#!/usr/bin/env bash
# kvcache-reap.sh -- eviction for the KVCACHE=disk tier. Run it from a timer (every minute).
#
# vLLM's fs secondary tier (tiering/fs/manager.py) has no capacity, quota or TTL setting: it
# writes block files and never deletes them. This script is the eviction policy.
#
# Policy: keep the filesystem at or below KVCACHE_TARGET_PCT, deleting oldest-first by mtime.
# The tier writes each block once, so mtime is insertion time and oldest-first approximates LRU
# at no I/O cost. Blocks younger than KVCACHE_MIN_AGE_MIN are left alone in normal running
# (they are the engine's hot set); only when the volume passes KVCACHE_EMERGENCY_PCT does it
# delete below that floor, because a full volume fails every store. Deleting a block the engine
# still wanted costs a recompute of that block (with patch_fs_failed_load.py applied).
#
# Fullness is `df` on the filesystem holding KVCACHE_ROOT, so give the tier ITS OWN filesystem:
# anything else written to the same volume is paid for by deleting cache blocks.
#
#   KVCACHE_ROOT=/path/to/KVCACHE_DISK/blocks ./kvcache-reap.sh     (default /kvcache/blocks;
#   DRY_RUN=1 lists, deletes nothing)
#
# KVCACHE_MIN_FREE_GB (0 = off): free space, in GB, that must remain after every run. Size it to
# the most the engine can write before the NEXT run finishes: peak store rate x the gap between
# runs, plus margin. Below it the emergency stage runs whatever %use says -- a percentage
# trigger (EMERGENCY_PCT) leaves too little room at a high store rate: 10% of a 94 GB volume is
# ~40 s of peak writes, under one timer cycle. Reference box (one R9700, Qwen3.8-27B MXFP4,
# DFlash x7, fp8 KV, 94 GB tier volume), 2026-10-05: peak fs store 2.25 GB per 10 s
# metrics interval (225 MB/s; analytic ceiling ~3,200 prefill t/s x 73 KB/token on disk), gap
# between runs <= 80 s (1-min timer + 10 s accuracy, runs take <= 3 s) -> 19.2 GB; x1.25 -> 24.
set -euo pipefail

ROOT=${KVCACHE_ROOT:-${KVCACHE_DISK:+$KVCACHE_DISK/blocks}}
ROOT=${ROOT:-/kvcache/blocks}
MIN_AGE_MIN=${KVCACHE_MIN_AGE_MIN:-15}      # HARD FLOOR: nothing younger is ever deleted
MAX_AGE_HOURS=${KVCACHE_MAX_AGE_HOURS:-0}   # 0 = age rule OFF, capacity alone governs; >0 = also delete anything older
TARGET_PCT=${KVCACHE_TARGET_PCT:-70}        # and keep %use at or below this
MAX_DELETE=${KVCACHE_MAX_DELETE:-4000}      # per-run cap so no cycle runs long; next cycle continues
EMERGENCY_PCT=${KVCACHE_EMERGENCY_PCT:-90}  # above this after Stage B, Stage C crosses MIN_AGE_MIN (0 = never)
EMERGENCY_MIN_AGE_MIN=${KVCACHE_EMERGENCY_MIN_AGE_MIN:-1}  # Stage C still spares blocks written in the last minute
TMP_AGE_MIN=${KVCACHE_TMP_AGE_MIN:-60}      # orphaned *.tmp older than this go too
MIN_FREE_GB=${KVCACHE_MIN_FREE_GB:-0}       # keep at least this much free after every run (0 = off); see header
DRY=${DRY_RUN:-}

[ -d "$ROOT" ] || { echo "kvcache-reap: $ROOT does not exist; nothing to do" >&2; exit 0; }

# A floor at or above the ceiling would make the age rule delete blocks the floor
# forbids -- incoherent rather than merely aggressive. Refuse instead of guessing.
# MAX_AGE_HOURS=0 means the age rule is off, so the guard does not apply -- without
# this exemption the script would exit 1 and do NO reaping at all, which on a
# filling volume is the dangerous failure, not the safe one.
if [ "$MAX_AGE_HOURS" -gt 0 ] && [ "$MIN_AGE_MIN" -ge $(( MAX_AGE_HOURS * 60 )) ]; then
  echo "kvcache-reap: MIN_AGE_MIN=${MIN_AGE_MIN}min >= MAX_AGE_HOURS=${MAX_AGE_HOURS}h; refusing to run" >&2
  exit 1
fi

pct() { df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9'; }
free_gb() { df -B1G --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }
# "at target" means both: %use at or below TARGET_PCT and at least MIN_FREE_GB free.
at_target() { [ "$(pct)" -le "$TARGET_PCT" ] && [ "$(free_gb)" -ge "$MIN_FREE_GB" ]; }

# A crashed store leaves <name>.bin.tmp<suffix> behind: io.py writes to a temp
# path and os.replace()s it. Those are never read and never reaped by vLLM.
if [ -z "$DRY" ]; then
  find "$ROOT" -type f -name '*.tmp*' -mmin "+$TMP_AGE_MIN" -delete 2>/dev/null || true
else
  find "$ROOT" -type f -name '*.tmp*' -mmin "+$TMP_AGE_MIN" -printf 'would remove stale tmp %p\n' 2>/dev/null || true
fi

removed=0
capped=0
floor_hit=0
# The loops below read from fd 3 via process substitution, so the body runs in
# this shell and `removed` survives the iteration. A pipe would not.
rm_one() {
  if [ -n "$DRY" ]; then echo "would remove $1"; else rm -f -- "$1" || return 0; fi
  removed=$((removed+1))
}

# --- Stage A: age. OFF by default: capacity alone decides. When KVCACHE_MAX_AGE_HOURS > 0 it
# --- runs every cycle whatever the usage.
if [ "$MAX_AGE_HOURS" -gt 0 ]; then
  while IFS= read -r -d '' -u 3 f; do
    if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
    [ -f "$f" ] || continue
    rm_one "$f"
  done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$(( MAX_AGE_HOURS * 60 ))" -print0)
fi
aged=$removed

# --- Stage B: capacity. Only if age alone did not keep us under target. Deletes
# --- oldest-first, and NEVER below the MIN_AGE_MIN floor -- that restriction is
# --- deliberate, so it is expressed in the find, not in a check a later edit could drop.
if ! at_target && [ "$capped" -eq 0 ]; then
  # NUL-delimited and mtime-sorted so filenames can contain anything.
  while IFS= read -r -d '' -u 3 line; do
    if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
    f=${line#* }
    [ -f "$f" ] || continue
    rm_one "$f"
    # df is not free; re-check in batches rather than per file.
    if [ $(( (removed - aged) % 200 )) -eq 0 ]; then
      at_target && break
    fi
  done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$MIN_AGE_MIN" -printf '%T@ %p\0' | sort -z -n 2>/dev/null)
  [ -z "$DRY" ] && ! at_target && floor_hit=1
fi

# --- Stage C: emergency. The floor held and the volume is still nearly full, so
# --- stores are about to fail (ENOSPC) for everything. Delete oldest-first below
# --- the floor, down to the target, sparing only the last EMERGENCY_MIN_AGE_MIN.
emergency=0
cur=$(pct)
low_free=0
[ "$MIN_FREE_GB" -gt 0 ] && [ "$(free_gb)" -lt "$MIN_FREE_GB" ] && low_free=1
if { { [ "$EMERGENCY_PCT" -gt 0 ] && [ "$cur" -gt "$EMERGENCY_PCT" ]; } || [ "$low_free" -eq 1 ]; } && [ "$capped" -eq 0 ]; then
  emergency=1
  while IFS= read -r -d '' -u 3 line; do
    if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
    f=${line#* }
    [ -f "$f" ] || continue
    rm_one "$f"
    if [ $(( removed % 200 )) -eq 0 ]; then
      at_target && break
    fi
  done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$EMERGENCY_MIN_AGE_MIN" -printf '%T@ %p\0' | sort -z -n 2>/dev/null)
  [ -z "$DRY" ] && at_target && floor_hit=0
fi

# Directory fan-out is <hhh>/<hh>_g<group>/, so emptied dirs accumulate. Only
# reachable when something was actually deleted, which is when they appear.
if [ -z "$DRY" ] && [ "$removed" -gt 0 ]; then
  find "$ROOT" -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

now=$(pct)
if [ "$removed" -eq 0 ]; then
  if [ "$MAX_AGE_HOURS" -gt 0 ]; then
    echo "kvcache-reap: ${now}% used, nothing older than ${MAX_AGE_HOURS}h and at/below the ${TARGET_PCT}% target; nothing to do"
  else
    echo "kvcache-reap: ${now}% used, at/below the ${TARGET_PCT}% target (age rule off); nothing to do"
  fi
else
  echo "kvcache-reap: removed $removed block(s) [${aged} by age, $((removed - aged)) for capacity]; now ${now}% used, $(free_gb) GB free"
fi
[ "$capped" -eq 1 ] && echo "kvcache-reap: stopped at the ${MAX_DELETE}-block per-run cap; the next cycle continues where this one left off"
[ "$emergency" -eq 1 ] && echo "kvcache-reap: EMERGENCY stage ran (volume above ${EMERGENCY_PCT}% or under ${MIN_FREE_GB} GB free, with every block under ${MIN_AGE_MIN} min); deleted below the floor, sparing the last ${EMERGENCY_MIN_AGE_MIN} min" >&2
if [ "$floor_hit" -eq 1 ]; then
  echo "kvcache-reap: WARNING ${now}% used / $(free_gb) GB free misses the target (${TARGET_PCT}%, ${MIN_FREE_GB} GB free), but every remaining block is" >&2
  echo "kvcache-reap:   younger than ${MIN_AGE_MIN} min. NOT deleting those -- recent blocks are the engine's hot set." >&2
  echo "kvcache-reap:   If this line repeats, the volume is too small for this workload; grow it, lower KVCACHE_MIN_AGE_MIN, or set" >&2
  echo "kvcache-reap:   KVCACHE_MAX_AGE_HOURS to a positive value to shed old blocks before pressure builds." >&2
fi
exit 0
