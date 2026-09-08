#!/usr/bin/env bash
# Reaper for the vLLM fs secondary KV tier.
#
# WHY THIS EXISTS: vllm/v1/kv_offload/tiering/fs/manager.py has no capacity,
# quota or TTL parameter, and SecondaryTierManager exposes no eviction hook.
# The tier writes and never deletes. Without this timer the filesystem fills
# and every subsequent store fails. This is not a tuning script; it is the
# eviction policy.
#
# AGE-BASED, changed 2026-09-06. The old policy did nothing below 80% used and
# then purged 80% -> 65% in one cycle: ~2,975 blocks, at the moment the cache is
# fullest and the engine busiest. Measured that afternoon, /kvcache went 55 GB ->
# 186 GB in 39 minutes of agentic load (~3.4 GB/min), so that purge was about an
# hour away, not the days first estimated from a quieter sample. Deleting
# steadily and early is the same total number of deletions -- you must delete
# what you write -- spread thin and taken at lower pressure.
#
# THE SAFETY PROPERTY IS MIN_AGE, NOT THE SCHEDULE. Deleting a block that a load
# is about to read is an ENGINE KILLER, not a missed hit: lookup stats the
# filesystem, so a reaped block normally just misses, but between lookup and load
# there is a window and offloading/worker.py:361 is a bare
# `assert transfer_result.success`. OffloadingConnector implements no
# get_block_ids_with_load_errors(), so kv_load_failure_policy=recompute cannot
# catch it. That window CANNOT be closed from here; it needs upstream. What this
# script can do is never delete a block the engine plausibly still wants.
# MIN_AGE_MIN is a hard floor that no rule -- capacity pressure included -- may
# cross. If the target cannot be met without crossing it, the script says so and
# stops, rather than delete young blocks.
#
# Ordering is oldest-mtime-first. The tier writes each block once and never
# modifies it (io.py short-circuits on os.path.exists), so mtime is insertion
# time: FIFO, which for a prefix cache approximates LRU at zero I/O cost. True
# atime-LRU would need the volume mounted at least `relatime` -- it is `noatime`
# today, so atime is meaningless here. That is a live remount and a separate
# decision, not something to do from inside a reaper.
#
# KNOWN ANTI-CORRELATION: a prefix shared by every conversation (a system prompt)
# is written once, early, and stays hot forever -- exactly the shape mtime-FIFO
# deletes first. That set is a handful of blocks out of thousands, and losing one
# costs a single recompute before the tier writes it back with a fresh mtime.
# Accepted with eyes open, not overlooked.
set -euo pipefail

ROOT=${KVCACHE_ROOT:-/kvcache/blocks}
MIN_AGE_MIN=${KVCACHE_MIN_AGE_MIN:-90}      # HARD FLOOR: nothing younger is ever deleted
MAX_AGE_HOURS=${KVCACHE_MAX_AGE_HOURS:-8}   # delete anything older, every cycle, regardless of %use
TARGET_PCT=${KVCACHE_TARGET_PCT:-65}        # and keep %use at or below this
MAX_DELETE=${KVCACHE_MAX_DELETE:-4000}      # per-run cap so no cycle runs long; next cycle continues
TMP_AGE_MIN=${KVCACHE_TMP_AGE_MIN:-60}      # orphaned *.tmp older than this go too
DRY=${DRY_RUN:-}

[ -d "$ROOT" ] || { echo "kvcache-reap: $ROOT does not exist; nothing to do" >&2; exit 0; }

# A floor at or above the ceiling would make the age rule delete blocks the floor
# forbids -- incoherent rather than merely aggressive. Refuse instead of guessing.
if [ "$MIN_AGE_MIN" -ge $(( MAX_AGE_HOURS * 60 )) ]; then
  echo "kvcache-reap: MIN_AGE_MIN=${MIN_AGE_MIN}min >= MAX_AGE_HOURS=${MAX_AGE_HOURS}h; refusing to run" >&2
  exit 1
fi

pct() { df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9'; }

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

# --- Stage A: age. Runs every cycle whatever the usage. This is the stage that
# --- stops the volume ever building toward a panic purge.
while IFS= read -r -d '' -u 3 f; do
  if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
  [ -f "$f" ] || continue
  rm_one "$f"
done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$(( MAX_AGE_HOURS * 60 ))" -print0)
aged=$removed

# --- Stage B: capacity. Only if age alone did not keep us under target. Deletes
# --- oldest-first, and NEVER below the MIN_AGE_MIN floor -- that restriction is
# --- the whole safety argument, so it is expressed in the find, not in a check
# --- that a later edit could drop.
cur=$(pct)
if [ "$cur" -gt "$TARGET_PCT" ] && [ "$capped" -eq 0 ]; then
  # NUL-delimited and mtime-sorted so filenames can contain anything.
  while IFS= read -r -d '' -u 3 line; do
    if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
    f=${line#* }
    [ -f "$f" ] || continue
    rm_one "$f"
    # df is not free; re-check in batches rather than per file.
    if [ $(( (removed - aged) % 200 )) -eq 0 ]; then
      cur=$(pct)
      [ "$cur" -le "$TARGET_PCT" ] && break
    fi
  done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$MIN_AGE_MIN" -printf '%T@ %p\0' | sort -z -n)
  cur=$(pct)
  [ -z "$DRY" ] && [ "$cur" -gt "$TARGET_PCT" ] && floor_hit=1
fi

# Directory fan-out is <hhh>/<hh>_g<group>/, so emptied dirs accumulate. Only
# reachable when something was actually deleted, which is when they appear.
if [ -z "$DRY" ] && [ "$removed" -gt 0 ]; then
  find "$ROOT" -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

now=$(pct)
if [ "$removed" -eq 0 ]; then
  echo "kvcache-reap: ${now}% used, nothing older than ${MAX_AGE_HOURS}h and at/below the ${TARGET_PCT}% target; nothing to do"
else
  echo "kvcache-reap: removed $removed block(s) [${aged} by age, $((removed - aged)) for capacity]; now ${now}% used"
fi
[ "$capped" -eq 1 ] && echo "kvcache-reap: stopped at the ${MAX_DELETE}-block per-run cap; the next cycle continues where this one left off"
if [ "$floor_hit" -eq 1 ]; then
  echo "kvcache-reap: WARNING ${now}% used is still above the ${TARGET_PCT}% target, but every remaining block is" >&2
  echo "kvcache-reap:   younger than ${MIN_AGE_MIN} min. NOT deleting those -- recent blocks are the engine's hot set," >&2
  echo "kvcache-reap:   and losing a load in flight kills EngineCore (see the header). If this line repeats, the" >&2
  echo "kvcache-reap:   volume is too small for this workload; grow it or lower KVCACHE_MAX_AGE_HOURS." >&2
fi
exit 0
