#!/usr/bin/env bash
# startup-cache/cache-reap.sh -- remove stale launcher cache trees.
#
# WHY THIS EXISTS: every launcher keys its cache dir on the image ID, the GPU arch and its tuning
# flags (STARTUP_CACHE_KEY plus a suffix), so every new image, card or flag combination gets a
# whole new tree, and nothing else removes the old ones. Left alone they grow without bound: on the
# box this was written on, nine stale trees held 17 GiB. This is the eviction policy for them.
#
# THE FAILURE MODE IS BENIGN, AND THAT SHAPES THE WHOLE DESIGN: deleting a tree that turns out to
# be live costs a SLOW NEXT BOOT (everything recompiles), not a wrong answer. Every byte in these
# trees is regenerable build output. So the job is to be conservative (never the live tree, never
# one in use) and bounded (never delay a boot), not clever.
#
# SELECTION RULE (three signals, ranked):
#   GUARD     the live tree (--live) is never deleted, and neither is any tree whose suffix equals
#             the live one (in-set, potentially live). Trees are the siblings "<--base>"*.
#   PRIMARY   the stamp <tree>/.last-boot-ok ("ts=<UTC ISO time>"), written at launch by
#             startup_cache_reap and again by a container on a successful boot. A stamp newer than
#             MIN_AGE_HOURS keeps the tree, unconditionally. An old or missing stamp does NOT mean
#             dead -- it falls through to the fallback.
#   FALLBACK  file atime. A tree with no fresh stamp is deletable only if its newest file atime is
#             older than MIN_AGE_HOURS. The stamp file itself is EXCLUDED from that scan: reading
#             it (the primary check above) refreshes its atime under relatime, and counting it once
#             made every stamped tree look in use forever, so this pruner silently deleted nothing
#             for weeks. Directory atimes are never used (any listing refreshes them), and the scan
#             uses stat-only tools (find -printf, du) so it does not touch file atimes itself.
#
# CAPACITY-GOVERNED, NOT AGE-GOVERNED: nothing happens until the filesystem is above TARGET_PCT
# used. MIN_AGE_HOURS is a hard floor no rule crosses, not a trigger. Whole trees only, oldest
# first, stopping as soon as the target is met, at most MAX_DELETE per run.
#
# BOUNDED: the caller wraps the run in `timeout`; a scan that cannot finish keeps the tree.
# DRY RUN IS THE DEFAULT: with no flag it prints what would go and deletes nothing. --apply (or
# CACHE_REAP_APPLY=1) deletes.
#
# The KV-cache disk tier has its own reaper (kv-cache/ops/kvcache-reap.sh); that volume is
# noatime and is not this script's business.

# --- identity: passed by the caller (the launcher knows these by construction) -------
LIVE=""      # the live tree ($CACHE). NEVER deleted.
BASE=""      # the sibling prefix, e.g. $HOME/.radiance-cache-w4a8- . Trees are "$BASE"*.
# --- tuning knobs (env) -----------------------------------------------------------
TARGET_PCT=${CACHE_REAP_TARGET_PCT:-65}          # free-space target: prune only above this
MIN_AGE_HOURS=${CACHE_REAP_MIN_AGE_HOURS:-24}   # HARD FLOOR: nothing with a newer file atime is deleted
MAX_DELETE=${CACHE_REAP_MAX_DELETE:-8}          # per-run cap on trees removed; next boot continues
APPLY=${CACHE_REAP_APPLY:-0}
DRY=1                                            # DRY RUN BY DEFAULT

usage() {
  echo "usage: cache-reap.sh --live <tree> --base <prefix> [--dry-run|--apply]" >&2
  echo "  env: CACHE_REAP_TARGET_PCT=${CACHE_REAP_TARGET_PCT} CACHE_REAP_MIN_AGE_HOURS=${MIN_AGE_HOURS}" >&2
  echo "       CACHE_REAP_MAX_DELETE=${MAX_DELETE} CACHE_REAP_APPLY=${APPLY}" >&2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --live)    [ $# -ge 2 ] || { usage; exit 2; }; LIVE="$2"; shift 2;;
    --base)    [ $# -ge 2 ] || { usage; exit 2; }; BASE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --apply)   APPLY=1; DRY=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "cache-reap: unknown arg $1" >&2; usage; exit 2;;
  esac
done

# A missing pruner, or a bad invocation, must never stop a boot: the launcher guards this
# call with an existence check and `|| true`. Fail loudly-but-harmlessly here.
[ -n "$LIVE" ] || { echo "cache-reap: --live <tree> is required" >&2; exit 2; }
[ -d "$LIVE" ] || { echo "cache-reap: live tree $LIVE does not exist; nothing to do" >&2; exit 0; }
[ -n "$BASE" ] || { echo "cache-reap: --base <prefix> is required (the launcher supplies it)" >&2; exit 2; }
# BASE is a prefix (e.g. $HOME/.radiance-cache-w4a8-), not necessarily an existing
# path: the real trees are "$BASE"<suffix>. An empty "$BASE"* glob is safe (no
# candidates -> "nothing to do"), so we do not require BASE itself to exist.

# DRY RUN IS THE DEFAULT when no flag is given. Only --apply (or the env) deletes.
[ "$APPLY" = 1 ] && DRY=0

# Derive the live suffix by prefix-stripping (no hard-coded flag names).
BASE_NAME=$(basename "$BASE")
LIVE_NAME=$(basename "$LIVE")
case "$LIVE_NAME" in
  "$BASE_NAME"*) LIVE_SUF="${LIVE_NAME#"$BASE_NAME"}";;
  *) LIVE_SUF="$LIVE_NAME";;
esac

# Used % of the filesystem the trees live on (the base dir's fs == the root fs here).
pct() { df --output=pcent "$(dirname "$BASE")" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# True (0) if <tree>/.last-boot-ok exists and its timestamp is within MIN_AGE_HOURS.
# The stamp is the unpoisonable primary: read O(1), never corrupted by inspection.
stamp_is_fresh() {
  local f="$1/.last-boot-ok" ts now
  [ -f "$f" ] || return 1
  ts=$(grep -m1 -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T' "$f" 2>/dev/null | head -1)
  [ -n "$ts" ] || return 1
  now=$(date -u +%s)
  ts=$(date -u -d "$ts" +%s 2>/dev/null) || return 1
  [ $(( now - ts )) -le $(( MIN_AGE_HOURS * 3600 )) ]
}

# Newest file atime of a tree, integer epoch seconds. find -printf stats (does not open
# contents -> does not poison). awk streams the max (reads to EOF, so no early-close
# SIGPIPE under `set -o pipefail`, and no full sort). This is the walk path; bounded by
# the caller's timeout -- if the walk is still running when the timeout fires, the tree
# is retained (give up rather than overrun the boot path).
newest_atime() {
  # .last-boot-ok is excluded: stamp_is_fresh() reads it, which refreshes its atime
  # (relatime), so counting it made every stamped tree look in use forever (see header).
  find "$1" -type f ! -name .last-boot-ok -printf '%A@\n' 2>/dev/null | awk 'NR==1{m=$1+0} ($1+0)>m{m=$1+0} END{if (NR>0) printf "%.0f", m}'
}

cur=$(pct)
[ -n "$cur" ] || { echo "cache-reap: could not read fs usage; doing nothing (safe)" >&2; exit 0; }
# CAPACITY-GOVERNED: below the target, do nothing. (The age floor is never the trigger.)
if [ "$cur" -le "$TARGET_PCT" ]; then
  echo "cache-reap: ${cur}% used, at/below the ${TARGET_PCT}% target; nothing to do"
  exit 0
fi

# ---- select candidate trees. Coarse: whole trees only. The live tree is never a
# candidate, and neither is any in-set tree. One bounded atime walk per out-of-set
# candidate. A slow walk that the caller's timeout kills simply leaves the tree retained.
# Collect "atime<TAB>path" lines, then sort oldest-atime-first.
cand_lines=""
now=$(date -u +%s)
for c in "$BASE"*; do
  [ -d "$c" ] || continue
  [ "$c" = "$LIVE" ] && continue                 # GUARD: never delete the live tree
  cn=$(basename "$c")
  case "$cn" in
    "$BASE_NAME"*) suf="${cn#"$BASE_NAME"}";;
    *) suf="$cn";;
  esac
  [ "$suf" = "$LIVE_SUF" ] && continue          # GUARD: in-set == potentially live
  stamp_is_fresh "$c" && continue              # PRIMARY: fresh stamp == served recently
  na=$(newest_atime "$c")                       # FALLBACK: newest file atime (bounded walk)
  [ -n "$na" ] || continue                     # no atime info -> retain (give up, don't guess)
  # Floor: a tree with a recently-accessed file is in use (or its atime is poisoned) -> protect.
  awk -v a="$na" -v n="$now" -v h="$MIN_AGE_HOURS" 'BEGIN{exit !(a > n - h*3600)}' && continue
  cand_lines="${cand_lines}${na}	${c}
"
done

# size (KiB) of a whole tree -- for the "how much would be reclaimed" report. du stats
# (does not open file contents -> does not poison file atimes); it does touch directory
# atimes, which we deliberately do not use.
size_kib() { du -sk "$1" 2>/dev/null | awk '{print $1}'; }

# ---- delete: whole trees, oldest-atime-first, stop as soon as the target is met.
# Coarse first by design: we never reach *inside* a tree (that is the slow, walk-heavy
# path and would touch the live tree, which is forbidden). If whole trees cannot meet
# the target we stop and warn, exactly like the KV reaper's floor branch.
removed=0
capped=0
reclaimed_kib=0
if [ -n "$cand_lines" ]; then
  # sort -n ascending on the atime field (oldest first); atime is the first tab field.
  while IFS=$'\t' read -r na path; do
    [ -n "$path" ] || continue
    [ "$path" = "$LIVE" ] && continue          # belt-and-suspenders guard
    [ "$removed" -ge "$MAX_DELETE" ] && { capped=1; break; }
    sz=$(size_kib "$path")
    if [ "$DRY" = 1 ]; then
      echo "cache-reap: [dry-run] would remove tree ${path} (~${sz} KiB, newest file atime ${na})"
    else
      echo "cache-reap: removing tree ${path} (~${sz} KiB)"
      rm -rf -- "$path" 2>/dev/null || true
    fi
    removed=$((removed+1))
    reclaimed_kib=$((reclaimed_kib + ${sz:-0}))
    # df is not free; re-check after each tree (there are only a few).
    cur=$(pct)
    [ -n "$cur" ] && [ "$cur" -le "$TARGET_PCT" ] && break
  done < <(printf '%s' "$cand_lines" | sort -t $'\t' -k1,1 -n)
fi

now=$(pct)
if [ "$removed" -eq 0 ]; then
  echo "cache-reap: ${now}% used; no tree that is out of set, without a fresh stamp and unread for ${MIN_AGE_HOURS}h (target ${TARGET_PCT}%)"
else
  if [ "$DRY" = 1 ]; then
    echo "cache-reap: [dry-run] would remove $removed tree(s) (~$((reclaimed_kib/1024)) MiB); target ${TARGET_PCT}%"
  else
    echo "cache-reap: removed $removed tree(s) (~$((reclaimed_kib/1024)) MiB); now ${now}% used"
  fi
fi
[ "$capped" = 1 ] && echo "cache-reap: stopped at the ${MAX_DELETE}-tree per-run cap; the next boot continues where this one left off"
if [ "$DRY" != 1 ] && [ -n "$now" ] && [ "$now" -gt "$TARGET_PCT" ]; then
  echo "cache-reap: WARNING ${now}% used is still above the ${TARGET_PCT}% target, but every remaining" >&2
  echo "cache-reap:   candidate is in-set, freshly stamped, or within the ${MIN_AGE_HOURS}h floor. NOT deleting those --" >&2
  echo "cache-reap:   the floor is a hard safety property. If this repeats, grow the filesystem." >&2
fi
exit 0
