#!/bin/bash
# Phase A reader -- cache-preemption-patch-plan.md R3.14.2.
#
# Prints the offload lookup-outcome counters and reads the verdict off them. The counters
# are installed by patch_kv_offload_lookup_outcomes.py at container launch, so they reset
# on every restart: a run of this script measures "since the engine last booted", and the
# engine uptime is printed alongside so the window is never ambiguous.
#
#   ./phasea-read.sh              read the counters now
#   ./phasea-read.sh --save NAME  also write a JSON snapshot under bench-history
#
# A counter with no series has never fired; that is itself a result, so absent lines are
# printed as 0 rather than omitted.
set -uo pipefail
PORT="${PORT:-5804}"
BASE="http://127.0.0.1:$PORT"

M=$(curl -s --max-time 30 "$BASE/metrics") || { echo "cannot reach $BASE/metrics"; exit 1; }
[ -n "$M" ] || { echo "empty /metrics from $BASE"; exit 1; }

val() {  # counter name -> integer value, 0 if the series does not exist
  echo "$M" | awk -v k="vllm:kv_offload_lookup_$1_total" \
    '$0 ~ "^"k"\\{" { gsub(/.*} /,""); printf "%d\n", $0+0; found=1; exit }
     END { if (!found) print "0" }'
}

CALLS=$(val calls)
HIT=$(val chunk_hit);        PEND=$(val chunk_hit_pending)
RETRY=$(val chunk_retry);    MISS=$(val chunk_miss);   OTHER=$(val chunk_other)
SW=$(val skip_short_window); ZH=$(val skip_zero_hit);  SR=$(val skip_short_result)
DB=$(val deferred_backend);  DL=$(val deferred_loading)
SERVED=$(val served);        STOK=$(val served_tokens)

PID=$(pgrep -f "VLLM::EngineCore" | head -1)
UP=$( [ -n "$PID" ] && ps -o etime= -p "$PID" | tr -d ' ' || echo "unknown")

printf '\nengine uptime %s   (counters reset on restart)\n\n' "$UP"
printf 'per-chunk backend results\n'
printf '  HIT           %12s\n  HIT_PENDING   %12s\n  RETRY         %12s\n' "$HIT" "$PEND" "$RETRY"
printf '  MISS          %12s\n  other         %12s\n' "$MISS" "$OTHER"
printf '\nwhole-lookup outcomes (these partition every call)\n'
printf '  calls              %12s\n' "$CALLS"
printf '  skip short window  %12s   bailed before querying: under one chunk to fetch\n' "$SW"
printf '  skip zero hit      %12s   backend hit nothing\n' "$ZH"
printf '  skip short result  %12s   queried, but the confirmed hit was under one chunk\n' "$SR"
printf '  deferred backend   %12s   prefix abandoned: a chunk was HIT_PENDING or RETRY\n' "$DB"
printf '  deferred loading   %12s   hit chunks already being loaded\n' "$DL"
printf '  SERVED             %12s   tokens %s\n' "$SERVED" "$STOK"

# The verdict. Deliberately only fires on a clear plurality of a meaningful sample -- a
# handful of calls after a restart says nothing, and saying so is the point.
echo
python3 - "$CALLS" "$SW" "$ZH" "$SR" "$DB" "$DL" "$SERVED" "$RETRY" "$MISS" "$HIT" "$PEND" <<'PY'
import sys
c, sw, zh, sr, db, dl, served, retry, miss, hit, pend = (float(x) for x in sys.argv[1:12])
if c < 50:
    print("VERDICT: too few lookups (%d) to read. Needs a day of normal traffic." % c)
else:
    named = {"A3 sub-chunk window (pre-query)": sw, "A3 sub-chunk window (post-query)": sr,
             "A2/A3 zero hit": zh, "A1 lookup defer (prefix abandoned)": db,
             "load-in-flight defer": dl, "hits are real": served}
    top, n = max(named.items(), key=lambda kv: kv[1])
    print("VERDICT: %s dominates -- %d of %d lookups (%.0f%%)." % (top, n, c, 100.0*n/c))
    if served and c:
        print("         served %d (%.1f%% of calls)" % (served, 100.0*served/c))
    print("         backend chunk results: HIT %d  HIT_PENDING %d  MISS %d  RETRY %d"
          % (hit, pend, miss, retry))
    if db and (pend or retry):
        # Which one abandoned the prefix decides the fix, so say which, not just "defer".
        if pend > 4 * retry:
            print("         TRIGGER: HIT_PENDING (%.0f:1) -- a GPU->CPU store has not landed."
                  % (pend / max(retry, 1)))
            print("         The block IS cached; ref_cnt is still -1. Not a retry livelock.")
        elif retry > 4 * pend:
            print("         TRIGGER: RETRY (%.0f:1) -- the backend is asking to be called again."
                  % (retry / max(pend, 1)))
        else:
            print("         TRIGGER: mixed HIT_PENDING/RETRY -- no single cause.")
PY

if [ "${1:-}" = "--save" ]; then
  OUT="$HOME/bench-history/phasea-${2:-$(date +%Y%m%d)}"
  mkdir -p "$OUT"
  printf '{"uptime":"%s","calls":%s,"chunk_hit":%s,"chunk_hit_pending":%s,"chunk_retry":%s,"chunk_miss":%s,"chunk_other":%s,"skip_short_window":%s,"skip_zero_hit":%s,"skip_short_result":%s,"deferred_backend":%s,"deferred_loading":%s,"served":%s,"served_tokens":%s}\n' \
    "$UP" "$CALLS" "$HIT" "$PEND" "$RETRY" "$MISS" "$OTHER" "$SW" "$ZH" "$SR" "$DB" "$DL" "$SERVED" "$STOK" \
    > "$OUT/counters.json"
  echo "wrote $OUT/counters.json"
fi
