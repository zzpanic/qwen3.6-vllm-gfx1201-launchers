#!/bin/bash
# Phase B reader -- cache-preemption-patch-plan.md R3.14.3.
#
# Phase A said 99.75% of production lookups defer on an in-flight store. Phase B stops
# them deferring and takes the ready prefix instead. This reads whether that worked, and
# it compares against the Phase A snapshot rather than against a number in someone's head.
#
#   ./phaseb-read.sh              read now, compare against the Phase A baseline
#   ./phaseb-read.sh --save NAME  also write a JSON snapshot under bench-history
#
# Windows differ (the baseline is 49 min, this run is whatever it is), so everything that
# is compared is compared as a SHARE OF LOOKUPS, never as a raw count.
#
# It refuses to report a verdict it cannot support: if the patch is not actually live, or
# there are too few lookups, it says so instead of printing a comparison that means nothing.
set -uo pipefail
PORT="${PORT:-5804}"
BASE="http://127.0.0.1:$PORT"
BASELINE="${BASELINE:-$HOME/bench-history/phasea-20260907-46min/counters.json}"

M=$(curl -s --max-time 30 "$BASE/metrics") || { echo "cannot reach $BASE/metrics"; exit 1; }
[ -n "$M" ] || { echo "empty /metrics from $BASE"; exit 1; }

val() {  # counter name -> integer value, 0 if the series does not exist
  echo "$M" | awk -v k="vllm:kv_offload_lookup_$1_total" \
    '$0 ~ "^"k"\\{" { gsub(/.*} /,""); printf "%d\n", $0+0; found=1; exit }
     END { if (!found) print "0" }'
}

# Separate probe, used for ONE counter only. A counter vLLM has registered but never
# incremented is reported as 0 by val() -- indistinguishable from a counter that does not
# exist because the patch is absent. Those mean opposite things, so the liveness check has
# to ask a different question: is the series registered at all?
registered() {
  echo "$M" | awk -v k="vllm:kv_offload_lookup_$1_total" \
    '$0 ~ "^"k"\\{" { print "yes"; found=1; exit } END { if (!found) print "no" }'
}

TRUNC_LIVE=$(registered pending_truncated)
TRUNC=$(val pending_truncated)
CALLS=$(val calls)
HIT=$(val chunk_hit);        PEND=$(val chunk_hit_pending)
RETRY=$(val chunk_retry);    MISS=$(val chunk_miss)
SW=$(val skip_short_window); ZH=$(val skip_zero_hit);  SR=$(val skip_short_result)
DB=$(val deferred_backend);  DL=$(val deferred_loading)
SERVED=$(val served);        STOK=$(val served_tokens)

gauge() {  # any metric name -> value, 0 if absent
  echo "$M" | awk -v k="$1" \
    '$0 ~ "^"k"\\{" { gsub(/.*} /,""); printf "%.0f\n", $0+0; found=1; exit }
     END { if (!found) print "0" }'
}
src() {  # prompt_tokens_by_source for one source label
  echo "$M" | awk -v s="source=\""$1"\"" \
    '/^vllm:prompt_tokens_by_source_total\{/ && index($0, s) { gsub(/.*} /,""); printf "%.0f\n", $0+0; found=1; exit }
     END { if (!found) print "0" }'
}

EXT=$(src external_kv_transfer); LOCALHIT=$(src local_cache_hit); LOCALCOMP=$(src local_compute)
LOADS=$(gauge "vllm:kv_offload_load_size_count")
LOADBYTES=$(gauge "vllm:kv_offload_load_bytes_total")

PID=$(pgrep -f "VLLM::EngineCore" | head -1)
UP=$( [ -n "$PID" ] && ps -o etime= -p "$PID" | tr -d ' ' || echo "unknown")
ENVFLAG=$(podman inspect qwen38-27b-vllm --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
  | awk -F= '/^RADIANCE_OFFLOAD_PENDING_IS_MISS=/{print $2}')

printf '\nengine uptime %s   RADIANCE_OFFLOAD_PENDING_IS_MISS=%s\n' "$UP" "${ENVFLAG:-unknown}"

# Is the patch actually in the running engine? vLLM registers every counter at startup, so
# an ABSENT series means the patch did not apply -- which would otherwise look exactly like
# "the patch applied and never truncated anything". Those are opposite conclusions.
if [ "$TRUNC_LIVE" = "no" ]; then
  echo
  echo "STOP: vllm:kv_offload_lookup_pending_truncated is not registered."
  echo "      The serve-ready-prefix patch is NOT live in this engine, whatever the env says."
  echo "      Check the boot log for: [radiance] WARNING: serve-ready-prefix patch did NOT apply"
  exit 2
fi
if [ "${ENVFLAG:-1}" = "0" ]; then
  echo "      (flag is 0: this engine is running UPSTREAM behaviour on purpose -- an A/B control run)"
fi

printf '\nthis run\n'
printf '  calls               %10s\n' "$CALLS"
printf '  SERVED              %10s   tokens %s\n' "$SERVED" "$STOK"
printf '  pending truncated   %10s   chunks declined instead of deferring the request\n' "$TRUNC"
printf '  deferred backend    %10s\n' "$DB"
printf '  deferred loading    %10s\n' "$DL"
printf '  skip zero hit       %10s\n' "$ZH"
printf '  skip short result   %10s\n' "$SR"
printf '  skip short window   %10s\n' "$SW"
printf '  chunks: HIT %s  HIT_PENDING %s  MISS %s  RETRY %s\n' "$HIT" "$PEND" "$MISS" "$RETRY"

# The counters above are branch frequencies. _lookup runs for every waiting request on every
# scheduler step, so one request can be counted hundreds of times and "served" does NOT mean
# "a request got a cache hit". What actually moved is below, from vLLM's own accounting.
printf '\nwhat actually transferred (vLLM'"'"'s own accounting, not ours)\n'
printf '  external load jobs  %10s   %s GB\n' "$LOADS" \
  "$(awk -v b="$LOADBYTES" 'BEGIN{printf "%.2f", b/1e9}')"
printf '  prompt tokens: external %s | GPU prefix cache %s | recomputed %s\n' \
  "$EXT" "$LOCALHIT" "$LOCALCOMP"
printf '  external share of prompt tokens: %s%%\n' \
  "$(awk -v e="$EXT" -v a="$LOCALHIT" -v c="$LOCALCOMP" 'BEGIN{t=e+a+c; printf "%.2f", t?100*e/t:0}')"

echo
python3 - "$BASELINE" "$CALLS" "$DB" "$SERVED" "$STOK" "$ZH" "$SR" "$SW" "$TRUNC" "$PEND" "$RETRY" "$EXT" "$LOCALHIT" "$LOCALCOMP" "$LOADS" <<'PY'
import json, sys, os
base_path, *rest = sys.argv[1:]
(calls, db, served, stok, zh, sr, sw, trunc, pend, retry,
 ext, localhit, localcomp, loads) = (float(x) for x in rest)

if calls < 50:
    print("VERDICT: too few lookups (%d) to read. Let it take real traffic." % calls)
    raise SystemExit(0)

# A restart wipes the /dev/shm CPU tier, so for a while after a boot every lookup misses
# the primary and goes to disk, which answers RETRY while it promotes. RETRY still defers
# by design -- Phase B did not touch it -- so a cold tier produces exactly the symptom
# Phase B is meant to cure, for a completely different reason. Say so rather than scoring it.
if retry > 4 * pend:
    print("HOLD: RETRY %d vs HIT_PENDING %d -- the CPU tier is still refilling from disk."
          % (retry, pend))
    print("      Those deferrals are promotions, not in-flight stores, and Phase B does not")
    print("      touch RETRY. This run cannot score the patch yet. Let the tier warm up.")
    print()

def pct(n, d):
    return 100.0 * n / d if d else 0.0

print("BRANCH FREQUENCIES -- how often each exit of _lookup was taken. _lookup runs for")
print("every waiting request on every scheduler step, so these are NOT per-request hit")
print("rates and 'served' is NOT 'a request got a cache hit'.")
print()
print("  this run:      %6.2f%% deferred   %6.2f%% reached the serve branch   (%d calls)"
      % (pct(db, calls), pct(served, calls), calls))

have_base = os.path.exists(base_path)
if have_base:
    b = json.load(open(base_path))
    bc = float(b["calls"])
    print("  Phase A base:  %6.2f%% deferred   %6.2f%% reached the serve branch   (%d calls, %s)"
          % (pct(float(b["deferred_backend"]), bc), pct(float(b["served"]), bc), bc, b["uptime"]))
    d_defer = pct(db, calls) - pct(float(b["deferred_backend"]), bc)
    print("  change:        %+6.2f pp deferred" % d_defer)
else:
    d_defer = None
    print("  (no Phase A baseline at %s)" % base_path)
print("  %d chunks truncated, %.2f per call" % (trunc, trunc / calls))

# The honest axis. Our counters live inside _lookup and cannot see whether a successful
# lookup turned into a transfer. vLLM's own accounting can, and during the first Phase B
# run the two disagreed by 444x -- 448 serve-branch hits against ONE load job. Report the
# transfer side separately and let it, not the branch count, decide the verdict.
total_prompt = ext + localhit + localcomp
print()
print("WHAT ACTUALLY TRANSFERRED")
print("  %d external load jobs; %d prompt tokens came from the external cache (%.2f%% of prompt)"
      % (loads, ext, pct(ext, total_prompt)))
if served > 0 and loads == 0:
    print("  NOTE: the serve branch fired %d times and NOTHING loaded. The lookup is fixed;"
          % served)
    print("        whatever consumes its answer is not. That is the next thing to look at.")
elif served > 3 * max(loads, 1):
    print("  NOTE: %d serve-branch hits against %d load job(s). Expected -- the same waiting"
          % (served, loads))
    print("        request is re-looked-up every step -- but it means the branch count cannot")
    print("        stand in for a hit rate.")

print()
if calls < 200 or loads < 5:
    print("VERDICT: NOT YET. The lookup path is measurable but only %d load job(s) have run."
          % loads)
    print("         A hit rate needs more traffic than this. Come back with more requests.")
elif d_defer is not None and d_defer < -5 and pct(ext, total_prompt) > 5.54:
    print("VERDICT: WORKED. Deferrals fell AND the external share of prompt tokens rose")
    print("         above the 5.54%% measured during Phase A.")
elif d_defer is not None and d_defer < -5:
    print("VERDICT: PARTIAL. Deferrals fell, but the external share of prompt tokens has not")
    print("         risen above Phase A's 5.54%. The lookup is fixed and something downstream")
    print("         of it is now the limit.")
else:
    print("VERDICT: NO IMPROVEMENT in the deferral rate.")
PY

if [ "${1:-}" = "--save" ]; then
  OUT="$HOME/bench-history/phaseb-${2:-$(date +%Y%m%d)}"
  mkdir -p "$OUT"
  printf '{"uptime":"%s","pending_is_miss":"%s","calls":%s,"chunk_hit":%s,"chunk_hit_pending":%s,"chunk_retry":%s,"chunk_miss":%s,"skip_short_window":%s,"skip_zero_hit":%s,"skip_short_result":%s,"deferred_backend":%s,"deferred_loading":%s,"served":%s,"served_tokens":%s,"pending_truncated":%s}\n' \
    "$UP" "${ENVFLAG:-unknown}" "$CALLS" "$HIT" "$PEND" "$RETRY" "$MISS" "$SW" "$ZH" "$SR" "$DB" "$DL" "$SERVED" "$STOK" "$TRUNC" \
    > "$OUT/counters.json"
  echo "wrote $OUT/counters.json"
fi
