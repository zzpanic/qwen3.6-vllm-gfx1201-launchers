#!/bin/bash
# check-r315-boot.sh -- did the reload actually pick up the R3.15 fix?
#
# Run this FIRST, right after the model comes back up and before any bench. It is read-only
# and takes about a second. It answers the one question a bench cannot: is the code that is
# running the code that was fixed? The patches are applied at container start from the
# host directory mounted at /house, so a stale mount, a failed patch or a launcher that
# skipped a step all look identical from outside -- the engine boots either way and then
# crashes hours later under real traffic, which is exactly how this bug was found.
#
# Checks, in order of how cheaply they fail:
#   1. the container is up and the engine answers /health
#   2. all eight house patches reported OK in the boot log
#   3. the KV geometry is the expected 9 groups (a different split means a different g8)
#   4. RADIANCE_OFFLOAD_MIXED_HIT is 1 -- the fix is only exercised when mixed hits serve
#   5. BOTH R3.15 hunks are present in the RUNNING engine's scheduler.py, read through
#      /proc/<pid>/root so it is the live file, not the copy on the host
#   6. no boundary assertion has fired yet this boot
set -uo pipefail

CONTAINER=${CONTAINER:-qwen38-27b-vllm}
fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

echo "== 1. container and engine =="
cpid=$(podman inspect "$CONTAINER" --format '{{.State.Pid}}' 2>/dev/null)
if [[ -z "$cpid" || "$cpid" == "0" ]]; then
  bad "container $CONTAINER is not running"; exit 1
fi
ok "container $CONTAINER up, pid $cpid"
started=$(podman inspect "$CONTAINER" --format '{{.State.StartedAt}}' 2>/dev/null)
ok "started $started"

echo "== 2. house patches =="
logs=$(podman logs "$CONTAINER" 2>&1)
napplied=$(grep -c "^  OK  " <<<"$logs")
nfailed=$(grep -cE "^  (FAIL|ERROR)" <<<"$logs")
grep -E "^  (OK|FAIL|ERROR)" <<<"$logs" | sed 's/^/    /'
if (( nfailed > 0 )); then bad "$nfailed patch hunk(s) failed"; else ok "$napplied patch hunks applied, 0 failed"; fi

echo "== 3. KV geometry =="
gline=$(grep -o "\[radiance\] kv cache groups:.*" <<<"$logs" | tail -1)
if [[ -z "$gline" ]]; then bad "no '[radiance] kv cache groups' line in the boot log"
elif grep -q "9 groups" <<<"$gline"; then ok "$gline"
else bad "unexpected geometry: $gline"; fi

echo "== 4. mixed-hit flag =="
mh=$(podman inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
     | grep '^RADIANCE_OFFLOAD_MIXED_HIT=' | cut -d= -f2)
case "$mh" in
  1) ok "RADIANCE_OFFLOAD_MIXED_HIT=1 (mixed hits are served; the fix is live)" ;;
  0) bad "RADIANCE_OFFLOAD_MIXED_HIT=0 -- the kill switch is on, the fixed path is unreachable" ;;
  "") warn "RADIANCE_OFFLOAD_MIXED_HIT unset in the container env; patch default is 1" ;;
  *) bad "RADIANCE_OFFLOAD_MIXED_HIT=$mh (expected 1)" ;;
esac

echo "== 5. the fix is in the RUNNING code =="
SCHED=/proc/$cpid/root/opt/vllm/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
if [[ ! -r "$SCHED" ]]; then
  bad "cannot read the running scheduler.py at $SCHED"
else
  if grep -q "num_chunks - required_window" "$SCHED"; then
    ok "hunk 3 present: the lookup scans low enough to confirm what the load reads"
  else
    bad "hunk 3 MISSING -- the lookup can still hand prepare_load an unconfirmed key"
  fi
  if grep -q "sliding_window_size_in_chunks is None and (" "$SCHED"; then
    ok "hunk 4 present: the boundary assertion is scoped to full-attention groups"
  else
    bad "hunk 4 MISSING -- the old unscoped boundary assertion is still live"
  fi
  d=$(grep -c 'RADIANCE_OFFLOAD_MIXED_HIT", "1"' "$SCHED")
  (( d > 0 )) && ok "flag defaults to 1 in the running code" \
              || warn "flag default in the running code is not 1"
fi

echo "== 6. nothing has fired yet =="
hits=$(grep -cE "offload boundary: local_tokens=|Block .* not found in cache" <<<"$logs")
if (( hits > 0 )); then
  bad "$hits boundary/prepare_load line(s) already in this boot:"
  grep -E "offload boundary: local_tokens=|Block .* not found in cache" <<<"$logs" | tail -5 | sed 's/^/    /'
else
  ok "no boundary assertion, no unconfirmed-key assertion this boot"
fi

echo
if (( fail )); then echo "RESULT: NOT CLEAR TO BENCH -- fix the above first"; exit 1; fi
echo "RESULT: clear to bench -- run ./mixedbench.py --yes"
