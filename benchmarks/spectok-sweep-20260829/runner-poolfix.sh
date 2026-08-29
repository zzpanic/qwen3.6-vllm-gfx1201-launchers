#!/usr/bin/env bash
# Corrective POOL-ONLY re-boot for SPECTOK arms C (K=6) and D (K=7) -- 2026-08-29.
#
# WHY THIS EXISTS: window-spectok-131072.sh boots each arm twice, but boot 1 uses a
# PROVISIONAL BATCHTOK computed from an assumed block size of 1616. For K=4 and K=5 the
# block really is 1616, so provisional == derived and boot 2 was genuinely warm -- its
# "GPU KV cache size" is the true pool. For K=6 (block 1632 -> BATCHTOK 4906) and K=7
# (block 1648 -> BATCHTOK 4956) the derived value DIFFERS from the provisional one, so
# boot 2 saw a new graph shape and COLD-COMPILED AGAIN: it reported ~6.88 GiB available
# instead of ~8.98, understating the pool by roughly 2.1 GiB.
#
# The BENCH numbers from that run are fine -- compilation completes before /health goes
# green, so BetterBench always ran on a fully compiled graph. Only the POOL figure is
# contaminated. This script re-boots C and D twice each with the CORRECT BATCHTOK already
# in place (the graph cache is warm from the sweep), and reads geometry only. No bench.
set -uo pipefail

OUT=${OUT:-./out}
LAUNCH=${LAUNCH:-./startup-qwen3.8-27b-vllm.sh}
NAME=qwen38-27b-window
PORT=8100
BASE=http://127.0.0.1:$PORT
mkdir -p "$OUT"

prod_env() {
  sed -n '/^prod_env() {$/,/^ENV$/p' ${RUNNER:-./runner.sh} \
    | sed '1,2d;$d'
}

stop_container() {
  podman stop -t 30 "$NAME" >/dev/null 2>&1
  podman rm -f "$NAME"      >/dev/null 2>&1
  for _ in $(seq 1 60); do
    podman ps --format '{{.Names}}' | grep -qx "$NAME" || break
    sleep 2
  done
}

boot_once() {
  local log="$1"; shift
  stop_container
  ( set -a
    eval "$(prod_env)"
    NAME="$NAME"
    for kv in "$@"; do export "${kv%%=*}=${kv#*=}"; done
    set +a
    exec "$LAUNCH" --port "$PORT"
  ) > "$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then echo "  healthy"; return 0; fi
    kill -0 "$pid" 2>/dev/null || { echo "  LAUNCHER EXITED -- see $log"; return 1; }
    sleep 2
  done
  echo "  TIMEOUT waiting for /health -- see $log"; return 1
}

geometry() {
  grep -E "Setting attention block size|Available KV cache memory|GPU KV cache size|Maximum concurrency|padding layers" "$1" \
    | sed 's/^.*\] //' | sed 's/^/  /'
}

fix_arm() {
  local arm="$1" K="$2" bt="$3"
  echo "=== POOL FIXUP $arm : SPECTOK=$K BATCHTOK=$bt ==="
  for n in 1 2; do
    echo "--- fixup boot $n ---"
    boot_once "$OUT/$arm.poolfix$n.log" "SPECTOK=$K" "BATCHTOK=$bt" || {
      echo "  $arm FAILED fixup boot $n"; return 1; }
    geometry "$OUT/$arm.poolfix$n.log"
  done
  echo
}

echo "### SPECTOK pool fixup, started $(date -Is)"
fix_arm C-spectok6 6 4906
fix_arm D-spectok7 7 4956
stop_container
echo "### done $(date -Is)"
