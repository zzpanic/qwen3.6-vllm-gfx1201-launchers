#!/usr/bin/env bash
# SPECTOK 4/5/6/7 at MAXLEN=131072 -- 2026-08-29.
#
# WHY 131072 AND NOT 204800: the K>=7 exclusion in [[dflash2-spectok-closed]] was a CONTEXT
# constraint, never a speed finding. The KV group-padding fix (g=5 -> 8) cut per-request
# cost ~21%, so the cost model now says every K in 4..8 clears 204800. pat chose to drop to
# 131072 to remove context as a variable and to sweep where MAXSEQS=2 is actually served
# (predicted 1.74x-1.89x concurrency, vs 1.22x-1.32x at 204800).
#
# BATCHTOK IS DERIVED PER ARM, NOT FIXED. The rule is BATCHTOK = n*block + 2*(K-1), and the
# attention block size itself MOVES WITH K (1616 at K=4, 1648 at K=7 -- measured 08-28).
# So arm 1 boots with a provisional value, we read the real block size out of that boot log,
# and arm 2 (the one that is benched) uses n=3 x the true block. Missing by 6 tokens drops a
# whole block ([[batchtok-align-block-plus-spec-reserve]]).
#
# Bench is BetterBench, --no-concurrency --no-prefill (pat: acceptance and speed only), and
# --greedy so cross-arm comparison is not swamped by sampling variance -- mean accepted
# length carries +-7% run-to-run noise and has cost this campaign retracted claims before.
# ACCEPTANCE IS NOT TAKEN FROM BETTERBENCH: it comes from vLLM's own /metrics counters,
# diffed pre/post, which are exact rather than sampled.
#
# Arm Z re-runs K=4 LAST as a same-config control. If Z disagrees with A, the window is
# drift-dominated and no arm ordering in between can be trusted.
set -uo pipefail

OUT=${OUT:-./out}
# NOTE (2026-09-05): this script was renamed to startup-qwen3.8-27b-int4.sh once a
# second 27B stack existed. Left verbatim below as the record of what was actually run;
# override LAUNCH= to re-run it today.
LAUNCH=${LAUNCH:-./startup-qwen3.8-27b-vllm.sh}
TOK=${MODEL_DIR:-./models/qwen3.8-27b-autoround}
SERVED=qwen3.8-27b-vllm
NAME=qwen38-27b-window
PORT=8100
BASE=http://127.0.0.1:$PORT
BB=${BB:-betterbench}
BBCFG=${BBCFG:-./config-prod.json}
CATS=(code reasoning chat math)
mkdir -p "$OUT"

prod_env() {
  cat <<'ENV'
SERVED=qwen3.8-27b-vllm
IMAGE=docker.io/stilldeadcode/vllm-radiance:0.9.3
DFLASH2_PATCH=1
DFLASH2_PATCHROOT=${DFLASH2_PATCHROOT:-./patches-0.9.3/vllm}
RADIANCE_SKINNY_GEMM=all
MODEL_DIR=${MODEL_DIR:-./models/qwen3.8-27b-autoround}
QUANT=auto_gptq
REASONING_EFFORT=low
MAXSEQS=2
MAXLEN=131072
GPUUTIL=0.95
KVBYTES=0
KVOFFLOAD=0
KV_GROUP_SIZE=auto
ATTN=R4D
ASYNCSCHED=0
SPEC=dflash2
DRAFT_DIR=${DRAFT_DIR:-./models/qwen38-27b-dflash2-int4}
DRAFT_ATTN=TRITON_ATTN
DRAFT_SAMPLE_METHOD=probabilistic
NGMIN=2
NGMAX=8
LMONLY=0
MAXPIX=4194304
MMIMGMAX=2
MMVIDMAX=0
ENV
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
  grep -E "Setting attention block size|Padding mamba page size|Available KV cache memory|GPU KV cache size|Maximum concurrency|padding layers|estimated maximum model length" "$1" \
    | sed 's/^.*\] //' | sed 's/^/  /'
}

blocksize_from() {  # echo the attention block size vLLM actually chose
  grep -oE "Setting attention block size to [0-9]+" "$1" | grep -oE "[0-9]+$" | tail -1
}

run_arm() {
  local arm="$1" K="$2"
  echo "=== ARM $arm : SPECTOK=$K MAXLEN=131072 ==="
  echo "--- boot 1 (cold graph, also tells us the true block size) ---"
  boot_once "$OUT/$arm.boot1.log" "SPECTOK=$K" "BATCHTOK=$((3*1616 + 2*(K-1)))" || {
    echo "  arm $arm FAILED boot 1"; geometry "$OUT/$arm.boot1.log"; return 1; }
  geometry "$OUT/$arm.boot1.log"

  local blk bt
  blk="$(blocksize_from "$OUT/$arm.boot1.log")"; blk="${blk:-1616}"
  bt=$((3*blk + 2*(K-1)))
  echo "  block size $blk -> BATCHTOK $bt  (3*$blk + 2*($K-1))"
  echo "$arm K=$K block=$blk batchtok=$bt" >> "$OUT/batchtok.txt"

  echo "--- boot 2 (warm, aligned BATCHTOK, this is the one benched) ---"
  boot_once "$OUT/$arm.boot2.log" "SPECTOK=$K" "BATCHTOK=$bt" || {
    echo "  arm $arm FAILED boot 2"; geometry "$OUT/$arm.boot2.log"; return 1; }
  geometry "$OUT/$arm.boot2.log"

  curl -s "$BASE/metrics" > "$OUT/$arm.metrics.pre.txt"
  "$BB" run --endpoint "$BASE/v1" --model "$SERVED" --config "$BBCFG" \
    --categories "${CATS[@]}" --runs 8 --warmup 2 --greedy \
    --no-concurrency --no-prefill --max-model-len 131072 \
    --out "$OUT/$arm.bb.json" > "$OUT/$arm.bb.log" 2>&1
  echo "  betterbench exit $?"
  curl -s "$BASE/metrics" > "$OUT/$arm.metrics.post.txt"
  echo
}

echo "### SPECTOK sweep @131072, started $(date -Is)"
run_arm A-spectok4 4
run_arm B-spectok5 5
run_arm C-spectok6 6
run_arm D-spectok7 7
run_arm Z-spectok4-control 4
stop_container
echo "### done $(date -Is)"
