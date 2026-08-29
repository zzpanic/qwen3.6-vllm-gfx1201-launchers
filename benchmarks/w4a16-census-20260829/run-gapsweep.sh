#!/usr/bin/env bash
# Isolated-kernel tile sweep over the keys the 2026-08-29 DFlash2 census showed UNCOVERED.
# Usage: ./run-gapsweep.sh <M_VALUES csv>
# No model is loaded, but the GPU must be free (stop llama-swap + podman stop first).
# Image is pinned to PRODUCTION's 0.9.3 -- the 0.5.8 pin in the 2026-08-14 runner is stale.
set -euo pipefail
MV="${1:-10,45,64,1616,4848}"
OUT="${OUT:-./out}"
PATCHDIR="${PATCHDIR:-../../patches}"   # holds bench_gapfill_gs128.py
IMAGE="docker.io/stilldeadcode/vllm-radiance:0.9.3"
CACHE_DIR="${CACHE_DIR:-./vllm-cache}"
HIST="${HIST:-./census-shapes.json}"
mkdir -p "$OUT"
[[ -s "$HIST" ]] || { echo "!!! census file missing/empty: $HIST" >&2; exit 2; }
trap 'podman rm -f w4a16-gapsweep >/dev/null 2>&1 || true' EXIT INT TERM
podman rm -f w4a16-gapsweep >/dev/null 2>&1 || true
echo "=== gap sweep: M_VALUES=$MV image=$IMAGE ==="
podman run --rm --name w4a16-gapsweep \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --shm-size 4g --security-opt seccomp=unconfined \
  -v "$(cd "$PATCHDIR" && pwd):/patch:ro" -v "$(cd "$OUT" && pwd):/out" \
  -v "$(cd "$(dirname "$HIST")" && pwd):/census:ro" -v "$CACHE_DIR:/cache" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
  -e M_VALUES="$MV" \
  --entrypoint python3 \
  "$IMAGE" /patch/bench_gapfill_gs128.py "/census/$(basename "$HIST")" \
  2>&1 | tee "$OUT/gapsweep.tsv"
