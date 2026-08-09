#!/usr/bin/env bash
# Standalone vLLM startup script: Qwen3.6-27B (DENSE, GDN hybrid, vision), int4 W4A16,
# on ONE AMD Radeon AI PRO R9700 (gfx1201, 32 GiB), via rootless podman + ROCm.
#
# Extracted from a homelab multi-model setup (originally driven by llama-swap, see
# https://github.com/mostlygeek/llama-swap) so it can run standalone: no llama-swap,
# no other tooling required. Every env var below has a working default from that setup;
# override any of them on the command line, e.g. `MAXSEQS=1 ./startup-qwen3.6-27b-vllm.sh`.
# See the companion 35B script (same image, MoE checkpoint instead of dense) for the
# vLLM/ROCm/AITER background this header doesn't repeat.
#
# Prerequisites:
#   - podman with ROCm/kfd access (rootless: --group-add keep-groups below is required --
#     numeric GIDs don't map through the rootless user namespace otherwise, and you'll see
#     "detected: 1 GPU(s)" followed by an architecture-check failure).
#   - the image: docker.io/stilldeadcode/vllm-radiance:0.5.8
#   - weights downloaded locally to MODEL_DIR, e.g.:
#       hf download Avesed/Qwen3.6-27B-INT4-W4A16 --local-dir ./models/qwen3.6-27b-int4
#     compressed-tensors pack-quantized, int4, group_size 32, symmetric. 18.39 GiB + the
#     810 MiB MTP head (mtp.safetensors -- MTP is ON by default here, see below; make sure
#     model.safetensors.index.json's 15 "mtp.*" entries are present, not stripped).
#
# WHY THIS CHECKPOINT'S KERNEL PATH IS *THE OPPOSITE* OF THE 35B MOE SCRIPT:
#   This model is dense -- no routed experts. Its quant config's ignore list is far
#   narrower (exempts only lm_head, the MTP head, the vision tower, and two tiny GDN
#   projections), so essentially everything else -- including the large GDN/attention
#   Linears -- runs int4 through vLLM's dense WNA16 kernel, not the fused-MoE path. Confirm
#   in the boot log:
#       "Using RDNAHybridW4A16LinearKernel"        <- expected PRESENT
#       "CompressedTensorsWNA16MoEMethod"           <- expected ABSENT (no experts)
#   Corollary: there's no fused-MoE kernel here at all, so this model sidesteps the 35B
#   script's "no tuned MoE config for this shape" problem entirely -- nothing to tune.
#
# KV CACHE is a GDN hybrid, and layer-arithmetic seriously undercounts it here. Naive
# per-layer math (4 kv_heads x 256 head_dim x 2 x 16 attention layers, plus the MTP head's
# own attention layer) gives ~34.0 KiB/token; the MEASURED cost on this card was
# 41.91 KiB/token, +23%. The gap is entirely mechanisms visible only in the boot log, not in
# the model config: the hybrid allocator pads the attention page size up to match the mamba
# (GDN state) page size, then pads mamba by a few percent to match back, then adds a small
# fixed page-alignment waste on top -- so the GDN state ends up co-allocated into the
# per-token KV accounting rather than sitting in a separate fixed slot-count term. LESSON:
# on any GDN hybrid, don't budget KV from layer arithmetic -- load once, read the boot log's
# "GPU KV cache size: N tokens" line, and treat that as ground truth.
#
# MTP IS ON BY DEFAULT HERE (opposite of the 35B script). The math is simply more favorable
# on this smaller model: mtp.safetensors is under half the size (0.79 GiB vs 1.57 GiB) while
# the KV pool it draws from is proportionally larger, so the trade is affordable where it
# wasn't on the 35B. SPECTOK=4 is a measured choice, not a default guess: a direct 4-arm
# ladder (SPECTOK 2/3/4/8) at four context depths showed num_speculative_tokens=4 winning at
# every depth by 17-48% decode throughput over 8, running at ~1.9x the engine step rate for
# only ~28% less accepted draft length per step -- i.e. the extra draft forward passes are
# not free, and a scheduler-code argument for "8 costs nothing" (source: reading
# max_num_new_slots_for_drafting() returning 0 for this config) was directly wrong when
# measured. 2 and 3 both underperform 4, so it's a real interior peak, not an edge value.
#
# GDN HYBRID CONSTRAINTS (load-bearing, not tuning knobs) -- same as the 35B script:
#   * --mamba-cache-mode align: GDN state is allocated per sequence slot; MAXSEQS must stay
#     small. Each slot costs ~2.4x more here than on the 35B (48 GDN layers vs 30, ssm_inner
#     6144 vs 4096), so this script defaults MAXSEQS=2, not 4.
#   * --max-num-batched-tokens must be >= 2240 (mamba-align floor).
#
# Attention backend, --no-async-scheduling, and the tokenizer.json truncation/padding defect
# (which breaks vision above ~511 image tokens) are all identical to the 35B script's header
# -- this script fixes the tokenizer defect automatically by default (TOKENIZER_FIX=1),
# rather than only warning, because a manual fix step is exactly what gets missed after a
# fresh checkpoint re-download and the failure is silent until someone tries vision.
set -euo pipefail

PORT="${PORT:-8000}"
NAME="${NAME:-qwen36-27b-vllm}"
MODEL_DIR="${MODEL_DIR:-./models/qwen3.6-27b-int4}"
CACHE_DIR="${CACHE_DIR:-./vllm-cache}"
IMAGE="${IMAGE:-docker.io/stilldeadcode/vllm-radiance:0.5.8}"
MAXLEN="${MAXLEN:-131072}"
GPUUTIL="${GPUUTIL:-0.98}"
ATTN="${ATTN:-ROCM_AITER_UNIFIED_ATTN}"
MAXSEQS="${MAXSEQS:-2}"        # per-sequence GDN state -- keep small, see header.
BATCHTOK="${BATCHTOK:-2560}"   # >= 2240 required by --mamba-cache-mode align.
LMONLY="${LMONLY:-0}"          # 1 = --language-model-only, drops the vision tower (~0.8 GiB
                               # of KV pool back if you don't need vision).
ASYNCSCHED="${ASYNCSCHED:-0}"  # 0 = pass --no-async-scheduling.
SPEC="${SPEC:-mtp}"            # mtp (default) | off | ngram | ngram_gpu
SPECTOK="${SPECTOK:-4}"        # measured winner, see header. Don't assume higher is better.
TOKENIZER_FIX="${TOKENIZER_FIX:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

case "$SPEC" in
  off) SPEC_ARGS="" ;;
  ngram|ngram_gpu)
    SPEC_ARGS="--speculative-config {\"method\":\"${SPEC}\",\"num_speculative_tokens\":${SPECTOK},\"prompt_lookup_min\":${NGMIN:-2},\"prompt_lookup_max\":${NGMAX:-8}}" ;;
  mtp|qwen3_5_mtp)
    if [[ ! -f "$MODEL_DIR/mtp.safetensors" ]]; then
      echo "[launcher] SPEC=$SPEC needs $MODEL_DIR/mtp.safetensors -- not present." >&2
      echo "[launcher]   hf download Avesed/Qwen3.6-27B-INT4-W4A16 mtp.safetensors --local-dir $MODEL_DIR" >&2
      exit 2
    fi
    if ! grep -q '"mtp\.' "$MODEL_DIR/model.safetensors.index.json" 2>/dev/null; then
      echo "[launcher] mtp.safetensors exists but the index has no mtp.* entries -- MTP would no-op." >&2
      exit 2
    fi
    SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${SPECTOK},\"attention_backend\":\"${ATTN}\",\"disable_padded_drafter_batch\":true}" ;;
  *) echo "[launcher] unknown SPEC=${SPEC}" >&2; exit 2 ;;
esac
[[ -n "$SPEC_ARGS" ]] && echo "[launcher] speculation: ${SPEC} x${SPECTOK}" >&2

if [[ "$ASYNCSCHED" == "1" ]]; then
  ASYNC_ARG=""
else
  ASYNC_ARG="--no-async-scheduling"
fi

if [[ "$LMONLY" == "1" ]]; then
  LM_ARG="--language-model-only"
else
  LM_ARG=""
fi

# Checkpoint defect: tokenizer.json ships with truncation/padding baked in at max_length
# 512 (residue of the quantization calibration run). Doesn't affect text, but silently
# truncates image-token runs above ~511, breaking vision above ~672px. Fixed automatically
# here (idempotent, keeps a .orig backup); set TOKENIZER_FIX=0 to only warn instead.
if [[ -f "$MODEL_DIR/tokenizer.json" ]]; then
  if python3 -c "
import json,sys
t=json.load(open('$MODEL_DIR/tokenizer.json'))
sys.exit(0 if (t.get('truncation') or t.get('padding')) else 1)
" 2>/dev/null; then
    if [[ "$TOKENIZER_FIX" == "1" ]]; then
      cp -n "$MODEL_DIR/tokenizer.json" "$MODEL_DIR/tokenizer.json.orig" 2>/dev/null || true
      python3 -c "
import json
p='$MODEL_DIR/tokenizer.json'
t=json.load(open(p))
t['truncation']=None; t['padding']=None
json.dump(t, open(p,'w'), ensure_ascii=False)
"
      echo "[launcher] tokenizer.json: stripped baked-in truncation/padding 512 (vision fix)." >&2
    else
      echo "[launcher] WARNING: tokenizer.json truncation/padding baked in -- vision degraded above ~511 image tokens. TOKENIZER_FIX=1 to auto-repair." >&2
    fi
  fi
fi

mkdir -p "$CACHE_DIR"
podman rm -f "$NAME" >/dev/null 2>&1 || true
trap 'podman rm -f "$NAME" >/dev/null 2>&1 || true' EXIT INT TERM

exec podman run --rm --name "$NAME" \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups \
  --shm-size 4g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR:/cache" \
  -p "127.0.0.1:${PORT}:8000" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 \
  -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e RADIANCE_PRESHUFFLE=1 -e RADIANCE_ATTN_TUNE=1 -e RADIANCE_GDN_WMMA=1 \
  -e RADIANCE_VIT_FLASH=1 -e RADIANCE_FUSE_RMS_QUANT=1 -e RADIANCE_DYNAMIC_DRAFT=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  "$IMAGE" \
    /model \
    --served-model-name qwen3.6-27b-vllm \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPUUTIL" \
    ${LM_ARG} \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$MAXSEQS" \
    --max-num-batched-tokens "$BATCHTOK" \
    --attention-backend "$ATTN" \
    --enable-prefix-caching --mamba-cache-mode align \
    ${SPEC_ARGS} \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    ${ASYNC_ARG} \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
