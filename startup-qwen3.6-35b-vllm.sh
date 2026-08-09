#!/usr/bin/env bash
# Standalone vLLM startup script: Qwen3.6-35B-A3B (MoE, GDN hybrid, vision), int4 W4A16,
# on ONE AMD Radeon AI PRO R9700 (gfx1201, 32 GiB), via rootless podman + ROCm.
#
# Extracted from a homelab multi-model setup (originally driven by llama-swap, see
# https://github.com/mostlygeek/llama-swap) so it can run standalone: no llama-swap,
# no other tooling required. Every env var below has a working default from that setup;
# override any of them on the command line, e.g. `MAXLEN=131072 ./startup-qwen3.6-35b-vllm.sh`.
#
# Prerequisites:
#   - podman with ROCm/kfd access (rootless: your user must be in the `render`/`video`
#     groups; --group-add keep-groups below is required for rootless podman specifically --
#     without it, numeric GIDs don't map through the user namespace and you'll see
#     "detected: 1 GPU(s)" followed by an architecture-check failure).
#   - the image: docker.io/stilldeadcode/vllm-radiance:0.5.8 (a ROCm/gfx1201-focused vLLM
#     build with AITER + "radiance" perf hooks; see its Docker Hub page for provenance).
#   - weights downloaded locally to MODEL_DIR, e.g.:
#       hf download Avesed/Qwen3.6-35B-A3B-INT4-W4A16 --local-dir ./models/qwen3.6-35b-a3b-int4
#     compressed-tensors pack-quantized, int4, group_size 32, symmetric. 22.28 GiB on disk.
#     Only the 256 expert FFNs are int4 -- attention, router, shared-expert, GatedDeltaNet,
#     lm_head, embeddings, vision tower and the (unused, see below) MTP head are bf16.
#
# WHY THIS CHECKPOINT'S KERNEL PATH IS WHAT IT IS:
#   `targets: ["Linear"]` in the checkpoint's quant config matches ONLY the expert FFNs --
#   every dense Linear is in a 502-entry ignore list -- so this model takes vLLM's
#   fused-MoE path, not the dense WNA16 linear-kernel path. Confirm in the boot log:
#       "Using CompressedTensorsWNA16MoEMethod"   <- expected PRESENT
#       "RDNAHybridW4A16LinearKernel"              <- expected ABSENT (no dense int4 here)
#
# NO TUNED MoE KERNEL CONFIG EXISTS for this shape (E=256 experts, N=512 intermediate) on
# this GPU as of image 0.5.8. vLLM ships Triton-autotuned block-size configs keyed by
# (device_name, num_experts, intermediate_size, dtype) under
# vllm/model_executor/layers/fused_moe/configs/*.json -- there are three for this exact
# card (device_name=AMD_Radeon_R9700) but all three are dtype=fp8_w8a8, none int4, and none
# match this (E,N). Expect "Using default MoE config. Performance might be sub-optimal!" in
# the log -- that's correct, not a misconfiguration. AITER's own precompiled native FP8 GEMM
# kernels (aiter_meta/hsa/<gfx-arch>/fp8gemm_blockscale/) don't reach this card either: only
# gfx942/gfx950/gfx1250 (MI300-class) have entries, no gfx1201. So on this shape/dtype you
# are on the fully generic, untuned kernel -- the single biggest open perf question here.
# The fix is either an upstream tuned-config contribution, or running vLLM's own
# benchmark_moe.py autotuner locally (Ray-based; as of this image, Ray's ROCm GPU detection
# fails with KeyError 'GPU' -- tuning locally needs that worked around first).
#
# KV CACHE: fp8 KV, 10 full-attention layers of 40 (rest are linear/GDN attention, which
# doesn't use the KV cache the same way) -> ~10.0 KiB/token. At gpu_memory_utilization 0.95
# on a 32 GiB card with ~22.3 GiB of int4 weights, expect a pool in the ~500-550k token
# range -- read the ACTUAL number from the boot log line "GPU KV cache size: N tokens";
# don't budget this from layer arithmetic alone (see the 27B script's header for why that
# undercounts on a GDN hybrid by ~20%+).
#
# MTP IS DELIBERATELY OFF. This checkpoint ships mtp.safetensors, and this image supports
# it (qwen3_5_mtp / Qwen3_5MoeMTP), but the draft weights and its extra attention layer come
# straight out of the same KV pool: ~200,000 tokens of a ~540,000-token pool, a 37% cut, to
# go from ~2 full-context agents to ~1.3. Not worth it here -- see the 27B script for the
# opposite call on a smaller model where the trade is affordable.
#
# GDN HYBRID CONSTRAINTS (load-bearing, not tuning knobs):
#   * --mamba-cache-mode align: GDN state is allocated PER SEQUENCE SLOT. The vLLM default
#     of 256 concurrent slots would eat ~12 GiB of GDN state alone and OOM a 32 GiB card --
#     MAXSEQS must stay small (4 here).
#   * --max-num-batched-tokens must be >= 2240; that's a floor imposed by mamba-align, not a
#     throughput tuning choice.
#
# Attention backend: swept three ROCm options on this card for this shape (head_size 256,
# fp8 KV). ROCM_AITER_UNIFIED_ATTN (AMD's aiter Triton kernel) beat vLLM's own TRITON_ATTN
# by 88-92% on deep prefill and the legacy-layout ROCM_ATTN by 97-99% on decode. Only
# TRITON_ATTN supports vLLM's KV connector/offload feature, so CPU KV offload and this
# attention backend are mutually exclusive on this card -- a real trade, not a bug.
#
# --no-async-scheduling: on this box's host (a 4-vCPU guest), enabling vLLM's async
# scheduling measured 42-45% SLOWER decode -- the per-step CPU-side scheduling work is the
# bottleneck here, so overlapping it with GPU execution hurts rather than helps. Your
# mileage may vary with more host CPU; it's a knob (ASYNCSCHED=1 to try the vLLM default).
#
# Checkpoint defect note: this repo's tokenizer.json ships with truncation/padding baked in
# at max_length 512 (residue of the quantization calibration run). It doesn't affect text,
# but silently truncates image-token runs above ~511 tokens, breaking vision on anything
# larger than roughly 672px. If you hit "Mismatch in `image` token count" on vision input,
# null out tokenizer.json's "truncation" and "padding" keys (keep a .orig backup first).
set -euo pipefail

PORT="${PORT:-8000}"
NAME="${NAME:-qwen36-35b-vllm}"
MODEL_DIR="${MODEL_DIR:-./models/qwen3.6-35b-a3b-int4}"
CACHE_DIR="${CACHE_DIR:-./vllm-cache}"
IMAGE="${IMAGE:-docker.io/stilldeadcode/vllm-radiance:0.5.8}"
MAXLEN="${MAXLEN:-262144}"     # = max_position_embeddings.
GPUUTIL="${GPUUTIL:-0.95}"
ATTN="${ATTN:-ROCM_AITER_UNIFIED_ATTN}"   # or TRITON_ATTN, ROCM_ATTN -- see header.
MAXSEQS="${MAXSEQS:-4}"        # per-sequence GDN state -- keep small, see header.
BATCHTOK="${BATCHTOK:-2560}"   # >= 2240 required by --mamba-cache-mode align.
AITERMOE="${AITERMOE:-0}"      # 1 costs a ~125s AITER JIT build + compile-cache invalidation
                               # on first load; measured no win on this shape.
ASYNCSCHED="${ASYNCSCHED:-0}"  # 0 = pass --no-async-scheduling (see header).
SPEC="${SPEC:-off}"            # off | ngram | ngram_gpu | mtp (needs mtp.safetensors, see
                               # header -- not shipped enabled here on purpose).
SPECTOK="${SPECTOK:-4}"

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
      exit 2
    fi
    SPEC_ARGS="--speculative-config {\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${SPECTOK},\"attention_backend\":\"${ATTN}\",\"disable_padded_drafter_batch\":true}" ;;
  *) echo "[launcher] unknown SPEC=${SPEC}" >&2; exit 2 ;;
esac
[[ -n "$SPEC_ARGS" ]] && echo "[launcher] speculation: ${SPEC} x${SPECTOK}" >&2

if [[ "$ASYNCSCHED" == "1" ]]; then
  ASYNC_ARG=""
else
  ASYNC_ARG="--no-async-scheduling"
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
  -e VLLM_ROCM_USE_AITER_MOE="$AITERMOE" \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e RADIANCE_PRESHUFFLE=1 -e RADIANCE_ATTN_TUNE=1 -e RADIANCE_GDN_WMMA=1 \
  -e RADIANCE_VIT_FLASH=1 -e RADIANCE_FUSE_RMS_QUANT=1 -e RADIANCE_DYNAMIC_DRAFT=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  "$IMAGE" \
    /model \
    --served-model-name qwen3.6-35b-vllm qwen3.6-35b \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPUUTIL" \
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
