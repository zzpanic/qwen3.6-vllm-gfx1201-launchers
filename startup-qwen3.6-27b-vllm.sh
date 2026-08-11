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
#   - weights downloaded locally to MODEL_DIR. TWO checkpoints are supported and both are
#     known to run; the default is the first:
#
#       # RECOMMENDED (default) -- Intel AutoRound, int4 GPTQ-format, group_size 128,
#       # symmetric, MTP head quantized INLINE (no separate mtp.safetensors):
#       hf download Intel/Qwen3.6-27B-int4-AutoRound --local-dir ./models/qwen3.6-27b-autoround
#       ./startup-qwen3.6-27b-vllm.sh
#
#       # ALTERNATIVE -- Avesed, compressed-tensors pack-quantized, int4, group_size 32,
#       # symmetric. 18.39 GiB + a separate 810 MiB mtp.safetensors:
#       hf download Avesed/Qwen3.6-27B-INT4-W4A16 --local-dir ./models/qwen3.6-27b-int4
#       MODEL_DIR=./models/qwen3.6-27b-int4 QUANT=compressed-tensors ./startup-qwen3.6-27b-vllm.sh
#
#     Either way, MTP is ON by default (see below) and this script checks
#     model.safetensors.index.json for "mtp.*" entries rather than for a filename --
#     Avesed ships the head as a separate file, AutoRound shards it inline, and the index
#     is the one check that is correct for both.
#
# WHY AutoRound IS THE DEFAULT, AND WHAT IT COSTS. AutoRound is a better quantization
# method on paper (sign-gradient-descent rounding rather than round-to-nearest with a
# per-group scale), which is the reason to prefer it -- but no quality benchmark was run
# here, so treat that as the published claim, not a measurement made in this repo. What
# WAS measured on this card, AutoRound vs Avesed, same launcher, same everything else:
#     KV cache pool   +17.4%  (255,387 vs 217,552 tokens at MAXLEN=131072,
#                              1.95x vs 1.66x concurrency)
#     prefill         +0.6% at >=20k depth, +2.1% at 5k
#     engine steps/s  +6.3%  (acceptance-controlled)
#     accepted len    -4.9%  <- its MTP draft head is int4 where Avesed's was BF16
#     net             wall-clock decode a wash at >=20k
# So: a materially larger KV pool, and speed roughly unchanged. The pool difference is
# the concrete reason to prefer it even setting quality aside.
#
# *** AutoRound NEEDS A THREE-KEY config.json PATCH ON THIS STACK, AND WITHOUT IT THE
# *** MODEL LOADS SILENTLY WRONG -- NOT WITH AN ERROR. See the ROUTING GUARD below;
# *** this script applies the patch for you by default (CONFIG_FIX=1).
#
# WHY THIS CHECKPOINT'S KERNEL PATH IS *THE OPPOSITE* OF THE 35B MOE SCRIPT:
#   This model is dense -- no routed experts. Its quant config's ignore list is far
#   narrower (exempts only lm_head, the MTP head, the vision tower, and two tiny GDN
#   projections), so essentially everything else -- including the large GDN/attention
#   Linears -- runs int4 through vLLM's dense WNA16 kernel, not the fused-MoE path. Confirm
#   in the boot log:
#       "Using RDNAHybridW4A16LinearKernel"        <- expected PRESENT
#       "CompressedTensorsWNA16MoEMethod"           <- expected ABSENT (no experts)
#   The full line differs by checkpoint -- same kernel, different owner:
#       AutoRound:  auto_gptq.py ... Using RDNAHybridW4A16LinearKernel for AutoGPTQLinearMethod
#       Avesed:     compressed_tensors_wNa16.py ... Using RDNAHybridW4A16LinearKernel for
#                   CompressedTensorsWNA16
#   Grep for the class name, not the surrounding text. If it is ABSENT on AutoRound, the
#   config.json routing patch below did not take -- see the ROUTING GUARD.
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
#
# TUNED_TILES=1 (default) bind-mounts patches/rdna_hybrid_w4a16.py over the RDNA hybrid
# W4A16 kernel's own untuned gfx1201 Triton tile heuristic (it ships tuned on a different
# model's shapes and group_size). The table carries BOTH checkpoints -- it is keyed on
# (group_size, K, N, M-bucket), so the AutoRound (gs=128) and Avesed (gs=32) rows coexist
# and neither can be hit by the other. Measured end-to-end, real server, baseline
# bracketed on both sides:
#     AutoRound / gs=128:  prefill +9.6 / +7.6 / +5.9 / +4.6% at 3.9k / 15.6k / 62k /
#                          120k prompt tokens (complete separation, 8/8 reps), and
#                          decode engine step rate +4.3%.
#     Avesed / gs=32:      prefill +9.7% at 4k down to +3.5% at 100k.
#
# NOTE, because an earlier version of this header said the opposite: this table moves
# DECODE too, not just prefill. The kernel's skinny-vs-Triton dispatch is a CONJUNCTION,
# `M <= 5 AND K*M <= 32768`, and decode breaks one or the other -- M is
# MAXSEQS x (SPECTOK+1) = 10 whenever two sequences decode at once, and down_proj's
# K=17408 blows the LDS budget even at M=2. A runtime census counted 2728 decode-shaped
# calls into the Triton kernel. Don't treat decode as an inert control when A/B-ing this.
#
# See patches/README.md for the full methodology, the runtime shape census, the
# isolated-kernel sweep data, and the caveats. It's a community-measured override, not
# something upstream has reviewed -- set TUNED_TILES=0 to run the kernel exactly as the
# image ships it. TUNED_TILES=1 is also what supplies the symmetric-GPTQ zero-point fix
# the AutoRound checkpoint needs, so TUNED_TILES=0 + AutoRound will crash on the first
# decode-shaped GEMM; see patches/README.md.
set -euo pipefail

PORT="${PORT:-8000}"
NAME="${NAME:-qwen36-27b-vllm}"
MODEL_DIR="${MODEL_DIR:-./models/qwen3.6-27b-autoround}"
QUANT="${QUANT:-auto_gptq}"    # auto_gptq   for Intel/Qwen3.6-27B-int4-AutoRound (default)
                               # compressed-tensors for Avesed/Qwen3.6-27B-INT4-W4A16
                               # It must AGREE with the checkpoint's config.json or vLLM
                               # raises a mismatch ValueError at load. For AutoRound that
                               # means the config.json patch below has to be applied too;
                               # the raw upstream file says "auto-round", which is neither.
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
GENCFG="${GENCFG:-1}"          # 1 (default) = apply Qwen3.6's "Thinking Mode, Precise
                               # Coding" sampling preset as the SERVER DEFAULT via
                               # --override-generation-config, instead of the checkpoint's
                               # own generation_config.json (temperature 1.0/top_k 20/
                               # top_p 0.95 -- Qwen's general-chat defaults). SERVER DEFAULT
                               # ONLY: any client request that sends its own value for a key
                               # is unaffected. 0 = leave the checkpoint's shipped config in
                               # force. Tune the five presets below to switch to Qwen's other
                               # documented presets (General Tasks / Instruct non-thinking).
GEN_TEMP="${GEN_TEMP:-0.6}"
GEN_TOPP="${GEN_TOPP:-0.95}"
GEN_TOPK="${GEN_TOPK:-20}"
GEN_MINP="${GEN_MINP:-0.0}"
GEN_REPPEN="${GEN_REPPEN:-1.0}"
                               # presence_penalty is deliberately absent: it's not in vLLM's
                               # --override-generation-config whitelist (server-default keys
                               # are temperature/top_k/top_p/min_p/repetition_penalty/
                               # max_new_tokens only), so it can only be set per-request by
                               # the client. Qwen's coding preset wants 0.0, which already
                               # matches the OpenAI-protocol client default, so there's no gap
                               # here -- but Qwen's other two presets want 1.5, and that value
                               # cannot be baked into the server this way.
SPEC="${SPEC:-mtp}"            # mtp (default) | off | ngram | ngram_gpu
SPECTOK="${SPECTOK:-4}"        # measured winner, see header. Don't assume higher is better.
TOKENIZER_FIX="${TOKENIZER_FIX:-1}"
CONFIG_FIX="${CONFIG_FIX:-1}"      # 1 = auto-apply the AutoRound config.json routing patch
                                   # (idempotent, keeps a .autoround-orig backup). 0 = only
                                   # check and refuse. Inert on non-AutoRound checkpoints.
TUNED_TILES="${TUNED_TILES:-1}"    # 0 to disable; bind-mounts patches/rdna_hybrid_w4a16.py, see header.
ALLOW_STOCK_KERNEL="${ALLOW_STOCK_KERNEL:-0}"   # see the TUNED_TILES=0 guard below.
PATCH_DIR="${PATCH_DIR:-./patches}"
KERNEL_PATH="/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"

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
    # Checked via the INDEX, not via a filename. Avesed ships the MTP head as a separate
    # mtp.safetensors; AutoRound shards its 29 MTP tensors INLINE. A `-f mtp.safetensors`
    # test would reject the AutoRound checkpoint outright. The index is correct for both,
    # and it is also what catches the real silent failure: an index with its mtp.* entries
    # stripped loads MTP as a no-op, with no error and no speculation.
    if ! grep -q '"mtp\.' "$MODEL_DIR/model.safetensors.index.json" 2>/dev/null; then
      echo "[launcher] SPEC=$SPEC needs MTP, but $MODEL_DIR/model.safetensors.index.json" >&2
      echo "[launcher] has no mtp.* entries -- MTP would load as a silent no-op." >&2
      echo "[launcher] Re-download the checkpoint (then re-apply the config.json patch)." >&2
      exit 2
    fi
    # ...and the shards those mtp.* entries point at must actually be on disk.
    if ! python3 -c "
import json,os,sys
idx=json.load(open('$MODEL_DIR/model.safetensors.index.json'))['weight_map']
missing={f for k,f in idx.items() if k.startswith('mtp.') and not os.path.exists(os.path.join('$MODEL_DIR',f))}
if missing: print(' '.join(sorted(missing)), file=sys.stderr)
sys.exit(1 if missing else 0)
"; then
      echo "[launcher] the shard(s) above hold the MTP tensors and are not present." >&2
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

# ROUTING GUARD -- the AutoRound checkpoint DOES NOT WORK unpatched, and it fails SILENTLY.
# Three defects in its shipped config.json, each with its own silent failure mode:
#
#   quant_method: "auto-round" -> "auto_gptq"
#       vLLM 0.26 ships an INC (Intel Neural Compressor) backend whose
#       inc.py override_quantization_method returns "inc" for ANY checkpoint declaring
#       quant_method "auto-round" -- ignoring --quantization entirely -- and config/model.py
#       then assigns that override. You land on INCWNA16LinearScheme instead of
#       RDNAHybridW4A16LinearKernel, with no error and no warning. Everything "works",
#       just on the wrong kernel.
#
#   desc_act: false            (added)
#       GPTQConfig.from_config demands the key. AutoRound never writes it. false is the
#       correct value here -- this checkpoint has zero g_idx tensors, i.e. no act-order.
#
#   modules_in_block_to_quantize: [...]   (added)
#       Without it, is_layer_gptq_quantized evaluates `any(x in prefix for x in [])`,
#       which is False for every layer, and THE WHOLE MODEL LOADS UNQUANTIZED -- silently.
#       (vLLM tries to recover by doing a Hugging Face *hub* lookup on the literal string
#       "/model", the in-container mount path, which of course finds nothing.)
#
# A re-download of the checkpoint reinstates all three, which is exactly when this earns
# its keep. CONFIG_FIX=1 (default) applies them idempotently and keeps the original beside
# the file as config.json.autoround-orig; CONFIG_FIX=0 only checks and refuses.
if [[ -f "$MODEL_DIR/config.json" ]]; then
  AR_STATE="$(python3 -c "
import json
q = json.load(open('$MODEL_DIR/config.json')).get('quantization_config', {}) or {}
m = q.get('quant_method')
if m == 'auto-round':                     print('needs-patch')
elif m == 'auto_gptq':
    ok = 'desc_act' in q and q.get('modules_in_block_to_quantize')
    print('ok' if ok else 'needs-patch')
else:                                     print('not-autoround')
" 2>/dev/null || echo unknown)"

  if [[ "$AR_STATE" == "needs-patch" ]]; then
    if [[ "$CONFIG_FIX" == "1" ]]; then
      cp -n "$MODEL_DIR/config.json" "$MODEL_DIR/config.json.autoround-orig" 2>/dev/null || true
      python3 -c "
import json
p = '$MODEL_DIR/config.json'
c = json.load(open(p))
q = c['quantization_config']
q['quant_method'] = 'auto_gptq'
q.setdefault('desc_act', False)
q.setdefault('lm_head', False)
q.setdefault('modules_in_block_to_quantize', [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
    'linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.out_proj',
])
json.dump(c, open(p, 'w'), indent=2)
"
      echo "[launcher] config.json: applied AutoRound routing patch (quant_method=auto_gptq," >&2
      echo "[launcher]   desc_act, modules_in_block_to_quantize). Original kept as" >&2
      echo "[launcher]   config.json.autoround-orig." >&2
    else
      echo "[launcher] $MODEL_DIR/config.json is NOT patched for auto_gptq routing, and" >&2
      echo "[launcher] CONFIG_FIX=0. Unpatched, vLLM's INC backend hijacks this checkpoint" >&2
      echo "[launcher] and serves it on the WRONG kernel -- or fully unquantized -- with no" >&2
      echo "[launcher] error. Required: quant_method=auto_gptq, desc_act," >&2
      echo "[launcher] modules_in_block_to_quantize. See the ROUTING GUARD comment above." >&2
      exit 2
    fi
  fi

  # Whatever the checkpoint is, --quantization has to agree with it or vLLM raises a
  # mismatch ValueError several minutes into the load.
  DECLARED="$(python3 -c "
import json
q = json.load(open('$MODEL_DIR/config.json')).get('quantization_config', {}) or {}
print(q.get('quant_method') or q.get('format') or 'none')
" 2>/dev/null || echo unknown)"
  case "$QUANT:$DECLARED" in
    auto_gptq:auto_gptq|compressed-tensors:pack-quantized|compressed-tensors:compressed-tensors) ;;
    *:unknown|*:none) ;;
    *) echo "[launcher] WARNING: QUANT=$QUANT but config.json declares '$DECLARED'." >&2
       echo "[launcher]   AutoRound wants QUANT=auto_gptq; Avesed wants compressed-tensors." >&2 ;;
  esac
fi

# The tuned-tiles file also carries the symmetric-GPTQ zero-point fix that the AutoRound
# checkpoint needs (GPTQ always ships a qzeros param, constant 7, even when symmetric; the
# stock kernel forwards that raw packed int32 straight into the skinny GEMM and dies on
# `assert zp.shape == (N, num_groups)`). So TUNED_TILES=0 is a supported baseline arm on
# the Avesed checkpoint and a crash on AutoRound. Refuse rather than crash on load.
if [[ "$TUNED_TILES" != "1" && "$QUANT" == "auto_gptq" && "$ALLOW_STOCK_KERNEL" != "1" ]]; then
  echo "[launcher] TUNED_TILES=0 with the AutoRound (GPTQ-format) checkpoint will crash:" >&2
  echo "[launcher]   the stock kernel has no symmetric-GPTQ zero-point handling." >&2
  echo "[launcher] Use the Avesed checkpoint for a stock-kernel baseline:" >&2
  echo "[launcher]   MODEL_DIR=./models/qwen3.6-27b-int4 QUANT=compressed-tensors TUNED_TILES=0" >&2
  echo "[launcher] ALLOW_STOCK_KERNEL=1 overrides this if you want to see it fail." >&2
  exit 2
fi

TILES_MOUNT_ARG=()
if [[ "$TUNED_TILES" == "1" ]]; then
  if [[ ! -f "$PATCH_DIR/rdna_hybrid_w4a16.py" ]]; then
    echo "[launcher] TUNED_TILES=1 but $PATCH_DIR/rdna_hybrid_w4a16.py not found." >&2
    exit 2
  fi
  TILES_MOUNT_ARG=(-v "$(cd "$PATCH_DIR" && pwd)/rdna_hybrid_w4a16.py:${KERNEL_PATH}:ro")
  echo "[launcher] tuned W4A16 prefill tile table: ON (see patches/README.md)" >&2
fi

# JSON must contain no spaces -- expanded unquoted below, same rule as SPEC_ARGS.
if [[ "$GENCFG" == "1" ]]; then
  GENCFG_ARG="--override-generation-config={\"temperature\":${GEN_TEMP},\"top_p\":${GEN_TOPP},\"top_k\":${GEN_TOPK},\"min_p\":${GEN_MINP},\"repetition_penalty\":${GEN_REPPEN}}"
  echo "[launcher] sampling override (server default only): temp=$GEN_TEMP top_p=$GEN_TOPP top_k=$GEN_TOPK min_p=$GEN_MINP rep_pen=$GEN_REPPEN" >&2
else
  GENCFG_ARG=""
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
  "${TILES_MOUNT_ARG[@]}" \
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
    --quantization "$QUANT" \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPUUTIL" \
    ${LM_ARG} \
    ${GENCFG_ARG} \
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
