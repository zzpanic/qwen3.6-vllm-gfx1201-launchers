#!/usr/bin/env bash
# Standalone vLLM startup script: Qwen3.8-27B (DENSE, GDN hybrid, vision), int4 W4A16,
# on ONE AMD Radeon AI PRO R9700 (gfx1201, 32 GiB), via rootless podman + ROCm.
#
# Qwen3.8-27B is a RETRAINED RELEASE OF THE SAME ARCHITECTURE as Qwen3.6-27B, not a new
# one -- checked against upstream config.json, not assumed:
#     model_type qwen3_5   arch Qwen3_5ForConditionalGeneration
#     hidden 5120   intermediate 17408   layers 64   heads 24/4   head_dim 256
#     vocab 248320
# Every tuning decision in the 3.6 script therefore transfers unchanged, including the
# W4A16 Triton tile table (keyed on GEMM shape, not model name -- see TUNED_TILES below).
# This script is the 3.6 one with a different checkpoint default and ONE new knob,
# REASONING_EFFORT, which the 3.8 chat template understands and the 3.6 template does not.
# Read the 3.6 script's header for the background this one doesn't repeat: kernel gate,
# GDN KV budgeting, MTP/SPECTOK measurements, the mamba-align batched-token floor.
#
# Prerequisites:
#   - podman with ROCm/kfd access (rootless: --group-add keep-groups below is required --
#     numeric GIDs don't map through the rootless user namespace otherwise).
#   - the image: docker.io/stilldeadcode/vllm-radiance:0.5.8
#   - weights:
#       hf download devan-carlin/Qwen3.8-27B-int4-AutoRound --local-dir ./models/qwen3.8-27b-autoround
#       ./startup-qwen3.8-27b-vllm.sh
#
# ---------------------------------------------------------------------------------------
# MEASURED ON THIS CARD (2026-08-15, this launcher, MAXLEN=131072, MAXSEQS=2, MTP x4)
# ---------------------------------------------------------------------------------------
#   depth (prompt tok)     prefill t/s   decode t/s   accepted len   engine steps/s
#   3,183                     1618.9        44.34         3.03            14.64
#   12,520                    1546.7        39.81         2.77            14.40
#   38,997                    1335.1        43.91         3.18            13.80
#   mean                      1500.2        42.69         2.99            14.28
#   2 passes x 3 depths. steps/s is the low-noise metric (counted, not derived);
#   decode and accepted-length carry ~+-6% run-to-run noise -- don't read them to 3 figures.
#
#   Load-time facts: weights 17.93 GiB, GPU KV cache 249,982 tokens = 1.91x at MAXLEN.
#
#   Quality, measured against the LIVE server through this exact config (fp8 KV + MTP):
#     GSM8K 5-shot          94.77%      (AMD Quark-Qronos 94.62, AMD Quark-AWQ 91.21,
#                                        Qwen BF16 base 93.33)
#     BFCL v4 single_turn   25.29% overall / Non-Live AST 87.46 / Live AST 81.87 /
#                           relevance detection 75.00 / irrelevance detection 83.62
#                                       (Qronos 23.80 overall, AWQ 24.06, BF16 base 24.38)
#   Both beat AMD's two published Quark checkpoints. See README.md for the caveats --
#   chiefly that these ran through this serving stack rather than AMD's in-process
#   enforce_eager path, so the columns are not strictly like-for-like.
# ---------------------------------------------------------------------------------------
#
# WHY THIS CHECKPOINT. There is no Intel AutoRound release for 3.8 (Intel published three
# for the 3.6 and none for this one). devan-carlin/Qwen3.8-27B-int4-AutoRound is the same
# RECIPE as Intel's 3.6 -- AutoRound, int4, group_size 128, symmetric, auto_round:auto_gptq
# packing, same block_name_to_quantize, same 48x linear_attn.in_proj_a/b fp16 exclusions --
# applied by a different (and unknown) author. That provenance gap was the reason to be
# suspicious of it; the GSM8K and BFCL numbers above are why that suspicion did not survive
# contact with a benchmark. The recipe transfers; it beats two AMD-authored checkpoints
# built with more sophisticated algorithms (Qronos is Hessian-based PTQ with a paper).
#
# ITS MTP DRAFT HEAD IS int4, AND THAT MATTERS. mtp.layers is inside this checkpoint's
# block_name_to_quantize, so the seven mtp.layers.0 Linears come in quantized. A sibling
# 3.8 checkpoint whose head was left BF16 was measured on this card at 0.51 GiB heavier
# and ~9% slower decode (steps/s 13.91 vs 14.28, decode 38.85 vs 42.69). Verify at load:
# "Model loading took 17.93 GiB" -- an 18.4x GiB figure means you have a BF16-head build.
# The AMD Quark checkpoints (Qronos and AWQ) both ship BF16 MTP heads and are rejected here
# for exactly this reason, their good eval numbers notwithstanding.
#
# *** AutoRound NEEDS A THREE-KEY config.json PATCH ON THIS STACK, AND WITHOUT IT THE
# *** MODEL LOADS SILENTLY WRONG -- NOT WITH AN ERROR. Identical mechanism and identical
# *** required values as the 3.6 script (this checkpoint's modules_in_block_to_quantize
# *** list is byte-identical to Intel's). See the ROUTING GUARD below; applied by default.
#
# REASONING_EFFORT is the one genuinely new knob (default `low` here). Qwen3.8's chat
# template takes a reasoning_effort variable and turns it into a system-prompt steer;
# Qwen3.6's template has no such variable and would ignore the flag SILENTLY, which is why
# the guard below hard-fails rather than warns. Three properties that are not obvious:
#   * It is a PROMPT STEER, not a decode cap. `low` injects "Keep your thinking brief and
#     focused, moving directly to the conclusion without unnecessary elaboration". Nothing
#     enforces it and it does not bound output length.
#   * `medium` sets NO instruction at all -- the template branches on xhigh and low only,
#     so medium is identical to unsteered.
#   * The whole block sits inside `if enable_thinking is undefined or enable_thinking is
#     true`, so with thinking off it does nothing whatsoever.
# SERVER DEFAULT ONLY: vLLM merges --default-chat-template-kwargs UNDER request values, so
# a client sending its own chat_template_kwargs.reasoning_effort still wins.
#
# TUNED_TILES=1 (default) bind-mounts patches/rdna_hybrid_w4a16.py over the image's own
# untuned gfx1201 Triton tile heuristic. THE SAME TABLE SERVES 3.6 AND 3.8 -- it is keyed
# on (group_size, K, N, M-bucket), and 3.8 is the same architecture at the same group_size,
# so every fused GEMM shape is identical. Re-confirmed by runtime census on a 3.8 load:
# 99.45% of kernel work lands on tuned rows. Nothing to re-sweep. It also supplies the
# symmetric-GPTQ zero-point fix this checkpoint needs to run at all -- TUNED_TILES=0 here
# is a crash on the first decode-shaped GEMM, not a slower baseline. See patches/README.md.
set -euo pipefail

PORT="${PORT:-8000}"
NAME="${NAME:-qwen38-27b-vllm}"
SERVED="${SERVED:-qwen3.8-27b-vllm}"
MODEL_DIR="${MODEL_DIR:-./models/qwen3.8-27b-autoround}"
QUANT="${QUANT:-auto_gptq}"    # must AGREE with the checkpoint's config.json or vLLM raises
                               # a mismatch ValueError several minutes into the load. The raw
                               # upstream file says "auto-round", which is neither -- the
                               # ROUTING GUARD below rewrites it.
CACHE_DIR="${CACHE_DIR:-./vllm-cache}"
IMAGE="${IMAGE:-docker.io/stilldeadcode/vllm-radiance:0.5.8}"
MAXLEN="${MAXLEN:-131072}"     # NOT 262144. The model's max_position_embeddings is 262144,
                               # but the measured KV pool is 249,982 tokens and vLLM refuses
                               # to start when max-model-len exceeds it. 131072 is also
                               # chosen to match MAXSEQS=2: 249,982/131,072 = 1.91x, i.e.
                               # almost exactly two full-length sequences. Raising MAXLEN
                               # buys context by spending the second concurrent slot
                               # (163,840 -> 1.53x; 196,608 -> 1.27x).
GPUUTIL="${GPUUTIL:-0.98}"
ATTN="${ATTN:-ROCM_AITER_UNIFIED_ATTN}"
MAXSEQS="${MAXSEQS:-2}"        # per-sequence GDN state -- keep small, see the 3.6 header.
BATCHTOK="${BATCHTOK:-2560}"   # >= 2240 required by --mamba-cache-mode align.
LMONLY="${LMONLY:-0}"          # 1 = --language-model-only, drops the vision tower.
ASYNCSCHED="${ASYNCSCHED:-0}"  # 0 = pass --no-async-scheduling.
GENCFG="${GENCFG:-1}"          # 1 = pin Qwen's "Thinking Mode / General Tasks" sampling preset
                               # as the SERVER DEFAULT via --override-generation-config rather
                               # than inheriting the checkpoint's own generation_config.json.
                               # Server default only: a client sending its own value for a key
                               # is unaffected.
GEN_TEMP="${GEN_TEMP:-1.0}"    # Qwen model card, Thinking Mode / General Tasks. This is the
                               # value every benchmark in README.md was measured at -- set it
                               # to 0.6 for the "Precise Coding" preset (what the 3.6 script
                               # here defaults to) if that suits your workload better, but then
                               # the README numbers no longer describe your server.
GEN_TOPP="${GEN_TOPP:-0.95}"
GEN_TOPK="${GEN_TOPK:-20}"
GEN_MINP="${GEN_MINP:-0.0}"
GEN_REPPEN="${GEN_REPPEN:-1.0}"
                               # presence_penalty is deliberately absent: not in vLLM's
                               # --override-generation-config whitelist, so it can only be
                               # set per-request by the client.
REASONING_EFFORT="${REASONING_EFFORT:-low}"
                               # xhigh (the template's own default) | medium | low | empty.
                               # EMPTY = don't pass the flag at all. See the header.
SPEC="${SPEC:-mtp}"            # mtp (default) | off | ngram | ngram_gpu
SPECTOK="${SPECTOK:-4}"        # measured winner on the 3.6 and carried over unchanged; a
                               # 4-arm ladder had 4 beating 8 by 17-48% decode at every
                               # depth. Not re-derived for 3.8. Don't assume higher is better.
TOKENIZER_FIX="${TOKENIZER_FIX:-1}"
CONFIG_FIX="${CONFIG_FIX:-1}"      # 1 = auto-apply the AutoRound config.json routing patch
                                   # (idempotent, keeps a .autoround-orig backup). 0 = only
                                   # check and refuse.
TUNED_TILES="${TUNED_TILES:-1}"    # 0 to disable; bind-mounts patches/rdna_hybrid_w4a16.py.
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
    # Checked via the INDEX, not via a filename: this checkpoint shards its MTP tensors
    # INLINE, so a `-f mtp.safetensors` test would reject it outright. The index is also
    # what catches the real silent failure -- an index with its mtp.* entries stripped
    # loads MTP as a no-op, with no error and no speculation.
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

  DECLARED="$(python3 -c "
import json
q = json.load(open('$MODEL_DIR/config.json')).get('quantization_config', {}) or {}
print(q.get('quant_method') or q.get('format') or 'none')
" 2>/dev/null || echo unknown)"
  case "$QUANT:$DECLARED" in
    auto_gptq:auto_gptq) ;;
    *:unknown|*:none) ;;
    *) echo "[launcher] WARNING: QUANT=$QUANT but config.json declares '$DECLARED'." >&2 ;;
  esac
fi

# The tuned-tiles file also carries the symmetric-GPTQ zero-point fix that any GPTQ-format
# AutoRound checkpoint needs (GPTQ always ships a qzeros param, constant 7, even when
# symmetric; the stock kernel forwards that raw packed int32 straight into the skinny GEMM
# and dies on `assert zp.shape == (N, num_groups)`). Unlike the 3.6 script there is no
# compressed-tensors alternative checkpoint here, so TUNED_TILES=0 has no working
# configuration at all on 3.8 -- refuse rather than crash on load.
if [[ "$TUNED_TILES" != "1" && "$QUANT" == "auto_gptq" && "$ALLOW_STOCK_KERNEL" != "1" ]]; then
  echo "[launcher] TUNED_TILES=0 with a GPTQ-format AutoRound checkpoint will crash: the" >&2
  echo "[launcher]   stock kernel has no symmetric-GPTQ zero-point handling. Every known" >&2
  echo "[launcher]   3.8-27B int4 checkpoint is GPTQ-format, so there is no stock-kernel" >&2
  echo "[launcher]   baseline available on this model. Use the 3.6 script's Avesed" >&2
  echo "[launcher]   (compressed-tensors) checkpoint if you need one." >&2
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
  echo "[launcher] tuned W4A16 tile table: ON (gs=128 rows, shared with 3.6 -- see patches/README.md)" >&2
fi

# JSON must contain no spaces -- expanded unquoted below, same rule as SPEC_ARGS.
if [[ "$GENCFG" == "1" ]]; then
  GENCFG_ARG="--override-generation-config={\"temperature\":${GEN_TEMP},\"top_p\":${GEN_TOPP},\"top_k\":${GEN_TOPK},\"min_p\":${GEN_MINP},\"repetition_penalty\":${GEN_REPPEN}}"
  echo "[launcher] sampling override (server default only): temp=$GEN_TEMP top_p=$GEN_TOPP top_k=$GEN_TOPK min_p=$GEN_MINP rep_pen=$GEN_REPPEN" >&2
else
  GENCFG_ARG=""
fi

# REASONING_EFFORT GUARD -- the point of this block. --default-chat-template-kwargs is
# passed straight to the Jinja renderer, so an unknown key is NOT an error anywhere in the
# stack: on a template without a reasoning_effort variable (every Qwen3.6 checkpoint) the
# flag would be accepted, logged by us, and have exactly zero effect. Fail instead.
RSNEFF_ARG=""
if [[ -n "$REASONING_EFFORT" ]]; then
  if ! grep -q 'reasoning_effort' "$MODEL_DIR/chat_template.jinja" 2>/dev/null; then
    echo "[launcher] REASONING_EFFORT=$REASONING_EFFORT but $MODEL_DIR/chat_template.jinja" >&2
    echo "[launcher] has no reasoning_effort variable -- the flag would be silently inert." >&2
    echo "[launcher] This knob is Qwen3.8+ only. Set REASONING_EFFORT= (empty) for a 3.6" >&2
    echo "[launcher] checkpoint, or use startup-qwen3.6-27b-vllm.sh." >&2
    exit 2
  fi
  # The template does `raise_exception` on an unrecognised value, and it renders PER
  # REQUEST -- so a typo here would load a healthy server that 500s every chat request
  # instead of failing at boot. Catch it now. The accepted set is read from the template
  # itself, so this stays honest if a later checkpoint changes it.
  if ! grep -q "resolved_reasoning_effort not in (.*'${REASONING_EFFORT}'" \
         "$MODEL_DIR/chat_template.jinja"; then
    echo "[launcher] REASONING_EFFORT='$REASONING_EFFORT' is not one of the values this" >&2
    echo "[launcher] template accepts. Its own check line is:" >&2
    grep -n "resolved_reasoning_effort not in" "$MODEL_DIR/chat_template.jinja" >&2
    echo "[launcher] The template raises on anything else -- that would be a 500 on every" >&2
    echo "[launcher] chat request, on a server that started up perfectly." >&2
    exit 2
  fi
  RSNEFF_ARG="--default-chat-template-kwargs={\"reasoning_effort\":\"${REASONING_EFFORT}\"}"
  echo "[launcher] reasoning effort (server default only): $REASONING_EFFORT" >&2
fi

# Checkpoint defect guard, carried from the 3.6 script: tokenizer.json can ship with
# truncation/padding baked in at max_length 512 (residue of a quantization calibration
# run). Doesn't affect text, but silently truncates image-token runs above ~511, breaking
# vision above ~672px. The devan-carlin 3.8 checkpoint does NOT have this defect -- the
# check is kept because a different 3.8 quant might, and the failure is silent.
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
    --served-model-name "$SERVED" \
    --quantization "$QUANT" \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPUUTIL" \
    ${LM_ARG} \
    ${GENCFG_ARG} \
    ${RSNEFF_ARG} \
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
