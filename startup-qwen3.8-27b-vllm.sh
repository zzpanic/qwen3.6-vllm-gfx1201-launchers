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
#   - target weights:
#       hf download devan-carlin/Qwen3.8-27B-int4-AutoRound --local-dir ./models/qwen3.8-27b-autoround
#   - draft weights (DFlash2 speculator, the default here -- see below):
#       hf download syvai/Qwen3.8-27B-DFlash2-W4A16 --local-dir ./models/qwen3.8-27b-dflash2-int4
#       ./startup-qwen3.8-27b-vllm.sh
#
# ---------------------------------------------------------------------------------------
# MEASURED ON THIS CARD (2026-08-22, this launcher, MAXLEN=204800, MAXSEQS=2, DFlash2 x4)
# ---------------------------------------------------------------------------------------
#   depth (prompt tok)     prefill t/s   decode t/s   accepted len   engine steps/s
#   3,183                     1685.2        62.40         3.01            20.72
#   12,520                    1605.8        61.10         3.05            20.05
#   38,997                    1378.1        56.68         3.05            18.60
#   77,851                    1151.3        47.62         2.84            16.74
#   steps/s is the low-noise metric (counted, not derived). decode and accepted-length
#   carry ~+-6% run-to-run noise -- a second pass of the same config read 56.42/54.50/
#   52.78/51.71 decode against identical steps/s, so read the STEP RATE, not tok/s.
#
#   Against MTP x4 at MATCHED DEPTH (both at MAXLEN=131072, same window, same card):
#                            MTP x4      DFlash2 x4    ratio
#     engine steps/s (mean)   14.03         19.07       1.36
#     accepted draft len       2.83          3.01       1.07
#     decode t/s (mean)       39.74         57.31       1.44
#   +44.2% decode. DFlash2 drafts K tokens in ONE parallel forward where MTP walks a
#   sequential chain of K, so the step is cheaper AND accepts slightly longer. Prefill is
#   not in that table because the drafter does not touch it -- speculation only runs in
#   decode -- and the two arms' prefill agreed inside noise.
#
#   WHAT IT COSTS: 8.2% of the KV pool. The draft model is resident (+1.08 GiB int4) and
#   its own KV comes out of the same budget: 250,148 tokens against MTP's 272,585 at
#   MAXLEN=204800. At MAXSEQS=2 that is 1.22x concurrency instead of 1.33x. If you would
#   rather have the second full-length slot than the speed, SPEC=mtp is still supported
#   and still correct -- see the SPEC knob below.
#
#   Load-time facts: "Model loading took 19.01 GiB" (target 17.93 + draft ~1.08, reported
#   as one figure), "GPU KV cache size: 250,148 tokens". Compare your own boot against
#   logs/boot-qwen3.8-27b-vllm-dflash2.log.
#
#   COLD-BOOT HAZARD: on the FIRST boot after any change to the image, MAXLEN or the
#   model graph, torch.compile runs cold and transiently costs ~2.28 GiB of the pool,
#   which can fail the boot with "estimated maximum model length is 198016". Re-run it.
#   The second boot loads the compiled graph from cache and succeeds. This is not a
#   misconfiguration and there is nothing to change.
#
#   Quality, measured against the LIVE server through this exact config (fp8 KV + spec):
#     GSM8K 5-shot          93.71% on DFlash2   (94.77% on MTP, same harness/config;
#                                        AMD Quark-Qronos 94.62, AMD Quark-AWQ 91.21,
#                                        Qwen BF16 base 93.33)
#       The 1.06 pp gap is 1.17 sigma on 1319 problems -- not a distinguishable
#       difference at this sample size. Both are temperature-1.0 numbers; greedy scores
#       ~97.6% and GSM8K is a poor discriminator between decoders either way.
#     BFCL v4 single_turn   25.29% overall / Non-Live AST 87.46 / Live AST 81.87 /
#                           relevance detection 75.00 / irrelevance detection 83.62
#                                       (Qronos 23.80 overall, AWQ 24.06, BF16 base 24.38)
#       *** MEASURED ON THE MTP CONFIG AND NOT YET RE-SCORED ON DFlash2. ***
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
# (With SPEC=dflash2 that line reports target AND draft as one figure: 19.01 GiB. Subtract
# ~1.08 GiB for the int4 draft before comparing.)
# The AMD Quark checkpoints (Qronos and AWQ) both ship BF16 MTP heads and are rejected here
# for exactly this reason, their good eval numbers notwithstanding.
# CAVEAT ADDED 2026-08-22: with SPEC=dflash2 (the default now) the target's MTP head is
# never called -- the speculator is the separate draft model. The head is then simply 0.51
# GiB of resident dead weight, so the argument above reduces from "int4 head drafts better"
# to "int4 head is smaller". It still points at the same checkpoint, for a weaker reason.
# The argument returns in full if you set SPEC=mtp.
#
# ---------------------------------------------------------------------------------------
# THE DFlash2 DRAFTER (SPEC=dflash2, the default)
# ---------------------------------------------------------------------------------------
# DFlash2 is a separate small draft model that proposes K tokens in ONE parallel forward
# (K mask-token rows), where MTP walks a sequential chain of K forwards. vLLM verifies the
# proposal with the same rejection sampler either way. Upstream vLLM PR #52816.
#
# THE IMAGE DOES NOT CONTAIN IT. vllm-radiance:0.5.8 is built on vLLM 0.26.0, which
# predates the PR, so this script bind-mounts a 10-file overlay from patches/dflash2/vllm
# over the image's vllm package (DFLASH2_PATCH=1, on by default whenever SPEC=dflash2).
# Every file carries a header naming the upstream commit it came from and any deviation.
# Read patches/dflash2/README.md before changing the image tag -- the overlay is pinned to
# 0.26.0 and will not apply cleanly to a different base.
#
# WHICH DRAFT CHECKPOINT. Two exist and both were measured here, at MAXLEN=131072 against
# the same MTP baseline, everything else held constant:
#
#   draft checkpoint                        +weights steps/s  acc.len  decode t/s
#   (MTP head in the target, no draft model)     -     14.03     2.83     39.74
#   z-lab/Qwen3.8-27B-DFlash2          (bf16) +3.47G   17.58     2.68     47.19  greedy
#   syvai/Qwen3.8-27B-DFlash2-W4A16    (int4) +1.08G   18.74     2.61     48.80  greedy
#   syvai/Qwen3.8-27B-DFlash2-W4A16    (int4) +1.08G   19.07     3.01     57.31  probabilistic
#   "+weights" is the measured "Model loading took" figure minus the 17.93 GiB target --
#   i.e. what actually comes out of the KV pool.
#
# TAKE THE int4 ONE, AND SAMPLE IT PROBABILISTICALLY. int4 is not a quality compromise
# here: it is both faster than bf16 and 2.39 GiB lighter, and that 2.39 GiB goes straight
# back into the KV pool. draft_sample_method is the bigger lever of the two -- `greedy`
# makes the drafter commit to its argmax, `probabilistic` samples from its distribution,
# which the target then accepts more often (2.61 -> 3.01 tokens per step, +15%). It is the
# single largest tuning win in this config and it is one word.
#
# The int4 draft is a PACKED compressed-tensors checkpoint, which merged upstream cannot
# load at all -- see the _dense_kv_rows() note in patches/dflash2/README.md.
#
# SPECTOK=4 IS MEASURED, NOT ASSUMED. K was swept over {4,6,7,8} on this card: 4 wins, and
# K>=7 cannot hold MAXLEN=204800 at all (the draft's own KV grows with K). Don't raise it.
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
MAXLEN="${MAXLEN:-204800}"     # NOT 262144. The model's max_position_embeddings is 262144,
                               # but the measured KV pool is 250,148 tokens with SPEC=dflash2
                               # and vLLM refuses to start when max-model-len exceeds it.
                               # 204800 leaves 1.22x concurrency at MAXSEQS=2 -- i.e. this
                               # default buys context and spends most of the second slot.
                               # Trade the other way by lowering it: 131072 gives 1.91x.
                               # Nothing here depends on the value except how much of the
                               # pool each sequence may claim.
                               # See the COLD-BOOT HAZARD note in the header before you
                               # conclude that a first-boot failure here means it's too high.
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
SPEC="${SPEC:-dflash2}"        # dflash2 (default) | mtp | off | ngram | ngram_gpu
                               # dflash2 = separate draft model, +44% decode over mtp, costs
                               # 8.2% of the KV pool. mtp = the target's own head, no extra
                               # weights, larger pool. Both are supported and correct; see
                               # the DFlash2 section in the header for the tradeoff.
SPECTOK="${SPECTOK:-4}"        # measured winner for BOTH mtp and dflash2 on this card. The
                               # 3.6 mtp ladder had 4 beating 8 by 17-48% decode at every
                               # depth; the 3.8 dflash2 sweep over {4,6,7,8} also lands on 4,
                               # and K>=7 cannot hold MAXLEN=204800. Don't assume higher is
                               # better -- it is not, in either mode.
DRAFT_DIR="${DRAFT_DIR:-./models/qwen3.8-27b-dflash2-int4}"
                               # SPEC=dflash2 only. The DFlash2 draft model. Default expects
                               # syvai/Qwen3.8-27B-DFlash2-W4A16 (int4, +1.08 GiB loaded).
DRAFT_ATTN="${DRAFT_ATTN:-TRITON_ATTN}"
                               # SPEC=dflash2 only, and it must NOT be the target's backend.
                               # DFlash2's draft attention is NON-CAUSAL (it attends across
                               # all K mask rows), and ROCM_AITER_UNIFIED_ATTN refuses a
                               # non-causal mask. TRITON_ATTN takes it. The target keeps
                               # ATTN=ROCM_AITER_UNIFIED_ATTN; the two are set independently.
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
                               # probabilistic | greedy. Measured +15% accepted length and
                               # +17% decode for probabilistic. See the header table.
DFLASH2_PATCH="${DFLASH2_PATCH:-1}"
                               # 1 = bind-mount patches/dflash2/vllm over the image's vllm
                               # package. REQUIRED for SPEC=dflash2 on this image (0.26.0
                               # predates upstream PR #52816). Ignored for other SPEC values,
                               # deliberately: one file in the overlay (v1/core/kv_cache_utils
                               # .py) is NOT DFlash2-gated and would also move MTP's prefix-
                               # hash granularity from 1664 to 832. Keeping the overlay scoped
                               # to SPEC=dflash2 means SPEC=mtp here is byte-for-byte the
                               # configuration the MTP numbers were measured on.
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
  dflash2)
    # The DFlash2 draft model is a SEPARATE checkpoint, not tensors inside the target.
    # Three things are checked, because each fails in a way that is either silent or
    # confusing several minutes into a load:
    if [[ "$DFLASH2_PATCH" != "1" ]]; then
      echo "[launcher] SPEC=dflash2 with DFLASH2_PATCH=0. vLLM 0.26.0 (this image) has no" >&2
      echo "[launcher] DFlash2 support at all -- it will fail on an unknown architecture." >&2
      echo "[launcher] The overlay in patches/dflash2/ is what adds it. See the header." >&2
      exit 2
    fi
    if [[ ! -d "$DRAFT_DIR" ]]; then
      echo "[launcher] SPEC=dflash2 needs a draft model at DRAFT_DIR=$DRAFT_DIR" >&2
      echo "[launcher]   hf download syvai/Qwen3.8-27B-DFlash2-W4A16 --local-dir $DRAFT_DIR" >&2
      exit 2
    fi
    # The architecture string is what routes it to the DFlash2 speculator. A plain Qwen3
    # draft in this slot loads as a generic draft model instead, quietly, and drafts badly.
    if ! grep -q 'DFlash2DraftModel' "$DRAFT_DIR/config.json" 2>/dev/null; then
      echo "[launcher] $DRAFT_DIR/config.json does not declare DFlash2DraftModel in" >&2
      echo "[launcher] architectures -- this is not a DFlash2 draft checkpoint." >&2
      exit 2
    fi
    # An ASYMMETRIC int4 draft would load and run and simply draft from a corrupted KV
    # precompute -- visible only as unexplained low acceptance. The overlay raises on it,
    # but say so here rather than 4 minutes into a load.
    if [[ -f "$DRAFT_DIR/config.json" ]] && \
       ! python3 -c "
import json,sys
c=json.load(open('$DRAFT_DIR/config.json'))
q=c.get('quantization_config') or {}
g=(q.get('config_groups') or {}).get('group_0') or {}
w=g.get('weights') or {}
sys.exit(0 if (not q or w.get('symmetric', True)) else 1)
" 2>/dev/null; then
      echo "[launcher] $DRAFT_DIR is an ASYMMETRIC quantized draft. The context-KV" >&2
      echo "[launcher] precompute in patches/dflash2/ dequantizes scale-only, with no" >&2
      echo "[launcher] zero-point term, so this would corrupt the draft SILENTLY." >&2
      exit 2
    fi
    SPEC_ARGS="--speculative-config {\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":${SPECTOK},\"attention_backend\":\"${DRAFT_ATTN}\",\"draft_sample_method\":\"${DRAFT_SAMPLE_METHOD}\"}"
    # Yes, "dflash" -- upstream's method string for DFlash2 is `dflash`; the `2` lives in
    # the draft model's architecture, not in the method name. vllm/config/speculative.py:61.
    ;;
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
[[ "$SPEC" == "dflash2" ]] && \
  echo "[launcher] draft: $DRAFT_DIR (attn=$DRAFT_ATTN sample=$DRAFT_SAMPLE_METHOD)" >&2

# DFlash2 overlay: bind-mount every .py under patches/dflash2/vllm over the matching path
# in the image's vllm package. Ten files, all from upstream PR #52816 (plus one local
# addition, see patches/dflash2/README.md). Mounted read-only and file-by-file so the rest
# of the package -- including radiance's own gfx1201 patches -- is untouched.
DFLASH2_MOUNT_ARGS=()
DRAFT_MOUNT_ARG=()
[[ "$SPEC" == "dflash2" ]] && DRAFT_MOUNT_ARG=(-v "$(cd "$DRAFT_DIR" && pwd):/draft:ro")
if [[ "$SPEC" == "dflash2" && "$DFLASH2_PATCH" == "1" ]]; then
  PATCHROOT="$(cd "$PATCH_DIR/dflash2/vllm" 2>/dev/null && pwd)" || {
    echo "[launcher] DFLASH2_PATCH=1 but $PATCH_DIR/dflash2/vllm is missing." >&2; exit 2; }
  SITE="/opt/vllm/lib/python3.12/site-packages/vllm"
  while IFS= read -r f; do
    rel="${f#$PATCHROOT/}"
    DFLASH2_MOUNT_ARGS+=(-v "$f:$SITE/$rel:ro")
  done < <(find "$PATCHROOT" -name '*.py' | sort)
  echo "[launcher] DFlash2 overlay: ${#DFLASH2_MOUNT_ARGS[@]} files over $SITE" >&2
  # A new package DIRECTORY cannot be created by bind-mounting files into it -- the
  # spec_decode/dflash2/ package only exists because __init__.py is mounted, and podman
  # will create the intermediate directory for that. If a future overlay adds a file under
  # a directory the image lacks AND no __init__.py beside it, check that assumption.
fi

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
  "${DFLASH2_MOUNT_ARGS[@]}" \
  "${DRAFT_MOUNT_ARG[@]}" \
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
