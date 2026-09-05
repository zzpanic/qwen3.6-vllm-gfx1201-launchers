#!/bin/bash
# startup-qwen3.8-27b-mxfp4.sh -- Qwen3.8-27B in NATIVE MXFP4 (W4A8) with an FP8 DFlash2
# drafter, on ONE AMD Radeon AI PRO R9700 (gfx1201 / RDNA4, 32 GiB).
#
# This is the MXFP4 sibling of startup-qwen3.8-27b-vllm.sh (int4 W4A16) in this repo. Same
# model, same card, different quantisation path and a different kernel stack. Run whichever
# suits you; they cannot run at the same time (each takes the whole GPU).
#
# WHAT THIS IS NOT: it is not a fork of anyone's kernels. The MXFP4 GEMM, the R4D attention
# and the DFlash2 integration are ggz14's radiance work, shipped in the
# stilldeadcode/vllm-radiance image. What this script contributes is the ARRANGEMENT --
# which knobs, at which values, on this card -- and every one of those values below carries
# the measurement that chose it. That is the part that is ours, and per the benchmarks in
# ./benchmarks it is worth a great deal more than it looks.
#
# PROVENANCE
#   image      docker.io/stilldeadcode/vllm-radiance:0.9.3  (vLLM 0.27.1, ROCm, AITER)
#   kernels    codeberg.org/ggz14/radiance-vllm-mxfp4       (clone it; see REPO below)
#   libr4d     pinned by the image. The pin is a CORRECTNESS pin, not a speed pin --
#              v0.4.0 produced NaNs in the GDN path (perplexity 653586) that looked exactly
#              like broken speculative decoding. Do not float it.
#
# TWO CHECKPOINTS are needed under $MODELS, both produced by the repo's setup-mxfp4.sh:
#   Qwen3.8-27B-MXFP4-mtpfp8   AMD's amd/Qwen3.8-27B-Quark-AWQ-MXFP4 with the MTP head
#                              requantized to fp8 by ./fp8_mtp.py. NOT optional for THAT
#                              checkpoint: its exclude list names the mtp.* layers as TENSOR
#                              names among 112 MODULE names, and quark matches modules -- so
#                              the exclusion never fires, vLLM applies the mxfp4 scheme to a
#                              bf16 head, and it asserts on a half-width parameter. A
#                              checkpoint that declares mtp.* in layer_quant_config loads
#                              as-is: point SNAP at it and skip fp8_mtp.py.
#   Qwen3.8-27B-DFlash2-FP8    the block-diffusion drafter used by SPEC_METHOD=dflash.
#                              fp8 and not mxfp4 on purpose -- 4-bit costs more acceptance
#                              than it saves in bandwidth, and AWQ does not rescue it.
#
# Everything below is `${VAR:-default}`, so any of it can be overridden from the environment
# without editing this file. THE DEFAULTS ARE THE MEASURED PRODUCTION CONFIGURATION -- unlike
# most launchers, you are not expected to tune this before it is fast.
#
# WHAT TO CHECK IN THE LOG (in order; the first two are the ones that matter)
#   "Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM"  -> our kernel won the selection
#   "[radiance] native MXFP4 enabled on gfx12x"           -> the aiter fp4 gate was relaxed
#   "[radiance] kv cache groups: size 8, 9 groups"        -> the KV group-padding patch fired
#   "[run] sampling=" / "[run] reasoning-effort="         -> what you will actually be served
#   The stock "current platform does not support native MXFP4/MXFP6" notice still prints and
#   is a FALSE ALARM -- it comes from a separate supports_mx() call, not the kernel gate.
#
#   startup-qwen3.8-27b-mxfp4.sh          serve on http://<host>:$PORT/v1
#   startup-qwen3.8-27b-mxfp4.sh -h       every knob, its default and what it does
#
# See ./TUNING.md for the measurement log behind these defaults and ./BACKGROUND.md for what
# the image is and which decisions here are ours rather than its defaults.

set -euo pipefail

# ---------------------------------------------------------------- usage / arguments
usage() {
  cat <<'USAGE'
 llama-swap-ggz14-27b.sh -- native MXFP4 Qwen3.8-27B on AMD RDNA4 (gfx1201)

GPU count, tensor-parallel size and KV cache size are all detected; nothing below has to be
edited to run on a host with a different number of cards.

  llama-swap-ggz14-27b.sh --port <N>   serve on http://<host>:<N>/v1 (llama-swap's contract;
                                       the container uses host networking and binds <N> directly)
  llama-swap-ggz14-27b.sh [ARGS]       any extra arguments are passed through to `vllm serve`

Everything is an environment variable; these are the ones worth knowing.

  MODELS=~/ai/models-mxfp4  directory holding the checkpoints (bind-mounted at /models)
  PORT=<--port value>       listen port; the --port argument (llama-swap) wins over this
  IMAGE=...:0.9.3           container image (CACHE is keyed to it -- move both together)
  RUNTIME=podman|docker     container runtime (auto-detected)
  CHAT_TEMPLATE=./qwen-fixed-v22.3.jinja
                            chat template; must be readable on the host

  SPEC_METHOD=dflash        speculative drafter: dflash (fastest, needs the DFlash2 checkpoint)
                            or mtp (uses the head inside the target, no extra download)
  SPEC=7 dflash / 4 mtp     speculative depth
  MAXSEQS=8                 max concurrent sequences
  MAXLEN=262144             max context length
  CHUNK=8192                prefill chunk (--max-num-batched-tokens)
  GPU_UTIL=0.98             VRAM fraction; use 0.75 for perplexity work (prompt_logprobs)

  TP=<auto>                 tensor-parallel size; defaults to the largest of 8/4/2/1 that the
                            detected cards can fill (head counts rule out 3, 6 and 12)
  GPUS=0,1                  HIP indices to serve on; defaults to every card with enough VRAM
  MIN_GPU_MIB=8192          VRAM floor for "usable"; excludes iGPUs from the count
  KV_MEM=auto               KV cache size: auto uses a pin measured for your hardware if
                            kv-profiles.tsv has one and lets vLLM profile if not; <bytes> pins
                            explicitly; 0 forces profiling. ./calibrate-kv.sh measures a pin
  ./gpu-detect.sh           print what was detected and which of these it would pick

  R4D_ATTN=1                R4D paged attention backend (0 = AITER unified attention)
  FAST_DRAFT=1              int2 draft head with an exact rerank
  MIN_M=0                   M above which the W4A8 kernel takes over from aiter (0 = always)
  AUTO_R4D=1                build the pinned libr4d on first run (cached); 0 uses the image's
  R4D_SO=<dir>              use your own libr4d checkout instead of building one
  EXTRA="--enforce-eager"   extra `vllm serve` flags (same as passing them as arguments)
  DRY_RUN=1                 print the container command instead of running it
  PREPARE_ONLY=1            do the one-time work (image, libr4d) and stop before serving

Full knob reference: README.md. Design notes and measurements: MXFP4-NOTES.md.
USAGE
}

PASSTHRU=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --port) PORT="$2"; shift 2 ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done

die() { echo "[serve-mxfp4] ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# This file lives outside the repo it was copied from, so the repo's own files (gpu-detect.sh,
# the r4d patch, the chat template, the /patches mount) are resolved from REPO instead.
REPO="$(realpath -m "${REPO:-$(dirname "$(realpath -m "$0")")/radiance-vllm-mxfp4}")"
# Hardware detection: how many usable AMD GPUs there are, which HIP indices they are, what TP
# fits them and the model's head counts, and whether a KV pin has been measured for them. Sets
# RAD_GPU_* / RAD_TP and defines rad_kv_lookup. See gpu-detect.sh for why a VRAM floor and not
# a count of render nodes. Sourced rather than run so a single scan serves every default below.
# shellcheck source=gpu-detect.sh
. "$REPO/gpu-detect.sh"

# ---------------------------------------------------------------- container runtime
# podman and docker differ in three places this script touches: `--replace` is podman-only,
# `--group-add keep-groups` is podman-only (docker wants numeric render/video GIDs), and docker
# needs the stale container removed by hand. Everything else is identical.
RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else die "no container runtime found" "install podman (preferred) or docker, then re-run"
  fi
fi
command -v "$RUNTIME" >/dev/null 2>&1 || die "RUNTIME=$RUNTIME is not on PATH"

RT_FLAGS=()
GROUP_FLAGS=()
if [ "$RUNTIME" = podman ]; then
  RT_FLAGS+=(--replace)
  GROUP_FLAGS+=(--group-add keep-groups)
else
  for g in render video; do
    gid=$(getent group "$g" 2>/dev/null | cut -d: -f3) || true
    if [ -n "$gid" ]; then GROUP_FLAGS+=(--group-add "$gid"); fi
  done
fi

# ---------------------------------------------------------------- host preflight
# Every check here fails with the command that fixes it. They are cheap, and each one stands for a
# failure that otherwise surfaces minutes later as a Python traceback from inside a TP worker.
preflight() {
  [ -e /dev/kfd ] || die "/dev/kfd is missing -- the amdgpu kernel driver is not loaded" \
      "this image ships ROCm userspace, but the kernel driver has to be on the host" \
      "check: ls -l /dev/kfd /dev/dri  and  dmesg | grep amdgpu"
  [ -d /dev/dri ] || die "/dev/dri is missing -- no GPU render nodes on this host"

  # gpu-detect.sh has already scanned. It counts only cards big enough to hold a shard, so a
  # host whose only amdgpu node is an iGPU lands here with zero rather than serving onto 2 GiB
  # of shared system memory and dying somewhere inside weight loading.
  [ "$RAD_GPU_COUNT" -gt 0 ] || die "no AMD GPU with at least ${RAD_MIN_GPU_MIB} MiB of VRAM" \
      "found:$([ -n "$RAD_GPU_SKIPPED" ] && echo "$RAD_GPU_SKIPPED" || echo " nothing on the amdgpu driver")" \
      "lower the floor with MIN_GPU_MIB=<mib>, or name the cards with GPUS=0,1"
  if [ "$RAD_GPU_COUNT" -gt "$TP" ]; then
    echo "[serve-mxfp4] note: $RAD_GPU_COUNT usable GPUs, serving on $TP (indices $GPU_IDS)." >&2
    echo "  TP must divide the model's head counts -- $RAD_TP_ALLOWED are the supported sizes." >&2
  fi

  [ -d "$MODELS" ] || die "MODELS=$MODELS does not exist" \
      "point MODELS at the directory holding your checkpoints, or run ./setup-mxfp4.sh"

  if ! "$RUNTIME" image exists "$IMAGE" >/dev/null 2>&1 &&
     ! "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[serve-mxfp4] pulling $IMAGE (a few GiB, once)"
    "$RUNTIME" pull "$IMAGE" || die "could not pull $IMAGE" "pull it by hand, or set IMAGE=<a local tag>"
  fi

  # A listening port is almost always the previous server or production still holding both GPUs.
  # PREPARE_ONLY is doing the one-time work, not serving, so a busy port is irrelevant there.
  [ "${PREPARE_ONLY:-0}" = 1 ] && return 0
  # The probe opens fd 3 in a SUBSHELL, so there is nothing to close here -- and closing it with
  # a bare `exec 3>&- 2>/dev/null` would apply that redirection to the shell itself and silence
  # every error message after it.
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    # Name the container holding it. "stop the container you find in `podman ps`" was not
    # enough on 2026-09-01: the server on the port had been started by running its script
    # directly, so `systemctl --user stop` was a no-op against it, the port stayed held, and
    # this check aborted a switch that looked like it should have worked. A container started
    # outside systemd is stopped with the runtime, not the unit -- so print the runtime command.
    local holder=""
    holder=$("$RUNTIME" ps --format '{{.Names}}' 2>/dev/null | head -20 | tr '\n' ' ')
    die "port $PORT is already in use" \
        "another server is running -- this one needs every GPU it serves on:" \
        "  running containers: ${holder:-<none: the port is held by a host process>}" \
        "  $RUNTIME stop <name>                   # works however the container was started" \
        "  systemctl --user stop qwen_vllm_paro   # ONLY if that unit started it -- check" \
        "                                         # \`systemctl --user is-active\` first, a" \
        "                                         # hand-started container is not systemd's" \
        "or serve on a different port: PORT=8081 ./serve-mxfp4.sh"
  fi

  [ -r "$CHAT_TEMPLATE" ] || die "chat template not readable: $CHAT_TEMPLATE" \
      "set CHAT_TEMPLATE=<path to a .jinja on the host>, or leave it unset to use the" \
      "one shipped in this repo (qwen-fixed-v22.3.jinja)"
}

# Image and cache MUST move together: cache dirs validate on model + torch/Triton version and must
# not be shared across configurations. Both defaulted to 0.7.4 / -074 long after production moved to
# 0.9.3 / -093, so anyone taking the defaults got a DIFFERENT server than the one being measured.
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
# NAME and SERVED move together with the entry key (the SAFETY rule in ~/ai/config.yaml):
# llama-swap forwards the requested model id verbatim and vLLM validates it, so if they
# drift apart the request 404s AFTER the model is already loaded.
NAME=${NAME:-qwen38-27b-mxfp4}
SERVED=${SERVED:-qwen3.8-27b-mxfp4}
PORT=${PORT:-8080}
# CHUNK 2048, and the reason is NOT alignment. Under R4D attention (R4D_ATTN=1 below) chunk
# alignment is worth exactly 0% -- a full 2x2 against KV pool size found it did nothing at any
# size. The "align BATCHTOK to n*block_size" rule is an AITER property and is inert here. 2048
# is chosen on three secondary criteria: it is the smallest torch.compile activation transient
# (which is the peak the KV pin below must leave room for), it drops AR_MAX_KB from 29696 to
# 24576, and it is a power of two. Read max_num_scheduled_tokens off the boot if you change it.
CHUNK=${CHUNK:-2048}
# 2 concurrent sequences. This is a SINGLE-USER box config: the whole 32 GiB goes to one long
# context rather than many short ones. Note MAXSEQS also moves the attention block size (1648 at
# K=7/seqs=2 under R4D; 1664 under AITER), so never carry a block size across entries -- read it
# off the boot. Raising it costs context roughly linearly.
MAXSEQS=${MAXSEQS:-2}
R4D_ATTN=${R4D_ATTN:-1}
# GDN in_proj merge (radiance_gdnmerge.py): in_proj_qkvz + in_proj_ba as ONE GEMM, removing 96
# GEMM launches and 48 activation quants per forward. Measured 2026-08-29: single-stream decode
# 26.25 -> 25.50 ms/step (-2.9%), prefill unchanged, all 48 layers merge; stacks with WPERM=1
# for 24.55 ms/step (-6.5%) at a 3% prefill cost. Output drift is split-K reassociation only
# (merged N crosses a dks boundary), same class as the decode kernel's own M-dependent split;
# gated with GSM8K 500q paired.  Resolved EARLY because the CACHE default is keyed on it:
# the merge changes the traced graph, and reusing a cache dir compiled without it replays a
# graph that still calls the two ORIGINAL projections -- whose weights the merge freed --
# and the engine dies at startup on an N=0 GEMM.
GDN_MERGE=${RADIANCE_GDN_MERGE_INPROJ:-1}
# AR/GEMM overlap (radiance_aroverlap.py) changes the traced graph too -- same cache rule.
AR_OVERLAP=${RADIANCE_AR_OVERLAP:-0}
# Norm+quant fusion (2026-08-30). Three pieces that only work TOGETHER: hoist the per-linear fp8
# activation quant into the traced graph (RADIANCE_MXFP4_HOIST_QUANT), swap the aiter pattern's
# replacement op for one that works on RDNA4 (RADIANCE_RMS_QUANT_FUSION + patch_rmsquant_fusion),
# and enable the vLLM passes themselves (pass_config.fuse_norm_quant/fuse_act_quant -- the piece
# the Aug-28 experiment missed: its serve config shows 'fuse_norm_quant': False, so that
# "neutral" result was a null test). Changes the traced graph => own cache suffix.
NQF=${RADIANCE_NORMQUANT_FUSION:-1}
# FP8 residual stream (radiance_arnq): fuse each RowParallel linear's post-AR epilogue
# (residual add + Gemma rmsnorm + per-token fp8 quant) into one HIP kernel and hand the next
# linear a pre-quantized (q, scale). Kernel is bit-identical to the traced path; the contract
# change is why it gets its own cache key and its own gate run. Requires NQF=1 and GDN_MERGE=1.
# TRAP: if the arnq installer SKIPS at startup (guard failure), the stock graph lands in the
# -fp8s cache dir, and because that trace never touched radiance_arnq.py the cache key cannot
# tell the difference afterwards -- a later fixed launch silently replays the stock graph
# (measured 2026-08-30: epilogue kernels 0/step, bench byte-identical). After fixing whatever
# made the installer skip, rm the -fp8s cache dir.
FP8S=${RADIANCE_FP8_STREAM:-1}
# NQF=1 and FP8S=1 are the DEFAULTS as of 2026-09-02: prod has served on them since 2026-08-30
# and a bare ./serve-mxfp4.sh must reproduce prod (it did not -- every restart needed the two
# overrides). Set either to 0 to fall back; the cache suffix follows.
#
# RADIANCE_MXFP4_A_TILED_MIN_M=513 (default, 0 = off): activations at M >= 513 are emitted
# fragment-tiled and the prefill GEMM reads them straight into WMMA registers
# (radiance_mxfp4_fp8_gemm_atiled). Measured 2026-09-02, BetterBench PP t/s vs the folded
# kernel: +9.7% @2k, +6..+8% @8k-64k, +0.4% @250k (attention-bound there); GSM8K 500q 98.00%
# (490/500). Must stay > 512 (the exact_nq decode epilogue writes row-major) and above
# RADIANCE_MXFP4_DECODE_MAX_M.
#
# RADIANCE_MXFP4_WPERM=1 + RADIANCE_MXFP4_DECODE_NT=1 (defaults since 2026-09-02): fragment-order
# weight layout plus nontemporal weight loads in the decode GEMM. Serve-level gate, same cache dir
# (weight layout only, the traced graph is untouched): bench_decode_ctx 23.95 -> 22.66 ms/step at
# ctx 0 and 27.23 -> 25.65 at 32k (-5.4/-5.6%), acceptance byte-identical (2.069); GSM8K 500q
# 97.40% (487/500); BetterBench prefill within +0.3..+3.3% of the WPERM=0 A-tiled sweep at every
# depth 2k-64k (the A-tiled prefill kernel is layout-neutral, which is what ended the old
# "WPERM costs prefill 7-11%" trade). NT is honoured only under WPERM=1 (2-3.6x SLOWER on the
# checkpoint layout - the kernel ignores it there). Set both to 0 to serve the checkpoint layout.
# Built with if-appends, NOT $([ ... ] && echo ...): a command substitution that "fails" (the
# test arm) makes the ASSIGNMENT fail, and under set -e that exits the script silently before a
# single line of output. It bit exactly when a flag was 0.
CACHE_SUF=""
if [ "$GDN_MERGE" = 1 ]; then CACHE_SUF="$CACHE_SUF-gdnm"; fi
if [ "$AR_OVERLAP" = 1 ]; then CACHE_SUF="$CACHE_SUF-arov"; fi
# -nqft, not -nqf: -nqf was the pass-only null experiment. TRACED_QUANT flips the traced graph
# via env alone (no hashed file changes), so it MUST key the cache dir.
if [ "$NQF" = 1 ]; then CACHE_SUF="$CACHE_SUF-nqft"; fi
if [ "$FP8S" = 1 ]; then CACHE_SUF="$CACHE_SUF-fp8s"; fi
# RADIANCE_GDN_NORM_QUANT=1 (default since 2026-09-02): the GDN RMSNormGated + per-token quant as
# ONE custom op (radiance::gdn_norm_quant) instead of the two inductor kernels per linear-attention
# layer. Serve gate: 22.51 -> 22.32 ms/step at ctx 0 (-0.8%), 25.8 -> 25.1 @32k; GSM8K 500q 97.60%;
# BetterBench single-pass update p50 -0.2 ms in every category, tok/update neutral. Not bit-exact
# (silu 1 ulp), so a single prompt's acc/draft moves -- judge it on multi-prompt tok/update. The
# compiled graph changes, so it keys the cache dir.
GNQ=${RADIANCE_GDN_NORM_QUANT:-1}
if [ "$GNQ" = 1 ]; then CACHE_SUF="$CACHE_SUF-gnq"; fi
# RADIANCE_GDN_STRIDED_GATES=1 (default 0, MEASURED NEUTRAL 2026-09-02): skips vLLM's .contiguous()
# on the GDN (b, a) gate slices. Serve A/B on top of GNQ: 22.34-22.41 vs 22.31-22.33 ms/step, output
# byte-identical, GSM8K 97.80% -- the copies are not on the critical path (or inductor re-packs
# the custom-op inputs anyway). Left dark; the graph changes, so it keys the cache dir.
SGATES=${RADIANCE_GDN_STRIDED_GATES:-0}
if [ "$SGATES" = 1 ]; then CACHE_SUF="$CACHE_SUF-sg"; fi
# RADIANCE_GDN_EMPTY_OUT=1 (default 0, MEASURED NEUTRAL 2026-09-02): core_attn_out via torch.empty,
# the rx5 fused_update zeroing the cudagraph pad rows itself. Serve A/B on top of GNQ: 22.28-22.32
# vs 22.29-22.33 ms/step, output byte-identical, GSM8K 500q @conc 8 98.00% (pad rows exercised).
# Correct but worthless: with the strided-gates result this says a ~1 us kernel plus its gap is
# hidden behind the queue at decode -- only kernel TIME moves the step now. Kept dark; keys the
# cache dir because the fill kernel leaves the graph.
EOUT=${RADIANCE_GDN_EMPTY_OUT:-0}
if [ "$EOUT" = 1 ]; then CACHE_SUF="$CACHE_SUF-eo"; fi
CACHE=${CACHE:-$HOME/.radiance-cache-w4a8-093$CACHE_SUF}

# --- multimodal budget knobs (ported from llama-swap-qwen36-27b.sh, 2026-09-05) ---------
# This launcher had NONE of these, and the checkpoint was never capped. The MXFP4
# processor_config.json ships size.longest_edge 16777216 with no max_pixels -- 16,384 visual
# tokens per image -- and the live boot confirms the cost:
#   "Encoder cache will be initialized with a budget of 16384 tokens, and profiled with
#    1 image items of the maximum feature size."
# That is the same configuration that OOMed the ViT on the int4 entry (qwen3_vl.py:2187,
# EngineDeadError, 2026-08-23), and the KV pool is charged for it twice: once as the retained
# encoder cache (16384 x 5120 x 2 B = 0.156 GiB) and once as the ViT activation inside the
# profile run's peak. The int4 checkpoint was capped to 4194304 that day; this one never was.
# NO LMONLY KNOB HERE, DELIBERATELY. pat wants the vision tower loaded (2026-09-05) and it is
# only 0.858 GiB of the 18.04 GiB checkpoint. These knobs are how it stays loaded SAFELY.
MAXPIX=${MAXPIX:-4194304}          # empty = leave processor_config.json alone. Non-empty = cap images
                            # at this many PIXELS via the idempotent repair further down.
                            # Visual tokens = pixels/1024 (16px patch x 2x2 merge), so
                            # 4194304 = 4096 tokens, the int4 entry's value.
MMIMGMAX=${MMIMGMAX:-2}      # empty = pass no --limit-mm-per-prompt. Non-empty = max images per
                            # request (vLLM 400s above it). The ViT activation spike scales
                            # with the NUMBER of images in one request, not just their pixels,
                            # so this is the second line of defence after MAXPIX.
MMVIDMAX=${MMVIDMAX:-0}      # empty = image key only. Set to 0 on vLLM >= 0.27.1: profile_run()
                            # picks the modality with the most tokens for its dummy encoder run
                            # and this checkpoint's video longest_edge is 25165824 against the
                            # image's 16777216. Capping video out keeps the profile on images.
SKIPMMPROF=${SKIPMMPROF:-0} # 1 = --skip-mm-profiling. Drops the dummy encoder run from the
                            # memory profile entirely (honoured at gpu_model_runner.py:6451 in
                            # this image). ONLY safe with MAXPIX set -- otherwise the pool is
                            # sized against a vision peak that can still arrive at runtime.
                            # Default off: the capped profile is the honest one.

# --- ROCm runtime env, needed to reproduce hifi/vllm-radlight (2026-09-05) --------------
# Not radiance knobs. These are read by the HIP/HSA runtime inside the container, so they do
# nothing unless forwarded with -e. Unset = not passed at all, which leaves every existing
# entry byte-identical to its behaviour before this block existed.
#   GPU_MAX_HW_QUEUES  ROCm default 4; radlight pins 1. Fewer hardware queues means less
#                      round-robin scheduling between them, which is where a dispatch-bound
#                      decode loses time.
#   HSA_ENABLE_MWAITX  ROCm default 0; radlight sets 1. Lets the host wait on MWAITX rather
#                      than a busy poll, which shortens the launch gap on short kernels.
#   HSA_ENABLE_INTERRUPT  radlight sets 1; we never forwarded it. Gates whether a host thread
#                      waiting on a completion signal may block on a KFD event + GPU interrupt
#                      instead of polling the signal in memory. In tension with MWAITX above,
#                      which is the polling path. Default on ROCm 7.14 is NOT documented
#                      anywhere in the install on this box -- if it is already 1, setting it
#                      is a no-op. Forwarded so the A/B can answer that.
# THE GFX1201 KERNEL / RUNTIME FLAGS -- the short version
#
# If you take one thing from this script, take this block and the four RADIANCE_* defaults
# further down. They are what separates a tuned R9700 from a stock one, they cost nothing, and
# almost nobody sets them.
#
#   R4D_ATTN=1            radiance's own attention instead of ROCM_AITER_UNIFIED_ATTN. Worth
#                         +1.7 / +5.6 / +24.9% PREFILL at 4k / 16k / 64k and +1.3/+1.5/+2.9%
#                         decode steps/s. The win GROWS with depth, which is why a shallow
#                         benchmark will tell you it does not matter. It also changes the
#                         attention block size (1648 vs AITER's 1664) and gives a slightly
#                         LARGER KV pool. Set below, not here.
#   GPU_MAX_HW_QUEUES=1   ROCm default is 4. Fewer hardware queues means less round-robin
#                         scheduling between them, which is where a dispatch-bound decode
#                         loses time. At MAXSEQS=2 this decode IS dispatch-bound.
#   HSA_ENABLE_MWAITX=1   ROCm default 0. Lets the host thread wait on the MWAITX instruction
#                         rather than a hot spin poll, shortening the launch gap on short
#                         kernels without burning a core.
#
# MEASURED AND REJECTED, so you do not have to repeat them:
#   HSA_ENABLE_INTERRUPT=1  radlight sets it; we measured it FLAT (decode -0.06/-0.18/-0.31%
#                         steps/s, prefill within cross-boot noise) and did not adopt it. It
#                         lets a host thread block on a KFD event + GPU interrupt instead of
#                         polling -- in direct tension with MWAITX above, which is the polling
#                         path. Forwarded but unset, so a re-test costs one line.
#   RADIANCE_MRV2=1       -29% steps/s here. MAXSEQS=2 leaves it nothing to amortise.
#   COMPILE_SIZES/COOP_RED  neither recovers the 22.66 ms/step it was supposed to; the static
#                         specializations change numerics enough to cost the drafter acceptance.
#   RADIANCE_FAST_DRAFT=1 2-bit draft head: 3.8 GiB of KV pool for +2.3% decode. The vendor's
#                         +16.6% is a TP=2 number and does not transfer to one card.
#
# These are read by the HIP/HSA runtime INSIDE the container, so they do nothing unless
# forwarded with -e. Unset = not passed at all.
GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-1}
HSA_ENABLE_MWAITX=${HSA_ENABLE_MWAITX:-1}
ROCM_ENV=()
if [ -n "${GPU_MAX_HW_QUEUES:-}" ]; then ROCM_ENV+=(-e "GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES"); fi
if [ -n "${HSA_ENABLE_MWAITX:-}" ]; then ROCM_ENV+=(-e "HSA_ENABLE_MWAITX=$HSA_ENABLE_MWAITX"); fi
if [ -n "${HSA_ENABLE_INTERRUPT:-}" ]; then ROCM_ENV+=(-e "HSA_ENABLE_INTERRUPT=$HSA_ENABLE_INTERRUPT"); fi
# prompt_logprobs allocates a ~1-1.7 GiB prompt x vocab logits transient that vLLM does not reserve
# for, and KV is sized to eat everything else -- 0.97 and even 0.92 OOM the engine on ppl.py. Use
# GPU_UTIL=0.75 for perplexity work, 0.98 for throughput.
# 0.98 is the ceiling on this box, not a guess: the card has 32624 MiB, and vLLM measures free
# memory AFTER its own HIP context and torch init exist, so it sees 31980 MiB. 0.99 asks for
# 31.54 GiB and fails at startup. 0.98 gives 857,399 KV tokens against 840,019 at 0.97 and
# survives a full 260k-prefill sweep with no OOM.
# 0.97. Note this is LARGELY INERT while KV_MEM below is set -- an explicit --kv-cache-memory
# overrides it and skips vLLM's memory profiling entirely. To trade context for decode speed,
# lower the PIN, not this. (Bigger KV pool = ~5% SLOWER decode over a 2.3 GiB swing; prefill is
# unaffected. This is also why a cold boot looks fast: never compare cold decode to warm.)
# For perplexity work set GPU_UTIL=0.75 and KV_MEM=0 -- prompt_logprobs allocates a 1-1.7 GiB
# transient vLLM does not reserve for, and 0.97 OOMs the engine.
GPU_UTIL=${GPU_UTIL:-0.97}
# KV cache size. Resolved further down, once the batch shape it depends on is known.
# THE SINGLE MOST IMPORTANT LINE IN THIS FILE if you run long contexts. 8.66 GiB, pinned.
#
# vLLM profiles free memory at boot and sizes the KV pool from what it sees. A COLD boot (empty
# torch.compile cache) sees ~2.3 GiB less than a warm one, because compile scratch is counted as
# permanent. So a MAXLEN sized against the warm pool serves happily for weeks and then REFUSES
# TO START the first time the compile cache is invalidated: "7.8 GiB KV cache is needed, which
# is larger than the available KV cache memory (7.08 GiB)". We hit exactly that.
#
# --kv-cache-memory skips profiling altogether, so the pool is identical cold and warm and the
# trap disappears. It is safe because compile peaks BEFORE the pool is allocated (~24.7 vs ~31.1
# GiB) -- the two peaks never coexist. vLLM prints the value it would "fully utilize" on its own
# boot line; this is set just under it. Set KV_MEM=0 to force profiling back on.
KV_MEM=${KV_MEM:-9300000000}
# Which drafter to speculate with.
#   mtp    -- the multi-token-prediction head inside the target checkpoint. One draft forward per
#             speculative position, so RADIANCE_DYNAMIC_DRAFT can stop the loop early.
#   dflash -- a separate block-diffusion drafter (DFlash2) that emits the whole block in ONE graphed
#             pass. Depth is fixed when its CUDA graph is captured, so DYNAMIC_DRAFT is inert and
#             num_speculative_tokens becomes a real tuning knob again.
# dflash is the default because it is what production serves and what the README's numbers were
# measured on; a default that does not match the shipped configuration silently invalidates any
# A/B run taken against it. It costs one extra 2 GiB download (setup-mxfp4.sh fetches it, and the
# check further down prints the command if it is missing). SPEC_METHOD=mtp needs no drafter at all
# and is the fallback if you do not want the second checkpoint.
SPEC_METHOD=${SPEC_METHOD:-dflash}
# Tensor parallelism, defaulted from the cards actually present. This was hardcoded to 2, which
# is right for the reference box and wrong for every host that is not it: a single-card user got
# a startup failure from inside a TP worker, and a four-card user got two idle cards.
TP=${TP:-$RAD_TP}
GPU_IDS=${GPU_IDS:-$RAD_GPU_INDICES}
# MODELS is bind-mounted at /models below, so SNAP and DRAFTER must live somewhere under it.
# Resolved HERE rather than next to SNAP further down: DRAFTER's default dereferences it, and under
# `set -u` that made an un-exported MODELS an "unbound variable" abort rather than a default.
MODELS="$(realpath -m "${MODELS:-$HOME/models-mxfp4}")"
# Drafter checkpoint for SPEC_METHOD=dflash. Must live under MODELS -- only MODELS is mounted.
DRAFTER=${DRAFTER:-$MODELS/Qwen3.8-27B-DFlash2-FP8}
# The drafter's own attention backend. It has to support FULL cuda graphs or vLLM logs "running the
# draft eagerly" and the single-pass draft loses its graph -- which is the entire point of dflash.
# TRITON_ATTN does; R4D is the target's backend and is what mtp uses for the drafter too.
DRAFT_ATTN=${DRAFT_ATTN:-TRITON_ATTN}
# Speculative depth.
#   mtp: measured on this build, 4 beats 8 at decode -- 59.8/60.2 tok/s against 53.1/58.6, because
#   acceptance falls (42.1% -> 33.7%) faster than the deeper drafts pay for themselves. The 0.5.8
#   baseline also ran 4, so this keeps the comparison honest as well as fast.
#   dflash: the drafter's block_size is 8; 7 is the shipped default and the depth is
#   CONTENT-DEPENDENT, so mind the corpus before re-tuning it. The 2026-08-29 sweep on
#   bench_decode_conc said 5 (+8-13% aggregate at every level) -- but that corpus asks for
#   deliberately non-repetitive prose, which is exactly the low-acceptance content where shallow
#   drafts win. On BetterBench's weighted mix (code 0.30), same build, back to back: SPEC=7
#   combined decode 184.3 t/s vs SPEC=5's 159.4 (+15.6% for 7) -- code/json/file_edit run
#   tok/update 4.7-6.0 at depth 7 and the cap at 5 truncates precisely that tail. 5 remains the
#   better setting for prose-heavy or batch-throughput serving (conc-8 562 vs 544 aggregate);
#   8 falls off DEC_MAX_TM at conc 8 (M=72>64, -25%). Tune acceptance-coupled knobs on the
#   weighted mix, not on a single content class.
#
#   RADIANCE_DYNAMIC_WIDTH (patch_dynwidth.py, default ON) mostly dissolves this trade: the
#   scheduler caps each request's VERIFY width from a per-request acceptance EMA (the DFlash2
#   draft pass is one fixed-cost graphed block either way), so prose sequences verify ~4 wide
#   while code keeps the full depth. Measured at base SPEC=7: weighted single-stream unchanged
#   (184.7 vs 184.3) with code tok/update intact, and conc-8 recovers static SPEC=5's batch
#   efficiency (steps 52-57 -> 46-47 ms, aggregate 391-413 -> 444-461 t/s). Lossless by
#   construction -- verification preserves the distribution at any proposal length.
if [ "$SPEC_METHOD" = dflash ]; then SPEC=${SPEC:-7}; else SPEC=${SPEC:-4}; fi
# The tuned drafter stack. The right default is NOT the same for both methods:
#   mtp    -- 1. The 2-bit draft head with an exact rerank is a straight win here (+6.5% decode).
#   dflash -- 1 as of 2026-08-27, WITH RERANK=64 (below). It used to be 0: FAST_DRAFT=1 crashed
#             this drafter at load with an IndexError in vLLM's rocm_unquantized_gemm_impl. That
#             was radiance_w4 freeing `layer.weight` to torch.empty(0) and DFlash2's fused
#             context-KV precompute then slicing it -- `k = weight.shape[1]` on a 1-D tensor. It no
#             longer fires because the pinned libr4d (b9e42ab) ships no w4a16 gemm_nt kernel, so
#             radiance_w4 disables itself and only the int2 head arms. IF LIBR4D IS EVER REBUILT
#             WITH r4d_gemm_w4a16_nt_m64, that crash path comes back and needs a guard in
#             patch_dflash_mxfp4_kv.py for a converted (0-element) weight.
#             Measured, ctx 0, 3 reps, interleaved A/B/A/B, dup-8gram 0.0% throughout:
#               bf16 head        30.13 ms/step | acc/draft 1.904 |  96.4 tok/s
#               int2 R=32        28.43         | acc/draft 1.804 |  98.6   (-5.3% acceptance)
#               int2 R=64        28.66         | acc/draft 1.904 | 101.3   (+5.1%)
if [ "$SPEC_METHOD" = dflash ]; then FAST_DRAFT=${FAST_DRAFT:-1}; else FAST_DRAFT=${FAST_DRAFT:-1}; fi
# Rerank width. RADIANCE_DRAFT_RERANK caps the candidate pool a TOP-K caller can draw from, because
# _radiance_topk_only blanks everything the rerank did not touch. mtp asks the head for an argmax
# and 32 is ample; DFlash2 asks for selector_top_k=16 and 32 costs 5.3% of acceptance. 64 restores
# it EXACTLY to the bf16 head's 1.904 for +0.23 ms, and 128/256 measure identical -- so the pool
# saturates at 4x K, and this is a ceiling to raise with selector_top_k, not a free parameter.
# 80 rather than 64 under dflash: VERIFY_HEAD needs 4x the SAMPLER's top_k (20 here) as well as 4x
# the drafter's selector_top_k (16). At 64 the verify gate rejects every sampled request and the
# feature silently does nothing. The drafter is indifferent -- 64/128/256 measured identical.
if [ "$SPEC_METHOD" = dflash ]; then RADIANCE_DRAFT_RERANK=${RADIANCE_DRAFT_RERANK:-80}; fi
# int2 TARGET verify head. ON under dflash as of 2026-08-27: the profile shows the bf16 lm_head is
# one 2.02 ms GEMM per step (5.9% of wall) and this reuses the drafter's int2 packing at zero extra
# VRAM. BetterBench single pass, combined decode 170.0 -> 174.9 t/s (+2.9%) with all eight
# categories +2.7 to +3.4%, conc 1/2/4 +2.8/+2.5/+1.6%, conc 8 neutral, prefill unchanged.
# Output-equivalent on everything measured: GSM8K 500q greedy identical (486/500 both), 8/8 greedy
# completions byte-identical, and 24/24 SEEDED SAMPLED completions byte-identical at the serve's own
# temperature 0.7 / top_p 0.95 / top_k 20.
if [ "$SPEC_METHOD" = dflash ]; then RADIANCE_VERIFY_HEAD=${RADIANCE_VERIFY_HEAD:-1}; fi
# Context length. Only lower it for diagnostics -- the FLA GDN fallback allocates against this,
# not against the chunk size, and OOMs at 262144.
# 204800, pinned. This is the largest context that fits ALONGSIDE the vision tower with the
# KV pin below, verified on a genuinely cold boot: 228,737 tokens of pool, 1.12x concurrency.
# Do NOT raise it by extrapolating tokens/GiB -- efficiency is max_model_len-dependent (218
# blocks/request at 131072, 308 at 204800), so arithmetic from another MAXLEN will lie to you.
MAXLEN=${MAXLEN:-204800}
# Chat template. It is mounted into the container by path, so it must exist ON THE HOST: this was
# hardcoded to a file under ~/.cache/huggingface that only ever existed on the box it was written
# on, which made a fresh clone fail at startup with a missing-file error from vllm rather than
# anything pointing at the cause. The repo ships the template, so the default works from a fresh
# clone; point CHAT_TEMPLATE at your own to override.
#
# qwen-fixed-v22.3.jinja is the default, NOT qwen3.8-enhanced.jinja (still in the repo). Measured
# 2026-09-02 on the same build, GSM8K 500q greedy conc 8: enhanced 96.00% (480/500, 14 answers
# ran to the 3072-token cap, 340 s) vs fixed-v22.3 98.00% (490/500, 0 truncated, 211 s). Every
# 97-98% record from Aug 24-31 was taken with fixed-v22.3; the 08-31 launcher rewrite silently
# switched prod to enhanced and the band dropped to 95-96% with runaway answers.
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$REPO/qwen-fixed-v22.3.jinja}
CHAT_TEMPLATE="$(realpath -m "$CHAT_TEMPLATE")"
# Server-default reasoning effort. UNSET means the template's own fallback applies, and
# qwen-fixed-v22.3.jinja falls back to 'medium' (line 18) -- that is what production served
# until 2026-09-05. The int4 entry has run 'low' for weeks via the same mechanism in
# llama-swap-qwen36-27b.sh; this is the MXFP4 launcher catching up, not a new idea.
#
# This is a SERVER DEFAULT ONLY: a client sending its own reasoning_effort still wins, so it
# steers the agentic traffic that sends nothing without taking the knob away from anyone.
#
# The guard matters. --default-chat-template-kwargs is passed straight to the template, so a
# template with no reasoning_effort variable, or one that rejects this value, fails SILENTLY
# (inert) or at first request (raise_exception) rather than at boot. Check both here, against
# the template we are actually about to mount.
# Sampling. These are the Qwen3.8 model card's Thinking Mode / General Tasks preset, which is
# also what generation_config.json ships and what the int4 entry has always served through the
# equivalent knobs in llama-swap-qwen36-27b.sh. It was 0.7 here from 2026-09-02 to 09-05 -- a
# benchmark-chasing number (temperature 1.0 costs ~5 GSM8K points) that silently made the MXFP4
# entry behave differently from the int4 one on identical prompts. Benchmarks are allowed to be
# their own thing; the model the user talks to is not. Reverted to the card.
#
# NOT set: min_p and repetition_penalty. The int4 launcher passes them at their no-op values
# (0.0 / 1.0), but vLLM 400s on min_p under spec decoding, so this launcher omits both rather
# than send a zero that only works by luck.
GEN_TEMP="${GEN_TEMP:-1.0}"
GEN_TOPP="${GEN_TOPP:-0.95}"
GEN_TOPK="${GEN_TOPK:-20}"
GENCFG_ARG="{\"temperature\":${GEN_TEMP},\"top_p\":${GEN_TOPP},\"top_k\":${GEN_TOPK}}"

# Server-default thinking budget. 'low' matches the int4 launcher in this repo. This is a
# SERVER DEFAULT ONLY -- a client sending its own reasoning_effort still wins -- so it steers
# the agentic traffic that sends nothing without taking the knob away from anyone. Set it to
# medium (the template's own fallback) or xhigh if you want longer chains of thought; the
# guard below fails at BOOT if your template cannot honour the value, because
# --default-chat-template-kwargs otherwise fails silently or at first request.
REASONING_EFFORT="${REASONING_EFFORT:-low}"
RSNEFF_ARG=""
if [ -n "$REASONING_EFFORT" ]; then
  grep -q 'reasoning_effort' "$CHAT_TEMPLATE" \
    || die "REASONING_EFFORT=$REASONING_EFFORT but the chat template has no reasoning_effort variable" \
           "template: $CHAT_TEMPLATE" \
           "the flag would be silently inert -- unset REASONING_EFFORT or use a template that reads it"
  case "$REASONING_EFFORT" in
    none|off|minimal|low|medium|high|xhigh) ;;
    *) die "REASONING_EFFORT='$REASONING_EFFORT' is not a value qwen-fixed-v22.3 resolves" \
           "accepted: none off minimal low medium high xhigh" ;;
  esac
  RSNEFF_ARG="--default-chat-template-kwargs={\"reasoning_effort\":\"$REASONING_EFFORT\"}"
fi

PATCHES_DIR="$(realpath -m "${PATCHES:-$REPO}")"
# A template inside the repo rides the /patches mount that is already there (already SELinux
# relabelled by its :z); anything else gets its own read-only mount.
CT_MOUNT=()
case "$CHAT_TEMPLATE" in
  "$PATCHES_DIR"/*) CT_PATH="/patches/${CHAT_TEMPLATE#"$PATCHES_DIR"/}" ;;
  *) CT_PATH=/chat-template.jinja; CT_MOUNT+=(-v "$CHAT_TEMPLATE:$CT_PATH:ro,z") ;;
esac

preflight
# A libr4d checkout DIRECTORY whose r4d.so is copied over the image's at container start. Leave
# unset and it is built for you (see AUTO_R4D just below); set it to use your own checkout.
# Needed because the GDN overflow fixes are upstream (StillDeadcode/libr4d PR #1, merged) but the
# only tag is still v0.4.0 and the 0.7.4 image pins v0.4.0 -- so the SHIPPED kernel predates the
# fix and NaNs the gated-delta-net output on this model: WikiText-2 PPL 653586 vs 8.3706. Once
# deadcode tags a release and ships an image pinning it, all of this can go away.
R4D_SO=${R4D_SO:-}
# Built automatically when R4D_SO is unset: libr4d is cloned at the pinned commit and compiled
# inside $IMAGE once, then cached and reused. Costs a few minutes on the first launch only.
# AUTO_R4D=0 opts out and runs the image stock kernel (broken on this model -- see above), and
# setting R4D_SO by hand still wins, so an existing checkout is never rebuilt behind your back.
R4D_PIN=${R4D_PIN:-b9e42ab}
R4D_CACHE=${R4D_CACHE:-$HOME/.cache/radiance-libr4d}
# r4d_radiance_extras.patch carries this repo's libr4d additions on top of the pinned commit:
# the 8-bit prefill attention legs (R4D_ATTN_FP8) and the fused GDN decode step
# (RADIANCE_GDN_FUSED_UPDATE). The build cache key carries a suffix so patched and stock builds
# coexist; bump the suffix whenever the patch content changes, or a stale build serves silently.
R4D_PATCH="$REPO/r4d_radiance_extras.patch"
R4D_KEY="$R4D_PIN"
if [ -f "$R4D_PATCH" ]; then R4D_KEY="$R4D_PIN-rx5"; fi   # rx5: fused_update zeroes the pad rows (o_rows arg)
if [ -z "$R4D_SO" ] && [ "${AUTO_R4D:-1}" = 1 ]; then
  if [ ! -f "$R4D_CACHE/$R4D_KEY/r4d.so" ]; then
    echo "[radiance] building libr4d $R4D_KEY in $IMAGE -- one time, a few minutes"
    rm -rf "$R4D_CACHE/.build"
    mkdir -p "$R4D_CACHE/.build"
    git clone -q https://codeberg.org/StillDeadcode/libr4d.git "$R4D_CACHE/.build"
    git -C "$R4D_CACHE/.build" checkout -q "$R4D_PIN"
    if [ "$R4D_KEY" != "$R4D_PIN" ]; then
      git -C "$R4D_CACHE/.build" apply "$R4D_PATCH"
    fi
    "$RUNTIME" run --rm --entrypoint bash -v "$R4D_CACHE/.build":/work:z -w /work \
      "$IMAGE" -c ./build.sh
    # publish only after a successful build, so an interrupted one is not cached as good
    mv "$R4D_CACHE/.build" "$R4D_CACHE/$R4D_KEY"
  fi
  R4D_SO="$R4D_CACHE/$R4D_KEY"
  echo "[radiance] libr4d $R4D_KEY -> $R4D_SO"
fi
if [ "${PREPARE_ONLY:-0}" = 1 ]; then
  echo "[radiance] prepared: image pulled and libr4d built -- ready to serve"
  exit 0
fi
# Where the hand-written W4A8 kernel takes over from aiter's W4A4 Triton path.
# DEFAULT 0 = never fall back; our kernel serves every M. The comparison is `x.shape[0] > MIN_M`,
# so MIN_M=1 would still route M=1 to aiter -- use 0, not 1.
#
# This was 16 until the decode kernel landed, for two separate reasons that are now both resolved:
#
#   CORRECTNESS. aiter's W4A4 path returns a WRONG result for N=5120 K=3072 (o_proj): captured from
#   a live serve and replayed against an fp32 reference, aiter lands at rel=1.066 with ~1/35th of the
#   correct magnitude, while ours is at rel=0.0017. That shape has no tuned table in mxfp4-configs/,
#   so it takes aiter's generic bands. At MIN_M=16 it went unnoticed in prefill (M=17, our kernel)
#   and poisoned decode (M=9, aiter) -- the fluent-looking garbage this build shipped with for an
#   afternoon.
#
#   SPEED. MIN_M=0 used to be a ~55% decode regression (54.3 ms/step against 35.1) because the only
#   kernel available at M<=16 was the prefill-tiled one, which at M=5 issues 51x more matrix MACs
#   than useful. RADIANCE_MXFP4_DECODE_MAX_M below fixes exactly that, so MIN_M=0 is now both
#   correct AND faster than the old default.
#
# Set it absurdly high to route everything to aiter -- only useful for bisecting.
MIN_M=${MIN_M:-0}
# The decode-kernel band must cover MAXSEQS x (SPEC+1) rows or the biggest verify batches fall
# onto the prefill tile: 64 covers the 8-stream default exactly (dflash SPEC=7 -> 8x8), 128
# covers 16 streams. Defaulted from MAXSEQS so the 8-and-under band routes IDENTICALLY to today.
if [ "${MAXSEQS:-8}" -gt 8 ]; then
  RADIANCE_MXFP4_DECODE_MAX_M=${RADIANCE_MXFP4_DECODE_MAX_M:-128}
fi

# All 304 linear layers run on the W4A8 kernel. RADIANCE_MXFP4_KERNEL_NK / _PERBLOCK_NK remain as
# shape-level bisect tools (N:K pairs) but are unset by default.
#
# They existed because the 64 layers at N=5120 K=3072 (gdn out_proj, attention o_proj) produced a
# broken model, which turned out NOT to be a kernel bug: those layers legitimately receive NaN in
# their activations -- one whole gated-delta-net head -- and per-token fp8 quantization turns a
# single NaN into a NaN row scale, poisoning the row. aiter tolerated the same input only because
# mxfp4 quantization squashes NaN to a finite code. RADIANCE_MXFP4_SANITIZE (default 1) fixes it.
# Extra vllm serve args, for bisecting (e.g. EXTRA="--enforce-eager").
EXTRA=${EXTRA:-}
# Cudagraph capture sizes; empty/none = vLLM's default list ([1,2,4] + multiples of 8).
# Finer sizes (3,5,6,7,10,12,14) were tried 2026-08-29 to un-pad dynamic-width single streams and
# measured NEUTRAL (181.3 vs 184.7 weighted, inside noise): the decode-band GEMMs are
# weight-stream-bound and nearly M-invariant below M~16 (tier7: gate_up 88.5 us at M=5 vs 88.7
# at M=8), so there was no single-stream width cost hiding behind the padding to recover --
# dynamic width's value is batching, where M crosses real cost and split-K boundaries. The knob
# stays for capture experiments; the default stays stock. SPEC=8 + dynamic width was measured in
# the same session: single-stream 184.9 (even), conc-8 405-427 vs 444-461 (LOSES -- cold-start
# batches run full width into the M=72>64 kernel cliff before the EMAs settle). 7 stays.
# CUDA-graph capture set narrowed to the batch shapes this config actually runs. With MAXSEQS=2
# and SPEC=7 the verify batch never exceeds 16 rows, so the stock capture set spends compile time
# and memory on shapes that are never dispatched.
CAPTURE_SIZES=${CAPTURE_SIZES:-[1,2,4,8,16]}
# Compilation-config entries accumulate into ONE flag: two --compilation-config instances would
# not merge (argparse keeps the last).
CC_ITEMS=""
if [ -n "$CAPTURE_SIZES" ] && [ "$CAPTURE_SIZES" != none ]; then
  CC_ITEMS="\"cudagraph_capture_sizes\":$CAPTURE_SIZES"
fi
# Static-shape inductor specializations for the decode batch sizes, and cooperative reductions.
# Both were in the serve that measured 22.66 ms/step (serve_final1.log, 2026-08-29) and neither
# made it into the launch defaults. Re-measured 2026-09-02 on the current stack (bench_decode_ctx
# ctx 0, gen 400, 2-3 reps each): defaults 23.91-23.98 ms/step at 2.069 acc/draft; COOP_RED=1
# alone 23.95-23.97 / 2.069 (neutral); COMPILE_SIZES=[1,2,4,8] alone 23.79-23.82 but acc/draft
# 1.837 (119 vs 128 tok/s, the static specializations change numerics enough to cost the
# drafter); both 23.81-23.85 / 1.771 (116 tok/s). Neither recovers 22.66; both stay OFF.
# COMPILE_SIZES="[1,2,4,8]"  COOP_RED=1
COMPILE_SIZES=${COMPILE_SIZES:-none}
COOP_RED=${COOP_RED:-0}
if [ -n "$COMPILE_SIZES" ] && [ "$COMPILE_SIZES" != none ]; then
  CC_ITEMS="${CC_ITEMS:+$CC_ITEMS,}\"compile_sizes\":$COMPILE_SIZES"
fi
if [ "$COOP_RED" = 1 ]; then
  CC_ITEMS="${CC_ITEMS:+$CC_ITEMS,}\"inductor_compile_config\":{\"triton.cooperative_reductions\":true}"
fi
if [ "$NQF" = 1 ]; then
  CC_ITEMS="${CC_ITEMS:+$CC_ITEMS,}\"pass_config\":{\"fuse_norm_quant\":true,\"fuse_act_quant\":true}"
fi
if [ -n "$CC_ITEMS" ]; then
  EXTRA="$EXTRA --compilation-config {$CC_ITEMS}"
fi
# PROFILE_DIR=1 arms the torch profiler (vLLM 0.27 moved it from VLLM_TORCH_PROFILER_DIR to CLI
# flags); traces land in $CACHE/prof, driven by POST /start_profile and /stop_profile.
if [ -n "${PROFILE_DIR:-}" ]; then
  mkdir -p "$CACHE/prof"
  # PROFILE_STACK=1 adds python stacks to the trace (bigger, slower flush; use for ATTRIBUTION
  # runs, not timing runs -- with_stack inflates the very gaps being measured).
  if [ "${PROFILE_STACK:-0}" = 1 ]; then WITH_STACK=true; else WITH_STACK=false; fi
  EXTRA="$EXTRA --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/prof --profiler-config.torch_profiler_with_stack=$WITH_STACK"
fi

SNAP="$(realpath -m "${SNAP:-$MODELS/Qwen3.8-27B-MXFP4-mtpfp8}")"
# -f follows symlinks, so a checkpoint assembled as a symlink farm into the HF cache fails
# this test on the HOST even though it resolves fine in the container, where the cache is
# bind-mounted at /root/.cache/huggingface. Accept a dangling symlink too and let the
# container be the judge; a genuinely absent checkpoint still has neither.
if [ ! -f "$SNAP/config.json" ] && [ ! -L "$SNAP/config.json" ]; then
  echo "no checkpoint at $SNAP" >&2
  echo >&2
  echo "Run the one-time setup, which downloads AMD's release and builds this checkpoint from it:" >&2
  echo >&2
  echo "  ./setup-mxfp4.sh" >&2
  echo >&2
  echo "It is not an optimization you can skip. AMD's release does not load as-is: its exclude list" >&2
  echo "names the bf16 mtp.* layers as TENSOR names (mtp.fc.weight) among module names, so quark's" >&2
  echo "module match never fires, vLLM applies the mxfp4 scheme to them, and it asserts on a" >&2
  echo "half-width parameter. ./fp8_mtp.py requantizes that head to fp8 and writes the matching" >&2
  echo "layer_quant_config; setup-mxfp4.sh just drives it for you." >&2
  echo >&2
  echo "A checkpoint that already declares mtp.* in layer_quant_config needs none of this --" >&2
  echo "point SNAP straight at it, e.g. the uncensored MXFP4 build linked in the README." >&2
  exit 1
fi
# HF_HUB_OFFLINE=1 inside the container and the cache mounts at /root/.cache/huggingface, so vllm
# must be handed the CONTAINER path -- a host path fails HF repo-id validation, not "not found".
# Derived from SNAP rather than hardcoded, so overriding SNAP actually redirects the server
# instead of silently serving whatever sits at the default name inside the mount.
case "$SNAP" in
  "$MODELS"/*) CSNAP="/models/${SNAP#"$MODELS"/}" ;;
  *) echo "SNAP ($SNAP) must be under MODELS ($MODELS): only MODELS is mounted into the" >&2
     echo "container. Move the checkpoint there, or set MODELS to a directory containing it." >&2
     exit 1 ;;
esac

# --- weight sharding: idempotent on-disk repair -----------------------------------------
# WHY. AMD ships this checkpoint as ONE 18.0 GiB model.safetensors. safetensors is mmap-based,
# so in theory the pages are reclaimable cache rather than committed memory -- but on a host
# with less RAM than about 1.5x the file, that theory stops protecting you: the loader pulls
# essentially the whole file through page cache while also holding staging buffers, and the
# box goes to swap. Splitting into SHARD parts bounds the working set to roughly one shard.
#
# It changes NOTHING about inference speed. The radiance kernel cache is keyed on tensor
# shapes, not on files, so this forces no recompile and the served model is bit-identical.
# It is purely a load-time / memory-pressure fix.
#
# IDEMPOTENT, like the processor_config.json repair below: it is a no-op once an index exists,
# so it runs every boot and costs nothing after the first. A re-download reinstates the
# monolith and the next boot re-shards it.
#
# SAFETY. The original is not touched until the new shards are written AND verified
# tensor-by-tensor (name, dtype, shape, and the safetensors __metadata__ header). This matters
# more here than it looks: this checkpoint carries the fp8 MTP head rewrite, and a shard step
# that silently dropped or re-typed those tensors would not fail loudly -- vLLM would apply the
# mxfp4 scheme to a bf16 head and assert on a half-width parameter. Any failure leaves the
# original in place and the boot continues unsharded.
#
# SHARD=0 or 1 disables. 4 is the default: ~4.5 GiB parts, which is what HF tooling produces
# anyway. Finer buys nothing -- once each shard fits comfortably, more files is just more index.
SHARD=${SHARD:-4}
if [ "${SHARD:-0}" -ge 2 ] && [ -f "$SNAP/model.safetensors" ] && [ ! -f "$SNAP/model.safetensors.index.json" ]; then
  # Disk check first: we hold the original and the new copy at once, so we need the model's
  # own size free, plus a 5% margin. Refusing here is much cheaper than a half-written model.
  _msz=$(stat -c%s "$SNAP/model.safetensors")
  _need=$(( _msz / 1024 * 105 / 100 ))                     # KiB
  _free=$(df -Pk "$SNAP" | awk 'NR==2{print $4}')
  if [ "$_free" -lt "$_need" ]; then
    echo "[shard] SKIPPING: need $(( _need / 1048576 )) GiB free next to the checkpoint," >&2
    echo "[shard]   have $(( _free / 1048576 )) GiB. Serving the monolithic weights as-is." >&2
  else
    echo "[shard] splitting model.safetensors into $SHARD parts (one-off; safe to interrupt)" >&2
    # Run inside the image: the host needs no python deps at all, and the container already
    # has the exact safetensors/torch the server will load with.
    _shard_py=$(mktemp /tmp/shard-safetensors.XXXXXX.py)
    cat > "$_shard_py" <<'SHARDPY'
import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

snap, nshard = sys.argv[1], int(sys.argv[2])
src = os.path.join(snap, "model.safetensors")
tmp = os.path.join(snap, ".shard-tmp")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)

with safe_open(src, framework="pt") as f:
    meta = f.metadata() or {}
    keys = list(f.keys())
    # Bytes per element, by safetensors dtype name. Explicit table rather than digit-scraping
    # the string: "F8_E4M3" would scrape to 843, and "BOOL" to nothing at all. These sizes only
    # decide how tensors are GROUPED, but a KeyError here would abort the boot, so unknown
    # dtypes fall back to 1 byte and merely produce slightly uneven shards.
    NBYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
              "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U16": 2, "U32": 4,
              "U64": 8, "BOOL": 1}
    sizes = {}
    for k in keys:
        sl = f.get_slice(k)
        n = 1
        for d in sl.get_shape():
            n *= d
        sizes[k] = n * NBYTES.get(sl.get_dtype(), 1)

total = sum(sizes.values())
target = total / nshard
groups, cur, acc = [], [], 0
for k in keys:                      # keep the checkpoint's own tensor order
    if cur and acc + sizes[k] > target and len(groups) < nshard - 1:
        groups.append(cur); cur, acc = [], 0
    cur.append(k); acc += sizes[k]
if cur:
    groups.append(cur)

index = {"metadata": {"total_size": total}, "weight_map": {}}
with safe_open(src, framework="pt") as f:
    for i, grp in enumerate(groups, 1):
        name = "model-%05d-of-%05d.safetensors" % (i, len(groups))
        save_file({k: f.get_tensor(k) for k in grp}, os.path.join(tmp, name), metadata=meta)
        for k in grp:
            index["weight_map"][k] = name
        print("[shard]   wrote %s (%d tensors)" % (name, len(grp)), file=sys.stderr)

# VERIFY before anything is destroyed: every tensor present exactly once, same dtype and
# shape as the original, and the header metadata carried over.
with safe_open(src, framework="pt") as fo:
    seen = {}
    for name in {v for v in index["weight_map"].values()}:
        with safe_open(os.path.join(tmp, name), framework="pt") as fn:
            assert (fn.metadata() or {}) == meta, "metadata lost in " + name
            for k in fn.keys():
                assert k not in seen, "duplicate tensor " + k
                seen[k] = True
                a, b = fo.get_slice(k), fn.get_slice(k)
                assert a.get_shape() == b.get_shape(), "shape changed: " + k
                assert a.get_dtype() == b.get_dtype(), "dtype changed: " + k
    missing = set(fo.keys()) - set(seen)
    assert not missing, "tensors dropped: %s" % sorted(missing)[:5]

with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as fh:
    json.dump(index, fh, indent=2)
for fn in os.listdir(tmp):
    os.replace(os.path.join(tmp, fn), os.path.join(snap, fn))
os.rmdir(tmp)
os.remove(src)
print("[shard] verified %d tensors across %d parts; monolith removed"
      % (len(index["weight_map"]), len(groups)), file=sys.stderr)
SHARDPY
    if "$RUNTIME" run --rm --entrypoint python3 \
         -v "$SNAP":"$SNAP":z -v "$_shard_py":/shard.py:ro,z \
         "$IMAGE" /shard.py "$SNAP" "$SHARD"; then
      :
    else
      echo "[shard] FAILED -- the original model.safetensors is untouched, serving it as-is." >&2
      rm -rf "$SNAP/.shard-tmp"
    fi
    rm -f "$_shard_py"
  fi
fi

# --- image pixel cap: idempotent processor_config.json repair ---------------------------
# Ported verbatim in intent from llama-swap-qwen36-27b.sh:760-792 (2026-08-23).
# WHY A FILE PATCH AND NOT --mm-processor-kwargs: Qwen3VLProcessor.__init__ in this
# transformers build takes **kwargs and DISCARDS them (only chat_template is read), so the
# vLLM flag is silently inert. The JSON is the only surface the processor actually reads.
# A re-download of the checkpoint reinstates the defect, so the repair runs every boot and
# is a no-op once the file is already at the target. min_pixels stays at the file's own
# size.shortest_edge (65536 here) when unset -- no forced upscaling of small images.
if [ -n "$MAXPIX" ] && [ -f "$SNAP/processor_config.json" ]; then
  if python3 -c "
import json,sys
ip=json.load(open('$SNAP/processor_config.json')).get('image_processor') or {}
sys.exit(0 if int(ip.get('max_pixels', 0)) == int('$MAXPIX') else 1)
" 2>/dev/null; then
    :
  else
    cp -n "$SNAP/processor_config.json" "$SNAP/processor_config.json.orig" 2>/dev/null || true
    python3 -c "
import json
p='$SNAP/processor_config.json'
d=json.load(open(p))
ip=d.setdefault('image_processor', {})
if 'min_pixels' not in ip:
    ip['min_pixels'] = (ip.get('size') or {}).get('shortest_edge', 65536)
ip['max_pixels'] = int('$MAXPIX')
json.dump(d, open(p,'w'), ensure_ascii=False, indent=2)
"
    echo "[launcher] processor_config.json: capped images at $MAXPIX pixels" >&2
    echo "[launcher]   ($(( MAXPIX / 1024 )) visual tokens max; 1024 px/token = 16px patch x 2x2 merge)" >&2
    echo "[launcher]   original preserved at processor_config.json.orig" >&2
  fi
fi

# No spaces inside the JSON: both of these are expanded UNQUOTED in the server-arg list
# below, same rule as $ASYNC_FLAG and $EXTRA.
MMIMG_ARG=""
if [ -n "$MMIMGMAX" ]; then
  MMVID_PART=""
  if [ -n "$MMVIDMAX" ]; then
    MMVID_PART=",\"video\":${MMVIDMAX}"
    echo "[launcher] max videos per request: $MMVIDMAX (0 = keep the 0.27.1 encoder profile on images)" >&2
  fi
  MMIMG_ARG="--limit-mm-per-prompt {\"image\":${MMIMGMAX}${MMVID_PART}}"
  echo "[launcher] max images per request: $MMIMGMAX (vLLM 400s above it)" >&2
fi

SKIPMM_ARG=""
if [ "$SKIPMMPROF" = 1 ]; then
  if [ -z "$MAXPIX" ]; then
    echo "[launcher] REFUSING SKIPMMPROF=1 without MAXPIX: the KV pool would be sized with no" >&2
    echo "[launcher] vision term at all while an uncapped image can still arrive at runtime." >&2
    exit 2
  fi
  SKIPMM_ARG="--skip-mm-profiling"
  echo "[launcher] --skip-mm-profiling: vision excluded from the memory profile (MAXPIX=$MAXPIX bounds it)" >&2
fi

if [ "$R4D_ATTN" = "1" ]; then ATTN=R4D; else ATTN=ROCM_AITER_UNIFIED_ATTN; fi

# Async scheduling overlaps the host's scheduling work with GPU execution, which is the standard
# answer to a large launch gap. vLLM refuses it together with disable_padded_drafter_batch, so the
# two are one switch here. The unpad lever is worth ~+50% single-stream on the 27B hybrids under
# MTP, where the drafter runs a SERIAL loop of forwards and the padding is paid once per position.
# Under dflash the drafter emits the whole block in one graphed pass, so it is worth re-testing
# which side of that trade wins.
ASYNC=${ASYNC:-0}
if [ "$ASYNC" = 1 ]; then ASYNC_FLAG="--async-scheduling"; UNPAD=false; else ASYNC_FLAG="--no-async-scheduling"; UNPAD=true; fi

# Speculative config, built here so the drafter path is validated before podman is invoked rather
# than surfacing as an HF repo-id error inside the worker.
if [ "$SPEC_METHOD" = dflash ]; then
  DRAFTER="$(realpath -m "$DRAFTER")"
  if [ ! -f "$DRAFTER/config.json" ]; then
    echo "no dflash drafter at $DRAFTER" >&2
    echo >&2
    echo "Fetch it (2 GiB), or let ./setup-mxfp4.sh do it:" >&2
    echo "  hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 --local-dir $DRAFTER" >&2
    echo >&2
    echo "Or serve without it, using the MTP head inside the target checkpoint instead:" >&2
    echo "  SPEC_METHOD=mtp ./serve-mxfp4.sh" >&2
    exit 1
  fi
  case "$DRAFTER" in
    "$MODELS"/*) CDRAFTER="/models/${DRAFTER#"$MODELS"/}" ;;
    *) echo "DRAFTER ($DRAFTER) must be under MODELS ($MODELS): only MODELS is mounted." >&2
       exit 1 ;;
  esac
  # disable_padded_drafter_batch is the single-stream lever (~+50% on the 27B hybrids) and the
  # image bakes the vLLM unpad patch it relies on; it applies to dflash as well as mtp.
  # DRAFT_SAMPLE=probabilistic drafts stochastically with vLLM's shared-Gumbel coupling
  # instead of argmax. Greedy one-hot drafts accept a token with only p_target(argmax);
  # matched sampling accepts with sum(min(p,q)), which is strictly >=.
  #
  # This is the single biggest free win in this file, and it is worth understanding before
  # you copy it. It was supposed to have a cost -- probabilistic drafting needs the full
  # draft-logits head, bypassing the int2 argmax fast path -- so it moves the two terms of
  # decode = steps/s x tokens-per-update in opposite directions. Isolated 2026-09-05 over
  # four ALTERNATING boots (prob/greedy/prob/greedy) at 4k/16k/50k depth:
  #
  #   acceptance (mean_len)  3.39 / 3.25 / 3.15   vs greedy  3.01 / 2.90 / 3.03   (+10%)
  #   decode t/s             +12.8% / +12.1% / +3.9%
  #   steps/s                FLAT to 0.1% on all four boots
  #
  # The draft-logits head is free here, so the acceptance gain is pure profit. Probabilistic
  # won 5 of the 6 within-depth pairings. Quote it as "about +10% acceptance, no step cost"
  # -- acceptance is content- and sampler-sensitive, and n=2 per arm fixes the sign, not the
  # magnitude (greedy returned 3.20 and 2.81 at the same depth on different boots).
  #
  # It interacts with temperature: a flatter target distribution makes p_target(argmax) fall
  # faster than sum(min(p,q)), so the edge is LARGER at this file's temperature 1.0 than at a
  # lower one. If you serve at 0.7 expect less, and re-measure rather than assuming.
  #
  # Two notes if you A/B this yourself. steps/s is the right metric for acceptance-NEUTRAL
  # knobs (it repeats to ~0.1% across cold boots) but it would have scored THIS knob as a
  # regression -- read mean_len too. And alternate boots: two samples from one boot are one
  # sample, which is how a phantom +2.3% prefill win got past us earlier the same day.
  DRAFT_SAMPLE=${DRAFT_SAMPLE:-probabilistic}
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$CDRAFTER\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$DRAFT_ATTN\",\"disable_padded_drafter_batch\":$UNPAD,\"draft_sample_method\":\"$DRAFT_SAMPLE\"}"
else
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"attention_backend\":\"$ATTN\",\"disable_padded_drafter_batch\":$UNPAD}"
fi

# The AR size gate compares the raw bf16 byte count: CHUNK x hidden(5120) x 2. Derive it rather
# than hardcoding it, so changing CHUNK cannot silently drop prefill back onto RCCL.
AR_MAX_KB=$(( (CHUNK * 5120 * 2) / 1024 + 4096 ))

# ---------------------------------------------------------------- KV cache size
# An explicit --kv-cache-memory OVERRIDES GPU_UTIL and skips vLLM's memory profiling entirely.
# It is worth having because that profiling is deliberately conservative: it subtracts the
# profile run's TRANSIENT activation peak plus the cudagraph estimate, both of which sit above
# what steady-state serving needs. On the reference box the difference is 0.93 GiB per rank,
# which is 5.7% of the cache -- but its size depends on the card, on the activation peak at
# CHUNK and on the cudagraph capture set, so it is measured, not computed. See kv-profiles.tsv.
#
#   KV_MEM=auto     (default) use a pin measured for this hardware and batch shape if one
#                   exists, otherwise let vLLM profile -- which is always safe
#   KV_MEM=<bytes>  pin explicitly, consulting neither the table nor the profiler
#   KV_MEM=0        force profiling on even where a measured pin exists
#
# The lookup is keyed on the batch shape as well as the hardware because MAXSEQS moves the
# cudagraph capture sizes and CHUNK moves the prefill transient; a pin measured at one shape is
# not valid at another. It is consulted only at the throughput GPU_UTIL, because the ppl.py
# prompt_logprobs transient is exactly what a pinned KV eats: with KV pinned, GPU_UTIL=0.75
# would no longer buy the headroom it exists to buy.
KV_SRC=explicit
if [ "$KV_MEM" = auto ]; then
  KV_MEM=""; KV_SRC=profiled
  if [ "$GPU_UTIL" = "0.98" ]; then
    KV_MEM=$(rad_kv_lookup "$RAD_GPU_SIG" "${MAXSEQS:-8}" "$CHUNK" "$MAXLEN" "$SPEC_METHOD")
    if [ -n "$KV_MEM" ]; then KV_SRC=measured; fi
  fi
fi
if [ "$KV_MEM" = "0" ]; then KV_MEM=""; KV_SRC=profiled; fi


mkdir -p "$CACHE"/{vllm,inductor,triton,aiter}

echo "[run] $RUNTIME $IMAGE | port $PORT | $SPEC_METHOD spec=$SPEC | model $CSNAP"
echo "[run] gpus=$RAD_GPU_COUNT x $RAD_GPU_NAME ($RAD_GPU_MIB MiB) tp=$TP hip=$GPU_IDS sig=$RAD_GPU_SIG"
echo "[run] attn=$ATTN chunk=$CHUNK ar_max_kb=$AR_MAX_KB fast_draft=$FAST_DRAFT rerank=${RADIANCE_DRAFT_RERANK:-32} vhead=${RADIANCE_VERIFY_HEAD:-0} min_m=$MIN_M fuse_rms=${RADIANCE_FUSE_RMS_QUANT:-1} preshuf=${RADIANCE_PRESHUFFLE:-1} util=$GPU_UTIL kv_mem=${KV_MEM:-none}($KV_SRC)"
if [ "$KV_SRC" = profiled ] && [ "$GPU_UTIL" = "0.98" ]; then
  echo "[run] no KV pin measured for $RAD_GPU_SIG at seqs=${MAXSEQS:-8} chunk=$CHUNK -- vLLM will"
  echo "[run]   profile for itself (safe). ./calibrate-kv.sh measures one and typically reclaims"
  echo "[run]   another ~5% of KV cache on hardware it has not seen before."
fi
echo "[run] cache=$CACHE"
echo "[run] chat-template=$CHAT_TEMPLATE"
echo "[run] sampling=$GENCFG_ARG"
echo "[run] reasoning-effort=${REASONING_EFFORT:-<template default: medium>}"
echo "[run] follow the log with: $RUNTIME logs -f $NAME    stop with: $RUNTIME stop $NAME"

# docker has no --replace, so a container left behind by a previous run has to go first.
if [ "$RUNTIME" != podman ]; then "$RUNTIME" rm -f "$NAME" >/dev/null 2>&1 || true; fi

# DRY_RUN=1 prints the command instead of running it -- for checking what a set of environment
# overrides actually produces, and for lifting the invocation into a unit file.
# PER-REQUEST METRICS IN THE LLAMA-SWAP WEBUI -- added 2026-09-05. pat noticed this entry
# showed blank prompt-speed/gen-speed columns while qwen3.8-27b-vllm filled them in. It is NOT
# a vLLM-vs-llama.cpp limitation (my first guess, and it was wrong): it is three flags that
# llama-swap-qwen36-27b.sh has carried since 2026-08-14 and this launcher never got. ALL THREE
# ARE REQUIRED -- dropping any one silently yields blank columns again:
#   --enable-per-request-metrics    the gate; emits the top-level `metrics` object. vLLM
#                                   REFUSES TO START alongside --disable-log-stats, so never
#                                   add that flag to this launcher.
#   --enable-force-include-usage    our clients STREAM, and in streaming the metrics ride the
#                                   final usage chunk, which vLLM emits only if forced.
#   --enable-prompt-tokens-details  not cosmetic. llama-swap computes prompt t/s as
#                                   (prompt_tokens - cached)/ttft, so without it every
#                                   prefix-cache hit is miscounted as real prefill and the
#                                   prefill number reads far too high.
# NOT fixed by these: draft/accept counts. llama-swap reads those only from llama.cpp's
# `timings` block; vLLM per-request metrics carry no speculative fields, so acceptance stays
# journal-only (the `SpecDecoding metrics:` lines).
# NB the flags themselves sit INSIDE the continued server-arg line below, with no comment
# between them: a '#' after a trailing backslash ends the command there and would silently
# drop --override-generation-config and --chat-template.
# STARTUP NOISE, defaults set 2026-09-05 at pat's request. Both are read by the image's
# entrypoint, not by this script, so they only take effect if forwarded with -e below.
#   RADIANCE_RUN_BWTEST=0    skips the GPU topology + bandwidth sweep at startup. Upstream
#                            defaults it on (~1s, backgrounded). Single card, TP=1, and the
#                            topology never changes, so it tells us nothing per boot.
#   RADIANCE_BANNER_PLAIN=1  disables ANSI colour in the banner. Everything we read comes back
#                            through journalctl, where the escape codes are just noise.
# Both remain overridable per-entry from config.yaml, since these are :- defaults.
exec ${DRY_RUN:+echo} "$RUNTIME" run "${RT_FLAGS[@]}" --rm --name "$NAME" --privileged --ipc=host --network=host \
  --device /dev/kfd --device /dev/dri "${GROUP_FLAGS[@]}" \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  ${ROCM_ENV[@]+"${ROCM_ENV[@]}"} \
  -e ROCR_VISIBLE_DEVICES="$GPU_IDS" -e HIP_VISIBLE_DEVICES="$GPU_IDS" -e HF_HUB_OFFLINE=1 \
  -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e RADIANCE_USE_R4D="${RADIANCE_USE_R4D:-1}" -e RADIANCE_USE_R4D_AR="${RADIANCE_USE_R4D_AR:-1}" -e RADIANCE_USE_R4D_AR_QUANT="${RADIANCE_USE_R4D_AR_QUANT:-1}" \
  -e RADIANCE_R4D_REPORT=1 -e RADIANCE_AR_MAX_KB="$AR_MAX_KB" \
  -e RADIANCE_PRESHUFFLE="${RADIANCE_PRESHUFFLE:-1}" -e RADIANCE_FUSE_RMS_QUANT="${RADIANCE_FUSE_RMS_QUANT:-1}" \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M="$MIN_M" \
  -e RADIANCE_FAST_DRAFT="$FAST_DRAFT" -e RADIANCE_DRAFT_TAU="${RADIANCE_DRAFT_TAU:-0.20}" \
  -e RADIANCE_DRAFT_RERANK="${RADIANCE_DRAFT_RERANK:-32}" \
  -e RADIANCE_DFLASH_SELECTOR_TOPK="${RADIANCE_DFLASH_SELECTOR_TOPK:-}" \
  -e RADIANCE_VERIFY_HEAD="${RADIANCE_VERIFY_HEAD:-0}" \
  -e RADIANCE_VERIFY_HEAD_MAX_M="${RADIANCE_VERIFY_HEAD_MAX_M:-32}" \
  -e RADIANCE_MXFP4_DEBUG="${RADIANCE_MXFP4_DEBUG:-0}" \
  -e RADIANCE_MXFP4_PUREQUANT="${RADIANCE_MXFP4_PUREQUANT:-0}" \
  -e RADIANCE_MXFP4_SYNC="${RADIANCE_MXFP4_SYNC:-0}" \
  -e RADIANCE_MXFP4_CLONE="${RADIANCE_MXFP4_CLONE:-0}" -e RADIANCE_MXFP4_CHECKX="${RADIANCE_MXFP4_CHECKX:-0}" \
  -e RADIANCE_MXFP4_PADOUT="${RADIANCE_MXFP4_PADOUT:-0}" \
  -e RADIANCE_MXFP4_TN4_MIN_M="${RADIANCE_MXFP4_TN4_MIN_M:-2048}" \
  -e RADIANCE_MXFP4_DECODE_MAX_M="${RADIANCE_MXFP4_DECODE_MAX_M:-64}" \
  -e RADIANCE_MXFP4_DECODE_NT="${RADIANCE_MXFP4_DECODE_NT:-1}" \
  -e RADIANCE_MXFP4_A_TILED_MIN_M="${RADIANCE_MXFP4_A_TILED_MIN_M:-513}" \
  -e RADIANCE_MXFP4_WPERM="${RADIANCE_MXFP4_WPERM:-1}" \
  -e RADIANCE_GDN_MERGE_INPROJ="$GDN_MERGE" \
  -e RADIANCE_GDN_NORM_QUANT="$GNQ" \
  -e RADIANCE_GDN_STRIDED_GATES="$SGATES" \
  -e RADIANCE_GDN_EMPTY_OUT="$EOUT" \
  -e R4D_ATTN_FP8="${R4D_ATTN_FP8:-3}" \
  -e RADIANCE_AR_OVERLAP="$AR_OVERLAP" \
  -e RADIANCE_GDN_FUSED_UPDATE="${RADIANCE_GDN_FUSED_UPDATE:-1}" \
  -e RADIANCE_DYNAMIC_WIDTH="${RADIANCE_DYNAMIC_WIDTH:-1}" \
  -e RADIANCE_DYNW_ALPHA="${RADIANCE_DYNW_ALPHA:-0.35}" \
  -e RADIANCE_DYNW_MARGIN="${RADIANCE_DYNW_MARGIN:-2}" \
  -e RADIANCE_DYNW_MIN="${RADIANCE_DYNW_MIN:-2}" \
  -e RADIANCE_DYNW_MIN_BATCH="${RADIANCE_DYNW_MIN_BATCH:-3}" \
  -e RADIANCE_AR_QNB="${RADIANCE_AR_QNB:-96}" \
  -e RADIANCE_AR_QNT="${RADIANCE_AR_QNT:-1024}" \
  ${PYTORCH_CUDA_ALLOC_CONF:+-e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"} \
  -e RADIANCE_AR_OVERLAP_MIN_M="${RADIANCE_AR_OVERLAP_MIN_M:-2048}" \
  -e RADIANCE_AR_OVERLAP_SLICES="${RADIANCE_AR_OVERLAP_SLICES:-4}" \
  -e RADIANCE_MXFP4_EPIFAST="${RADIANCE_MXFP4_EPIFAST:-1}" \
  -e RADIANCE_MXFP4_R4D_DECODE_MAX_M="${RADIANCE_MXFP4_R4D_DECODE_MAX_M:-0}" \
  -e RADIANCE_TOPK_TRITON_MIN_ROWS="${RADIANCE_TOPK_TRITON_MIN_ROWS:-1}" \
  -e RADIANCE_SKINNY_GEMM="${RADIANCE_SKINNY_GEMM:-1}" \
  -e RADIANCE_DFLASH_CALIB="${RADIANCE_DFLASH_CALIB:-}" \
  -e RADIANCE_DFLASH_CALIB_TOKENS="${RADIANCE_DFLASH_CALIB_TOKENS:-200000}" \
  -e RADIANCE_MXFP4_HOIST_QUANT="${RADIANCE_MXFP4_HOIST_QUANT:-$NQF}" \
  -e RADIANCE_MXFP4_TRACED_QUANT="${RADIANCE_MXFP4_TRACED_QUANT:-$NQF}" \
  -e RADIANCE_FP8_STREAM="$FP8S" \
  -e RADIANCE_RMS_QUANT_FUSION="${RADIANCE_RMS_QUANT_FUSION:-$NQF}" \
  -e RADIANCE_MXFP4_SHADOW="${RADIANCE_MXFP4_SHADOW:-}" \
  -e RADIANCE_MXFP4_SANITIZE="${RADIANCE_MXFP4_SANITIZE:-0}" \
  -e RADIANCE_GDN_PATHS="${RADIANCE_GDN_PATHS:-both}" \
  -e RADIANCE_GDN_NANTRACE="${RADIANCE_GDN_NANTRACE:-0}" \
  -e RADIANCE_MXFP4_KERNEL_N="${RADIANCE_MXFP4_KERNEL_N:-}" \
  -e RADIANCE_MXFP4_KERNEL_NK="${RADIANCE_MXFP4_KERNEL_NK:-}" \
  -e RADIANCE_MXFP4_CHECKALL="${RADIANCE_MXFP4_CHECKALL:-}" \
  -e RADIANCE_MXFP4_MHIST="${RADIANCE_MXFP4_MHIST:-0}" \
  -e RADIANCE_MXFP4_DECODE_KS="${RADIANCE_MXFP4_DECODE_KS:-}" \
  -e RADIANCE_MXFP4_DECODE_BK="${RADIANCE_MXFP4_DECODE_BK:-}" \
  -e RADIANCE_MXFP4_CHECK_MAX_M="${RADIANCE_MXFP4_CHECK_MAX_M:-128}" \
  -e RADIANCE_MXFP4_PERBLOCK_NK="${RADIANCE_MXFP4_PERBLOCK_NK:-}" \
  -e RADIANCE_MXFP4_REFLINEAR="${RADIANCE_MXFP4_REFLINEAR:-0}" \
  -e RADIANCE_RUN_BWTEST="${RADIANCE_RUN_BWTEST:-0}" \
  -e RADIANCE_BANNER_PLAIN="${RADIANCE_BANNER_PLAIN:-1}" \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e AITER_ROOT_DIR=/cache/aiter -e TRITON_CACHE_AUTOTUNING=1 \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  -v "$MODELS":/models \
  -v "$CACHE":/cache \
  -v "${PATCHES:-$REPO}":/patches:z \
  ${CT_MOUNT[@]+"${CT_MOUNT[@]}"} \
  ${R4D_SO:+-v "$R4D_SO":/r4d:z} \
  ${R4D_SO:+-e R4D_SO="$R4D_SO"} \
  --entrypoint bash \
  "$IMAGE" -lc '
    set -e
    SP=/opt/vllm/lib/python3.12/site-packages
    cd /patches
    python3 patch_quark_mxfp4.py
    python3 patch_ar_maxbytes.py
    python3 patch_topk_triton_rows.py
    python3 patch_dflash_calib.py
    python3 patch_dflash_mxfp4_kv.py
    python3 patch_rmsquant_fusion.py
    python3 patch_verify_head.py
    python3 patch_kv_group_size.py
    python3 patch_topk_composite.py
    python3 patch_gdn_shared_build.py
    python3 patch_dflash_selector_topk.py
    python3 patch_gdn_merge_inproj.py
    python3 patch_dynwidth.py
    python3 patch_ar_geometry.py
    python3 patch_gdn_glue.py
    # Non-fatal: fixes content=null on thinking-off requests; not required to serve.
    python3 patch_qwen3_thinkoff.py \
      || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"
    cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
    # radiance_drafthead.py is copied too so RADIANCE_DRAFT_RERANK can be swept without an
    # image rebuild. The repo copy was byte-identical to the 0.9.3 one before that knob existed.
    cp radiance_mxfp4.py radiance_gdn.py radiance_rmsquant.py radiance_drafthead.py \
       radiance_verifyhead.py radiance_gdnmerge.py radiance_aroverlap.py radiance_topk.py \
       radiance_arnq.py "$SP"/
    hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 $(python3 -m pybind11 --includes) \
      radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
    # Optional patched libr4d. R4D_SO is the DIRECTORY of a libr4d checkout built from main --
    # it is bind-mounted at /r4d and its r4d.so replaces the one in the image. For an image
    # rebuild, the Dockerfile supports the same substitution through R4D_REPO / R4D_VERSION.
    if [ -n "${R4D_SO:-}" ] && [ -f /r4d/r4d.so ]; then
      cp /r4d/r4d.so "$SP"/r4d.so
      echo "[radiance] using patched r4d.so from $R4D_SO"
    fi
    # Leave /patches before exec. It is a bind mount of the repo, and a stale
    # radiance_mxfp4_fp8.so left there by a `make` shadows the one just compiled into
    # site-packages, because the working directory precedes it on sys.path. That is not a
    # hypothetical: an Aug-20 build sat there and silently served a kernel 17 hours older than
    # its own source, producing fluent-looking garbage with no error anywhere in the log.
    cd /
    exec /opt/radiance_entrypoint.sh "$@"' _ \
     "$CSNAP" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
    --kv-cache-dtype fp8 --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" \
    ${KV_MEM:+--kv-cache-memory "$KV_MEM"} \
    --max-model-len "$MAXLEN" --max-num-seqs "${MAXSEQS:-8}" --max-num-batched-tokens "$CHUNK" \
    --attention-backend "$ATTN" \
    --speculative-config "$SPEC_CFG" \
    $ASYNC_FLAG $EXTRA $MMIMG_ARG $SKIPMM_ARG \
    --enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --enable-per-request-metrics --enable-force-include-usage --enable-prompt-tokens-details \
    --override-generation-config "$GENCFG_ARG" \
    --chat-template "$CT_PATH" \
    $RSNEFF_ARG \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}
