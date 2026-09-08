#!/bin/bash
# llama-swap launcher for qwen3.8-27b-ggz14.
#
# PROVENANCE -- THREE LAUNCHERS, ONE LINEAGE
#
#   ggz14's serve-mxfp4.sh          codeberg.org/ggz14/radiance-vllm-mxfp4 (v0.11.0)
#     |                             the upstream script. Owns the MXFP4 GEMM, R4D
#     |                             attention and DFlash2 integration -- none of that
#     |                             is ours, and none of it is forked here.
#     v
#   startup-qwen3.8-27b-mxfp4.sh    ../startup-qwen3.8-27b-mxfp4.sh, in this repo.
#     |                             The working MXFP4 build: same model, same card,
#     |                             NO KV offload, and every knob carrying the
#     |                             measurement that chose it. GET THIS SERVING
#     |                             FIRST. If it does not serve, nothing in
#     |                             kv-cache/ will either.
#     v
#   THIS FILE                       serve-mxfp4-kvcache-base.sh. The same launcher
#                                   plus the offload delta: the GPU -> /dev/shm ->
#                                   /kvcache staging, the seven house patches in
#                                   ../patches/, and the RADIANCE_* gates on them.
#
# The tuning defaults below are NOT maintained here. They are copied across from
# startup-qwen3.8-27b-mxfp4.sh when the release is assembled, so the two engines
# cannot drift -- and so that what you measure with the cache on is the same engine
# you measured with it off. If you have tuned the MXFP4 launcher for your own
# hardware, carry the same values over.
#
# The delta is seven items and nothing else: HOUSE resolution; the KVOFF_* knob
# block; the /dev/shm fit check and RAM clamp; the fs-tier --kv-transfer-config
# builder; the extra container mounts and RADIANCE_* env; the seven /house patch
# lines in the prelude; and --kv-transfer-config on the serve line. See
# ./README.md for the table.
#
# Keeping this in sync: re-copy from the upstream serve-mxfp4.sh when ggz14's repo
# updates, then re-apply the six llama-swap edits listed below and the offload delta.
#
# House copy (2026-09-05) of ggz14's serve-mxfp4.sh, repo at
# kv-cache/../radiance-vllm-mxfp4 (codeberg.org/ggz14/radiance-vllm-mxfp4, v0.11.0,
# image stilldeadcode/vllm-radiance:0.9.3). Serves Qwen3.8-27B in native MXFP4
# (4-bit) with the FP8 DFlash2 drafter on this box's single R9700 (TP=1, auto-detected
# by the gpu-detect.sh sourced below).
#
# The only differences from the upstream script are the ones llama-swap needs:
#   * takes --port <N> (llama-swap's contract); the container uses --network=host and
#     binds 0.0.0.0:<N> directly, so no port mapping
#   * NAME defaults to qwen38-27b-ggz14; SERVED (new knob) defaults to qwen3.8-27b-ggz14
#     and is the single --served-model-name (upstream ships Qwen3.8 Qwen3.6 Qwen3.8-MXFP4)
#   * MODELS defaults to $HOME/models-mxfp4 (where this box's checkpoints live)
#   * adds --rm, so a killed launcher cannot orphan its container (the house rule)
#   * REPO (new knob, defaults to radiance-vllm-mxfp4 beside kv-cache/) locates the repo's own files
#     (gpu-detect.sh, r4d_radiance_extras.patch, the chat template, the /patches mount),
#     which upstream resolves relative to the script itself
# Everything else -- the knob defaults, the patch prelude, the env list, the entrypoint
# exec, the vllm serve arguments -- is byte-identical to upstream. Re-copy this file from
# the upstream serve-mxfp4.sh when the repo updates, then re-apply these six edits.
#
#   serve-mxfp4-kvcache-base.sh --port <N>   start the server (launched by llama-swap)
#   serve-mxfp4-kvcache-base.sh -h           every knob, its default and what it does
#
# It needs two checkpoints under $MODELS, both produced by setup-mxfp4.sh:
#   Qwen3.8-27B-MXFP4-mtpfp8   AMD's amd/Qwen3.8-27B-Quark-AWQ-MXFP4 with the MTP head requantized
#                              to fp8 by ./fp8_mtp.py. NOT optional for THAT checkpoint: its exclude
#                              list does name the mtp.* layers, but as tensor names (mtp.fc.weight,
#                              all 15 .weight-suffixed) among 112 module names, and quark matches
#                              modules -- so the exclusion never fires, vLLM applies the mxfp4
#                              scheme to a bf16 head, and it asserts on a half-width parameter.
#                              A checkpoint that declares mtp.* in layer_quant_config (or excludes
#                              it by module name) loads as-is: point SNAP at it and skip fp8_mtp.py.
#                              The drafter is fp8 and not mxfp4 on purpose -- 4-bit costs more
#                              acceptance than it saves in bandwidth, and AWQ does not rescue it.
#   Qwen3.8-27B-DFlash2-FP8    the block-diffusion drafter used by SPEC_METHOD=dflash (the default).
#                              SPEC_METHOD=mtp uses the head inside the target and needs no drafter.
#
# Everything below is `${VAR:-default}`, so any of it can be overridden from the environment
# without editing this file. The defaults are the measured production configuration; each one
# carries the measurement that chose it in the comment above it.
#
# WHAT TO CHECK IN THE LOG
#   "Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM"  -> our kernel won the selection
#   "[radiance] native MXFP4 enabled on gfx12x"           -> the aiter fp4 gate was relaxed
#   the R4D selections table (RADIANCE_R4D_REPORT=1)      -> which kernels bound, and why not
#   the stock "current platform does not support native MXFP4/MXFP6" notice still prints and is a
#   false alarm; it comes from a separate supports_mx() call.
#
# This entry takes the whole R9700, so it cannot be loaded together with the other whole-GPU
# vLLM entry on this box (qwen3.8-27b-vllm, container qwen38-27b-vllm). To switch, stop the
# other container first:  podman stop qwen38-27b-vllm   (and the reverse to go back).
#
# The measurements behind the defaults, the numerics reference, the 0.5.8 baseline and the history
# of this file are in MXFP4-NOTES.md; the user-facing documentation is in README.md.

set -euo pipefail

# ---------------------------------------------------------------- usage / arguments
usage() {
  cat <<'USAGE'
 serve-mxfp4-kvcache-base.sh -- native MXFP4 Qwen3.8-27B on AMD RDNA4 (gfx1201)

GPU count, tensor-parallel size and KV cache size are all detected; nothing below has to be
edited to run on a host with a different number of cards.

  serve-mxfp4-kvcache-base.sh --port <N>   serve on http://<host>:<N>/v1 (llama-swap's contract;
                                       the container uses host networking and binds <N> directly)
  serve-mxfp4-kvcache-base.sh [ARGS]       any extra arguments are passed through to `vllm serve`

Everything is an environment variable; these are the ones worth knowing.

  MODELS=$HOME/models-mxfp4  directory holding the checkpoints (bind-mounted at /models)
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
REPO="$(realpath -m "${REPO:-$(cd "$(dirname "$(realpath -m "${BASH_SOURCE[0]}")")" && pwd)/../../radiance-vllm-mxfp4}")"
# HOUSE is our own code that runs against ggz14's tree but is NOT part of it: the offload
# boundary patch and the KV tier bench. It used to live loose inside $REPO, which is an
# upstream clone ignored by ~/ai's .gitignore -- so those files were tracked by nothing and a
# `git clean` in that checkout would have deleted them with no copy anywhere. They now live in
# a tracked directory and ride their own mount; $REPO stays pristine.
HOUSE="$(realpath -m "${HOUSE:-$(cd "$(dirname "$(realpath -m "${BASH_SOURCE[0]}")")" && pwd)/../patches}")"
[ -d "$HOUSE" ] || die "HOUSE=$HOUSE does not exist (house patches + KV bench live there)"
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
    # OUR OWN ORPHAN IS NOT A CONFLICT, measured 2026-09-06: this cost a recovery. A crash, or
    # a stop whose SIGKILL landed on the podman client rather than the container, leaves a
    # container of our own $NAME Up -- holding the port, the GPU and the /dev/shm region. The
    # launch already reclaims that name (podman `--replace` at RT_FLAGS, docker the `rm -f`
    # just before exec) and BOTH stop a running container before creating the new one, so the
    # port is free by the time podman binds it. But this guard ran first and killed the
    # launcher, which llama-swap surfaces only as "upstream command exited prematurely" -- an
    # error naming neither the port nor the orphan. Recovery then needed a hand-typed
    # `podman stop`, which is not something to have to remember at the wrong moment.
    #
    # Scope is strictly a container named EXACTLY $NAME. Any other holder still aborts: that
    # really is a second server wanting a GPU this entry needs all of. With --network=host
    # there is no port mapping to inspect, so "a container of our name is Up" is the closest
    # available proof of ownership -- and if it is up on our port, replacing it is right
    # whichever process is listening.
    local self_st=""
    self_st="$("$RUNTIME" ps --filter "name=^${NAME}$" --format '{{.Status}}' 2>/dev/null | head -1)"
    if [[ "$self_st" == Up* ]]; then
      echo "[serve-mxfp4] port $PORT is held by our OWN container $NAME ($self_st)." >&2
      echo "[serve-mxfp4]   Treating it as an orphan from a crash or a killed stop; the launch" >&2
      echo "[serve-mxfp4]   will replace it. If your supervisor thought this model was stopped," >&2
      echo "[serve-mxfp4]   it was holding the GPU the whole time -- $RUNTIME ps ; amd-smi monitor" >&2
    else
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
  fi

  [ -r "$CHAT_TEMPLATE" ] || die "chat template not readable: $CHAT_TEMPLATE" \
      "set CHAT_TEMPLATE=<path to a .jinja on the host>, or leave it unset to use the" \
      "one shipped in this repo (qwen-fixed-v22.3.jinja)"
}

# Image and cache MUST move together: cache dirs validate on model + torch/Triton version and must
# not be shared across configurations. Both defaulted to 0.7.4 / -074 long after production moved to
# 0.9.3 / -093, so anyone taking the defaults got a DIFFERENT server than the one being measured.
IMAGE=${IMAGE:-stilldeadcode/vllm-radiance:0.9.3}
# NAME and SERVED move together with the entry key (the SAFETY rule in <your llama-swap config.yaml>):
# llama-swap forwards the requested model id verbatim and vLLM validates it, so if they
# drift apart the request 404s AFTER the model is already loaded.
NAME=${NAME:-qwen38-27b-ggz14}
SERVED=${SERVED:-qwen3.8-27b-ggz14}
PORT=${PORT:-8080}
HSA_ENABLE_MWAITX=${HSA_ENABLE_MWAITX:-1}
GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-1}
MAXSEQS=${MAXSEQS:-2}
CHUNK=${CHUNK:-2048}
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
GPU_UTIL=${GPU_UTIL:-0.97}
# KV cache size. Resolved further down, once the batch shape it depends on is known.
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
MODELS="$(realpath -m "${MODELS:-$HOME/ai/models-mxfp4}")"
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

REASONING_EFFORT="${REASONING_EFFORT:-}"
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

# ---------------------------------------------------------------------------
# CPU KV offload (second-tier prefix cache in system RAM).
#
# vLLM's --kv-offloading-size takes ABSOLUTE GiB only (config/cache.py: it is a
# plain `float | None`) -- there is no percentage form and no "auto", so any
# machine-relative sizing has to happen out here.
#
#   KV_OFFLOAD=off     no offload at all.
#   KV_OFFLOAD=auto    size it from what this box actually has spare (default).
#   KV_OFFLOAD=50%     percentage of TOTAL system RAM, then clamped as below.
#   KV_OFFLOAD=11.5    absolute GiB, still clamped -- an oversized value is
#                      lowered to what fits rather than failing the boot.
#
# The buffer lives in /dev/shm, and we run --ipc=host, so the tmpfs that matters
# is the HOST's. Two hard limits, and the first one surprises people:
#
#   1. /dev/shm defaults to 50% of RAM. `df -h` ROUNDS IT UP -- a 23.46 GiB box
#      reports "12G" for a tmpfs that is really 11.732 GiB. Size against
#      statvfs, never against df, or you will ask for 12 and get a dead boot.
#   2. The region is pre-faulted (MADV_POPULATE_WRITE) and pinned
#      (cudaHostRegister), so it can never swap. Every byte here is a byte
#      permanently denied to page cache and to the server's own anonymous
#      memory. KVOFF_KEEP_FREE_GIB is the floor we refuse to eat into.
#
# Overshooting the tmpfs fails the START -- it does not degrade gracefully, and
# on 0.27.1 it dies without a log line, which is a miserable thing to debug.
# Hence the clamp is mandatory, not advisory.
KV_OFFLOAD=${KV_OFFLOAD:-off}
KVOFF_MIN_GIB=${KVOFF_MIN_GIB:-4}        # below this a second tier is not worth the RAM; -> off
KVOFF_KEEP_FREE_GIB=${KVOFF_KEEP_FREE_GIB:-3}   # page cache + headroom left for the rest of the box
KVOFF_SHM_MARGIN_MIB=${KVOFF_SHM_MARGIN_MIB:-256} # podman locks + multiprocessing semaphores
KVOFF_BACKEND=${KVOFF_BACKEND:-native}

# --- L3: disk-backed secondary tier (OPTIONAL, default OFF) ------------------
# vLLM registers a filesystem secondary tier ("fs") behind the CPU primary tier,
# reached through TieringOffloadingSpec. It is NOT reachable via
# --kv-offloading-backend, whose only legal values are native|lmcache
# (config/cache.py:40) -- it needs an explicit --kv-transfer-config, built below.
#
#   KVOFF_DISK=/kvcache   host path of a dedicated filesystem -> enables the tier
#   KVOFF_DISK=""         off (DEFAULT)
#
# THREE THINGS THAT WILL BITE, all verified in the 0.27.1 tree:
#
#   1. PYTHONHASHSEED. Block filenames are content hashes chained from NONE_HASH,
#      and kv_cache_utils.py:112 seeds that from os.urandom(32) when
#      PYTHONHASHSEED is unset. Every restart would then hash the same tokens to
#      DIFFERENT filenames, orphaning the entire cache on disk -- silently, with a
#      100% miss rate and no error. We pin it. Changing it invalidates the cache.
#   2. NO EVICTION. tiering/fs/manager.py has no capacity, quota or TTL parameter,
#      and SecondaryTierManager exposes no eviction hook: this tier writes and
#      never deletes. An external reaper is MANDATORY, not advisory. See
#      kvcache-reap.service/.timer.
#   3. O_DIRECT. tiering/fs/io.py probes it per-directory and uses it when the
#      filesystem supports it (ext4 does). Reads and writes therefore BYPASS the
#      page cache entirely, so spare RAM cannot act as a read cache in front of
#      this tier -- the only productive home for spare RAM is the primary tier.
KVOFF_DISK=${KVOFF_DISK:-}
KVOFF_DISK_MNT=/kvcache                                   # path INSIDE the container
# Measured on the 512 GB zvol, 2026-09-07, O_DIRECT (the tier's own path), MB/s:
#   reads    1 thr  87.6 | 4 thr 268.9 | 8 thr 719.3 | 16 thr 435.6  <- peak at 8
#   writes   1 thr 1100                                              <- never the limit
# So reads get 8 (I/O-bound waiting on the array, so exceeding the 4 cores is fine,
# and 16 regresses on raidz contention); writes stay at 4, since 1.1 GB/s at a single
# thread is already 24x the 44.6 MB/s this tier actually stores.
# CAVEAT on those read figures: they were taken shortly after writing the test files,
# and with sync=disabled the host may still have had them in a pending txg, so the
# absolute numbers are optimistic. The SHAPE is what these settings rest on.
KVOFF_DISK_RTHREADS=${KVOFF_DISK_RTHREADS:-8}             # vLLM default is 16
KVOFF_DISK_WTHREADS=${KVOFF_DISK_WTHREADS:-4}             # vLLM default is 16; 4C4T box
KVOFF_HASHSEED=${KVOFF_HASHSEED:-0}
# RADIANCE_OFFLOAD_MIXED_HIT: 1 = upstream behaviour, 0 = patch_offload_mixed_hit.py's
# guard is active (external hits declined for requests that also hit the GPU prefix
# cache). DEFAULT IS 1 -- deliberately the crashing behaviour. On 2026-09-06 the L3 tier
# was reading back at 70.4% external hit rate and the guard would have thrown away an
# unknown, possibly large share of that, so we run upstream until the patch's one-shot
# diagnostic dump identifies the offending group. Flip to 0 to trade hit rate for a
# crash-free engine.
KVOFF_MIXED_HIT=${KVOFF_MIXED_HIT:-1}
# RADIANCE_OFFLOAD_PENDING_IS_MISS: 1 = a lookup that meets a chunk whose store has not
# landed yet takes the ready prefix it has already found; 0 = upstream, which defers the
# whole request and hopes for a longer prefix a step later. DEFAULT IS 1, because Phase A
# measured what upstream's hope is worth here: 11,849 of 11,878 production lookups deferred,
# 3 served, and the trigger was HIT_PENDING over RETRY at 408 to 1. A bench hits because an
# idle box has no in-flight stores; production stores 61 GB a boot and never goes quiet.
# This is the A/B switch for that change -- flip to 0 to get upstream behaviour back without
# unpatching anything. See kv-cache/cache-preemption-patch-plan.md R3.14.3.
KVOFF_PENDING_IS_MISS=${KVOFF_PENDING_IS_MISS:-1}
# RADIANCE_OFFLOAD_EAGLE_GROUPS: 1 = annotate the EAGLE/MTP draft KV group positionally on
# the hybrid grouping path; 0 = upstream, which only does it for DeepSeek-V4 and therefore
# not for us. DEFAULT IS 1. Without it no group is annotated, and the offload scheduler's
# fail-safe flags ALL NINE groups as draft groups -- the boot log says so verbatim
# ("draft attention groups [0, 1, 2, 3, 4, 5, 6, 7, 8] detected"). That withholds the newest
# chunk of every conversation from the store for the whole of a decode and shortens every
# servable prefix by a chunk, nine times over. Verify after a restart: the line must read
# "[8]". See kv-cache/patch_kv_offload_eagle_groups.py and upstream PR #55390.
KVOFF_EAGLE_GROUPS=${KVOFF_EAGLE_GROUPS:-1}
# RADIANCE_MAMBA_STORE_STRIDE: keep every Nth Mamba/GDN snapshot instead of one per chunk.
# 1 = off (upstream). DEFAULT IS 8. A Mamba group holds ONE recurrent state and the load
# path reads exactly one chunk of it, but the store path writes 27 MB per group per chunk --
# ~70 snapshots for a long conversation where the GPU itself keeps 2. At N=8 the bytes per
# 8 chunks fall from 72 units to 30 (0.417x), so the CPU tier holds ~276,900 tokens instead
# of 115,360: 1.21x the GPU cache instead of 0.50x. That is what lets a conversation still
# be in RAM on the follow-up turn, and a CPU hit costs 1-2 s against the measured 64.26 s
# fs->CPU promotion. It costs prefix: hits are truncated down to an N-chunk (13,184-token)
# boundary. See kv-cache/patch_kv_offload_mamba_stride.py, plan R3.13.
KVOFF_MAMBA_STRIDE=${KVOFF_MAMBA_STRIDE:-8}
# RADIANCE_FS_FANOUT_TARGET_MB / RADIANCE_FS_FANOUT_MAX: how far one filesystem-tier job is
# split across the tier's own thread pool. DEFAULT IS 256 MiB / 0 (0 = use the byte budget).
# The tier is given 8 read and 4 write threads and, upstream, uses exactly one of them: both
# submit paths call enqueue_*(job_id, 1, [task]) and that single task is a serial loop over
# every block file in the job. vllm/fs_io_C.abi3.so has no threads of its own (no pthread, no
# io_uring in its symbol table -- only PyEval_SaveThread), so a promotion reads ~200 files of
# 27 MB one after another at queue depth 1. That is the 64.26 s fs->CPU promotion.
# Upstream PR #49225 fixes this with a 32 MiB budget divided by the block size; our block is
# already 27,000,832 bytes, so 32 would give a fanout of 2 (and 1 as soon as a second job is
# in flight). 256 MiB gives 8 batches for a promotion and 4 for a store -- the whole pool.
# Set KVOFF_FS_FANOUT_MAX=1 to restore upstream one-task-per-job exactly, without unpatching.
# See kv-cache/patch_kv_offload_fs_fanout.py, plan R3.14.
KVOFF_FS_FANOUT_MB=${KVOFF_FS_FANOUT_MB:-256}
KVOFF_FS_FANOUT_MAX=${KVOFF_FS_FANOUT_MAX:-0}
# KVOFF_BLOCKS_PER_CHUNK: connector 'blocks_per_chunk' -- how many KV blocks share one
# offloaded chunk (and one file per group). Empty = omit the key = vLLM's default of 1.
# Raising it coarsens the offload grid AND, via resolve_mamba_align_size, the external
# hit-window rounding. Under test 2026-09-07 to find out whether a Mamba group's per-chunk
# payload stays one fixed-size state (a free N-fold volume cut) or becomes N states
# (bundling only). See kv-cache/cache-preemption-patch-plan.md R3.12.5. Leave EMPTY for
# production until that is answered.
KVOFF_BLOCKS_PER_CHUNK=${KVOFF_BLOCKS_PER_CHUNK:-}
# KVOFF_DISK_SUBDIR: subdirectory of KVOFF_DISK holding the block tree. Change it to run an
# experiment against an isolated tier -- the on-disk config.json is written once and NOT
# rewritten when the geometry changes, so a differing blocks_per_chunk must not share a root.
KVOFF_DISK_SUBDIR=${KVOFF_DISK_SUBDIR:-blocks}

# KVOFF_POLICY: eviction policy for the CPU PRIMARY tier (the 16 GiB /dev/shm region).
#   lru  = vLLM's default (cpu/manager.py:45, cpu/spec.py:133).
#   arc  = Adaptive Replacement Cache: T1 (recency) + T2 (frequency) with B1/B2
#          ghost lists that retune the split on every hit.
# Why this matters here: the tier holds ~508k tokens, about 13 prompts of 34k. Under
# LRU a single sweep of new material walks the whole tier out -- exactly the pattern
# measured 2026-09-08, where p1..p10 fully evicted p0. ARC is scan-resistant: one-shot
# blocks land in T1 and are evicted from there, while a prefix that has been hit twice
# is promoted to T2 and survives the sweep. A recurring system prompt or document head
# is precisely the T2 case.
# SAFETY, verified 2026-09-08 by reading policies/arc.py:112-170: ARC.evict() skips any
# block with ref_cnt != 0 AND any key in the `protected` set, so it honours the ref_cnt
# eviction protection (tiering/manager.py principle 5) without needing the
# mark_evictable/mark_non_evictable hooks -- those are no-op base methods
# (policies/base.py:92,96) that LRU overrides only as an indexing optimisation.
# COST: the ghost lists hold up to cache_capacity KEYS each (no block data), trimmed in
# evict(). Keys only, so the overhead is metadata, not tier capacity.
KVOFF_POLICY=${KVOFF_POLICY:-lru}

# KVOFF_STORE_THRESHOLD: admission filter. A block must be SEEN in lookup() this many
# times before it is eligible to be stored in the CPU tier (cpu/manager.py:173).
#   0 or 1 = off (vLLM default; every block is admitted on first sight)
#   2      = admit only on the second sighting
# This is the other half of scan resistance: it stops single-use prompts from ever
# entering the tier, rather than letting them in and then evicting something. The
# trade-off is a one-occurrence admission delay -- content is not cached until its
# second appearance, so a prefix reused exactly twice is never served from cache.
# Independent of KVOFF_POLICY; A/B them separately.
KVOFF_STORE_THRESHOLD=${KVOFF_STORE_THRESHOLD:-0}
KVOFF_TIER_ARG=""

# --- garbage collection -----------------------------------------------------
# Leftovers survive a restart and break the NEXT boot, in ways whose error
# messages point nowhere near the cause. GC runs on every start, regardless of
# KV_OFFLOAD.
#
#   GC_ORPHANS=on    reap unreferenced /dev/shm regions, report orphan
#                    containers (DEFAULT).
#   GC_ORPHANS=off   touch nothing, report nothing.
GC_ORPHANS=${GC_ORPHANS:-on}
# DRY_RUN must stay side-effect free: report what GC would do, delete nothing.
[ -n "${DRY_RUN:-}" ] && GC_DRY=1 || GC_DRY=""
gc_do() { if [ -n "$GC_DRY" ]; then echo "[gc] (dry-run, not executing) $*" >&2; else "$@"; fi; }

# Leftover 1: the container. A crash or a `systemctl restart` can leave one
# holding VRAM while the supervisor believes the model is stopped.
#
# This only REPORTS. The launch already reclaims the name -- podman via
# `--replace` (set above), docker via the `rm -f` just before exec -- and both
# handle a running container too, so a second removal path here would be one
# more thing to disagree with the first. What the launch does NOT do is tell
# you it happened, and a silently-replaced orphan is exactly the condition
# worth knowing about: it means something died without cleaning up, and if it
# was still running it was holding VRAM the whole time. Scope is strictly our
# own $NAME -- a launcher must never touch containers it did not create.
gc_report_orphan_container() {
  [ "$GC_ORPHANS" = off ] && return 0
  local st
  st="$("$RUNTIME" ps -a --filter "name=^${NAME}$" --format '{{.Status}}' 2>/dev/null | head -1)"
  [ -z "$st" ] && return 0
  if [[ "$st" == Up* ]]; then
    echo "[gc] WARNING: container $NAME is ALREADY RUNNING ($st) and will be replaced." >&2
    echo "[gc]   If your supervisor thinks this model is stopped, that was an orphan holding VRAM." >&2
    echo "[gc]   Check with: $RUNTIME ps ; amd-smi monitor" >&2
  else
    echo "[gc] stale container $NAME found ($st); the launch will reclaim the name." >&2
  fi
}

# Leftover 2: a restart orphans the container but NOT its /dev/shm region. The stale mmap
# keeps its full size, so the next boot finds the tmpfs full and cannot place
# its own buffer. Reap regions that no live process still holds -- fuser is the
# precise test, so we only delete what is genuinely unreferenced, and a
# concurrently running second model keeps its buffer.
kvoff_reap_orphans() {
  local f
  [ "$GC_ORPHANS" = off ] && return 0
  shopt -s nullglob
  for f in /dev/shm/vllm_offload_*.mmap; do
    if command -v fuser >/dev/null 2>&1; then
      fuser -s "$f" 2>/dev/null && continue
    elif command -v lsof >/dev/null 2>&1; then
      lsof -t -- "$f" >/dev/null 2>&1 && continue
    else
      continue   # no way to prove it is unused; leave it alone
    fi
    echo "[gc] reaping orphaned shm region $(basename "$f") ($(( $(stat -c %s "$f") / 1024 / 1024 )) MiB)" >&2
    gc_do rm -f -- "$f"
  done
  shopt -u nullglob
}

# Resolve KV_OFFLOAD -> GiB, clamped to both limits. Echoes "" for no offload.
kvoff_resolve() {
  local want="$1" shm_free_b mem_total_b mem_avail_b cur_b=0 f
  # statvfs, not df: df rounds and would overstate the ceiling.
  read -r shm_free_b mem_total_b mem_avail_b <<<"$(
    python3 - <<'EOF'
import os
s = os.statvfs('/dev/shm')
free = s.f_bavail * s.f_frsize
mt = ma = 0
for line in open('/proc/meminfo'):
    k, v = line.split(':', 1)
    if k == 'MemTotal':     mt = int(v.split()[0]) * 1024
    elif k == 'MemAvailable': ma = int(v.split()[0]) * 1024
print(free, mt, ma)
EOF
  )" || return 0

  case "$want" in
    off|no|none|0) return 0 ;;
  esac

  local G=$((1024*1024*1024))
  # Ceiling 1: what the tmpfs can actually hold, less a margin for the small
  # shm files that come and go while the server runs.
  local shm_cap_b=$(( shm_free_b - KVOFF_SHM_MARGIN_MIB * 1024 * 1024 ))
  # Ceiling 2: what RAM can spare without starving page cache. MemAvailable
  # already discounts reclaimable cache, so subtract the floor we want left.
  local ram_cap_b=$(( mem_avail_b - KVOFF_KEEP_FREE_GIB * G ))

  local req_b
  case "$want" in
    auto)  req_b=$(( shm_cap_b < ram_cap_b ? shm_cap_b : ram_cap_b )) ;;
    *%)    req_b=$(python3 -c "print(int($mem_total_b * float('${want%\%}') / 100))") ;;
    *)     req_b=$(python3 -c "print(int(float('$want') * $G))") ;;
  esac

  (( req_b > shm_cap_b )) && req_b=$shm_cap_b
  (( req_b > ram_cap_b )) && req_b=$ram_cap_b

  # Both ceilings go NEGATIVE on a tight box (MemAvailable below KEEP_FREE, or
  # a full tmpfs), and a negative request must mean "no offload", never a
  # negative flag value. Guard that before the size test.
  (( req_b <= 0 )) && return 0
  # ...and do the minimum test in python: KVOFF_MIN_GIB is a user-supplied
  # value, and bash arithmetic cannot compare against a fractional one -- it
  # errors and the `&&` never fires, which used to emit a negative size.
  python3 -c "import sys; sys.exit(0 if $req_b < float('$KVOFF_MIN_GIB') * $G else 1)" && return 0

  # vLLM wants GiB; one decimal is plenty and keeps us under the clamp.
  python3 -c "import math; print(f'{math.floor($req_b / $G * 10) / 10:g}')"
}

# config.yaml may already pass --kv-offloading-size through EXTRA. That is the
# explicit, per-entry setting and it wins; adding a second copy of the flag here
# would leave vLLM parsing a duplicate.
gc_report_orphan_container
kvoff_reap_orphans

if [[ "$EXTRA" != *--kv-offloading-size* ]]; then
  KVOFF_GIB="$(kvoff_resolve "$KV_OFFLOAD")"
  if [[ -n "$KVOFF_GIB" ]]; then
    echo "[kv-offload] KV_OFFLOAD=$KV_OFFLOAD -> --kv-offloading-size $KVOFF_GIB (backend $KVOFF_BACKEND)" >&2
    EXTRA="$EXTRA --kv-offloading-size $KVOFF_GIB --kv-offloading-backend $KVOFF_BACKEND"
  elif [[ "$KV_OFFLOAD" != off ]]; then
    echo "[kv-offload] KV_OFFLOAD=$KV_OFFLOAD but too little spare RAM/shm to place a useful buffer; disabled." >&2
  fi
else
  echo "[kv-offload] --kv-offloading-size already set in EXTRA; leaving it alone." >&2
  # ...but still check it against the tmpfs. An explicit size bypasses
  # kvoff_resolve's clamp, and the offload region is pre-faulted with
  # MADV_POPULATE_WRITE: asking for more than /dev/shm can hold does not degrade,
  # it kills the START, and on 0.27.1 it does so without a log line. Refusing here
  # with a message is strictly better than that. statvfs, never df -- df rounds up.
  _kvoff_req="$(sed -n 's/.*--kv-offloading-size[= ]*\([0-9.]*\).*/\1/p' <<<"$EXTRA")"
  if [[ -n "$_kvoff_req" ]]; then
    _kvoff_shm="$(python3 -c "import os;s=os.statvfs('/dev/shm');print(s.f_blocks*s.f_frsize/2**30)")"
    if python3 -c "import sys;sys.exit(0 if float('$_kvoff_req')+0.25>float('$_kvoff_shm') else 1)"; then
      die "--kv-offloading-size $_kvoff_req GiB does not fit /dev/shm ($_kvoff_shm GiB usable).
     The region is pre-faulted, so this would fail the START with no log line.
     Fix ONE of:
       * mount -o remount,size=28G /dev/shm   (and add it to /etc/fstab --
         see kv-cache/ops/etc-fstab-snippets/kvcache.fstab; the kernel default is
         50% of RAM, which is why more RAM alone does not lift this)
       * lower --kv-offloading-size in the config.yaml entry
     Note df will disagree with this check: it rounds the tmpfs UP."
    fi
    # SECOND guard: /dev/shm is only a LIMIT, so fitting the tmpfs proves nothing
    # about the box actually having the RAM. Since 2026-09-08 shm is 28G to allow a
    # 24 GiB tier after the 32 -> 40 GB upgrade -- which means a 24 GiB request now
    # PASSES the tmpfs check on a 32 GB box and then OOMs it on the pre-fault. Clamp
    # to what MemTotal can hold instead of trusting the mount.
    # KVOFF_RAM_RESERVE_GIB is everything that is NOT the tier, plus headroom.
    # Measured 2026-09-08 (free -m, 16 GiB tier resident): total 32090 MB, used 22687,
    # shared 16728 -> non-tier usage ~5.9 GiB. 15 GiB of reserve leaves ~9 GiB spare,
    # which is what the box runs with today, and makes the clamp land on exactly the
    # right value at both sizes:  40 GB -> 24 GiB tier,  32 GB -> 16 GiB tier.
    # So a missed or failed RAM upgrade silently degrades to today's known-good
    # configuration instead of failing the boot.
    KVOFF_RAM_RESERVE_GIB=${KVOFF_RAM_RESERVE_GIB:-15}
    _kvoff_ram="$(python3 -c "
import re
mt=int(re.search(r'MemTotal:\s+(\d+)', open('/proc/meminfo').read()).group(1))
print('%.2f' % (mt/1048576))")"
    _kvoff_allow="$(python3 -c "print('%.2f' % max(0.0, float('$_kvoff_ram') - float('$KVOFF_RAM_RESERVE_GIB')))")"
    if python3 -c "import sys;sys.exit(0 if float('$_kvoff_req')>float('$_kvoff_allow') else 1)"; then
      _kvoff_clamped="$(python3 -c "import math;print(int(math.floor(float('$_kvoff_allow'))))")"
      if [[ "$_kvoff_clamped" -lt 4 ]]; then
        echo "[kv-offload] MemTotal ${_kvoff_ram} GiB - reserve ${KVOFF_RAM_RESERVE_GIB} GiB leaves" >&2
        echo "[kv-offload]   only ${_kvoff_allow} GiB; too little for a useful tier. DISABLING offload." >&2
        EXTRA="$(sed -E 's/--kv-offloading-size[= ]*[0-9.]+//; s/--kv-offloading-backend[= ]*[a-z]+//' <<<"$EXTRA")"
        KVOFF_DISK=""
      else
        echo "[kv-offload] *** CLAMPED: requested ${_kvoff_req} GiB, but MemTotal is only ${_kvoff_ram} GiB." >&2
        echo "[kv-offload]   ${_kvoff_ram} - ${KVOFF_RAM_RESERVE_GIB} reserve = ${_kvoff_allow} GiB usable -> tier ${_kvoff_clamped} GiB." >&2
        echo "[kv-offload]   If you expected the full ${_kvoff_req} GiB, the RAM upgrade did not reach this" >&2
        echo "[kv-offload]   guest -- it is a VM, so the HYPERVISOR allocation must be raised too." >&2
        EXTRA="$(sed -E "s/(--kv-offloading-size[= ]*)[0-9.]+/\1$_kvoff_clamped/" <<<"$EXTRA")"
        _kvoff_req="$_kvoff_clamped"
      fi
    fi
    echo "[kv-offload] explicit size ${_kvoff_req} GiB fits /dev/shm (${_kvoff_shm} GiB) and RAM (${_kvoff_ram} GiB)." >&2
  fi
fi

# Wire the secondary tier. cpu_bytes_to_use is deliberately ABSENT from this JSON:
# config/vllm.py:933 unconditionally .update()s it from --kv-offloading-size, so
# putting it here too would just be a value that loses. Everything else we set
# survives, because _post_init_kv_transfer_config only creates a default config
# when none was passed -- ours is preserved and merged into.
# NOTE: $EXTRA is word-split at the call site, so this JSON must contain NO spaces.
# kv_load_failure_policy: scheduler.py:130 defaults to recompute, but line 148
# overwrites it from the config WHENEVER one is passed -- and the dataclass
# default (config/kv_transfer.py:69) is "fail". Reaching the fs tier requires
# passing a config, so without this field we silently take "fail" instead of
# the default. Set it back to what we would have had.
# CAVEAT, measured 2026-09-06: this is INERT for OffloadingConnector. The
# recompute path is driven by get_block_ids_with_load_errors(), which only
# nixl, mooncake, flexkv and lmcache implement; OffloadingConnector inherits
# the base returning an empty set. Worse, offloading/worker.py:361 is a bare
# `assert transfer_result.success`, so a failed load kills EngineCore outright
# rather than failing the request either way. Keep the field for when upstream
# wires it up; do NOT rely on it to survive a reaped block.
if [[ -n "$KVOFF_DISK" ]]; then
  if [[ "$EXTRA" != *--kv-offloading-size* ]] && [[ -z "${KVOFF_GIB:-}" ]]; then
    echo "[kv-offload] KVOFF_DISK is set but there is no CPU primary tier; the fs tier" >&2
    echo "[kv-offload]   cannot reach the GPU on its own (tiering/spec.py) -- disabling it." >&2
    KVOFF_DISK=""
  elif [[ ! -d "$KVOFF_DISK" ]]; then
    echo "[kv-offload] KVOFF_DISK=$KVOFF_DISK does not exist; disabling the disk tier." >&2
    KVOFF_DISK=""
  else
    KVOFF_BPC_JSON=""
    KVOFF_POLICY_JSON=""
    if [[ "$KVOFF_POLICY" != lru ]]; then
      KVOFF_POLICY_JSON=",\"eviction_policy\":\"$KVOFF_POLICY\""
    fi
    if [[ "$KVOFF_STORE_THRESHOLD" -ge 2 ]] 2>/dev/null; then
      KVOFF_POLICY_JSON="$KVOFF_POLICY_JSON,\"store_threshold\":$KVOFF_STORE_THRESHOLD"
    fi
    if [[ -n "$KVOFF_BLOCKS_PER_CHUNK" ]]; then
      KVOFF_BPC_JSON=",\"blocks_per_chunk\":$KVOFF_BLOCKS_PER_CHUNK"
    fi
    KVOFF_TIER_ARG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\"$KVOFF_BPC_JSON$KVOFF_POLICY_JSON,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KVOFF_DISK_MNT/$KVOFF_DISK_SUBDIR\",\"n_read_threads\":$KVOFF_DISK_RTHREADS,\"n_write_threads\":$KVOFF_DISK_WTHREADS}]}}"
    mkdir -p "$KVOFF_DISK/$KVOFF_DISK_SUBDIR"
    echo "[kv-offload] L3 disk tier: $KVOFF_DISK -> $KVOFF_DISK_MNT/$KVOFF_DISK_SUBDIR (PYTHONHASHSEED=$KVOFF_HASHSEED," >&2
    if [[ -n "$KVOFF_BLOCKS_PER_CHUNK" ]]; then
      echo "[kv-offload]   NON-DEFAULT blocks_per_chunk=$KVOFF_BLOCKS_PER_CHUNK" >&2
    fi
    echo "[kv-offload]   CPU tier policy=$KVOFF_POLICY store_threshold=$KVOFF_STORE_THRESHOLD" >&2
    echo "[kv-offload]   ${KVOFF_DISK_RTHREADS}r/${KVOFF_DISK_WTHREADS}w threads, $(df -h --output=size "$KVOFF_DISK" | tail -1 | tr -d ' ') volume, NO built-in eviction)" >&2
  fi
fi
# ---------------------------------------------------------------------------
# Cudagraph capture sizes; empty/none = vLLM's default list ([1,2,4] + multiples of 8).
# Finer sizes (3,5,6,7,10,12,14) were tried 2026-08-29 to un-pad dynamic-width single streams and
# measured NEUTRAL (181.3 vs 184.7 weighted, inside noise): the decode-band GEMMs are
# weight-stream-bound and nearly M-invariant below M~16 (tier7: gate_up 88.5 us at M=5 vs 88.7
# at M=8), so there was no single-stream width cost hiding behind the padding to recover --
# dynamic width's value is batching, where M crosses real cost and split-K boundaries. The knob
# stays for capture experiments; the default stays stock. SPEC=8 + dynamic width was measured in
# the same session: single-stream 184.9 (even), conc-8 405-427 vs 444-461 (LOSES -- cold-start
# batches run full width into the M=72>64 kernel cliff before the EMAs settle). 7 stays.
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
# Greedy fill against a target RECOMPUTED from what is left, not a fixed total/nshard. With a
# fixed target every group that closes under it pushes its slack into the final group: the
# first cut of this checkpoint produced 2.4/4.6/4.6/6.9 GiB, and the 6.9 -- not the mean -- is
# what bounds the loader's working set. Re-deriving remaining/left after each cut spreads the
# slack instead of accumulating it, and it self-corrects if the size estimate above is off.
groups, cur, acc = [], [], 0
remaining, left = total, nshard
for k in keys:                      # keep the checkpoint's own tensor order
    target = remaining / left if left > 1 else float("inf")
    if cur and acc + sizes[k] > target:
        groups.append(cur); remaining -= acc; left -= 1; cur, acc = [], 0
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
    dst = os.path.join(snap, fn)
    os.replace(os.path.join(tmp, fn), dst)
    os.chmod(dst, 0o644)             # the container umask writes 0600; match the checkpoint
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
  # instead of argmax. Greedy one-hot drafts accept with only p_target(argmax); matched
  # sampling accepts with sum(min(p,q)), which is strictly >=.
  #
  # ISOLATED AND MEASURED 2026-09-05 (4 alternating boots, bench-live 4k/16k/50k, at the
  # serve's temperature 1.0): acceptance +10% (mean_len 3.39/3.25/3.15 vs greedy's
  # 3.01/2.90/3.03, winning 5 of 6 within-depth pairings), decode +12.8/+12.1/+3.9%, and
  # steps/s FLAT to 0.1% -- so the full draft-logits head this was supposed to cost is free
  # on this config. The old "measure acceptance vs that cost before defaulting" note is
  # answered: there is no cost to weigh. This was the prime suspect for the radlight decode
  # gap and it holds up.
  #
  # The edge is temperature-dependent (a flatter target makes p_target(argmax) fall faster
  # than sum(min(p,q))), so it is LARGER at our temperature 1.0 than at the 0.7 served until
  # 2026-09-05. Full write-up: bench-history/draft-sample-isolation-20260905/notes.md.
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
exec ${DRY_RUN:+echo} "$RUNTIME" run "${RT_FLAGS[@]}" --rm --name "$NAME" --privileged --ipc=host --network=host --ulimit memlock=-1 \
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
  ${KVOFF_DISK:+-v "$KVOFF_DISK":"$KVOFF_DISK_MNT"} \
  ${KVOFF_DISK:+-e PYTHONHASHSEED="$KVOFF_HASHSEED"} \
  -e RADIANCE_OFFLOAD_MIXED_HIT="$KVOFF_MIXED_HIT" \
  -e RADIANCE_OFFLOAD_PENDING_IS_MISS="$KVOFF_PENDING_IS_MISS" \
  -e RADIANCE_OFFLOAD_EAGLE_GROUPS="$KVOFF_EAGLE_GROUPS" \
  -e RADIANCE_MAMBA_STORE_STRIDE="$KVOFF_MAMBA_STRIDE" \
  -e RADIANCE_FS_FANOUT_TARGET_MB="$KVOFF_FS_FANOUT_MB" \
  -e RADIANCE_FS_FANOUT_MAX="$KVOFF_FS_FANOUT_MAX" \
  -v "$CACHE":/cache \
  -v "${PATCHES:-$REPO}":/patches:z \
  -v "$HOUSE":/house:z \
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
    # House patch: lives in /house, not /patches. PYTHONPATH lets it import the ggz14
    # _patchlib the same way the in-repo patches do (script dir, not cwd, is sys.path[0]).
    PYTHONPATH=/patches python3 /house/patch_offload_mixed_hit.py
    # House patch: KV offload store-path instrumentation. Adds gauges only, no behaviour
    # change. Without it an allocation failure is unattributable and the lookup-delay
    # histogram tops out at 10 s. See kv-cache/cache-preemption-patch-plan.md R3.6.
    PYTHONPATH=/patches python3 /house/patch_kv_offload_instrumentation.py
    # House patch: KV offload lookup-outcome counters. Instrumentation only. Answers why
    # production gets ~0 external hits when the bench gets an exact 85,696-token one, by
    # counting every terminal branch of the lookup path. See R3.14.2. Remove once Phase A
    # has read its answer -- it is a diagnostic, not a permanent metric.
    # Non-fatal by design: a diagnostic must never be able to keep the engine from
    # serving. Hunks are ordered so a mid-way failure still leaves a consistent file
    # (metric names first, then the call sites that use them).
    PYTHONPATH=/patches python3 /house/patch_kv_offload_lookup_outcomes.py \
      || echo "[radiance] WARNING: lookup-outcome counters did not apply; Phase A metrics will be absent"
    # House patch: serve the ready prefix instead of deferring on an in-flight store.
    # This is the fix Phase A pointed at, and the only BEHAVIOUR change in this block --
    # everything above it is instrumentation. Gated on RADIANCE_OFFLOAD_PENDING_IS_MISS so
    # it is an A/B, not a one-way door. Must run AFTER the lookup-outcome patch: it adds a
    # counter to the same two files and anchors on lines that patch inserts. See R3.14.3.
    # Non-fatal, but not harmless if it fails: the engine would run upstream behaviour while
    # the env var claims otherwise, so the warning names that explicitly, and phaseb-read.sh
    # checks the counter series exists before it reports anything.
    PYTHONPATH=/patches python3 /house/patch_kv_offload_serve_ready_prefix.py \
      || echo "[radiance] WARNING: serve-ready-prefix patch did NOT apply -- lookups will still defer on an in-flight store, whatever RADIANCE_OFFLOAD_PENDING_IS_MISS says"
    # House patch: annotate the EAGLE/MTP draft KV group positionally, so the offload
    # scheduler stops treating all nine groups as draft groups. Prerequisite for the stride
    # patch below: while every group is flagged eagle, storable_chunks() drops the trailing
    # chunk of each group during decode and the store grid stops lining up with the hit
    # window. Non-fatal: on failure the flag-them-all fallback is todays behaviour anyway.
    PYTHONPATH=/patches python3 /house/patch_kv_offload_eagle_groups.py \
      || echo "[radiance] WARNING: eagle-group annotation did NOT apply -- all nine KV groups will be treated as MTP draft groups, whatever RADIANCE_OFFLOAD_EAGLE_GROUPS says"
    # House patch: R3.13 Mamba store cadence -- the capacity lever. Must run AFTER the
    # eagle-group patch. The two halves (store grid, lookup rounding) are ordered so that a
    # mid-way failure degrades safely: lookup-only means a coarser hit window with a full
    # store, which costs prefix but stays correct.
    PYTHONPATH=/patches python3 /house/patch_kv_offload_mamba_stride.py \
      || echo "[radiance] WARNING: mamba store-cadence patch did NOT apply -- every chunk will store all six Mamba groups, whatever RADIANCE_MAMBA_STORE_STRIDE says"
    # House patch: R3.14 fs-tier job fanout, the port of upstream PR #49225. Order-independent
    # of the three above -- it touches a different file (v1/kv_offload/tiering/fs/manager.py)
    # and anchors nowhere near the cascade-backlog gauge the instrumentation patch adds there.
    # Non-fatal: on failure every fs job keeps running on one thread, which is todays
    # behaviour, so the promotion stays slow but nothing is wrong.
    PYTHONPATH=/patches python3 /house/patch_kv_offload_fs_fanout.py \
      || echo "[radiance] WARNING: fs fanout patch did NOT apply -- every filesystem job will run on a single thread, whatever RADIANCE_FS_FANOUT_TARGET_MB says"
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
  ${KVOFF_TIER_ARG:+--kv-transfer-config "$KVOFF_TIER_ARG"} \
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
