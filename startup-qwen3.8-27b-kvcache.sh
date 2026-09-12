#!/bin/bash
# Launcher for the KV-cache offload implementation: entry `qwen3.8-27b-kvcache`.
#
# ============================================================================
# WHAT THIS IS
# ============================================================================
# This is the KV-cache work as a SEPARATE, PUBLISHABLE ENTRY. Everything the
# three-tier offload needs -- the tier sizes, the nine house patches, the
# eviction policy, the fs tier, the stride, the GC contract -- is pinned HERE,
# in one file, with the reasoning inline, rather than being spread across the
# production entry's env block in config.yaml.
#
# It exists for two reasons:
#
#   1. DISTRIBUTION. Several people want this KV-cache stack and it is easier to
#      hand over one launcher + the kv-cache/ directory than to ask someone to
#      reproduce a 600-line config.yaml entry. This file plus its sibling docs
#      IS the deliverable.
#
#   2. ISOLATION. KV-cache experiments change tier geometry, and a changed tier
#      geometry is exactly the kind of thing that silently invalidates a
#      benchmark. Running them on their own entry, with their own container
#      name and their own block tree on disk, means an experiment can never
#      quietly pollute the production entry's cache or its numbers.
#
# It is a THIN WRAPPER, deliberately. It sets environment and then execs the
# real launcher (llama-swap-ggz14-27b.sh), which stays the single source of
# truth for how the model is served. Nothing about the model, the kernels, the
# drafter or the vLLM invocation is duplicated here -- only the KV-cache
# configuration. If the base launcher changes, this entry inherits the change.
#
# ============================================================================
# STATUS: WORKING, CORRECT, UNOPTIMISED -- READ BEFORE PUBLISHING OR TRUSTING
# ============================================================================
# This is a WORKING, CORRECT, UNOPTIMISED implementation delivered as a patch
# stack -- not a working implementation. It makes a three-tier KV-cache offload work
# on a GDN hybrid model on a single consumer AMD card (gfx1201 / RDNA4), and it
# is measured doing so: 81.8% of prompt tokens were served without recomputation
# over ~14 h of real agent work, and an 85,696-token disk hit was bit-identical
# to a cold recompute. The defects that were found are not still open -- they
# were fixed, and then the fixes were tested. It is a starting point, not a
# product, and it is published at this maturity deliberately: several people want
# the capability and the author has neither the time nor the specialist expertise
# to carry it to completion alone.
#
# WHAT THIS NEEDS TO BE A RELEASE
# (in order -- each stage's output is the next stage's input):
#
#   1. BENCHMARK HOOKS, METRICS AND A REPRODUCIBLE HARNESS. FIRST.
#      Tidy the instrumentation that already exists (the lookup-outcome and
#      instrumentation patches) into a coherent metrics surface, and write a
#      REPRODUCIBLE cache-metrics script in the spirit of BetterBench but aimed
#      at the tier rather than raw throughput: per-tier hit share, promotion
#      latency, load_bytes, read:write ratio, prefill avoided. It must NOT
#      repeat BetterBench's mistake -- vLLM matches the prefix cache by block
#      CONTENT, not by chained prefix, so varying a nonce in block 0 leaves the
#      body self-caching and the "cold" arm is not cold. Making the cold arm
#      genuinely cold is the hard part, and the reason this stage is first:
#      every claim below it is unfalsifiable until it exists.
#
#   2. CONTINUE THE REVIEW OF EXISTING WORK, AND WRITE AN IMPLEMENTATION PLAN.
#      kv-cache-references.md is the review so far -- every PR, paper and blog
#      assessed, each with a ruling. Continue it, then plan. This includes
#      reconciling the nine house patches against current vLLM/radiance HEAD
#      and DELETING each in favour of the upstream implementation wherever one
#      exists (the eagle-groups fix has a counterpart in PR #52047 (NOT merged as #55390; #52047 does not cover this model); the fs
#      fanout was ported from PR #49225). AVOID REIMPLEMENTATION -- a house
#      patch that duplicates merged upstream work is a liability, not an asset:
#      one more thing to rebase, and it will silently diverge. Score upstream
#      work by APPLICABILITY, not by merge status; an unmerged PR that fits is
#      worth more than a merged one that does not.
#
#   3. TUNABLE ACCURACY, AND A PoC OF THE MOST PROMISING QUALITY OPTIONS.
#      Correctness here is a dial, not a boolean, and the dial is currently
#      welded. Expose the accuracy/cost trade-offs as CONFIG TOGGLES rather
#      than constants -- the mamba stride N first, then the store threshold and
#      the pending-is-miss behaviour -- so a deployment can pick its point on
#      the curve and a benchmark can sweep it. Then prototype the most
#      promising quality option. The strongest candidate is THE EXACTNESS FIX:
#      replay the <= one-block gap from the stride checkpoint, turning the
#      stride from an approximation into an exact reconstruction. Designed but
#      not built; the single well-described gap to the full method.
#
#   4. REFACTORING. These patches were written one at a time, each to answer a
#      specific question, and it shows: they monkey-patch by string surgery,
#      they carry an implicit dependency ORDER that is stated in prose (the
#      launcher's comments and patches/README.md) rather than enforced in code,
#      and the gating env vars are inconsistent in naming and in
#      whether 0 or 1 means "upstream". This wants to be a single coherent
#      module with an explicit interface, not eight scripts in a trench coat.
#      It lands here, not earlier, because stages 2 and 3 decide how much of it
#      survives to be refactored at all.
#
#   5. SPEED OPTIMISATION. Nothing here has been tuned; it has only been made
#      to work. The known ceilings are all measured and all documented -- the
#      fs tier serves at ~117 MB/s against a ~101 MB/s recompute break-even
#      (1.16x, i.e. barely worth doing), the promotion path is strictly staged
#      through the CPU tier so disk and RAM contend for the same region, and
#      there is no DMA path from disk to VRAM on this hardware. Which of those
#      are real limits and which are just untuned is, in most cases, not yet
#      established -- and stage 1 is what settles it.
#
#   6. NORMAL SOFTWARE ENGINEERING. Code review, tests, CI, packaging, a real
#      release. None of it has happened. There is no test suite; correctness has
#      been established by hand, per-change, against a live endpoint.
#
# The specific limitations are enumerated in kv-cache-known-issues.md and
# kv-cache-future-work.md; the three that matter most:
#
#   * The mamba stride (N=8) is TRUNCATE-AND-RECOMPUTE; the cost is compute,
#     not accuracy. A lookup rounds DOWN to the nearest kept snapshot boundary,
#     so the engine only ever requests a state it actually kept -- what is
#     served is EXACT, and the gap (up to 13,184 tokens, ~6,592 on average) is
#     recomputed on the normal prefill path. The price is that recompute plus
#     the dead zone (prefixes under 13,184 tokens get a zero external hit). The
#     exactness fix (replay the <= one-block gap) is DESIGNED BUT NOT BUILT.
#     See future-work §L.
#
#   * A failed offload load KILLS EngineCore. `assert transfer_result.success`,
#     and OffloadingConnector has no get_block_ids_with_load_errors(), so
#     kv_load_failure_policy=recompute is inert here. This is why the reaper's
#     MIN_AGE floor is a hard safety property and not a tuning knob.
#
#   * The published BetterBench numbers for this stack are POLLUTED and must
#     never be cited. vLLM matches the prefix cache by block CONTENT, not by
#     chained prefix, and BetterBench's "cold (nonce)" varies only block 0, so
#     the body self-caches. The "tier earned its keep" measurement (~2.45M tokens served from the
#     RAM/disk tiers ~= ~26 min of prefill avoided at the honest cold rate) is
#     method only, not citable -- it was taken in the polluted pre-R3.15 window;
#     re-measurement is `status-2026-09-10.md` section 4.
#
# ============================================================================
# ONE GPU
# ============================================================================
# This entry takes the whole R9700, exactly like qwen3.8-27b-vllm. They cannot
# be loaded together and llama-swap will not evict a ttl:0 entry, so the switch
# is manual:
#     podman stop qwen38-27b-vllm       then use qwen3.8-27b-kvcache
#     podman stop qwen38-27b-kvcache    then use qwen3.8-27b-vllm
#
# ============================================================================
# USAGE
# ============================================================================
#     llama-swap-qwen38-27b-kvcache.sh --port <N>    (llama-swap's contract)
#     llama-swap-qwen38-27b-kvcache.sh -h            this text, then the base
#                                                    launcher's full knob list
#
# Every knob below is overridable from the environment (config.yaml `env:`),
# and every knob the base launcher understands still works and still wins if
# you set it -- this file only supplies DEFAULTS.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOUSE="${HOUSE:-$HERE/kv-cache/patches}"
BASE_LAUNCHER="${BASE_LAUNCHER:-$HERE/kv-cache/launcher/serve-mxfp4-kvcache-base.sh}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  echo
  echo "=== base launcher knobs ($BASE_LAUNCHER) ==="
  exec "$BASE_LAUNCHER" -h
fi

[[ -x "$BASE_LAUNCHER" ]] || { echo "kvcache: base launcher not found or not executable: $BASE_LAUNCHER" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. IDENTITY -- distinct container + served id.
#
# The entry key in config.yaml, SERVED, and NAME must move together. llama-swap
# forwards the requested model id verbatim and vLLM validates it, so if they
# drift the request 404s AFTER the model has finished loading -- a six-minute
# way to discover a typo. cmdStop in config.yaml must name the same container.
# ---------------------------------------------------------------------------
export NAME="${NAME:-qwen38-27b-kvcache}"
export SERVED="${SERVED:-qwen3.8-27b-kvcache}"

# ---------------------------------------------------------------------------
# 2. THE CPU PRIMARY TIER (L2) -- size in GiB.
#
# Set EXPLICITLY, not via KV_OFFLOAD=auto, and that is deliberate: `auto` sizes
# from MemAvailable at boot, which depends on what else happened to be resident
# at that moment. A benchmark entry whose tier size depends on boot timing is
# useless. Explicit + clamped is reproducible.
#
# 24 GiB is the known-good size on this box (39.17 GiB guest). It holds roughly
# 419,000 tokens at 61,440 bytes/token as stored -- about 12 prompts of 34k.
# See kv-cache-operations.md §3 for the full sizing table and the OOM warning
# before you raise it.
#
# THE TIER MUST FIT /dev/shm. It is pre-faulted (MADV_POPULATE_WRITE) and pinned
# (cudaHostRegister); overshooting the tmpfs fails the START and on vLLM 0.27.1
# it dies WITHOUT A LOG LINE. Keep /dev/shm a few GiB above the tier.
# ---------------------------------------------------------------------------
KVCACHE_TIER_GIB="${KVCACHE_TIER_GIB:-24}"

# KVOFF_RAM_RESERVE_GIB: everything that is NOT the tier, plus headroom. The
# base launcher clamps the tier down to (MemTotal - this) and says so loudly in
# the boot log. Measured non-tier footprint on this box is ~7 GiB, so 15 is
# generous; raise the tier and you must lower this in the same edit or the
# clamp will silently give you back the old size. Do NOT remove the clamp --
# it is the only thing between a typo and an unbootable box.
export KVOFF_RAM_RESERVE_GIB="${KVOFF_RAM_RESERVE_GIB:-15}"

# ---------------------------------------------------------------------------
# 3. THE FS SECONDARY TIER (L3) -- the disk.
#
# Set KVCACHE_DISK="" to turn the disk tier off entirely and run RAM-only. That
# is a supported configuration and costs nothing but capacity: the disk reads at
# ~117 MB/s against a ~101 MB/s recompute break-even, i.e. only 1.16x, so
# serving from disk is barely faster than recomputing. See operations §1.
#
# THREE THINGS THAT WILL BITE (all verified in the 0.27.1 tree):
#   1. PYTHONHASHSEED must be pinned. Block filenames are content hashes chained
#      from NONE_HASH, seeded from os.urandom(32) when unset -- so every restart
#      would hash the same tokens to DIFFERENT filenames and orphan the whole
#      on-disk cache, silently, at a 100% miss rate with no error. The base
#      launcher pins it to 0.
#   2. NO EVICTION. tiering/fs/manager.py has no capacity, quota or TTL
#      parameter and exposes no eviction hook: this tier writes and NEVER
#      deletes. An external reaper is MANDATORY. See §6.
#   3. O_DIRECT. Reads and writes bypass the page cache in both directions, so
#      spare RAM cannot act as a read cache in front of this tier. The only
#      productive home for spare RAM is the primary tier.
# ---------------------------------------------------------------------------
export KVOFF_DISK="${KVCACHE_DISK-/kvcache}"

# Its own block tree, NOT the production entry's `blocks`.
#
# Sharing would warm faster, but it would also mean an experiment on this entry
# writes into the tree production reads from. Block keys already include the
# model and the block geometry, so the two trees would not corrupt each other --
# but they would share the reaper's capacity budget, and a measurement taken
# against a tree someone else is writing into is not a measurement. The lesson
# is expensive and already learned once: see the BetterBench pollution note above.
#
# To deliberately share production's warm cache, set KVCACHE_DISK_SUBDIR=blocks.
export KVOFF_DISK_SUBDIR="${KVCACHE_DISK_SUBDIR:-blocks-kvcache}"

# Read/write thread counts. Measured on the 512 GB zvol via O_DIRECT, MB/s:
#   reads   1thr 87.6 | 4thr 268.9 | 8thr 719.3 | 16thr 435.6   <- peak at 8
#   writes  1thr 1100                                           <- never the limit
# 16 regresses on raidz contention; writes stay low because 1.1 GB/s at one
# thread is already ~24x what this tier actually stores.
export KVOFF_DISK_RTHREADS="${KVOFF_DISK_RTHREADS:-8}"
export KVOFF_DISK_WTHREADS="${KVOFF_DISK_WTHREADS:-4}"

# ---------------------------------------------------------------------------
# 4. THE HOUSE PATCHES -- the eight, and why each is on.
#
# Applied at container start, in dependency order, from this directory (bind
# mounted at /house). The upstream clone stays pristine. Each is gated by an
# env var so it can be reverted without unpatching.
# ---------------------------------------------------------------------------

# (a) mixed-hit guard. Stops OffloadingConnector killing the engine on a mixed
#     local+external prefix hit. 1 = serve mixed hits = default = safe, 0 = decline = the retained kill switch.
export KVOFF_MIXED_HIT="${KVOFF_MIXED_HIT:-1}"

# (b) REMOVED 2026-09-12 -- there used to be a KVOFF_PENDING_IS_MISS knob here, described as
#     "THE ONE THAT MAKES THE FS TIER ACTUALLY SERVE". It was the gate on
#     patch_kv_offload_serve_ready_prefix.py, which was deleted along with the
#     promotion-refusal thesis it was built on (0 refusals across 5,416 promotions at a
#     100%-full CPU tier). The code no longer reads the variable at all, so setting it did
#     nothing except print a reassuring line in the boot log. Upstream's wait-for-the-
#     promotion behaviour -- which is what the knob emulated at 0 -- is now simply what runs.
#     Do not reintroduce it without the patch.

# (c) eagle-groups. vLLM's MTP-draft-group annotator is hard-gated to
#     DeepSeek-V4, so on this model production was flagging ALL NINE KV groups
#     as draft groups. The patch flags only group 8. The boot log line must read
#     [8], not all nine. Upstream PR #52047 (NOT merged as #55390; #52047 does not cover this model).
export KVOFF_EAGLE_GROUPS="${KVOFF_EAGLE_GROUPS:-1}"

# (d) mamba stride N=8. THE CAPACITY LEVER, and the approximation -- see the
#     status warning at the top of this file. 0.417x the bytes, which lifts the
#     RAM tier from 0.50x the GPU cache to 1.21x. Set 1 for exact-but-huge.
export KVOFF_MAMBA_STRIDE="${KVOFF_MAMBA_STRIDE:-8}"

# (e) fs fanout. Splits one promotion across up to N parallel read tasks
#     (upstream does one task per job). Ported from PR #49225. It splits the
#     work exactly 8 ways as designed -- and the disk still gives ~117 MB/s
#     either way, because this device is the limit, not the concurrency. Kept
#     because it is correct and free, not because it measured a win.
#     Set KVOFF_FS_FANOUT_MAX=1 to restore upstream exactly without unpatching.
export KVOFF_FS_FANOUT_MB="${KVOFF_FS_FANOUT_MB:-256}"
export KVOFF_FS_FANOUT_MAX="${KVOFF_FS_FANOUT_MAX:-0}"

# (f) + (g) instrumentation and lookup-outcomes are unconditional -- they only
#     add metrics. They are what makes any of this diagnosable:
#     kv_offload_cpu_cache_evictable_perc and _free_perc are the two terms
#     prepare_store() tests for admission; kv_offload_fs_inflight_jobs is the
#     cascade backlog that pins the CPU tier shut.
#
#     DO NOT read kv_offload_cpu_cache_usage_perc as residency. It subtracts
#     evictable blocks, so it means "fraction pinned by in-flight transfers" and
#     reads 0.0 at idle with hundreds of GB on the fs tier.

# (h) tier report metrics -- also unconditional, also metrics-only. Adds the 19
#     per-tier `vllm:kv_offload_tier_*` series (load/store bytes, tokens and
#     latency histograms per tier, capacity, occupancy, reads-before-evict,
#     eviction-to-reuse, prefill stall) that `tools/tierreport.py` reads to
#     answer "is my RAM the right size, is my disk too slow, is this cache
#     worth running at all". It NEEDS (f) applied first -- it wraps the same
#     call sites -- so it is last in the apply order. Everything it emits is a
#     counter or a histogram, never a gauge, because the report is scraped once
#     from a long-lived server: an instantaneous level carries no information
#     after three weeks of uptime.

# ---------------------------------------------------------------------------
# 5. THE CPU TIER EVICTION POLICY.
#
# ARC, not LRU. The tier holds ~22 prompts, and LRU's classic failure is exactly
# our access pattern: write one prompt, read a dozen others, and the first is
# evicted before it is ever reused. We measured that and misread it as an
# experimental artifact for some time. ARC (T1 recency + T2 frequency + B1/B2
# ghost lists) is scan-resistant and costs no extra RAM or CPU.
#
# This is also the correct resolution of the "turn it over to ZFS as an ARC"
# instinct -- right idea, wrong layer. The ARC that helps is inside vLLM.
#
# Reversible: set lru. Pure policy swap, no on-disk or on-wire format change.
export KVOFF_POLICY="${KVOFF_POLICY:-arc}"

# Admission filter: a block must be SEEN in lookup() this many times before it
# is stored. 0 = store everything (default). NOT recommended above 1 without a
# deliberate A/B: at 2, a prefix reused exactly twice is never served at all,
# because admission is delayed by one occurrence.
export KVOFF_STORE_THRESHOLD="${KVOFF_STORE_THRESHOLD:-0}"

# ---------------------------------------------------------------------------
# 6. THE GARBAGE COLLECTOR -- MANDATORY, NOT OPTIONAL.
#
# The fs tier writes and never deletes (see §3.2). Two layers:
#   * startup: the base launcher reports orphan containers and reaps
#     unreferenced /dev/shm regions, fuser-gated.
#   * runtime: kvcache-reap.sh on a systemd timer, 5-minute cadence, 2-stage
#     age/capacity, with a MIN_AGE=90min HARD SAFETY FLOOR.
#
# The floor is a safety property, not a tuning knob: reaping a block that is
# in flight kills EngineCore (see the status warning at the top). Never lower it
# and never hand-delete a young block.
#
# This wrapper only WARNS if the timer is missing -- it will not start it for
# you, because installing a system unit is not a launcher's job.
# ---------------------------------------------------------------------------
if [[ -n "${KVOFF_DISK:-}" ]]; then
  if ! systemctl is-enabled kvcache-reap.timer >/dev/null 2>&1; then
    echo "[kvcache] *** WARNING: fs tier is ON but kvcache-reap.timer is not enabled." >&2
    echo "[kvcache]     This tier NEVER deletes on its own. Without the reaper, $KVOFF_DISK" >&2
    echo "[kvcache]     fills until the filesystem is full. Install it with:" >&2
    echo "[kvcache]       sudo cp $HERE/../kvcache-reap.{service,timer} /etc/systemd/system/" >&2
    echo "[kvcache]       sudo systemctl daemon-reload" >&2
    echo "[kvcache]       sudo systemctl enable --now kvcache-reap.timer" >&2
  fi
fi

# ---------------------------------------------------------------------------
# 7. WIRE THE TIER SIZE IN, then hand over.
#
# Passed via EXTRA so it takes the base launcher's EXPLICIT-size path (which
# validates against /dev/shm and applies the RAM clamp) rather than the
# MemAvailable-derived `auto` path. If the caller already put a size in EXTRA,
# theirs wins and we add nothing.
# ---------------------------------------------------------------------------
EXTRA="${EXTRA:-}"
if [[ "$EXTRA" != *--kv-offloading-size* ]]; then
  EXTRA="$EXTRA --kv-offloading-size $KVCACHE_TIER_GIB --kv-offloading-backend ${KVOFF_BACKEND:-native}"
fi
export EXTRA

echo "[kvcache] entry=$SERVED container=$NAME" >&2
echo "[kvcache]   CPU tier ${KVCACHE_TIER_GIB} GiB, policy=${KVOFF_POLICY}, reserve=${KVOFF_RAM_RESERVE_GIB} GiB" >&2
echo "[kvcache]   fs tier ${KVOFF_DISK:-<off>}${KVOFF_DISK:+/$KVOFF_DISK_SUBDIR}" >&2
echo "[kvcache]   mamba_stride=${KVOFF_MAMBA_STRIDE} eagle_groups=${KVOFF_EAGLE_GROUPS}" >&2
echo "[kvcache]   base launcher: $BASE_LAUNCHER" >&2

exec "$BASE_LAUNCHER" "$@"
