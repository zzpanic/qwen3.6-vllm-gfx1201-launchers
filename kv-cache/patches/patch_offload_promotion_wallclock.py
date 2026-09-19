#!/usr/bin/env python3
"""Consolidated: promotion_refusal_instrumentation then wallclock_reanchored (folds the apply-bundle wrapper).
Applied in this order; the second patch anchors on the first patch output, so the
order is load-bearing. A first-patch failure aborts before the second runs.
"""
"""radiance: attribute the KV-offload promotion refusals (task 00, Phase A).

WHY
===
When a secondary tier (fs) reports a block HIT, the tiering manager promotes it
into the primary (CPU) tier by calling the primary's prepare_write() (==
prepare_store()). If the primary tier cannot immediately free a slot,
prepare_store() returns None and the promotion is refused; the tiering manager
then reports the block as MISS (tiering/manager.py:397) and the engine prefills
from scratch.

That refusal increments NOTHING. Every refusal is folded into
lookup_chunk_miss_total and is indistinguishable from a genuine miss, which is
why this took reading the code to find rather than reading /metrics. The fix is
to name the refusal: count it, label it by source tier, split the two
prepare_store() return-None paths (they have different fixes), and count the
promotions that DID land so a refusal RATE is computable.

WHAT IT ADDS
============
All names are `vllm:kv_offload_promotion_*` and carry a single `tier` label
whose value is the SOURCE tier's tier_type ("fs", ...) -- the tier the block was
found on, not the CPU tier it failed to land in.

    promotion_refused{tier}                 a secondary hit could not be promoted
                                           because the primary tier was full
    promotion_refused_no_evictable{tier}    the refusal path: not enough
                                           evictable blocks (fix: reserved
                                           headroom, B2)
    promotion_refused_protected{tier}       the refusal path: the evictable
                                           blocks are held by concurrent
                                           references (fix: bounded retry, B1)
    promotion_initiated{tier}              a promotion that landed
                                           (refusal rate =
                                           refused / (refused + initiated))

The two paths need separate counters because they have different fixes:
  * NO_EVICTABLE: there are not enough ref_cnt==0 blocks to evict. The tier is
    simply full. A reserved headroom (B2) keeps a slice free so a promotion
    always has somewhere to land.
  * PROTECTED: there are enough evictable blocks in count, but the specific
    ones the policy would pick are held (ref_cnt>0) by a concurrent request
    mid-step. A bounded retry (B1) waits for the references to drop.

The two are distinguished by making prepare_store() return a
PrepareStoreRefusal(reason) sentinel instead of None on those two paths. The
normal GPU->CPU store path (tiering/manager.py:613) treats the sentinel exactly
like today's None (a failure); only the promotion caller reads the reason.

RISK
====
Instrumentation only. A refusal still returns False -> MISS exactly as today; a
successful promotion is unchanged. The only behavioural surface touched is that
prepare_store() now returns a small sentinel object instead of None on the two
refusal paths, and the two call sites both check for it with `is None or
isinstance(..., PrepareStoreRefusal)`, so a future upstream refactor back to None
degrades to "the reason counters read zero" rather than crashing. Every
emission is guarded so a rename degrades to "no metric".

Metric definitions are registered in get_connector_metric_definitions() at the
connector level so they exist for every offloading spec; OffloadPromMetrics.
observe() asserts that every emitted name is registered.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KVO_BASE = SP / "vllm/v1/kv_offload" / "base.py"
CPU_MANAGER = SP / "vllm/v1/kv_offload" / "cpu" / "manager.py"
TIERING_BASE = SP / "vllm/v1/kv_offload" / "tiering" / "base.py"
TIERING_MANAGER = SP / "vllm/v1/kv_offload" / "tiering" / "manager.py"
CONN_METRICS = (
    SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"
)

print("radiance: KV offload promotion-refusal instrumentation")

# ---------------------------------------------------------------------------
# 1. base.py: the PrepareStoreRefusal sentinel, next to PrepareStoreOutput.
# ---------------------------------------------------------------------------
apply(
    KVO_BASE,
    """@dataclass
class PrepareStoreOutput:
    keys_to_store: list[OffloadKey]
    store_spec: LoadStoreSpec
    evicted_keys: list[OffloadKey]""",
    """@dataclass
class PrepareStoreOutput:
    keys_to_store: list[OffloadKey]
    store_spec: LoadStoreSpec
    evicted_keys: list[OffloadKey]


class PrepareStoreRefusal:
    \"\"\"radiance promotion-refusal: prepare_store() declined to allocate.

    Carries which of the two refusal paths caused it, because they have
    different fixes (NO_EVICTABLE -> reserved headroom; PROTECTED -> bounded
    retry) and therefore need different counters. The normal store path treats
    this exactly like today's None; only the promotion caller reads `reason`.
    \"\"\"

    NO_EVICTABLE = "no_evictable"
    PROTECTED = "protected"

    def __init__(self, reason: str) -> None:
        self.reason = reason""",
    "class PrepareStoreRefusal:",
    "1  base.py: PrepareStoreRefusal sentinel",
)

# ---------------------------------------------------------------------------
# 2. tiering/base.py: the metric-name table entries.
# ---------------------------------------------------------------------------
apply(
    TIERING_BASE,
    """    LOOKUP_SYNC_DELAY = "vllm:kv_offload_tiering_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_tiering_lookup_async_delay_seconds\"""",
    """    LOOKUP_SYNC_DELAY = "vllm:kv_offload_tiering_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_tiering_lookup_async_delay_seconds"

    # radiance promotion-refusal: a secondary-tier hit that could not be
    # promoted because the primary tier was full. Counted by SOURCE tier.
    PROMOTION_REFUSED = "vllm:kv_offload_promotion_refused"
    PROMOTION_REFUSED_NO_EVICTABLE = (
        "vllm:kv_offload_promotion_refused_no_evictable"
    )
    PROMOTION_REFUSED_PROTECTED = "vllm:kv_offload_promotion_refused_protected"
    PROMOTION_INITIATED = "vllm:kv_offload_promotion_initiated\"""",
    "PROMOTION_INITIATED = \"vllm:kv_offload_promotion_initiated\"",
    "2  tiering/base.py: promotion-refusal name table",
)

# ---------------------------------------------------------------------------
# 3. offloading/metrics.py: register the definitions (label `tier`).
# ---------------------------------------------------------------------------
apply(
    CONN_METRICS,
    """from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    TierReportMetrics,
)""",
    """from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    TierReportMetrics,
)
from vllm.v1.kv_offload.tiering.base import TieringOffloadingMetrics""",
    "from vllm.v1.kv_offload.tiering.base import TieringOffloadingMetrics",
    "3a offloading/metrics.py: import TieringOffloadingMetrics",
)

apply(
    CONN_METRICS,
    """        TierReportMetrics.STALL_SECONDS: OffloadingCounterMetadata(
            documentation=(
                "Seconds requests spent waiting on this tier before being "
                "allocated. This is what a slow device costs, and it is the "
                "number that says a tier is not merely failing to help but "
                "actively hurting."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
    }""",
    """        TierReportMetrics.STALL_SECONDS: OffloadingCounterMetadata(
            documentation=(
                "Seconds requests spent waiting on this tier before being "
                "allocated. This is what a slow device costs, and it is the "
                "number that says a tier is not merely failing to help but "
                "actively hurting."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        # radiance promotion-refusal: a secondary-tier hit that could not be
        # promoted because the primary tier was full. Without these, every
        # such refusal is folded into lookup_chunk_miss_total and is
        # indistinguishable from a genuine miss. Labelled by source tier.
        TieringOffloadingMetrics.PROMOTION_REFUSED: OffloadingCounterMetadata(
            documentation=(
                "Secondary-tier hits refused promotion because the primary "
                "tier could not immediately free a slot, by source tier. "
                "Verdict (2026-09-12): 0 refusals across 5,416 promotions "
                "with the CPU tier pinned at 100% for 90+ min. The 'a full "
                "primary tier causes refusals' thesis did not hold, so this "
                "counter stays 0; it is kept as the evidence and as the "
                "tripwire in case the CPU tier is ever shrunk (a deferred, "
                "not cancelled, decision)."
            ),
            labelnames=("tier",),
        ),
        TieringOffloadingMetrics.PROMOTION_REFUSED_NO_EVICTABLE: (
            OffloadingCounterMetadata(
                documentation=(
                    "Refusals because there were not enough evictable blocks; "
                    "the tier is simply full. The fix is reserved headroom "
                    "(kv_offload_promotion_headroom_frac, B2)."
                ),
                labelnames=("tier",),
            )
        ),
        TieringOffloadingMetrics.PROMOTION_REFUSED_PROTECTED: (
            OffloadingCounterMetadata(
                documentation=(
                    "Refusals because the evictable blocks are held by "
                    "concurrent references. The fix is a bounded retry "
                    "(the promotion retry bound, B1)."
                ),
                labelnames=("tier",),
            )
        ),
        TieringOffloadingMetrics.PROMOTION_INITIATED: OffloadingCounterMetadata(
            documentation=(
                "Promotions that landed. Refusal rate = refused / "
                "(refused + initiated); shares the source-tier label."
            ),
            labelnames=("tier",),
        ),
    }""",
    "TieringOffloadingMetrics.PROMOTION_INITIATED: OffloadingCounterMetadata(",
    "3b offloading/metrics.py: register promotion-refusal definitions",
)

# ---------------------------------------------------------------------------
# 4. cpu/manager.py: import the sentinel, widen the annotation, and turn the
#    two return-None paths into labelled sentinels.
# ---------------------------------------------------------------------------
apply(
    CPU_MANAGER,
    """    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    TierReportMetrics,
)""",
    """    OffloadKey,
    PrepareStoreOutput,
    PrepareStoreRefusal,
    ReqContext,
    RequestOffloadingContext,
    TierReportMetrics,
)""",
    "    PrepareStoreRefusal,\n    ReqContext,",
    "4a cpu/manager.py: import PrepareStoreRefusal",
)

apply(
    CPU_MANAGER,
    """    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:""",
    """    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | PrepareStoreRefusal | None:""",
    "-> PrepareStoreOutput | PrepareStoreRefusal | None:",
    "4b cpu/manager.py: widen prepare_store annotation",
)

apply(
    CPU_MANAGER,
    """            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None""",
    """            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                # radiance promotion-refusal: not enough evictable blocks; the
                # fix is a reserved headroom (B2). Distinct from PROTECTED.
                return PrepareStoreRefusal(PrepareStoreRefusal.NO_EVICTABLE)""",
    "return PrepareStoreRefusal(PrepareStoreRefusal.NO_EVICTABLE)",
    "4c cpu/manager.py: NO_EVICTABLE refusal sentinel",
)

apply(
    CPU_MANAGER,
    """            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None""",
    """            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                # radiance promotion-refusal: the evictable blocks are held by
                # concurrent references; the fix is a bounded retry (B1).
                return PrepareStoreRefusal(PrepareStoreRefusal.PROTECTED)""",
    "return PrepareStoreRefusal(PrepareStoreRefusal.PROTECTED)",
    "4d cpu/manager.py: PROTECTED refusal sentinel",
)

# ---------------------------------------------------------------------------
# 5. tiering/manager.py: import the sentinel, count refusals by source tier and
#    path in _initiate_promotion, count initiations, and treat the sentinel as
#    a failure in the GPU->CPU store path.
# ---------------------------------------------------------------------------
apply(
    TIERING_MANAGER,
    """    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierReportMetrics,
    get_offload_block_hash,
)""",
    """    OffloadPolicy,
    PrepareStoreOutput,
    PrepareStoreRefusal,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierReportMetrics,
    get_offload_block_hash,
)""",
    "    PrepareStoreRefusal,\n    ReqContext,\n    RequestOffloadingContext,\n    ScheduleEndContext,",
    "5a tiering/manager.py: import PrepareStoreRefusal",
)

apply(
    TIERING_MANAGER,
    """        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the block as unavailable
            # rather than retrying indefinitely.
            return False

        store_spec = primary_write_result.store_spec""",
    """        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None or isinstance(
            primary_write_result, PrepareStoreRefusal
        ):
            # radiance promotion-refusal: attribute the refusal by source tier
            # and by which prepare_store() path caused it. Still a refusal:
            # today's MISS behaviour is unchanged.
            self._count_promotion_refusal(tier, primary_write_result)
            return False

        # radiance promotion-refusal: a promotion that landed.
        self._stats.increase_counter(
            TieringOffloadingMetrics.PROMOTION_INITIATED, 1, (tier.tier_type,)
        )
        store_spec = primary_write_result.store_spec""",
    "self._count_promotion_refusal(tier, primary_write_result)",
    "5b tiering/manager.py: count refusals and initiations",
)

apply(
    TIERING_MANAGER,
    """        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _flush_pending_promotions(self) -> None:""",
    """        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _count_promotion_refusal(self, tier: SecondaryTierManager, result) -> None:
        \"\"\"radiance promotion-refusal: attribute a refused promotion.

        Counts the refusal by source tier and by which prepare_store() path
        caused it (they have different fixes). Guarded: a metrics path must
        never be able to fail a lookup.
        \"\"\"
        try:
            label = (tier.tier_type,)
            self._stats.increase_counter(
                TieringOffloadingMetrics.PROMOTION_REFUSED, 1, label
            )
            if isinstance(result, PrepareStoreRefusal):
                if result.reason == PrepareStoreRefusal.NO_EVICTABLE:
                    self._stats.increase_counter(
                        TieringOffloadingMetrics.PROMOTION_REFUSED_NO_EVICTABLE,
                        1,
                        label,
                    )
                else:
                    self._stats.increase_counter(
                        TieringOffloadingMetrics.PROMOTION_REFUSED_PROTECTED,
                        1,
                        label,
                    )
        except Exception:
            pass

    def _flush_pending_promotions(self) -> None:""",
    "def _count_promotion_refusal",
    "5c tiering/manager.py: _count_promotion_refusal helper",
)

apply(
    TIERING_MANAGER,
    """        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None""",
    """        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None or isinstance(
            primary_result, PrepareStoreRefusal
        ):
            return None""",
    "if primary_result is None or isinstance(",
    "5d tiering/manager.py: treat the sentinel as a store failure",
)

"""Time each secondary-tier transfer ONCE, in wall clock, not per I/O batch.

WHY
===
`patch_kv_offload_tier_report.py` wraps every I/O batch a secondary tier enqueues
(`_tr_time_task` on the fs tier) and SUMS the per-batch wall times into
`kv_offload_tier_load_seconds{tier}` / `..._store_seconds{tier}`. On a tier whose
I/O runs on a worker pool -- the fs tier runs 8 read and 4 write threads -- that
sum is THREAD-SUMMED seconds, not wall time. Eight batches each taking 1 s while
their siblings share the device sum to 8 s, but the wall clock advanced 1 s.

Consequences, both already live in `tierreport.py`:

  * Every rate derived from those counters (MB/s, equivalent tok/s, the break-even
    and value verdicts) is a LOWER BOUND, not a measurement. `tierreport.py`
    prints `>= 28 MB/s` and returns UNKNOWN for the verdicts it cannot honestly
    compute, with a `--serial-io` flag for deployments where the counters really
    are wall time.
  * The register records the reopen condition explicitly: "When the patch is
    changed to time the whole promotion once instead of each batch"
    (`kv-cache-closed-decisions.md` section 4).

This patch writes that change.

WHAT IT CHANGES
===============
A secondary-tier transfer (a promotion, or a cascade store) is now timed ONCE,
end to end, in wall clock: from the moment the batch is SUBMITTED to the moment
its LAST task COMPLETES. That span is observable in the tiering manager, which is
the only place that sees both ends:

  * start   -- `tier.submit_load()` in `_flush_pending_promotions()` and
               `tier.submit_store()` in `complete_store()` / the request-level
               cascade. Recorded into a side map keyed by job id.
  * end     -- the job appears in `tier.get_finished_jobs()` in
               `_process_finished_jobs()`. The pool's `JobState` only reports a
               job finished once every one of its tasks has called `task_done()`,
               so that moment IS "the last task completed".

The difference is charged to a NEW, clearly-named counter per job. No new
threading primitive: the side map is touched only on the scheduler thread (both
submit and completion run there), so it needs no lock.

Because the per-batch signal is a real, useful diagnostic (pool busy-seconds,
from which pool saturation follows: thread-seconds / wall-seconds ~= concurrency),
it is KEPT, but under a new name so nothing is silently repurposed:

  * `kv_offload_tier_load_seconds` / `_store_seconds`
      WERE: thread-summed per-batch wall time (the defect).
      NOW:  wall-clock whole-job time, one sample per job. The name is now
           honest -- "the seconds this tier spent moving bytes" is literally the
           wall time of the transfer. This is the reopen condition the register
           was waiting on.
  * `kv_offload_tier_load_thread_seconds` / `_store_thread_seconds`  (NEW)
      The thread-summed per-batch total that `..._load_seconds` used to be,
      preserved for pool diagnosis.
  * `kv_offload_tier_load_latency_seconds` / `_store_latency_seconds`  (UNCHANGED)
      The per-batch latency HISTOGRAM. A batch's wall time, measured on the one
      pool thread that runs it, is a real quantity and stays as is.

The byte, per-batch-op, and per-batch-latency emitters in the fs tier are
untouched; only which seconds counter the per-batch accumulator is charged to
changes (`..._load_seconds` -> `..._load_thread_seconds`).

RISK
====
Instrumentation only. No scheduling, eviction, transfer or correctness path
changes behaviour; every hunk either adds a side map, records a monotonic
timestamp, or re-points an already-computed accumulator at a different (newly
registered) counter name. The fs tier's per-batch `_tr_time_task` wrapper is left
entirely in place -- it still feeds the byte counter, the per-batch op count,
the per-batch latency histogram, and (now) the thread-summed seconds counter.

Every emission site is guarded so a rename or refactor upstream degrades to "no
metric" rather than crashing the metrics path, in the same style as
`patch_kv_offload_tier_report.py`. The wall-clock span degrades to "no counter"
if a job is completed without a recorded submit time (the `pop(..., None)` guard),
so a bookkeeping drift can never fail a load or a store.

REVERT
======
This patch is additive on top of `patch_kv_offload_tier_report.py`. To revert:
delete the two `..._thread_seconds` names from `base.py` and their definitions from
`offloading/metrics.py`; drop the `self._tr_job_wall` map and the
`_tr_start_job()` helper from `tiering/manager.py` and its three call sites; and
change the fs tier's `_tr_emit_tier_report()` loop back from
`TierReportMetrics.LOAD_THREAD_SECONDS` / `STORE_THREAD_SECONDS` to
`LOAD_SECONDS` / `STORE_SECONDS`. The live engine is read-only; revert by
restoring the pre-patch files and re-loading the model.

 RUNS AFTER: patch_kv_offload_tier_report.py (it adds TierReportMetrics and the
 fs tier's _tr_emit_tier_report that this patch re-points).
 BUNDLE NOTE (task 20): this is the re-anchored copy. Its hunk 2 now runs AFTER
 patch_promotion_refusal_instrumentation.py (the CONN_METRICS definitions dict
 tail moves from the STALL_SECONDS entry to the PROMOTION_INITIATED entry).
 Apply order for the bundle: house 1-8 (..tier_report) -> 00-instrumentation
 -> 00-fix -> this file. See apply-bundle.py.
 """

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KVO = SP / "vllm/v1/kv_offload"
KVO_BASE = KVO / "base.py"
TIERING_MANAGER = KVO / "tiering/manager.py"
FS_MANAGER = KVO / "tiering/fs/manager.py"
CONN_METRICS = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"

print("radiance: KV offload tier-report -- wall-clock whole-job timing")

# --- prerequisite -----------------------------------------------------------
# This patch re-points the fs tier's seconds accumulator and adds whole-job
# timing to the tiering manager. Both sit on top of the tier-report patch: it
# adds the TierReportMetrics name table, the fs tier's _tr_emit_tier_report(),
# and the manager's _tr_tokens_per_hash slot. A missing marker means the base
# patch is not applied, which is a different failure than a version drift.
if "class TierReportMetrics:" not in KVO_BASE.read_text():
    raise SystemExit(
        "[radiance] wall-clock timing requires "
        "patch_kv_offload_tier_report.py to be applied first "
        "(it creates TierReportMetrics and the fs tier's _tr_emit_tier_report)."
    )
if "self._tr_tokens_per_hash" not in TIERING_MANAGER.read_text():
    raise SystemExit(
        "[radiance] wall-clock timing requires "
        "patch_kv_offload_tier_report.py to be applied first "
        "(it creates TieringOffloadingManager._tr_tokens_per_hash)."
    )
# Bundle prerequisite: this copy's hunk 2 anchors on the CONN_METRICS
# definitions dict tail that patch_promotion_refusal_instrumentation.py leaves
# there (the PROMOTION_INITIATED entry). A missing marker means the
# instrumentation patch did not run, which is a wrong order, not a version
# drift -- say so explicitly rather than failing the anchor opaquely.
if "PROMOTION_INITIATED: OffloadingCounterMetadata(" not in CONN_METRICS.read_text():
    raise SystemExit(
        "[radiance] this re-anchored wall-clock patch requires "
        "patch_promotion_refusal_instrumentation.py to be applied first "
        "(it adds TieringOffloadingMetrics.PROMOTION_INITIATED to the "
        "definitions dict this hunk anchors on)."
    )

# ---------------------------------------------------------------------------
# 1. The two new counter names, in the leaf module everyone already imports.
#    The wall-clock fix reuses the existing LOAD_SECONDS / STORE_SECONDS names;
#    these are the preserved per-batch (thread-summed) signal.
# ---------------------------------------------------------------------------
apply(
    KVO_BASE,
    '''    LOAD_BYTES = "vllm:kv_offload_tier_load_bytes"
    LOAD_SECONDS = "vllm:kv_offload_tier_load_seconds"
    LOAD_OPS = "vllm:kv_offload_tier_load_ops"
    LOAD_LATENCY = "vllm:kv_offload_tier_load_latency_seconds"
    STORE_BYTES = "vllm:kv_offload_tier_store_bytes"
    STORE_SECONDS = "vllm:kv_offload_tier_store_seconds"
    STORE_OPS = "vllm:kv_offload_tier_store_ops"
    STORE_LATENCY = "vllm:kv_offload_tier_store_latency_seconds"''',
    '''    LOAD_BYTES = "vllm:kv_offload_tier_load_bytes"
    # Wall-clock whole-job time: timed once per transfer, from submit to the
    # last task completing, at the tiering manager (which is the only place
    # that sees both ends). On a pooled tier this is the honest denominator
    # for a rate; it used to be the thread-summed per-batch total, which is
    # now under LOAD_THREAD_SECONDS.
    LOAD_SECONDS = "vllm:kv_offload_tier_load_seconds"
    LOAD_OPS = "vllm:kv_offload_tier_load_ops"
    # Per-batch load time, thread-summed over the tier's worker pool. NOT wall
    # time on a pooled tier: N concurrent batches each taking 1 s sum to N s.
    # Kept to gauge how busy the pool is (thread-seconds / wall-seconds ~=
    # concurrency), not for a rate.
    LOAD_THREAD_SECONDS = "vllm:kv_offload_tier_load_thread_seconds"
    # Per-batch latency histogram; a batch's own wall time, measured on the one
    # pool thread that runs it, is a real quantity. Unchanged.
    LOAD_LATENCY = "vllm:kv_offload_tier_load_latency_seconds"
    STORE_BYTES = "vllm:kv_offload_tier_store_bytes"
    STORE_SECONDS = "vllm:kv_offload_tier_store_seconds"
    STORE_OPS = "vllm:kv_offload_tier_store_ops"
    STORE_THREAD_SECONDS = "vllm:kv_offload_tier_store_thread_seconds"
    STORE_LATENCY = "vllm:kv_offload_tier_store_latency_seconds"''',
    "LOAD_THREAD_SECONDS = \"vllm:kv_offload_tier_load_thread_seconds\"",
    "1  kv_offload/base.py: thread-summed seconds names",
)

# ---------------------------------------------------------------------------
# 2. Register the two new definitions at the CONNECTOR level, so the assertion
#    in OffloadPromMetrics.observe() cannot be tripped by emitting them.
# ---------------------------------------------------------------------------
apply(
    CONN_METRICS,
    # Re-anchored for the bundle: the definitions dict's tail is no longer the
    # STALL_SECONDS entry (patch_promotion_refusal_instrumentation.py, hunk 3b,
    # inserts four PROMOTION_* entries after it before the closing brace). We
    # anchor on the PROMOTION_INITIATED entry + the closing brace -- the tail
    # AFTER that patch runs -- and insert the two thread-seconds defs at the
    # very end of the dict. This makes the wall-clock patch order-dependent on
    # the instrumentation patch (the wrapper enforces that order); running the
    # wall-clock patch without it would fail the anchor (count == 0), which is
    # the safe failure mode (a named error, not a mis-ordered file).
    '''        TieringOffloadingMetrics.PROMOTION_INITIATED: OffloadingCounterMetadata(
            documentation=(
                "Promotions that landed. Refusal rate = refused / "
                "(refused + initiated); shares the source-tier label."
            ),
            labelnames=("tier",),
        ),
    }''',
    '''        TieringOffloadingMetrics.PROMOTION_INITIATED: OffloadingCounterMetadata(
            documentation=(
                "Promotions that landed. Refusal rate = refused / "
                "(refused + initiated); shares the source-tier label."
            ),
            labelnames=("tier",),
        ),
        TierReportMetrics.LOAD_THREAD_SECONDS: OffloadingCounterMetadata(
            documentation=(
                "Per-batch load time, thread-summed over the tier's worker "
                "pool. NOT wall time on a pooled tier: N concurrent batches "
                "each taking 1 s sum to N s. Use it to gauge how busy the "
                "pool is (thread-seconds / wall-seconds ~= concurrency), not "
                "for a rate. Wall-clock whole-job load time is "
                "kv_offload_tier_load_seconds."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.STORE_THREAD_SECONDS: OffloadingCounterMetadata(
            documentation=(
                "Per-batch store time, thread-summed over the tier's worker "
                "pool. NOT wall time on a pooled tier (same caveat as "
                "kv_offload_tier_load_thread_seconds). Wall-clock whole-job "
                "store time is kv_offload_tier_store_seconds."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
    }''',
    "TierReportMetrics.LOAD_THREAD_SECONDS: OffloadingCounterMetadata(",
    "2  offloading/metrics.py: thread-summed seconds definitions",
)

# ---------------------------------------------------------------------------
# 3. The tiering manager: the only place that sees both the submit and the
#    completion of a secondary-tier transfer, so the whole job is timed once,
#    in wall clock, there.
# ---------------------------------------------------------------------------
apply(
    TIERING_MANAGER,
    '''        self._tr_tokens_per_hash: int = 0''',
    '''        self._tr_tokens_per_hash: int = 0

        # radiance wall-clock: submit time of each in-flight secondary-tier
        # transfer job, so the whole job is timed once, from submit to the
        # last task completing. Keyed by job id, one entry per job, drained at
        # completion. Scheduler-thread only (both submit and completion run
        # there), so no lock. Bounded like the other side maps; a job
        # abandoned by reset is dropped FIFO rather than leaking.
        self._tr_job_wall: dict[JobId, tuple[float, bool]] = {}
        self._tr_job_wall_cap = 4096''',
    "self._tr_job_wall_cap = 4096",
    "3a tiering/manager.py: per-job wall-clock submit map",
)

apply(
    TIERING_MANAGER,
    '''    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id''',
    '''    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _tr_start_job(self, job_metadata: JobMetadata) -> None:
        """radiance wall-clock: start the whole-job wall timer.

        A secondary-tier transfer (a promotion, or a cascade store) is timed
        once, end to end, in wall time -- from the moment the batch is
        submitted to the moment its last task completes -- instead of per I/O
        batch. On a pooled tier several batches run at once, so summing their
        per-batch wall times overstates the wall time by up to the pool width
        and understates every rate derived from it. This records the submit
        time; _process_finished_jobs() closes the span when the job completes.

        Scheduler-thread only. A job abandoned by reset is dropped FIFO.
        """
        if len(self._tr_job_wall) >= self._tr_job_wall_cap:
            self._tr_job_wall.pop(next(iter(self._tr_job_wall)), None)
        self._tr_job_wall[job_metadata.job_id] = (
            time.monotonic(), job_metadata.is_promotion
        )''',
    "def _tr_start_job",
    "3b tiering/manager.py: _tr_start_job helper",
)

apply(
    TIERING_MANAGER,
    '''                self._transfer_jobs[job_id] = job_metadata
                # radiance tier-report (P1): a promotion is this tier's hit.
                self._tr_count_hit(tier.tier_type, entry.keys)
                tier.submit_load(job_metadata)''',
    '''                self._transfer_jobs[job_id] = job_metadata
                # radiance tier-report (P1): a promotion is this tier's hit.
                self._tr_count_hit(tier.tier_type, entry.keys)
                tier.submit_load(job_metadata)
                # radiance wall-clock: start the whole-job timer (promotion).
                self._tr_start_job(job_metadata)''',
    "# radiance wall-clock: start the whole-job timer (promotion).",
    "3c tiering/manager.py: start timer on submit_load (promotion)",
)

apply(
    TIERING_MANAGER,
    '''            for tier in self.secondary_tiers:
                job_metadata = self.create_store_job(keys, req_context)
                tier.submit_store(job_metadata)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().''',
    '''            for tier in self.secondary_tiers:
                job_metadata = self.create_store_job(keys, req_context)
                tier.submit_store(job_metadata)
                # radiance wall-clock: start the whole-job timer (cascade).
                self._tr_start_job(job_metadata)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().''',
    "# radiance wall-clock: start the whole-job timer (cascade).",
    "3d tiering/manager.py: start timer on complete_store (cascade)",
)

apply(
    TIERING_MANAGER,
    '''        for tier in request_level_tiers:
            job_metadata = self.create_store_job(ready_keys, req_context)
            tier.submit_store(job_metadata)''',
    '''        for tier in request_level_tiers:
            job_metadata = self.create_store_job(ready_keys, req_context)
            tier.submit_store(job_metadata)
            # radiance wall-clock: start the whole-job timer (request-level).
            self._tr_start_job(job_metadata)''',
    "# radiance wall-clock: start the whole-job timer (request-level).",
    "3e tiering/manager.py: start timer on request-level store",
)

apply(
    TIERING_MANAGER,
    '''                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                if job_metadata.is_promotion:''',
    '''                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                # radiance wall-clock: close the span. The submit time was
                # recorded when this job was submitted (a promotion, or a
                # cascade store); the moment it now appears in
                # get_finished_jobs() is the moment its last task completed --
                # the pool's JobState only reports a job finished once every
                # task has called task_done(). The difference is the whole
                # transfer's wall time, timed once, not the sum of the per-
                # batch times a worker pool accumulates in parallel.
                _tr_started = self._tr_job_wall.pop(job_id, None)
                if _tr_started is not None:
                    _tr_submit, _tr_is_promotion = _tr_started
                    _tr_wall = time.monotonic() - _tr_submit
                    _tr_secs_name = (
                        TierReportMetrics.LOAD_SECONDS
                        if _tr_is_promotion
                        else TierReportMetrics.STORE_SECONDS
                    )
                    self._stats.increase_counter(
                        _tr_secs_name, _tr_wall, (tier.tier_type,)
                    )

                if job_metadata.is_promotion:''',
    "# radiance wall-clock: close the span.",
    "3f tiering/manager.py: close the whole-job span on completion",
)

# ---------------------------------------------------------------------------
# 4. The fs tier: re-point the per-batch accumulator's seconds onto the NEW
#    thread-summed name, freeing LOAD_SECONDS / STORE_SECONDS for the manager's
#    wall-clock emission. The byte, per-batch-op, and per-batch-latency
#    emitters are untouched.
# ---------------------------------------------------------------------------
apply(
    FS_MANAGER,
    '''            (
                "load",
                TierReportMetrics.LOAD_BYTES,
                TierReportMetrics.LOAD_SECONDS,
                TierReportMetrics.LOAD_OPS,
                TierReportMetrics.LOAD_LATENCY,
            ),
            (
                "store",
                TierReportMetrics.STORE_BYTES,
                TierReportMetrics.STORE_SECONDS,
                TierReportMetrics.STORE_OPS,
                TierReportMetrics.STORE_LATENCY,
            ),''',
    '''            (
                "load",
                TierReportMetrics.LOAD_BYTES,
                # The per-batch wall time, summed over the pool, is thread-
                # time not wall time; it moves to the thread-summed name.
                # Wall-clock whole-job load time is LOAD_SECONDS, emitted by
                # the tiering manager.
                TierReportMetrics.LOAD_THREAD_SECONDS,
                TierReportMetrics.LOAD_OPS,
                TierReportMetrics.LOAD_LATENCY,
            ),
            (
                "store",
                TierReportMetrics.STORE_BYTES,
                TierReportMetrics.STORE_THREAD_SECONDS,
                TierReportMetrics.STORE_OPS,
                TierReportMetrics.STORE_LATENCY,
            ),''',
    "TierReportMetrics.LOAD_THREAD_SECONDS,",
    "4  fs/manager.py: per-batch seconds move to the thread-summed name",
)

print("[radiance] promotion-refusal + wallclock bundle applied (promotion first)")
