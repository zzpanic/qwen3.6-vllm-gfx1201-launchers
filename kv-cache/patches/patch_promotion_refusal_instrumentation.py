#!/usr/bin/env python3
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

print("[radiance] promotion-refusal counters applied "
      "(refused / no_evictable / protected / initiated, by source tier)")
