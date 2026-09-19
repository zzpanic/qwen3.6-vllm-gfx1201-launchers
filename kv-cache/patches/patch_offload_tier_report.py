#!/usr/bin/env python3
"""Per-tier, lifetime-cumulative instrumentation for the KV offload tiers.

WHY
===
An operator running a layered KV cache has three questions and today the serving
stack cannot answer any of them:

  1. Have I allocated too much (or too little) RAM to the CPU tier?
  2. Is my disk too slow -- should I buy NVMe?
  3. Is turning this layered cache on adding value at all?

The reason it cannot answer them is not that the numbers are missing; it is that
the numbers that exist are the wrong SHAPE:

  * `vllm:kv_offload_load_bytes_total` and `_load_time_total` carry NO LABELS.
    On this box they read 11.8 GB/s aggregate, which is CPU-tier speed. The
    disk's contribution is buried inside that average, so a slow disk is
    undetectable from the aggregate. Every tier question needs a `tier` label.

  * `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}` is exact
    and trustworthy -- it sums to `prompt_tokens_total` -- but it puts BOTH
    offload tiers in one bucket. It cannot say whether the RAM tier or the disk
    tier did the work, which is the column the entire sizing argument rests on.

  * Everything else the tiers expose is a GAUGE: `cpu_cache_free_perc`,
    `cpu_cache_evictable_perc`, `fs_inflight_jobs`, `kv_cache_usage_perc`. A
    gauge is an instantaneous sample. Scraped once from a server that has been
    up for three weeks it carries almost no information: "the RAM tier is full
    right now" is not a sizing statement, "the RAM tier was above 95% full for
    80% of the time" is. So this patch adds COUNTERS AND HISTOGRAMS ONLY. The
    two exceptions are tier capacity and used bytes, which are configuration
    rather than judgment.

That last point is the design constraint the whole patch list follows from. The
report this feeds (`tierreport.py`) is meant to be run ONCE, against a server
that has been serving the operator's OWN traffic for as long as it has been up.
A replay benchmark cannot answer "is my RAM the right size", because the answer
depends entirely on the operator's prefix-reuse pattern, not on a synthetic
corpus. So every number the report depends on has to survive being read exactly
once, long after the fact -- which means monotonic counters and distributions,
never levels.

WHAT IT ADDS
============
All names are `vllm:kv_offload_tier_*` and all carry a single `tier` label whose
value is `"cpu"` for the primary tier and the configured `tier_type` (e.g.
`"fs"`) for each secondary tier.

  P1 -- attribution
    tier_hit_blocks_total{tier}          distinct block hashes served by a tier
    tier_hit_tokens_total{tier}          the same, in tokens

  P2 -- speed and volume
    tier_load_bytes_total{tier}          bytes moved toward the GPU
    tier_load_seconds_total{tier}        time spent moving them
    tier_load_ops_total{tier}            number of transfer batches
    tier_load_latency_seconds{tier}      HISTOGRAM of per-batch latency
    tier_store_bytes_total{tier}         ... and the same six for stores
    tier_store_seconds_total{tier}
    tier_store_ops_total{tier}
    tier_store_latency_seconds{tier}

  P3 -- size
    tier_capacity_bytes{tier}            gauge, configuration
    tier_used_bytes{tier}                gauge, configuration
    tier_occupancy_ratio{tier}           HISTOGRAM of occupancy over time

  P4/P6 -- the sizing verdict
    tier_reads_before_evict{tier}        HISTOGRAM, buckets 0,1,2,4,8,16,32,64
    tier_evictions_total{tier}
    tier_evicted_bytes_total{tier}
    tier_eviction_to_reuse_seconds{tier} HISTOGRAM
    tier_lookup_miss_evicted_total{tier}

  P5 -- the downside
    tier_stall_seconds_total{tier}       time a request waited on this tier

Why those two histograms in particular are the point of the exercise:

  * `reads_before_evict` with its mass at 0 means you are storing blocks nobody
    ever reads back. The tier is OVERSIZED (or the store policy is too eager).
  * `eviction_to_reuse_seconds` with its mass at the low end means you evict
    blocks that are asked for again seconds later. The tier is UNDERSIZED. That
    is cache thrashing, measured rather than inferred.
  * Both at once means the tier is the wrong SHAPE, not the wrong size -- it is
    churning on cold data while hot data gets evicted, which is an eviction
    policy problem and no amount of RAM will fix it.

No occupancy number, however finely sampled, can distinguish those three cases.

TWO ATTRIBUTION SUBTLETIES, ENCODED HERE ON PURPOSE
===================================================
1. DISTINCT HASHES, NOT KEYS. An `OffloadKey` is (block_hash, kv_group_idx), so
   one logical block is looked up once per KV cache group. Counting keys would
   multiply token counts by the group count -- and the multiplier is NOT uniform
   (in this build the Mamba groups store 1-in-8, the attention groups 1-in-1),
   so it cannot be divided back out afterwards. Every token figure here counts
   `len({block_hash for key in keys})` and multiplies by `tokens_per_hash`.

2. THE PROMOTING-TIER TRAP. This is a staged design: a block that lives on disk
   is promoted fs -> CPU and then loaded CPU -> GPU. So the CPU tier's numbers
   INCLUDE work that originated on disk. If the report just read
   `tier_hit_tokens{cpu}` it would credit the disk's contribution to RAM and
   conclude the disk tier is dead weight. This patch therefore counts the CPU
   tier at `prepare_load()` (everything handed to the GPU, promotions included)
   and each secondary tier at promotion; the report SUBTRACTS to get
   cpu-originated volume. Both numbers are emitted; neither is a lie on its own,
   but only the pair is usable. `tierreport.py` does the subtraction.

RISK
====
Instrumentation only. No scheduling, eviction, transfer or correctness path
changes behaviour; every hunk either adds a counter update or adds a field.

The three places that carry real (small) risk, and why they are acceptable:

  * `DirectionalTransferStats` gains a `times: list[float]` field. This object
    crosses the worker -> scheduler IPC boundary. It is a plain dataclass whose
    sibling field `sizes: list[int | float]` already crosses the same boundary
    as a list, so the new field is structurally identical to one that is known
    to serialize. Without it there is no per-transfer latency for the CPU tier,
    only an aggregate mean -- and the mean is exactly what hides the tail that
    hurts (a p50 of 40 ms with a p99 of 9 s reads as "fine" in the mean).

  * `CPUOffloadingManager` gains two side maps keyed by `OffloadKey`: read
    counts and an eviction ghost list. Both are bounded (`num_blocks * 4`
    entries, FIFO-trimmed), which is a few hundred KB at production tier sizes.
    `BlockStatus` is a `ctypes.Structure` with fixed fields, so the read count
    could not be stored on the block itself.

  * The fs tier wraps its two I/O entry points in a timing closure. The wrapper
    is `try/finally` around the original call and records into a lock-guarded
    accumulator, so a failing I/O is still timed and an exception still
    propagates unchanged.

Every emission site is guarded so that a rename or a refactor upstream degrades
to "no metric" rather than crashing the metrics path, in the same style as
`patch_kv_offload_instrumentation.py`.

Metric definitions are registered in `get_connector_metric_definitions()` rather
than on a spec class, so they exist for every offloading spec. `OffloadPromMetrics.observe()`
asserts that every emitted name is registered; registering at the connector level
means that assertion cannot be tripped by running a spec we did not think of.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KVO = SP / "vllm/v1/kv_offload"
KVO_BASE = KVO / "base.py"
CPU_MANAGER = KVO / "cpu/manager.py"
TIERING_MANAGER = KVO / "tiering/manager.py"
TIERING_SPEC = KVO / "tiering/spec.py"
FS_MANAGER = KVO / "tiering/fs/manager.py"
CONN = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading"
CONN_METRICS = CONN / "metrics.py"
CONN_COMMON = CONN / "common.py"
CONN_SCHEDULER = CONN / "scheduler.py"

print("radiance: KV offload tier-report instrumentation")

# --- prerequisite -----------------------------------------------------------
# The filesystem tier has no get_stats() of its own in stock vLLM; the backlog
# gauge added by patch_kv_offload_instrumentation.py is what creates it, and
# hunk 8d extends that method rather than competing with it. So this patch must
# run AFTER the instrumentation patch. _patchlib would already fail loudly on
# the missing anchor, but a missing anchor reads like a version drift, which is
# the wrong thing to go looking for -- say what is actually wrong instead.
if "FS_INFLIGHT_JOBS" not in FS_MANAGER.read_text():
    raise SystemExit(
        "[radiance] tier-report requires patch_kv_offload_instrumentation.py "
        "to be applied first (it creates FileSystemTierManager.get_stats)."
    )

# ---------------------------------------------------------------------------
# 1. The shared metric-name table, in the leaf module everyone already imports.
# ---------------------------------------------------------------------------
apply(
    KVO_BASE,
    """@dataclass(frozen=True)
class OffloadingMetricMetadata:
    documentation: str
    labelnames: tuple[str, ...] = ()""",
    '''class TierReportMetrics:
    """radiance: per-tier, lifetime-cumulative KV offload instrumentation.

    One label, `tier`, whose value is "cpu" for the primary tier and the
    configured tier_type for each secondary tier. Counters and histograms only:
    these are read ONCE, from a long-lived server, against the operator's own
    traffic -- a gauge scraped that way carries no information. Capacity and
    used bytes are the exception: they are configuration, not judgment.
    """

    TIER = ("tier",)
    # The label value for the primary tier. Secondary tiers label themselves
    # with their configured tier_type ("fs", "p2p", ...), so nothing else here
    # needs to know the tier list.
    CPU = "cpu"

    # P1: which tier actually served the block.
    HIT_BLOCKS = "vllm:kv_offload_tier_hit_blocks"
    HIT_TOKENS = "vllm:kv_offload_tier_hit_tokens"

    # P2: volume, time and per-batch latency, per tier and per direction.
    LOAD_BYTES = "vllm:kv_offload_tier_load_bytes"
    LOAD_SECONDS = "vllm:kv_offload_tier_load_seconds"
    LOAD_OPS = "vllm:kv_offload_tier_load_ops"
    LOAD_LATENCY = "vllm:kv_offload_tier_load_latency_seconds"
    STORE_BYTES = "vllm:kv_offload_tier_store_bytes"
    STORE_SECONDS = "vllm:kv_offload_tier_store_seconds"
    STORE_OPS = "vllm:kv_offload_tier_store_ops"
    STORE_LATENCY = "vllm:kv_offload_tier_store_latency_seconds"

    # P3: size. Occupancy is a distribution over time, not a level.
    CAPACITY_BYTES = "vllm:kv_offload_tier_capacity_bytes"
    USED_BYTES = "vllm:kv_offload_tier_used_bytes"
    OCCUPANCY_RATIO = "vllm:kv_offload_tier_occupancy_ratio"

    # P4/P6: the sizing verdict.
    READS_BEFORE_EVICT = "vllm:kv_offload_tier_reads_before_evict"
    EVICTIONS = "vllm:kv_offload_tier_evictions"
    EVICTED_BYTES = "vllm:kv_offload_tier_evicted_bytes"
    EVICTION_TO_REUSE = "vllm:kv_offload_tier_eviction_to_reuse_seconds"
    MISS_EVICTED = "vllm:kv_offload_tier_lookup_miss_evicted"

    # P5: what the tier costs when it is slow.
    STALL_SECONDS = "vllm:kv_offload_tier_stall_seconds"

    # Sub-millisecond (CPU memcpy) through minutes (a saturated spinning disk).
    LATENCY_BUCKETS = (
        0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,
        0.1, 0.5, 1, 5, 10, 30, 60, 300,
    )
    # Integer read counts; edges at .5 so a block read exactly N times lands in
    # the bucket labelled N and "never read back" is exactly the le="0.5" bin.
    READS_BUCKETS = (0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5)
    # One second to one day: how long after eviction the block was wanted again.
    REUSE_BUCKETS = (1, 5, 15, 60, 300, 900, 3600, 14400, 86400)
    OCCUPANCY_BUCKETS = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0)


@dataclass(frozen=True)
class OffloadingMetricMetadata:
    documentation: str
    labelnames: tuple[str, ...] = ()''',
    "class TierReportMetrics:",
    "1  kv_offload/base.py: TierReportMetrics name table",
)

# ---------------------------------------------------------------------------
# 2. Register the definitions at the CONNECTOR level, so they exist for every
#    offloading spec. OffloadPromMetrics.observe() asserts that every emitted
#    name is registered; registering here means that assertion cannot be
#    tripped by a spec we did not anticipate.
# ---------------------------------------------------------------------------
apply(
    CONN_METRICS,
    """from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
)""",
    """from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    TierReportMetrics,
)""",
    "    TierReportMetrics,\n)",
    "2a offloading/metrics.py: import TierReportMetrics",
)

apply(
    CONN_METRICS,
    """        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),
    }""",
    '''        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),
        # radiance tier-report: everything below carries a `tier` label so one
        # scrape describes every tier separately. See patch_kv_offload_tier_report.py.
        TierReportMetrics.HIT_BLOCKS: OffloadingCounterMetadata(
            documentation=(
                "Distinct KV blocks served by this tier. The CPU tier is counted "
                "at the GPU-facing load, so it INCLUDES blocks a secondary tier "
                "promoted into it; subtract the secondary tiers to get the "
                "volume that originated in CPU."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.HIT_TOKENS: OffloadingCounterMetadata(
            documentation=(
                "Tokens served by this tier (distinct blocks x tokens per hash). "
                "Same promoting-tier caveat as kv_offload_tier_hit_blocks."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.LOAD_BYTES: OffloadingCounterMetadata(
            documentation="Bytes this tier moved toward the GPU.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.LOAD_SECONDS: OffloadingCounterMetadata(
            documentation="Seconds this tier spent moving bytes toward the GPU.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.LOAD_OPS: OffloadingCounterMetadata(
            documentation="Load transfer batches issued against this tier.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.LOAD_LATENCY: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of per-batch load latency for this tier, in seconds. "
                "A histogram rather than a mean because the mean hides the tail "
                "that actually costs time to fetch."
            ),
            buckets=TierReportMetrics.LATENCY_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.STORE_BYTES: OffloadingCounterMetadata(
            documentation="Bytes written into this tier.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.STORE_SECONDS: OffloadingCounterMetadata(
            documentation=(
                "Seconds spent writing into this tier. This is the tier's "
                "overhead term: it is paid on every store whether or not the "
                "block is ever read back."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.STORE_OPS: OffloadingCounterMetadata(
            documentation="Store transfer batches issued against this tier.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.STORE_LATENCY: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of per-batch store latency for this tier, in seconds."
            ),
            buckets=TierReportMetrics.LATENCY_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.CAPACITY_BYTES: OffloadingGaugeMetadata(
            documentation=(
                "Configured capacity of this tier, in bytes. For a filesystem "
                "tier this is the size of the filesystem holding the block "
                "store, since that is what actually bounds it."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.USED_BYTES: OffloadingGaugeMetadata(
            documentation=(
                "Bytes currently held by this tier. For a filesystem tier this "
                "is the filesystem's used bytes, which may include data that is "
                "not KV blocks -- it is the number that decides whether the "
                "tier runs out of room."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.OCCUPANCY_RATIO: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of this tier's occupancy, sampled once per metrics "
                "interval. The DISTRIBUTION is the sizing signal -- 'above 95% "
                "full for 80% of the time' is a statement about capacity, "
                "'95% full right now' is not."
            ),
            buckets=TierReportMetrics.OCCUPANCY_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.READS_BEFORE_EVICT: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of how many times a block was read back before this "
                "tier evicted it. Mass in the le=0.5 bucket means blocks are "
                "being stored that nobody ever reads: the tier is oversized, or "
                "the store policy is too eager."
            ),
            buckets=TierReportMetrics.READS_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.EVICTIONS: OffloadingCounterMetadata(
            documentation="Blocks evicted from this tier.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.EVICTED_BYTES: OffloadingCounterMetadata(
            documentation="Bytes evicted from this tier.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.EVICTION_TO_REUSE: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of the time between this tier evicting a block and a "
                "later lookup asking for that same block, in seconds. Mass at "
                "the low end is cache thrashing measured rather than inferred: "
                "the tier is undersized."
            ),
            buckets=TierReportMetrics.REUSE_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.MISS_EVICTED: OffloadingCounterMetadata(
            documentation=(
                "Lookups that missed on a block this tier previously held and "
                "evicted. Distinguishes 'never cached' from 'cached and thrown "
                "away too early'; only the second justifies buying capacity."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.STALL_SECONDS: OffloadingCounterMetadata(
            documentation=(
                "Seconds requests spent waiting on this tier before being "
                "allocated. This is what a slow device costs, and it is the "
                "number that says a tier is not merely failing to help but "
                "actively hurting."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
    }''',
    "TierReportMetrics.HIT_BLOCKS: OffloadingCounterMetadata(",
    "2b offloading/metrics.py: tier-report metric definitions",
)

# ---------------------------------------------------------------------------
# 3. The CPU primary tier: read counts, evictions, the ghost list, occupancy.
#    This is where P4 and P6 live, and P4 is the reason the report is worth
#    running at all -- it turns a storage purchase from an argument into a
#    measurement.
# ---------------------------------------------------------------------------
apply(
    CPU_MANAGER,
    """from collections import OrderedDict
from collections.abc import Collection, Iterable""",
    """import time
from collections import OrderedDict
from collections.abc import Collection, Iterable""",
    "import time\nfrom collections import OrderedDict",
    "3a cpu/manager.py: import time",
)

apply(
    CPU_MANAGER,
    """    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (""",
    """    RequestOffloadingContext,
    TierReportMetrics,
)
from vllm.v1.kv_offload.cpu.common import (""",
    "    TierReportMetrics,\n)\nfrom vllm.v1.kv_offload.cpu.common import (",
    "3b cpu/manager.py: import TierReportMetrics",
)

apply(
    CPU_MANAGER,
    """        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )""",
    '''        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

        # radiance tier-report: state for the sizing verdict. BlockStatus is a
        # ctypes.Structure with fixed fields, so the per-block read count cannot
        # live on the block and needs a side map. Both maps are FIFO-bounded.
        self._tr_tier_name: str = "cpu"
        # Bytes per tier block; set by the primary-tier subclass, which is the
        # only place the mmap layout is known. 0 disables the byte counters.
        self._tr_block_bytes: int = 0
        self._tr_cap: int = max(1024, num_blocks * 4)
        # key -> number of prepare_load() reads since it was stored.
        self._tr_reads: OrderedDict[OffloadKey, int] = OrderedDict()
        # key -> time.monotonic() at eviction. A lookup that hits here is a
        # would-have-hit: the block WAS cached and was thrown away too early.
        self._tr_ghost: OrderedDict[OffloadKey, float] = OrderedDict()
        self._tr_reads_before_evict: list[int] = []
        self._tr_reuse_delays: list[float] = []
        self._tr_evictions: int = 0
        self._tr_miss_evicted: int = 0''',
    "self._tr_reads_before_evict",
    "3c cpu/manager.py: tier-report state",
)

apply(
    CPU_MANAGER,
    """        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT""",
    """        block = self._policy.get(key)
        if block is None:
            # radiance tier-report (P4/P6): was this block ever here? A miss on
            # a block we evicted is the only kind of miss that argues for buying
            # capacity; a miss on a block we never had does not.
            evicted_at = self._tr_ghost.pop(key, None)
            if evicted_at is not None:
                self._tr_miss_evicted += 1
                self._tr_reuse_delays.append(time.monotonic() - evicted_at)
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT""",
    "radiance tier-report (P4/P6): was this block ever here?",
    "3d cpu/manager.py: would-have-hit on lookup miss",
)

apply(
    CPU_MANAGER,
    """            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)""",
    """            block.ref_cnt += 1
            blocks.append(block)
            # radiance tier-report (P4): a prepare_load IS the read-back.
            self._tr_reads[key] = self._tr_reads.get(key, 0) + 1
            self._tr_reads.move_to_end(key)
            if len(self._tr_reads) > self._tr_cap:
                self._tr_reads.popitem(last=False)
        return self._get_load_store_spec(keys, blocks)""",
    "radiance tier-report (P4): a prepare_load IS the read-back.",
    "3e cpu/manager.py: count read-backs",
)

apply(
    CPU_MANAGER,
    """            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)""",
    """            _tr_now = time.monotonic()  # radiance tier-report
            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)
                # radiance tier-report (P4): how many times was it read back
                # before it died, and when did it die? Those two distributions
                # are what tell an operator whether the tier is too big, too
                # small, or the wrong shape.
                self._tr_reads_before_evict.append(self._tr_reads.pop(key, 0))
                self._tr_evictions += 1
                self._tr_ghost[key] = _tr_now
                if len(self._tr_ghost) > self._tr_cap:
                    self._tr_ghost.popitem(last=False)""",
    "radiance tier-report (P4): how many times was it read back",
    "3f cpu/manager.py: eviction accounting",
)

apply(
    CPU_MANAGER,
    """        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0""",
    """        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0

        # radiance tier-report: a reset is not an eviction. Dropping the ghost
        # list here stops a post-reset miss being reported as a would-have-hit.
        self._tr_reads.clear()
        self._tr_ghost.clear()""",
    "radiance tier-report: a reset is not an eviction.",
    "3g cpu/manager.py: clear tier-report state on reset",
)

apply(
    CPU_MANAGER,
    """        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats""",
    '''        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        self._tr_emit_tier_report(stats)

        return stats

    # ---- radiance tier-report -------------------------------------------
    def _tr_emit_tier_report(self, stats: OffloadingConnectorStats) -> None:
        """Flush this tier's cumulative sizing counters onto the stats payload.

        Everything here is a counter or a histogram, because the report that
        reads it scrapes ONCE from a server that may have been up for weeks. The
        two gauges (capacity, used) are configuration rather than judgment.

        Guarded on the side maps: a subclass that bypasses __init__ degrades to
        "no tier metrics" instead of crashing the metrics path.
        """
        if getattr(self, "_tr_reads", None) is None:
            return
        label = (self._tr_tier_name,)
        block_bytes = self._tr_block_bytes

        if self._num_blocks > 0:
            resident = self._num_allocated_blocks - len(self._free_list)
            # Occupancy as a distribution over time: one observation per
            # metrics interval. Reading the level once tells you nothing.
            stats.observe_histogram(
                TierReportMetrics.OCCUPANCY_RATIO,
                resident / self._num_blocks,
                label,
            )
            if block_bytes:
                stats.set_gauge(
                    TierReportMetrics.CAPACITY_BYTES,
                    self._num_blocks * block_bytes,
                    label,
                )
                stats.set_gauge(
                    TierReportMetrics.USED_BYTES, resident * block_bytes, label
                )

        for num_reads in self._tr_reads_before_evict:
            stats.observe_histogram(
                TierReportMetrics.READS_BEFORE_EVICT, num_reads, label
            )
        self._tr_reads_before_evict.clear()

        for delay in self._tr_reuse_delays:
            stats.observe_histogram(
                TierReportMetrics.EVICTION_TO_REUSE, delay, label
            )
        self._tr_reuse_delays.clear()

        if self._tr_evictions:
            stats.increase_counter(
                TierReportMetrics.EVICTIONS, self._tr_evictions, label
            )
            if block_bytes:
                stats.increase_counter(
                    TierReportMetrics.EVICTED_BYTES,
                    self._tr_evictions * block_bytes,
                    label,
                )
            self._tr_evictions = 0

        if self._tr_miss_evicted:
            stats.increase_counter(
                TierReportMetrics.MISS_EVICTED, self._tr_miss_evicted, label
            )
            self._tr_miss_evicted = 0''',
    "def _tr_emit_tier_report",
    "3h cpu/manager.py: emit tier-report counters",
)

# ---------------------------------------------------------------------------
# 4. The tiering manager: tier attribution (P1) and per-tier stall time (P5).
# ---------------------------------------------------------------------------
apply(
    TIERING_MANAGER,
    """    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec""",
    """    ScheduleEndContext,
    TierReportMetrics,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec""",
    "    TierReportMetrics,\n    get_offload_block_hash,\n)",
    "4a tiering/manager.py: imports",
)

apply(
    TIERING_MANAGER,
    """    # time.monotonic() of this request's first deferred secondary-tier lookup;
    # None once consumed (observed) or while no secondary lookup is pending.
    secondary_lookup_start_time: float | None = None""",
    """    # time.monotonic() of this request's first deferred secondary-tier lookup;
    # None once consumed (observed) or while no secondary lookup is pending.
    secondary_lookup_start_time: float | None = None
    # radiance tier-report (P5): which tier that deferral is waiting on, so the
    # wait can be charged to the device that caused it. None when the deferral
    # came from a RETRY rather than a promotion, in which case no tier is
    # identifiable and no stall time is attributed.
    secondary_lookup_tier: str | None = None""",
    "secondary_lookup_tier: str | None = None",
    "4b tiering/manager.py: RequestState.secondary_lookup_tier",
)

apply(
    TIERING_MANAGER,
    """        self._kv_memoryview = mmap_region.create_kv_memoryview()""",
    """        self._kv_memoryview = mmap_region.create_kv_memoryview()

        # radiance tier-report: bytes per CPU-tier block. This is the only place
        # the mmap row stride is known, and the byte-valued capacity and
        # eviction counters need it. Guarded: a layout change leaves the byte
        # counters unemitted rather than crashing the metrics path.
        _tr_strides = getattr(self._kv_memoryview, "strides", None)
        self._tr_block_bytes = int(_tr_strides[0]) if _tr_strides else 0""",
    "radiance tier-report: bytes per CPU-tier block.",
    "4c tiering/manager.py: CPU tier block size",
)

apply(
    TIERING_MANAGER,
    """        # Buffers manager-level observations (e.g. lookup delay) between
        # get_stats() calls; merged in and reset each time get_stats() runs.
        self._stats = OffloadingConnectorStats()""",
    """        # Buffers manager-level observations (e.g. lookup delay) between
        # get_stats() calls; merged in and reset each time get_stats() runs.
        self._stats = OffloadingConnectorStats()

        # radiance tier-report: tokens covered by one offloaded block hash. Set
        # by the spec after construction (only the spec knows it); 0 leaves the
        # token counters unemitted and the block counters still correct.
        self._tr_tokens_per_hash: int = 0""",
    "self._tr_tokens_per_hash: int = 0",
    "4d tiering/manager.py: tokens_per_hash slot",
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

    def _tr_count_hit(self, tier_name: str, keys: Collection[OffloadKey]) -> None:
        """radiance tier-report (P1): attribute served blocks/tokens to a tier.

        Counts DISTINCT BLOCK HASHES, not keys. An OffloadKey is
        (block_hash, kv_group_idx), so one logical block is looked up once per
        KV cache group; counting keys would multiply every token figure by the
        group count. Worse, the multiplier is not uniform -- in this build the
        Mamba groups store 1-in-8 while the attention groups store 1-in-1 -- so
        it could not be divided back out afterwards.

        The CPU tier is counted at prepare_load(), i.e. everything handed to the
        GPU, which INCLUDES blocks a secondary tier promoted into CPU. Each
        secondary tier is counted at promotion. Neither number is wrong, but
        only the pair is usable: the report subtracts the secondary tiers from
        the CPU total to get the volume that really originated in RAM. Without
        that subtraction a disk tier doing all the work reads as dead weight.

        Guarded, because a metrics path must never be able to fail a load.
        """
        try:
            num_blocks = len({get_offload_block_hash(key) for key in keys})
        except Exception:
            return
        if not num_blocks:
            return
        label = (tier_name,)
        self._stats.increase_counter(TierReportMetrics.HIT_BLOCKS, num_blocks, label)
        if self._tr_tokens_per_hash:
            self._stats.increase_counter(
                TierReportMetrics.HIT_TOKENS,
                num_blocks * self._tr_tokens_per_hash,
                label,
            )''',
    "def _tr_count_hit",
    "4e tiering/manager.py: _tr_count_hit helper",
)

apply(
    TIERING_MANAGER,
    '''        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        return self.primary_tier.prepare_load(keys, req_context)''',
    '''        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        # radiance tier-report (P1): this is the GPU-facing read, so it is the
        # honest place to measure what the CPU tier delivered -- including
        # blocks promoted into it from a secondary tier this step.
        self._tr_count_hit("cpu", keys)
        return self.primary_tier.prepare_load(keys, req_context)''',
    'self._tr_count_hit("cpu", keys)',
    "4f tiering/manager.py: count CPU-served blocks",
)

apply(
    TIERING_MANAGER,
    """                self._transfer_jobs[job_id] = job_metadata
                tier.submit_load(job_metadata)""",
    """                self._transfer_jobs[job_id] = job_metadata
                # radiance tier-report (P1): a promotion is this tier's hit.
                self._tr_count_hit(tier.tier_type, entry.keys)
                tier.submit_load(job_metadata)""",
    "radiance tier-report (P1): a promotion is this tier's hit.",
    "4g tiering/manager.py: count secondary-tier promotions",
)

apply(
    TIERING_MANAGER,
    """                if (
                    req_state is not None
                    and promoted
                    and req_state.secondary_lookup_start_time is None
                ):
                    req_state.secondary_lookup_start_time = lookup_start""",
    """                if (
                    req_state is not None
                    and promoted
                    and req_state.secondary_lookup_start_time is None
                ):
                    req_state.secondary_lookup_start_time = lookup_start
                    # radiance tier-report (P5): remember whose device the
                    # request is now waiting on.
                    req_state.secondary_lookup_tier = tier.tier_type""",
    "req_state.secondary_lookup_tier = tier.tier_type",
    "4h tiering/manager.py: remember the stalling tier",
)

apply(
    TIERING_MANAGER,
    """        req_state.secondary_lookup_start_time = None
        self._stats.observe_histogram(
            TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )""",
    """        req_state.secondary_lookup_start_time = None
        delay = time.monotonic() - start_time
        self._stats.observe_histogram(
            TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY,
            delay,
        )
        # radiance tier-report (P5): the same wait, charged to the tier that
        # caused it, as a cumulative counter. A histogram says how bad the worst
        # wait was; this says how many hours of the server's life the device
        # cost, which is the term that decides whether a tier is a net loss.
        tier_name = req_state.secondary_lookup_tier
        req_state.secondary_lookup_tier = None
        if tier_name:
            self._stats.increase_counter(
                TierReportMetrics.STALL_SECONDS, delay, (tier_name,)
            )""",
    "radiance tier-report (P5): the same wait, charged to the tier",
    "4i tiering/manager.py: per-tier stall seconds",
)

# ---------------------------------------------------------------------------
# 5. The spec hands the manager the hash granularity, which is the only thing
#    it needs to turn block counts into token counts.
# ---------------------------------------------------------------------------
apply(
    TIERING_SPEC,
    """            self._manager = tiering_manager""",
    """            # radiance tier-report: only the spec knows how many tokens one
            # offloaded block hash covers, and the manager needs it to report
            # tokens rather than blocks.
            tiering_manager._tr_tokens_per_hash = int(self.tokens_per_hash)
            self._manager = tiering_manager""",
    "tiering_manager._tr_tokens_per_hash",
    "5  tiering/spec.py: hand over tokens_per_hash",
)

# ---------------------------------------------------------------------------
# 6. The worker transfer path already reports bytes and seconds for the
#    GPU<->primary-tier copy, but only as a SUM. A sum cannot answer "what does
#    a fetch cost me"; the mean of a bimodal distribution is a number no request
#    ever experienced. So carry the per-batch durations alongside the per-batch
#    sizes that are already carried, and turn them into a histogram at the
#    scheduler. This is the P2 timing for the CPU tier.
# ---------------------------------------------------------------------------
apply(
    CONN_COMMON,
    """class DirectionalTransferStats:
    bytes: int = 0
    time: float = 0.0
    sizes: list[int | float] = field(default_factory=list)""",
    """class DirectionalTransferStats:
    bytes: int = 0
    time: float = 0.0
    sizes: list[int | float] = field(default_factory=list)
    # radiance tier-report (P2): per-batch durations, parallel to `sizes`.
    # Travels the same worker->scheduler path `sizes` already travels, so it
    # adds no new serialization requirement -- only more of an element type
    # that is already crossing.
    times: list[float] = field(default_factory=list)""",
    "times: list[float] = field(default_factory=list)",
    "6a offloading/common.py: DirectionalTransferStats.times",
)

apply(
    CONN_COMMON,
    """        return DirectionalTransferStats(
            bytes=self.bytes + other.bytes,
            time=self.time + other.time,
            sizes=[*self.sizes, *other.sizes],
        )""",
    """        return DirectionalTransferStats(
            bytes=self.bytes + other.bytes,
            time=self.time + other.time,
            sizes=[*self.sizes, *other.sizes],
            times=[*self.times, *getattr(other, "times", ())],
        )""",
    'times=[*self.times, *getattr(other, "times", ())],',
    "6b offloading/common.py: aggregate times",
)

apply(
    CONN_COMMON,
    """    def record(self, num_bytes: int, time: float) -> None:
        self.bytes += num_bytes
        self.time += time
        self.sizes.append(num_bytes)""",
    """    def record(self, num_bytes: int, time: float) -> None:
        self.bytes += num_bytes
        self.time += time
        self.sizes.append(num_bytes)
        self.times.append(time)""",
    "self.times.append(time)",
    "6c offloading/common.py: record the duration",
)

# ---------------------------------------------------------------------------
# 7. Re-emit the worker transfer stats with a tier label (P2, CPU side).
#
#    Why "cpu" is hard-coded and why that is honest: this path reports the
#    GPU<->primary-tier copy, and the primary tier is a
#    CPUPrimaryTierOffloadingManager by type in the tiering spec, or the CPU
#    manager itself in the plain offloading spec. A secondary tier's device I/O
#    never reaches the worker -- it is issued by the tier itself -- which is
#    exactly why the fs tier has to time its own batches further down.
#
#    The unlabelled counters are left untouched. Anyone alerting on them keeps
#    working; the labelled series are additive.
# ---------------------------------------------------------------------------
apply(
    CONN_SCHEDULER,
    """    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    make_offload_key,
)""",
    """    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    TierReportMetrics,
    make_offload_key,
)""",
    "    TierReportMetrics,\n    make_offload_key,",
    "7a offloading/scheduler.py: import",
)

apply(
    CONN_SCHEDULER,
    """                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )""",
    """                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
                # radiance tier-report (P2): the same totals, attributed.
                _tr_cpu = (TierReportMetrics.CPU,)
                transfer_stats.increase_counter(
                    TierReportMetrics.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                    _tr_cpu,
                )
                transfer_stats.increase_counter(
                    TierReportMetrics.LOAD_SECONDS,
                    meta.transfer_stats.load.time,
                    _tr_cpu,
                )
                transfer_stats.increase_counter(
                    TierReportMetrics.LOAD_OPS,
                    len(meta.transfer_stats.load.sizes),
                    _tr_cpu,
                )
                for _tr_secs in getattr(meta.transfer_stats.load, "times", ()):
                    transfer_stats.observe_histogram(
                        TierReportMetrics.LOAD_LATENCY, _tr_secs, _tr_cpu
                    )""",
    "radiance tier-report (P2): the same totals, attributed.",
    "7b offloading/scheduler.py: tier-labelled load stats",
)

apply(
    CONN_SCHEDULER,
    """                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )""",
    """                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
                # radiance tier-report (P2): the store-side tax, attributed.
                # This is the "cost" half of the net-value sum -- a tier that
                # serves nothing still charges the server this much.
                _tr_cpu_st = (TierReportMetrics.CPU,)
                transfer_stats.increase_counter(
                    TierReportMetrics.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                    _tr_cpu_st,
                )
                transfer_stats.increase_counter(
                    TierReportMetrics.STORE_SECONDS,
                    meta.transfer_stats.store.time,
                    _tr_cpu_st,
                )
                transfer_stats.increase_counter(
                    TierReportMetrics.STORE_OPS,
                    len(meta.transfer_stats.store.sizes),
                    _tr_cpu_st,
                )
                for _tr_secs in getattr(meta.transfer_stats.store, "times", ()):
                    transfer_stats.observe_histogram(
                        TierReportMetrics.STORE_LATENCY, _tr_secs, _tr_cpu_st
                    )""",
    "radiance tier-report (P2): the store-side tax, attributed.",
    "7c offloading/scheduler.py: tier-labelled store stats",
)

# ---------------------------------------------------------------------------
# 8. The filesystem tier times its own device I/O (P2) and reports its size
#    (P3).
#
#    This is the half of the report that cannot be borrowed from anywhere else.
#    A secondary tier's reads and writes never travel the worker transfer path,
#    so the engine's existing load_bytes/load_time counters contain NOTHING
#    from this device -- they are pure CPU-tier numbers. Divide them and you
#    get ~11.8 GB/s, which is memcpy speed, and the disk is invisible inside
#    it. That single fact is why an operator today cannot answer "is my disk
#    too slow".
# ---------------------------------------------------------------------------
apply(
    FS_MANAGER,
    """import functools
import json
import os""",
    """import functools
import json
import os
import threading
import time""",
    "import threading\nimport time",
    "8a fs/manager.py: imports",
)

apply(
    FS_MANAGER,
    """        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)""",
    """        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

        # radiance tier-report (P2/P3): accumulator drained by get_stats().
        # Touched from pool threads, so it is lock-guarded; the lock is held
        # only for a few integer adds, never across the I/O itself.
        self._tr_lock = threading.Lock()
        self._tr_acc = {
            "load_bytes": 0,
            "load_seconds": 0.0,
            "load_ops": 0,
            "load_lat": [],
            "store_bytes": 0,
            "store_seconds": 0.0,
            "store_ops": 0,
            "store_lat": [],
        }
        # Latency samples are bounded. get_stats() drains them every metrics
        # interval, but if metrics are disabled nothing ever drains them, and
        # an unbounded list on an I/O path is a leak. Dropping samples costs a
        # percentile some precision; the byte and second counters stay exact.
        self._tr_lat_cap = 4096
        try:
            self._tr_root_dir = os.path.dirname(
                self.file_mapper.get_config_file_path()
            )
        except Exception:
            self._tr_root_dir = None
        # statvfs is a syscall against the block device's filesystem; sampling
        # it on every metrics interval would put it on the hot path for a
        # number that moves at the speed of a disk filling up.
        self._tr_statvfs_every = 256
        self._tr_statvfs_countdown = 0
        self._tr_capacity_bytes = 0.0
        self._tr_used_bytes = 0.0
        self._tr_install_timing()""",
    "self._tr_install_timing()",
    "8b fs/manager.py: tier-report state",
)

apply(
    FS_MANAGER,
    """    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()""",
    '''    def _tr_install_timing(self) -> None:
        """radiance tier-report (P2): time this tier's own device I/O.

        The pool's bound enqueue methods are wrapped rather than the submit_*
        bodies, because that keeps the measurement independent of HOW a job was
        split into tasks -- one task per job, or one per fanout batch. What is
        timed is whatever callable actually runs on a pool thread, which is the
        real unit of device work, and the wrapper survives any future change to
        the splitting policy.

        Wrapping is per-instance and idempotent. If anything about the pool is
        not what is expected, timing is simply not installed: an unmeasured
        tier is a missing row in a report, a broken tier is a broken server.
        """
        pool = getattr(self, "_pool", None)
        if pool is None:
            return
        for name, is_load in (("enqueue_load", True), ("enqueue_store", False)):
            original = getattr(pool, name, None)
            if original is None or getattr(original, "_tr_wrapped", False):
                continue
            try:
                setattr(pool, name, self._tr_wrap_enqueue(original, is_load))
            except AttributeError:
                # A __slots__ pool cannot be wrapped; leave it unmeasured.
                return

    def _tr_wrap_enqueue(self, original, is_load: bool):
        def enqueue(job_id, n_tasks, tasks):
            return original(
                job_id, n_tasks, [self._tr_time_task(t, is_load) for t in tasks]
            )

        enqueue._tr_wrapped = True
        return enqueue

    def _tr_time_task(self, task, is_load: bool):
        """Wrap one I/O batch so its wall time and size are recorded.

        Byte count comes off the partial's own arguments -- (paths, view,
        offsets, block_size, o_direct) -- so it is the bytes this batch was
        actually asked to move, not an estimate. If the shape is not what is
        expected the batch is still timed and simply contributes no bytes,
        which degrades a bandwidth figure rather than losing a latency sample.
        """
        num_bytes = 0
        args = getattr(task, "args", None)
        if args is not None and len(args) >= 4:
            try:
                num_bytes = len(args[0]) * int(args[3])
            except (TypeError, ValueError):
                num_bytes = 0

        def timed(*a, **kw):
            start = time.monotonic()
            try:
                return task(*a, **kw)
            finally:
                # try/finally, so a failed read is still charged for the time
                # it burned. A device that fails slowly is the worst case an
                # operator can have, and it must not vanish from the report.
                self._tr_record(is_load, num_bytes, time.monotonic() - start)

        return timed

    def _tr_record(self, is_load: bool, num_bytes: int, seconds: float) -> None:
        acc = getattr(self, "_tr_acc", None)
        if acc is None:
            return
        prefix = "load" if is_load else "store"
        with self._tr_lock:
            acc[prefix + "_bytes"] += num_bytes
            acc[prefix + "_seconds"] += seconds
            acc[prefix + "_ops"] += 1
            samples = acc[prefix + "_lat"]
            if len(samples) < self._tr_lat_cap:
                samples.append(seconds)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()''',
    "def _tr_install_timing",
    "8c fs/manager.py: I/O timing wrappers",
)

apply(
    FS_MANAGER,
    '''        pool = getattr(self, "_pool", None)
        inflight = getattr(pool, "_inflight_jobs", None) if pool is not None else None
        if not isinstance(inflight, int):
            return None
        stats = OffloadingConnectorStats()
        stats.set_gauge(self.FS_INFLIGHT_JOBS, float(inflight))
        return stats''',
    '''        pool = getattr(self, "_pool", None)
        inflight = getattr(pool, "_inflight_jobs", None) if pool is not None else None
        stats = OffloadingConnectorStats()
        # The backlog gauge is now optional rather than the gate: the
        # tier-report counters below are the numbers the sizing argument rests
        # on, and losing all of them because a debug gauge could not be read
        # would be the wrong trade.
        if isinstance(inflight, int):
            stats.set_gauge(self.FS_INFLIGHT_JOBS, float(inflight))
        self._tr_emit_tier_report(stats)
        return stats

    def _tr_emit_tier_report(self, stats) -> None:
        """radiance tier-report: drain the I/O accumulator, sample the size."""
        from vllm.v1.kv_offload.base import TierReportMetrics

        acc = getattr(self, "_tr_acc", None)
        if acc is None:
            return
        with self._tr_lock:
            snapshot = dict(acc)
            acc.update(
                {
                    "load_bytes": 0,
                    "load_seconds": 0.0,
                    "load_ops": 0,
                    "load_lat": [],
                    "store_bytes": 0,
                    "store_seconds": 0.0,
                    "store_ops": 0,
                    "store_lat": [],
                }
            )

        label = (self.tier_type,)
        for prefix, bytes_name, seconds_name, ops_name, lat_name in (
            (
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
            ),
        ):
            if snapshot[prefix + "_ops"]:
                stats.increase_counter(bytes_name, snapshot[prefix + "_bytes"], label)
                stats.increase_counter(
                    seconds_name, snapshot[prefix + "_seconds"], label
                )
                stats.increase_counter(ops_name, snapshot[prefix + "_ops"], label)
            for seconds in snapshot[prefix + "_lat"]:
                stats.observe_histogram(lat_name, seconds, label)

        # Size (P3). The fs tier holds no capacity number of its own -- it
        # writes files until the filesystem says no -- so the filesystem IS the
        # capacity, and its used bytes are what decide whether the tier runs
        # out of room. That number can include data which is not KV blocks;
        # the metric documentation says so, and the report repeats it.
        self._tr_statvfs_countdown -= 1
        if self._tr_statvfs_countdown <= 0 and self._tr_root_dir:
            self._tr_statvfs_countdown = self._tr_statvfs_every
            try:
                st = os.statvfs(self._tr_root_dir)
                self._tr_capacity_bytes = float(st.f_blocks) * float(st.f_frsize)
                self._tr_used_bytes = (
                    float(st.f_blocks - st.f_bfree) * float(st.f_frsize)
                )
            except OSError:
                pass
        if self._tr_capacity_bytes > 0.0:
            stats.set_gauge(
                TierReportMetrics.CAPACITY_BYTES, self._tr_capacity_bytes, label
            )
            stats.set_gauge(TierReportMetrics.USED_BYTES, self._tr_used_bytes, label)
            stats.observe_histogram(
                TierReportMetrics.OCCUPANCY_RATIO,
                self._tr_used_bytes / self._tr_capacity_bytes,
                label,
            )''',
    "def _tr_emit_tier_report",
    "8d fs/manager.py: emit the tier report",
)

print("[radiance] tier-report metrics applied -- 19 tier-labelled series "
      "(P1..P6 of tier-report-metrics-plan.md)")
