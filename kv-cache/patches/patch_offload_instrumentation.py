#!/usr/bin/env python3
"""Make the KV offload store path observable. Instrumentation only -- no behaviour change.

WHY THIS EXISTS (see cache-preemption-patch-plan.md Revision 3):

The 2026-09-07 tierbench baseline (salt 9e3561) showed the offload tier serving ZERO bytes
across an 11-minute run: both re-read phases and the 3-client concurrent phase came back as
full recomputes, and `kv_offload_load_bytes_total` never moved. The cause is on the STORE
side, not the read side:

  * The CPU->fs cascade is a *load* from the CPU tier's point of view -- it reads CPU blocks
    in order to write them to disk -- so `prepare_load` pins every block queued for the fs
    tier until its write lands (cpu/manager.py:130, released in complete_load at :154).
  * `CPUOffloadingManager.prepare_store` (cpu/manager.py:186) returns None when
    `num_blocks_to_evict > self._num_evictable_cache_blocks`, i.e. when too few blocks have
    ref_cnt == 0.
  * GPU->CPU ran at 10.31 GB/s; CPU->fs drained at 79 MB/s. 130:1. The tier fills with
    pinned blocks, admission fails, nothing is retained, every request recomputes.

Three numbers were needed to see that and none of them were emitted. This patch adds them.

WHAT IT CHANGES

  1+2. Widens both `lookup_async_delay` histograms. Their top finite bucket was 10 s, and the
       measured delay was 60.6 s per observation against a 60.95 s TTFT -- every real stall
       landed in +Inf, so the one histogram that already saw the problem could not express it.
       New buckets run to 600 s. Widening buckets is backwards compatible: `_count` and `_sum`
       keep their meaning, only the resolution improves.

  3+4+5. Adds `kv_offload_cpu_cache_evictable_perc` and `kv_offload_cpu_cache_free_perc`.
       `prepare_store` admits a batch iff `len(keys_to_store) <= free + evictable`, and
       NEITHER term was observable. Note the existing `cpu_cache_usage_perc` does NOT measure
       residency -- manager.py:297 subtracts evictable blocks, so it means "fraction pinned by
       in-flight transfers" and reads 0.0 at idle with 228 GB sitting on the fs tier. These two
       new gauges are the ones that explain an allocation failure.

  6. Adds `kv_offload_fs_inflight_jobs` on the fs tier: the cascade backlog in jobs, straight
     off `DualQueueThreadPool._inflight_jobs`. This is the quantity that drives the pinning.
     `SecondaryTierManager` already exposes both hooks (`build_metric_definitions` at
     tiering/base.py:307 and `get_stats` at :313, fanned out from tiering/spec.py:136 and
     tiering/manager.py:826) -- the fs tier simply never overrode them. No plumbing needed in
     v1/metrics/loggers.py.

RISK: low. Every hunk either adds a gauge or lengthens a bucket tuple. No control flow is
touched, so this cannot change what is stored, evicted, promoted or served. The two new gauge
reads are guarded with getattr so a future rename degrades to "no gauge" rather than a crash
in the metrics path.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

KVO = SP / "vllm/v1/kv_offload"
TIERING_SPEC = KVO / "tiering/spec.py"
FS_MANAGER = KVO / "tiering/fs/manager.py"
CPU_COMMON = KVO / "cpu/common.py"
CPU_SPEC = KVO / "cpu/spec.py"
CPU_MANAGER = KVO / "cpu/manager.py"
CONN_METRICS = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"

# Shared bucket ladder for both async-delay histograms. The old ladder stopped at 10 s.
# 20/30 catch a slow-but-working promotion; 60/120 are the observed stall band; 300/600
# exist so a pathological case is still distinguishable from +Inf.
_WIDE = """                    5,
                    10,
                    20,
                    30,
                    60,
                    120,
                    300,
                    600,
                ),"""

# --- hunk 1: tiering LOOKUP_ASYNC_DELAY buckets --------------------------------------
apply(
    TIERING_SPEC,
    """                    "secondary-tier lookup until the request is allocated or "
                    "finishes, in seconds."
                ),
                buckets=(
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
                    5,
                    10,
                ),""",
    """                    "secondary-tier lookup until the request is allocated or "
                    "finishes, in seconds."
                ),
                # radiance: extended past 10 s -- a measured 60.6 s stall was
                # invisible in +Inf. See cache-preemption-patch-plan.md R3.5.
                buckets=(
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
"""
    + _WIDE,
    "radiance: extended past 10 s",
    "tiering lookup_async_delay buckets",
)

# --- hunk 2: connector LOOKUP_ASYNC_DELAY buckets ------------------------------------
apply(
    CONN_METRICS,
    """                "deferring and the following lookup resolving, or request "
                "finish, in seconds."
            ),
            buckets=(
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                5,
                10,
            ),""",
    """                "deferring and the following lookup resolving, or request "
                "finish, in seconds."
            ),
            # radiance: extended past 10 s. See cache-preemption-patch-plan.md R3.5.
            buckets=(
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                5,
                10,
                20,
                30,
                60,
                120,
                300,
                600,
            ),""",
    "radiance: extended past 10 s",
    "connector lookup_async_delay buckets",
)

# --- hunk 3: CPU tier metric names ---------------------------------------------------
apply(
    CPU_COMMON,
    '    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"',
    '''    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"
    # radiance: the two terms prepare_store() actually compares against. See R3.2.
    CPU_CACHE_EVICTABLE_PERC = "vllm:kv_offload_cpu_cache_evictable_perc"
    CPU_CACHE_FREE_PERC = "vllm:kv_offload_cpu_cache_free_perc"''',
    "CPU_CACHE_EVICTABLE_PERC",
    "cpu tier metric names",
)

# --- hunk 4: CPU tier gauge definitions ----------------------------------------------
apply(
    CPU_SPEC,
    """            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),""",
    """            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_EVICTABLE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of the CPU KV-cache tier holding cached blocks "
                    "that are reclaimable right now (ref_cnt == 0). Together "
                    "with the free fraction this is the admission headroom "
                    "prepare_store() tests; when it reaches 0 stores are "
                    "refused and nothing is retained."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_FREE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of the CPU KV-cache tier that is unallocated."
                ),
            ),""",
    "CPU_CACHE_EVICTABLE_PERC",
    "cpu tier gauge definitions",
)

# --- hunk 5: emit the CPU tier gauges ------------------------------------------------
apply(
    CPU_MANAGER,
    """        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)""",
    """        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        # radiance: prepare_store() admits a batch iff
        #   len(keys_to_store) <= free + evictable
        # and neither term was observable. See cache-preemption-patch-plan.md R3.2.
        if self._num_blocks > 0:
            stats.set_gauge(
                CPUOffloadingMetrics.CPU_CACHE_EVICTABLE_PERC,
                self._num_evictable_cache_blocks / self._num_blocks,
            )
            stats.set_gauge(
                CPUOffloadingMetrics.CPU_CACHE_FREE_PERC,
                self._get_num_free_blocks() / self._num_blocks,
            )""",
    "CPU_CACHE_EVICTABLE_PERC",
    "cpu tier gauge emission",
)

# --- hunk 6: fs tier cascade backlog -------------------------------------------------
# SecondaryTierManager already declares both hooks and defaults them to {} / None; the fs
# tier just never overrode them. Imports are done inside the methods so the module's import
# block (and its cycle ordering) is left alone.
apply(
    FS_MANAGER,
    """    @override
    def take_events(self) -> Iterable[OffloadingEvent]:""",
    '''    FS_INFLIGHT_JOBS = "vllm:kv_offload_fs_inflight_jobs"

    @classmethod
    def build_metric_definitions(cls, extra_config):
        """radiance: expose the cascade backlog. See cache-preemption-patch-plan.md R3.2."""
        from vllm.v1.kv_offload.base import OffloadingGaugeMetadata

        return {
            cls.FS_INFLIGHT_JOBS: OffloadingGaugeMetadata(
                documentation=(
                    "Number of KV offload jobs queued or in flight on the "
                    "filesystem tier's thread pool. Each job holds a ref on the "
                    "CPU-tier blocks it is reading, so a sustained backlog is "
                    "what starves prepare_store() of evictable blocks."
                ),
            ),
        }

    def get_stats(self):
        """radiance: emit the cascade backlog. Guarded -- a rename must not crash metrics."""
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        pool = getattr(self, "_pool", None)
        inflight = getattr(pool, "_inflight_jobs", None) if pool is not None else None
        if not isinstance(inflight, int):
            return None
        stats = OffloadingConnectorStats()
        stats.set_gauge(self.FS_INFLIGHT_JOBS, float(inflight))
        return stats

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:''',
    "FS_INFLIGHT_JOBS",
    "fs tier cascade backlog gauge",
)

print("  DONE  kv offload instrumentation")
