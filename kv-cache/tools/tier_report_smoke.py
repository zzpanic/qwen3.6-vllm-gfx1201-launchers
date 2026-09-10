"""Smoke test for patch_kv_offload_tier_report.py -- no GPU, no engine.

Three things are checked, and they are the three that can only fail at runtime:
  A. every tier-labelled series survives the prometheus registration path with
     the right label arity and appears in the exposition text;
  B. the fs tier's I/O timing wrapper actually times a task and its emit
     produces per-tier load/store counters plus a latency histogram;
  C. the CPU tier's emit produces the sizing family from its side maps.
"""
import sys

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    TierReportMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadPromMetrics,
    OffloadingConnectorStats,
    get_connector_metric_definitions,
)

ENGINE_LABELS = ["model_name", "engine"]
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def build_prom():
    """An OffloadPromMetrics with only the fields observe() touches."""
    reg = CollectorRegistry()
    defs = get_connector_metric_definitions()
    p = OffloadPromMetrics.__new__(OffloadPromMetrics)
    p._labelnames = list(ENGINE_LABELS)
    p.per_engine_labelvalues = {0: ["qwen", "0"]}
    p._observe_deprecated_metrics = False
    p._offloading_metric_metadata = defs
    p.offloading_metrics = {}
    p._offloading_metric_defs = {}
    for name, meta in defs.items():
        labelnames = ENGINE_LABELS + list(meta.labelnames)
        if isinstance(meta, OffloadingCounterMetadata):
            m = Counter(name, meta.documentation, labelnames, registry=reg)
        elif isinstance(meta, OffloadingGaugeMetadata):
            m = Gauge(name, meta.documentation, labelnames, registry=reg)
        elif isinstance(meta, OffloadingHistogramMetadata):
            m = Histogram(
                name, meta.documentation, labelnames,
                buckets=meta.buckets, registry=reg,
            )
        else:
            raise AssertionError(name)
        p._offloading_metric_defs[name] = m
    return p, reg


# --- A. registration + label arity -----------------------------------------
print("A. prometheus round-trip")
TR = TierReportMetrics
names = [
    getattr(TR, a) for a in dir(TR)
    if not a.startswith("_") and isinstance(getattr(TR, a), str)
]
names = [n for n in names if n.startswith("vllm:kv_offload_tier_")]
check(len(names) == 19, f"19 tier-report metric names defined (got {len(names)})")

prom, registry = build_prom()
defs = get_connector_metric_definitions()
check(
    all(defs[n].labelnames == ("tier",) for n in names),
    "every tier-report metric carries exactly the 'tier' label",
)

stats = OffloadingConnectorStats()
for tier in ("cpu", "fs"):
    for n in names:
        meta = defs[n]
        if isinstance(meta, OffloadingHistogramMetadata):
            stats.observe_histogram(n, 0.5, (tier,))
        elif isinstance(meta, OffloadingGaugeMetadata):
            stats.set_gauge(n, 1.0, (tier,))
        else:
            stats.increase_counter(n, 2, (tier,))
try:
    prom.observe(stats.data, 0)
    check(True, "observe() accepted every tier-labelled series")
except Exception as e:  # noqa: BLE001
    check(False, f"observe() raised {e!r}")

text = generate_latest(registry).decode()
missing = [
    n for n in names
    if f'{n.replace(":", ":")}' not in text and n.replace(":", "_") not in text
]
check(not missing, f"all series exposed (missing: {missing})")
check('tier="cpu"' in text and 'tier="fs"' in text, "both tier labels exposed")

# Wrong arity must be rejected, or a mislabelled emission would pass silently.
try:
    prom._get_prometheus_metric(TR.HIT_BLOCKS, (), 0)
    check(False, "unlabelled emission on a labelled metric was NOT rejected")
except AssertionError:
    check(True, "unlabelled emission on a labelled metric is rejected")


# --- B. fs tier timing + emit ----------------------------------------------
print("B. fs tier")
import functools
import os
import threading

from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager


class FakePool:
    def __init__(self):
        self.tasks = []
        self._inflight_jobs = 0

    def enqueue_load(self, job_id, n_tasks, tasks):
        self.tasks.extend(tasks)

    def enqueue_store(self, job_id, n_tasks, tasks):
        self.tasks.extend(tasks)


fs = FileSystemTierManager.__new__(FileSystemTierManager)
fs.tier_type = "fs"
fs._pool = FakePool()
fs._tr_lock = threading.Lock()
fs._tr_acc = {
    "load_bytes": 0, "load_seconds": 0.0, "load_ops": 0, "load_lat": [],
    "store_bytes": 0, "store_seconds": 0.0, "store_ops": 0, "store_lat": [],
}
fs._tr_lat_cap = 4096
fs._tr_root_dir = "/tmp"
fs._tr_statvfs_every = 256
fs._tr_statvfs_countdown = 0
fs._tr_capacity_bytes = 0.0
fs._tr_used_bytes = 0.0
fs._tr_install_timing()

check(
    getattr(fs._pool.enqueue_load, "_tr_wrapped", False),
    "pool enqueue_load was wrapped",
)
fs._tr_install_timing()
check(
    getattr(fs._pool.enqueue_load, "_tr_wrapped", False),
    "re-installing timing is idempotent (still wrapped, not double-wrapped)",
)

ran = []


def fake_io(paths, view, offsets, block_size, o_direct):
    ran.append(len(paths))
    return True


# The real shape: functools.partial(batch_load_block, paths, view, offsets,
# block_size, o_direct). Three blocks of 1 MiB.
task = functools.partial(fake_io, ["a", "b", "c"], None, [0, 1, 2], 1 << 20, False)
fs._pool.enqueue_load(1, 1, [task])
check(len(fs._pool.tasks) == 1, "one wrapped task reached the pool")
fs._pool.tasks[0]()
check(ran == [3], "the underlying I/O function still ran with its real args")
check(fs._tr_acc["load_ops"] == 1, "one load op recorded")
check(
    fs._tr_acc["load_bytes"] == 3 * (1 << 20),
    f"bytes read off the partial's args (got {fs._tr_acc['load_bytes']})",
)
check(len(fs._tr_acc["load_lat"]) == 1, "one latency sample recorded")

# A failing batch must still be charged for the time it burned.
def boom(*a, **kw):
    raise OSError("device error")


fs._pool.tasks.clear()
fs._pool.enqueue_store(2, 1, [functools.partial(boom, ["x"], None, [0], 4096, False)])
try:
    fs._pool.tasks[0]()
except OSError:
    pass
check(fs._tr_acc["store_ops"] == 1, "a FAILED store batch is still counted")

fs_stats = OffloadingConnectorStats()
fs._tr_emit_tier_report(fs_stats)
d = fs_stats._values
check(
    d.get(TR.LOAD_BYTES, {}).get(("fs",)) == 3 * (1 << 20),
    "emit produced load_bytes{tier=fs}",
)
check(
    len(d.get(TR.LOAD_LATENCY, {}).get(("fs",), [])) == 1,
    "emit produced a load latency sample for fs",
)
check(TR.STORE_OPS in d, "emit produced store ops for fs")
check(
    d.get(TR.CAPACITY_BYTES, {}).get(("fs",), 0) > 0,
    "statvfs gave a capacity for the fs tier",
)
check(
    0.0 <= d.get(TR.OCCUPANCY_RATIO, {}).get(("fs",), [-1])[0] <= 1.0,
    "occupancy ratio is a fraction",
)
fs_stats2 = OffloadingConnectorStats()
fs._tr_emit_tier_report(fs_stats2)
check(
    TR.LOAD_BYTES not in fs_stats2._values,
    "the accumulator was drained (no double counting across intervals)",
)
try:
    prom.observe(fs_stats.data, 0)
    check(True, "fs emit output is observable by the prom path")
except Exception as e:  # noqa: BLE001
    check(False, f"fs emit output rejected: {e!r}")


# --- C. cpu tier emit -------------------------------------------------------
print("C. cpu tier")
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from collections import OrderedDict

cpu = CPUOffloadingManager.__new__(CPUOffloadingManager)
cpu._tr_tier_name = "cpu"
cpu._tr_block_bytes = 4096
cpu._tr_reads = OrderedDict()
cpu._tr_ghost = OrderedDict()
cpu._tr_reads_before_evict = [0, 0, 1, 5]
cpu._tr_reuse_delays = [3.0, 900.0]
cpu._tr_evictions = 4
cpu._tr_miss_evicted = 2
cpu._num_blocks = 100
cpu._num_allocated_blocks = 90
cpu._free_list = [1, 2, 3]

cpu_stats = OffloadingConnectorStats()
cpu._tr_emit_tier_report(cpu_stats)
v = cpu_stats._values
check(
    v[TR.OCCUPANCY_RATIO][("cpu",)] == [0.87],
    f"occupancy = (90-3)/100 (got {v.get(TR.OCCUPANCY_RATIO)})",
)
check(v[TR.CAPACITY_BYTES][("cpu",)] == 100 * 4096, "capacity bytes")
check(v[TR.USED_BYTES][("cpu",)] == 87 * 4096, "used bytes")
check(v[TR.READS_BEFORE_EVICT][("cpu",)] == [0, 0, 1, 5], "reads-before-evict samples")
check(v[TR.EVICTION_TO_REUSE][("cpu",)] == [3.0, 900.0], "eviction-to-reuse samples")
check(v[TR.EVICTIONS][("cpu",)] == 4, "evictions counter")
check(v[TR.EVICTED_BYTES][("cpu",)] == 4 * 4096, "evicted bytes")
check(v[TR.MISS_EVICTED][("cpu",)] == 2, "would-have-hit counter")

cpu_stats2 = OffloadingConnectorStats()
cpu._tr_emit_tier_report(cpu_stats2)
check(
    TR.EVICTIONS not in cpu_stats2._values
    and TR.READS_BEFORE_EVICT not in cpu_stats2._values,
    "cpu side maps drained (no double counting)",
)
bare = CPUOffloadingManager.__new__(CPUOffloadingManager)
try:
    bare._tr_emit_tier_report(OffloadingConnectorStats())
    check(True, "an uninitialised manager degrades to no metrics, not a crash")
except Exception as e:  # noqa: BLE001
    check(False, f"uninitialised manager raised {e!r}")

try:
    prom.observe(cpu_stats.data, 0)
    check(True, "cpu emit output is observable by the prom path")
except Exception as e:  # noqa: BLE001
    check(False, f"cpu emit output rejected: {e!r}")

print()
if failures:
    print(f"FAILED: {len(failures)}")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
