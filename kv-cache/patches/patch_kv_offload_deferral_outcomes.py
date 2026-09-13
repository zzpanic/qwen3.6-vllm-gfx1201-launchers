#!/usr/bin/env python3
"""radiance: instrument what happens AFTER a KV-offload lookup deferral (task 50 / task 32).

WHY
===
The engine shows `lookup_deferred_backend_total` and `lookup_chunk_retry_total`,
but nothing says what happened AFTER the engine waited. `lookup_deferred_backend`
counts whole lookups, so a request that defers for N steps inflates it N-fold,
and the terminal state (did it get a hit, give up, or die still waiting) is
folded back into the generic `lookup_served` / `lookup_skip_*` counters. There
is no per-request view of the deferral, no measure of how many *steps* a request
waited, and no view of the cost of unbounded deferral.

Task 50 measured 15 full-recompute intervals in an hour in which NOT ONE
lookup-outcome counter moved, and a `lookup_partition` drift of a constant +32.
That drift, however, is NOT a silent exit: `lookup_calls` is incremented only
inside `_lookup()`, and every `_lookup()` call lands in exactly one of the six
outcome buckets (verified against the installed tree), so at zero traffic
`drift == lookup_skip_short_result + lookup_deferred_loading` (the two buckets
the partition deliberately does not sum). The one lookup-path exit that genuinely
records no outcome counter at all is the `transfer_jobs` branch of
`get_num_new_matched_tokens` (scheduler.py): it returns before `_lookup()` is
called, so it increments neither `lookup_calls` nor any outcome, and is invisible
in every existing counter. This patch gives the deferral a per-request
lifecycle (counters 1-4 + the depth histogram) and names that one silent exit.

WHAT IT ADDS (all `vllm:kv_offload_lookup_*`, connector-level, registered in
get_connector_metric_definitions() so the OffloadPromMetrics.observe() assertion
cannot trip)
==================================================
    deferral_total                 distinct requests that deferred (denominator)
    deferral_served                deferred, then got a hit (>0)
    deferral_gave_up               deferred, then prefilled (0)
    deferral_unresolved            deferred, finished/aborted still waiting
    deferral_depth                 histogram of scheduler steps waited
                                   (first deferral -> resolution/finish)
    transfer_jobs_deferred         the one silent exit: the request was declined
                                   because it still had in-flight transfers
    served + gave_up + unresolved = deferral_total (a clean partition)

Instrumentation only. No behaviour change: a deferral still defers, a served
result still serves, a 0 still prefills, and the transfer_jobs branch still
returns `None, False` (a WAIT, per the scheduler's docstring -- with the
deployed RADIANCE_OFFLOAD_PENDING_IS_MISS=0 it is a delay, not a miss). The
only code added is three dataclass fields with defaults, three guarded helper
methods, and guarded counter emissions.

RISK
====
Every emission is wrapped in try/except -> pass, so a metric path can never fail
a lookup or a request. The three new dataclass fields all default (False / 0 /
False), so an existing request shape is unchanged. Nothing existing is modified;
lines are only added.

PRELUDE POSITION
================
Runs AFTER the task-20 bundle (house 1-8 -> promotion-refusal -> wall-clock).
Its off_metrics.py anchors (the `LOOKUP_SERVED_TOKENS` name-table entry and the
`LOOKUP_SERVED_TOKENS` definition) are stable across the bundle -- the bundle's
off_metrics.py changes land at the dict TAIL (the `PROMOTION_INITIATED` entry,
then the wall-clock thread-seconds defs) and in base.py, none of which touch
these anchors -- so there is no forward dependency on the bundle. It is placed
last so the live patched tree (house 1-8 + bundle) is exactly the tree it
anchors against, the same re-anchoring task 15 was forced into.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
OFF_SCHED = (
    SP
    / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
)
OFF_METRICS = (
    SP
    / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"
)

print("radiance: KV offload deferral-outcome + transfer_jobs-deferred counters")

# ---------------------------------------------------------------------------
# 1. metrics.py: the six new counter names, in the _ConnectorMetricName table,
#    beside the other lookup terminal-outcome names.
# ---------------------------------------------------------------------------
apply(
    OFF_METRICS,
    """    LOOKUP_SERVED_TOKENS = "vllm:kv_offload_lookup_served_tokens"
""",
    """    LOOKUP_SERVED_TOKENS = "vllm:kv_offload_lookup_served_tokens"

    # radiance deferral-outcomes (task 50/32): the per-request lifecycle of a
    # deferred lookup, plus the one lookup-path exit that records no outcome.
    LOOKUP_DEFERRAL_TOTAL = "vllm:kv_offload_lookup_deferral_total"
    LOOKUP_DEFERRAL_SERVED = "vllm:kv_offload_lookup_deferral_served_total"
    LOOKUP_DEFERRAL_GAVE_UP = "vllm:kv_offload_lookup_deferral_gave_up_total"
    LOOKUP_DEFERRAL_UNRESOLVED = (
        "vllm:kv_offload_lookup_deferral_unresolved_total"
    )
    LOOKUP_DEFERRAL_DEPTH = "vllm:kv_offload_lookup_deferral_depth"
    LOOKUP_TRANSFER_JOBS_DEFERRED = (
        "vllm:kv_offload_lookup_transfer_jobs_deferred_total"
    )
""",
    'LOOKUP_TRANSFER_JOBS_DEFERRED = (\n        "vllm:kv_offload_lookup_transfer_jobs_deferred_total"\n    )',
    "1  offloading/metrics.py: deferral-outcome name table",
)

# ---------------------------------------------------------------------------
# 2. metrics.py: the six definitions, registered at the connector level. Placed
#    after the LOOKUP_SERVED_TOKENS definition (before the tier-report block)
#    so they sit beside their siblings and are independent of the bundle's
#    tail-of-dict insertions.
# ---------------------------------------------------------------------------
apply(
    OFF_METRICS,
    """        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),
        # radiance tier-report: everything below carries a `tier` label so one
""",
    """        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),
        # radiance deferral-outcomes (task 50/32): what happened AFTER a lookup
        # deferred. served + gave_up + unresolved = deferral_total. The
        # transfer_jobs counter is the one lookup-path exit that records no
        # outcome counter at all (it returns before _lookup()); it is a WAIT,
        # not a miss. Instrumentation only.
        _ConnectorMetricName.LOOKUP_DEFERRAL_TOTAL: OffloadingCounterMetadata(
            documentation=(
                "Distinct requests whose offload lookup deferred (first "
                "deferral). The denominator for the deferral-outcome "
                "partition; lookup_deferred_backend counts whole lookups, so a "
                "request deferring N steps inflates it N-fold."
            ),
        ),
        _ConnectorMetricName.LOOKUP_DEFERRAL_SERVED: OffloadingCounterMetadata(
            documentation=(
                "Deferred requests that then resolved to a usable external hit "
                "(>0 tokens after the wait)."
            ),
        ),
        _ConnectorMetricName.LOOKUP_DEFERRAL_GAVE_UP: OffloadingCounterMetadata(
            documentation=(
                "Deferred requests that then resolved to 0 (gave up and "
                "prefilled). served + gave_up + unresolved = deferral_total."
            ),
        ),
        _ConnectorMetricName.LOOKUP_DEFERRAL_UNRESOLVED: (
            OffloadingCounterMetadata(
                documentation=(
                    "Deferred requests that finished/aborted with the lookup "
                    "still pending -- the cost of unbounded deferral."
                )
            )
        ),
        _ConnectorMetricName.LOOKUP_DEFERRAL_DEPTH: OffloadingHistogramMetadata(
            documentation=(
                "Scheduler steps a deferred request waited, first deferral to "
                "resolution/finish. Steps decouple the wait from batch size and "
                "separate the normal 1-2-step async round-trip from a long one "
                "(a hung fs read / a livelock)."
            ),
            buckets=(1.5, 2.5, 3.5, 5.5, 10.5, 20.5, 50.5, 100.5),
        ),
        _ConnectorMetricName.LOOKUP_TRANSFER_JOBS_DEFERRED: (
            OffloadingCounterMetadata(
                documentation=(
                    "Times get_num_new_matched_tokens declined a request "
                    "because it still had in-flight transfers (the "
                    "transfer_jobs branch). The only lookup-path exit that "
                    "records no outcome counter, so it was invisible; it "
                    "returns a WAIT (query again later), not a miss."
                )
            )
        ),
        # radiance tier-report: everything below carries a `tier` label so one
""",
    "LOOKUP_DEFERRAL_TOTAL: OffloadingCounterMetadata(",
    "2  offloading/metrics.py: deferral-outcome definitions",
)

# ---------------------------------------------------------------------------
# 3. scheduler.py: the three per-request lifecycle fields on RequestOffloadState.
# ---------------------------------------------------------------------------
apply(
    OFF_SCHED,
    """    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False
""",
    """    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False
    # radiance deferral-outcomes (task 50/32): per-request _lookup deferral
    # lifecycle. `deferred` marks the first deferred step (deferral_total is
    # emitted once); `deferral_depth` counts the steps waited; `deferral_
    # settled` closes the lifecycle on resolution (served/gave_up) or finish
    # (unresolved) so the depth histogram is recorded exactly once.
    deferred: bool = False
    deferral_depth: int = 0
    deferral_settled: bool = False
""",
    "    deferral_settled: bool = False",
    "3  offloading/scheduler.py: RequestOffloadState deferral fields",
)

# ---------------------------------------------------------------------------
# 4. scheduler.py: the three guarded helpers, beside the async-delay observer.
# ---------------------------------------------------------------------------
apply(
    OFF_SCHED,
    """        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _generate_job_id(self) -> int:
""",
    """        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _count_deferral_step(self, req_status: "RequestOffloadState") -> None:
        \"\"\"radiance task 50/32: count one step of a _lookup deferral.

        On the first deferred step mark the request and emit deferral_total
        (the denominator); every deferred step adds one to the depth.
        Guarded: a metrics path must never fail a lookup.
        \"\"\"
        try:
            if not req_status.deferral_settled:
                req_status.deferral_depth += 1
            if not req_status.deferred:
                req_status.deferred = True
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.LOOKUP_DEFERRAL_TOTAL
                )
        except Exception:
            pass

    def _settle_deferral(
        self, req_status: "RequestOffloadState", num_hit_tokens: int
    ) -> None:
        \"\"\"radiance task 50/32: settle a _lookup deferral that resolved.

        num_hit_tokens is a real (non-None) result: >0 = the request got a hit
        after the wait (deferral_served); 0 = it gave up and prefilled
        (deferral_gave_up). Records the terminal state and the depth, then
        closes the lifecycle. Guarded: a metrics path must never fail a lookup.
        \"\"\"
        try:
            if req_status.deferred and not req_status.deferral_settled:
                if num_hit_tokens > 0:
                    self._connector_stats.increase_counter(
                        _ConnectorMetricName.LOOKUP_DEFERRAL_SERVED
                    )
                else:
                    self._connector_stats.increase_counter(
                        _ConnectorMetricName.LOOKUP_DEFERRAL_GAVE_UP
                    )
                self._connector_stats.observe_histogram(
                    _ConnectorMetricName.LOOKUP_DEFERRAL_DEPTH,
                    float(req_status.deferral_depth),
                )
                req_status.deferral_settled = True
        except Exception:
            pass

    def _finish_deferral(self, req_status: "RequestOffloadState") -> None:
        \"\"\"radiance task 50/32: a request finished while still deferred.

        The lookup was still pending (deferred, not settled) when the request
        ended: count it as deferral_unresolved (the cost of unbounded deferral)
        and record the depth. Guarded: a metrics path must never fail a request.
        \"\"\"
        try:
            if req_status.deferred and not req_status.deferral_settled:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.LOOKUP_DEFERRAL_UNRESOLVED
                )
                self._connector_stats.observe_histogram(
                    _ConnectorMetricName.LOOKUP_DEFERRAL_DEPTH,
                    float(req_status.deferral_depth),
                )
                req_status.deferral_settled = True
        except Exception:
            pass

    def _generate_job_id(self) -> int:
""",
    "def _count_deferral_step",
    "4  offloading/scheduler.py: deferral lifecycle helpers",
)

# ---------------------------------------------------------------------------
# 5. scheduler.py: the one silent exit -- the transfer_jobs branch. It returns
#    before _lookup(), so it never increments lookup_calls or any outcome; this
#    is the counter that makes it visible.
# ---------------------------------------------------------------------------
apply(
    OFF_SCHED,
    """        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False
""",
    """        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            # radiance task 50/32: this is the only lookup-path exit that
            # records no outcome counter (it returns before _lookup()). It is a
            # WAIT, not a miss. Guarded: a metrics path must never fail a
            # lookup.
            try:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.LOOKUP_TRANSFER_JOBS_DEFERRED
                )
            except Exception:
                pass
            return None, False
""",
    "LOOKUP_TRANSFER_JOBS_DEFERRED\n                )",
    "5  offloading/scheduler.py: count the transfer_jobs silent exit",
)

# ---------------------------------------------------------------------------
# 6. scheduler.py: count the _lookup deferral steps (denominator + depth) and
#    settle the lifecycle on resolution. The num_hit_tokens is None branch is
#    the _lookup deferral; the else branch (a real 0 or >0) is the resolution,
#    which also covers the mixed-hit kill-switch path that sets 0 without a
#    _lookup call.
# ---------------------------------------------------------------------------
apply(
    OFF_SCHED,
    """            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
            else:
                self._maybe_observe_lookup_async_delay(req_status)
""",
    """            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
                # radiance task 50/32: a deferred step (denominator + depth).
                self._count_deferral_step(req_status)
            else:
                self._maybe_observe_lookup_async_delay(req_status)
                # radiance task 50/32: the deferred lookup resolved (served or
                # gave up); settle the lifecycle.
                self._settle_deferral(req_status, num_hit_tokens)
""",
    "self._count_deferral_step(req_status)",
    "6  offloading/scheduler.py: count deferral steps + settle on resolution",
)

# ---------------------------------------------------------------------------
# 7. scheduler.py: a request that finished while still deferred is unresolved.
# ---------------------------------------------------------------------------
apply(
    OFF_SCHED,
    """        self._maybe_observe_lookup_async_delay(req_status)

        # Update offload keys with final block hash so _build_store_jobs can
""",
    """        self._maybe_observe_lookup_async_delay(req_status)
        # radiance task 50/32: if the request ended while its lookup was still
        # pending, that deferral resolved as unresolved.
        self._finish_deferral(req_status)

        # Update offload keys with final block hash so _build_store_jobs can
""",
    "self._finish_deferral(req_status)",
    "7  offloading/scheduler.py: count a finish-while-deferred as unresolved",
)

print("[radiance] deferral-outcome + transfer_jobs-deferred counters applied "
      "(deferral_total/served/gave_up/unresolved + depth + transfer_jobs)")
