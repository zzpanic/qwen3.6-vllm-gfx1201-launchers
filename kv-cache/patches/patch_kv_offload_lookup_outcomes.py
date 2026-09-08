#!/usr/bin/env python3
"""Phase A of the closeout plan: make the offload LOOKUP path observable.
Instrumentation only -- no behaviour change.

WHY THIS EXISTS (cache-preemption-patch-plan.md R3.14.2):

A bench on this server got a real 85,696-token external hit, bit-identical to a cold
recompute (R3.10, R3.11). Production, on the same server, gets ~0 external hits across
~1.4M queried tokens -- with the mixed-hit guard OFF, so that is not the cause (R3.12.6).
Four mechanisms can produce that zero and nothing measured so far distinguishes them:

  A1 retry livelock      -- any RETRY makes _maximal_prefix_lookup set defer_lookup, so
                            _lookup returns None and the request is delayed, never served.
  A2 cascade vs think time -- 83 s of fs->CPU staging is longer than a user's pause, so the
                            chunks the next turn needs are not queryable yet: MISS on
                            chunks we know were written.
  A3 hit window < 1 chunk -- _lookup returns 0 outright when there is less than one
                            1,648-token chunk to fetch. With the GPU prefix cache already
                            serving 11.5%, the remainder may routinely be under that floor.
  A4 the eagle haircut   -- misclassified full-attention groups take the eagle pop without
                            the compensating query extension (R3.14.1). Bounded at 1,648
                            tokens, so it only matters as A3's accomplice.

Each mechanism leaves a different fingerprint in the lookup outcome distribution, and
every one of those outcomes is already an explicit branch in the source. This patch counts
them. It adds no branches of its own.

WHAT IT CHANGES

  1+2. metrics.py: twelve new counter names and their OffloadingCounterMetadata entries.
       No labels, so nothing downstream has to learn a new label schema.

  3.   scheduler.py: a _count_lookup_result helper, and a call to it at BOTH backend lookup
       sites -- _maximal_prefix_lookup (full-attention groups) and _sliding_window_lookup
       (Mamba/GDN groups). Counting only one would silently omit six of nine groups.

  4-9. scheduler.py: one counter at each terminal branch of _lookup -- the three distinct
       `return 0` sites (which are three different stories and must not be conflated), both
       `return None` sites, and the success path.

HOW TO READ THE RESULT (R3.14.2 exit criterion)

  mostly ..._chunk_retry / ..._deferred_backend  -> A1, fix the livelock
  mostly ..._chunk_miss                          -> A2, a residency/timing fix, not volume
  mostly ..._skip_short_window / _skip_short_result -> A3: production's misses are smaller
                                                    than the cache's granularity, and no
                                                    amount of R3.13 rescues that
  mostly ..._served                              -> the hits are real; the ~0 is a metrics
                                                    bug, not a cache bug

RISK: low, and lower than the R3.6 instrumentation patch this sits beside. Every hunk either
adds a counter definition or inserts one `increase_counter` call on a line of its own. No
condition, no return value and no control flow is touched, so this cannot change what is
looked up, stored, evicted, promoted or served. The helper is defensive: an unrecognised
LookupResult member is counted as `..._chunk_other` rather than raising inside a lookup.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

OFFL = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading"
METRICS = OFFL / "metrics.py"
SCHED = OFFL / "scheduler.py"

# ---------------------------------------------------------------- 1. metric names

apply(
    METRICS,
    '''    LOOKUP_SYNC_DELAY = "vllm:kv_offload_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_lookup_async_delay_seconds"
    ALLOCATION_FAILURE = "vllm:kv_offload_allocation_failure"''',
    '''    LOOKUP_SYNC_DELAY = "vllm:kv_offload_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_lookup_async_delay_seconds"
    ALLOCATION_FAILURE = "vllm:kv_offload_allocation_failure"

    # radiance: lookup outcome autopsy. See cache-preemption-patch-plan.md R3.14.2.
    # Per-chunk results from the backend, at both lookup sites.
    LOOKUP_CHUNK_HIT = "vllm:kv_offload_lookup_chunk_hit"
    LOOKUP_CHUNK_HIT_PENDING = "vllm:kv_offload_lookup_chunk_hit_pending"
    LOOKUP_CHUNK_RETRY = "vllm:kv_offload_lookup_chunk_retry"
    LOOKUP_CHUNK_MISS = "vllm:kv_offload_lookup_chunk_miss"
    LOOKUP_CHUNK_OTHER = "vllm:kv_offload_lookup_chunk_other"
    # Terminal outcomes of a whole _lookup call. These partition every exit.
    LOOKUP_CALLS = "vllm:kv_offload_lookup_calls"
    LOOKUP_SKIP_SHORT_WINDOW = "vllm:kv_offload_lookup_skip_short_window"
    LOOKUP_SKIP_ZERO_HIT = "vllm:kv_offload_lookup_skip_zero_hit"
    LOOKUP_SKIP_SHORT_RESULT = "vllm:kv_offload_lookup_skip_short_result"
    LOOKUP_DEFERRED_BACKEND = "vllm:kv_offload_lookup_deferred_backend"
    LOOKUP_DEFERRED_LOADING = "vllm:kv_offload_lookup_deferred_loading"
    LOOKUP_SERVED = "vllm:kv_offload_lookup_served"
    LOOKUP_SERVED_TOKENS = "vllm:kv_offload_lookup_served_tokens"''',
    "LOOKUP_CHUNK_HIT",
    "metrics: lookup outcome counter names",
)

# ---------------------------------------------------------------- 2. metric definitions

apply(
    METRICS,
    '''        _ConnectorMetricName.ALLOCATION_FAILURE: OffloadingCounterMetadata(
            documentation=(
                "Number of KV offload store allocation attempts that failed."
            ),
        ),
    }''',
    '''        _ConnectorMetricName.ALLOCATION_FAILURE: OffloadingCounterMetadata(
            documentation=(
                "Number of KV offload store allocation attempts that failed."
            ),
        ),
        # radiance: lookup outcome autopsy (cache-preemption-patch-plan.md R3.14.2).
        _ConnectorMetricName.LOOKUP_CHUNK_HIT: OffloadingCounterMetadata(
            documentation="Offload chunk lookups that returned HIT.",
        ),
        _ConnectorMetricName.LOOKUP_CHUNK_HIT_PENDING: OffloadingCounterMetadata(
            documentation=(
                "Offload chunk lookups that returned HIT_PENDING: present but "
                "not yet readable, which defers the request."
            ),
        ),
        _ConnectorMetricName.LOOKUP_CHUNK_RETRY: OffloadingCounterMetadata(
            documentation=(
                "Offload chunk lookups that returned RETRY: location uncertain. "
                "A high count against a low served count is the retry livelock."
            ),
        ),
        _ConnectorMetricName.LOOKUP_CHUNK_MISS: OffloadingCounterMetadata(
            documentation="Offload chunk lookups that returned MISS.",
        ),
        _ConnectorMetricName.LOOKUP_CHUNK_OTHER: OffloadingCounterMetadata(
            documentation=(
                "Offload chunk lookups returning a LookupResult this build does "
                "not recognise. Should stay at zero."
            ),
        ),
        _ConnectorMetricName.LOOKUP_CALLS: OffloadingCounterMetadata(
            documentation="Calls into the offload lookup path.",
        ),
        _ConnectorMetricName.LOOKUP_SKIP_SHORT_WINDOW: OffloadingCounterMetadata(
            documentation=(
                "Lookups abandoned before querying the backend because less "
                "than one chunk remained to fetch."
            ),
        ),
        _ConnectorMetricName.LOOKUP_SKIP_ZERO_HIT: OffloadingCounterMetadata(
            documentation="Lookups where the backend hit no chunks at all.",
        ),
        _ConnectorMetricName.LOOKUP_SKIP_SHORT_RESULT: OffloadingCounterMetadata(
            documentation=(
                "Lookups abandoned after querying because the confirmed hit was "
                "under one chunk."
            ),
        ),
        _ConnectorMetricName.LOOKUP_DEFERRED_BACKEND: OffloadingCounterMetadata(
            documentation=(
                "Lookups deferred because the backend asked to be retried."
            ),
        ),
        _ConnectorMetricName.LOOKUP_DEFERRED_LOADING: OffloadingCounterMetadata(
            documentation=(
                "Lookups deferred because hit chunks were already being loaded."
            ),
        ),
        _ConnectorMetricName.LOOKUP_SERVED: OffloadingCounterMetadata(
            documentation="Lookups that returned a usable external hit.",
        ),
        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),
    }''',
    "LOOKUP_CHUNK_HIT: OffloadingCounterMetadata",
    "metrics: lookup outcome counter definitions",
)

# ---------------------------------------------------------------- 3. the helper

apply(
    SCHED,
    "    def _maximal_prefix_lookup(",
    '''    def _count_lookup_result(self, result) -> None:
        """radiance: count one backend chunk lookup by outcome (R3.14.2).

        Called from both lookup sites. Never raises: an unrecognised result is
        counted as `other` rather than breaking a lookup for a metric.
        """
        name = _LOOKUP_RESULT_COUNTERS.get(
            result, _ConnectorMetricName.LOOKUP_CHUNK_OTHER
        )
        self._connector_stats.increase_counter(name)

    def _maximal_prefix_lookup(''',
    "_count_lookup_result",
    "scheduler: lookup result helper",
)

# The mapping lives next to the LookupResult import so a rename fails loudly at import
# time rather than silently counting everything as `other`.
apply(
    SCHED,
    'MATCHER_LOCALITY_KEY = "locality"',
    '''MATCHER_LOCALITY_KEY = "locality"

# radiance: LookupResult -> counter name, for _count_lookup_result (R3.14.2).
_LOOKUP_RESULT_COUNTERS = {
    LookupResult.HIT: _ConnectorMetricName.LOOKUP_CHUNK_HIT,
    LookupResult.HIT_PENDING: _ConnectorMetricName.LOOKUP_CHUNK_HIT_PENDING,
    LookupResult.RETRY: _ConnectorMetricName.LOOKUP_CHUNK_RETRY,
    LookupResult.MISS: _ConnectorMetricName.LOOKUP_CHUNK_MISS,
}''',
    "_LOOKUP_RESULT_COUNTERS = {",
    "scheduler: lookup result mapping",
)

# ---------------------------------------------------------------- 4. both lookup sites

apply(
    SCHED,
    """            result = self.manager.lookup(key, req_context)
            match result:""",
    """            result = self.manager.lookup(key, req_context)
            self._count_lookup_result(result)  # radiance R3.14.2
            match result:""",
    "self._count_lookup_result(result)",
    "scheduler: count prefix-lookup results",
)

apply(
    SCHED,
    """            match self.manager.lookup(keys[idx], req_context):""",
    """            _lookup_result = self.manager.lookup(keys[idx], req_context)
            self._count_lookup_result(_lookup_result)  # radiance R3.14.2
            match _lookup_result:""",
    "_lookup_result = self.manager.lookup(keys[idx], req_context)",
    "scheduler: count sliding-window-lookup results",
)

# ---------------------------------------------------------------- 5. terminal outcomes

apply(
    SCHED,
    "        num_computed_tokens = req_status.num_locally_computed_tokens",
    """        self._connector_stats.increase_counter(  # radiance R3.14.2
            _ConnectorMetricName.LOOKUP_CALLS
        )
        num_computed_tokens = req_status.num_locally_computed_tokens""",
    "_ConnectorMetricName.LOOKUP_CALLS",
    "scheduler: count lookup calls",
)

apply(
    SCHED,
    """                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0""",
    """                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_SKIP_SHORT_WINDOW
                    )
                    return 0""",
    "_ConnectorMetricName.LOOKUP_SKIP_SHORT_WINDOW\n",
    "scheduler: count pre-query short window",
)

apply(
    SCHED,
    """                if num_hit_chunks == 0:
                    return 0""",
    """                if num_hit_chunks == 0:
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_SKIP_ZERO_HIT
                    )
                    return 0""",
    "_ConnectorMetricName.LOOKUP_SKIP_ZERO_HIT\n",
    "scheduler: count zero-hit lookups",
)

apply(
    SCHED,
    """                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0""",
    """                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_SKIP_SHORT_RESULT
                    )
                    return 0""",
    "_ConnectorMetricName.LOOKUP_SKIP_SHORT_RESULT\n",
    "scheduler: count post-query short result",
)

apply(
    SCHED,
    """            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None""",
    """            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            self._connector_stats.increase_counter(  # radiance R3.14.2
                _ConnectorMetricName.LOOKUP_DEFERRED_BACKEND
            )
            return None""",
    "_ConnectorMetricName.LOOKUP_DEFERRED_BACKEND\n",
    "scheduler: count backend-deferred lookups",
)

apply(
    SCHED,
    """                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None""",
    """                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_DEFERRED_LOADING
                    )
                    return None""",
    "_ConnectorMetricName.LOOKUP_DEFERRED_LOADING\n",
    "scheduler: count load-deferred lookups",
)

apply(
    SCHED,
    """        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens""",
    """        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        self._connector_stats.increase_counter(  # radiance R3.14.2
            _ConnectorMetricName.LOOKUP_SERVED
        )
        self._connector_stats.increase_counter(  # radiance R3.14.2
            _ConnectorMetricName.LOOKUP_SERVED_TOKENS, num_hit_tokens
        )
        return num_hit_tokens""",
    "_ConnectorMetricName.LOOKUP_SERVED\n",
    "scheduler: count served lookups",
)
