#!/usr/bin/env python3
"""Attribute each offload `_lookup` miss to a reason -- task-65 revision of the
staged out/59/patch_kv_offload_miss_reason.py.

WHY THIS REVISED VERSION EXISTS
==============================
Two problems with the staged 59, found by task 65:

1. **It collides with the staged 50 patch in `metrics.py`** and the two cannot
   be applied in either order. Both insert their counter names right after the
   `LOOKUP_SERVED_TOKENS` name line, and both insert their definitions right
   after the `LOOKUP_SERVED_TOKENS` definition / before the `# radiance
   tier-report` comment. 50's name-line insertion breaks 59's `class
   _TransferType` anchor; 59's definition insertion breaks 50's tier-report
   comment anchor. Verified on a copy of the live tree: 50->59 fails 59 hunk 1,
   59->50 fails 50 hunk 2.

2. **It ships two series it cannot populate** -- the `*_mandatory` reason. 59's
   own GATE section says `mandatory` "reads 0 until a tier surfaces a
   correctness-forced-miss signal"; no tier in the live tree sets
   `req_status.mandatory_miss`, so both `*_mandatory` counters are never
   incremented, never exported (the absent-counter trap: absent == 0), and add a
   `mandatory_miss` dataclass field + a branch per miss for nothing. That
   violates 59's own stated standard ("refusing to ship labels it could not
   populate").

WHAT THIS REVISED VERSION DELIVERS
==================================
  zero_hit      -> { unresolved }
  short_result  -> { unresolved }

The `mandatory` reason is DROPPED (re-add it when a tier surfaces the
correctness-forced-miss signal). The `defect` reason (a stale `absent`, task 58)
stays referenced, not duplicated, exactly as before.

+2 series (was +4): `zero_hit_unresolved`, `short_result_unresolved`. Both
unlabeled (incremented with no labelvalues, so they read 0 when never incremented).
Fixed small enum, no unbounded label.

RE-ANCHORING (order-independent w.r.t. 50)
=========================================
  names -> inserted just before `class _TransferType:` (end of the
           `_ConnectorMetricName` table). 50 only inserts after the
           `LOOKUP_SERVED_TOKENS` name line and preserves the two blank lines
           before `class _TransferType:`, so this slot is present before and
           after 50, in either order.
  defs  -> inserted just above the
           `_ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(`
           definition. 50's hunk-2 anchor preserves that definition line, so this
           slot is present before and after 50, in either order.
Neither 50 nor this patch now touches the same location, so they apply in either
order. (The scheduler `mandatory_miss` field is gone, so there is no
`finished_signaled`-region collision either.)

COST (hot path)
===============
One counter increment per `zero_hit` / `short_result` miss call, NO syscall. No
`mandatory_miss` field, no per-miss branch. Nothing adds a `stat` to the lookup
path. Engine behaviour unchanged; only counters are added.

RECONCILIATION (checked by inv_lookup_subpartition, the task-65 kvvalidate patch)
================================================================================
  top  (existing): lookup_calls == short_window + zero_hit + short_result
                           + deferred_backend + deferred_loading + served
  sub  (new):       zero_hit      == zero_hit_unresolved
                    short_result  == short_result_unresolved
The parent and child increment back-to-back in the same `_lookup` call, so a
non-atomic scrape can only drift by the in-flight lookups (bounded by active
work, exactly like the top partition).

REVERT
======
Delete the two `LOOKUP_MISS_*_UNRESOLVED` name constants and their two metadata
entries from metrics.py; delete the two reason-attribution increments
(zero_hit / short_result) from scheduler.py. The live engine is read-only;
revert by restoring the pre-patch files.

  RUNS AFTER: the house KV-offload prelude (needs `lookup_outcomes` for the six
  terminal counters) + the task-20 bundle + task 58 (currently last). Order-
  independent w.r.t. the staged 50 patch (see RE-ANCHORING).
"""

import os
import sysconfig
from pathlib import Path

from _patchlib import apply

# Default: the live tree. Overridable for a throwaway rehearsal / verification
# against a tree copy (no engine touched).
SP = Path(os.environ.get(
    "RADIANCE_KVOFFLOAD_DIR",
    sysconfig.get_paths()["purelib"]
    + "/vllm/distributed/kv_transfer/kv_connector/v1/offloading",
))
METRICS = SP / "metrics.py"
SCHED = SP / "scheduler.py"

print("radiance: KV offload -- attribute each _lookup miss to a reason (task-65 v2: 2 live series, re-anchored to coexist with 50)")

if not METRICS.exists() or not SCHED.exists():
    raise SystemExit(f"[radiance] offloading metrics.py / scheduler.py missing ({SP})")

# ---------------------------------------------------------------------------
# 1. metrics.py: the two reason sub-counter NAMES, before `class _TransferType`.
# ---------------------------------------------------------------------------
apply(
    METRICS,
    """
class _TransferType:""",
    """
    LOOKUP_MISS_ZERO_UNRESOLVED = "vllm:kv_offload_lookup_zero_hit_unresolved"
    LOOKUP_MISS_SHORT_UNRESOLVED = "vllm:kv_offload_lookup_short_result_unresolved"


class _TransferType:""",
    'LOOKUP_MISS_ZERO_UNRESOLVED = "vllm:kv_offload_lookup_zero_hit_unresolved"',
    "1  metrics.py: reason sub-counter names (before class _TransferType)",
)

# ---------------------------------------------------------------------------
# 2. metrics.py: the two reason sub-counter METADATA, above the
#    LOOKUP_SERVED_TOKENS definition (the slot 50 preserves, so order-independent).
# ---------------------------------------------------------------------------
apply(
    METRICS,
    """        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(""",
    """        # radiance R3.14.3: reason sub-counters. Each sums to its parent
        # (zero_hit / short_result); kvvalidate checks the sub-partition.
        # `unresolved` is the 0-cost default (a miss whose fine cause is not
        # determined: never-stored / evicted / prefix-changed / sub-floor).
        # The `mandatory` reason is deferred until a tier surfaces a
        # correctness-forced-miss signal; the `defect` reason is the task-58
        # tier lie counter, referenced not duplicated.
        _ConnectorMetricName.LOOKUP_MISS_ZERO_UNRESOLVED: OffloadingCounterMetadata(
            documentation=(
                "zero_hit misses attributed to an undetermined cause "
                "(never-stored / evicted / prefix-changed / sub-floor). "
                "Sums to ...skip_zero_hit."
            ),
        ),
        _ConnectorMetricName.LOOKUP_MISS_SHORT_UNRESOLVED: OffloadingCounterMetadata(
            documentation=(
                "short_result misses attributed to an undetermined cause "
                "(evicted tail / prefix-changed / sub-floor). Sums to "
                "...skip_short_result."
            ),
        ),
        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(""",
    "LOOKUP_MISS_ZERO_UNRESOLVED: OffloadingCounterMetadata(",
    "2  metrics.py: reason sub-counter metadata (above served_tokens def)",
)

# ---------------------------------------------------------------------------
# 3. scheduler.py: attribute the zero_hit miss (0-cost default, no branch).
# ---------------------------------------------------------------------------
apply(
    SCHED,
    """                if num_hit_chunks == 0:
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_SKIP_ZERO_HIT
                    )
                    return 0""",
    """                if num_hit_chunks == 0:
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_SKIP_ZERO_HIT
                    )
                    # radiance R3.14.3: attribute the miss (0-cost default:
                    # unresolved -- a miss whose fine cause is not determined at
                    # the lookup site; never-stored / evicted / prefix-changed /
                    # sub-floor are all an absent file to a single stat). The
                    # `mandatory` reason is deferred until a tier surfaces it;
                    # the `defect` reason (stale `absent`) is the task-58 tier
                    # lie counter, referenced not duplicated.
                    self._connector_stats.increase_counter(
                        _ConnectorMetricName.LOOKUP_MISS_ZERO_UNRESOLVED
                    )
                    return 0""",
    "_ConnectorMetricName.LOOKUP_MISS_ZERO_UNRESOLVED",
    "3  scheduler.py: attribute the zero_hit miss",
)

# ---------------------------------------------------------------------------
# 4. scheduler.py: attribute the short_result miss (0-cost default, no branch).
# ---------------------------------------------------------------------------
apply(
    SCHED,
    """                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_SKIP_SHORT_RESULT
                    )
                    return 0""",
    """                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    self._connector_stats.increase_counter(  # radiance R3.14.2
                        _ConnectorMetricName.LOOKUP_SKIP_SHORT_RESULT
                    )
                    # radiance R3.14.3: attribute the miss (0-cost default:
                    # unresolved).
                    self._connector_stats.increase_counter(
                        _ConnectorMetricName.LOOKUP_MISS_SHORT_UNRESOLVED
                    )
                    return 0""",
    "_ConnectorMetricName.LOOKUP_MISS_SHORT_UNRESOLVED",
    "4  scheduler.py: attribute the short_result miss",
)

print("[radiance] miss-reason (task-65 v2) applied "
      "(zero_hit / short_result -> unresolved; +2 live series; "
      "mandatory deferred; defect = task-58 tier lie counter, referenced not duplicated)")
