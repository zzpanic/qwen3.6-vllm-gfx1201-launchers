#!/usr/bin/env python3
"""Phase B of the closeout plan: stop throwing away hits we have already found.
Behaviour change, gated on one environment variable.

WHY THIS EXISTS (cache-preemption-patch-plan.md R3.14.3):

Phase A measured the lookup path over 49 minutes of real production traffic and the
answer was not close. Of 11,878 lookups, 11,849 -- 99.75% -- abandoned the prefix and
returned None, and only 3 served. The per-chunk histogram named the trigger: HIT_PENDING
1,579,971 against RETRY 3,869, a ratio of 408 to 1. So the mechanism is not the retry
livelock the plan had guessed at. It is this:

  HIT_PENDING means the block IS in the cache. cpu/manager.py found it. Its ref_cnt is
  still -1, which policies/base.py defines as "the write has not landed yet". Upstream
  treats that as a reason to WAIT: both lookup loops set defer_lookup, and then
  `return hit_count if not defer_lookup else None` discards a hit count they had already
  earned -- 575,335 confirmed HITs across the measurement window -- so that the request
  can be re-looked-up a step later in the hope of a longer prefix.

  That trade is sound on a quiet machine and catastrophic on a busy one. Waiting only pays
  if the pending write lands before the next lookup and nothing else starts writing. This
  server stored 61 GB in one boot; there is no quiet instant to wait for. Of the requests
  that ever escaped the wait, the mean escape took 0.94 s -- and 25 escaped in 49 minutes.

  It also explains why every bench hit and production never did. A bench runs on an idle
  box: no in-flight stores, no HIT_PENDING, no defer, clean serve. The bench and production
  were never measuring the same system.

WHAT IT CHANGES

  A HIT_PENDING chunk stops meaning "wait for this" and starts meaning "this one is not
  readable yet" -- exactly what MISS already means to both loops. Nothing else moves.

  1. scheduler.py: the _RADIANCE_PENDING_IS_MISS flag, read once at import beside the
     mixed-hit flag it sits next to, so the whole change is one env var to revert.

  2. scheduler.py, _maximal_prefix_lookup (the two full-attention groups and the MTP
     group): HIT_PENDING ends the prefix scan instead of extending it and deferring. The
     function returns the confirmed-ready prefix.

  3. scheduler.py, _sliding_window_lookup (the six Mamba/GDN groups, window = 1 chunk):
     HIT_PENDING resets the consecutive-hit run instead of extending it and deferring, so
     the backwards scan keeps going and finds an OLDER window that is ready. A Mamba group
     needs one specific state, so truncating is the only honest answer -- but an older
     state is still a real hit, and today we take neither.

  4+5. metrics.py: one counter, ..._lookup_pending_truncated, so the effect is measurable
     rather than assumed. It counts chunks this patch declined; against the Phase A
     counters it says directly whether the truncation bought serves.

WHY THIS CANNOT SERVE WRONG DATA

  A pending chunk is never claimed. Both loops end up reporting FEWER chunks than upstream
  would, never more, and every chunk they do report returned HIT -- ready, ref_cnt >= 0,
  the same state the existing serve path already requires. Upstream never serves a pending
  chunk either; it waits for it. So this changes how much of a hit we take, not whether the
  bytes are valid. R3.11 established those bytes are bit-identical to a cold recompute.

  The floor is unchanged: _lookup still returns 0 when the truncated result is under one
  1,648-token chunk, so a short truncation costs a load that was not worth doing.

WHAT WOULD NOT WORK, AND WHY IT IS NOT HERE

  tiering/manager.py:321 short-circuits on a primary HIT_PENDING and never asks the disk
  tier, which looks like a second bug worth fixing. It is not worth fixing. A secondary-tier
  HIT does not serve the block: it starts a promotion into the primary tier and returns
  RETRY. The block is already inbound to the primary, so consulting disk would schedule a
  redundant copy of a write already in flight. Leave it alone.

REVERTING: set RADIANCE_OFFLOAD_PENDING_IS_MISS=0 and restart. The patch stays applied and
every branch reverts to upstream behaviour, which makes this an A/B rather than a one-way
door.

RISK: moderate, and confined. It changes what a lookup reports, so it can change which
requests take an external hit and how long that hit is. It cannot change the content of a
hit, cannot make a lookup claim an unready block, and cannot raise from inside the lookup.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])

OFFL = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading"
METRICS = OFFL / "metrics.py"
SCHED = OFFL / "scheduler.py"

# ---------------------------------------------------------------- 1. the flag

apply(
    SCHED,
    '''_RADIANCE_ALLOW_MIXED_HIT = os.environ.get("RADIANCE_OFFLOAD_MIXED_HIT", "0") == "1"''',
    '''_RADIANCE_ALLOW_MIXED_HIT = os.environ.get("RADIANCE_OFFLOAD_MIXED_HIT", "0") == "1"
# radiance R3.14.3: treat a HIT_PENDING chunk as the end of the ready prefix rather than as
# a reason to defer the whole request. Phase A measured 99.75% of production lookups
# deferring on HIT_PENDING at 408:1 over RETRY, and 3 serves in 49 minutes. Set to 0 to get
# upstream's wait-for-it behaviour back without unpatching anything.
_RADIANCE_PENDING_IS_MISS = (
    os.environ.get("RADIANCE_OFFLOAD_PENDING_IS_MISS", "1") == "1"
)''',
    "_RADIANCE_PENDING_IS_MISS",
    "scheduler: pending-is-miss flag",
)

# ---------------------------------------------------------------- 2. prefix lookup
# Full-attention and MTP groups. `break` here leaves the enclosing `for`, exactly as the
# MISS arm below it already does; `match` is not a loop, so this is the same exit.

apply(
    SCHED,
    '''                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1''',
    '''                case LookupResult.HIT_PENDING:
                    if _RADIANCE_PENDING_IS_MISS:
                        # radiance R3.14.3: the block is cached, but its store has not
                        # landed. Upstream waits; production never gets a quiet instant
                        # to wait for, so end the ready prefix here and serve it.
                        self._connector_stats.increase_counter(
                            _ConnectorMetricName.LOOKUP_PENDING_TRUNCATED
                        )
                        break
                    defer_lookup = True
                    hit_count += 1''',
    "radiance R3.14.3: the block is cached, but its store has not",
    "scheduler: prefix lookup takes the ready prefix",
)

# ---------------------------------------------------------------- 3. sliding-window lookup
# The six Mamba/GDN groups, window = 1 chunk. Resetting the run rather than breaking lets
# the backwards scan reach an older state that IS ready -- a shorter hit, but a real one.

apply(
    SCHED,
    '''                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1''',
    '''                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    if _RADIANCE_PENDING_IS_MISS:
                        # radiance R3.14.3: not readable now, so it cannot be part of
                        # this window. Keep scanning backwards for an older window that
                        # is ready rather than deferring the whole request.
                        self._connector_stats.increase_counter(
                            _ConnectorMetricName.LOOKUP_PENDING_TRUNCATED
                        )
                        consecutive_hits = 0
                    else:
                        defer_lookup = True
                        consecutive_hits += 1''',
    "radiance R3.14.3: not readable now, so it cannot be part of",
    "scheduler: sliding-window lookup falls back to an older ready state",
)

# ---------------------------------------------------------------- 4. metric name

apply(
    METRICS,
    '''    LOOKUP_SERVED_TOKENS = "vllm:kv_offload_lookup_served_tokens"''',
    '''    LOOKUP_SERVED_TOKENS = "vllm:kv_offload_lookup_served_tokens"
    # radiance R3.14.3: chunks declined because their store had not landed.
    LOOKUP_PENDING_TRUNCATED = "vllm:kv_offload_lookup_pending_truncated"''',
    "LOOKUP_PENDING_TRUNCATED",
    "metrics: pending-truncated counter name",
)

# ---------------------------------------------------------------- 5. metric definition

apply(
    METRICS,
    '''        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),
    }''',
    '''        _ConnectorMetricName.LOOKUP_PENDING_TRUNCATED: OffloadingCounterMetadata(
            documentation=(
                "Offload chunks declined because their store had not landed "
                "(radiance R3.14.3). Each one shortened a hit that upstream "
                "would have deferred outright."
            ),
        ),
        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),
    }''',
    "LOOKUP_PENDING_TRUNCATED: OffloadingCounterMetadata",
    "metrics: pending-truncated counter definition",
)
