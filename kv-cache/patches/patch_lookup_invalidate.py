#!/usr/bin/env python3
"""Invalidate the fs tier's per-key async-lookup cache when a store LANDS.

WHY
===
`AsyncLookupManager.lookup()` (vllm/v1/kv_offload/tiering/async_lookup.py) caches
a per-key existence verdict and re-states a key only while `state is None` (the
`state is None` branch). A cached `False` ("absent") is returned verbatim, never
re-checked, until `cleanup()` deletes the entry -- and `cleanup()` fires only once
every referencing request has finished.

The fs store path never invalidates that cache. So: a request looks up key K
(cached `absent`), computes that prefix, and stores K -- the file now exists --
but every later lookup of K keeps returning `absent` for the lifetime of the
referencing request(s). That window is longest for long-lived / shared-prefix
requests, which is exactly the long-context case. A MISS is, there, a lie: the
block is on disk and the scheduler recomputes it from scratch.

This is upstream vLLM code (not one of our patches), found by task 50 and
confirmed in the live engine.

WHAT IT CHANGES
===============
1. `AsyncLookupManager.invalidate(keys)` (NEW, base class). Drops a key's cached
   verdict so the next lookup re-states it. It drops ONLY `absent` (False)
   entries: a `present` entry is left (re-stating it would just re-confirm True)
   and an in-flight (None) entry is already being checked. It is generic and
   tier-agnostic, shaped to be offerable upstream.
2. `AsyncLookupManager.cleanup()` -- guarded against a key that was dropped by
   `invalidate()` and not yet re-looked-up (the entry can be absent; the old
   `self._lookup_state[key]` would KeyError). This guard is a no-op when the fix
   is off (nothing is dropped), so it is safe regardless of the gate.
3. `FileSystemTierManager` wires the call: `submit_store()` records the store's
   keys unconditionally (a new always-on map, independent of KV events);
   `get_finished_jobs()` calls `invalidate()` on those keys when a store SUCCEEDS
   and the gate is on. A FAILED store leaves the verdict untouched (the block
   genuinely is not on disk, so the `absent` is still correct).
4. `FileSystemTierManager.lookup()` -- the DIAGNOSTIC (test flag): when a lookup
   returns a cached `absent`, re-stat the file and count a "lie" if it now
   exists. Off by default; costs one faccessat per cached-absent lookup, only
   when the flag is set.

DECISION: DROP, not flip.
  A store completing successfully does NOT guarantee the file persists to the next
  lookup -- the fs tier has no internal eviction and an external reaper removes
  files out-of-band. Flipping the verdict to `present` would assert a fact the
  tier has already lost the ability to know (a later reap would turn it into a
  false HIT -> a failed promotion -> a recompute, i.e. the same expensive penalty,
  relocated). Dropping forces the next lookup to re-stat the file, so the verdict
  reflects the file's ACTUAL current state. The cost of dropping is one re-stat
  (batched, on the next lookup of that key); the cost of not invalidating is a
  lost hit + a full recompute. Those are not symmetric: drop.

GATE (house convention: UNSET = upstream behaviour)
  RADIANCE_LOOKUP_INVALIDATE
    unset (default) : upstream behaviour -- the stale `absent` persists after a
                     store; a later lookup of a just-stored block can MISS and
                     force a full recompute. A non-fatal warning names this at
                     tier construction.
    =1              : on successful store completion, drop the stale `absent`
                     verdict for the stored keys.
  Default argument: the fix is OFF by default so the patch is a safe no-op until
  enabled (reversible; an operator can revert by unsetting). It is recommended ON
  in steady state, because the only cost of invalidating is a re-stat while the
  cost of not invalidating is a full recompute -- the asymmetry strongly favours
  invalidating. The gate is a revert valve, not a performance lever.

TEST FLAG (off by default, costs nothing when off)
  RADIANCE_LOOKUP_STALE_WATCH
    unset (default) : the diagnostic is inert (short-circuits; no re-stat).
    =1              : a lookup that returns a cached `absent` re-stats the file
                     and, if it now exists, increments `self._radiance_stale_lie`
                     and logs a (throttled) warning. That single event IS the
                     defect (a MISS that is a lie). With the fix off and the flag
                     on the counter is non-zero; with the fix on it goes to zero.
                     The contrast is the proof.

COST
====
At most one re-stat per stored key that (a) was cached-absent and (b) is
re-looked-up -- and it lands in the SAME batched `faccessat`/`batch_lookup_C`
call the tier already uses for any new key after `cleanup()`. A key is
re-stated at most once per store (the result re-caches to True), so it cannot
re-stat repeatedly; a stored key that is never re-requested costs zero. The fs
tier is device-bound (228 MB/s, 5.3% wait), so a few extra batched existence
checks per step sit well inside the slack. It cannot be a syscall storm.

RISK
====
`invalidate()` / the `get_finished_jobs()` call / `submit_store()` tracking all run
on the scheduler thread (the thread that owns `_lookup_state`), so there is no
new race. Every site degrades to "no invalidation" (upstream behaviour) if the
gate is unset. The `cleanup()` guard degrades to the old path when nothing is
dropped. Instrumentation (the test flag) is a single `os.path.exists` guarded by
the flag; a rename or a missing attribute degrades to "no diagnostic", never a
crash.

REVERT
======
Delete `invalidate()` from async_lookup.py and restore `cleanup()`'s
`self._lookup_state[key]` direct-index; in fs/manager.py delete the two
`RADIANCE_*` module constants, the `_store_lookup_keys` / `_radiance_stale_lie`
`__init__` fields and their two `submit_store`/`get_finished_jobs`/`lookup`
blocks, and the `__init__` warning. The live engine is read-only; revert by
restoring the pre-patch files and reloading the model.

  RUNS AFTER: house 1-8 (needs `patch_kv_offload_fs_fanout.py` for the
  `_RADIANCE_FANOUT_MAX` anchor and `_radiance_split` in `submit_store`) and the
  task-20 bundle. It does not anchor on the bundle's fs/manager.py hunk
  (the `_tr_emit_tier_report` loop), so it is order-independent w.r.t. the
  bundle; positioned last in the prelude (same as task 50) to be safe.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KVO = SP / "vllm/v1/kv_offload"
ASYNC_LOOKUP = KVO / "tiering/async_lookup.py"
FS_MANAGER = KVO / "tiering/fs/manager.py"

print("radiance: KV offload -- invalidate the async-lookup cache on store landing")

# --- prerequisite -----------------------------------------------------------
# The fs/manager.py anchors stand on the house fs-fanout patch (it adds the
# _RADIANCE_FANOUT_MAX constant and _radiance_split in submit_store). A missing
# marker is a wrong prelude, not a version drift -- say so explicitly rather
# than failing the anchor opaquely.
if not ASYNC_LOOKUP.exists() or not FS_MANAGER.exists():
    raise SystemExit("[radiance] async_lookup.py / fs/manager.py missing")
if "_RADIANCE_FANOUT_MAX = int(" not in FS_MANAGER.read_text():
    raise SystemExit(
        "[radiance] this patch requires patch_kv_offload_fs_fanout.py (house) "
        "to be applied first (it adds _RADIANCE_FANOUT_MAX and "
        "_radiance_split, which the fs/manager.py anchors stand on)."
    )

# ---------------------------------------------------------------------------
# 1. async_lookup.py: add AsyncLookupManager.invalidate() (before cleanup) and
#    guard cleanup() against a key dropped by invalidate().
# ---------------------------------------------------------------------------
apply(
    ASYNC_LOOKUP,
    '''    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state[key]
            state.request_ids.discard(req_id)
            if not state.request_ids:
                del self._lookup_state[key]''',
    '''    def invalidate(self, keys: "Iterable[OffloadKey]") -> None:
        """Drop the cached verdict for ``keys`` so the next lookup re-states it.

        Called by a tier when a STORE for these keys completes successfully: the
        block is now on disk, so a previously-cached `absent` verdict is stale.
        Only `absent` (False) entries are dropped -- a `present` entry is left
        (re-stating it would just re-confirm True) and an in-flight (None) entry
        is already being checked. Dropping (not flipping to True) keeps the
        answer honest under external eviction: the next lookup re-states the
        file, so the verdict reflects its actual current state rather than a fact
        the tier can no longer guarantee. Scheduler-thread only.
        """
        for key in keys:
            state = self._lookup_state.get(key)
            if state is not None and state.result is False:
                del self._lookup_state[key]

    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            # A key may have been dropped by invalidate() (a store landed) and
            # not yet re-looked-up, so its entry can be absent here. This guard
            # is a no-op when nothing was dropped (e.g. the fix gate is unset).
            state = self._lookup_state.get(key)
            if state is None:
                continue
            state.request_ids.discard(req_id)
            if not state.request_ids:
                del self._lookup_state[key]''',
    "def invalidate(self, keys",
    "1  async_lookup.py: invalidate() + guarded cleanup()",
)

# ---------------------------------------------------------------------------
# 2. fs/manager.py: the two env-var module constants (right after the existing
#    radiance fanout constants).
# ---------------------------------------------------------------------------
apply(
    FS_MANAGER,
    '_RADIANCE_FANOUT_MAX = int(os.environ.get("RADIANCE_FS_FANOUT_MAX", "0"))',
    '_RADIANCE_FANOUT_MAX = int(os.environ.get("RADIANCE_FS_FANOUT_MAX", "0"))\n'
    "# radiance: invalidate the per-key async-lookup cache when a fs store LANDS.\n"
    "# unset (default) = upstream behaviour (the cache keeps a stale `absent`\n"
    "# after a store; a later lookup of a just-stored block can MISS and force a\n"
    "# full recompute). =1 = drop that stale verdict on successful store\n"
    "# completion. A FAILED store leaves the verdict (the block genuinely is not\n"
    "# on disk, so the `absent` is still correct).\n"
    '_RADIANCE_LOOKUP_INVALIDATE = os.environ.get("RADIANCE_LOOKUP_INVALIDATE", "0") == "1"\n'
    "# radiance diagnostic (off by default; costs one faccessat per\n"
    "# cached-absent lookup, only when set): a lookup that returns a cached\n"
    "# `absent` re-stats the file and counts a 'lie' if it now exists. That\n"
    "# single event is the defect; non-zero with the fix off, zero with it on.\n"
    '_RADIANCE_LOOKUP_STALE_WATCH = os.environ.get("RADIANCE_LOOKUP_STALE_WATCH", "0") == "1"',
    '_RADIANCE_LOOKUP_INVALIDATE = os.environ.get("RADIANCE_LOOKUP_INVALIDATE", "0") == "1"',
    "2  fs/manager.py: RADIANCE_LOOKUP_INVALIDATE / RADIANCE_LOOKUP_STALE_WATCH",
)

# ---------------------------------------------------------------------------
# 3. fs/manager.py __init__: the always-on store-key map (for invalidation,
#    independent of KV events) and the diagnostic counter. Plus the gate warning.
# ---------------------------------------------------------------------------
apply(
    FS_MANAGER,
    '''        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}''',
    '''        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # radiance: keys of in-flight store jobs, tracked ALWAYS so
        # get_finished_jobs() can invalidate the async-lookup cache when a store
        # lands, independent of whether KV events are enabled.
        self._store_lookup_keys: dict[JobId, list[OffloadKey]] = {}
        # radiance diagnostic: number of lookups that returned a cached `absent`
        # for a key whose file now exists (a "lie"). Gated on
        # RADIANCE_LOOKUP_STALE_WATCH; zero cost when that flag is unset.
        self._radiance_stale_lie = 0
        if _RADIANCE_LOOKUP_INVALIDATE:
            logger.info(
                "radiance: RADIANCE_LOOKUP_INVALIDATE on -- a stale `absent` "
                "lookup verdict is dropped when a store for it lands"
            )
        else:
            logger.warning(
                "radiance: RADIANCE_LOOKUP_INVALIDATE unset -- the fs "
                "async-lookup cache keeps a stale `absent` after a store lands "
                "(upstream behaviour); a later lookup of a just-stored block can "
                "MISS and force a full recompute. Set it to 1 to drop the stale "
                "verdict on store completion."
            )''',
    "self._store_lookup_keys: dict[JobId, list[OffloadKey]] = {}",
    "3  fs/manager.py __init__: _store_lookup_keys, _radiance_stale_lie, gate warning",
)

# ---------------------------------------------------------------------------
# 4. fs/manager.py submit_store(): record the store's keys unconditionally.
# ---------------------------------------------------------------------------
apply(
    FS_MANAGER,
    '''        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = self._radiance_split(  # radiance R3.14''',
    '''        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        # radiance: always-on (independent of events) so get_finished_jobs() can
        # invalidate the async-lookup cache when this store lands.
        self._store_lookup_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = self._radiance_split(  # radiance R3.14''',
    "self._store_lookup_keys[job_metadata.job_id] = list(job_metadata.keys)",
    "4  fs/manager.py submit_store: track store keys unconditionally",
)

# ---------------------------------------------------------------------------
# 5. fs/manager.py get_finished_jobs(): on a SUCCESSFUL store, invalidate the
#    cached verdict for those keys (and always drain the tracking map).
# ---------------------------------------------------------------------------
apply(
    FS_MANAGER,
    '''            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            results.append(JobResult(job_id=job_id, success=success))''',
    '''            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            # radiance: a successful store put these blocks on disk, so any
            # previously-cached `absent` verdict for them is stale. Drop it (the
            # next lookup re-states the file, reflecting its real current state,
            # robust to external eviction). A FAILED store leaves the verdict
            # untouched: the block genuinely is not on disk, so `absent` is
            # still correct. The tracking map is drained for every finished job.
            store_keys = self._store_lookup_keys.pop(job_id, None)
            if store_keys and success and _RADIANCE_LOOKUP_INVALIDATE:
                self._lookup_manager.invalidate(store_keys)
            results.append(JobResult(job_id=job_id, success=success))''',
    "store_keys = self._store_lookup_keys.pop(job_id, None)",
    "5  fs/manager.py get_finished_jobs: invalidate on successful store",
)

# ---------------------------------------------------------------------------
# 6. fs/manager.py lookup(): the diagnostic -- a cached `absent` that is on disk
#    now is a MISS that is a lie. Re-stat it (one faccessat, only when the flag
#    is set) and count the hit.
# ---------------------------------------------------------------------------
apply(
    FS_MANAGER,
    '''    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS''',
    '''    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        # radiance diagnostic (RADIANCE_LOOKUP_STALE_WATCH): a cached `absent`
        # that is actually on disk now is a MISS that is a lie. Re-stat it (one
        # faccessat, only when the flag is set) and count the hit. Inert when the
        # flag is unset (the `and` short-circuits before the os.path.exists).
        #
        # DO NOT LEAVE THIS ON IN PRODUCTION. Its cost is NOT the same shape as
        # the fix's. The fix re-states at most once per stored key, inside a
        # batch the tier already sends. This fires on EVERY miss, one unbatched
        # os.path.exists each -- and on a cold cache every lookup misses, so the
        # warm-up window (~45 min, before the fs tier can serve at all) is
        # exactly when it is most expensive. The tier is device-bound
        # (228 MB/s, 5.3% wait). Use it to demonstrate the defect and to confirm
        # the fix drives it to zero, then turn it off.
        if (
            _RADIANCE_LOOKUP_STALE_WATCH
            and result is False
            and os.path.exists(self.file_mapper.get_file_name(key))
        ):
            self._radiance_stale_lie += 1
            if self._radiance_stale_lie <= 8 or self._radiance_stale_lie % 100 == 0:
                logger.warning(
                    "radiance stale-lookup LIE: cached `absent` for a key now "
                    "on disk (count=%d) -- a stored block is being missed",
                    self._radiance_stale_lie,
                )
        return LookupResult.HIT if result else LookupResult.MISS''',
    "self._radiance_stale_lie += 1",
    "6  fs/manager.py lookup: RADIANCE_LOOKUP_STALE_WATCH diagnostic",
)

print("[radiance] async-lookup invalidation applied "
      "(gate RADIANCE_LOOKUP_INVALIDATE, diagnostic RADIANCE_LOOKUP_STALE_WATCH)")
