#!/usr/bin/env python3
"""Forget the fs tier's cached lookup verdict for keys whose LOAD failed.

WHY
===
`AsyncLookupManager` (vllm/v1/kv_offload/tiering/async_lookup.py) caches one existence verdict
per key and keeps it while any request that looked the key up is still alive. When a promotion
(fs -> CPU) fails -- the file was reaped after the lookup stat-ed it, came up short, or hit an
I/O error -- `fs/io.py` deletes the file and the job reports failure; the tiering manager drops
the half-written CPU block (`cpu/manager.py complete_store(success=False)`). Nothing tells the
lookup cache. The request's next lookup gets the cached `present`, promotes again, the read
fails again (the file is gone), and so on until the request finishes -- which it cannot, because
it is waiting for that block. EngineCore does not crash; the request hangs until the client gives
up.

Measured 2026-09-19 18:06 (bow-20260919/tools/badfile_test.py): one g6 block of a 24k-token
prefix truncated to 4 KiB after a restart (RAM tier empty, disk the only copy). The request never
completed: 240 s client timeout, 294 failed read jobs (1 EIO, then ENOENT ~1.2/s), loop stopped
only when the client disconnected. The trigger in production is the reaper (kvcache-reap.service)
deleting a block between a request's lookup and its promotion -- rare, but storms make that
window seconds long.

WHAT IT CHANGES
===============
1. `AsyncLookupManager.forget(keys)` (NEW): drop the cached verdict for these keys whatever it
   says (except in-flight None, already being re-checked). Unlike `invalidate()`, which only drops
   a stale `absent` after a store lands, this drops a `present` that a failed read has disproved.
   The next lookup re-states the file: gone -> MISS -> the request recomputes that block.
2. `FileSystemTierManager.submit_load()` records the job's keys; `get_finished_jobs()` forgets
   them when the load FAILED and the gate is on. The map is drained for every finished job.

GATE (house convention: UNSET = upstream behaviour)
  RADIANCE_FS_FAILED_LOAD_FORGET  unset/0 = upstream (the loop), 1 = forget on failed load.
  The launcher defaults it to 1 (KVOFF_FS_FAILED_LOAD_FORGET).

COST: nothing on the success path (one dict insert + pop per load job). On a failed load, one
re-stat of the failed keys on their next lookup, batched like any other lookup.

RISK: scheduler thread only (the thread that owns _lookup_state and runs get_finished_jobs), so
no new race. A missing anchor is fatal to this patch only; the prelude treats it as non-fatal
and names the consequence.

RUNS AFTER patch_kv_offload_fs_fanout.py (anchors on its `_radiance_split` in submit_load and its
`_RADIANCE_FANOUT_MAX` constant) and patch_kv_offload_lookup_cache_invalidate.py (anchors after
its `invalidate()`).

REVERT: unset the gate (behaviour), or drop the prelude line and reload (code).
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KVO = SP / "vllm/v1/kv_offload"
ASYNC_LOOKUP = KVO / "tiering/async_lookup.py"
FS_MANAGER = KVO / "tiering/fs/manager.py"

print("radiance: KV offload -- forget the lookup verdict when an fs load fails")

s = FS_MANAGER.read_text() if FS_MANAGER.exists() else ""
if "_RADIANCE_FANOUT_MAX = int(" not in s:
    raise SystemExit("[radiance] needs patch_kv_offload_fs_fanout.py applied first")
if "def invalidate(self, keys" not in (ASYNC_LOOKUP.read_text() if ASYNC_LOOKUP.exists() else ""):
    raise SystemExit("[radiance] needs patch_kv_offload_lookup_cache_invalidate.py applied first")

# 1. async_lookup.py: forget() right before cleanup().
apply(
    ASYNC_LOOKUP,
    '''    def cleanup(self, req_id: str) -> None:''',
    '''    def forget(self, keys: "Iterable[OffloadKey]") -> None:
        """Drop the cached verdict for ``keys`` whatever it is (radiance).

        Called by a tier when a LOAD of these keys FAILED: the cached `present`
        has been disproved (the file was reaped, short, or unreadable, and the
        tier deleted it). Left in place it makes the waiting request promote the
        same missing file forever. An in-flight (None) entry is left -- it is
        already being re-stated. Scheduler-thread only.
        """
        for key in keys:
            state = self._lookup_state.get(key)
            if state is not None and state.result is not None:
                del self._lookup_state[key]

    def cleanup(self, req_id: str) -> None:''',
    "def forget(self, keys",
    "1  async_lookup.py: forget()",
)

# 2. fs/manager.py: the gate constant.
apply(
    FS_MANAGER,
    '_RADIANCE_FANOUT_MAX = int(os.environ.get("RADIANCE_FS_FANOUT_MAX", "0"))',
    '_RADIANCE_FANOUT_MAX = int(os.environ.get("RADIANCE_FS_FANOUT_MAX", "0"))\n'
    '# radiance: forget the lookup verdict of keys whose fs load failed (see\n'
    '# patch_kv_offload_fs_failed_load.py). Unset/0 = upstream behaviour.\n'
    '_RADIANCE_FAILED_LOAD_FORGET = os.environ.get("RADIANCE_FS_FAILED_LOAD_FORGET", "0") == "1"',
    "_RADIANCE_FAILED_LOAD_FORGET =",
    "2  fs/manager.py: gate constant",
)

# 3. fs/manager.py: the per-job key map, next to the stock store-events map.
apply(
    FS_MANAGER,
    "        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}\n",
    "        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}\n"
    "        # radiance: load job -> keys, so a FAILED load can forget its lookup verdicts.\n"
    "        self._radiance_load_keys: dict[JobId, list[OffloadKey]] = {}\n",
    "self._radiance_load_keys: dict[JobId",
    "3  fs/manager.py: load key map",
)

# 4. fs/manager.py: record the keys at submit_load.
apply(
    FS_MANAGER,
    '''    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = self._radiance_split(  # radiance R3.14''',
    '''    def submit_load(self, job_metadata: JobMetadata) -> None:
        self._radiance_load_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = self._radiance_split(  # radiance R3.14''',
    "self._radiance_load_keys[job_metadata.job_id] =",
    "4  fs/manager.py: submit_load records keys",
)

# 5. fs/manager.py: forget on a failed load, drain on every finished job.
apply(
    FS_MANAGER,
    '''            results.append(JobResult(job_id=job_id, success=success))
        return results''',
    '''            load_keys = self._radiance_load_keys.pop(job_id, None)
            if load_keys and not success and _RADIANCE_FAILED_LOAD_FORGET:
                self._lookup_manager.forget(load_keys)
            results.append(JobResult(job_id=job_id, success=success))
        return results''',
    "self._lookup_manager.forget(load_keys)",
    "5  fs/manager.py: forget on failed load",
)
