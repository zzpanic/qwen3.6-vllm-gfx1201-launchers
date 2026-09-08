#!/usr/bin/env python3
"""R3.14 -- fan one fs-tier job out across the thread pool.

This is the port of upstream vLLM PR #49225 ("[KV offload] Batch FS tier read/write
tasks across threads"). The PR itself does NOT apply to our tree -- 44 of its 56
removed lines in manager.py and 10 of 29 in thread_pool.py are not present in the
installed files, because its base carries two later refactors we do not have
(per-task transfer timing in JobState, and per-key load-failure attribution via
TransferJob/mark_miss). What follows is its mechanism, re-expressed against the
code we actually run.

THE PROBLEM

The fs tier is configured with 8 read threads and 4 write threads. It never uses
them. Both submit paths end in the same line:

    self._pool.enqueue_store(job_metadata.job_id, 1, [task])
    self._pool.enqueue_load(job_metadata.job_id, 1, [task])

n_tasks = 1. One job becomes one task, one task is picked up by one worker, and
that worker calls batch_store_block / batch_load_block, which is a serial loop
over every block file in the job. The C extension does not rescue this: the
symbol table of vllm/fs_io_C.abi3.so has no pthread, no io_uring and no aio --
only PyEval_SaveThread / PyEval_RestoreThread. It releases the GIL and then reads
the files one after another.

So a job of N blocks runs at queue depth 1 regardless of how many threads the
tier was given. That is where the 64-second fs->CPU promotion comes from: a
multi-chunk promotion opens and reads ~200 files of 27,000,832 bytes each, in
sequence, on a single thread. The device is never asked for more than one read
at a time, so it cannot be the thing that is slow.

WHAT THIS CHANGES

Split the job into several partials, each covering a slice of the block list, and
hand all of them to the pool at once. Nothing in thread_pool.py changes, because
the pool ALREADY supports many tasks per job and the manager simply never used
it: JobState(job_id, n_tasks) counts n_tasks completions, publishes the job once
when the last one lands, and decrements _inflight_jobs once. The cascade-backlog
gauge (FS_INFLIGHT_JOBS) therefore keeps counting jobs, not tasks, and drain_jobs
/ wait_idle keep working unchanged.

HOW MANY BATCHES -- and why our default is not upstream's

Upstream picks the batch count from a byte budget: enough bytes to saturate the
device in one call, then no more. Its constant is 32 MiB, divided by the block
size, capped by the thread count, and divided by the number of jobs already in
flight so that concurrent jobs share the pool fairly.

That constant assumes small blocks. Our block IS 27,000,832 bytes, so
ceil(32 MiB / 27 MB) = 2: upstream's heuristic would conclude that one of our
blocks already saturates the device and would give us a fanout of 2, or 1 as soon
as a second job is in flight. The measured promotion rate says otherwise -- 64 s
for a multi-GB read is far under what this disk sustains, which is the signature
of queue depth 1, not of a saturated device.

So the budget is a knob. RADIANCE_FS_FANOUT_TARGET_MB defaults to 32 here (the
upstream value, so the patch alone changes almost nothing); the launcher raises
it. RADIANCE_FS_FANOUT_MAX overrides the byte budget with a flat cap. Setting
RADIANCE_FS_FANOUT_MAX=1 restores exactly today's behaviour.

SAFETY

- Failure semantics are unchanged. JobState ORs the success flags, so if any
  batch raises, the job is reported failed exactly as a whole-job failure is
  today. A batch that fails still removes its own offending file inside
  _load_block, as before.
- Parallel writers do not collide: _get_tmp_suffix() is thread-local, so each
  batch builds its own temp names, and each file is still published by an atomic
  os.replace.
- A job of 0 or 1 blocks takes the old single-partial path byte for byte. That
  matters: returning an empty task list for an empty job would create a JobState
  that never completes and would hang drain_jobs.
- _inflight_jobs is read without the pool lock. That is deliberate and already
  precedented by the radiance get_stats() gauge -- the value only picks a batch
  count, so a stale read costs nothing.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _patchlib import apply  # noqa: E402

VLLM = Path(sys.prefix) / "lib" / f"python3.{sys.version_info.minor}" / "site-packages" / "vllm"
if not VLLM.exists():
    import vllm as _v
    VLLM = Path(_v.__file__).parent

MGR = VLLM / "v1" / "kv_offload" / "tiering" / "fs" / "manager.py"

print("[radiance] R3.14 fs tier job fanout (upstream PR #49225)")

# --- 1. the knobs and the two helpers ---------------------------------------------------
apply(
    MGR,
    anchor="from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool",
    new='''from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# radiance R3.14 (upstream PR #49225): how far to fan a single fs job out across the
# pool. TARGET_MB is the byte budget that decides the batch count; MAX overrides it
# with a flat cap (MAX=1 restores the old one-task-per-job behaviour exactly).
_RADIANCE_FANOUT_TARGET_BYTES = int(
    float(os.environ.get("RADIANCE_FS_FANOUT_TARGET_MB", "32")) * (2**20)
)
_RADIANCE_FANOUT_MAX = int(os.environ.get("RADIANCE_FS_FANOUT_MAX", "0"))


def _radiance_fanout_degree(
    n_blocks: int,
    n_threads: int,
    total_threads: int,
    inflight_jobs: int,
    block_size: int,
) -> int:
    """How many batches to split a job of n_blocks into.

    Splitting only pays while the device is not already saturated, and once several
    jobs are in flight the threads are busy anyway -- so the budget is divided by the
    in-flight job count. Beyond that, extra batches buy queue entries and wake-ups
    without moving more bytes.
    """
    if n_blocks <= 1 or n_threads <= 1:
        return 1
    if _RADIANCE_FANOUT_MAX > 0:
        budget = _RADIANCE_FANOUT_MAX
    elif block_size > 0:
        budget = -(-_RADIANCE_FANOUT_TARGET_BYTES // block_size)
    else:
        budget = total_threads
    # No more reads can be outstanding than there are workers to issue them.
    budget = min(budget, total_threads)
    jobs = max(1, inflight_jobs)
    return max(1, min(-(-budget // jobs), n_blocks, n_threads))


def _radiance_batches(n_items: int, n_batches: int):
    """Yield (start, stop) slices splitting n_items evenly, largest remainder first.

    Order is preserved and every item lands in exactly one slice.
    """
    q, r = divmod(n_items, n_batches)
    start = 0
    for i in range(min(n_items, n_batches)):
        stop = start + (q + 1 if i < r else q)
        yield start, stop
        start = stop''',
    sentinel="_RADIANCE_FANOUT_TARGET_BYTES",
    label="1 fs manager: fanout knobs and helpers",
)

# --- 2. remember the thread counts ------------------------------------------------------
apply(
    MGR,
    anchor='''        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )''',
    new='''        # radiance R3.14: kept so submit_store/submit_load can size the fanout. The
        # pool does not expose them and we do not want to reach into its internals.
        self._radiance_n_read_threads = n_read_threads
        self._radiance_n_write_threads = n_write_threads
        self._radiance_total_threads = max(1, n_read_threads + n_write_threads)
        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )''',
    sentinel="_radiance_total_threads",
    label="2 fs manager: record the pool thread counts",
)

# --- 3. the splitter --------------------------------------------------------------------
apply(
    MGR,
    anchor='''    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:''',
    new='''    def _radiance_split(self, io_fn, paths, offsets, n_threads):
        """radiance R3.14: turn one job into a list of partials, one per batch.

        Returns a single full-range partial for jobs of 0 or 1 blocks, which is
        byte-for-byte the old behaviour -- an empty list would build a JobState that
        never completes and would hang drain_jobs.
        """
        total = len(paths)
        if total <= 1:
            return [
                functools.partial(
                    io_fn,
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
            ]
        n_batches = _radiance_fanout_degree(
            total,
            n_threads if n_threads > 0 else self._radiance_total_threads,
            self._radiance_total_threads,
            # Unlocked read, as in get_stats(): it only picks a batch count.
            getattr(self._pool, "_inflight_jobs", 0),
            self._block_size,
        )
        return [
            functools.partial(
                io_fn,
                paths[a:b],
                self._primary_kv_view,
                offsets[a:b],
                self._block_size,
                self._use_o_direct,
            )
            for a, b in _radiance_batches(total, n_batches)
        ]

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:''',
    sentinel="def _radiance_split",
    label="3 fs manager: _radiance_split helper",
)

# --- 4. STORE path ----------------------------------------------------------------------
apply(
    MGR,
    anchor='''        task = functools.partial(
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])''',
    new='''        tasks = self._radiance_split(  # radiance R3.14
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._radiance_n_write_threads,
        )
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)''',
    sentinel="tasks = self._radiance_split(  # radiance R3.14\n            batch_store_block,",
    label="4 fs manager: fan the store path out",
)

# --- 5. LOAD path -----------------------------------------------------------------------
apply(
    MGR,
    anchor='''        task = functools.partial(
            batch_load_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )

        self._pool.enqueue_load(job_metadata.job_id, 1, [task])''',
    new='''        tasks = self._radiance_split(  # radiance R3.14
            batch_load_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._radiance_n_read_threads,
        )
        self._pool.enqueue_load(job_metadata.job_id, len(tasks), tasks)''',
    sentinel="tasks = self._radiance_split(  # radiance R3.14\n            batch_load_block,",
    label="5 fs manager: fan the load path out",
)

print(f"[radiance] R3.14 applied -- target "
      f"{os.environ.get('RADIANCE_FS_FANOUT_TARGET_MB', '32')} MiB, "
      f"max {os.environ.get('RADIANCE_FS_FANOUT_MAX', '0')} (1 = off)")
