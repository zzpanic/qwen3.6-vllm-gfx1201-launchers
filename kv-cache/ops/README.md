# Operations files

## The reaper is MANDATORY for the disk tier

The default build has no disk tier and needs none of this section. With
`KVCACHE_DISK_TIER=1` (or any `KVCACHE_DISK`), the fs tier writes and **never deletes**: `tiering/fs/manager.py` has no
capacity, quota or TTL parameter and exposes no eviction hook. Without an
external reaper the filesystem fills until it is full.

    sudo cp kvcache-reap.sh /usr/local/bin/        # or wherever, then fix the unit
    sudo cp kvcache-reap.service kvcache-reap.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now kvcache-reap.timer

It keeps the filesystem at or below 70% (`KVCACHE_TARGET_PCT`), oldest blocks first, and
leaves blocks younger than 15 minutes (`KVCACHE_MIN_AGE_MIN`) alone unless the volume passes
90%, because a full volume fails every store. A deleted block costs a recompute of that block
(with `patch_fs_failed_load.py` applied), not a crash. It works from `df` on the filesystem that
holds `KVCACHE_ROOT` (default `/kvcache/blocks`), so **give the tier its own filesystem**:
anything else stored there is paid for by deleting cache blocks.

**Size the free-space floor to the write rate, not the volume** (`KVCACHE_MIN_FREE_GB`, 24 in
the shipped unit). A percentage trigger alone is not enough: at 90% a 94 GB volume has ~9 GB
left, which peak prefill fills in about 40 s -- less than one timer cycle -- and below 90% the
15-minute floor stops the normal stage from touching the hot blocks. The reference box logged
15,486 ENOSPC store failures in five days that way. Below `KVCACHE_MIN_FREE_GB` the emergency
stage runs whatever `%use` says, and both stages delete until usage is at or below the target
AND that much is free. Size it as **peak store rate x gap between runs x 1.25**: read the peak
from the engine's `kv_offload_tier_store_bytes:('fs',)` metric (per 10 s interval), and the gap
is the timer cadence plus its accuracy. A failed store itself is safe -- each block file is
written to a temp name and renamed -- it just wastes the recompute.

## After every reload: check it is serving

    watch -n 5 python3 ../tools/kvwatch.py   # either build: hit rates and bytes moved
    python3 ../tools/kvvalidate.py           # disk build ONLY: the patches are live

`kvvalidate.py` is read-only and takes about a second. The patches are applied at
container start from the mounted source directory, so a stale mount, a failed hunk
or a launcher that skipped a step all look identical from outside — the engine
boots either way and then fails hours later under real traffic, which is how the
R3.15 bug was found. The script reads the RUNNING engine's files through
`/proc/<pid>/root`, not the copies on the host, because those are the ones that
can disagree. On the default build it reports FAILs that are not real, because the
counters it compares are not exported; there, read the boot log instead — the
      `[kvcache]` lines name the build, and each of the five non-fatal default patches
      (eagle-group, mamba-stride, reconcile re-ask, swa-align/touch-all, last-block
      align) prints a `[radiance] WARNING` if it did not apply (the sixth, mixed-hit,
      stops the boot).

## The mounts

`etc-fstab-snippets/` holds the `/dev/shm` and cache-filesystem lines. The sizing
rule is in `docs/SETUP.md` → Sizing; only the `/dev/shm` line is needed by the
default build.
