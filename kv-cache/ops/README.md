# Operations files

## The reaper is MANDATORY for the experimental disk tier

The default build has no disk tier and needs none of this section. With
`KVCACHE_EXPERIMENTAL=1` (or any `KVCACHE_DISK`), the fs tier writes and **never deletes**: `tiering/fs/manager.py` has no
capacity, quota or TTL parameter and exposes no eviction hook. Without an
external reaper the filesystem fills until it is full.

    sudo cp kvcache-reap.sh /usr/local/bin/        # or wherever, then fix the unit
    sudo cp kvcache-reap.service kvcache-reap.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now kvcache-reap.timer

`MIN_AGE=90min` is a **hard safety floor, not a tuning knob**. Reaping a block
that is in flight kills EngineCore outright — there is no load-failure recovery
path in this connector. Never lower it; never hand-delete a young block.

## After every reload: check it is serving

    watch -n 5 python3 ../tools/kvwatch.py   # either build: hit rates and bytes moved
    python3 ../tools/kvvalidate.py           # experimental build ONLY: the patches are live

`kvvalidate.py` is read-only and takes about a second. The patches are applied at
container start from the mounted source directory, so a stale mount, a failed hunk
or a launcher that skipped a step all look identical from outside — the engine
boots either way and then fails hours later under real traffic, which is how the
R3.15 bug was found. The script reads the RUNNING engine's files through
`/proc/<pid>/root`, not the copies on the host, because those are the ones that
can disagree. On the default build it reports FAILs that are not real, because the
counters it compares are not exported; there, read the boot log instead — the
`[kvcache]` lines name the build, and each of the two non-fatal default patches
prints a `[radiance] WARNING` if it did not apply (the third, mixed-hit, stops the
boot).

## The mounts

`etc-fstab-snippets/` holds the `/dev/shm` and cache-filesystem lines. The sizing
rule is in `docs/SETUP.md` → Sizing; only the `/dev/shm` line is needed by the
default build.
