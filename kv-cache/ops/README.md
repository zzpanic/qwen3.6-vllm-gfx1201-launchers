# Operations files

## The reaper is MANDATORY, not optional

The fs (disk) tier writes and **never deletes**: `tiering/fs/manager.py` has no
capacity, quota or TTL parameter and exposes no eviction hook. Without an
external reaper the filesystem fills until it is full.

    sudo cp kvcache-reap.sh /usr/local/bin/        # or wherever, then fix the unit
    sudo cp kvcache-reap.service kvcache-reap.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now kvcache-reap.timer

`MIN_AGE=90min` is a **hard safety floor, not a tuning knob**. Reaping a block
that is in flight kills EngineCore outright — there is no load-failure recovery
path in this connector. Never lower it; never hand-delete a young block.

## The mounts

`etc-fstab-snippets/` holds the `/dev/shm` and cache-filesystem lines, with the
sizing arithmetic in comments. Sizing procedure and the OOM cliff are in
`docs/kv-cache-operations.md` §2 and §3 — read §3.0 before enlarging the tier.
