# Setup and tools

Everything here assumes the launcher from the repository root. Each change says how to verify it
and how to undo it.

## Prerequisites

- One AMD gfx1201 card (R9700, 32 GB), ROCm, podman or docker.
- A filesystem for the disk tier. Ordinary SATA SSD is fine — the tier reads at 228 MB/s and is
  device-bound, so a faster disk helps and a slower one is the limit. Give it what you can spare;
  a 204,800-token context is ~12.6 GB offloaded.
- `/dev/shm` **larger than the RAM tier**. The tier is a shared-memory region; if `/dev/shm` cannot
  hold it the boot dies with no log line at all.

```bash
python3 -c "import os;s=os.statvfs('/dev/shm');print('/dev/shm: %.2f GiB'%(s.f_blocks*s.f_frsize/2**30))"
sudo mount -o remount,size=32G /dev/shm          # live, no restart
```
Persist it by editing the `tmpfs /dev/shm` line in `/etc/fstab` (back the file up first).

## The reaper is required

Nothing bounds the disk tier's growth on its own. **Install the reaper before serving**, or the
filesystem fills.

```bash
sudo cp ops/kvcache-reap.{sh,service,timer} /etc/systemd/system/   # .sh to /usr/local/bin
sudo systemctl enable --now kvcache-reap.timer
systemctl is-enabled kvcache-reap.timer          # want: enabled
```

It is capacity-governed with an absolute minimum-age floor, so it will not delete a block that a
running request still needs. Mask it during correctness runs — it will otherwise delete blocks
mid-test and produce a failure that is not real.

## Running

```bash
DRY_RUN=1 ./startup-qwen3.8-27b-kvcache.sh       # prints the container command, runs nothing
./startup-qwen3.8-27b-kvcache.sh                 # serve
```

Two settings do the sizing. Set them together — the RAM tier must fit inside `/dev/shm`:

| setting | meaning |
|---|---|
| `--kv-offloading-size` | RAM tier, GiB. Size at `(concurrent agents + 1) × context × 61,440 B` |
| `KVOFF_DISK` | disk tier path. Unset it to run RAM-only |

Leave ~15 GB of system RAM beyond the tier for the engine, drafter and working set. Dropping the
disk tier keeps the RAM tier, which is the one that always pays (~45× a recompute); the disk tier
is insurance against eviction.

Confirm it came up:
```bash
sudo journalctl -u llama-swap --since "10 min ago" --no-pager | grep -i kv-offload
```

To start over, stop the server, `rm -rf` the tier directory, start it again. Nothing on disk needs
migrating between versions — the tier is a cache and is always safe to delete cold.

## Tools

**`tools/kvvalidate.py`** — the one to run first. Read-only, stdlib-only, no GPU, one HTTP read.
Against a live busy endpoint it re-establishes five invariants, **names the regime you are in
before you draw a conclusion**, and marks every performance figure `SOLID`, `REGIME` or
`UNRESOLVED`.

```bash
python3 tools/kvvalidate.py                      # text
python3 tools/kvvalidate.py --json               # machine-readable, severity per invariant
```

Two things it will tell you that look alarming and are not. A freshly booted engine reports the
**warm-up** regime: the disk tier cannot serve until the RAM tier fills, so for the first ~45
minutes it looks dead when it is merely cold — judge nothing before then. And it warns about a
small gap in vLLM's own lookup accounting; that is an engine defect, not your setup, which is why
it warns rather than fails.

**`tools/tierreport.py`** — tier sizing and read rates from the live counters.

**`bench/correctbench.py`** — the correctness gates. CT5 is a negative control and gates the rest;
nothing else runs unless a deliberately corrupted output is correctly flagged.

```bash
python3 bench/correctbench.py --test ct5 --yes    # ~6 s, no GPU pressure
python3 bench/correctbench.py --test all --yes    # ~75 min, pushes real prefills
```

**Run correctness on a quiet card.** A single co-tenant request is enough to flip a near-tie token
with no cache involved — measured here as 0/10 divergent alone versus 10/10 with one co-tenant.
Every probe is tagged with whether it was contended; judge on the uncontended ones. Contention can
only create a spurious divergence, never hide a real one.

## Reading the metrics directly

Token read rate for a tier is
`kv_offload_tier_hit_tokens_total{t} / kv_offload_tier_load_seconds_total{t}`.

**Never** use `kv_offload_tier_load_latency_seconds_sum` as a duration — it is thread-summed and
overstates wall clock by ~7.6×. Judge a tier's value by `load_bytes`, never by the chunk or token
hit counters, which repeat on every scheduler step.
