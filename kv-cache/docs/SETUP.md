# Setup and tools

Everything here assumes the launcher from the repository root. Each change says how to verify it
and how to undo it.

There are two builds (see the README): the **default**, GPU → RAM with three patches, and the
**experimental** one, `KVCACHE_EXPERIMENTAL=1`, which adds a disk tier and the instrumentation.
Sections marked *experimental* do not apply to the default.

## Prerequisites

- One AMD gfx1201 card (R9700, 32 GB), ROCm, podman or docker.
- `/dev/shm` **larger than the RAM tier**. The tier is a shared-memory region; if `/dev/shm` cannot
  hold it the boot dies with no log line at all. The kernel default is half of RAM, which is usually
  too small.
- *Experimental only:* a filesystem for the disk tier. Ordinary SATA SSD is fine — the tier reads
  at 228 MB/s and is device-bound, so a faster disk helps and a slower one is the limit.

```bash
python3 -c "import os;s=os.statvfs('/dev/shm');print('/dev/shm: %.2f GiB'%(s.f_blocks*s.f_frsize/2**30))"
sudo mount -o remount,size=32G /dev/shm          # live, no restart
```
Persist it by editing the `tmpfs /dev/shm` line in `/etc/fstab` (back the file up first).

## Sizing

**The RAM tier holds at least 2× the smaller of the GPU KV pool and max-model-len**, both in
tokens. That is the only sizing rule.

| smaller of GPU pool and max-model-len | RAM tier, at least | at 61,440 B/token |
|---|---|---|
| 100k | 200k tokens | 11.4 GiB |
| 200k | 400k tokens | 22.9 GiB |
| 262,144 (Qwen3.8's max) | 524,288 tokens | 30.0 GiB |

**The launcher applies it for you.** `KVCACHE_TIER_GIB` defaults to `auto`: it takes the GPU pool
from `KV_MEM` (or, if the pool is not pinned, uses `MAXLEN` alone, which can only over-size),
computes the minimum, rounds up to a whole GiB, and checks that it fits both `/dev/shm` and
`MemTotal − KVOFF_RAM_RESERVE_GIB` (15 GiB by default, kept for the engine, drafter and OS). **If it
does not fit, offload is disabled for that boot** and the log says what did not fit:

```
[kvcache] tier auto: 2 x min(GPU pool ~228,771, MAXLEN 204,800) x 61440 B/token = 23.4 GiB recommended minimum -> 24 GiB
[kvcache] *** KV-CACHE OFFLOAD DISABLED: the recommended minimum does not fit this system.
```

To run a different size anyway, set `KVCACHE_TIER_GIB=<GiB>`. An explicit size is taken as given,
still refused if `/dev/shm` cannot hold it, and still clamped to `MemTotal − KVOFF_RAM_RESERVE_GIB`.

The two bytes-per-token figures are model-specific (`KVCACHE_OFFLOAD_BPT`, `KVCACHE_GPU_BPT`). To
derive yours: the GPU figure is `--kv-cache-memory` bytes divided by the boot log's `GPU KV cache
size` in tokens. The offloaded figure is per-group bytes per token × the number of groups actually
stored per chunk. Here every stored block is 27,000,832 B for 1,648 tokens = 16,384 B/token per
group, and a chunk stores 2 attention groups + 1 draft group + 6 Mamba groups at the 1-in-8 stride:
16,384 × (2 + 1 + 6/8) = 61,440 B/token.

## The reaper — *experimental*, and then required

Nothing bounds the disk tier's growth on its own. **Install the reaper before serving the
experimental build**, or the filesystem fills. The default build has no disk tier and does not
need it.

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
DRY_RUN=1 ./startup-qwen3.8-27b-kvcache.sh                         # prints the container command, runs nothing
./startup-qwen3.8-27b-kvcache.sh                                   # serve, default build
KVCACHE_EXPERIMENTAL=1 ./startup-qwen3.8-27b-kvcache.sh            # serve, experimental build
```

| setting | meaning |
|---|---|
| `KVCACHE_EXPERIMENTAL` | `0` (default) GPU → RAM, three patches; `1` adds the disk tier and the instrumentation |
| `KVCACHE_TIER_GIB` | RAM tier, GiB. `auto` (default) applies the sizing rule above |
| `KVCACHE_DISK` | disk tier path. Defaults to `/kvcache` on the experimental build, unset on the default |

Confirm it came up — the first `[kvcache]` line names the build and the tier size:
```bash
sudo journalctl -u llama-swap --since "10 min ago" --no-pager | grep -E '\[kvcache\]|kv-offload'
```

*Experimental:* to start over, stop the server, `rm -rf` the tier directory, start it again.
Nothing on disk needs migrating between versions — the tier is a cache and is always safe to delete
cold.

## Tools

**`tools/kvwatch.py`** — the live view, and the one tool that works on **both** builds, because it
reads only metrics upstream vLLM exports:

```bash
watch -n 5 python3 tools/kvwatch.py
```

GPU and offload-tier hit rates (lifetime and since the last refresh — read the second), load/store
bytes, and the last few requests with how much of each was cached. It reads the `qwen3.8-27b-kvcache`
entry through llama-swap on `:1234`; point it elsewhere with
`KVWATCH_METRICS=http://127.0.0.1:<port>/metrics`. The per-request table needs llama-swap and is
skipped without it. The first refresh shows no rates — it has nothing to difference against yet.

**`tools/kvvalidate.py`** — *experimental build.* Read-only, stdlib-only, no GPU, one HTTP read.
Against a live busy endpoint it re-establishes five invariants, **names the regime you are in
before you draw a conclusion**, and marks every performance figure `SOLID`, `REGIME` or
`UNRESOLVED`. On the default build the counters it reads do not exist.

```bash
python3 tools/kvvalidate.py                      # text
python3 tools/kvvalidate.py --json               # machine-readable, severity per invariant
```

One thing it will tell you that looks alarming and is not: a freshly booted engine reports the
**warm-up** regime, because the disk tier cannot serve until the RAM tier fills. For the first ~45
minutes it looks dead when it is merely cold — judge nothing before then.

One thing it tells you that you should not ignore is `lookup_partition`. vLLM's `_lookup` has six
terminal exits and each increments one counter, so they must sum to `lookup_calls` exactly. A
positive drift means some lookup exited without recording an outcome, and every rate derived from
those buckets is then a lower bound rather than a measurement. A negative drift means an outcome
was counted twice, which makes the accounting unusable. Either way the number to chase is the
exit, not the counter.

**`tools/tierreport.py`** — *experimental build.* Tier sizing and read rates from the live counters.

**`bench/correctbench.py`** — *experimental build*, since it serves from the disk tier. The
correctness gates. CT5 is a negative control and gates the rest; nothing else runs unless a
deliberately corrupted output is correctly flagged.

```bash
python3 bench/correctbench.py --test ct5 --yes    # ~6 s, no GPU pressure
python3 bench/correctbench.py --test all --yes    # ~75 min, pushes real prefills
```

**Run correctness on a quiet card.** A single co-tenant request is enough to flip a near-tie token
with no cache involved — measured here as 0/10 divergent alone versus 10/10 with one co-tenant.
Every probe is tagged with whether it was contended; judge on the uncontended ones. Contention can
only create a spurious divergence, never hide a real one.

## Reading the metrics directly

On either build, whether the tier serves is
`external_prefix_cache_hits_total / external_prefix_cache_queries_total`, and what it moved is
`kv_offload_load_bytes_total` and `kv_offload_store_bytes_total`.

*Experimental:* token read rate for a tier is
`kv_offload_tier_hit_tokens_total{t} / kv_offload_tier_load_seconds_total{t}`.

**Never** use `kv_offload_tier_load_latency_seconds_sum` as a duration — it is thread-summed and
overstates wall clock by ~7.6×. Judge a tier's value by `load_bytes`, never by the chunk or token
hit counters, which repeat on every scheduler step.
