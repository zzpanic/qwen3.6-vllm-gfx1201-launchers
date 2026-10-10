# KV-cache offload

The prefix cache can spill off the GPU into RAM and then disk, so a returning conversation
is **restored instead of recomputed**. The restored output is **bit-identical** to a cold run.
Everything for it is in [`kv-cache/`](kv-cache/), and `startup-qwen3.8-27b-mxfp4.sh` turns it on by default.

## The benefit

An agent session is a long prompt that grows a little every turn. vLLM's prefix cache already reuses
that prompt, but only while it stays in GPU memory. On one 32 GB card the GPU pool holds 329,035
tokens. Two agents, or one agent plus anything else, push each other's history out.

Without offload, the next turn of an evicted conversation re-runs prefill over the **whole** history.
At this card's 2,200-3,000 tokens/s, that is 30-50 s at 100k tokens before the first output token.

With offload, the evicted blocks have already been copied to RAM, and from there to disk. The next
turn loads them back:
- from RAM at about 12 GB/s, roughly 170x faster than recomputing (that is the PCIe 3.0 x16 ceiling
  of the reference board; SYSTEM.md);
- from disk at about 1 GB/s, roughly 15x faster.

Only the genuinely new tokens are prefilled.

### Speedup, measured

Three agent-style sessions growing to about 110k tokens each, served round-robin, so that together
they outgrow the GPU pool from turn 4 on (`kv-cache/bench/turnbench.py`).
- **No cache**: the same request replayed with a fresh cache salt.
- **Times**: whole requests, prefill plus 320 generated tokens.
- **Brackets**: tokens served from the tier.

| turn | prompt | no cache | RAM only | RAM + disk |
|---|--:|--:|--:|--:|
| B4 | 65,981 | 30.7 s | 12.8 s (49k) | 12.8 s (49k) |
| A5 | 80,895 | 37.8 s | 15.1 s (63k) | 15.2 s (63k) |
| C5 | 81,756 | 38.8 s | 13.2 s (67k) | 13.3 s (67k) |
| A6 | 95,502 | 46.7 s | 15.5 s (77k) | 15.5 s (77k) |
| B6 | 95,278 | 46.9 s | 21.6 s (67k) | 16.6 s (77k) |
| C6 | 96,116 | 47.2 s | 27.9 s (53k) | 18.0 s (77k) |
| A7 | 110,063 | 51.1 s | 44.2 s (21k) | 14.9 s (92k) |
| B7 | 108,878 | 51.8 s | 42.4 s (28k) | 16.5 s (92k) |
| C7 | 110,553 | 52.9 s | 49.6 s (11k) | 15.5 s (95k) |

- A 16 GiB RAM tier carries the sessions until they outgrow it, at turn 6.
- The disk tier keeps every later turn near 15 s, against about 52 s with no cache, a **3.4x** speedup.
- All 21 turns served cleanly in both modes.

This table was measured on ggz14's single-GPU build (fp16 GDN state, `MAXSEQS=3`, 220k context, a
248,235-token pool) with this patch set applied. That is how the work was offered upstream in
[ggz14/radiance-vllm-mxfp4#52](https://codeberg.org/ggz14/radiance-vllm-mxfp4/pulls/52).

### Where hits came from, lifetime of one boot

Sample `kv-cache/tools/kvtable.py` output. It was taken under light traffic, so the disk column is one
block's worth of evidence.

|  | L0 GPU | L1 RAM | L2 disk | recompute |
|---|---|---|---|---|
| Served, lifetime | 85.8% | 6.5% | 0.0% | 7.7% |
| Moves data | in place | RAM -> GPU at 12.0 GB/s | disk -> RAM at 1.0 GB/s | - |
| Tokens/s equivalent | - | ~334k | 29k | 2k |
| vs recompute | - | 173x | 15x | 1x |
| Batch latency p50 / p99 | - | 0.25 s / 0.50 s | 0.03 s / 0.50 s | - |

## Exact, not approximate

A turn served from the cache produces the **same tokens and the same logprobs** as a cold prefill of
the same prompt, at fp16 and at fp32 GDN state. That is stricter than upstream vLLM promises. It holds
serially; under concurrent batches the batch shape itself changes the numerics.

What makes it exact (2026-10-10, **canonical prefill chunks**, `KVOFF_CANONICAL_CHUNKS=1`): a block's
KV and recurrent state depend on the LENGTH of the prefill chunk they were computed in, not only on
the tokens. Stock vLLM stops a chunk at the shared-prefix boundary of a cache hit, which a cold prefill
never does, so a cached resume continued on a different chunk grid and differed from cold in the low
bits (fp16 state shows it more, but fp32 has the same mechanism). The launcher now prefills on one
fixed grid from token 0, resumes only on that grid, and caches only blocks computed in a full grid
chunk. The cost: prompts shorter than one grid chunk (~3.5k tokens) are never served from cache.

Checking it:
- `kv-cache/bench/turnbench.py --yes` compares every cached turn of 3 sessions x 7 turns against a
  cold twin: 21/21 exact on both the RAM-only and the disk build.
- `kv-cache/tools/val_resume.py --evict 9` pushes a conversation out through every tier and checks
  the resume against a cold run.

Not exact, and kept off: **non-exact split-KV prefill kernels**. They change the summation order,
so they are not used (KERNEL.md).

## How it is set up

| Knob | Default | What it does |
|---|---|---|
| `KVCACHE_TIER_GIB` | `16` | RAM tier size in GiB (below). `auto` = one full max-context prefill at dense bytes; `0` turns offload off. |
| `KVCACHE_DISK_TIER` | `1` | Disk tier on, with its three behavioural fixes (failed-load forget, fs fan-out, lookup invalidation). `0` = RAM tier only. |
| `KVOFF_MINIMAL` | `1` | `0` adds the six instrumentation patches (counters, a debug event sink, the per-tier report) for `kvvalidate.py` and `tierreport.py`. Diagnosis only: it makes the log and `/metrics` much noisier. |
| `KVCACHE_DISK` | `/kvcache` | Disk tier filesystem. If it does not exist the launcher refuses to start and says how to fix it (create it, point `KVCACHE_DISK` elsewhere, or `KVCACHE_DISK_TIER=0`). |
| `KVOFF_POLICY` | `arc` | RAM tier eviction: a prefix hit twice (a system prompt, a document head) survives a sweep of one-off blocks. |
| `KVOFF_MAMBA_STRIDE` | `4` | Store the GDN state every 4th chunk instead of every chunk. A GDN layer holds one recurrent state, not a per-token history, so storing it every chunk was 4x write amplification. |
| `KVOFF_PROMPT_ONLY` | `true` | Offload prompt tokens only, not generated ones. |

### Sizing: 16 GiB of RAM and a 128 GiB disk at 75%

**Recommended: a 16 GiB RAM tier, and a 128 GiB disk-tier volume that the reaper keeps at 75% full**
(96 GiB of cache; the other 25% is headroom the reaper keeps free). Those are the defaults
(`KVCACHE_TIER_GIB=16`, `KVCACHE_TARGET_PCT=75`).

**Why they are smaller than before: checkpoint retention** (2026-10-11, `KVOFF_VRAM_RETENTION`,
`KVOFF_TIER_RETENTION`, on by default). A cached conversation is attention KV for every token plus
GDN and drafter checkpoints. Real agent traffic only ever resumes at the end of the previous prompt,
at the shared system prompt, or a few chunks below the end -- never deep in mid-history -- so the tiers
now keep checkpoints only there, plus a sparse backoff, instead of at every chunk. Attention stays
complete, and exactness is unchanged: retention decides what stays cached, never what is computed.
On a replay of real agent sessions this kept the same hits as dense storage with about half the RAM
and disk, and with 3-4 agents in parallel it served far more follow-up turns from cache (4 agents
56% -> 77%, 3 agents 69% -> 85%).

**What the sizes hold.** vLLM reports the GPU KV pool and the context per request; neither says how
much *finished* conversation stays ready to resume. The launcher prints that on the first request
(`[radiance] warm history ...`, with the recommended minimums for your box). At 100k-token
conversations, on the reference box:

| | with retention | dense (before) |
|---|--:|--:|
| GPU pool | ~314k tokens | ~166k |
| RAM tier, 16 GiB | ~355k | ~233k |
| disk tier, 96 GiB usable | ~2.2M | ~1.4M |

(Measured from that line at a 12 GiB RAM tier and 66 GiB of disk, scaled to these sizes: about
46 KB per cached token in the tiers with retention, 74 KB dense. Context per request is unchanged.)

**The minimums.**
- **RAM tier: ~10 GiB at 262,144 context** (8 GiB at 204,800). A disk hit is restored disk -> RAM ->
  GPU, so the RAM tier must be able to stage one full-length resume (attention for every chunk plus
  one checkpoint). Below that, the longest conversations cannot come back from disk.
- **Disk tier: ~16 GB per concurrent agent, 32 GB at the least.** That is ~350k tokens of warm
  history per agent, the point past which more storage stopped adding hits on real agent traffic.
- The tier is pinned memory in `/dev/shm`, so it never swaps and is permanently taken from the host:
  `/dev/shm` must be larger than the tier (20G for 16 GiB), and the box needs the tier plus ~15 GiB
  for the engine, drafter and OS. If either does not fit, offload is turned off for that boot and the
  log says why.

### The disk tier and the reaper

- **Size.** 128 GiB with the reaper at 75% gives 96 GiB of cache, ~2.2M tokens of finished
  conversations. A smaller volume works and holds proportionally less; the launcher warns below
  128 GiB. Use a dedicated filesystem: the tier writes with O_DIRECT, so page cache does not help it,
  and it should not share space with anything that matters.
- **The reaper is mandatory.** vLLM's filesystem tier has no capacity limit and never deletes.
  Without the reaper (`kv-cache/ops/kvcache-reap.{sh,service,timer}`), the volume fills until writes
  fail. It keeps the volume at or below `KVCACHE_TARGET_PCT` (75), oldest blocks first, and the
  launcher warns at boot if `kvcache-reap.timer` is not enabled.
- **Why it also keeps 24 GB free.** `KVCACHE_MIN_FREE_GB=24` is the peak measured write rate
  (225 MB/s) x 80 s x 1.25, so the disk cannot fill between reaps whatever the percentage says.
- **Hash seed.** Block filenames are content hashes, and the launcher pins `PYTHONHASHSEED` so a
  restart finds the same files. Changing it orphans the whole disk cache.
- **Clearing it.** `./clear-cache.sh` wipes it per model, or everything except the models you keep.

`kv-cache/docs/SETUP.md` has the installation steps, the fstab lines and the derivation of the
sizing numbers.

## What the patches do

The offload path is vLLM 0.27.1's own `OffloadingConnector`, plus house patches in
`kv-cache/patches/` that make it work on a GDN hybrid with a drafter:
- **Mixed hits**: a prefix partly on the GPU and partly in a tier.
- **Drafter groups**: the DFlash2 drafter's KV groups get annotated.
- **GDN store stride.**
- **Re-asked turns**: reconcile a re-asked turn.
- **Alignment**: the prompt's last block and the sliding-window touch order.
- **Disk-tier failures**: forget a failed disk load and recompute that one block.
- **Instrumentation**: the counters behind `kvtable.py`, `kvvalidate.py` and `tierreport.py`.

Every behaviour change sits behind an environment gate whose unset state is stock vLLM.
`kv-cache/README.md` lists each patch. Two of them are generic vLLM fixes (`patch_sched_align_last_block`,
`patch_fs_failed_load`) whose intended destination is upstream.

## Watching it

```bash
python3 kv-cache/tools/kvtable.py --url http://127.0.0.1:<port>/metrics   # per-tier hits, capacity, speed
python3 kv-cache/tools/kvwatch.py                                          # live
```

With `KVOFF_MINIMAL=0` (the instrumentation), `kvvalidate.py` and `tierreport.py` also explain why a
lookup did *not* hit.

## Evidence scope

Everything was measured on radiance 0.9.3 / vLLM 0.27.1 on one R9700. Stock vLLM does not run this
model on this card (the RDNA4 path is the radiance overlay plus libr4d), so there is no stock control
arm.
