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
- from RAM at about 12 GB/s, roughly 170x faster than recomputing;
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
the same prompt. That is stricter than upstream vLLM promises. It holds serially; under concurrent
batches the batch shape itself changes the numerics.

Checking it:
- `kv-cache/bench/turnbench.py --yes` compares every cached turn of 3 sessions x 7 turns against a
  cold twin: 21/21 exact on both the RAM-only and the disk build.
- `kv-cache/tools/val_resume.py --evict 9` pushes a conversation out through every tier and checks
  the resume against a cold run.

Two things are not exact, and the launcher keeps both off:
- **fp16 GDN state** (`MAMBA_SSM_FP16=1`): a GDN state restored from RAM or disk at fp16 answers
  garbage, while fp32 round-trips bit-exactly. This is open work; see ROADMAP.md section 1.
- **Non-exact split-KV prefill kernels**: they change the summation order, so they are not used
  (KERNEL.md).

## How it is set up

| Knob | Default | What it does |
|---|---|---|
| `KVCACHE_TIER_GIB` | `auto` | RAM tier size: one full max-context prefill (below). `<GiB>` sets it; `0` turns offload off. |
| `KVCACHE_DISK_TIER` | `1` | Disk tier on, with the full instrumented patch set. `0` = RAM tier only, minimal patch set. |
| `KVCACHE_DISK` | `/kvcache` | Disk tier filesystem. If it does not exist, the boot says so and serves RAM-only. |
| `KVOFF_POLICY` | `arc` | RAM tier eviction: a prefix hit twice (a system prompt, a document head) survives a sweep of one-off blocks. |
| `KVOFF_MAMBA_STRIDE` | `4` | Store the GDN state every 4th chunk instead of every chunk. A GDN layer holds one recurrent state, not a per-token history, so storing it every chunk was 4x write amplification. |
| `KVOFF_PROMPT_ONLY` | `true` | Offload prompt tokens only, not generated ones. |

### Sizing the RAM tier

The tier lives in `/dev/shm` and is pinned memory, so it can never swap and every byte of it is
permanently taken from the host. The launcher sizes it from the model's geometry:

**Bytes per offloaded token.** Every 1,648-token chunk stores 2 attention groups plus the drafter
group, and the 6 GDN groups every 4th chunk. Each group is 16,384 B per token, so:

16,384 x (3 + 6/4) = **73,728 B/token**

(For checking: the engine reports 636 slots for 16 GiB, which is about 233k tokens.)

**The rule: the RAM tier holds one full max-context prefill.** A conversation at the configured
context length (`MAXLEN`) survives eviction from the GPU in full and resumes from RAM. `auto` sizes it
as `MAXLEN` x 73,728 B, plus 2% for chunk rounding, rounded up to a whole GiB. It must fit both
`/dev/shm` and `MemTotal - 15 GiB`; if it does not, offload is turned off for that boot and the log
says why.

| Context (`MAXLEN`) | RAM tier |
|---|---|
| 100k | 8 GiB |
| 204,800 | 15 GiB |
| 262,144 (the default, Qwen3.8's maximum) | **19 GiB** |

For two full-length sessions alternating, set `KVCACHE_TIER_GIB` to twice that (37 GiB at 262,144),
RAM permitting.

**Correction:** an earlier version of this repository used 61,440 B/token. That figure assumed a
1-in-8 GDN store stride, but the launcher runs 1-in-4. At 61,440, a 16 GiB tier was thought to hold
about 280k tokens; it holds about 233k.

### The disk tier: 128 GiB or more, and the reaper

- **Size.** A full 262,144-token conversation takes about 18 GiB on disk, and the reaper keeps
  24 GiB free (below). **128 GiB or more is recommended**: that keeps about five full-length
  conversations. A smaller volume works, but holds fewer, and the launcher warns below 128 GiB.
  Use a dedicated filesystem: the tier writes with O_DIRECT, so page cache does not help it, and it
  should not share space with anything that matters.
- **The reaper is mandatory.** vLLM's filesystem tier has no capacity limit and never deletes.
  Without the reaper (`kv-cache/ops/kvcache-reap.{sh,service,timer}`), the volume fills until writes
  fail. The launcher warns at boot if `kvcache-reap.timer` is not enabled.
- **Why 24 GiB free.** The reaper runs every 5 minutes and keeps
  `KVCACHE_MIN_FREE_GB=24` free. That is the peak measured write rate (225 MB/s) x 80 s x 1.25, so
  the disk cannot fill between reaps.
- **Hash seed.** Block filenames are content hashes, and the launcher pins `PYTHONHASHSEED` so a
  restart finds the same files. Changing it orphans the whole disk cache.

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

With the disk tier (the full patch set), `kvvalidate.py` and `tierreport.py` also explain why a lookup
did *not* hit.

## Evidence scope

Everything was measured on radiance 0.9.3 / vLLM 0.27.1 on one R9700. Stock vLLM does not run this
model on this card (the RDNA4 path is the radiance overlay plus libr4d), so there is no stock control
arm.
