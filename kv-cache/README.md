# KV-cache offload for a GDN hybrid on a single AMD card

KV-cache offload for hybrid (Gated-DeltaNet + attention + drafter) models on one card,
GPU → RAM → disk. Served by vLLM 0.27.1 / radiance 0.9.3 on one AMD Radeon AI PRO R9700
(gfx1201 / RDNA4, 32 GB, TP=1) — here, Qwen3.8-27B (Gated-DeltaNet hybrid, MXFP4 weights,
FP8 KV) at a 204,800-token context. It exists so that when **more than one client shares a
slot** (`--max-num-seqs`), a returning client's prefix is **loaded back from RAM (or disk)
instead of reprocessed**: the prefix is kept off the card, and on the next turn it is restored
rather than recomputed.

## Why this build: bit-identical, not fast

This build does not chase peak speed; it is focused on **bit-identical KV-cache serving**. That
means a cached turn produces the **same tokens and the same logprobs** as a cold prefill of the
same prompt — a bar stricter than upstream promises. It holds **serially** (one turn at a time);
it is not expected under concurrent batches, where the batch shape changes the numerics.

## Two options

Both run from `startup-qwen3.8-27b-kvcache.sh`; one variable chooses between them.

```bash
./startup-qwen3.8-27b-kvcache.sh                    # option 1 (default)
KVCACHE_EXPERIMENTAL=1 ./startup-qwen3.8-27b-kvcache.sh   # option 2 (experimental)
DRY_RUN=1 ./startup-qwen3.8-27b-kvcache.sh        # print the command, run nothing
```

| | Option 1 — GPU → RAM (default) | Option 2 — GPU → RAM → disk (experimental) |
|---|---|---|
| select with | nothing | `KVCACHE_EXPERIMENTAL=1` |
| house patches | the 6 "always" behavioural | all 15 (the 6 + 9 instrumentation / disk-tier) |
| needs | `/dev/shm` for the RAM tier | that, plus a filesystem (`KVCACHE_DISK`, default `/kvcache`) and the reaper |
| tools | `kvwatch.py` | `kvwatch.py`, `kvvalidate.py`, `tierreport.py` |

Option 1 is the release default (`KVOFF_MINIMAL=1`, no disk tier). Option 2 adds the disk tier
and the full instrumented set; `kvvalidate.py` is option 2 only — its counters are not exported
on option 1, and it reports false FAILs there.

**Sizing — one rule.** The RAM tier holds **at least 2 × the smaller of the GPU KV pool and
max-model-len**, both in tokens. The GPU pool is the `GPU KV cache size` line in the boot log;
Qwen3.8's max-model-len tops out at 262,144.

| smaller of GPU pool and max-model-len | RAM tier, at least |
|---|---|
| 100k | 200k tokens |
| 200k | 400k tokens |
| 262,144 (e.g. a 300k pool) | 524,288 tokens |

`KVCACHE_TIER_GIB=auto` applies this and checks it against `/dev/shm` and system RAM (keeping
15 GiB back). **If it does not fit, offload is disabled for that boot and the log says why.**
Set `KVCACHE_TIER_GIB=<GiB>` to override. (Convert at the *offloaded* bytes/token — not the
on-chip figure — per [`docs/SETUP.md`](docs/SETUP.md).)

## The tier table

Sample output of `tools/kvtable.py` — **2026-09-19, lifetime since the 18:57 boot; light traffic
(the fs tier served 1,648 tokens)**. Produce your own with
`python3 kv-cache/tools/kvtable.py --url <engine>/metrics`.

|  | L0 GPU | L1 RAM | L2 SSD | recompute |
|---|---|---|---|---|
| Served, lifetime | 85.8% | 6.5% | 0.0% | 7.7% |
| Capacity | 229k tok (8.7 GiB) | 22.0 GiB | 93.9 GiB | - |
| Bytes per token | 40 KB | 35 KB | 32 KB | - |
| Moves data | in place | RAM → GPU at 12.0 GB/s | SSD → RAM at 1.0 GB/s | - |
| Tokens/s equivalent | - | ~334k | 29k | 2k |
| vs recompute | - | 173x | 15x | 1x |
| Batch latency p50 / p99 | - | 0.25 s / 0.50 s | 0.03 s / 0.50 s | - |
| How full | always | 22 of 22 GiB | 61 of 94 GiB | - |

## Checking exactness yourself

`turnbench` is the gate: 3 sessions × 7 turns, every **cached** turn compared token-for-token and
logprob-for-logprob against a **cold twin** of the same prompt. Both builds pass it
**21/21 exact** (logprobs bit-identical, spec counters equal): option 1 (RAM only) and option 2
(RAM + disk). Run it on a quiet card — a co-tenant request can flip a near-tie token with no
cache involved. One command:

```bash
python3 kv-cache/bench/turnbench.py --yes
# against something other than the launcher's own entry:
TURNBENCH_BASE=http://127.0.0.1:8000 python3 kv-cache/bench/turnbench.py --yes
```

A corrupt or truncated block on the disk tier is not fatal: the read fails, that one block is
recomputed, and the block is re-stored intact.

## The patch set

Anchored against vLLM 0.27.1 + radiance 0.9.3, applied at container start in a real dependency
order. Every behaviour change sits behind an env gate whose **unset state is stock vLLM**; nothing
hard-codes a model name, group index or block size. **Default = the 6 "always"; experimental = all 15.**

| # | patch | does | scope |
|---|---|---|---|
| 1 | `patch_offload_mixed_hit` | mixed GPU / tier hit accounting | default |
| 2 | `patch_offload_instrumentation` | per-tier load/store metrics | exp |
| 3 | `patch_offload_lookup_metrics` | lookup-outcome counters | exp |
| 4 | `patch_eagle_groups` | annotate the eagle / draft KV groups | default |
| 5 | `patch_mamba_stride` | recurrent-state store stride (4) | default |
| 6 | `patch_reconcile_reask` | reconcile a re-asked turn (memo off) | default |
| 7 | `patch_swa_align_touch` | sliding-window align / touch | default |
| 8 | `patch_sched_align_last_block` | align the prompt's last block — **upstream candidate** | default |
| 9 | `patch_offload_debug_instrument` | debug hooks | exp |
| 10 | `patch_offload_fs_fanout` | fs read fan-out | exp |
| 11 | `patch_offload_tier_report` | tier-report metrics | exp |
| 12 | `patch_offload_promotion_wallclock` | promotion + wall-clock timing | exp |
| 13 | `patch_lookup_invalidate` | lookup invalidation (off by default) | exp |
| 14 | `patch_fs_failed_load` | forget a failed disk load — **upstream candidate** | exp |
| 15 | `patch_offload_miss_deferral_metrics` | miss / deferral counters | exp |

The two marked rows are generic vLLM fixes, not model-specific. The intended destination is
**upstream vLLM**: check each patch against current HEAD and drop it wherever upstream has since
implemented it.

## How this differs from ggz14

A deliberate re-configuration of ggz14's TP=1 stack — this launcher is a house copy of its
`serve-mxfp4.sh`, and it **does not adopt ggz14's single-GPU profile**. Only the settings that
matter (ggz14 column = `serve-mxfp4.sh` TP=1 single-GPU profile at `980f891`):

| setting | ggz14 (TP=1 profile) | here | why |
|---|---|---|---|
| KV offload tier | none (no `--kv-transfer-config` at all) | GPU → RAM (→ disk) | **this is the point** |
| mamba ssm dtype | fp16 (conv bf16) | **fp32 ssm** | a tier can't round-trip a 16-bit temporal state |
| context (`--max-model-len`) | 65,536 | 204,800 | extra context is gold; fits the cold pin |
| `--max-num-seqs` | 8 | 2 | single-user agentic; serialises eviction |
| prefill chunk (`--max-num-batched-tokens`) | 4,096 | 2,048 | smallest activation transient |
| `--gpu-memory-utilization` | 0.98 | 0.97 | 0.98 → profiling → device-lost without a pin (the pin overrides it regardless) |
| `--kv-cache-memory` | auto → profile pin 6,535,819,798 | explicit pin 9,300,000,000 | no cold/warm gap |
| GDN_LAZY | 1 | 0 | profile default, not adopted |
| draft sampling | greedy | probabilistic | +10% acceptance, measured, no cost |
| generation temp | 0.7 | 1.0 | model-card preset; 0.7 was benchmark-chasing |
| capture sizes | `[1 … 64]` | `[1,2,4,8,16]` | capture only small-M decode shapes |

## Evidence scope

Everything here was measured on **radiance 0.9.3 / vLLM 0.27.1 on one R9700**. **Stock vLLM does
not run this model on this card** (the RDNA4 path is the radiance overlay + libr4d), so there is
no stock control arm; any contribution must say so up front.
