# KV-cache offload for a GDN hybrid on a single AMD card

**KV-cache offload for Qwen3.8-27B** (Gated-DeltaNet hybrid, MXFP4 weights, FP8 KV), served by
vLLM 0.27.1 / radiance 0.9.3 on one **AMD Radeon AI PRO R9700** (gfx1201 / RDNA4, 32 GB, TP=1), at
a 204,800-token context. It comes in two options from one launcher: **GPU → RAM** (the default) and
**GPU → RAM → disk** (experimental).

## Why: more than one client per slot

A slot is one sequence the engine serves at a time (`--max-num-seqs`). When **more than one client
shares a slot**, they take turns, and each turn can push another client's prefix out of the GPU
cache. Without offload, that client **reprocesses its whole prefix** on its next turn. With offload,
the prefix is kept off the card and **loaded back instead of reprocessed**.

## Two options

Both run from `startup-qwen3.8-27b-kvcache.sh`; one variable chooses between them.

### Option 1 — GPU → RAM (default)

```bash
./startup-qwen3.8-27b-kvcache.sh
```

- **What it is:** a RAM tier in `/dev/shm` behind the GPU prefix cache, plus three behavioural
  patches: mixed-hit, eagle-groups, mamba-stride.
- **What it needs:** `/dev/shm` big enough for the RAM tier. The launcher sizes the tier (see
  [Sizing](#sizing)) and turns offload off, saying why, if it does not fit.
- **Tools:** `tools/kvwatch.py`.

### Option 2 — GPU → RAM → disk (experimental)

```bash
KVCACHE_EXPERIMENTAL=1 ./startup-qwen3.8-27b-kvcache.sh
```

- **What it is:** the same RAM tier with a disk tier behind it, plus the full instrumented patch set.
- **What it needs:** `/dev/shm` as above, a dedicated filesystem (`KVCACHE_DISK`, default
  `/kvcache`), and the **reaper** — the disk tier never deletes on its own.
- **Tools:** `tools/kvwatch.py`, `tools/kvvalidate.py`, `tools/tierreport.py`.

| | Option 1 — GPU → RAM (default) | Option 2 — GPU → RAM → disk (experimental) |
|---|---|---|
| select with | nothing | `KVCACHE_EXPERIMENTAL=1` |
| house patches | 3 behavioural | those 3 + instrumentation + two disk-tier patches |
| needs | `/dev/shm` for the RAM tier | that, plus a filesystem and the reaper |
| tools | `kvwatch.py` | `kvwatch.py`, `kvvalidate.py`, `tierreport.py` |

## What you can expect

**Both options:** the RAM tier reads at **337,408 tok/s (11.8 GB/s)** (measured on option 2).

**Both options — the floor:** a prefix under **13,184 tokens** gets no offload hit, because the
recurrent-state snapshots are kept every 8th chunk.

**Option 2:** the disk tier read at 3,868 tok/s (228 MB/s, ordinary SATA SSD); whether that beats a
recompute is not settled. One long run on this option — 31.9M prompt tokens of agent coding work,
2–3 agents, ~14 h, saturated:

| | share of prompt tokens |
|---|---|
| GPU prefix cache | 64.5% |
| offload tiers (RAM + disk) | 17.5% |
| **served without recompute** | **82.0%** |
| recomputed | 18.0% |

Its raw `/metrics` ships in `examples/`; reproduce the report with:

```bash
python3 kv-cache/tools/kvvalidate.py --markdown --metrics-file kv-cache/examples/metrics-snapshot-20260912.txt
```

Full output: [`examples/EXAMPLE-REPORT.md`](examples/EXAMPLE-REPORT.md).

## Correctness

`bench/correctbench.py` compares cached output against a cold recompute: a negative control (CT5),
cold / GPU / disk / mixed hits on one long prompt (CT1), cross-prompt isolation (CT2, CT3), chunk
boundaries (CT4) and the recurrent-state stride (CT6). It uses the disk tier, so it runs on option 2.

Last run: CT1, CT2, CT5 and CT6 passed; CT3 and CT4 failed only on probes that shared the card with
another request, and CT3 passed on a re-run on a quiet card. **Run it on a quiet card** — a
co-tenant request can change a near-tie token with no cache involved.

## Sizing

**One rule: the RAM tier holds at least 2× the smaller of the GPU KV pool and max-model-len**, both
in tokens. The GPU pool is the `GPU KV cache size` line in the boot log; Qwen3.8's max-model-len
tops out at 262,144.

| smaller of GPU pool and max-model-len | RAM tier, at least | at 61,440 B/token |
|---|---|---|
| 100k | 200k tokens | 11.4 GiB |
| 200k | 400k tokens | 22.9 GiB |
| 262,144 (e.g. a 300k pool at Qwen3.8's max) | 524,288 tokens | 30.0 GiB |

Convert at the *offloaded* bytes per token, which is not the GPU's: 61,440 B here, against 40,652 B
on-chip. Both are model-specific; [`docs/SETUP.md`](docs/SETUP.md) shows how to derive yours.

The launcher applies it in both options: `KVCACHE_TIER_GIB=auto` computes the minimum from `MAXLEN`
and `KV_MEM` (24 GiB on this stack) and checks it against `/dev/shm` and system RAM, keeping 15 GiB
back. **If it does not fit, offload is disabled for that boot and the log says why.** Set
`KVCACHE_TIER_GIB=<GiB>` to override.

## Getting started

```bash
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers
DRY_RUN=1 ./startup-qwen3.8-27b-kvcache.sh                         # option 1: prints the command, runs nothing
DRY_RUN=1 KVCACHE_EXPERIMENTAL=1 ./startup-qwen3.8-27b-kvcache.sh  # option 2: same
```

If that prints a sane command the launcher is wired correctly; the boot log's `[kvcache]` lines
name the option and the tier size. Then read [`docs/SETUP.md`](docs/SETUP.md).

**Watching it — both options:**

```bash
watch -n 5 python3 kv-cache/tools/kvwatch.py
```

Hit rates, bytes loaded and stored, and recent requests with how much of each was cached. It reads
through llama-swap on `:1234`; otherwise set `KVWATCH_METRICS=http://127.0.0.1:<port>/metrics`.

**Checking it — option 2:** `python3 kv-cache/tools/kvvalidate.py` checks the cache's invariants
against a live endpoint and names the regime it is in. **Do not run it against option 1** — the
counters it compares are not exported there, and it reports FAILs that are not real.

## What this is

Anchored patches against vLLM 0.27.1 + radiance 0.9.3, applied at container start: three in
option 1, the full set in option 2 — see [`patches/APPLY-ORDER.txt`](patches/APPLY-ORDER.txt). The
intended destination is upstream vLLM: check each patch against current HEAD and drop it wherever
upstream has since implemented it.

**Author:** zzpanic — [github.com/zzpanic](https://github.com/zzpanic). Reproductions on other
hardware are wanted more than anything else here: if you run this on a different card, please open
an issue with your boot log's `[kvcache]` lines and the `External prefix cache hit rate` it reaches
— or, on option 2, `tools/kvvalidate.py` output.
