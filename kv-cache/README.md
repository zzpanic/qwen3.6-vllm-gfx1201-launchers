# KV-cache offload for a GDN hybrid on a single AMD card

**GPU → RAM KV-cache offload for Qwen3.8-27B** (Gated-DeltaNet hybrid, MXFP4 weights, FP8 KV),
served by vLLM 0.27.1 / radiance 0.9.3 on one **AMD Radeon AI PRO R9700** (gfx1201 / RDNA4,
32 GB, TP=1), at a 204,800-token context. A third tier on disk is available as an experimental
build.

On one card, **the cheapest prefill is the one you do not do.** Run two agents and the second
evicts the first, which then re-prefills its whole prefix — tens of thousands of tokens, ~45 s
of wall clock during which the card belongs to nobody. This caches that prefix in system RAM so
the turn starts immediately.

As far as we can tell this is **the only hybrid KV offload that runs on RDNA4**. LMCache ships
its ROCm build for Instinct only (gfx942/gfx950) and its tracker has no RDNA issues.

## Two builds — use the default

| | **default (recommended)** | experimental |
|---|---|---|
| select with | nothing | `KVCACHE_EXPERIMENTAL=1` |
| tiers | GPU → RAM | GPU → RAM → disk |
| house patches | 3 behavioural: mixed-hit, eagle-groups, mamba-stride | those 3 + the instrumentation set + two disk-tier patches (fs fanout, lookup invalidation, the latter gated off) |
| needs | `/dev/shm` sized for the RAM tier | that, plus a dedicated filesystem and the **reaper** |
| observability | upstream only — `external_prefix_cache_hits/queries`, `kv_offload_store_bytes`: *whether* the tier serves | + per-tier counters, `tools/kvvalidate.py`, `tools/tierreport.py`: *why* it did or did not |

The default is the patch set the author's own production entry runs. It is small on purpose: the
three patches are the behavioural changes the RAM tier needs, and every one of them is needed for
correctness or capacity on this model. The experimental build adds everything that was needed to
*find* those three. On ordinary storage it is unresolved whether the disk tier is even faster than
a recompute (somewhere between 0.55× and 2.2×), and it needs an external garbage collector because
vLLM's fs tier never deletes. Turn it on if you want to work on the tier, not to get a faster cache.

## What you can expect

The RAM tier reads at **337,408 tok/s (11.8 GB/s)** — against a recompute, a hit is effectively
free. That rate does not depend on workload. It was measured on the experimental build; the default
build loads from RAM through the same copy path.

**The floor:** a prefix under **13,184 tokens** gets no external hit at all — the recurrent-state
snapshots are kept every 8th chunk, and a shorter prefix never reaches one. If your prefixes are
below ~13K tokens this will not help you.

The one long measured run was on the **experimental** build: **31.9M prompt tokens** of real agent
coding work, 2–3 agents on one card, ~14 h. It is a single saturated run, and a fresh boot will not
reproduce it:

| | share of prompt tokens |
|---|---|
| GPU prefix cache | 64.5% |
| offload tiers (RAM + disk) | 17.5% |
| **served without recompute** | **82.0%** |
| recomputed | 18.0% |

Its raw `/metrics` ships in `examples/`, so the report is checkable without the hardware:

```bash
python3 kv-cache/tools/kvvalidate.py --markdown --metrics-file kv-cache/examples/metrics-snapshot-20260912.txt
```

The full output is [`examples/EXAMPLE-REPORT.md`](examples/EXAMPLE-REPORT.md). The same run's disk
tier read at 3,868 tok/s (228 MB/s, device-bound, ordinary SATA SSD), and prefill against it is
unresolved between 0.52× and 2.08× of recompute. The default build has no equivalent report,
because the counters that produce it are among the patches it leaves out.

## Correctness

Exactness was the design requirement, and it is tested rather than asserted. `bench/correctbench.py`
runs six gates against a live endpoint. It exercises the disk tier, so it runs against the
experimental build, whose patches are a superset of the default's:

- **CT5** — negative control. Deliberately mutated output *must* be flagged, identical output must
  not. Nothing else runs unless this passes, because a suite that has never failed is untested.
- **CT1** — one ~90k-token prompt served cold, from GPU, from disk and mixed; every path compared
  to the cold reference.
- **CT2 / CT3** — a second prompt sharing a body but differing in its leading block must not be
  served another prompt's blocks, with GPU and tier counters asserted separately at zero.
- **CT4** — boundary sweep across chunk edges (±1 token), the case most likely to hide a defect.
- **CT6** — measures the recurrent-stride store against cold recompute at nine prefix lengths.

Last run: **CT1, CT2, CT5 PASS. CT4 bit-identical on all 11 uncontended probes. CT6
`max|dlogprob| = 0.0` at all nine lengths**, non-boundary included — the stride truncates how far
back a hit reaches, it does not approximate the state it returns.

**CT3 and CT4 both reported FAIL in that run, and both were contention.** CT4's three failing
offsets are each tagged `contended=True` while all eleven uncontended probes are bit-identical.
CT3 failed with the cache serving *nothing* on either side (`gpu_hits=0, ext_hits=0` — two
recomputes disagreeing), and **passed on a re-run against a quiet card**: `tokens_identical=True,
max|dlogprob|=0.0`. So the CT3 pass comes from a second run, not the full-suite one. Said plainly
because a suite that only ever passes is a suite nobody has tested — and because the contention
effect below is the whole reason those failures are not defects. An
85,696-token disk hit was bit-identical to a cold recompute, and across the run there were 5,423
promotions with 0 refused.

One caveat that matters when you run this yourself: **a co-tenant request is enough to flip a
near-tie token with no cache involved.** Measured here, an uncontended prompt diverged 0/10 times
and the same prompt with one co-tenant diverged 10/10. The harness tags every probe, and contention
can only create a spurious divergence — never hide a real one. Judge on uncontended probes.

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

**You do not have to do this by hand.** The launcher's default, `KVCACHE_TIER_GIB=auto`, computes
the minimum from `MAXLEN` and `KV_MEM` — on this stack 2 × 204,800 × 61,440 B = 23.4 GiB, so 24 GiB
— and checks it against `/dev/shm` and system RAM, keeping 15 GiB back for the engine, drafter and
OS. **If the minimum does not fit, offload is disabled for that boot and the log says why.** Set
`KVCACHE_TIER_GIB=<GiB>` to override.

## Getting started

```bash
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers
DRY_RUN=1 ./startup-qwen3.8-27b-kvcache.sh                         # default build; prints the command, runs nothing
DRY_RUN=1 KVCACHE_EXPERIMENTAL=1 ./startup-qwen3.8-27b-kvcache.sh  # experimental build, same
```

If that prints a sane command the launcher is wired correctly; the boot log's `[kvcache]` lines
name the build and the tier size. Once it is serving, watch the cache work — this reads only
upstream metrics, so it works on both builds:

```bash
watch -n 5 python3 kv-cache/tools/kvwatch.py
```

It shows GPU and offload-tier hit rates since the last refresh, the bytes the tier loaded and
stored, and the last few requests with how much of each was cached. It reads through llama-swap on
`:1234`; without llama-swap, point it at the engine with
`KVWATCH_METRICS=http://127.0.0.1:<port>/metrics`.

Then read [`docs/SETUP.md`](docs/SETUP.md) — prerequisites, sizing, and for
the experimental build the reaper (**required**, not advisory: without it the disk tier grows
without bound), the tools, and how to read the metrics without falling into the two traps that
cost this project the most time.

On the experimental build, `python3 kv-cache/tools/kvvalidate.py` checks a running system: it
re-establishes five invariants against a live busy endpoint, names which regime you are in before
you draw a conclusion, and marks every performance figure `SOLID`, `REGIME` or `UNRESOLVED`. It is
read-only and stdlib-only. Its `lookup_partition` check is the one to watch: the engine's six
terminal lookup buckets must sum to `lookup_calls` exactly, and any drift means a lookup exited
without recording an outcome — so every rate below it is a lower bound until you find the exit.
**Do not run it against the default build:** the counters it compares are not exported there, and
it reports FAILs that are not real.

## What this is

Anchored patches against vLLM 0.27.1 + radiance 0.9.3, applied at container start: three in the
default build, the full set in the experimental one — see
[`patches/APPLY-ORDER.txt`](patches/APPLY-ORDER.txt). It is a working, correct, unoptimised
implementation, not a proof of concept, and the intended destination is upstream vLLM: check each
patch against current HEAD and drop it wherever upstream has since implemented it.

Setup, sizing, tools and the metric traps are in [`docs/SETUP.md`](docs/SETUP.md). The patches
carry their own reasoning as comments — including, on the instrument that settled each one, the
things this project got wrong on the way.

**Author:** zzpanic — [github.com/zzpanic](https://github.com/zzpanic). Reproductions on other
hardware are wanted more than anything else here: if you run this on a different card, please open
an issue with your boot log's `[kvcache]` lines and the `External prefix cache hit rate` it reaches
— or, on the experimental build, `tools/kvvalidate.py` output.
