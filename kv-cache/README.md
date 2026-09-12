# Three-tier KV-cache offload for a GDN hybrid on a single AMD card

**GPU → RAM → disk KV-cache offload for Qwen3.8-27B** (Gated-DeltaNet hybrid, MXFP4 weights,
FP8 KV), served by vLLM 0.27.1 / radiance 0.9.3 on one **AMD Radeon AI PRO R9700**
(gfx1201 / RDNA4, 32 GB, TP=1), at a 204,800-token context.

On one card, **the cheapest prefill is the one you do not do.** Run two agents and the second
evicts the first, which then re-prefills its whole prefix — tens of thousands of tokens, ~45 s
of wall clock during which the card belongs to nobody. This caches that prefix in RAM and on
disk so the turn starts immediately.

As far as we can tell this is **the only hybrid KV offload that runs on RDNA4**. LMCache ships
its ROCm build for Instinct only (gfx942/gfx950) and its tracker has no RDNA issues.

## What you can expect

Measured over **31.5M prompt tokens** of real agent coding work — 2–3 agents on one card, ~14 h:

| | share of prompt tokens |
|---|---|
| GPU prefix cache | 64.1% |
| offload tiers (RAM + disk) | 17.7% |
| **served without recompute** | **81.8%** |
| recomputed | 18.2% |

Tier read rates, which do not depend on workload:

| tier | token rate | bandwidth | note |
|---|---|---|---|
| RAM | 337,408 tok/s | 11.8 GB/s | effectively free against a recompute |
| disk | 3,868 tok/s | 228 MB/s | device-bound (5.3% wait), ordinary SATA SSD |

**The floor:** a prefix under **13,184 tokens** gets no external hit at all — the recurrent-state
snapshots are kept every 8th chunk, and a shorter prefix never reaches one. If your prefixes are
below ~13K tokens this will not help you.

**Not tuned.** Prefill throughput against the disk tier is unresolved between 0.52× and 2.08× of
recompute; the RAM tier is unambiguous at ~45× the worst-case prefill. Treat disk as insurance
against eviction, RAM as the tier that pays.

## Correctness

Exactness was the design requirement, and it is tested rather than asserted. `bench/correctbench.py`
runs six gates against a live endpoint:

- **CT5** — negative control. Deliberately mutated output *must* be flagged, identical output must
  not. Nothing else runs unless this passes, because a suite that has never failed is untested.
- **CT1** — one ~90k-token prompt served cold, from GPU, from disk and mixed; every path compared
  to the cold reference.
- **CT2 / CT3** — a second prompt sharing a body but differing in its leading block must not be
  served another prompt's blocks, with GPU and tier counters asserted separately at zero.
- **CT4** — boundary sweep across chunk edges (±1 token), the case most likely to hide a defect.
- **CT6** — measures the recurrent-stride store against cold recompute at nine prefix lengths.

Last run, on the shipped stack: **CT1, CT2, CT3, CT5 PASS. CT4 bit-identical on all 11
uncontended probes. CT6 `max|dlogprob| = 0.0` at all nine lengths**, non-boundary included — the
stride truncates how far back a hit reaches, it does not approximate the state it returns. An
85,696-token disk hit was bit-identical to a cold recompute, and across the run there were 5,423
promotions with 0 refused.

One caveat that matters when you run this yourself: **a co-tenant request is enough to flip a
near-tie token with no cache involved.** Measured here, an uncontended prompt diverged 0/10 times
and the same prompt with one co-tenant diverged 10/10. The harness tags every probe, and contention
can only create a spurious divergence — never hide a real one. Judge on uncontended probes.

## Sizing

Both numbers are model-specific; derive yours as shown in [`docs/SETUP.md`](docs/SETUP.md). Here:

- on-GPU pool **40,652 B/token**, offload **61,440 B/token** (offload costs 1.51× per token)
- one 204,800-token context = **8.33 GB** on-chip, **12.58 GB** offloaded

Size the RAM tier at **(concurrent agents + 1) × context × 61,440 B**, the `+1` being the staging
slot a promotion needs. Ours is 24 GiB — 2.05 full contexts, which fits two agents and not a third
— and leave ~15 GB of system RAM for the engine, drafter and working set. Disk is cheap: give it
what you can. At a 32K context the whole thing is ~2 GB and you probably do not need this.

## Getting started

```bash
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers
DRY_RUN=1 ./startup-qwen3.8-27b-kvcache.sh     # prints the container command, runs nothing
```

If that prints a sane command the launcher is wired correctly. Then read
[`docs/SETUP.md`](docs/SETUP.md) — prerequisites, the reaper (**required**, not advisory: without
it the disk tier grows without bound), sizing, the tools, and how to read the metrics without
falling into the two traps that cost this project the most time.

To check a running system, `python3 tools/kvvalidate.py` re-establishes five invariants against a
live busy endpoint, names which regime you are in before you draw a conclusion, and marks every
performance figure `SOLID`, `REGIME` or `UNRESOLVED`. It is read-only and stdlib-only. It will warn
about a small gap in vLLM's own lookup accounting; that is an engine defect, not your setup.

## What this is

Nine anchored patches against vLLM 0.27.1 + radiance 0.9.3, applied at container start — see
[`patches/APPLY-ORDER.txt`](patches/APPLY-ORDER.txt). It is a working, correct, unoptimised
implementation, not a proof of concept, and the intended destination is upstream vLLM: check each
patch against current HEAD and drop it wherever upstream has since implemented it.

Setup, sizing, tools and the metric traps are in [`docs/SETUP.md`](docs/SETUP.md). The patches
carry their own reasoning as comments — including, on the instrument that settled each one, the
things this project got wrong on the way.

**Author:** zzpanic — [github.com/zzpanic](https://github.com/zzpanic). Reproductions on other
hardware are wanted more than anything else here: if you run this on a different card, please open
an issue with `tools/kvvalidate.py` output.
