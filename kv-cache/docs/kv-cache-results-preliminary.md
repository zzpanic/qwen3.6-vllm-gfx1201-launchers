# Preliminary results — improved KV caching policy

> **Status: PRELIMINARY, SMALL SAMPLE.** One run, one machine, one workload, by the
> author. It is published as an early indication of the shape of the result, not as a
> benchmark. Nothing here has been repeated, held out, or independently reproduced, and
> the sample is small enough that any individual figure could move materially on a
> second run. Treat it as "this is roughly what happened", not "this is what it does".
>
> §4 lists the figures that do not currently reconcile with each other or with the
> measured storage density. They are flagged rather than corrected, because the raw
> observations are the author's and it is not for a reader — or an assistant — to quietly
> adjust someone's data.

---

## 1. Setup

| | |
|---|---|
| Hardware | AMD Radeon AI PRO R9700 (gfx1201, 32 GB GDDR6) |
| Software | vLLM 0.27.1 (radiance 0.9.3) + ggz14 + zzpanic github loader + custom patches |
| Model | Qwen 3.8-27B (24 Mamba + 8 attention layers) |
| Workload | An `opencode` session — code review plus testing/validation — with concurrent open-webui chat sessions using web search |
| Baseline | Unpatched vLLM 0.27.1, same workload |

The workload is worth dwelling on, because it is the point. This is **not a synthetic
benchmark**: it is a real mixed session, with an agentic coding tool and interactive chat
competing for the same cache. That is the case the tier exists to serve — long context
that comes back — and it is also why the sample is small and uncontrolled.

## 2. Results

Each row is the share of prompt tokens served from that tier.

| Tier | Size (GiB) | Hit ratio | Fetch latency | Equivalent tok/sec |
|---|---|---|---|---|
| GPU VRAM | 9.3 | 52% | 0 (local) | n/a (local) |
| CPU RAM | 24 | 11% | 1–2 s | 336,777 (promote) |
| Filesystem | 215 / 511 | 7% | 38 s | 2,214 (stage) |
| Recompute | — | 30% | 45 s | 1,993 (prefill) |
| **Total** | — | **100%** | — | — |

### What fits in each store

| Store | Capacity (tokens) |
|---|---|
| GPU | 228,737 |
| CPU RAM | 419,430 |
| Filesystem | 8,317,057 (36× the GPU cache; 3,506,944 stored) |

### Wall clock

Baseline cold recompute 45 s; all timings based on a 100,000-token prefix.

| | Latency | vs baseline |
|---|---|---|
| GPU hit | 0 s | baseline |
| RAM hit | 1–2 s | 95% faster |
| Disk hit | 38 s | 15% faster |
| Recompute | 45 s | baseline |

## 3. Interpretation

A two-tier KV offload (RAM + disk) serves long-context follow-ups from cache instead of
recomputing. **70% of the prompt is served from cache** — 52% VRAM + 11% RAM + 7% disk —
and only 30% needs recompute. The disk tier alone holds 8,317,057 tokens and recovers a
further 7% that would otherwise be a full recompute, at 38 s against a 45 s baseline.

The headline is the **11% from RAM**, not the 7% from disk. A RAM hit is essentially free
against a 45 s recompute; a disk hit saves 15%. The disk tier's contribution is real but
thin, and that matches what the component measurements say independently: the fs tier
reads at ~117 MB/s against a ~101 MB/s recompute break-even, a **1.16×** advantage. The
45 s → 38 s figure here is **1.18×** — arrived at by a completely different route, on a
different workload, and landing in the same place. That agreement is the single most
encouraging thing in this table, precisely because nothing was tuned to produce it.

## 4. Figures that do not yet reconcile

Flagged, not corrected. Each needs a second run to settle.

**a) Layer counts.** The setup says 24 Mamba + 8 attention. The model's own `config.json`
declares **48 linear-attention + 16 full-attention** layers (`full_attention_interval: 4`,
64 layers total). The stated counts are exactly half, so this looks like a transcription
error rather than a different model — but it should be corrected at the source before
anyone builds on it, because the attention-layer count is what sets storage density.

**b) Storage density, three different values.** The capacities imply three densities that
do not agree, and none matches the measured one:

| From | Implied bytes/token |
|---|---|
| CPU RAM: 419,430 tokens in 24 GiB | ~61,440 |
| Filesystem: 8,317,057 tokens in 511 GiB | ~65,977 |
| Filesystem: 3,506,944 stored in 215 GiB | ~65,846 |
| **Measured directly on this stack** | **33,808** |

The two filesystem figures agree with each other, which suggests they share a source. The
RAM figure differs from them and both differ from the measured 33,808 B/token — which is
itself well established: 16 full-attention layers × 2,048 B/token/layer plus one MTP
layer accounts for 103% of it, i.e. attention arithmetic alone explains the whole number.
At 33,808 B/token a 24 GiB tier holds ~762,000 tokens, not 419,430. Something is being
counted differently in at least two of these rows.

**c) The tok/sec column does not follow from the wall clock.** Against the stated
100,000-token prefix:

| Row | Stated tok/s | Implied by stated latency | Stated latency | Implied by stated tok/s |
|---|---|---|---|---|
| RAM promote | 336,777 | 50,000–100,000 (1–2 s) | 1–2 s | 0.30 s |
| Disk stage | 2,214 | 2,632 (38 s) | 38 s | 45.2 s |
| Recompute | 1,993 | 2,222 (45 s) | 45 s | 50.2 s |

No single prefix length reconciles the two columns, so they were most likely measured on
different prefixes or different runs and tabulated together. **The wall-clock table is the
one to trust** — it is the directly observed quantity, and its internal ratios (45 → 38 s
= 15.6% faster, matching the stated 15%) are self-consistent. The tok/sec column should
be regarded as derived and currently unverifiable.

**d) "Two-tier" vs three tiers.** The text says two-tier, the architecture is three
(GPU → RAM → disk). Both readings are defensible — there are two *offload* tiers behind
the GPU — but it is worth stating which is meant, since the staging constraint (nothing
reaches the GPU except through the RAM tier) is what makes the disk tier's numbers look
the way they do.

## 5. What this does not show

- **No quality measurement.** The recurrent-state store is approximate by design (stride
  N=8), and this run says nothing about whether the served answers were as good. That
  matters more here than in a normal cache benchmark, because a stale state degrades
  output rather than failing loudly.
- **No control over cache state between conditions.** A real mixed session cannot be
  held at a fixed cache fill, and comparing tiers at different fills is the specific trap
  that invalidated an earlier benchmark of this stack.
- **No repetition, no error bars.** One run.
- **One machine, one model, one workload.**

## 6. How to turn this into a real result

In rough order of value:

1. Fix the layer counts and settle the density question (§4a, §4b) — everything derived
   downstream depends on them.
2. Re-run with the wall clock and the throughput figures taken from the *same* prefix, so
   §4c reconciles.
3. Repeat enough times to state a range rather than a point.
4. Add a quality check alongside the speed one — the stride is an approximation and its
   cost belongs in the same table as its benefit.
5. Hold the cache fill identical across conditions, or state plainly that it was not.

`bench/tierbench.py` exists for step 5: it sizes each eviction from the tier capacities
it reads out of the engine's boot log and **refuses to report a phase whose tier state did
not come out as intended.**
