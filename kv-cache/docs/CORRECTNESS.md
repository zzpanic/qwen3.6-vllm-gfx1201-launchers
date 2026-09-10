# CORRECTNESS -- KV-cache correctness suite

`correctbench.py` closes out "is the cache correct?" for every serving path that can be
exercised **against the running endpoint, with no model reload**. It imports and drives
the three validated harnesses (`tierbench.py`, `equivbench.py`, `mixedbench.py`); it does
not modify them.

## The property under test

> Serving any part of a prompt from any cache tier must produce **exactly** the output a
> full recompute of that same prompt produces — **and** a prompt that shares no
> legitimate chained prefix must **never** be served from another prompt's blocks.

Both halves matter. R3.15 was a violation of the second: the tier matched blocks whose
content was identical while their chained prefix was not. CT2 is the direct regression
test for it.

## Hard rules (enforced by the harnesses, honoured here)

- **No reload / restart / unload / `systemctl`.** Endpoint only. If the engine stops
  answering or starts refusing requests, the run aborts and writes the report; nothing is
  restarted.
- **`temperature=0` everywhere. `min_p` is never sent** — every request goes through
  `E.ask()` or `T.chat()`, neither of which sends `min_p` (this build returns 400 for it
  under spec decoding).
- **`T.budget_check()` is the wall-clock cap** (90 min from import). CT4 and CT6 pace
  their offsets/probes to it and report how many they reached.

## How to run

```
./correctbench.py --test ct5 --dry-run     print the plan, send nothing
./correctbench.py --test ct5 --yes          negative controls only (no GPU work) -- run this FIRST
./correctbench.py --test all  --yes         CT5 first, then CT1-CT4 + CT6 under one 90-min cap
./correctbench.py --test ct1  --yes         one test (CT5's gate still runs first)
./correctbench.py --test hammer --yes       repeat one or more offsets N times (see below)
```

Reports go to `$HOME/audit/stress/tierbench/CORRECT-<runid>.{md,json}`, matching the
existing harnesses. **Exit status 0 only if CT5's gate passes and CT1-CT4 all PASS with a
measured zero noise floor. CT6 never affects the exit status.**

## Status semantics

- **PASS** — the test reached its assertion and it held.
- **FAIL** — the test reached its assertion and it did **not** hold (a real finding), or
  the engine produced fatal log lines / died.
- **INCONCLUSIVE** — the test could not establish what it exists to test: the noise floor
  was non-zero (the run is void), a phase claimed MIXED but had `gpu_hits` or `ext_hits`
  at zero (it did not actually test the mixed path), or a cap cut it short. **Never
  reported as a pass.**
- **NOT RUN** — the CT5 gate failed (so CT1-CT4 are not trustworthy), or the run aborted.

## How to read a failure

- **Noise floor non-zero** (coldA vs coldB `tokens_identical=False` or
  `max|dlogprob|>0`): the decode pipeline itself is not bit-identical on this config.
  Every later comparison is then uninterpretable, so the test is **void**, not failed.
  Report it as such.
- **MIXED not built** (a mixed phase with `gpu_hits=0` or `ext_hits=0`): the mixed code
  path was never exercised. **INCONCLUSIVE**, never a pass. The `mixedbench` recipe
  (evict everything, re-warm only a head) is the one known to build it; a naive partial
  eviction overshoots straight to full OFFLOAD (`gpu_hits=0`).
- **A phase's tokens diverge from the cold reference**: that path served a divergent
  answer. This is the thing the suite exists to catch — report the first-divergent index
  and the `top3`/margin context `E.compare` attaches, so a near-tie flip (nondeterminism)
  is separable from a real wrong-token flip (corruption).
- **A CT2/CT3 counter is non-zero on the second prompt**: the prompt was served from
  another prompt's blocks despite a different leading block — the R3.15 defect. The GPU
  counter and the tier counter are asserted **separately and both at zero**, because on
  2026-09-10 the GPU counter read 0 while the tier served 98.1% of the prompt from
  another prompt's blocks; a combined assertion would have passed on a broken engine.

## Per test

### CT5 — negative controls (no GPU work; runs first)
Prove the instrument can report failure. A suite that has never failed has not been
tested. It checks, in memory:
- `E.compare(r, mutated)` must report divergence (a first-divergent index and a
  non-zero `max|dlogprob|`);
- `E.compare(r, r)` (identical) must **not** be flagged — this is what proves the
  control is **selective**, i.e. it can fail;
- `T.classify` on a zero-hit delta → `RECOMPUTE`; on a GPU-hit delta → `GPU`; on an
  external-hit delta → `OFFLOAD`;
- a deliberately repeated `cache_salt` must be detected as a hit (a short prompt, > one
  1648-token block, so there is actually something to cache).

**Gate: if CT5's gate does not pass, CT1-CT4 are NOT RUN — nothing else the suite says
counts.**

### CT1 — path equivalence matrix (the core test)
One ~90k-token prompt served every way, all compared to a cold reference:
`coldA` (fresh salt, RECOMPUTE) → `coldB` (fresh salt, same text, RECOMPUTE = the noise
floor) → `gpu` (immediate repeat) → drain/evict → `offload` → evict-all/re-warm a head
→ `mixed`. Assertion: every phase's tokens identical to `coldA` with
`max|dlogprob| == 0`, with `coldA`-vs-`coldB` establishing that zero is achievable in
this run.

### CT2 — salt isolation (the A1 / R3.15 regression test)
Two prompts, byte-identical body, different leading block, same `cache_salt`. Run A, let
it settle, run B, assert on B **separately and both at zero**: `prefix_cache_hits_total`
delta (GPU) and `external_prefix_cache_hits_total` delta (tier). Also compare B's output
to an independent cold run of B. This is the test that fails if R3.15 is ever reverted or
regressed.

### CT3 — cross-prompt contamination at depth
As CT2, but evict past the CPU tier first so the **tier is the only possible source**,
with a shared body long enough to span many blocks. Catches a tier that ignores the chain
only once the GPU cannot answer.

### CT4 — boundary sweep of the mixed path
R3.15 was a boundary defect, so vary the boundary. `tokens_per_chunk` is read from the
detected geometry (never hard-coded). Sweep the GPU-resident head across
`k * tokens_per_chunk` and ±1 token for several k (including k=1 and the largest k that
still leaves a tail). At each offset: no crash, no new boundary/unconfirmed-key assertion
in the engine log, and output identical to the reference. This is the test most likely to
find a residual defect; it is paced to the cap and reports how many offsets it reached.

## The contention guard — read this before believing any divergence

**A co-tenant request in the batch is enough to flip a near-tie token, with no cache
involved at all.** Measured directly: one prompt, 128 tokens at `temperature=0`, run alone
diverges **0/10** times; run with a single co-tenant in flight it diverges **10/10**, every
time at the same token index, with the losing candidate ahead by a margin of only 0.125.
Batch composition changes GEMM shapes and reduction order, the logits move slightly, and a
near-tie tips. The output is the model's own rank-2 candidate, not corruption.

So every assertion of bit-identical output needs to know whether it was alone.
`contention_guard(m0, m1)` snapshots `vllm:num_requests_running` and
`vllm:request_success_total` immediately before and after **each probe** and tags that
probe CONTENDED if anything else ran in the window. Contended probes are **recorded and
reported, never silently dropped** — dropping them would hide how much of a run was
contaminated.

The asymmetry is what makes this workable: **contention can only create a spurious
divergence, it can never hide a real one.** Therefore an uncontended run is the valid test
for a real defect, and a *contended* PASS is stronger evidence than an uncontended one.

`--test hammer` repeats a small number of offsets many times (`--reps`, `--offsets`), with
the guard on every rep, which is how an intermittent divergence gets attributed: if it is a
boundary defect it will show up in the uncontended reps. This is how CT4's three original
FAILs were settled — 9/9 bit-identical uncontended at the decisive offset, and the only
divergence anywhere was a contended recompute.

### CT6 — stride-boundary exactness probe (A2 / D4) — MEASURE, DO NOT ASSERT
The N=8 Mamba stride store is **approximate by design** at non-boundary positions.
`correctbench` compares a cached-serve output against a cold recompute at prefix lengths
that are, and are not, multiples of `8 * tokens_per_chunk`, and **records the divergence
as a number**. A non-zero divergence here is a **finding, not a failure**: it quantifies a
known, documented design limitation and is what decides whether the stride store is safe
to keep. **CT6 never fails the run and never sets a non-zero exit status. Do not "fix"
the stride store on the strength of a non-zero number.**

## If something goes wrong

If the engine returns errors, starts refusing requests, or the endpoint stops answering:
**stop, do not attempt recovery, do not restart anything.** The run aborts, writes the
report (with the partial phases and any fatal log lines), and ends. A hung engine is
recoverable by a human; a half-restarted one during an unattended run is a lost
afternoon.
