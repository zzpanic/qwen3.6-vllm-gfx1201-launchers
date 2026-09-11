# Three-tier KV-cache offload for a GDN hybrid on a single AMD card

**A GPU → RAM → disk KV-cache offload for Qwen3.8-27B (Gated-DeltaNet hybrid, MXFP4
weights, FP8 KV) served by vLLM 0.27.1 / radiance 0.9.3 on one AMD Radeon AI PRO R9700
(gfx1201, 32 GB, TP=1).**

In one sentence: on a single 32 GB card serving a 204,800-token context, **the cheapest
prefill is the one you do not do**, and this is a working — if unfinished — implementation
of not doing it.

---

## Synopsis: the problem this exists to solve

One card. One model hot at a time. A 204,800-token context window, and **two concurrency
slots**, which sounds like room for two workers and is not.

**The convenience-slot rule.** Measured on this box, the second stream is not free — and
not in the way people expect. Two decode streams produce the *same total* throughput as
one (aggregate 48.9 tok/s at one running request, 49.4 tok/s at two): continuous batching
buys nothing here, because DFlash2 speculative decoding at K=7 already makes a single
stream eight rows wide per step, so the batching headroom is spent before a second client
arrives. Worse, a *concurrent prefill* costs the decoding stream about **29%**, and the
scheduler's chunk budget is only ~2,036 tokens, so one 65k prefill is ~32 chunks and 50–120
seconds of taxed decode for whoever else is on the card. In one observed window there were
38 preemptions across 83 requests and the prefix-cache hit rate fell from 88.6% to 65.1%.

So the standing policy here is: **agents issue one request at a time; the second slot is
for short convenience requests only** — a chat message, a quick lookup — never a second
long context. The criterion is *context length*, not "human vs agent". `max_num_seqs: 2`
is correct for that policy: 1 kills the convenience slot, 3+ invites preemption.

**Which is exactly why the cache tier matters.** If you cannot buy concurrency, the only
remaining lever for multi-agent work on one card is **not recomputing what you already
computed**. Agentic coding tools are the ideal case and the worst case at once: every turn
re-sends a prefix that is almost entirely identical to the last one, tens of thousands of
tokens of it, and a 100k-token cold prefill costs ~45 s of wall clock during which the
card belongs to nobody else. Serve that prefix from a cache and the turn starts
immediately, the convenience slot stays usable, and two agents can share one card serially
without either of them paying for the other's prefill.

That is the whole design intent: **trade RAM and cheap disk for prefill time, so that a
single card behaves like it has more of the one resource it cannot buy.**

---

## ⚠️ Status: PROOF OF CONCEPT

Read this before you quote any number from this repository.

This is a **proof of concept**. It demonstrates that a three-tier KV-cache offload can be
made to work on a GDN hybrid model on a single consumer AMD card, and it is measured doing
so. **That is the whole of the claim.** It is not production code and it is not a
research-grade reproduction. It carries one **deliberate approximation** — the mamba N=8
stride, described under [Future work](#future-work), which serves recurrent state up to
eight chunks stale — and that one is by design rather than a defect. The defects that were
found are not still open: they were fixed, and then the fixes were tested. What follows is
the state of that work, not a disclaimer.

**Correctness status, 2026-09-10.** A defect that made the offload tier serve KV
belonging to a *different prompt* was found and fixed (R3.15). It was not a benchmark
problem: unfixed, the tier served **98.1%** of tokens on prompts that shared no legitimate
prefix, while the GPU prefix cache — looking at the same prompts in the same window —
correctly refused **all** of them. The fix is confirmed live by `check-r315-boot.sh`, and
a controlled reverse test isolating that one file reproduces the defect on demand.

**What has since been attacked, and held.** The remaining ways the cache could be wrong
were enumerated and tested one at a time:

| The suspicion | Outcome |
|---|---|
| A disk-tier hit might not reproduce a cold recompute | **Exact.** An 85,696-token disk hit was bit-identical |
| The mixed local+external boundary might corrupt output (three CT4 failures) | **Not a defect.** 9/9 bit-identical uncontended; every original failure was co-tenancy moving a near-tie logit |
| `cache_salt` might not reach the tier's key, letting one tenant read another's blocks | **Honoured.** The isolation claim is **refuted** — the salt chains into the offload key |
| The correctness harness itself might be incapable of failing | **It fails when it should.** A negative-control gate blocks the suite if the instrument cannot detect a divergence it was handed |

A co-tenant can *create* a spurious divergence but never hide a real one, so the uncontended
runs above are the valid test — that asymmetry is in [`CORRECTNESS.md`](docs/CORRECTNESS.md).
Each row's evidence, and the one condition that would reopen it, is in
[`kv-cache-closed-decisions.md`](docs/kv-cache-closed-decisions.md), which also lists the
claims this project has published and withdrawn.

**What is still not validated.** The framework is **assumed correct** — enough to work on
and to benchmark against — and **not yet validated correct**. What is owed is a named list,
[`status-2026-09-10.md`](docs/status-2026-09-10.md) §4; the two that matter most are showing
the *generated tokens* differ under a deliberately re-armed defect, and reproducing both
defects on a **stock** vLLM 0.27.1 build, which is what blocks filing upstream.

**Every number in this repository taken before 2026-09-10 is uncitable** — the harness was
sound, the engine under it was not. Re-measuring them is outstanding work, not a formality.

**Where the effort stands.** Roadmap stage 1 is half delivered. The per-tier metrics exist,
are patched into the engine and are verified live — `tools/tierreport.py` turns one scrape
of your own traffic into a verdict on whether the RAM tier is the right size, whether the
disk is too slow, and whether the layered cache is adding value at all — and, where the
counters cannot honestly answer one of those, says so instead of answering anyway. The
controlled A/B harness with a genuinely cold arm, the other half, does not exist yet.

It is published at this maturity **deliberately**. Several people want this capability;
the author has neither the time nor the specialist expertise to carry it to completion
alone, and a working-but-flawed starting point that says exactly where it is flawed is
more useful to those people than nothing. **It is a starting point, not a product.**
[Future work](#future-work) is the ordered list of what it would take to make it real, and
[Point your own coding agent at this](#point-your-own-coding-agent-at-this) is how to
start.

---

## Overview: what it actually does

Three tiers, each a fallback for the one above:

| Tier | Where | Size here | Notes |
|---|---|---|---|
| **L1** | GPU, fp8 KV | ~228,700 tokens / 9.3 GB | vLLM's own prefix cache, pinned |
| **L2** | RAM, `/dev/shm` | 24 GiB ≈ 762,000 tokens | pre-faulted + pinned; **ARC** eviction |
| **L3** | disk, dedicated fs | 511 GiB | `O_DIRECT`, 8 read / 4 write threads |

Wired through `--kv-transfer-config` (`TieringOffloadingSpec` / `OffloadingConnector`).

**The architecture is strictly staged and this is the single most important thing to
understand about it.** From `tiering/manager.py`: *"Primary tier is the gateway —
secondary tiers cannot access GPU memory directly; all data flows through the CPU primary
tier"* and *"blocks in secondary tiers must be promoted to the primary tier before GPU can
access them"*. Consequences:

- The disk tier **cannot** serve the GPU on its own. Turn off the CPU tier and the disk
  tier turns off with it.
- Disk and RAM **contend for the same region**. `lookup()` returns MISS not only when a
  block is absent but also when *"primary is full and cannot accept a promotion"*.
- There is **no DMA path** disk → VRAM. Foreclosed three ways here: no cuFile/GPUDirect/
  DMA-BUF anywhere in the vLLM tree; the cache device is virtio_blk inside a VM, so there
  is no real PCIe endpoint to peer with; and GPUDirect Storage is NVIDIA/cuFile and does
  not exist for gfx1201.

**Storage density is at the architectural floor.** 33,808 bytes/token measured, and
attention arithmetic alone accounts for 103% of it (16 full-attention layers × 2,048
B/token/layer, plus one MTP layer). FP8 is already one byte; GQA is already 6:1; the Mamba
layers are already stored 1-in-8. There is no density work left to do here — a fact worth
knowing before anyone spends a week on compression.

Eight house patches make it work; they are in [`patches/`](patches/) with their apply
order, and [`kv-cache-current-implementation.md`](docs/kv-cache-current-implementation.md)
explains what each one does and which environment variable gates it.

---

## The benchmark

Two numbers can be stated, and both come with their limits attached.

> ⚠️ **Both were measured before the R3.15 correctness fix (2026-09-10)** and are
> retained here as method, not as results. Some fraction of the "tokens served from the
> tier" below were served against unconfirmed keys — that is, served *wrongly*. The
> direction of the error is known, its size is not. These must be re-measured on the fixed
> engine; see [`status-2026-09-10.md`](docs/status-2026-09-10.md) §4.

**The tier earned its keep.** Approximately **2.45M tokens served from the RAM and disk
tiers**, ≈ **26 minutes of prefill avoided** at the honest cold rate (~1,555 tok/s).

**A preliminary end-to-end run** on a real mixed workload — an `opencode` session doing
code review and testing, with concurrent chat — measured per-tier hit shares of the prompt
tokens:

| Tier | Size (GiB) | Hit ratio | Fetch latency | vs 45 s cold recompute |
|---|---|---|---|---|
| GPU VRAM | 9.3 | 52% | 0 (local) | — |
| CPU RAM | 24 | 11% | 1–2 s | ~95% faster |
| Filesystem | 215 / 511 | 7% | 38 s | 15% faster |
| Recompute | — | 30% | 45 s | baseline |

**70% of prompt tokens served from cache.** The headline is the **11% from RAM**, not the
7% from disk: a RAM hit is essentially free against a 45 s recompute, while a disk hit
saves only 15%.

> **This is one small-sample run and is labelled as such.** One machine, one workload, one
> pass, by the author, nothing held out or reproduced.
> [`kv-cache-results-preliminary.md`](docs/kv-cache-results-preliminary.md) carries it in
> full — including **§4, the figures in it that do not yet reconcile** with each other or
> with the measured storage density. They are flagged there rather than quietly corrected.

The encouraging part is not the headline but a coincidence. The disk tier's speedup here
(45 s → 38 s, **1.18×**) lands on the same number the component measurement reached by a
completely different route: the fs tier reads at ~117 MB/s against a ~101 MB/s recompute
break-even, **1.16×**. Different workload, different instrument, same answer, and nothing
was tuned to produce it.

**What must never be cited.** The BetterBench numbers for this stack are polluted. vLLM
matches the prefix cache by block **content**, not by chained prefix, and BetterBench's
"cold (nonce)" run varies only block 0 — so the body of the prompt self-caches and the
"cold" arm is not cold. Everything else in these documents is either a component
measurement (documented with its method) or invalid (documented as invalid).

---

## Future work

The three limitations that matter most, then the order of work.

- **The mamba stride (N=8) is approximate by design.** The store keeps only every Nth
  chunk's recurrent state, and a lookup rounds **down** to the nearest kept boundary, so
  the served state can be up to 8 chunks stale. It is a good approximation because the
  Gated-DeltaNet gate gives the state finite effective memory — recent tokens dominate —
  but it *is* an approximation, and it measurably raises the needle-in-a-haystack failure
  rate.
- **A failed offload load kills EngineCore.** `assert transfer_result.success`, and
  `OffloadingConnector` exposes no `get_block_ids_with_load_errors()`, so
  `kv_load_failure_policy=recompute` is inert here. This is why the reaper's `MIN_AGE`
  floor is a hard safety property and not a tuning knob.
- **The disk tier is barely worth doing on this hardware.** 1.16× over recompute. Whether
  that is the device or the implementation is not yet established.

### The roadmap — in the author's intended order

Each stage's output is the next stage's input, and **the order is deliberate**: nothing
after stage 1 can be judged without stage 1, and the accuracy work comes before the speed
work because it is pointless to optimise a path that is still returning approximate state.

**1. Benchmark hooks, metrics, and a reproducible harness. First.**
Tidy up the instrumentation that already exists (the lookup-outcome and instrumentation
patches) into a coherent metrics surface, and write a **reproducible cache-metrics script**
in the spirit of BetterBench but aimed at the tier rather than at raw token throughput:
hit share per tier, promotion latency, `load_bytes`, read:write ratio, and the recompute
avoided. **It must not repeat BetterBench's mistake** — vLLM matches the prefix cache by
block *content*, not by chained prefix, so varying a nonce in block 0 leaves the body of
the prompt self-caching and the "cold" arm is not cold. Getting the cold arm genuinely
cold is the hard part of this stage and the reason it comes first: every claim below it is
unfalsifiable until it exists.

**Part of this now exists.** `patches/patch_kv_offload_tier_report.py` adds 19
`tier`-labelled series — per-tier hit blocks and tokens, load/store bytes, seconds, ops and
latency histograms, capacity, an occupancy *distribution*, reads-before-evict,
eviction-to-reuse, would-have-hit and prefill stall time — and
[`tools/tierreport.py`](tools/README.md) turns one `/metrics` scrape into `results.md` +
`results.json` with a verdict per operator question: too much RAM, too little RAM, disk too
slow, disk actively hurting, and whether the layered cache is adding value at all. The
design note is [`docs/tier-report-metrics-plan.md`](docs/tier-report-metrics-plan.md).

It is deliberately **lifetime-cumulative and read-once**, which sidesteps the cold-arm
problem rather than solving it: it reports what the operator's own traffic actually did, so
there is no synthetic corpus to get wrong. Everything a *judgment* rests on is a monotonic
counter or a histogram (the only gauges are `tier_capacity_bytes` and `tier_used_bytes` — configuration, not judgment), because a gauge scraped once off a long-lived server
says almost nothing. What is still missing from this stage is the other half — the
controlled A/B harness with a genuinely cold arm, which is what would let a *change* be
measured rather than a deployment described.

Two specific jobs belong here, both already scoped by earlier work:

- **Confirm or kill the busy-wait.** The strongest open lead in the repository is that the
  deferral loop is a busy-wait costing ~1.9 ms an iteration — ~19 s of spinning for 103 ms
  of real disk I/O. It is fitted to three points and **not yet confirmed against
  scheduler-step counters**. If it holds, it explains why a request that is 92% served from
  cache still loses to a cold recompute, and it is a bigger prize than any tuning below.
- **Fix the correctness check.** The current one passes on a five-character `ACK` response
  and therefore proves nothing about tens of thousands of tokens of replayed KV. It needs a
  prompt demanding a long, content-dependent answer before a pass means anything.

[`kv-cache-historical.md`](docs/kv-cache-historical.md) §7 lists every other instrument
defect found the hard way. Read it before designing the harness, not after.

**2. Continue the review of existing work, and produce an implementation plan.**
[`kv-cache-references.md`](docs/kv-cache-references.md) is the review so far — every PR,
paper and blog already assessed, each with a ruling. Continue it, then write the plan.
Concretely, this includes reconciling the eight house patches against current
vLLM/radiance HEAD and **deleting each one in favour of the upstream implementation
wherever one exists**: at least one already has an upstream counterpart (the eagle-groups
fix is PR #55390; the fs fanout was ported from PR #49225). Two open PRs already measure as
directly applicable and should be handled first: **#54327** adds bounded capacity and LRU
eviction to the fs tier at **100% applicability** — it would retire the mandatory external
reaper — and **#54743** adds the filtered-group primitive the stride work should be built
*on* rather than beside. The measured applicability table is
[`kv-cache-references.md`](docs/kv-cache-references.md) §3a. *Avoid reimplementation* — a
house patch duplicating merged upstream work is a liability, not an asset: one more thing
to rebase, and it will silently diverge. Score upstream work by **applicability, not by
merge status**; an unmerged PR that fits is worth more than a merged one that does not.

**3. Tunable accuracy, and a proof of concept of the most promising quality options.**
Correctness here is not binary — it is a dial, and right now the dial is welded in one
position. Expose the accuracy/cost trade-offs as **configuration toggles** rather than
constants (the mamba stride `N` being the obvious first one; the store threshold and the
pending-is-miss behaviour are others), so that a deployment can choose its point on the
curve and a benchmark can sweep it. Then build a proof of concept of the most promising
quality options. The strongest candidate is the **exactness fix**: replaying the ≤
one-block gap from the stride checkpoint, which turns the mamba stride from an
approximation into an exact reconstruction. It is designed and unbuilt, and it is the
single well-described gap to the full method — see
[`kv-cache-future-work.md`](docs/kv-cache-future-work.md).

**4. Refactoring.**
These patches were written one at a time, each to answer a specific question, and it
shows. They monkey-patch by string surgery. They carry an implicit dependency **order**
documented only in the launcher. Their gating environment variables are inconsistent in
naming and in whether `0` or `1` means "upstream behaviour". This wants to be a single
coherent module with an explicit interface, not eight scripts in a trench coat. It lands
here rather than earlier because stages 2 and 3 decide how much of it survives to be
refactored.

**5. Speed optimisation.**
Nothing here has been tuned; it has only been made to work. The known ceilings are
measured and documented — the fs tier's 1.16×; the strictly staged promotion path, so disk
and RAM contend for the same region; no DMA path from disk to VRAM on this hardware.
**Which of those are real limits and which are merely untuned is, in most cases, not yet
established** — and stage 1 is what settles it.

**6. Normal software engineering.**
Code review, tests, CI, packaging, a real release. **None of it has happened.** There is no
test suite; correctness has been established by hand, per change, against a live endpoint.

### Where this is meant to end up — upstream, and general

The target of all six stages is **upstream vLLM**, not a fork that lives here permanently.
That is a constraint on how the work is done, not a wish about where it might land:

- **The model is the vehicle, not the point.** Everything here is measured on
  Qwen3.8-27B-MXFP4 because that is what this machine serves, and one day that checkpoint
  will be old hat. The parts worth keeping are the ones that outlive it: a tier that stores
  and serves KV, a stride that checkpoints recurrent state, a promotion path between tiers.
  Where a mechanism genuinely *is* architecture-specific — the mamba stride is, it exists
  only because Gated-DeltaNet has a recurrent state to checkpoint at all — it should be
  selected from the model's own declared layout (which KV groups are attention, which are
  recurrent), never from the model's name. The eagle-groups patch is the cautionary case in
  miniature: it exists because vLLM's own draft-group annotator is hard-gated to one model
  family, and every other model silently got the wrong answer.
- **Hyper-specialisation is an explicit non-goal.** It is possible to chase the last few
  percent by tuning to one checkpoint on one card until nothing else loads. That is
  deliberately not what this is for. vLLM is a general serving runtime and should remain
  one; a change that buys this model 5% and costs another model its ability to start is a
  bad trade at any margin.
- **Not breaking existing capability is a requirement of every stage,** including the ones
  that look like housekeeping. It is why every behaviour-changing patch here is behind an
  environment gate whose unset state is upstream behaviour, and why the eight patches are
  to be *deleted* in favour of upstream implementations wherever one exists rather than
  maintained alongside them. Stage 6's tests and CI are not tidiness for its own sake; they
  are the price of admission for anything that asks other people to run it.

None of this is aimed at the specialised forks. The gfx1201 work this repository stands on
— radiance, libr4d, the MXFP4 build — is what makes this card usable at all, and the
multi-GPU tensor-parallel work in that cluster is genuinely impressive engineering. It is
simply a different aim: those forks exist to make one architecture excellent on hardware
the runtime otherwise ignores, and this wants to end up in the runtime everybody already
has.

---

## Point your own coding agent at this

This repository is written to be handed to an agent, not just read. The documents are
deliberately long and state their own reasoning and their own doubts, because that is what
an agent needs in order to not repeat work that has already been done and discarded.

**1. Get it.**

```bash
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers/kv-cache
```

**2. Prime the agent.** Give it, in this order:

| Read | Why |
|---|---|
| this `README.md` | the claim, its limits, and the shape of the work |
| [`docs/kv-cache-handover.md`](docs/kv-cache-handover.md) | the full state of play and the resume path |
| [`docs/kv-cache-known-issues.md`](docs/kv-cache-known-issues.md) | **before writing anything** — the hard "never do X" list |
| [`docs/kv-cache-references.md`](docs/kv-cache-references.md) | every PR, paper and experiment already reviewed, each with a ruling |
| [`docs/kv-cache-historical.md`](docs/kv-cache-historical.md) | the experimental record — including the claims that were **retracted**, and why. Read it before re-running an experiment that looks obvious |
| [`docs/kv-cache-closed-decisions.md`](docs/kv-cache-closed-decisions.md) | **the register of what is settled**, each row carrying the one condition that would reopen it. Read this before proposing work; the historical record is the reasoning behind it |
| [`patches/README.md`](patches/README.md) | what the eight patches need in order to run, which two are fatal on failure, and the env gate on each |
| [`tools/README.md`](tools/README.md) | the tier sizing/speed report: what it needs, how to test it with no engine, and how to read `--calibrate` honestly |

A prompt that works: *"Read kv-cache/README.md, then docs/kv-cache-handover.md and
docs/kv-cache-known-issues.md, check docs/kv-cache-closed-decisions.md for whether the
question is already settled, and skim docs/kv-cache-historical.md for what has already
been tried and retracted. This is a proof of concept with known correctness errors,
and the roadmap in the README is in the author's intended order. Start at stage 1: audit
what cache metrics the existing patches already expose, and propose a reproducible harness
that measures per-tier hit share and prefill avoided with a genuinely cold arm — read
bench/README.md first for why 'cold' is the hard part. Do not write code in this pass."*

**3. Get the plain MXFP4 build serving first.** `startup-qwen3.8-27b-mxfp4.sh` at
the repository root is the same model on the same card with **no cache work at all**,
and every knob in it carries the measurement that chose it. It is the shorter path to
a working engine, and if it does not serve, nothing here will either. The cache
launcher is that same structure plus a seven-item delta — and its tuning defaults are
copied from it when the release is built, so the two cannot drift. Base your own work
on that structure: `kv-cache/launcher/README.md` lists the delta in full, and it is
the whole job if you want to add the cache to a launcher of your own.

**4. Pick a stage from [Future work](#future-work).** Stage 1 is not optional
throat-clearing — until the metrics harness exists, nothing you change afterwards can be
shown to have helped. Three good first tasks, in increasing size:

- **Small, self-contained:** normalise the patch gating variables so that unset always
  means upstream behaviour, and move the apply order and its dependencies into code rather
  than leaving them stated in `patches/README.md` and the launcher's comments.
- **Medium, and the actual starting point:** the reproducible cache-metrics harness of
  stage 1 — with a *genuinely* cold arm. Read [`bench/`](bench/)'s README first; the
  content-vs-prefix-hash trap is documented there and it is what makes this non-trivial.
- **Large, the real prize:** the exactness fix inside stage 3. It is designed and unbuilt,
  and it is the difference between "approximate" and "correct".

**5. Run it.** [`docs/kv-cache-operations.md`](docs/kv-cache-operations.md) is the runbook
— turning the disk tier off, resizing `/dev/shm`, resizing the KV tier, each with commands,
verification and undo. Read [Hard requirements](#hard-requirements--not-advisory) first;
two of them fail *silently* if ignored.

**6. Measure it.** [`bench/`](bench/) ships the harnesses but **no result files**,
deliberately — single-machine numbers, some of them known polluted. Its README documents
the content-vs-prefix-hash trap that invalidated an earlier round of measurement, which is
the mistake most likely to waste your first day.

Different hardware is welcome and wanted. Nothing here has been reproduced on another
machine, and the conclusions should be assumed hardware-specific until they are.

---

## Documentation map

Read in this order. Every document is written to be actionable without opening the next.

| Document | What it holds |
|---|---|
| **[`kv-cache-handover.md`](docs/kv-cache-handover.md)** | **Read first.** The map, the current state, and the resume path. |
| [`kv-cache-results-preliminary.md`](docs/kv-cache-results-preliminary.md) | **Preliminary results.** One small-sample run on a real mixed workload: 70% of prompt tokens served from cache. Includes §4, the figures in it that do not yet reconcile. |
| [`kv-cache-historical.md`](docs/kv-cache-historical.md) | **The experimental record.** Every hypothesis, measurement, correction and retraction, in order — so you do not re-run a settled experiment or build on a withdrawn claim. |
| [`kv-cache-closed-decisions.md`](docs/kv-cache-closed-decisions.md) | **The register of closed decisions.** The same closures as a scannable table rather than a narrative, with the evidence and the reopen condition for each. Consult it first; it exists so the narrative documents do not have to be read end to end to find out whether a question is already answered. |
| [`kv-cache-operations.md`](docs/kv-cache-operations.md) | The runbook: turn the disk tier off, resize `/dev/shm`, resize the KV tier. Each with commands, verification and undo. |
| [`kv-cache-current-implementation.md`](docs/kv-cache-current-implementation.md) | What is actually built and running: the three tiers, the eight patches with gates and order, the two-layer GC, the serve invocation. |
| [`kv-cache-known-issues.md`](docs/kv-cache-known-issues.md) | Every problem, gotcha and limitation as Symptom / Root cause / Impact / Status, by severity, plus the hard "never do X" list. |
| [`kv-cache-future-work.md`](docs/kv-cache-future-work.md) | The plan, the reuse-refresh mechanism and its `L` analysis, and the limitations to state up front. |
| [`kv-cache-references.md`](docs/kv-cache-references.md) | Every PR, paper, blog and local artifact reviewed, each with a status and a ruling. |
| [`status-2026-09-09.md`](docs/status-2026-09-09.md) | The investigation snapshot: the BetterBench pollution finding, the DASC/DAMP findings, the dead ends. |
| [`cache-preemption-patch-plan.md`](docs/cache-preemption-patch-plan.md) | The deeper patch plan. **Revision 3 supersedes parts of Revision 2 (and the original) — read Revision 3 first, then Revision 2.** |
| [`README.house.md`](docs/README.house.md) | The original house-files README: how `/house` is wired, `tierbench`, the instrumentation patch. |

---

## The launcher

`../startup-qwen3.8-27b-kvcache.sh` serves the entry **`qwen3.8-27b-kvcache`**. It is the
KV-cache configuration in one file, every knob documented inline with the reasoning and
the measured cost of changing it. Run `../startup-qwen3.8-27b-kvcache.sh -h` for the
full list.

It is a **thin wrapper**, deliberately: it sets environment and execs the real launcher,
which stays the single source of truth for how the model is served. Nothing about the
model, kernels, drafter or vLLM invocation is duplicated.

It runs on its own container (`qwen38-27b-kvcache`) and its own on-disk block tree
(`blocks-kvcache`), so a KV-cache experiment can never quietly pollute the production
entry's cache or its numbers. That isolation is not paranoia — it is the direct lesson of
the BetterBench pollution above.

**It takes the whole GPU.** It and the production entry cannot be loaded together.

---

## Hard requirements — not advisory

- **`PYTHONHASHSEED` must be pinned** (the launcher pins it to `0`). Block filenames are
  content hashes chained from `NONE_HASH`, which is seeded from `os.urandom(32)` when the
  variable is unset. Leave it unset and every restart hashes the same tokens to
  *different* filenames, orphaning the entire on-disk cache — silently, at a 100% miss
  rate, with no error anywhere.
- **The fs tier never deletes.** `tiering/fs/manager.py` has no capacity, quota or TTL
  parameter and exposes no eviction hook. **An external reaper is mandatory**
  (`kvcache-reap.sh` + its systemd timer). Without it the filesystem fills. **There is an
  upstream fix for this and it applies at 100%** — PR #54327 adds bounded capacity and LRU
  eviction to that exact file. Testing it is one of the first jobs on the list; if it holds,
  the reaper becomes legacy.
- **Never delete a young block.** The reaper's `MIN_AGE=90min` floor is a safety property:
  reaping an in-flight block kills EngineCore (see the limitations above).
- **`O_DIRECT` in both directions.** Spare RAM cannot act as a read cache in front of the
  fs tier. The only productive home for spare RAM is the primary tier.

---

## The development machine — and why it shaped every choice

Everything here is single-machine, single-configuration, and the machine is a modest one.
That is not an apology: **most of the design decisions in this repository are downstream of
these constraints**, and they only make sense if you can see them.

| | |
|---|---|
| Host CPU | Intel Core i5-7600 — **4 cores, 4 threads**, no SMT |
| Host RAM | 64 GB DDR4-2400 (4 × 16 GB) |
| Host OS | TrueNAS Community Edition 25.10.4 — the NAS *is* the hypervisor |
| Motherboard | ASUS P10S WS — **PCIe 3.0 x16** to the GPU |
| GPU | AMD Radeon AI PRO R9700, gfx1201 / RDNA4, 32 GB, TP=1, **passed through to the guest** |
| Guest | a TrueNAS VM (QEMU/KVM, i440FX + OVMF) with **40 GiB RAM** (39.17 GiB usable) and the GPU on passthrough |
| Guest OS | Ubuntu 24.04.4 LTS, kernel 6.8 — everything in this repository runs *inside* the guest |
| Cache filesystem | a zvol on a **TrueNAS raidz array of 4 × 4 TB drives**, presented to the guest as `virtio_blk` |
| Model | Qwen3.8-27B MXFP4 weights + FP8 KV, 16 full-attention + 48 Gated-DeltaNet layers |
| Serving | vLLM 0.27.1 + radiance 0.9.3 overlay, R4D attention, DFlash2 spec decoding |

### What each constraint decided

**Four cores, four threads.** The engine, the O_DIRECT reader threads, the reaper timer and
the client all share four hardware threads with no SMT. This is why a **busy-wait** in the
deferral loop is not a minor inefficiency here: ~1.9 ms per deferral × 10,000 deferrals is
~19 seconds of spinning for 103 ms of actual disk I/O (see
[`kv-cache-historical.md`](docs/kv-cache-historical.md) §3.3). On a 32-thread host the same
loop would be cheaper and might never have been noticed — which is a reason to trust the
finding, not to discount it.

**40 GiB in the guest, and the tier is pinned.** The CPU tier is **pre-faulted and mlocked**,
so it can never be swapped or reclaimed under pressure. Of 39.17 GiB, vLLM's own non-tier
footprint — weights staging, Python heap, HIP host allocations, page cache — measures about
**7 GiB and spikes during prefill**. That is the whole reason the operations runbook
recommends a 28 GiB tier rather than the 30 GiB the arithmetic appears to allow: the
headroom is not spare, it is working memory. There is no memory hotplug in this guest, so
the ceiling cannot be raised at runtime.

**PCIe 3.0 x16.** ~15.75 GB/s theoretical to the card. The measured CPU→GPU promotion copy
of **11.8 GB/s** is therefore close to the bus limit and is *not* a software problem — do
not go looking for one. It also means the RAM tier is about as fast as it can be here, which
is why a RAM hit is effectively free against a 45 s recompute and why the RAM tier, not the
disk tier, is the headline of the results.

**A raidz array of spinning disks behind a zvol behind virtio_blk.** This is the single most
consequential constraint in the repository. raidz gives you roughly one disk's worth of
random-read IOPS regardless of width, and the storage path adds a zvol and a virtio layer on
top — and the host running that storage stack is the *same* four-thread box, because TrueNAS
is the hypervisor, so ZFS checksumming, ARC and raidz parity are spending the same cores the
engine is. The measured result is **~117 MB/s**, against a **~101 MB/s** recompute break-even —
a **1.16×** advantage, which is why the disk tier is real but thin, and why it contributes
7% where the RAM tier contributes 11%. **An NVMe device is roughly 19× this**, and the disk
tier's whole economic case changes on that hardware. If you have NVMe, the most interesting
thing you can do with this repository is re-run the tier measurement and tell everyone what
happened.

**GPU passthrough into a VM.** The cache device has no real PCIe identity from the guest's
point of view, so there is no peer to DMA with even in principle — one of the three
independent reasons there is no disk→VRAM path here. It is also why **any host-limited
property must never be sized from inside the guest**: the guest misreports the CPU model and
the PCIe link speed, and believing it will send you down a false trail.

**Nothing here has been reproduced on other hardware,** and every conclusion should be
assumed hardware-specific until it is. Different hardware — more cores, more RAM, NVMe, a
bare-metal host — is wanted, and would settle several of the open questions above outright.

---

## Licence and attribution

The house patches and documents here are the author's own work, built on and against vLLM
and the radiance overlay; upstream code carries its own licences. Where a patch was ported
from an upstream PR it says so, with the PR number, in the patch file itself.

**Author:** zzpanic — <zzpanic@gmail.com>, [github.com/zzpanic](https://github.com/zzpanic).

Reproductions on other hardware are wanted more than anything else here. If you run this
on a different card, a different storage stack, or an NVMe device, the results are worth
sending on whether they agree or not — a disagreement is more useful than a confirmation,
because nothing in this repository has been reproduced anywhere but the one machine
described above. Issues on the repository are the better route for anything others should
see; mail is for what does not belong in public.
