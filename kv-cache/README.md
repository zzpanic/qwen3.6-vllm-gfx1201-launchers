# Three-tier KV-cache offload for a GDN hybrid on a single AMD card

**A GPU → RAM → disk KV-cache offload for Qwen3.8-27B (Gated-DeltaNet hybrid, MXFP4
weights, FP8 KV), served by vLLM 0.27.1 / radiance 0.9.3 on one AMD Radeon AI PRO R9700
(gfx1201 / RDNA4, 32 GB, TP=1).**

On a single 32 GB card serving a 204,800-token context, **the cheapest prefill is the one
you do not do.** This is a working, correct, unoptimised implementation of not doing it —
measured over ~14 hours of real agent work, in which **81.8% of prompt tokens were served
without recomputation.**

---

## The wall this exists to get past

Run two agents on one card and the second evicts the first — which then re-prefills its
prefix from scratch. That is the single-card concurrency wall, and it is the reason this
work exists.

One card. One model hot at a time. A 204,800-token context, and **two concurrency slots**
that sound like room for two workers and is not. Measured on this box, the second stream is
not free — and not in the way people expect. Two decode streams produce the *same total*
throughput as one (48.9 tok/s at one running request, 49.4 tok/s at two): continuous
batching buys nothing here, because DFlash2 speculative decoding at K=7 already makes a
single stream eight rows wide per step, so the batching headroom is spent before a second
client arrives. Worse, a *concurrent prefill* costs the decoding stream about **29%**, and
the scheduler's chunk budget is only ~2,036 tokens, so one 65k prefill is ~32 chunks and
50–120 seconds of taxed decode for whoever else is on the card. In one observed window there
were 38 preemptions across 83 requests and the prefix-cache hit rate fell from 88.6% to
65.1%.

So the standing policy is: **agents issue one request at a time; the second slot is for
short convenience requests only** — a chat message, a quick lookup — never a second long
context. The criterion is *context length*, not "human vs agent". `max_num_seqs: 2` is
correct for that policy: 1 kills the convenience slot, 3+ invites preemption.

**If you cannot buy concurrency, the only remaining lever for multi-agent work on one card
is not recomputing what you already computed.** Agentic coding tools are the ideal case and
the worst case at once: every turn re-sends a prefix that is almost entirely identical to
the last one, tens of thousands of tokens of it, and a 100k-token cold prefill costs ~45 s
of wall clock during which the card belongs to nobody else. Serve that prefix from a cache
and the turn starts immediately, the convenience slot stays usable, and two agents can share
one card serially without either of them paying for the other's prefill.

That is the whole design intent: **trade RAM and cheap disk for prefill time, so that a
single card behaves like it has more of the one resource it cannot buy.**

---

## What it does, in the one number that does not move

**81.8% of prompt tokens were served without recomputation.** Over **31.5M** prompt tokens
of real agent coding work — two to three concurrent agents on this one card for ~14 h — the
GPU prefix cache caught 64.1% of them, the offload tiers the other 17.7%, and only 18.2%
was recomputed. It improved through the run (recompute was 21.8% early on). This is the
robust number: it does not depend on *which* layer did the work, so it does not wobble when
the workload shifts.

Two more that do not depend on the workload at all, and are the ones to hold on:

| tier | token read rate | bandwidth | what the wait fraction means |
|---|---|---|---|
| fs (disk) | **3,868 tok/s** | 228 MB/s | 5.3% — the tier is device-bound, not starved |
| CPU (RAM) | **337,408 tok/s** | 11.8 GB/s | 0.0% — essentially free against a recompute |

And the correctness fact that matters most: an **85,696-token disk hit was bit-identical to
a cold recompute** — the disk tier serves *exact* state, not an approximation.
(LMCache's own documentation concedes its disk tier is not bit-exact.) Across the run there
were **5,423 promotions, 0 refused**: the tier never declined to hand back a block it held.

---

## Why this might work for your card

**This runs on RDNA4 — gfx1201, a consumer/workstation AMD card.** That is the line that
should not be buried: for that class of hardware, this is, as far as we can tell, the only
hybrid KV offload there is. LMCache — the mainstream KV offload for LLMs — ships its ROCm
build for Instinct only (gfx942 / gfx950), and its issue tracker contains zero issues
mentioning RDNA. If you are on an AMD card that is not an Instinct, and you are serving a
model with a recurrent-state (hybrid) architecture that wants its state cached across a
restart, the path is not obvious. This is that path, demonstrated on the card that matters
for a workstation: the R9700, gfx1201, 32 GB, TP=1.

---

## Try it in ten minutes

The whole chain — the tier sizing, the RAM clamp, the eight house patches, the fs-tier
config — can be exercised **without a GPU window.** `DRY_RUN=1` prints the container command
instead of running it, and stays side-effect free (the startup GC reports what it would do
and deletes nothing):

```bash
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers

# Print the exact `podman run` command the launcher would execute — runs nothing,
# touches no GPU:
DRY_RUN=1 ./startup-qwen3.8-27b-kvcache.sh
```

If that prints a sane command, the launcher is wired correctly on your box. To actually
serve, drop `DRY_RUN=1` and install the reaper first — see [Hard requirements](#hard-requirements--not-advisory).
Sizing is in [How much do you need](#how-much-do-you-need--ram-and-disk-anchored-to-your-vram).

---

## How do you know it worked

`kvvalidate.py` is the welcome mat, not a warning label. It runs against a **live, busy
endpoint** — the condition most downstream users will actually have — and re-establishes five
invariants that held on 2026-09-12, then **names the regime you are in before you draw a
conclusion.** Read-only, stdlib only, no GPU, no engine change: it issues exactly one read
and never a state-changing request.

A pass is a real result, not a tautology: every invariant is checked against independently
incremented sources, never against a derived quantity. The tool encodes the three traps this
project learned the expensive way as guards:

- never report the load-latency histogram as a duration — it is thread-summed and overstates
  wall clock ~7×; the wall clock is `tier_load_seconds_total{fs}` only.
- judge a tier's value by `load_bytes` (bytes actually read from disk), never by the
  chunk/token hit counters, which repeat on every scheduler step.
- never print a derived quantity as a finding when it is an algebraic identity of the
  measured ones.

---

## What it actually does

Three tiers, each a fallback for the one above:

| Tier | Where | Size here | Notes |
|---|---|---|---|
| **L1** | GPU, fp8 KV | ~228,700 tokens / 9.3 GB | vLLM's own prefix cache, pinned |
| **L2** | RAM, `/dev/shm` | 24 GiB ≈ 419,000 tokens | pre-faulted + pinned; **ARC** eviction |
| **L3** | disk, dedicated fs | 511 GiB | `O_DIRECT`, 8 read / 4 write threads |

Wired through `--kv-transfer-config` (`TieringOffloadingSpec` / `OffloadingConnector`).

**The architecture is strictly staged, and this is the single most important thing to
understand about it.** From `tiering/manager.py`: *"Primary tier is the gateway — secondary
tiers cannot access GPU memory directly; all data flows through the CPU primary tier"* and
*"blocks in secondary tiers must be promoted to the primary tier before GPU can access them."*
Consequences:

- The disk tier **cannot** serve the GPU on its own. Turn off the CPU tier and the disk
  tier turns off with it.
- Disk and RAM **contend for the same region**. `lookup()` returns MISS not only when a
  block is absent but also when *"primary is full and cannot accept a promotion."*
- There is **no DMA path** disk → VRAM. Foreclosed three ways here: no cuFile/GPUDirect/
  DMA-BUF anywhere in the vLLM tree; the cache device is virtio_blk inside a VM, so there
  is no real PCIe endpoint to peer with; and GPUDirect Storage is NVIDIA/cuFile and does
  not exist for gfx1201.

**Storage density is 61,440 bytes/token as stored — not at the floor.** Two full-attention
groups every chunk (32,768 B/token, irreducible), one MTP/draft group every chunk (16,384
B/token, recomputable), and the six Mamba groups at a 1-in-8 stride (12,288 B/token). FP8
is already one byte; GQA is already 6:1. The 33,808 B/token attention-only figure is the
theoretical floor, not what is on disk — and there *is* density work left: dropping the
recomputable draft group reaches a 45,056 B/token floor. **It is not reachable by simply
declining to store the draft group:** the multi-group lookup returns zero for the whole hit
when any group has no stored chunks (`offloading/scheduler.py:816`), so dropping g8 costs
every offload hit rather than 27% of the bytes.

Eight house patches make it work; they are in [`patches/`](patches/) with their apply order,
and [`docs/kv-cache-current-implementation.md`](docs/kv-cache-current-implementation.md)
explains what each one does and which environment variable gates it.

---

## How much do you need — RAM and disk, anchored to your VRAM

Not a rule of thumb, and it does not need to be. Size is set by your **model's offload
density**, your **context length**, and your **target concurrency — not by how much VRAM
your card has.** The offload tiers are **61,440 B/token**; the on-GPU pool is
**40,652 B/token**. Offload storage costs **1.51× per token** what the GPU pool does, so
you cannot size the tiers by copying the GPU pool's bytes. That is structural, not waste:
the GPU keeps **one** Mamba recurrent state per live sequence (O(1) in sequence length),
while the offload store snapshots that state **every 8th chunk** so a prefix can be resumed
(its footprint grows O(tokens/8)), and g8's 3/8 padding sits on top of that.

On this box the wall lands at a specific, measurable place: **the third deep agent is the
cliff** — the first two fit, the third has nowhere to go. The CPU tier is the primary tier
and the only one with direct GPU access, so it must hold a whole context or a promotion
cannot land. Ours holds **2.05 full contexts** (24 GiB). Two agents fit; the third does not.

> **Rule — size the CPU tier in contexts, not bytes: `CPU tier ≥ (concurrent agents + 1) ×
> context × 61,440 B/token`.** The `+1` is the free promotion-staging slot the tier must
> keep available so a `fs → GPU` load can land.

Per full 204,800-token context: on-chip `204,800 × 40,652 = 8.33 GB`; offload
`204,800 × 61,440 = 12.58 GB` (11.7 GiB). The CPU-tier, disk and system-RAM columns do not
change with your GPU pool — only "contexts on-chip" does. Size the RAM and disk once, for
your model and your concurrency; buy the GPU for what it buys.

```
B/tok (offload) = 61,440        # measured; model-specific (recipe below)
B/tok (GPU)     = 40,652        # measured

contexts on-chip   = GPU_pool_bytes / (context × B/tok_GPU)
CPU tier (bytes)   = (agents + 1) × context × B/tok_offload
system RAM (bytes) = CPU_tier_bytes + ~15 GB
disk (bytes)       = (sessions to keep resumable) × context × B/tok_offload
```

The CPU tier lives in `/dev/shm` and **cannot grow at runtime** — what you pin at boot is
the ceiling for the life of the process. Ours is 24 GiB of 39 GB = **61% of system RAM**
(the ~15 GB of headroom is where the engine, drafter and non-KV working set live). **The
fs tier has no quota, TTL or eviction hook of its own — it writes and never deletes;** the
reaper is the eviction policy and defaults to a 65% target. Size the disk for the sessions
you want to keep resumable, and let the reaper do the rest.

**If your context is 32K instead of 204,800, everything shrinks proportionally** — per-context
offload is `32,768 × 61,440 = 2.01 GB` (1.88 GiB). At 32K you are almost certainly below
the floor (see the dead zone), so these are the sizing shape, not a promise of benefit.

**Disk speed: report the token rate, and be honest about the ratio.** Measure the load rate
the only way that is safe — wall clock:
`kv_offload_tier_hit_tokens_total{t} / kv_offload_tier_load_seconds_total{t}`. **Never** the
load-latency histogram: it is byte-identical to `thread_seconds`, 7.6× inflated, and still
unfixed.

| tier | tok/s | bandwidth |
|---|---|---|
| fs (disk) | **3,868** | 228 MB/s |
| CPU | **337,408** | 11.8 GB/s |

**The CPU tier is unambiguously worth its RAM.** 337,408 tok/s beats even prefill at its
*fastest* measured rate by ~49×, and prefill is almost always slower than that. For the
**disk tier, do not quote a speed-up ratio.** The natural prefill denominator is a sum over
overlapping requests — the same trap as `thread_seconds` — which puts the disk tier anywhere
between **2.2× and 0.55×** recompute. Resolving it cleanly needs a controlled single-stream
wall-clock prefill measurement, which has not been run. So, honestly, without apology: the
CPU tier is a clear win; the disk tier's margin depends on how much prefill concurrency you
run, and we have not yet measured it cleanly. `kvvalidate.py` and `tierreport.py` will tell
you, on your hardware. Two effects the tok/s comparison misses, both favouring the tier: a
disk load is **asynchronous** across 8 read threads and overlaps other work, whereas prefill
*occupies* the GPU; and a load **frees** GPU time for the other requests. Parity in tok/s
is not parity in value.

**The floor — read this before buying disks.** Prefixes under **13,184 tokens** get **zero
external hit** — that is the **Mamba dead zone** (below), the stride/capacity trade where the
snapshots do not line up with a lookup. If your prompts and prefixes sit below ~13K tokens,
this is not the tool for you, no matter how large the tiers are.

**To derive your own B/token** (it is model-specific): `config.json` in the store root lists
the **groups** and `tokens_per_block` for each; per-group **file size ÷ `tokens_per_block`**
= B/token for that group; **sum** the groups, **dividing any strided group by its stride**
(the Mamba groups are strided — written every 8th chunk). Drop the result into the formula
above and the whole table redoes itself for your model and context.

---

## The measurements — and the ones that do not move

**What the disk does when it is actually needed — quoted as a range, not a point.** Across
13 saturated-regime intervals with real disk activity, the external (tier → GPU) share ran
**min 0.0%, median 36.3%, max 86.7%.** The peak window, where agents were actively evicting
each other — the goal-1 scenario:

```
23:35  86.7% external   720,176 fs tokens
23:46  84.1% external   594,928 fs tokens
23:57  73.4% external   459,792 fs tokens
```

The median and the range are quoted together on purpose. "73–87%" by itself is the best three
intervals of thirteen and is not defensible on its own. And the external *share* is not a
headline: it moves with the workload (three hours apart on the same engine it read 36.2%,
then 17.7% — because the agents had settled into long continuing conversations and the GPU
prefix cache was catching the work before the tier was ever consulted). A falling external
share with a rising cache share and falling recompute is success, not regression. So the
robust headline is 81.8% served without recomputation, not any single external figure.

**The token read rates** (the regime-independent ones, wall-clock method above): fs
**3,868 tok/s** (228 MB/s, 5.3% wait — device-bound), CPU **337,408 tok/s** (11.8 GB/s,
0.0% wait). **Never** the latency histogram for either.

---

### The warm-up: a feature, and the reason it ships with a regime tool

The fs (disk) tier **cannot serve until the CPU tier saturates** — on this box, ~45 minutes
of real traffic. In every sample while `cpu_free > 0` the `fs_hit` delta is exactly 0; it
turns non-zero in nearly every sample after `cpu_free` reaches 0. The CPU tier fills 8.18 →
25.76 GB over that window. Benchmark inside it and you measure nothing.

That is not a bug we could not fix; it is how a tiered cache behaves, and we understand it
well enough to package it. **KV offload tiers only engage under memory pressure. Benchmark
one cold and you will measure nothing — so this ships with a tool that tells you which
regime you are in before you draw a conclusion.** `kvvalidate.py` names the regime
(saturated / post-boot / mid-fill); the mid-fill one is the liar, and it is warned, never
refused.

### The Mamba dead zone

The stride is **truncate-and-recompute, not serve-an-approximate-state.** `resolve_mamba_align_size`
rounds `max_hit_size_tokens` **down** to a kept Mamba snapshot boundary, so the engine only
ever requests a snapshot it actually kept — what is served is **exact**, and the gap (up to
13,184 tokens, ~6,592 on average) is recomputed by the normal prefill path. The cost is
**compute, not accuracy** (the 85,696-token bit-identical disk hit corroborates it).

The price of that stride is the **dead zone: any prefix shorter than 13,184 tokens gets a
zero external hit and is recomputed in full, even though its attention chunks are on disk.**
The snapshots simply do not line up with a short lookup. This is the stride/capacity trade,
stated plainly: if your prompts and prefixes sit below ~13K tokens, this is not the tool for
you, no matter how large the tiers are.

---

## The first benchmark, reproduced as given

> ⚠️ **Everything below this line was measured before the R3.15 correctness fix
> (2026-09-10) and is retained as *method, not as result.*** Some fraction of the "tokens
> served from the tier" were served against unconfirmed keys — that is, served *wrongly*.
> The direction of the error is known, its size is not. These must be re-measured on the
> fixed engine; see [`docs/status-2026-09-10.md`](docs/status-2026-09-10.md) §4.

**The tier earned its keep.** Approximately **2.45M tokens served from the RAM and disk
tiers**, ≈ **26 minutes of prefill avoided** at the honest cold rate (~1,555 tok/s).

**A preliminary end-to-end run** on a real mixed workload — an `opencode` session doing code
review and testing, with concurrent chat — measured per-tier hit shares of the prompt tokens:

| Tier | Size (GiB) | Hit ratio | Fetch latency | vs 45 s cold recompute |
|---|---|---|---|---|
| GPU VRAM | 9.3 | 52% | 0 (local) | — |
| CPU RAM | 24 | 11% | 1–2 s | ~95% faster |
| Filesystem | 215 / 511 | 7% | 38 s | 15% faster |
| Recompute | — | 30% | 45 s | baseline |

**70% of prompt tokens served from cache.** The headline of that run is the **11% from RAM**,
not the 7% from disk: a RAM hit is essentially free against a 45 s recompute, while a disk
hit saves only 15%.

> **This is one small-sample run and is labelled as such.** One machine, one workload, one
> pass, by the author, nothing held out or reproduced.
> [`docs/kv-cache-results-preliminary.md`](docs/kv-cache-results-preliminary.md) carries it
> in full — including **§4, the figures in it that do not yet reconcile** with each other or
> with the measured storage density. They are flagged there rather than quietly corrected:
> *"They are flagged rather than corrected, because the raw observations are the author's
> and it is not for a reader — or an assistant — to quietly adjust someone's data."*

The encouraging part of that run is not the headline but a coincidence. The disk tier's
speedup there (45 s → 38 s, **1.18×**) lands on the same number the component measurement
reached by a completely different route: the fs tier reads at ~117 MB/s against a ~101 MB/s
recompute break-even, **1.16×**. Different workload, different instrument, same answer, and
nothing was tuned to produce it.

**What must never be cited.** The BetterBench numbers for this stack are polluted. vLLM
matches the prefix cache by block **content**, not by chained prefix, and BetterBench's
"cold (nonce)" run varies only block 0 — so the body of the prompt self-caches and the
"cold" arm is not cold. Everything else in these documents is either a component measurement
(documented with its method) or invalid (documented as invalid).

---

## Correctness: what was attacked, and held

**The state of the claim is: working, correct, unoptimised.** It is a working, correct,
unoptimised implementation delivered as a patch stack — not a proof of concept. The defects
that were found are not still open: they were fixed, and then the fixes were tested.

**A defect that made the offload tier serve KV belonging to a *different prompt* was found
and fixed** (R3.15, 2026-09-10). It was not a benchmark problem: unfixed, the tier served
**98.1%** of tokens on prompts that shared no legitimate prefix, while the GPU prefix cache
— looking at the same prompts in the same window — correctly refused **all** of them. The fix
is confirmed live by `check-r315-boot.sh`, and a controlled reverse test isolating that one
file reproduces the defect on demand.

The remaining ways the cache could be wrong were enumerated and tested one at a time:

| The suspicion | Outcome |
|---|---|
| A disk-tier hit might not reproduce a cold recompute | **Exact.** An 85,696-token disk hit was bit-identical |
| The mixed local+external boundary might corrupt output (three CT4 failures) | **Not a defect.** 9/9 bit-identical uncontended; every original failure was co-tenancy moving a near-tie logit |
| `cache_salt` might not reach the tier's key, letting one tenant read another's blocks | **Honoured.** The isolation claim is **refuted** — the salt chains into the offload key |
| The correctness harness itself might be incapable of failing | **It fails when it should.** A negative-control gate blocks the suite if the instrument cannot detect a divergence it was handed |

A co-tenant can *create* a spurious divergence but never hide a real one, so the uncontended
runs above are the valid test — that asymmetry is in [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md).
Each row's evidence, and the one condition that would reopen it, is in
[`docs/kv-cache-closed-decisions.md`](docs/kv-cache-closed-decisions.md), which also lists
the claims this project has published and withdrawn.

**What is still not validated.** The framework is **assumed correct** — enough to work on and
to benchmark against — and **not yet validated correct.** The systematic correctness suite has
not been run; per the plan it happens **pre-optimisation, after shipping.** What is owed is a
named list, [`docs/status-2026-09-10.md`](docs/status-2026-09-10.md) §4; the two that matter
most are showing the *generated tokens* differ under a deliberately re-armed defect, and
reproducing both defects on a **stock** vLLM 0.27.1 build, which is what blocks filing
upstream.

**Every number in this repository taken before 2026-09-10 is uncitable** — the harness was
sound, the engine under it was not. Re-measuring them is outstanding work, not a formality.

---

## What it costs, and what is not tuned yet

The limitations, stated plainly and without apology:

- **The mamba stride is truncate-and-recompute, and its cost is compute.** A prefix is
  truncated to the last kept Mamba snapshot and the gap (up to 13,184 tokens, ~6,592 on
  average) is recomputed on the normal prefill path. What is served is exact; the price is
  the recompute of that gap, plus the dead zone (short prefixes below 13,184 tokens get a
  zero external hit). The exactness fix — replay the ≤ one-block gap from the stride
  checkpoint, turning the truncate into an exact reconstruction — is designed but not built.
- **A failed offload load kills EngineCore.** `assert transfer_result.success`, and
  `OffloadingConnector` exposes no `get_block_ids_with_load_errors()`, so
  `kv_load_failure_policy=recompute` is inert here. This is why the reaper's `MIN_AGE`
  floor is a hard safety property and not a tuning knob.
- **The disk tier is barely worth doing on this hardware.** ~1.16× over recompute. Whether
  that is the device (a raidz of spinning disks behind a zvol behind virtio_blk — an NVMe
  device is ~19× this) or the implementation is not yet established.

**The code is implementation-grade; the delivery is a patch stack.** It is eight anchored
string-surgery patches applied at container start against a version-bound vendor image
(vLLM 0.27.1 + radiance 0.9.3), which is what makes it precise and what makes it brittle in
exactly the same way. That is the honest shape of it, said where it is useful rather than
in the headline.

**It is a starting point, not a product.** It is published at this maturity deliberately:
several people want this capability, the author has neither the time nor the specialist
expertise to carry it to completion alone, and a working starting point that says exactly
where it is flawed is more useful than nothing. [The roadmap](#the-roadmap--in-the-authors-intended-order)
is the ordered list of what it would take to make it real, and
[Point your own coding agent at this](#point-your-own-coding-agent-at-this) is how to start.

---

## Where it sits upstream

This is on the upstream path, not beside it. vLLM's own connector documentation states that
hybrid models are **"currently not optimized for the offloading connector"** — which is
precisely the gap this work fills, and the reason the gfx1201/RDNA4 result is worth
reporting back. The **llm-d filesystem backend is the in-tree `FileSystemTierManager` this
runs**, with the same authors (Ozeri, Harnik, IBM). The eight house patches are a
transitional layer: the destination is to reconcile each against current vLLM/radiance HEAD
and **delete it in favour of the upstream implementation wherever one exists.** Two open PRs
already measure as directly applicable — **#54327** (bounded capacity + LRU eviction to the fs
tier, 100% applicability; would retire the mandatory external reaper) and **#54743** (the
filtered-group primitive the stride work should build on).

The target of all the work below is **upstream vLLM**, not a fork that lives here permanently.
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
  deliberately not what this is for. vLLM is a general serving runtime and should remain one;
  a change that buys this model 5% and costs another model its ability to start is a bad
  trade at any margin.
- **Not breaking existing capability is a requirement of every stage,** including the ones
  that look like housekeeping. It is why every behaviour-changing patch here is behind an
  environment gate whose unset state is upstream behaviour, and why the eight patches are to
  be *deleted* in favour of upstream implementations wherever one exists rather than
  maintained alongside them.

None of this is aimed at the specialised forks. The gfx1201 work this repository stands on —
radiance, libr4d, the MXFP4 build — is what makes this card usable at all, and the
multi-GPU tensor-parallel work in that cluster is genuinely impressive engineering. It is
simply a different aim: those forks exist to make one architecture excellent on hardware the
runtime otherwise ignores, and this wants to end up in the runtime everybody already has.

---

## The roadmap — in the author's intended order

Each stage's output is the next stage's input, and **the order is deliberate**: nothing after
stage 1 can be judged without stage 1, and the accuracy work comes before the speed work
because it is pointless to optimise a path that is still returning approximate state.

**1. Benchmark hooks, metrics, and a reproducible harness. First.**
Tidy up the instrumentation that already exists (the lookup-outcome and instrumentation
patches) into a coherent metrics surface, and write a **reproducible cache-metrics script**
in the spirit of BetterBench but aimed at the tier rather than at raw token throughput: hit
share per tier, promotion latency, `load_bytes`, read:write ratio, and the recompute avoided.
**It must not repeat BetterBench's mistake** — vLLM matches the prefix cache by block
*content*, not by chained prefix, so varying a nonce in block 0 leaves the body of the prompt
self-caching and the "cold" arm is not cold. Getting the cold arm genuinely cold is the hard
part of this stage and the reason it comes first: every claim below it is unfalsifiable until
it exists.

**Part of this now exists.** `patches/patch_kv_offload_tier_report.py` adds 19 `tier`-labelled
series — per-tier hit blocks and tokens, load/store bytes, seconds, ops and latency histograms,
capacity, an occupancy *distribution*, reads-before-evict, eviction-to-reuse, would-have-hit
and prefill stall time — and [`tools/tierreport.py`](tools/README.md) turns one `/metrics`
scrape into `results.md` + `results.json` with a verdict per operator question: too much RAM,
too little RAM, disk too slow, disk actively hurting, and whether the layered cache is adding
value at all. The design note is
[`docs/tier-report-metrics-plan.md`](docs/tier-report-metrics-plan.md).

It is deliberately **lifetime-cumulative and read-once**, which sidesteps the cold-arm problem
rather than solving it: it reports what the operator's own traffic actually did, so there is
no synthetic corpus to get wrong. Everything a *judgment* rests on is a monotonic counter or
a histogram (the only gauges are `tier_capacity_bytes` and `tier_used_bytes` — configuration,
not judgment), because a gauge scraped once off a long-lived server says almost nothing. What
is still missing from this stage is the other half — the controlled A/B harness with a
genuinely cold arm, which is what would let a *change* be measured rather than a deployment
described.

Two specific jobs belong here, both already scoped by earlier work:

- **Confirm or kill the busy-wait.** The strongest open lead in the repository is that the
  deferral loop is a busy-wait costing ~1.9 ms an iteration — ~19 s of spinning for 103 ms
  of real disk I/O. It is fitted to three points and **not yet confirmed against
  scheduler-step counters**. If it holds, it explains why a request that is 92% served from
  cache still loses to a cold recompute, and it is a bigger prize than any tuning below.
- **Fix the correctness check.** The current one passes on a five-character `ACK` response
  and therefore proves nothing about tens of thousands of tokens of replayed KV. It needs a
  prompt demanding a long, content-dependent answer before a pass means anything.

[`docs/kv-cache-historical.md`](docs/kv-cache-historical.md) §7 lists every other instrument
defect found the hard way. Read it before designing the harness, not after.

**2. Continue the review of existing work, and produce an implementation plan.**
[`docs/kv-cache-references.md`](docs/kv-cache-references.md) is the review so far — every PR,
paper and blog already assessed, each with a ruling. Continue it, then write the plan.
Concretely, this includes reconciling the eight house patches against current
vLLM/radiance HEAD and **deleting each one in favour of the upstream implementation wherever
one exists**: at least one already has an upstream counterpart (the eagle-groups fix is PR
#55390; the fs fanout was ported from PR #49225). Two open PRs already measure as directly
applicable and should be handled first: **#54327** adds bounded capacity and LRU eviction to
the fs tier at **100% applicability** — it would retire the mandatory external reaper — and
**#54743** adds the filtered-group primitive the stride work should be built *on* rather
than beside. The measured applicability table is
[`docs/kv-cache-references.md`](docs/kv-cache-references.md) §3a. *Avoid reimplementation* — a
house patch duplicating merged upstream work is a liability, not an asset: one more thing to
rebase, and it will silently diverge. Score upstream work by **applicability, not by merge
status**; an unmerged PR that fits is worth more than a merged one that does not.

**3. Tunable accuracy, and a proof of concept of the most promising quality options.**
Correctness here is not binary — it is a dial, and right now the dial is welded in one
position. Expose the accuracy/cost trade-offs as **configuration toggles** rather than
constants (the mamba stride `N` being the obvious first one; the store threshold and the
pending-is-miss behaviour are others), so that a deployment can choose its point on the curve
and a benchmark can sweep it. Then build a proof of concept of the most promising quality
options. The strongest candidate is the **exactness fix**: replaying the ≤ one-block gap from
the stride checkpoint, which turns the mamba stride from an approximation into an exact
reconstruction. It is designed and unbuilt, and it is the single well-described gap to the
full method — see [`docs/kv-cache-future-work.md`](docs/kv-cache-future-work.md).

**4. Refactoring.**
These patches were written one at a time, each to answer a specific question, and it shows.
They monkey-patch by string surgery. They carry an implicit dependency **order** documented
only in the launcher. Their gating environment variables are inconsistent in naming and in
whether `0` or `1` means "upstream behaviour". This wants to be a single coherent module with
an explicit interface, not eight scripts in a trench coat. It lands here rather than earlier
because stages 2 and 3 decide how much of it survives to be refactored.

**5. Speed optimisation.**
Nothing here has been tuned; it has only been made to work. The known ceilings are measured
and documented — the fs tier's 1.16×; the strictly staged promotion path, so disk and RAM
contend for the same region; no DMA path from disk to VRAM on this hardware. **Which of
those are real limits and which are merely untuned is, in most cases, not yet established** —
and stage 1 is what settles it.

**6. Normal software engineering.**
Code review, tests, CI, packaging, a real release. **None of it has happened.** There is no
test suite; correctness has been established by hand, per change, against a live endpoint.

---

## Point your own coding agent at this

This repository is written to be handed to an agent, not just read. The documents are
deliberately long and state their own reasoning and their own doubts, because that is what
an agent needs in order to not repeat work that has already been done and discarded.

To be plain about the shape of it, in the place where that is useful: **the code is
implementation-grade, and the delivery is a patch stack** — eight anchored string-surgery
patches applied at container start against a version-bound vendor image (vLLM 0.27.1 +
radiance 0.9.3). That is what makes it precise and what makes it brittle in the same way.

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
question is already settled, and skim docs/kv-cache-historical.md for what has already been
tried and retracted. This is a working, correct, unoptimised implementation delivered as a
proof-of-concept patch stack, and the roadmap in the README is in the author's intended
order. Start at stage 1: audit what cache metrics the existing patches already expose, and
propose a reproducible harness that measures per-tier hit share and prefill avoided with a
genuinely cold arm — read bench/README.md first for why 'cold' is the hard part. Do not
write code in this pass."*

**3. Get the plain MXFP4 build serving first.** `startup-qwen3.8-27b-mxfp4.sh` at the
repository root is the same model on the same card with **no cache work at all**, and every
knob in it carries the measurement that chose it. It is the shorter path to a working engine,
and if it does not serve, nothing here will either. The cache launcher is that same
structure plus a **seven-item delta** — and its tuning defaults are copied from it when the
release is built, so the two cannot drift. Base your own work on that structure:
`kv-cache/launcher/README.md` lists the delta in full, and it is the whole job if you want
to add the cache to a launcher of your own.

**4. Pick a stage from [The roadmap](#the-roadmap--in-the-authors-intended-order).** Stage
1 is not optional throat-clearing — until the metrics harness exists, nothing you change
afterwards can be shown to have helped. Three good first tasks, in increasing size:

- **Small, self-contained:** normalise the patch gating variables so that unset always means
  upstream behaviour, and move the apply order and its dependencies into code rather than
  leaving them stated in `patches/README.md` and the launcher's comments.
- **Medium, and the actual starting point:** the reproducible cache-metrics harness of stage
  1 — with a *genuinely* cold arm. Read [`bench/`](bench/)'s README first; the
  content-vs-prefix-hash trap is documented there and it is what makes this non-trivial.
- **Large, the real prize:** the exactness fix inside stage 3. It is designed and unbuilt,
  and it is the difference between "approximate" and "correct".

**5. Run it.** [`docs/kv-cache-operations.md`](docs/kv-cache-operations.md) is the runbook —
turning the disk tier off, resizing `/dev/shm`, resizing the KV tier, each with commands,
verification and undo. Read [Hard requirements](#hard-requirements--not-advisory) first; two
of them fail *silently* if ignored.

**6. Measure it.** [`bench/`](bench/) ships the harnesses but **no result files**,
deliberately — single-machine numbers, some of them known polluted. Its README documents the
content-vs-prefix-hash trap that invalidated an earlier round of measurement, which is the
mistake most likely to waste your first day.

Different hardware is welcome and wanted. Nothing here has been reproduced on another machine,
and the conclusions should be assumed hardware-specific until they are.

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
| [`cache-preemption-patch-plan.md`](docs/cache-preemption-patch-plan.md) | The deeper patch plan. **Revision 3 supersedes parts of Revision 2 (and the original) — read Revision 3 first, then Revision 2 in [`kv-cache-historical.md`](docs/kv-cache-historical.md).** |
| [`README.house.md`](docs/README.house.md) | The original house-files README: how `/house` is wired, `tierbench`, the instrumentation patch. |

---

## The launcher

`../startup-qwen3.8-27b-kvcache.sh` serves the entry **`qwen3.8-27b-kvcache`**. It is the
KV-cache configuration in one file, every knob documented inline with the reasoning and the
measured cost of changing it. Run `../startup-qwen3.8-27b-kvcache.sh -h` for the full list.

It is a **thin wrapper**, deliberately: it sets environment and execs the real launcher,
which stays the single source of truth for how the model is served. Nothing about the model,
kernels, drafter or vLLM invocation is duplicated.

It runs on its own container (`qwen38-27b-kvcache`) and its own on-disk block tree
(`blocks-kvcache`), so a KV-cache experiment can never quietly pollute the production entry's
cache or its numbers. That isolation is not paranoia — it is the direct lesson of the
BetterBench pollution above.

**It takes the whole GPU.** It and the production entry cannot be loaded together.

---

## Hard requirements — not advisory

- **`PYTHONHASHSEED` must be pinned** (the launcher pins it to `0`). Block filenames are
  content hashes chained from `NONE_HASH`, which is seeded from `os.urandom(32)` when the
  variable is unset. Leave it unset and every restart hashes the same tokens to *different*
  filenames, orphaning the entire on-disk cache — silently, at a 100% miss rate, with no
  error anywhere.
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
[`docs/kv-cache-historical.md`](docs/kv-cache-historical.md) §3.3). On a 32-thread host the
same loop would be cheaper and might never have been noticed — which is a reason to trust
the finding, not to discount it.

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
engine is. The measured result is **~117 MB/s**, against a **~101 MB/s** recompute break-even
— a **1.16×** advantage, which is why the disk tier is real but thin, and why it contributes
7% where the RAM tier contributes 11%. **An NVMe device is roughly 19× this**, and the disk
tier's whole economic case changes on that hardware. If you have NVMe, the most interesting
thing you can do with this repository is re-run the tier measurement and tell everyone what
happened.

**GPU passthrough into a VM.** The cache device has no real PCIe identity from the guest's
point of view, so there is no peer to DMA with even in principle — one of the three
independent reasons there is no disk→VRAM path here. It is also why **any host-limited
property must never be sized from inside the guest**: the guest misreports the CPU model and
the PCIe link speed, and believing it will send you down a false trail.

**Nothing here has been reproduced on other hardware,** and every conclusion should be assumed
hardware-specific until it is. Different hardware — more cores, more RAM, NVMe, a bare-metal
host — is wanted, and would settle several of the open questions above outright.

---

## Licence and attribution

The house patches and documents here are the author's own work, built on and against vLLM
and the radiance overlay; upstream code carries its own licences. Where a patch was ported
from an upstream PR it says so, with the PR number, in the patch file itself.

**Author:** zzpanic — <zzpanic@gmail.com>, [github.com/zzpanic](https://github.com/zzpanic).

Reproductions on other hardware are wanted more than anything else here. If you run this on
a different card, a different storage stack, or an NVMe device, the results are worth sending
on whether they agree or not — a disagreement is more useful than a confirmation, because
nothing in this repository has been reproduced anywhere but the one machine described above.
Issues on the repository are the better route for anything others should see; mail is for
what does not belong in public.
