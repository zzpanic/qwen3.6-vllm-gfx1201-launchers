# Three-tier KV-cache offload for a GDN hybrid on a single AMD card

**A GPU → RAM → disk KV-cache offload for Qwen3.8-27B (Gated-DeltaNet hybrid, MXFP4
weights, FP8 KV) served by vLLM 0.27.1 / radiance 0.9.3 on one AMD Radeon AI PRO R9700
(gfx1201, 32 GB, TP=1).**

---

## ⚠️ Status: PROOF OF CONCEPT

Read this section before you read anything else, and before you quote any number
from this repository.

This is a **proof of concept**. It demonstrates that a three-tier KV-cache offload can
be made to work on a GDN hybrid model on a single consumer AMD card, and it is measured
doing so. **That is the whole of the claim.** It is not production code, it is not a
research-grade reproduction, and it has **known correctness errors in the
implementation**.

It is published at this maturity **deliberately**. Several people want this capability;
the author has neither the time nor the specialist expertise to carry it to completion
alone, and a working-but-flawed starting point that says exactly where it is flawed is
more useful to those people than nothing. **It is a starting point, not a product.**

### What this needs before it is anything more than a proof of concept

In order — each stage's output is the next stage's input.

**1. Upstream reconciliation — first, and before any code is written.**
Several of the seven house patches were written against a gap that may since have been
closed upstream, and at least one has a known upstream counterpart already (the
eagle-groups fix is PR #55390; the fs fanout was ported from PR #49225). Every patch
here must be checked against current vLLM/radiance HEAD and **deleted in favour of the
upstream implementation wherever one exists**. *Avoid reimplementation* — a house patch
that duplicates merged upstream work is a liability, not an asset: one more thing to
rebase, and it will silently diverge. Score upstream work by **applicability, not by
merge status**; an unmerged PR that fits is worth more than a merged one that does not.

**2. Refactoring.**
These patches were written one at a time, each to answer a specific question, and it
shows. They monkey-patch by string surgery. They carry an implicit dependency **order**
documented only in the launcher. Their gating environment variables are inconsistent in
naming and in whether `0` or `1` means "upstream behaviour". This wants to be a single
coherent module with an explicit interface, not seven scripts in a trench coat.

**3. Optimisation.**
Nothing here has been tuned; it has only been made to work. The known ceilings are
measured and documented — the fs tier serves at **~117 MB/s against a ~101 MB/s
recompute break-even** (1.16×, i.e. barely worth doing); the promotion path is strictly
staged through the CPU tier, so disk and RAM contend for the same region; there is no
DMA path from disk to VRAM on this hardware. **Which of those are real limits and which
are merely untuned is, in most cases, not yet established.**

**4. The exactness fix.**
The one designed-but-unbuilt piece: replaying the ≤ one-block gap from the stride
checkpoint, which turns the mamba stride from an approximation into an exact
reconstruction. This is the single well-described gap to the full method. See
[`kv-cache-future-work.md`](docs/kv-cache-future-work.md).

**5. Normal software engineering.**
Code review, tests, CI, packaging, a real release. **None of it has happened.** There is
no test suite; correctness has been established by hand, per change, against a live
endpoint.

### The three limitations that matter most

- **The mamba stride (N=8) is approximate by design.** The store keeps only every Nth
  chunk's recurrent state, and a lookup rounds **down** to the nearest kept boundary, so
  the served state can be up to 8 chunks stale. It is a good approximation because the
  Gated-DeltaNet gate gives the state finite effective memory — recent tokens dominate —
  but it *is* an approximation, and it measurably raises the needle-in-a-haystack failure
  rate. The exact fix is stage 4 above.
- **A failed offload load kills EngineCore.** `assert transfer_result.success`, and
  `OffloadingConnector` exposes no `get_block_ids_with_load_errors()`, so
  `kv_load_failure_policy=recompute` is inert here. This is why the reaper's `MIN_AGE`
  floor is a hard safety property and not a tuning knob.
- **The BetterBench numbers for this stack are polluted and must never be cited.** vLLM
  matches the prefix cache by block **content**, not by chained prefix, and BetterBench's
  "cold (nonce)" varies only block 0 — so the body self-caches. The only defensible
  quantitative claim is the one below.

### The one honest number

The tier **earned its keep**: approximately **2.45M tokens served from the RAM and disk
tiers**, ≈ **26 minutes of prefill avoided** at the honest cold rate (~1,555 tok/s).

Everything else is either a component measurement (documented with its method) or
invalid (documented as invalid).

---

## What it actually does

Three tiers, each a fallback for the one above:

| Tier | Where | Size here | Notes |
|---|---|---|---|
| **L1** | GPU, fp8 KV | ~228,700 tokens / 9.3 GB | vLLM's own prefix cache, pinned |
| **L2** | RAM, `/dev/shm` | 24 GiB ≈ 762,000 tokens | pre-faulted + pinned; **ARC** eviction |
| **L3** | disk, dedicated fs | 511 GiB | `O_DIRECT`, 8 read / 4 write threads |

Wired through `--kv-transfer-config` (`TieringOffloadingSpec` / `OffloadingConnector`).

**The architecture is strictly staged and this is the single most important thing to
understand about it.** From `tiering/manager.py`: *"Primary tier is the gateway —
secondary tiers cannot access GPU memory directly; all data flows through the CPU
primary tier"* and *"blocks in secondary tiers must be promoted to the primary tier
before GPU can access them"*. Consequences:

- The disk tier **cannot** serve the GPU on its own. Turn off the CPU tier and the disk
  tier turns off with it.
- Disk and RAM **contend for the same region**. `lookup()` returns MISS not only when a
  block is absent but also when *"primary is full and cannot accept a promotion"*.
- There is **no DMA path** disk → VRAM. Foreclosed three ways here: no cuFile/GPUDirect/
  DMA-BUF anywhere in the vLLM tree; the cache device is virtio_blk inside a VM, so
  there is no real PCIe endpoint to peer with; and GPUDirect Storage is NVIDIA/cuFile
  and does not exist for gfx1201.

**Storage density is at the architectural floor.** 33,808 bytes/token measured, and
attention arithmetic alone accounts for 103% of it (16 full-attention layers × 2,048
B/token/layer, plus one MTP layer). FP8 is already one byte; GQA is already 6:1; the
Mamba layers are already stored 1-in-8. There is no density work left to do here.

---

## Documentation map

Read in this order. Every document is written to be actionable without opening the next.

| Document | What it holds |
|---|---|
| **[`kv-cache-handover.md`](docs/kv-cache-handover.md)** | **Read first.** The map, the current state, and the resume path. |
| [`kv-cache-operations.md`](docs/kv-cache-operations.md) | The runbook: turn the disk tier off, resize `/dev/shm`, resize the KV tier. Each with commands, verification and undo. |
| [`kv-cache-current-implementation.md`](docs/kv-cache-current-implementation.md) | What is actually built and running: the three tiers, the seven patches with gates and order, the two-layer GC, the serve invocation. |
| [`kv-cache-known-issues.md`](docs/kv-cache-known-issues.md) | Every problem, gotcha and limitation as Symptom / Root cause / Impact / Status, by severity, plus the hard "never do X" list. |
| [`kv-cache-future-work.md`](docs/kv-cache-future-work.md) | The plan, the reuse-refresh mechanism and its `L` analysis, and the limitations to state up front. |
| [`kv-cache-references.md`](docs/kv-cache-references.md) | Every PR, paper, blog and local artifact reviewed, each with a status and a ruling. |
| [`status-2026-09-09.md`](docs/status-2026-09-09.md) | The investigation snapshot: the BetterBench pollution finding, the DASC/DAMP findings, the dead ends. |
| [`cache-preemption-patch-plan.md`](docs/cache-preemption-patch-plan.md) | The deeper patch plan. **Revision 2 supersedes parts of the original.** |
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
entry's cache or its numbers. That isolation is not paranoia — it is the direct lesson
of the BetterBench pollution above.

**It takes the whole GPU.** It and the production entry cannot be loaded together.

---

## Hard requirements — not advisory

- **`PYTHONHASHSEED` must be pinned** (the launcher pins it to `0`). Block filenames are
  content hashes chained from `NONE_HASH`, which is seeded from `os.urandom(32)` when
  the variable is unset. Leave it unset and every restart hashes the same tokens to
  *different* filenames, orphaning the entire on-disk cache — silently, at a 100% miss
  rate, with no error anywhere.
- **The fs tier never deletes.** `tiering/fs/manager.py` has no capacity, quota or TTL
  parameter and exposes no eviction hook. **An external reaper is mandatory**
  (`kvcache-reap.sh` + its systemd timer). Without it the filesystem fills.
- **Never delete a young block.** The reaper's `MIN_AGE=90min` floor is a safety
  property: reaping an in-flight block kills EngineCore (see the limitations above).
- **`O_DIRECT` in both directions.** Spare RAM cannot act as a read cache in front of the
  fs tier. The only productive home for spare RAM is the primary tier.

---

## Environment this was built and measured on

Everything here is single-machine, single-configuration. Nothing has been reproduced on
other hardware, and the conclusions should be assumed hardware-specific until they are.

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700, gfx1201 / RDNA4, 32 GB, TP=1 |
| Model | Qwen3.8-27B MXFP4 weights + FP8 KV, 16 full-attention + 48 Gated-DeltaNet layers |
| Serving | vLLM 0.27.1 + radiance 0.9.3 overlay, R4D attention, DFlash2 spec decoding |
| Host | Hyper-V guest, 39.17 GiB RAM, 4C4T; cache device is a virtio_blk zvol |

The host being a VM matters more than it looks: it is why there is no memory hotplug, why
the cache device has no real PCIe identity, and why any host-limited property must never
be sized from inside the guest.

---

## Licence and attribution

The house patches and documents here are the author's own work, built on and against
vLLM and the radiance overlay; upstream code carries its own licences. Where a patch was
ported from an upstream PR it says so, with the PR number, in the patch file itself.
