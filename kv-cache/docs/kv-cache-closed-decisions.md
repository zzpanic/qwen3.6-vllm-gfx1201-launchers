# Closed decisions — the register

Every question in this project that has been **settled**, with the evidence that settled it
and, in each case, **the one thing that would reopen it**.

## Why this document exists

The other documents in `docs/` are narrative: they record what was tried, in the order it
was tried, including the parts that turned out to be wrong. That is the right shape for an
experimental record and the wrong shape for the question an engineer actually arrives with,
which is almost always *"has this already been decided, and on what grounds?"* Answering
that from a 2,300-line chronological plan means reading the whole thing and hoping you
noticed the paragraph where it was withdrawn.

So the closures live here, once, in a form you can scan. The narrative stays where it is.

**Read the reopen column before proposing work.** A closure here is not a claim that the
question is uninteresting — several of these were closed *because* they were interesting
enough to measure properly. It is a claim that re-deriving the answer costs a day and buys
nothing, unless the named condition has changed. If it has, reopen it; that is what the
column is for.

**What this register is not.** It is not a summary of the project's findings — see
[`kv-cache-historical.md`](kv-cache-historical.md) for the reasoning and the numbers behind
each row, and [`kv-cache-known-issues.md`](kv-cache-known-issues.md) §E for the shorter list
of things that are operationally dangerous rather than merely settled. Rows here point at
both.

---

## 1. Correctness

| Question | Ruling | Evidence | Reopen if |
|---|---|---|---|
| Does the mixed local+external boundary corrupt output? (CT4) | **No defect.** The divergence was co-tenancy. | 9/9 bit-identical uncontended at the decisive offset; the only divergence anywhere was a *contended* recompute. [`CORRECTNESS.md`](CORRECTNESS.md) | A divergence appears in an **uncontended** rep. **Never** on rep count — the author ruled 9 clean uncontended reps sufficient at this stage. |
| Does `cache_salt` reach the offload tier's key, or can one tenant read another's blocks? | **It is honoured. No isolation defect.** The earlier claim is **refuted**. | The salt enters at block 0 and chains through `parent_block_hash` into the offload key, so there is no unsalted path to the tier; six fresh salts read 0 bytes on both tiers. The "4 GB under a fresh salt" was a **global counter delta over a shared window**, not a per-request read. | A fresh-salt read is demonstrated with **per-request** attribution. A delta on a global counter is not evidence and will not reopen this. |
| Is a disk-tier hit exact? | **Yes, bit-identical.** | An 85,696-token disk hit matched a cold recompute exactly. [`kv-cache-historical.md`](kv-cache-historical.md) §9 | A divergence on an uncontended pure-fs hit. |
| Did the tier ever serve blocks whose content matched but whose prefix did not? | **Yes — and it is fixed (R3.15).** | The reverse test armed the defect: unfixed, the tier served **98.1%** cross-nonce false hits while the GPU cache served 0%. | — (kept here as the reason the fix must never be reverted) |
| Is the N=8 Mamba stride store exact? | **No, and it is not meant to be.** Approximate by design away from kept boundaries. | [`kv-cache-known-issues.md`](kv-cache-known-issues.md) A2/D4; CT6 measures the divergence and never fails the run. | Not a closure to reopen — a documented design limitation. Do not "fix" it on the strength of a non-zero CT6 number. |

## 2. Capacity and density

| Question | Ruling | Evidence | Reopen if |
|---|---|---|---|
| Can KV be made denser? | **No. 34 KB/token is the architectural floor.** | 16 attention layers × 2048 B. [`kv-cache-historical.md`](kv-cache-historical.md) §2 | The model architecture changes. Not by compression work on this one. |
| Are the Mamba layers the storage waste? | **No — retracted twice.** The waste is **temporal, not spatial**. | Groups 0–5 are 97–99% dense; the MTP group is ~6%. Same §2. | — |
| Is `blocks_per_chunk` a lever? | **No.** Investigated and closed. | [`kv-cache-historical.md`](kv-cache-historical.md) §9 | — |
| Does the mamba-align cache shortfall need fixing? | **No — by design.** Every turn pays 1,664–3,327 tokens. | R3.12 | — |

## 3. The device, and what to buy

| Question | Ruling | Evidence | Reopen if |
|---|---|---|---|
| How fast is the fs tier's device? | **~117 MB/s. Do not re-bench it.** | Established from three independent directions. [`kv-cache-historical.md`](kv-cache-historical.md) §5 | The device is physically replaced. |
| Is the disk the bottleneck, or the software above it? | **The device.** | R3.9 exonerated the software; the fanout port then split a promotion exactly 8 ways and the rate did not move. | — |
| Does parallelising a promotion help? (PR #49225, R3.14 fanout) | **No win.** Kept because it is correct and free, not because it measured better. | Splits 8 ways exactly; ~117 MB/s either way. | — |
| Should storage be upgraded? | **Yes — this REVERSES the earlier "do not buy disks".** The fs tier sits at **1.16×** break-even: 101 MB/s to tie recompute, device does 117. | [`kv-cache-historical.md`](kv-cache-historical.md) §5 | — (this is the standing recommendation, not a closure against work) |
| GPUDirect / DMA disk→VRAM? | **Foreclosed, three separate ways.** | See [`../README.md`](../README.md) | — |

## 4. Method — how measurements here are allowed to be made

These are closed *procedural* decisions. They cost more to relearn than any result in the
table above, because each was learned by publishing a wrong number first.

| Question | Ruling | Evidence | Reopen if |
|---|---|---|---|
| How should upstream work be scanned? | **By applicability against the installed tree** — fetch each PR's diff and count how many of its removed lines still exist verbatim. **Never by merge status or issue prominence.** | The picture inverted when redone this way. [`kv-cache-historical.md`](kv-cache-historical.md) §8; the measured table is in [`kv-cache-references.md`](kv-cache-references.md) §3a | — |
| Does an offload hit cost ~78 s? | **Withdrawn.** No read-back was measured — it was a lookup stall. | R3.1 | — |
| Was the 80 s vs 14 s promotion comparison valid? | **Withdrawn.** The arms did not start from an identical cache fill. | R3.9-era retraction | — |
| Is request wall time a valid instrument for tier work? | **No.** It bundles queue, stall and compute. | Same | — |
| Can a global cumulative counter be attributed to one request? | **No.** A delta over a shared window measures everything on the engine, not your request. | The refuted `cache_salt` claim was built on exactly this error (§1) | — |
| Is a divergence seen under contention evidence of a defect? | **No — and the asymmetry matters.** Contention can only *create* a spurious divergence, never hide a real one. So an **uncontended** run is the valid test, and a *contended* PASS is stronger than an uncontended one. | Uncontended 0/10, contended 10/10, always at the same near-tie token, margin 0.125. [`CORRECTNESS.md`](CORRECTNESS.md) | — |
| Is it enough to validate a cache-correctness fix on the fixed build? | **No. Arm the defect and show the instrument catches it.** | [`kv-cache-known-issues.md`](kv-cache-known-issues.md) §E.6 | — |
| Can `/metrics` be assumed not to expose something? | **No — enumerate it first.** A metric assumed absent was present, and the "≤25%" figure it produced was a lower bound, not a measurement. | — | — |

---

## Still open — deliberately not in this register

For contrast, so that "not listed here" is not read as "settled":

- **The controlled A/B harness with a genuinely cold arm.** Half of roadmap stage 1. The
  metrics half now exists ([`tier-report-metrics-plan.md`](tier-report-metrics-plan.md) and
  `tools/tierreport.py`); this half does not.
- **Why a 64 s promotion is still not explained.** [`kv-cache-historical.md`](kv-cache-historical.md) §6.
- **`--max-num-seqs` 4 → 2.** Costs nothing to try, never tried under a controlled replay.
- **Everything in [`kv-cache-known-issues.md`](kv-cache-known-issues.md) §C.**
