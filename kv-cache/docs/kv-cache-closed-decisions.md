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
| What is the reuse L — and is the tier's `tokens_per_hash` the quantity C1 calls `tokens_per_chunk`? (C1) | **Yes, on this deployment: both are 1,648, so L ≤ 8 × 1,648 = 13,184 tokens, ~6,592 on average.** With `blocks_per_chunk = 1`, one offloaded chunk is one block, so the scheduler's chunk size *is* the group block size, which is the hash granularity the tier records as `tokens_per_hash`. | On-disk run config `tokens_per_hash: 1648` and per-group `tokens_per_block: 1648` (`/kvcache/blocks/_models_Qwen3.8-27B-MXFP4-mtpfp8_5f69fc4578e4/config.json:126,:18`); live engine read through `/proc/<pid>/root`: `tokens_per_chunk = tokens_per_block × blocks_per_chunk` (`offloading/scheduler.py:171`) with `blocks_per_chunk = 1` (`offloading/config.py:67` — the live `kv_connector_extra_config` sets neither `block_size` nor `blocks_per_chunk`); the engine's own boot asserts (`tokens_per_block % tokens_per_hash == 0`, `offloading/config.py:60`; all mamba groups agree, `offloading/scheduler.py:172`) passed with the engine serving. | A change to the block size or the stride N. |
| Does the mamba-align cache shortfall need fixing? | **No — by design.** Every turn pays 1,664–3,327 tokens. | R3.12 | — |

## 3. The device, and what to buy

| Question | Ruling | Evidence | Reopen if |
|---|---|---|---|
| How fast is the fs tier's device? | **~117 MB/s. Do not re-bench it.** | Established from three independent directions. [`kv-cache-historical.md`](kv-cache-historical.md) §5 | The device is physically replaced. |
| Is the disk the bottleneck, or the software above it? | **The device.** | R3.9 exonerated the software; the fanout port then split a promotion exactly 8 ways and the rate did not move. | — |
| Does parallelising a promotion help? (PR #49225, R3.14 fanout) | **No win.** Kept because it is correct and free, not because it measured better. | Splits 8 ways exactly; ~117 MB/s either way. | — |
| Should storage be upgraded? | **Yes — this REVERSES the earlier "do not buy disks".** The fs tier sits at **1.16×** break-even: 101 MB/s to tie recompute, device does 117. | [`kv-cache-historical.md`](kv-cache-historical.md) §5 | — (this is the standing recommendation, not a closure against work) |
| GPUDirect / DMA disk→VRAM? | **Foreclosed, three separate ways.** | See [`../README.md`](../README.md) | — |
| Can the CPU tier be grown past 24 GiB? | **No, not on this hardware.** `--kv-offloading-size 24` is a permanent ceiling, not a tuning choice. | The guest has 39 GB total, ~8 GB available, 24 GB already shared to the tier (`free -g`, 4 vCPU Skylake guest, 2026-09-11). | More host RAM. Until then, **reducing bytes/token is the only lever that raises CPU-tier residency** — see the density row below. |
| Should the fs tier bypass the CPU tier and load straight to GPU? | **No — out of scope by decision** (pat, 2026-09-11), independently of the GPUDirect foreclosure above. | Ruled out for reasons beyond the tiering design; the staged-promotion architecture stays. | pat reopening it. |
| What is the fs tier's stored density, and what is its floor? | **61,440 B/token today; 45,056 B/token is the achievable floor.** Per 1,648-token chunk: 2 self-attn groups every chunk (32,768 B/tok, irreducible), 1 MTP/draft group every chunk (16,384 B/tok, **recomputable**), 6 Mamba groups at a 1-in-8 stride (12,288 B/tok, cadence-tunable only). | File census 2026-09-11: every block file is exactly 27,000,832 B = 16,384 B/token/group; g0–g5 at 73 files, g6/g7 at 636, g8 at 632 (0.1148 measured stride rate vs 1/8). Independently matches `patch_kv_offload_mamba_stride.py:78-79`'s own arithmetic (30 files per 8 chunks). | A change to the group layout, the stride N, or the block size. |

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
| Is `kv_offload_tier_load_seconds` wall time? | **No, on any tier whose I/O runs on a worker pool** (here `fs`, 8 read threads). The wrapper times each pool task and sums, so the total is thread-time and every rate derived from it is a **floor**, understated by up to the pool width. The latency histogram is likewise a *batch* service time, not a request stall — and this tier **queues deliberately rather than preempting and evicting**, so a long batch time is the trade, not a fault. The only wall-clock figure is `prefill_stall_seconds{tier}`. | `tierreport.py` now prints these rates with `≥`, brackets them against the p99 batch, and suppresses the "too slow" / "actively hurting" / "ceiling" verdicts on such a tier; `--serial-io` opts out. [`tier-report-metrics-plan.md`](tier-report-metrics-plan.md) §4 | When the patch is changed to time the whole promotion once instead of each batch — queued for the next model reload, since a reload resets these counters |
| Is the running container's state inspectable, given `podman exec` fails? | **Yes — read it through `/proc`.** `/proc/$(podman inspect <container> --format '{{.State.Pid}}')/root/...` exposes the whole container filesystem read-only from the host. | Used to pre-flight all 31 hunks of patch 8 against the **live** engine before a reload: 31/31 clean. | — (`podman exec` itself is still blocked by RLIMIT_MEMLOCK; that is [`kv-cache-known-issues.md`](kv-cache-known-issues.md) B2) |

## 5. Retracted — claims this project published and then withdrew

These were stated in earlier revisions of the documents in `docs/`, and are **wrong**. They
are collected here so the narrative documents can state the current answer plainly instead
of carrying a correction beside every sentence. If you are holding an old copy of this
repository, or a summary written from one, check it against this table.

| The claim, as published | The correction | Where it came from |
|---|---|---|
| *"vLLM radiance matches the prefix cache by block content, not by chained prefix"* — i.e. the GPU prefix cache was blamed for the polluted BetterBench run (A1) | **Wrong.** The GPU prefix cache chains correctly and always did. The defect was in the **offload tier's** mixed-hit lookup, which could hand `prepare_load` a key it had not confirmed. Fixed by R3.15. | `status-2026-09-09.md` §3, repeated into `kv-cache-future-work.md` §0 |
| *"The N=8 stride is the mechanism behind A1"* | **Wrong, and a separate matter.** A1 was unconfirmed keys; the stride approximation is an independent, deliberate design limitation that R3.15 neither caused nor cured. | `kv-cache-known-issues.md` A2 |
| *"The CPU tier holds ~508,000 tokens in 16 GiB"*, and the stride pair *"115,360 (N=1) → 276,900 (N=8)"* | **Superseded twice over.** The tier is **24 GiB** — `kv_offload_tier_capacity_bytes{tier="cpu"}` reads 25.76 GB on the live engine — and at 61,440 B/token as stored that is **~419,000 tokens** (§3). The old token figures also rest on a Mamba-storage share retracted in §2, so they cannot simply be rescaled to the larger tier. | `kv-cache-current-implementation.md`, `kv-cache-future-work.md`, `kv-cache-handover.md` |
| *"fs reads peak at 719.3 MB/s at 8 threads"* | **Not the rate the tier gets.** That was a threaded microbenchmark; in service the device delivers **~117 MB/s regardless of fanout**, against a ~101 MB/s recompute break-even (§3). | `kv-cache-future-work.md`, tier-latency table |
| *"~2.45M tokens served ≈ 26 min of prefill avoided"* presented as **the valid quantitative claim** | **Uncitable**, along with every other number taken before 2026-09-10 — not because the harness was wrong but because the engine under it was serving mis-chained KV. Retained as method. Re-measurement is `status-2026-09-10.md` §4 items 2–3. | `kv-cache-future-work.md`, `kv-cache-handover.md` |
| *"Live container state cannot be inspected"* (because `podman exec` fails) | **It can**, read-only, through `/proc/<pid>/root` — see §4. This also unblocks the `tokens_per_chunk` question, which was recorded as blocked rather than merely undone. | `kv-cache-known-issues.md` B2/C1 |
| *"An offload hit costs ~78 s"*; *"promotion: 80 s vs 14 s"*; *"the Mamba layers are the storage waste"*; *"do not buy disks"* | All withdrawn. See §4, §4, §2 and §3 respectively. | various |

---

## 6. Operating the scarce resource — window types

Added 2026-09-11. Windows are the scarcest resource in this project; everything else runs
unattended and free. There are **three** kinds and conflating them wastes them:

| window | what it is | constraints |
|---|---|---|
| **Reload** | the model restarts, arming patches | **resets every lifetime counter** — anything not captured first is gone. llama-swap may stay up. Short. |
| **GPU exclusive** | llama-swap stopped, the GPU used exclusively | required for any timing comparison. **Ask pat in conversation first, every time; never `/unload`; keep the model hot afterwards.** Long. |
| **Quiet** | no reload, no exclusivity, but **no competing traffic** | the resume test needs ≥60 min of genuine idle. **The kvq runners must be stopped first** or their traffic is what gets measured. |

**Sequencing rule for a reload window, decided 2026-09-11.** Landing counters and behaviour
changes together leaves **no "before"**: the instrument and the change arrive at the same
moment and their effects cannot be separated. So: **reload 1** lands every patch with all
behaviour changes at their neutral default (the precedent is `RADIANCE_MAMBA_STORE_STRIDE=1`
disabling the stride patch entirely), making counters live while behaviour is unchanged, and
the instrumented baseline is collected then. **Reload 2** flips knobs on against that
baseline. A patch with no neutral off value does not go in a bundle.

**Ordering rules within any window:** shared setup is paid once; anything that resets counters
precedes anything needing a long history; nothing that warms the cache precedes something
needing it cold; and the shortest activity that could *invalidate* the rest runs first — a
failed boot check should end the window at minute two, not minute ninety.

## 7. Plan exclusions (moved from `cache-preemption-patch-plan.md` R3.14.7)

The closeout plan's "explicitly not doing" list, moved out of the narrative so the register
carries it in one place. **Reopen conditions:** not stated in the source text — for the disk and
CPU-tier rows see §3 above (device physically replaced / more host RAM) and `blocks_per_chunk`
in §2 (—); the attention-only-offload and cascade-scheduling rows carry no reopen condition
anywhere in the source.

Unchanged from R3.10.7 and restated so it is in one place: **no new disks** (the device is
exonerated, R3.9.2), **no CPU-tier resizing** (16 GiB is already at the host's limit, and
R3.12.3 shows the fix is fewer bytes per token, not more bytes), **no attention-only offload**
(it breaks resume correctness for a hybrid model), **no cascade scheduling work** (R3.10.2
withdrew the premise), and **no `blocks_per_chunk` changes** (R3.12.5a).

> **Two of these five are superseded and must not be read as current policy (2026-09-11).**
> - *No new disks* rested on R3.9.2 exonerating the device. Later measurement put the fs tier
>   at **break-even** — it needs ~101 MB/s to tie recompute and the device gives ~117 — so the
>   disk is the constraint, not an exonerated component, and NVMe is ~19x. The exclusion stands
>   only as "no disk purchase has been made", not as "a faster disk would not help".
> - *No CPU-tier resizing* cites "16 GiB is already at the host's limit". The tier is now
>   **24 GiB**, and 24 GiB is the real ceiling (the guest has 39 GB total, 8 available).
>   The conclusion survives — the tier cannot grow further on this hardware — but the number
>   in the reasoning is two revisions old.
>
> The other three (attention-only offload, cascade scheduling, `blocks_per_chunk`) stand as
> written. Task 12 (2026-09-11) independently reinforces the first of those: a group with no
> stored chunks zeroes the entire multi-group hit.

## Deferred — decided, not forgotten

- **The remaining correctness validation runs** (`status-2026-09-10.md` §4 items 1, 2, 3, 5,
  6). **Deferred by pat, 2026-09-11**: they are long-running, the engine-under-test has
  passed them repeatedly, and no new information is expected. Items 2, 3 and 6 are genuine
  re-measurements; item 4 was already met by CT5's gate.
  **Two are not repeats and are deferred anyway, knowingly:** item 1 (output correctness with
  the defect *armed* — invisible to any probe run on patched code, and the difference between
  an inferred and a demonstrated wrong-output bug) and item 5 (**stock reproduction, which
  gates filing anything upstream**). The harness for item 5 exists and self-tests clean
  (`stock_repro.py --self-test` → `GATE: PASS`); only the two-arm run is missing, and it needs
  an unpatched build plus a GPU exclusive window.
  **Consequence, stated so it is not rediscovered:** the arm remains *assumed correct, not yet
  validated correct*. **Reopen condition:** submitting anything upstream — at that moment
  item 5 becomes a blocker, not before.

## Still open — deliberately not in this register

For contrast, so that "not listed here" is not read as "settled":

- **The controlled A/B harness with a genuinely cold arm.** Half of roadmap stage 1. The
  metrics half now exists ([`tier-report-metrics-plan.md`](tier-report-metrics-plan.md) and
  `tools/tierreport.py`); this half does not.
- **REFUTED (2026-09-11/12): the "priority defect" — fs-tier hit reported as `MISS` when the CPU tier is full.**
  The *mechanism* is real and was correctly traced: in
  `v1/kv_offload/tiering/manager.py:386` a secondary-tier HIT returns
  `LookupResult.MISS if not promoted else LookupResult.RETRY`; `_initiate_promotion`
  (`:474`) returns `False` whenever `primary_tier.prepare_write()` returns `None`, which
  `v1/kv_offload/cpu/manager.py:218-233` does when eviction cannot free a slot ("Eviction
  will fail", or the ARC policy's candidates are pinned). But the *empirical claim* — that
  this fires at scale and is why the fs tier "cannot serve under the pressure it was built
  for" — is **refuted**: `kv_offload_promotion_refused_total{fs}` measured **0 across 5,416
  promotions** with the CPU tier pinned at 100% for 90+ minutes, the exact condition the
  thesis predicted would produce refusals. In its working regime the tier serves 73–87% of
  external tokens from disk. The original observation that motivated this (two concurrent
  streams, CPU tier pinned at `used_bytes == capacity_bytes`, 333 evictions / 8.99 GB,
  `lookup_chunk_miss_total` 35 → 6,906, `tier_load_ops{fs}` unchanged at 47 while the tier
  absorbed 9 GB of writes) came from a **warm-up / residency window** — the CPU tier had not
  yet filled, so the fs tier was short-circuited on primary HITs and never consulted
  (`TieringOffloadingManager.lookup` returns on a primary HIT). That made a working tier look
  dead. There was no counter on the refusal path at the time, which is why the observation
  was indistinguishable from genuine misses in `/metrics` and was only found by reading the
  code. The agreed fix shape (bounded `RETRY` + reserved promotion headroom, task 00) is
  **not needed to make the tier serve**; the B1/B2 patches are retained as instrumentation
  only. `cpu_cache_evictable_perc = 1.0` is not counter-evidence: it is sampled between
  scheduler steps, when nothing is pinned. See `out/FAILURE-TAXONOMY.md` §E and
  `../WHERE-WE-ARE.md` §1–§4.
- **Why the fs tier reads ~59 KiB per token served** — *largely resolved 2026-09-11*: stored
  density is 61,440 B/token (see §3), so the fs tier reads close to everything it stored. The
  residue is that the CPU tier reads 35,292 B/token for the same served data, which is
  believed to be read-side (fs reads whole 27,000,832 B group-files, CPU reads only what it
  serves) but is **not yet confirmed**.
- **Where the traffic actually goes, and how much pressure changes it.** Single stream:
  GPU prefix cache 89.8% of prompt tokens, recompute 9.0%, offload tiers 1.1%. Two
  concurrent streams: GPU 50.5%, recompute 18.6%, **offload tiers 30.9%**. The tiers only
  matter under concurrency, and the GPU pool arithmetic says why —
  `kv_cache_size_tokens` 228,737 with `kv_cache_max_concurrency` 1.117, so at a 110k context
  limit exactly 2.08 streams fit and a third must evict.
- **Why a 64 s promotion is still not explained.** [`kv-cache-historical.md`](kv-cache-historical.md) §6.
- **`--max-num-seqs` 4 → 2.** Costs nothing to try, never tried under a controlled replay.
- **Everything in [`kv-cache-known-issues.md`](kv-cache-known-issues.md) §C.**
