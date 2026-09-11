# The experimental record — what was tried, measured, and retracted

This document exists so that a contributor does not spend a week re-running an
experiment whose answer is already known, and — just as importantly — does not build on
a claim that was later withdrawn.

The raw run directories, counter snapshots and result files are **not** shipped: they are
single-machine, some are known polluted, and a number is worth less than the reasoning
that produced it. What is shipped is the reasoning, in order, with each conclusion marked
**STANDS**, **CORRECTED** or **RETRACTED**. Where a claim was corrected, both the original
and the correction are given — the original is usually the more instructive half, because
it shows which plausible reading was wrong.

All work below was done on one machine (see the environment table in `README.md`) between
2026-08-29 and 2026-09-09.

---

## 0. How to read this

The investigation ran in numbered plan revisions, R3.x, defined in
`cache-preemption-patch-plan.md`. The ones that produced a shipped patch:

| Rev | What it is | Outcome |
|---|---|---|
| R3.6 | Store-path instrumentation | Shipped. Gauges only. |
| R3.13 | Mamba store cadence (stride N) | Shipped. The capacity lever. |
| R3.13a | Eagle/MTP group annotation | Shipped. Fixed a real bug. |
| R3.14 | fs-tier job fanout (port of PR #49225) | Shipped, **buys nothing measurable**. |
| R3.14.2 | Lookup-outcome counters | Shipped. Diagnostic; remove when done. |
| R3.14.3 | Pending-is-miss / serve-ready-prefix | Shipped, **and it was the bug**. See §3. |

---

## 1. The AITER block — VOID

**Original claim:** KV offload could not be enabled because the AITER attention backend
did not support it.

**Status: VOID.** Under the R4D attention backend the block does not exist. Offload has
been live since, at 11.5 GiB and later 24 GiB. Two traps were found on the way and both
still apply:

- **`df -h` rounds the tmpfs up.** Do not use it to confirm a `/dev/shm` resize; use
  `findmnt --verify` and the engine's own boot line.
- **Restarting the service orphans the `/dev/shm` region** (and the container). Any
  workflow that restarts must check for the orphan afterwards.

Cost of having offload on at all: ~3.3% prefill.

---

## 2. "The Mamba layers are the waste" — RETRACTED, twice

**Original claim (a):** the six GDN/Mamba KV groups were ~15% of the stored bytes and
mostly padding, so there was large density work available.

**Correction:** measured storage density is **33,808 B/token**, and the full-attention
arithmetic alone accounts for **103%** of it — 16 full-attention layers × 2,048
B/token/layer plus one MTP layer. The Mamba groups contribute ~0–2%, the MTP draft group
(g8) ~6%. **There is no density work left.** Do not re-open compression here.

**Original claim (b):** the Mamba waste is *spatial* — the groups are sparsely populated.

**Correction: the waste is TEMPORAL, not spatial.** Groups g0–g5 are 97–99% dense. What
was wasteful was storing a recurrent state for *every* chunk when the state is
reconstructible from a nearby checkpoint. That is what the stride patch (R3.13) addresses,
and it is why the fix is a cadence, not a compression.

---

## 3. Why the disk tier served nothing — the long one

This is the central thread of the investigation and it changed direction three times.

### 3.1 First hypothesis: the Mamba stride's alignment grid — RETRACTED

The stride patch rounds a lookup **down** to the nearest kept boundary, so the natural
suspicion was that the lookup grid and the store grid did not line up and the request
missed.

An A/B/A was run: three fresh boots, full production environment, only
`KVOFF_MAMBA_STRIDE` varied. Fill with a 34,052-token prompt, evict it from GPU and CPU
with ten more 34k prompts, re-read it — by then it can only come from disk.

| | stride 8 (A1) | stride 1 (B) | stride 8 (A2) |
|---|---|---|---|
| re-read of p0 | 13.5 s | 13.4 s | 13.5 s |
| re-read miss delta | +84 | **+0** | **+0** |
| `chunk_retry` before re-read | 2,266 | 1,484,712 | 3,087 |

A cold 34k prefill costs ~13.3 s. **Every arm was a full recompute.** Stride 1 removed
the alignment mismatch entirely — misses went to zero, retries rose 655× — and changed
the re-read not at all, while making the fill 35% slower from the 8× write volume.

**Alignment was never reached.** The hypothesis is retracted.

### 3.2 The actual mechanism: R3.14.3 cancelled its own wait — STANDS

Traced in `offloading/scheduler.py`, reproduced identically in three independent arms:

1. chunk 0 is in the fs tier → `RETRY` → the manager starts the disk→CPU promotion and
   defers the request.
2. a few milliseconds later the promotion has allocated its CPU blocks but the data has
   not arrived → the key is present-but-not-readable → `HIT_PENDING`.
3. `RADIANCE_OFFLOAD_PENDING_IS_MISS=1` (the then-default) breaks the prefix lookup at
   `local_idx = 0` with `hit_count = 0`.
4. the `num_hit_chunks == 0` guard returns 0 **unconditionally** — no defer, no wait.
5. the request recomputes all 34,052 tokens.

Three scheduler steps, and it gives up on a seconds-long disk read. R3.14.3's own comment
explains the error: it was written for an in-flight **store** (GPU→CPU), where waiting is
pointless because the data is already in GPU. `HIT_PENDING` does not distinguish a store
from a **promotion** (disk→CPU), where waiting is the entire point. It truncated both.

**Metric trap, worth its own line:** `kv_offload_load_size_count` and `load_bytes_total`
scraped empty in every arm. They were not empty — **the whole `kv_offload_load_*` family
did not exist in those boots.** The series are registered lazily on first use, so their
*absence* is itself proof that zero loads were ever issued. (Also: there is no
`lookup_chunk_hit_total`; the family is `lookup_chunk_hit_pending_total`.)

### 3.3 The fix works, and made things slower — STANDS, and is still open

Flipping `KVOFF_PENDING_IS_MISS=0` made the wait real. The load path came alive for the
first time on this host: 18 metric series that had never existed before, real bytes moved
at ~11.8 GB/s from the CPU tier.

**The prediction was wrong.** The re-read did not fall below its ~13.5 s cold cost; it
rose to **17.5 s**, 30% slower than control and slower than recomputing from scratch.

| | pim=1 | pim=0 |
|---|---|---|
| stride 8 | 13.5 s, 0 loads (×4 runs) | 17.5 s, +1 load / 1.06 GB |
| stride 1 | 13.4 s, 0 loads | 18.8 s, +1 load / 1.22 GB |

Completing that 2×2 settled two things at once. **Alignment is not the second terminator**
— stride 1 changes nothing in either column. And **the cache was serving all along**: one
load *operation* moved 1.22 GB, which against p0's 1.384 GB KV footprint is **88.2%** of
the prompt, independently confirmed by `prompt_tokens_details` showing `cached=31312` of
34,052 tokens (**92.0%**) on the first request after a fresh boot, i.e. from the external
tier.

So the cache serves, and the request is *still* slower than recomputing. The leading
hypothesis — fitted to three points and **not yet confirmed against scheduler-step
counters** — is that the deferral loop is a **busy-wait costing ~1.9 ms per iteration**:
10,000 deferrals × 1.9 ms ≈ 19 s of spinning, against `load_time_total` of **103 ms** of
actual I/O. Nineteen seconds of spinning for a tenth of a second of disk work.

**This is the single most valuable open lead in the repository**, and it is why the
roadmap puts metrics first. Confirm it against scheduler-step counters before acting on
it.

### 3.4 An earlier reading of the same counters — CORRECTED

`lookup_served` is a **branch counter, not a hit rate**. In a 5h43m production window it
read 448 — and all 448 belonged to a *single* request, re-looked-up ~420 times over one
60-second window, once per scheduler step, admitted once. The accompanying reading that
"the external share fell to 4.14%, below Phase A's 5.54%" was a **denominator artifact**:
external stayed frozen at one load while the GPU prefix cache did more work.

A second correction from the same window: an earlier note said Phase B's serve hits were
one request in one window and therefore marginal. Over a 6h36m window it served **448 of
755 lookups (59.3%)**. Phase B is not marginal; it works.

That one hit is also the cleanest single measurement in the whole investigation:

    external load        4.46 GB, 130,192 tokens  (34,264 B/token — the known read shape)
    CPU -> GPU copy      0.378 s   (11.8 GB/s)
    fs  -> CPU promotion 64.26 s   (ONE observation)

---

## 4. The eagle-group bug — FIXED, and it was real

Straight off the boot log:

    KV offloading: EAGLE/MTP draft attention groups [0,1,2,3,4,5,6,7,8] detected

**All nine KV groups flagged as speculative-draft groups.** Only g8 is one. The cause:
DFlash2 sets `use_eagle()` but no group carries `is_eagle_group`, so vLLM's fallback marks
every group eagle — and vLLM's positional annotator is hard-gated to DeepSeek-V4, so it
never fires for this model. While every group is flagged eagle, `storable_chunks()` drops
the trailing chunk of *each* group during decode and the store grid stops lining up with
the hit window.

Fixed by `patch_kv_offload_eagle_groups.py`. Deployed and verified 2026-09-08; with the
stride patch it took the write volume from 189 files to **75**. Upstream counterpart:
**PR #55390** — only ~11% of its diff applies to this tree (radiance has its own
`_annotate_eagle_groups_deepseek_v4`), but it is small and it confirms the fix *shape*:
annotate MTP draft groups **positionally** on the hybrid grouping path. **Port the idea,
not the diff.**

---

## 5. The disk is the limit — STANDS, and reverses an earlier instruction

R3.14 ported PR #49225's fs-tier job fanout by hand (the PR itself does not apply: 44 of
56 removed lines in `manager.py` are absent here, because this tree predates two upstream
refactors). The port is manager-only, five edits; `thread_pool.py` needed no change at all
— the pool already accepted N tasks per job, the manager simply never used it.

**The mechanism works exactly as designed and buys nothing measurable.**

| | control (upstream, 1 task) | fanout (256 MiB budget) |
|---|---|---|
| bytes promoted | 1570.8 MB | 1570.8 MB |
| threads that read | **1** | **8** (5 × 206.0 MB, 3 × 180.2 MB) |
| effective read rate | 116.3 MB/s | 119.2 MB/s |

The split is exact — 61 blocks, and 5×8 + 3×7 = 61 is precisely the largest-remainder
shape. The disk does not care. **The device is the limit, not the queue depth**, arrived
at from a new direction and agreeing with the earlier component measurement: ~117 MB/s
against a ~101 MB/s recompute break-even, **1.16×**.

Note also that upstream's own constant (32 MiB ÷ block size) would have produced a fanout
of **2** here, dropping to 1 with a second job in flight — the PR as written would have
done essentially nothing on this hardware. The budget was therefore made a knob.

**A retracted intermediate result, and the lesson attached to it.** An earlier control run
appeared to show **80.0 s against the fanout arm's 14.3 s — a 5.6× difference**. It is
**withdrawn.** The control was taken straight after a fresh boot with an empty CPU tier, so
it promoted 21,305.5 MB where the fanout arm promoted 1,570.8 MB. Nothing about them was
comparable. Giving the control the identical eleven-prompt fill made the difference vanish.
**Never compare two promotion arms without an identical cache fill.**

One loose end recorded rather than concluded: that confounded run moved 21,305.5 MB in
92.4 s across 8 threads = **230.6 MB/s**, roughly double the ~117 MB/s seen elsewhere.
Those threads were serving *eight concurrent jobs*, not one job split eight ways, and the
files may have been warm in the page cache. **Whether job-level concurrency genuinely
reaches ~230 MB/s where task-level fanout stalls at ~117 is not settled**, and would need
its own controlled test. It does not change the R3.14 verdict, which is about a single job.

**Consequence:** "DO NOT BUY DISKS" — issued when it looked as though zero reads were
being served — is **reversed**. Reads are real, the device is the ceiling, and NVMe is
~19× this device. But **do not re-bench this disk**: it has been measured from three
independent directions and it gives ~117 MB/s.

---

## 6. Why the 64 s promotion is still not explained — OPEN

R3.14 was taken up because the fs→CPU promotion measured 64.26 s and single-threaded
execution was the obvious suspect. It was a genuine defect — confirmed at the binary
level, since `nm -D` and `strings` on `vllm/fs_io_C.abi3.so` show no `pthread`, no
`io_uring` and no `aio`, only `PyEval_SaveThread`/`RestoreThread`, so the C extension
cannot have been parallelising internally either.

Fixing it did not move the number, because the number was never about threads. At
~117 MB/s, 64 s of promotion is ~7.3 GB of reads, which is what a large multi-request
working set actually costs on this device. **The lever, if there is one, is reading fewer
bytes — not reading them on more threads.** That is the same direction R3.13 took.

---

## 7. Instrument defects found the hard way

Every one of these cost real time. They are the reason stage 1 of the roadmap is stage 1.

- **Request wall time is the wrong instrument for tier work.** A cold 34k prefill costs
  ~13.3 s on this box regardless, and the promotion runs underneath it. Both arms of a
  fanout A/B returned in ~13.5 s and that number says nothing about either. Judge the tier
  by `load_bytes`.
- **Never compare promotion arms without an identical cache fill** (§5).
- **Content-hash pollution.** vLLM matches the prefix cache by block *content*, not by
  chained prefix. A harness that makes a run "cold" by varying a nonce in **block 0 only**
  does not get a cold run — the body self-caches from the previous iteration. This
  invalidated an entire BetterBench run for this stack. **Vary the whole body.**
- **A tier phase that did not happen still reports a number.** An early harness pushed
  27 GB through a 16 GiB CPU tier, so its "CPU re-read" phase was really a third disk read
  and all three re-read phases returned the same ~78 s. `tierbench.py` refuses to report a
  phase whose tier state did not come out as intended, and that refusal is the whole reason
  it exists.
- **CPython 3.12 does not push `threading.Thread(name=...)` down to the OS thread `comm`**,
  so the twelve pool workers cannot be found by name — they all appear as `python3`.
  Per-thread `/proc/<pid>/task/*/io` counters are the only way to see a split.
- **The correctness check that proves nothing.** Reference and test hashed identically —
  but the response was **five characters** (`\n\nACK`), because the prompt was "read this
  and reply ACK". That proves nothing about 34,052 tokens of replayed KV. **Replace the
  check prompt with one demanding a long, content-dependent answer** before treating a
  pass as meaningful. This is still outstanding.
- **Harness defects that take the server down.** One run ended with `systemctl start
  llama-swap`; polkit's cached authorisation had lapsed over the ~35 min run, and
  production was left down. Another health-checked twice with no patience for a ~4.5 min
  boot. Any script that takes an exclusive window must verify the restore and say so
  loudly, or hand the restore back explicitly.

---

## 8. The upstream sweep, and the lesson it produced

The first two upstream sweeps found nothing and concluded no fix existed. Both were wrong,
and wrong in the same way: they judged candidates by **merge status** and looked only at
marquee RFCs.

Redone as an **applicability test** — for each PR, fetch the diff and count how many of its
removed lines still exist verbatim in the installed tree — the picture inverted. Two PRs
apply directly, and one of them (#54327) would replace a component this proof of concept
currently carries in a shell script. The measured table is in
`kv-cache-references.md` §3a.

**LESSON, and it generalises past this project:** scan upstream by **applicability against
the installed tree**, not by merge status or issue prominence. The useful work was in small
open `[Bugfix][KV Offload]` PRs updated within the last week, not in the headline RFCs. An
unmerged PR that applies is worth more than a merged one that does not.

---

## 9. What was measured and then deliberately left alone

> The full register, including the closures reached elsewhere in this document and the
> condition that would reopen each one, is
> [`kv-cache-closed-decisions.md`](kv-cache-closed-decisions.md). The list below is the
> subset that belongs to the experimental record; the reasoning behind every row is above.

- **Storage density** — at the architectural floor (§2). Closed.
- **The disk device** — ~117 MB/s from three directions (§5). Closed; do not re-bench.
- **`blocks_per_chunk`** — investigated, closed.
- **Disk-hit correctness** — an 85,696-token disk hit was verified **bit-identical** to a
  cold recompute. The store/load path is exact; the approximation is the *stride*, and only
  the stride (see `kv-cache-known-issues.md`).
- **GPUDirect / DMA disk→VRAM** — foreclosed three separate ways (see `README.md`). Closed.

---

## Appendix — the superseded plan revisions, moved from `cache-preemption-patch-plan.md`

The two earlier revisions of `cache-preemption-patch-plan.md` — Revision 2 (2026-09-07) and
Revision 1 (the original, §0–§9) — are reproduced verbatim below, for the reasoning. Revision 3
in the plan supersedes both where they disagree.

# Revision 2 — 2026-09-07 (review of rev 1 against the live server)

**Everything below the next `---` is rev 1, preserved for its reasoning. Where rev 1 and this section disagree, this section wins.**

Rev 1's thesis — *retain, don't recompute* — is still right. What was wrong is the cost model underneath it, and therefore the build order. **Do not start on the reaper.** Start at §R2.3.

Verified against: the live server (container `qwen38-27b-vllm`, engine PID 5899, booted 2026-09-06 21:50 — the same boot the audit and `hitrate-bench.py` ran on, so its cumulative counters still contain those runs); the vLLM 0.27.1 tree inside the container (`/proc/5899/root/opt/vllm/lib/python3.12/site-packages/vllm/`); `~/audit/stress/`; `<repo>/kvcache-reap.sh`; `/etc/fstab`.


## R2.1 The 12 s read-back is synthetic. Measured, a read-back costs ~78 s.

> **WITHDRAWN by Revision 3 (R3.1).** No read-back was measured here — the offload tier served zero bytes. The ~78 s was a lookup stall plus a full recompute.

`~/audit/stress/results/results.md`, `max_tokens=16`, the same ~90k-token prompt in every phase:

| phase | served from | Δ read | wall |
|---|---|---|---|
| WARM / HIT-GPU / SUSTAINED | GPU (97.0% hit) | 0 GB | **2.15 / 2.21 / 2.16 s** |
| RE-READ-CPU / FS / FS2 | offload (95.1% ext hit) | 2.796 GB | **77.0 / 78.5 / 78.0 s** |

`hitrate-bench.py` reproduces the shape independently: 2.1 s on a GPU hit, 84.6 s on an offload hit. The 4.88% that genuinely recomputes is ~4,400 tokens ≈ 2.6 s at this plan's own 1,699 tok/s calibration, so it accounts for none of the gap.

Effective rate: **2.796 GB / ~75 s ≈ 37 MB/s.** Cold recompute of the same ~90k tokens at 1,699 tok/s is **~53 s**. As the system stands today, **read-back is slower than recompute** — the opposite of §0 and §1.3.

§1.3 attributes the 77 s to a "just-wrote / pending-txg / first-read-back artifact". The artifact runs the other way: `llama-swap-ggz14-27b.sh` annotates its own `dd` figures as *"taken shortly after writing the test files, and with sync=disabled the host may still have had them in a pending txg, so the absolute numbers are optimistic."* The 12 s is the synthetic number; the ~78 s appeared three times consecutively and a fourth time in a separate harness.

**This does not kill the plan** — see R2.2.

## R2.2 The 75 s is not in the transfers, and the disk cannot explain it

> **SUPERSEDED by Revision 3 (R3.2).** The 75 s is in the store path, not the read path: the fs cascade pins the CPU tier shut (130:1 fill-vs-drain).

Live counters (`/metrics`, cumulative since boot):

| leg | bytes | time | rate |
|---|---|---|---|
| GPU→CPU (store) | 340.3 GB / 1,540 transfers | 33.95 s | **10.0 GB/s** |
| CPU→GPU (read-back) | 44.9 GB / 23 transfers | 3.82 s | **11.8 GB/s** |
| fs→CPU (promotion) | — | — | **no counter exists** |

`vllm:kv_offload_load_*` is numerically identical to `vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` — the same leg exposed twice. The secondary-tier leg is not instrumented at all.

So the 2.796 GB spends **~0.24 s** entering the GPU. (This *confirms* the "CPU tier ≈ 0.3 s" figure for that leg — what was never measured is an end-to-end CPU-tier hit as a request experiences it. Note also that `RE-READ-CPU` did not measure the CPU tier: `EVICT-1` wrote 27.19 GB through a 16 GiB tier, which evicts a 2.8 GB prefix by construction, so all three re-read phases were fs reads. That is why 77.0 / 78.5 / 78.0 s are indistinguishable.)

The remaining ~75 s sits inside the window measured by `vllm:kv_offload_tiering_lookup_async_delay_seconds` — *"wall-clock from a request's first deferred secondary-tier lookup until the request is allocated or finishes"* — currently **911.1 s across 160 observations**, with nothing decomposing it.

**The array cannot explain that window.** 2.796 GB costs 3.9 s at the launcher's O_DIRECT figure (8 threads, 719 MB/s) and 11.7 s at the audit's pessimistic settled figure (240 MB/s). On the most disk-unfavourable number in the tree the media is **under a sixth** of the stall; on the optimistic one, a twentieth.

**Consequence for hardware:** more spindles and a passthrough controller target the smallest term. Fixing the array completely would move a re-read from ~78 s to ~66 s. Do not provision disks until R2.3 has produced a number.

**Consequence for the plan:** the hardware already demonstrates 11.8 GB/s into the GPU and at worst 240 MB/s off the array. A 2.8 GB read-back *should* cost 4–12 s. The 5× win in §0 is achievable — it is being lost in the promotion path, which no reaper policy, no admission policy and no disk touches.

**One measured lead:** `vllm:kv_offload_allocation_failure_total = 194`. In `tiering/manager.py::_initiate_promotion`, a full primary tier returns `False`, which `lookup()` converts to `LookupResult.MISS` — the block is *abandoned*, not retried. That is a path by which a cache hit becomes a recompute, and it is already firing 194 times on this boot.

## R2.3 START HERE — instrument the promotion leg

**Build sheet with exact anchors: R2.9.2.** This section is the *why*; R2.9 is the *how*.

This is the smallest job in the plan and it gates everything else, including the disks.

Add, in the tiering manager / fs tier:
- a timer around the fs→CPU transfer itself (submit_load → job completion), exposed as a histogram per tier — the direct analogue of `kv_offload_load_time_total` for the secondary leg;
- a count of **retry rounds per request** (how many scheduler steps a request spends in `LookupResult.RETRY` before it is allocated);
- a counter for promotions abandoned because the primary was full, distinguished from `kv_offload_allocation_failure_total`'s store-side failures.

Anchors: `v1/kv_offload/tiering/manager.py` — `lookup()` ≈282, `_initiate_promotion()` ≈380, `_flush_pending_promotions()` ≈429, `_process_finished_jobs()` ≈246. `v1/kv_offload/tiering/fs/manager.py` for the tier-side transfer.

**Candidate mechanisms this will separate** (all read out of the 0.27.1 tree, none yet proven):
1. **Per-step promotion cadence.** `_initiate_promotion` defers `submit_load` to `_flush_pending_promotions()`, called once per scheduler step from `on_schedule_end()`; `_maybe_process_finished_jobs()` polls *"at most once per step"*. A 61-block prefix may need many steps, each doing nothing else.
2. **Abandoned promotions.** A full primary → `MISS` → the block is dropped from the promotion set for that pass (the 194 above).
3. **The array.** Bounded at ≤1/6 of the window by R2.2.

Decision rule once the timer exists: if the fs leg is the bulk, provision disks. If the retry-round count is the bulk, the fix is in the promotion path and disks change nothing.

## R2.4 Aspect-by-aspect corrections

### Aspect 1 (reaper) — demoted, and not implementable as scoped

- **Problem statement is stale.** §1.4 / §2.2 describe a FIFO 80%→65% reaper. That was rewritten on 2026-09-06. `kvcache-reap.sh` is now age-based: Stage A deletes anything older than `MAX_AGE_HOURS=8` **every cycle regardless of usage**; Stage B is the capacity stage, still oldest-mtime-first, floored at `MIN_AGE_MIN=90`. A new sort key only changes Stage B.
- **LRQ cannot be a reaper-only change.** Last-*query* time is not observable from outside the engine: `/kvcache` is mounted `noatime,nodiratime` (live mount and `/etc/fstab`), and the fs tier writes each block once and never touches it again. Three options, cheapest first: remount `relatime` (gives a one-bit "ever read since written", which for a write-once cache is most of the value); remount `strictatime` (true LRU, one metadata write per read); or have vLLM emit query times — which is a vLLM patch and contradicts §5's "not a vLLM patch". Sidecar-per-readback is the most expensive of the three and §2.2 picks it by default.
- **The safety property is missing from Aspect 1.** `MIN_AGE_MIN` exists because `offloading/worker.py` does a bare `assert transfer_result.success` and `OffloadingConnector` implements no `get_block_ids_with_load_errors()`, so `kv_load_failure_policy=recompute` cannot catch a block reaped between lookup and load — it kills EngineCore. **Any new sort key must preserve the floor.** See the header comment in `kvcache-reap.sh`.

### Aspect 2 (CPU-tier admission) — right intent, wrong mechanism; a supported one exists

- **The described mechanism does not exist.** `TieringOffloadingManager`'s docstring: *"Always offload to all tiers — when a block is stored to the primary tier, it is cascaded to ALL secondary tiers"* and *"secondary tiers cannot access GPU memory directly; all data flows through the CPU primary tier."* fs is fed **from** CPU. So "admit the hot blocks to CPU, demote the rest to fs" is not expressible: a block not admitted to CPU never reaches fs either, so `VLLM_CPU_OFFLOAD_DEMOTE_TO=fs` collapses to `drop` — the setting §2.3 reserves for "only when the fs tier is full".
- **Do it as a `CachePolicy` instead — no patch at all.** `TieringOffloadingSpec` accepts `eviction_policy` plus `cache_policy_module_path`, and `CachePolicyFactory.get_cache_policy_cls` loads an out-of-tree class — *"an out-of-tree policy needs no register_cache_policy() call at all, just this module path passed through config"*. The ABC (`v1/kv_offload/cpu/policies/base.py`) exposes `get`, `insert`, `remove`, `touch(keys, req_context)`, `evict`, `mark_evictable`, `mark_non_evictable` — everything a recency or refcount policy needs, with the query signal already wired in. Model it on `policies/lru.py` / `policies/arc.py`.
- **It now has a measured motivation:** the 194 allocation failures in R2.2. Keeping the 16 GiB primary from being saturated by cascade traffic is exactly what reduces abandoned promotions.
- Ship it as a module + two config keys in `KVOFF_TIER_ARG`, not as `patch_offload_cpu_admission`. It survives image bumps; a connector patch does not.

### Aspect 3 (queue on overflow) — will deadlock as written; fix before enabling once

- **Deadlock.** The insertion point is inside the *running*-request loop (`scheduler.py` ≈576-637). Deferring instead of preempting leaves the request in `self.running` still holding its blocks, and the outer `if new_blocks is None: break` skips every running request behind it too. Two long requests, pool full, both needing one more block: neither is scheduled, neither finishes, neither frees anything — every step, forever. Preemption is precisely the tie-break that makes that state recoverable. §3's *"no request starves as long as max_num_seqs is bounded"* does not hold, and §4.3 frames the risk as one step of GPU idle rather than a hang.
- **It inverts the admission gate.** The waiting phase is gated on `if not preempted_reqs` (≈684). A deferral does not populate `preempted_reqs`, so the patch *unblocks* admission of new waiting requests in exactly the step the pool overflowed — the opposite of the intent. Gate the waiting phase on the defer flag too.
- **The counter will not run.** `self._iteration_stats` does not exist on the scheduler (zero occurrences in `scheduler.py`) — the §2.4 snippet raises `AttributeError` on the first defer. `num_preempted_reqs` is incremented at `stats.py:449` from an `EngineCoreEventType.PREEMPTED` event during output processing, not by the scheduler. A deferred counter must travel the same event route: emit an event from the scheduler, count it in `IterationStats.update_from_events`.
- Minimum safe shape: defer only when at least one *other* running request is still schedulable this step, and fall back to stock preemption otherwise. That keeps the tie-break available and makes the deadlock unreachable.

### §5 — "no existing patch touches `scheduler.py`" is false

`patch_dynwidth.py` (`TARGET = SP / "vllm/v1/core/sched/scheduler.py"`) and `patch_offload_mixed_hit.py` both patch it. The `scheduler.py:2924-2928` env-var block §2.1 says to copy **is** patch_dynwidth's own insertion. Handle apply order and anchor-text collision; add the new patch after both in the boot list.

## R2.5 The recompute/miss diagnostic does not transfer to live traffic

Live right now: Q = 18.249 M, H = 13.665 M, E = 1.234 M → **(Q − H − E)/Q = 18.4%** (GPU 74.9%, external 26.9%). Against §2.6's "baseline 3–5%" and §6.5's "well above the baseline confirms the crawl", the server is crawling. It is not.

The reason is structural: every first-ever prompt contributes ~100% miss to a cumulative counter, so the fraction measures **workload novelty, not retention**. It is a sound A/B statistic *inside* `hitrate-bench.py` — fixed prefix, controlled re-query, delta counters, which is where 3.05% and 4.88% came from — and it is not a server-level health threshold.

Keep it bench-only, or redefine it on windowed deltas with the novel-prefix term separated out. **Do not gate the build on the cumulative figure.**

## R2.6 The workload premise (§1.1) is contradicted by the previous boot

§1.1 says the real pattern is 1 long + 1 short, co-fitting, and that the 38 preemptions were a stress-test artifact. True *for this boot* — all 38 came from the audit's 2-concurrent phases, and `hitrate-bench.py` logged 0.

But the previous boot's **live opencode traffic** recorded **151 preemptions across 136 requests** at identical geometry (228,737 tokens, 138 blocks, `Maximum concurrency for 204,800 tokens per request: 1.12x`), with 49 of those 136 requests over 100,000 tokens — see `~/vllm-kv-cache-offload-reuse.md` §2 and §3.8, read off `/metrics`, not a stress test.

Two >100k requests cannot co-fit in 138 blocks, and `--max-num-seqs 2` admits both — nothing serializes them. **"2-long would serialize" is an assumption about usage, not a property of the config.** Pat confirms the normal shape is multiple endpoints/software chatting plus opencode, i.e. the multi-client regime, not the quiet single-client one this boot has seen. So Aspect 3 is not a corner case and cannot stay a permanent "safety net".

## R2.7 "Size the fs tier to the working set" (§6.2) is not reachable at 512 GB

The audit measured 100–135 MB/s net-new store under load (119 GB over the stress window); the settled rate is 44.6 MB/s = 3.85 TB/day. 512 GB is one to four hours of load. Current state: 102 GB used, 4,048 blocks, oldest block 4.4 h old at light load.

§6.2's criterion — the high-water exceeds the total evicted KV — cannot be met by sizing, only by storing less. `kvcache-reap.sh`'s own header reaches the same conclusion from the other direction (*"if this line repeats, the volume is too small for this workload"*). State the goal as a residency window in hours, not as "hold the working set".

## R2.8 Revised build order

> **SUPERSEDED by Revision 3 (R3.6).** Nothing reaches the point where retention policy applies; fix the store-path livelock first.

1. **Instrument the promotion leg (R2.3).** Nothing else is decidable until this exists. Smallest job in the plan. → build sheet **R2.9.2**.
2. **Aspect 2, rewritten as an out-of-tree `CachePolicy` module** (R2.4). No patch, measured motivation, survives image bumps. → build sheet **R2.9.3**.
3. **Disks** — only if R2.3 says the fs leg is the bulk of the window. One number decides it.
4. **Aspect 1 (reaper)** — secondary, per pat; and it is already updated. If revisited, try the `relatime` remount before writing sidecars, and preserve `MIN_AGE_MIN`.
5. **Aspect 3** — needs the deadlock and the admission-gate inversion fixed before it is enabled even once. Under R2.6's multi-client regime it is eventually load-bearing, not optional.

**Also worth testing early, cheaply:** pat describes a "prompt rebuilding stall" under heavy load, which assumes the cache missed. R2.1 shows a *hit* also costs ~78 s. During a stall, if `kv_offload_tiering_lookup_async_delay_seconds` climbs while `prefix_cache_hits_total` also climbs, the cache is working and the promotion path is the stall — a different bug from the one this plan was written to fix, and one R2.3 would localise.


## R2.9 Implementation appendix for steps 1 and 2 (build sheet)

Everything in this appendix was read out of the running container's tree
(`/proc/5899/root/opt/vllm/lib/python3.12/site-packages/vllm/`) on 2026-09-07. It exists so the
coding agent does not have to rediscover the metric plumbing. Line numbers are from that tree and
will drift; the anchors are the function and class names.

### R2.9.1 How an offload metric reaches `/metrics` (five hops)

1. **Definition.** A metric is declared by a `build_metric_definitions(cls, extra_config)`
   classmethod returning `{name: OffloadingMetricMetadata}` —
   `OffloadingCounterMetadata` / `OffloadingGaugeMetadata` / `OffloadingHistogramMetadata`
   (`v1/kv_offload/base.py`). Three places implement it and they compose:
   - `CPUOffloadingSpec.build_metric_definitions` — `cpu/spec.py:31`
   - `TieringOffloadingSpec.build_metric_definitions` — `tiering/spec.py:82`, which calls `super()`
     **and then fans out to every configured secondary tier** at `tiering/spec.py:137`:
     `metrics.update(tier_cls.build_metric_definitions(tier_config))`
   - `SecondaryTierManager.build_metric_definitions` — `tiering/base.py:307`, default `{}`.
     **The fs tier does not override it. This is the hole to fill.**
2. **Registration.** `OffloadPromMetrics.__init__`
   (`distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:328`) merges
   `spec_cls.build_metric_definitions(extra_config)` with the connector's own definitions and
   creates one Prometheus object per name in `_create_metric` (`:386`). **Nothing else needs
   touching** — no edit to `v1/metrics/loggers.py`, no allowlist.
3. **Emission.** Code writes into an `OffloadingConnectorStats` via `increase_counter` /
   `set_gauge` / `observe_histogram` (`metrics.py:275-308`). The payload is self-describing and
   survives IPC.
4. **Collection.** `TieringOffloadingManager.get_stats()` (`tiering/manager.py:822`) already
   collects `primary_tier.get_stats()`, **then loops over `self.secondary_tiers` calling
   `tier.get_stats()`** (`:829`) and aggregates, then merges its own `self._stats` buffer
   (allocated at `:224`). So both hooks a new metric needs are already wired.
5. **Export.** `OffloadPromMetrics.observe()` (`metrics.py:481`) dispatches by type onto the
   Prometheus objects. `assert key in self._offloading_metric_defs` — **a metric emitted in hop 3
   but not declared in hop 1 raises, it does not silently vanish.** That assert is the
   verification for free.

Template to copy: `CPUOffloadingManager.get_stats` (`cpu/manager.py:293-330`) — builds a fresh
`OffloadingConnectorStats`, writes gauges/counters/histograms, clears its accumulators, returns.

### R2.9.2 Step 1 build sheet — instrument the promotion leg

**Deliver as one patch script**, `patch_tiering_metrics.py`, in `<repo>/ggz14-mxfp4/`, using
`from _patchlib import apply` (`apply(path, anchor, new, sentinel, label)` — idempotent, one-shot,
fatal on a non-unique anchor, `ast.parse` before write). Copy the shape of
`patch_offload_mixed_hit.py`. Add it to the boot list in `llama-swap-ggz14-27b.sh` (~line 1393,
next to `patch_offload_mixed_hit.py`).

One patch rather than an out-of-tree tier subclass because counter **(c)** below is unreachable
from a tier — see the note at the end of this section.

Four edits:

**(a) `tiering/fs/manager.py` — time the fs leg.** `FileSystemTierManager` already tracks jobs by
id: `submit_load` (`:221`) enqueues, `get_finished_jobs` (`:234`) drains `self._pool.get_finished()`.
- in `__init__`: `self._load_t0: dict[JobId, tuple[float, int]] = {}` and a list
  `self._load_samples: list[tuple[float, int]] = []`
- in `submit_load`: `self._load_t0[job_metadata.job_id] = (time.perf_counter(), len(job_metadata.keys) * self._block_size)`
- in `get_finished_jobs`: on each completed id, pop `_load_t0` and append
  `(perf_counter() - t0, nbytes)` to `_load_samples`
- do the same for `submit_store` (`:207`) if cheap — the cascade write competes for the same
  threadpool and may be what starves reads. `n_read_threads=8` / `n_write_threads=4` are set in the
  launcher at `KVOFF_DISK_RTHREADS` / `KVOFF_DISK_WTHREADS`.
- add `build_metric_definitions` (classmethod, `@override`) and `get_stats` to the class. The spec
  already calls both.

**(b) `tiering/fs/manager.py` — count retry rounds per request.** `FileSystemTierManager.lookup`
(`:200`) returns `LookupResult.RETRY` when `self._lookup_manager.lookup(...)` returns `None`. Count
per `req_context.req_id` (`ReqContext` is at `base.py:91`, `req_id: str`), and observe the total as
a histogram in `on_request_finished` (`:265`), which is where `self._lookup_manager.cleanup` already
runs. **This is the number that decides step 3**: many retry rounds with a fast fs leg means the
stall is the per-step promotion cadence, not the disk.

**(c) `tiering/manager.py` — count abandoned promotions.** `_initiate_promotion` returns `False`
when the primary tier has no room, and `lookup()` converts that to `LookupResult.MISS` — the block
is dropped rather than retried. Add
`self._stats.increase_counter(TieringOffloadingMetrics.PROMOTION_ABANDONED)` on that path;
`self._stats` already exists (`:224`) and is already drained by `get_stats` (`:838`). Declare the
name in `TieringOffloadingMetrics` (`tiering/base.py:37`) and its metadata in
`TieringOffloadingSpec.build_metric_definitions` (`tiering/spec.py:82`). This is the counter that
distinguishes a full primary from a slow disk, and it is why the whole job is one patch rather than
a tier subclass: a `SecondaryTierManager` cannot see this decision.

**(d) `tiering/spec.py` — widen the `LOOKUP_ASYNC_DELAY` buckets.** They currently top out at
`10` (`tiering/spec.py:110-122`). The stall being chased is ~75 s, so **every observation of
interest is already in `+Inf`** — which is why the live `_sum` of 911.11 s over 160 observations
cannot be decomposed. Extend to `(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)` and give
every new histogram in (a) and (b) the same range. Getting this wrong makes the rest of step 1
useless.

Suggested names (flat, `vllm:` prefixed, matching the existing convention):
`vllm:kv_offload_fs_load_seconds`, `vllm:kv_offload_fs_load_bytes`,
`vllm:kv_offload_fs_store_seconds`, `vllm:kv_offload_fs_retry_rounds`,
`vllm:kv_offload_tiering_promotion_abandoned`.

**Verify:**
```
curl -s localhost:1234/upstream/qwen3.8-27b-vllm/metrics | grep kv_offload_fs
```
Then run **`tierbench.py`** (this directory; see `README.md`) before and after the patch. Do not
use `~/audit/stress/harness.py` for this: its EVICT-1 phase pushes 27.19 GB through a 16 GiB CPU
tier, so its "RE-READ-CPU" phase is really a third fs read — which is why its three re-read
phases returned 77.0 / 78.5 / 78.0 s. tierbench sizes each eviction from the tier capacities it
reads out of the boot log, so its `cpu` and `fs` phases land in the tiers they claim, and it marks
a phase INVALID rather than reporting a number from a cache state it did not achieve.

The patch is correct if and only if, in tierbench's output:

- `fs_load_seconds` delta is ~0 in the **cpu** phase and large in the **fs** phase — that is the
  separation nothing on this build can currently make;
- `fs_load_seconds` sum is well under the `fs` phase's wall time;
- the report's **Attribution** section shows a residue. Budget: fs-leg sum + retry rounds × step
  time should account for the ~75 s. Whatever is left over is a third mechanism and needs its own
  hunt — tierbench prints that residue as a percentage of wall time so it cannot be overlooked.

### R2.9.3 Step 2 build sheet — Aspect 2 as an out-of-tree `CachePolicy`

**No patch at all.** `CachePolicyFactory.get_cache_policy_cls(name, module_path)`
(`cpu/policies/factory.py:39`) imports `name` from `module_path` when it is not a registered
policy, asserts `issubclass(policy_cls, CachePolicy)`, and logs an experimental-API warning. It is
reached from `tiering/spec.py:185-186` (`cache_policy=self.eviction_policy`,
`cache_policy_module_path=self.cache_policy_module_path`) → `tiering/manager.py:96,102`.

1. **Write** `<repo>/ggz14-mxfp4/radiance_cachepolicy.py` with a class subclassing
   `CachePolicy` (`cpu/policies/base.py:36`). Required: `get`, `insert`, `remove`,
   `touch(keys, req_context)`, `evict`, `clear`; optional: `mark_evictable`,
   `mark_non_evictable`. `__init__(self, cache_capacity: int)`. Model it on
   `cpu/policies/lru.py`; `cpu/policies/arc.py` is the worked example of a non-trivial policy.
   `touch()` is the query signal Aspect 1 was trying to reconstruct from the filesystem — here it
   is handed to you.
2. **Ship it** by adding the filename to the existing `cp radiance_*.py "$SP"/` line in
   `llama-swap-ggz14-27b.sh` (~line 1408). It then imports as `radiance_cachepolicy` from
   site-packages — no `PYTHONPATH`, no mount changes. (`/patches` is bind-mounted, but the boot
   block deliberately `cd /`s out of it before exec; see the comment there about a stale `.so`
   shadowing site-packages.)
3. **Enable it** by adding two keys to `KVOFF_TIER_ARG` (`llama-swap-ggz14-27b.sh:884`):
   `"eviction_policy":"<ClassName>","cache_policy_module_path":"radiance_cachepolicy"`.
   Absent both keys the default stays `lru` (`tiering/spec.py:29`), so this is reversible by
   deleting two keys.
4. **Target:** reduce `kv_offload_allocation_failure_total` (194 on this boot) and the new
   `promotion_abandoned` counter from R2.9.2(c), without hurting
   `external_prefix_cache_hits_total`. Both are cumulative — A/B on deltas over a fixed
   `tierbench.py --yes --concurrent 3` run, not on absolutes. The concurrent phase is the one
   that generates allocation pressure; the single-client phases will not move these counters.

### R2.9.4 What this appendix deliberately does not specify

- **The policy's actual eviction rule** (step 2). Recency, refcount and a hard cap are all
  plausible; the numbers from step 1 should pick it. Write the module against whichever the data
  supports, not against Aspect 2's original `none|recency|refcount|cap` flag list.
- **Step 3 (disks)** is a decision, not code: one comparison of the fs-leg sum against the ~75 s
  window. Do not provision anything until R2.9.2 has produced that number.
- **Steps 4 and 5** stay as scoped in R2.4.


---

## 0. TL;DR (what to build)

> **SUPERSEDED IN PART — see R2.1, R2.2, R2.8.** The 12 s read-back is a synthetic figure; measured, an offload hit costs ~78 s, and the disk accounts for at most a sixth of that. The 5× target stands; the build order does not.

**The aim:** **never regenerate a cached prefix from scratch**, given the storage backend reads back a 2.8 GB prefix in ~12 s (8-thread, cold) vs 59 s to re-prefill — a **5× win**. A cached prefix is always cheaper to read back than to regenerate, for long prefixes.

**The real failure mode:** as the opencode session and the chat conversation **grow**, their combined context converges on the 138-block GPU pool limit. At that point the pool overflows and requests get evicted. The key distinction:
- **Retained (offloaded):** the evicted KV is kept in the fs (12 s) / CPU (0.3 s) tier and read back — a memory transfer, cheap, barely disruptive (H2D copy, not GPU compute).
- **Dropped (not retained):** the evicted KV is gone (the offload tiers are full, or the reaper already evicted it) → the only option is **recompute (59 s)** — a big GPU prefill that starves the co-batched in-flight requests → **the server crawls.**

So the crawl = the evicted KV being **dropped, not retained**. The work is **RETENTION**: keep the evicted KV in the offload tiers so every overflow is a read-back, not a recompute.

Three aspects (each flag-gated, reversible, default off = byte-identical to stock):
1. **Reaper improvement (core):** evict **least-recently-*queried*** blocks (not oldest-*written*), so the reaper doesn't drop the prefix you're about to read back. The primary fix for the crawl.
2. **Selective CPU-tier admission (speed tier):** only migrate the *hottest* evicted blocks to the 16 GiB CPU tier (0.3 s read-back); demote the rest to fs (12 s) / drop. Keeps the fast tier holding the active conversation's prefix.
3. **Queue on overflow (safety net):** for the 2-long case (which you'd serialize), defer (queue) instead of preempt, so the prefix isn't dropped. A safety net, not the main work.

Plus the **diagnostic** (evicted-but-not-retained rate) to confirm the root cause, and the **prerequisite metrics** (read-back latency, time-saved) to verify the fix.

**Scoping — make it work first, optimize the shape after.** The goal is to prove retention works (the evicted KV is read back, not recomputed). The final shape is a *later* optimization, and there is hardware headroom to tune it: the current **HDD L3 tier can be replaced** (other disks + controllers are available on this box), and **slightly more system RAM can be assigned for a larger L2 (CPU) tier**. Do not over-optimize the shape up front — implement the three aspects, confirm the recompute/miss fraction → ~baseline (3–5%) under a growing-conversation workload (retention holds), *then* pick the final disk/L2/reaper shape.

---

## 1. Background / why this work

### 1.1 The serving pattern (what actually runs)

> **SUPERSEDED — see R2.6.** The 1-long + 1-short premise holds only for this quiet boot. Live opencode traffic on the previous boot logged 151 preemptions across 136 requests at identical geometry.
- **1 long (opencode, ~100k → ~61 blocks) + 1 short (chat prompt, a few blocks) ≈ 66 blocks < 138 → co-fit, no preemption.**
- **2 short (a few + a few) ≈ 10–40 blocks < 138 → co-fit, no preemption.**
- **2 long would serialize** (not run in parallel) — the *only* preemption case, and it is **not** the real workload.
- So `max_num_seqs=2` already serves the real pattern. The 38 preemptions measured in the stress test were the **2-long case (an artifact, not the real workload).**

### 1.2 The real failure mode: growing conversations → the server crawls
As the opencode session and the chat conversation grow, their combined context converges on the 138-block pool limit. At that point the pool overflows and requests get evicted. See §0 for the retained-vs-dropped distinction. **The crawl is the evicted KV being dropped (not retained) → recomputed (59 s).** The fix is retention.

### 1.3 The read-back is a win (settled rate, not the stress-test artifact)

> **SUPERSEDED — see R2.1.** The 12 s is the artifact and the ~78 s is the settled figure, not the other way round. As built, read-back is currently *slower* than recompute.
- Cold prefill: **~1,699 tok/s** (160,063 tok / 94.2 s) → a 100k recompute ≈ **59 s**.
- Settled (cold, single-pass) fs read-back: **~225–240 MB/s** (8-thread barely beats 1-thread — the array's total read bandwidth is the bottleneck, not the thread count) → a 2.8 GB read-back ≈ **12 s**.
- The 77 s stress-test read-back was a **just-wrote / pending-txg / first-read-back artifact**, not the real rate.
- So a cached prefix is always cheaper to read back than to regenerate, for long prefixes. The aim is achievable.

### 1.4 The mechanism (code path)

> **SUPERSEDED IN PART — see R2.4 (Aspect 1).** The FIFO reaper described here was replaced by an age-based one on 2026-09-06.
- Preemption: `v1/core/sched/scheduler.py` — the allocation/preemption block (≈ 582-637) and `_preempt_request()` (≈ 1274). On `allocate_slots() is None` it preempts a running request.
- The waiting gate (≈ line 687): `if num_running >= self.max_num_running_reqs: break` — why `--max-num-seqs 1` already yields "queue, don't preempt" for the 2-long case.
- On preemption, `_preempt_request` sets `num_computed_tokens = 0` and `self.waiting.prepend_request(request)` (≈ line 1314); on reschedule, the KV is read back (offload) or recomputed (`kv_load_failure_policy=recompute`).
- **The reaper** (`<repo>/kvcache-reap.sh`): FIFO (oldest-*written*), 80%→65%, ~5 min cadence. It is the *only* eviction policy for the fs tier, and it is the **risk** (it can evict a hot prefix before it's re-queried → recompute).

### 1.5 The fix (retention)
Keep the evicted KV in the offload tiers so every overflow is a read-back (12 s / 0.3 s), not a recompute (59 s):
1. The fs tier must hold the total evicted KV (the 80% high-water = 409 GB must exceed it).
2. The reaper must evict least-recently-*queried* (not oldest-*written*) — so it doesn't drop the prefix you're about to read back.
3. The selective CPU admission keeps the hottest in the fast CPU tier (0.3 s).
4. The queue-on-overflow (safety net) stops the 2-long case from dropping the prefix.

---

## 2. The exact change

### 2.1 The flags
Follow the existing radiance env-var pattern (`scheduler.py:2924-2928`, e.g. `_RAD_DYNW = _rad_os.environ.get("RADIANCE_DYNAMIC_WIDTH", "0") == "1"`). All default to original behavior (zero behavior change when unset — this is what makes the patch reversible/safe):
```python
# Aspect 3 (safety net) — consumed by the scheduler:
_RAD_QUEUE_ON_OVERFLOW = _rad_os.environ.get("VLLM_QUEUE_ON_OVERFLOW", "0") == "1"
# Aspect 2 (speed tier) — consumed by the offloading connector:
_RAD_CPU_OFFLOAD_POLICY  = _rad_os.environ.get("VLLM_CPU_OFFLOAD_POLICY", "none")   # none|recency|refcount|cap
_RAD_CPU_OFFLOAD_DEMOTE = _rad_os.environ.get("VLLM_CPU_OFFLOAD_DEMOTE_TO", "fs") # fs|drop
_RAD_CPU_OFFLOAD_CAP_N  = int(_rad_os.environ.get("VLLM_CPU_OFFLOAD_CAP_N", "64"))
# Aspect 1 (core) — the reaper is a standalone script (kvcache-reap.sh); its policy is
# a config flag / env var read by the reaper, not the vLLM process:
#   VLLM_REAP_POLICY=lru|refcount|rrq   (rrq = least-recently-queried, the new default when set)
```
- Name them `VLLM_*` (or `RADIANCE_*` to match the other radiance vars — pick one scheme and be consistent across all aspects).

### 2.2 The reaper improvement (core — the primary fix for the crawl)

> **SUPERSEDED — see R2.4 (Aspect 1).** Demoted to step 4. The reaper is already age-based, and last-query time is not observable under a `noatime` mount.
**The problem:** `kvcache-reap.sh` evicts by FIFO (oldest-*written*), 80%→65%. A prefix written a while ago but re-queried recently (hot) is evicted first → forces a recompute. As conversations grow, the fs tier fills and the reaper starts dropping the very prefixes that are about to be read back.

**The fix:** change the eviction order from oldest-*written* (FIFO) to **least-recently-*queried*** (LRQ). A block is evicted only if it has been the least-recently-re-queried for the longest. This requires tracking, per block, its **last-query timestamp** (or a reference count):
- **Source of the last-query time:** the offloading connector's read-back events (when a block is read back from the fs tier, record the time) and the prefix-cache hit events. Persist per-block (or per-request) query-time alongside the block file (e.g., a sidecar metadata file, or an append to the block filename / a separate index).
- **Eviction order:** when the reaper fires (80%), sort candidate blocks by last-query time ascending (oldest query first) and delete the lowest until 65%. A block never re-queried since it was written is evicted before a recently re-queried one.
- **Reference-count variant (`refcount`):** evict blocks with the lowest reference count (shared prefixes / high re-query likelihood are kept).
- **The NUL-safe filename handling and stale-`.tmp` cleanup stay** (the existing reaper's safe-file handling is preserved; only the *order* changes).

**Where (code):** `<repo>/kvcache-reap.sh` (the standalone reaper script). Add the query-time tracking + the LRQ/refcount eviction order. The reaper already runs every ~5 min (`kvcache-reap.timer`); the change is the sort key, not the cadence.

**Why it's the primary fix:** it directly stops the reaper from dropping the prefix you're about to read back — the exact cause of the crawl (§1.2). Combined with sizing the fs tier to the total evicted KV (§6.2), it ensures the evicted KV is retained, so every overflow is a read-back (12 s), not a recompute (59 s).

### 2.3 Selective CPU-tier offload admission (speed tier)

> **SUPERSEDED — see R2.4 (Aspect 2).** The mechanism as described does not exist (fs is fed *from* CPU, so "demote to fs" collapses to "drop"). Rebuild it as an out-of-tree `CachePolicy` module — no patch required.
**The problem:** the CPU RAM tier is a **small (16 GiB), fast (10.6 GB/s read-back)** buffer. Under current behavior, **every** evicted block migrates to it. The working set is much larger than 16 GiB, so the CPU tier **thrashes** — blocks are written in and evicted out *before* they are re-queried, the hit rate is low, and the fast read-back is underutilized. The blocks actually re-queried (the hot ones) get pushed out by less-important blocks.

**The fix:** when a request's blocks are evicted (offloaded), **score the request and admit only a SELECTED subset to the CPU tier; demote the rest** to the fs tier (capacity) or drop them (recompute). This keeps the fast CPU tier holding the *hot* blocks (high hit rate → fast read-backs) and the fs tier holding the *cold* ones (capacity).

**Selection granularity:** per-request (a request's blocks are offloaded together as a unit; the policy ranks *requests*).

**The policy (`VLLM_CPU_OFFLOAD_POLICY`):**
| Value | Score / rule | Admit to CPU | Demote the rest |
|-------|-------------|--------------|-----------------|
| `none` (default) | — | **all** evicted requests (current, thrashing) | — |
| `recency` | last-access time | the most-recently-accessed requests (up to the cap) | to `VLLM_CPU_OFFLOAD_DEMOTE_TO` |
| `refcount` | # of other requests sharing the block (shared prefixes) | the high-reference requests (up to the cap) | to `VLLM_CPU_OFFLOAD_DEMOTE_TO` |
| `cap` | hard rate cap | at most `VLLM_CPU_OFFLOAD_CAP_N` requests/step (ordered by recency) | to `VLLM_CPU_OFFLOAD_DEMOTE_TO` |

- `VLLM_CPU_OFFLOAD_DEMOTE_TO` = `fs` (demote to the 512 GB fs tier — capacity) or `drop` (drop — recompute on re-query).
- **Default `fs` is correct** (the fs read-back, ~12 s, beats the 59 s recompute, so retaining blocks in the fs tier is a win; `drop` is only for when the fs tier is full).
- The cap (`VLLM_CPU_OFFLOAD_CAP_N`, default 64) bounds the per-step CPU influx for `recency`/`refcount`/`cap`.

**Where (code):** the offloading connector's eviction/offload path — the `TieringOffloadingSpec` (where an evicted block is written to the **CPU primary tier**). Insert the admission decision before the CPU write: score the request → if admitted (top of the policy ranking, within the cap), write to CPU; else write to the demotion target (fs) or drop.

### 2.4 Queue on overflow (safety net — for the 2-long case you'd serialize)

> **SUPERSEDED — see R2.4 (Aspect 3).** As written this deadlocks, inverts the waiting-phase admission gate, and its counter raises `AttributeError` on the first defer.
**File:** `v1/core/sched/scheduler.py`. **Location:** the allocation/preemption block in `_schedule()` (≈ lines 582-637).

**Current code (≈ 586-637):**
```python
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        ...  # budget bookkeeping (token_budget, encoder budget, req_index)
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break
```

**Patched code** — insert a single guard after the "can be scheduled" `break`, before the preemption:
```python
                    # PATCH: defer (queue) instead of preempting (safety net for 2-long).
                    if self.queue_on_overflow:
                        # Not scheduled this step; the request stays in its
                        # queue (running/w) and is retried next step. No KV
                        # is dropped -> no recompute, no readback.
                        self._iteration_stats.num_deferred_reqs += 1
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        ...  # UNCHANGED (original preemption)
```

**Why it's a safety net (not the main work):** the real serving pattern (1 long + 1 short, or 2 short) **co-fits** (no preemption). Only the 2-long case (which you'd serialize) preempts. This guard makes that case queue instead of dropping the prefix. For the real pattern it is a no-op (no overflow → the guard never fires).

### 2.5 The counters
Follow the exact `num_preempted_reqs` pattern:
- **Field:** add `self.num_deferred_reqs = 0` to the `IterationStats` class (`v1/metrics/stats.py:349`, next to `self.num_preempted_reqs = 0` at line 356). (Increment site: the `+= 1` at stats.py:449 shows where `num_preempted_reqs` is bumped; the scheduler increments `num_deferred_reqs` on the defer path.)
- **Log-line:** in `v1/metrics/loggers.py`, `self.num_preemptions += iteration_stats.num_preempted_reqs` (line 150) and the log-args (lines 283-285) — add the deferred analogue.
- **Prometheus counter:** mirror `counter_num_preempted_reqs` (loggers.py:661-667, created via `create_metric_per_engine`) — add `counter_scheduling_deferred` named `vllm:scheduling_deferred`, and the per-engine `.inc()` at loggers.py:1191-1192 analogue.
- **Aspect-2 counters** (incremented in the offloading connector, not the scheduler): add `cpu_offload_admitted_total` and `cpu_offload_demoted_total{target}` as prometheus counters (same `create_metric_per_engine` pattern); increment at the admission decision in §2.3.

### 2.6 The diagnostic (primary) + prerequisite metrics

> **SUPERSEDED — see R2.5.** The cumulative miss fraction measures workload novelty, not retention; it reads 18.4% on a healthy live server. Keep it bench-only.
**The primary diagnostic — the recompute/miss fraction (existing metrics, NO new counter needed):**
```
recompute_miss_frac = (prefix_queries − prefix_hits − ext_hits) / prefix_queries
  prefix_queries = vllm:prefix_cache_queries_total
  prefix_hits    = vllm:prefix_cache_hits_total
  ext_hits       = vllm:external_prefix_cache_hits_total
```
- A token hits at most one tier, so `(Q − H − E) ≥ 0` = the tokens that missed **both** the GPU prefix cache and the offload tier → **recomputed from scratch** (the failure).
- **Baseline (retained, working): ~3–5%** (the new-suffix tokens + the last partial block). **Failure (dropped, plummets): jumps toward 100%.**
- **Validated:** the benchmark (`~/audit/stress/hitrate-bench.py`) measured **4.88%** on an evicted-then-re-queried prefix (95.12% offload hit) — i.e. RETAINED. If it jumps to ~100%, the evicted prefix was DROPPED (the hit rate plummets). This is the root-cause signal for the crawl (§1.2) and the primary "is it working / is it plummets" probe.
- Because it is computed from the three **existing** metrics above, it is a **drop-in diagnostic** (no patch required to start measuring it).

**The optional refinement — `evicted_not_retained_total` counter** (attributes the misses to the eviction path specifically): count preemptions/evictions where the evicted KV was **not** followed by a `CPU_to_GPU`/`fs` read-back (i.e. dropped, not retained). A high rate isolates the eviction→drop path. (The recompute/miss fraction above is the primary signal; this counter refines it to the eviction path.)

**Prerequisite metrics (verify the fix's speed benefit):**
- `kv_offload_readback_latency_seconds{tier="cpu|fs"}` **histogram** (read-back rate, per tier).
- `kv_offload_time_saved_seconds_total` **counter** (Σ of (recompute_time − readback_time) when readback < recompute; the "is the offload actually helping" signal).

---

## 3. Interaction with existing code (what NOT to break)

> **SUPERSEDED IN PART — see R2.4 (Aspect 3).** "No request starves as long as `max_num_seqs` is bounded" does not hold.
- **`preempted_reqs`** (used at ≈ line 684): with the safety-net flag set, no preemptions occur on the overflow path → `preempted_reqs` stays empty → Phase-2 queues waiting requests via the line-687 gate. No change needed.
- **`reset_preempted_req_ids`** (≈ lines 628, 1315): stays empty (no preemptions on the overflow path).
- **The line-687 gate** (`if num_running >= self.max_num_running_reqs: break`): unchanged; still queues waiting requests when at max. This is what makes `max_num_seqs=1` already work for the 2-long case.
- **`requires_kv_delivery` / `drop_stale_output`:** the deferred request does **not** go through `_preempt_request`, so no stale-output drop occurs. Verify this is the desired semantics (it is: a deferred request keeps its in-flight output; only a *preempted* one drops it).
- **The reaper's NUL-safe filenames + stale-`.tmp` cleanup** (the existing `kvcache-reap.sh` behavior): **preserved** — only the *sort key* changes (FIFO → LRQ/refcount).
- **Retry/no-starvation** (safety net): deferred requests are retried every scheduling step; running requests finish and free the pool, so no request starves as long as `max_num_seqs` is bounded.

---

## 4. Risks / edge cases

> **SUPERSEDED IN PART — see R2.4 (Aspect 3).** The Aspect 3 risk is a hang, not one step of idle GPU.
1. **Reaper LRQ metadata overhead (§2.2):** tracking per-block last-query time adds a sidecar/index write per read-back. Keep it cheap (a small sidecar file or an in-memory index flushed on reap). Verify it doesn't slow the read-back path.
2. **Reaper LRQ cold-start (§2.2):** a block never re-queried has no query time — treat "never queried" as the oldest (evicted first), which is correct (a never-re-queried block is the least valuable).
3. **GPU idle time (safety net, §2.4):** queue-on-overflow lets the GPU idle. For the 2-long case (which you'd serialize) this is the desired trade; for the real pattern it's a no-op (no overflow).
4. **CPU-tier admission thrash (§2.3):** the policy must be stable across steps (a block admitted to CPU this step shouldn't be demoted next step unless a hotter block arrives). Verify the admission is monotonic within a request's lifetime.
5. **fs-tier read-back under high concurrency:** the ~240 MB/s is the array's *aggregate* read ceiling. A single large read-back (~12 s for 2.8 GB) is fine; several overlapping large read-backs would contend for the same bandwidth and each slow down. At your serving pattern (1 long + 1 short co-fit) this is a non-issue.
6. **Offload-tier sizing:** the fs high-water (409 GB) must exceed the total evicted KV, or the reaper triggers and drops prefixes. Size the fs tier to the working set (§6.2).
7. **Spec-decode (EAGLE/MTP):** the deferred request's `spec_token_ids` are untouched (only `_preempt_request` clears them — which the safety net skips). Verify no stale spec tokens on resume.

---

## 5. Integration with the radiance patch set

> **SUPERSEDED IN PART — see R2.4 (§5).** `patch_dynwidth.py` and `patch_offload_mixed_hit.py` both already patch `scheduler.py`.
- **Separate patch files:** `<repo>/ggz14-mxfp4/patches/patch_offload_cpu_admission` (Aspect 2, the connector) + `patch_scheduler_queue_on_overflow` (Aspect 3, `scheduler.py`). **Aspect 1 (the reaper) is a change to `<repo>/kvcache-reap.sh`** (the standalone script), not a vLLM patch — add the LRQ/refcount sort key + the query-time tracking.
- **Gated:** `VLLM_QUEUE_ON_OVERFLOW`, `VLLM_CPU_OFFLOAD_POLICY`, `VLLM_CPU_OFFLOAD_DEMOTE_TO`, `VLLM_CPU_OFFLOAD_CAP_N`, `VLLM_REAP_POLICY` (all default off/original → zero behavior change when unset).
- **Apply order:** add the vLLM patches to the boot patch list in `patch_all.sh` / the launcher; the reaper change is a direct edit to `kvcache-reap.sh` (no patch file needed, or a patch file for consistency). Verify the vLLM patches apply cleanly (no existing patch touches `scheduler.py` — confirmed).
- **Idempotent/reversible:** each flag off = original behavior; the patch set can be removed without a rebuild (just unset the env vars).

---

## 6. Verification / test plan
### 6.1 Smoke (each aspect on vs off)
- **All off:** `num_preemptions` unchanged, `scheduling_deferred` = 0, `cpu_offload_admitted_total` = all evicted, the recompute/miss fraction at baseline (3–5%), `evicted_not_retained_total` unchanged → confirms zero behavior change (the safety invariant).
- **Reaper LRQ on:** a hot (recently re-queried) block is **kept**; a cold (never re-queried) block is **evicted first** → the order is firing.
- **Aspect 2, policy `recency`/`cap`:** `cpu_offload_admitted_total` bounded by the cap, `cpu_offload_demoted_total{target}` > 0 → the selection is firing.
- **Aspect 3 (safety net) + 2-concurrent large:** `num_preemptions` → ~0, `scheduling_deferred` increments, the 2nd waits, the 1st runs at full speed.

### 6.2 A/B: retention (the primary test — reaper + fs sizing)

> **SUPERSEDED — see R2.7.** The stated criterion is not reachable by sizing at 512 GB. State the goal as a residency window in hours.
The test that confirms the crawl is fixed. Run a **growing-conversation** workload (long opencode-style sessions that grow over many turns) with:
- **Control:** current reaper (FIFO, oldest-written).
- **Treatment:** LRQ reaper + fs tier sized to the total evicted KV.
Compare:
| Metric | Expect with retention |
|--------|----------------------|
| recompute/miss fraction (the miss rate, the primary diagnostic) | **→ ~baseline (3–5%)** (the evicted KV is retained) |
| offload (ext) hit rate on the re-query | high (the evicted KV is read back) |
| `kv_offload_time_saved_seconds_total` | **rises** (recompute cost removed) |
| end-to-end re-query latency (the growing conversation) | **not regressed** (no 59 s recompute stalls) |
| server throughput under growing load | **does not crawl** |

**Decision criterion:** retention wins if the recompute/miss fraction → ~baseline (3–5%) and the growing-conversation workload no longer crawls (no 59 s recompute stalls).

### 6.3 A/B: selective CPU admission (speed tier)
Run a re-query-heavy workload with `VLLM_CPU_OFFLOAD_POLICY=none` vs `recency` (and `cap`, `refcount` as variants); compare:
| Metric | Expect with policy |
|--------|-------------------|
| `cpu_offload_admitted_total` | bounded by the cap (vs. unbounded under `none`) |
| `cpu_offload_demoted_total{target}` | > 0 (the cold blocks are demoted) |
| CPU-tier hit rate | **improves** (hot blocks retained) |
| `kv_offload_readback_latency_seconds{tier="cpu"}` | **improves** (hot reads dominate) |

**Decision criterion:** the policy wins if the CPU-tier hit rate / read-back latency improves (hot blocks retained) without regressing re-query latency.

### 6.4 A/B: queue on overflow (safety net)
Run the **2-long** case (the one you'd serialize) with and without `VLLM_QUEUE_ON_OVERFLOW=1`; compare `num_preemptions_total` (expect → ~0 with the flag), `scheduling_deferred`, P95 latency. (A safety-net test, lower priority than §6.2.)

### 6.5 The diagnostic (confirm the root cause before patching)
Before implementing, confirm the crawl's root cause with the **recompute/miss fraction** (the primary diagnostic, §2.6 — computed from the three existing metrics, no new counter needed):
```
recompute_miss_frac = (prefix_queries − prefix_hits − ext_hits) / prefix_queries
```
- **The A/B probe:** run `~/audit/stress/hitrate-bench.py` — it A/Bs a control (no-evict) vs preemption (evict) re-query of the same ~100k prefix and reports the recompute/miss fraction. Baseline (retained) ≈ 3–5%; failure (dropped) ≈ toward 100%.
- **The real-workload signal:** on a growing-conversation workload, watch the **cumulative** recompute/miss fraction over time (re-run `hitrate-bench.py` periodically, or track the running `vllm:prefix_cache_*` / `vllm:external_prefix_cache_hits` counters). When the offload tiers fill and the FIFO reaper starts evicting hot prefixes, the recompute/miss fraction rises from ~3–5% toward the miss rate — that is the plummet.
- **Decision criterion:** a cumulative recompute/miss fraction well above the ~3–5% baseline on a growing-conversation workload confirms the crawl's root cause (§1.2) and justifies the reaper + fs-sizing work. After the patch, the same workload should hold at ~baseline (the evicted KV is retained, not dropped).

---

## 7. The no-patch alternative (reference)

> **SUPERSEDED IN PART — see R2.8.** The genuine no-patch option is now Aspect 2 as an out-of-tree `CachePolicy` module.
- **`--max-num-seqs 1`** already yields "queue, don't preempt" for the 2-long case via the line-687 gate (a 2nd request can't be admitted when 1 is running at max, so it waits; no preemption). Zero code risk. But for the real pattern (1 long + 1 short co-fit), `max_num_seqs=2` already works (no preemption).
- **The reaper is the primary work and has no flag shortcut** — the LRQ eviction order + fs sizing must be implemented (there's no stock vLLM knob for "evict least-recently-queried"). This is the change that fixes the crawl.

---

## 8. Out of scope / open questions
- **Final-shape optimization (make it work first, then tune):** the *shape* (which disk/controller for L3, L2 size, reaper cadence, fs high-water) is a later concern. Hardware headroom: the **current HDD L3 tier can be replaced** (other disks + controllers available on this box), and **slightly more system RAM can be assigned for a larger L2 (CPU) tier**. Sequence: (1) implement the three aspects, (2) confirm the recompute/miss fraction → ~baseline (3–5%, retention works) via `hitrate-bench.py` on a growing-conversation workload, (3) then choose the final shape.
- **Cost-based recompute patch** ("if readback > recompute, skip the readback and recompute") — a separate, harder change (needs a per-block readback-time estimate); the settled read-back (12 s) already beats recompute (59 s), so this is moot for the normal case.
- **fs-tier read-back speed fix** (`n_read_threads` up / NVMe) — orthogonal; the ~240 MB/s is the array's aggregate ceiling, so more threads won't help; a faster array (NVMe) would.
- **Reaper LRQ metadata format:** sidecar file vs in-memory index vs append to block filename — open (keep it cheap, §4.1).
- **Reaper `refcount` vs `rrq` default:** which best matches the workload's re-query pattern (shared-prefix-heavy → `refcount`; recency-heavy → `rrq`). Open.
- **fs-tier sizing:** the exact high-water (409 GB) vs the real total evicted KV — tune to the workload's working set. Open.
- **Aspect 1 + 3 interaction:** with the safety net on (no preemption), the CPU/fs-tier population comes purely from LRU evictions of finished blocks; confirm the Aspect-2 admission still applies on that path (it does — the decision is at the CPU-write site, independent of *why* the block was evicted). Verify in the combined A/B.

---

## 9. Files / locations reference
| Item | Location |
|------|----------|
| Reaper (core change) | `<repo>/kvcache-reap.sh` — add LRQ/refcount sort key + query-time tracking (the `kvcache-reap.timer` cadence stays) |
| Aspect-2 change (CPU admission) | the offloading connector's CPU-write/eviction site (the `TieringOffloadingSpec` primary-tier write path) |
| Aspect-3 change (safety net) | `v1/core/sched/scheduler.py` — allocation/preemption block ≈ 582-637; `_preempt_request` ≈ 1274; gate ≈ 687; env-var pattern ≈ 2924-2928 |
| Per-step stats (field) | `v1/metrics/stats.py` — `IterationStats` class line 349; `num_preempted_reqs` 356 / 449 |
| Metric emission | `v1/metrics/loggers.py` — counter 661-667; `.inc()` 1191-1192; log-line 150, 283-285 |
| Readback metric + diagnostic (prereq) | `TieringOffloadingSpec` load path (the offloading connector) |
| **Primary diagnostic (recompute/miss fraction)** | computed from the 3 existing metrics (`vllm:prefix_cache_queries_total` − `vllm:prefix_cache_hits_total` − `vllm:external_prefix_cache_hits_total`) / `prefix_queries` — **no new counter needed** |
| **A/B probe (the hit-rate/miss benchmark)** | `~/audit/stress/hitrate-bench.py` (control vs preemption re-query; reports the recompute/miss fraction; baseline ≈ 3–5%, failure ≈ toward 100%). Report: `~/audit/stress/hitrate/REPORT.md` |
| New patch file(s) | `<repo>/ggz14-mxfp4/patches/patch_offload_cpu_admission` (Aspect 2) + `patch_scheduler_queue_on_overflow` (Aspect 3) — the reaper is a direct edit to `kvcache-reap.sh` |
| Launcher (flag wiring) | `<repo>/llama-swap-ggz14-27b.sh` (the serve env, alongside `RADIANCE_*`): `VLLM_QUEUE_ON_OVERFLOW`, `VLLM_CPU_OFFLOAD_POLICY`, `VLLM_CPU_OFFLOAD_DEMOTE_TO`, `VLLM_CPU_OFFLOAD_CAP_N`, `VLLM_REAP_POLICY` |
| Existing patch plans (style ref) | `<repo>/ggz14-mxfp4/patches/patch_gdn_merge_plan.md`, `patch_radiance_w4a8_plan.md` |

**Acceptance:** all flags default off ⇒ byte-identical to stock. **Reaper LRQ on** ⇒ a hot (recently re-queried) block is kept, a cold (never re-queried) block is evicted first; the **recompute/miss fraction → ~baseline (3–5%)** under a growing-conversation workload (retention holds); the workload no longer crawls (no 59 s recompute stalls); `kv_offload_time_saved` rises. **Aspect 2 on** ⇒ `cpu_offload_admitted_total` bounded by the cap, `cpu_offload_demoted_total{target}` > 0, CPU-tier hit rate + read-back latency improve. **Aspect 3 on** ⇒ `num_preemptions` → ~0 in the 2-long case, `scheduling_deferred` > 0, no starvation. All three aspects are independent and can be shipped/toggled separately; the **reaper + fs sizing is the primary deliverable**.
