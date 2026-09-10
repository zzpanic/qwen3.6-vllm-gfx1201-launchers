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
