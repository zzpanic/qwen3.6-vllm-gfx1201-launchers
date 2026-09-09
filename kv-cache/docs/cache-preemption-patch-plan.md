# Patch plan — "retain, don't recompute" (KV retention for growing conversations)

**Target build:** `qwen3.8-27b-vllm` (mxfp4), single R9700 32 GB, **vLLM 0.27.1** custom (radiance patch set, `<repo>/ggz14-mxfp4/`).
**Status:** PLAN for a second coding agent. **Superseded in part by Revision 3 below (2026-09-07) — read that first, then Revision 2. Where they disagree, the higher revision number wins.** Three flag-gated, reversible changes; the primary work is **retention**.

---

# Revision 3 — 2026-09-07 (first deterministic measurement; rev 2's cost model is withdrawn)

**This section supersedes Revision 2 where they disagree, and Revision 2 supersedes rev 1. Read in that order.**

Rev 2 was written from cumulative Prometheus counters on a server that had been up for hours.
Revision 3 is written from `tierbench.py`, which controls what is in each tier before it measures.
The result overturns the central number rev 2 was built on.

**Start at R3.14.2b — Phase A is ANSWERED: 99.75% of production lookups defer, and the trigger
is HIT_PENDING (a store still in flight), not RETRY. That retires A2, A3 and the
bench-vs-production puzzle, and it demotes R3.13. Then R3.14.3, which is the fix Phase A named:
it is BUILT and waiting on a restart.** Then read R3.14 for the plan both phases sit in, and
after that R3.13 (the build sheet, now demoted but not withdrawn), R3.12, R3.11, R3.10 — newest
first; R3.10 corrects R3.9.1, R3.12 reframes R3.10.4, and R3.14.1 corrects the build ordering
R3.13.5 proposed. Short version: the disk tier *can* serve a hit (85,696 tokens, run `9e4d2b`),
that hit is **bit-identical to a cold recompute** (run `9e6993` — 128/128 tokens and every
logprob exact), and the work left is **in the lookup path** — production throws away hits it
has already found because a later chunk's store is still in flight (R3.14.2b). Cutting the
once-per-chunk Mamba snapshot (R3.12.3, R3.13) is still worth doing, because **it is what makes
the RAM tier able to serve a hit at all**, but it is now second-order. **R3.12.2 and R3.12.6
correct two claims made earlier in this document.** R3.12.5 named the one cheap experiment that
might have made that a config change; **R3.12.5a ran it, and the answer is no —
`blocks_per_chunk` bundles blocks into bigger files, it does not deduplicate Mamba state, so
the store-policy patch is required.** R3.1–R3.9 are kept because the retractions inside them
are the map of what not to re-try.

## R3.1 — The finding: the offload tier served zero bytes

Baseline run, salt `9e3561`, 2026-09-07 13:54, prefix 90,029 tokens:

| phase | expected | served by | valid | wall s | TTFT s | ext hit tok | load GB |
|---|---|---|---|---|---|---|---|
| cold | RECOMPUTE | RECOMPUTE | yes | 43.41 | 43.12 | 0 | 0.000 |
| gpu | GPU | GPU | yes | 2.27 | 1.97 | 87,344 | 0.000 |
| cpu | OFFLOAD | **RECOMPUTE** | **NO** | 43.51 | 43.21 | 0 | 0.000 |
| fs | OFFLOAD | **RECOMPUTE** | **NO** | 42.95 | 42.66 | 0 | 0.000 |

Concurrent phase (3 clients, 117k-token prompts): 351,008 prefix-cache queries, **0 hits**,
0 external hits, 42.0 GB stored, **0 bytes loaded**, 61 allocation failures.

`kv_offload_load_bytes_total` did not move by a single byte across the entire 11-minute run.
Both re-read phases came back within 1.3% of the cold recompute, because they *were* cold recomputes.

**Rev 2's R2.1 ("read-back ~78 s vs 53 s recompute, so read-back is currently slower than
recompute") is WITHDRAWN.** No read-back was measured. The 78 s was a lookup stall followed by
a full recompute. There is no evidence in this system about how fast the fs tier reads, because
the fs tier has never served a read.

## R3.2 — Root cause: the fs cascade pins the CPU tier shut

The store path is `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1108`
(line 1154 in the patched runtime):

```python
store_output = self.manager.prepare_store(new_offload_keys, req_status.req_context)
if store_output is None:
    self._connector_stats.increase_counter(_ConnectorMetricName.ALLOCATION_FAILURE)
    logger.warning("Request %s: cannot store chunks", req_id)
    continue
```

`CPUOffloadingManager.prepare_store` (`vllm/v1/kv_offload/cpu/manager.py:186`) returns `None` here:

```python
num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()
if num_blocks_to_evict > 0:
    if num_blocks_to_evict > self._num_evictable_cache_blocks:
        # Eviction will fail.
        return None
```

`_num_evictable_cache_blocks` counts only blocks with `ref_cnt == 0`. A block is pinned by
`prepare_load` (`cpu/manager.py:130`) and released by `complete_load` (`cpu/manager.py:154`).
**The CPU→fs cascade is a load from the CPU tier's point of view** — it reads CPU blocks in order
to write them to disk — so every block queued for the fs tier is pinned until its disk write lands.

Measured on this box during the run:

- GPU→CPU: 483.2 GB / 46.9 s = **10.31 GB/s**
- CPU→fs: 5,276 files x 27,000,832 B over 30 min = **79.1 MB/s**
- ratio: **130 : 1**

The CPU tier fills 130x faster than the cascade drains it. Pinned blocks accumulate, evictable
blocks go to zero, `prepare_store` returns `None`, and nothing is retained.

> ### ⚠ CORRECTION (R3.9) — the two rate figures above are wrong, the mechanism is not
>
> **79.1 MB/s is not a rate the cascade ever runs at.** It is a 30-minute average of a
> process that alternates between ~202 MB/s and *exactly zero*. A minute-resolution mtime
> histogram of the block tree (R3.9) shows the cascade stops dead the moment the engine
> takes load and resumes the minute it goes idle — 21 consecutive minutes of zero blocks
> written across the middle of the run. Averaging "full speed" with "completely stopped"
> produces a number that describes neither.
>
> So **"130 : 1" is not a bandwidth mismatch and there is no fill-vs-drain ratio to
> improve.** The tier is *blocked*, not *slow*. Every conclusion below that reasons from
> the ratio — "a tenfold faster drain still leaves 13:1", "spindles move the cliff later" —
> is arguing about the wrong quantity. The conclusions those arguments reached (don't buy
> disks; break the pin coupling) happen to survive, but for a different and stronger reason:
> a faster disk cannot help something that is not running at all.
>
> What *is* still correct in R3.2 is the pinning mechanism itself — `prepare_load` pins,
> the cascade holds the pin until the write lands, evictable goes to zero, `prepare_store`
> returns `None`. That chain is confirmed by the gauges in R3.8 and is unaffected.

## R3.3 — The retry livelock

On `None`, the code `continue`s **without** calling `advance_stored_idx`. The request therefore
re-offers the same chunks on the next scheduler step, and the next, indefinitely.

Observed: **342 `cannot store chunks` warnings across 19 distinct requests**, 03:56:36–04:04:48 UTC,
approximately one per request per second for the whole run. This exactly matches the
`kv_offload_allocation_failure_total` delta (194 → 536 = +342).

This is the "prompt rebuilding stall pattern under heavy load" that started this investigation.
It is not slow read-back. It is a store path that livelocks under back-pressure, so every
subsequent request misses and recomputes from scratch.

## R3.4 — `cpu_cache_usage_perc` does not mean what its name suggests

`cpu/manager.py:297`:

```python
num_used = self._num_allocated_blocks - len(self._free_list) - self._num_evictable_cache_blocks
usage = num_used / self._num_blocks
```

Evictable cached blocks are **subtracted**. The gauge is *"fraction of the tier pinned by in-flight
transfers"*, not *"fraction of the tier holding data"* — which is what its own help text says:
"Fraction of CPU KV-cache space currently pinned by active transfers".

Consequences:
- At idle it reads **0.0** while 228 GB of blocks sit on the fs tier. It is not a residency gauge.
- Under the concurrent phase it reached **0.956** — i.e. 96% of the tier pinned, ~28 of 636 blocks
  admissible. That is the saturation signal.
- Rev 2 read 0.8255 as "82% full". That reading was wrong; it meant "82% pinned".

`write_usage_perc` stayed at **0.0** throughout, so none of the pinning was GPU→CPU stores in
flight. `read_usage_perc` carried the entire 0.871–0.956. It is all fs-cascade pinning.

## R3.5 — What the async lookup delay actually measures

Concurrent phase: `tiering_lookup_async_delay` +181.69 s over **3** observations = **60.6 s each**,
against a measured TTFT of **60.95 s** on the same requests.

The request blocks for ~61 s on a tier lookup and receives **zero bytes**. The stall is the lookup
itself, not a transfer. Rev 2's R2.9.2(d) — widen the `LOOKUP_ASYNC_DELAY` buckets past their 10 s
ceiling — is still required, and is now the single highest-value instrumentation change, because
this histogram is the one that already sees the stall and cannot express it.

> **DONE — see R3.8.** With the ceiling raised to 600 s the distribution is bimodal with an
> empty 5–60 s band: lookups resolve in seconds or stall for over a minute, with nothing in
> between. The 60.6 s average above was two catastrophic observations, not a uniform delay.

## R3.6 — Revised build order

Rev 2's order (instrument → CachePolicy → disks → reaper → Aspect 3) assumed the problem was
retention policy. It is not: nothing reaches the point where policy applies.

1. ~~**Widen `LOOKUP_ASYNC_DELAY` buckets**~~ (rev 2 R2.9.2(d)). **DONE 2026-09-07** — `patch_kv_offload_instrumentation.py` hunks 1–2. Result in R3.8.
2. ~~**Add an evictable-blocks gauge and a cascade-queue-depth gauge.**~~ **DONE 2026-09-07** — hunks 3–6, giving `cpu_cache_evictable_perc`, `cpu_cache_free_perc` and `fs_inflight_jobs`. Result in R3.8. `_num_evictable_cache_blocks`
   is the variable that decides admission and is currently unobservable. Without it, every
   conclusion about the store path is inference.
3. **Fix the retry livelock** (R3.3). **UNBLOCKED by R3.8.** At minimum, back off a request that has failed admission
   rather than re-offering the same chunks every step.
4. **Break the pin coupling** (R3.2) — the structural fix. **UNBLOCKED by R3.8; 4a is now the highest-value single change.** Options, in the order I would try them:
   a. Bound the cascade queue, so a full queue stops accepting new CPU stores instead of pinning
      the tier. Turns a livelock into clean back-pressure.
   b. Copy the block out before the disk write, so the fs cascade does not hold a CPU pin at all.
      Costs one memcpy per block; removes the coupling entirely.
   c. Make the cascade selective — do not send everything to disk, only what a policy says is
      worth keeping. This is where rev 2's `CachePolicy` work belongs, one stage later than
      rev 2 placed it.
5. ~~**Then, and only then, the disks.**~~ **CLOSED by R3.9 — do not do this.** `fio` on
   `/kvcache` was run and the disk is not the constraint: it sustains **470 MB/s** on a
   single-threaded O_DIRECT sequential write and **223.8 MB/s** on an exact replica of the
   cascade's own write shape, which is what the cascade actually achieves *when it runs*.
   The cascade's problem is that it does not run under load at all, and no disk fixes that.
   Spindles are off the table until (4) is done and re-measured.
6. Reaper, Aspect 3: unchanged from rev 2, still last.

## R3.7 — What is still unmeasured

Do not let R3.1's table be read as "the fs tier is slow". It has never been observed serving a
read. Its read latency, its promotion path, and whether promotion works at all are **open**.
The first run in which `kv_offload_load_bytes_total` moves during a `cpu` or `fs` phase is the
first real datum about tier read performance. Everything before that is about the store path.

> **Still true after run `9e3d5e` (see R3.8).** Two full runs, zero bytes read back.

## R3.8 — Post-instrumentation confirmation (run `9e3d5e`, GPU window 2026-09-07)

R3.1–R3.5 were inferred from counters that were never designed to answer the question. This
section is the same experiment re-run with R3.6 tasks 1 and 2 applied
(`patch_kv_offload_instrumentation.py`), so the mechanism is now **observed** rather than
reconstructed. Instrumentation only — no control flow was touched — so the phase results are
expected to be identical to the baseline, and they are.

Comparability: identical capacities to baseline `9e3561` (GPU 228,737 tokens, CPU primary
636 blocks), same prefix (90,027 tokens), same script.

### What did not change (as designed)

| phase | baseline `9e3561` TTFT | post-patch `9e3d5e` TTFT | served by |
|---|---|---|---|
| cold | 43.12 s | 44.68 s | RECOMPUTE |
| gpu | 1.97 s | 1.89 s | GPU |
| cpu | 43.21 s | 42.50 s | RECOMPUTE — **still INVALID** |
| fs | 42.66 s | 43.15 s | RECOMPUTE — **still INVALID** |

`kv_offload_size_sum{transfer_type="CPU_to_GPU"}` finished the run at **0.0 bytes, 0
transfers**. That is a second independent confirmation of R3.1: across two full runs and
~22 minutes of load, the offload tier has still never served a single byte back.

The store side is healthy and is not the bottleneck: 503 stores, 135.6 GB, 12.03 s of
transfer time — **11.3 GB/s** GPU→CPU.

### The mechanism, now measured directly

Sampled every 2 s (`baselines/gauges-9e3d5e.csv`, columns
`t,evictable,free,pinned,fs_inflight,alloc_fail,load_bytes,store_bytes`):

* `free` is **0.000 for the entire run.** The CPU tier is fully allocated from the first
  store onward, so admission depends *entirely* on the evictable count. This is why R3.2's
  reasoning had to go through evictable blocks rather than free ones.
* `evictable` falls **0.453 → 0.019** while `fs_inflight` climbs **39 → 70 jobs**. The two
  move together, which is the pin coupling of R3.2 made visible: every block queued for the
  fs cascade is held by a `prepare_load` ref and is therefore not evictable.
* Allocation failures begin at **t = 63 s**, the exact sample at which `evictable` reaches
  0.019, and climb monotonically to 133 by t = 209 s and 443 by the end of the run.

That is the full causal chain — cascade backlog → evictable collapse → `prepare_store`
returns `None` → livelock — with every link on a timestamp.

### It is a livelock, not a deadlock

After load stopped, the tier recovered on its own: `evictable` returned to **1.0**,
`fs_inflight` to **0**, `pinned` to **0**. Nothing is permanently wedged. The tier drains
perfectly well the moment you stop asking it to store. This matters for the fix: the target
is throughput coupling under sustained load, not a lost-wakeup bug.

### The stall is a cliff, not a slope

The R3.5 histogram topped out at 10 s, so the real shape was invisible. With the buckets
widened to 600 s, 36 observations decompose as:

| bucket | tiering-level | connector-level |
|---|---|---|
| ≤ 0.01 s | 6 | 6 |
| 0.1 – 1.0 s | 9 | 9 |
| 1 – 5 s | 19 | 20 |
| **5 – 60 s** | **0** | **0** |
| 60 – 120 s | 1 | 1 |
| 120 – 300 s | 1 | 0 |
| sum | 225.86 s | 106.86 s |

**The 5–60 s band is empty.** There is no gradual queueing regime: a lookup either resolves
in seconds or it waits behind the fs drain for over a minute. This is the signature of the
"prompt rebuilding stall pattern under heavy load" that started this investigation — it is
bimodal because admission is gated by a binary condition (is there an evictable block?), not
by a queue that lengthens smoothly.

Note the tiering level carries one observation the connector level does not: a lookup of
**120–300 s**, i.e. the tier stalled substantially longer than the connector recorded. The
two sums differ by 119 s for the same 36 lookups.

### Consequence for R3.6 task 5 (the spinning disks) — read this before buying anything

> **SUPERSEDED by R3.9.** The reasoning below is built on the 130:1 ratio, which R3.9 shows
> is not a real quantity. The *conclusion* — don't buy disks — is correct and is now
> established directly by `fio`. Read R3.9 instead; this is kept only to show the path.

~~The fill:drain ratio is **130:1**. Even a tenfold improvement in fs write bandwidth leaves
13:1, which still collapses evictable under sustained load; it would move the cliff later,
not remove it.~~ **More spindles cannot fix this on their own — task 4 (breaking the pin
coupling) is mandatory, and task 5 is an amplifier for it, not a substitute.**

~~Before any disk work, one cheap question must be answered first: **is 79 MB/s the disk or the
cascade?**~~ **Answered in R3.9: it is neither. It is the cascade not running.**

### Incidental: fs-tier residue

`/kvcache` holds **330 GB** against **135.6 GB** stored this boot, so roughly 195 GB is
carry-over from earlier boots, at 65% of a 512 GB filesystem. Not urgent and not on the
critical path, but it is the concrete case for the reaper (Aspect 1) once the store path is
fixed.

### Build-order status after this run

* Task 1 (widen `LOOKUP_ASYNC_DELAY` buckets) — **done**, and it paid for itself immediately.
* Task 2 (evictable + cascade-depth gauges) — **done**; produced the causal chain above.
* Task 3 (retry livelock) and task 4 (pin coupling) — **unblocked**. The measurement they were
  waiting on is in. Task 4a (bound the cascade queue) is now the highest-value single change,
  because `free` = 0.000 throughout means an unbounded queue can always consume the whole tier.
* Task 5 (disks) — gated on the `fio` question above.
* Task 6 (reaper / Aspect 3) — unchanged, last.

Artefacts: `baselines/REPORT-9e3d5e.md`, `baselines/RESULTS-9e3d5e.json`,
`baselines/gauges-9e3d5e.csv`, `baselines/metrics-final-9e3d5e.txt`.

---

## R3.9 — The disk is exonerated, the ratio was an artefact, and the geometry was wrong

**This section corrects R3.2 and R3.8 and supersedes R3.6 task 5.** Three separate errors
are fixed here. Two of them were mine and both pointed the work at hardware that is fine.

### R3.9.1 ~~The cascade does not run under load — it is blocked, not slow~~ — **WRONG, corrected in R3.10.2**

The decisive measurement is a minute-resolution histogram of block-file mtimes under
`/kvcache/blocks/..._r0`, taken across the two baseline runs:

| wall clock | blocks written per minute | what the engine was doing |
|---|---|---|
| 13:54 – 14:06 | ~400/min | run `9e3561` starting, light load |
| 14:07 – 14:27 | **0** — twenty-one consecutive minutes | under load |
| 14:28 – 14:40 | ~380/min | resumes the minute load ends |

Two independent instances (one mid-run at 14:06→14:11, one covering the entire post-patch
run at 14:16→14:28). Blocks stop reaching disk the moment requests arrive and resume the
minute the engine goes idle. When it runs unobstructed the cascade drains at **202.1 MB/s**.

This is what makes the "79.1 MB/s" of R3.2 meaningless: it is the average of 202 MB/s and
zero. **There is no fill-vs-drain ratio.** The correct statement is that the drain is
*suspended* while the engine serves, which is also why every "the fs tier served zero
bytes" result was real and correctly measured — the benchmark asked for data that genuinely
was not on the disk yet. The lookups were right; the tier was empty.

> ### ⚠ CORRECTION (R3.10.2) — the histogram is real, the conclusion drawn from it is not
>
> The mtime histogram above is sound data. The inference *"load suspends the cascade"* is not.
> R3.10 watched the cascade directly through `kv_offload_fs_inflight_jobs` plus a `df`-derived
> block count sampled every 10 s. During the 107 s eviction burst — the engine prefilling
> ~270,000 tokens flat out — the disk gained **693 blocks = 18.7 GB, i.e. 175 MB/s written
> while under full load**. When the load stopped it kept draining at 101–167 MB/s across ten
> consecutive samples while the backlog fell 61 → 0 in-flight jobs. It does not stall under
> load; the backlog *grows* under load (5 → 67 in-flight) because the store path feeds it
> faster than the device can absorb, and that is a different problem with a different fix.
>
> So the 21 blank minutes were not a suspended drain. They were an **empty queue**: the store
> path was refusing admissions (`kv_offload_allocation_failure` climbing, `evictable_perc`
> → 0.019 in R3.8), so no blocks entered the CPU tier and there was nothing for the cascade
> to cascade. Same symptom, opposite cause — and it points the fix at the store path
> (R3.3 retry livelock) rather than at cascade scheduling.
>
> What survives from R3.9.1: the 79.1 MB/s figure and the "130 : 1" ratio stay retired, and
> the "fs tier served zero bytes" results stay correctly measured.

### R3.9.2 `fio` on `/kvcache` — the disk was never the problem

Run with no GPU window, `/dev/vda` (dedicated to `/kvcache`):

| test | shape | result |
|---|---|---|
| A | 1 job, 1 MiB blocks, O_DIRECT sequential | **470.1 MB/s** |
| B | 16 jobs, 1 MiB blocks, O_DIRECT | 455.1 MB/s — *no scaling, device-limited* |
| C | 16 jobs, one 27,000,832 B write per file, O_DIRECT | 223.8 MB/s |
| Cb | as C, buffered (no O_DIRECT) | 269.6 MB/s |
| D | Python replica of `_store_block`, 16 threads | 204.9 MB/s |

Test D reproduces the cascade's exact write shape and lands within 1% of the 202.1 MB/s the
real cascade achieves when it runs. **The cascade is already at its ceiling for the shape it
uses.** The device has ~2.3x more to give, but only at a different write shape (larger
sequential runs), and B shows more threads buy nothing.

*Retracted:* an earlier single-pair comparison suggested chunking the 27 MB write into 1 MiB
pieces was worth +36%. A repeated A/B (unchunked 222.4 / 127.5 / 134.3 MB/s; chunked 207.4 /
171.1 / 171.1) shows run-to-run variance on this device of 127–222 MB/s for identical work,
which is larger than the effect. **There is no measured chunking win.** Do not act on it.

### R3.9.3 The tier geometry — the CPU tier is not a cache

Read off the fs tier's own on-disk layout (`config.json` plus a `stat` of any block file),
which is exact where the metrics-derived figure is not:

```
tokens_per_hash   = 1,648        blocks_per_file = 1        kv_cache_groups = 9
block file size   = 27,000,832 B  (16 KiB per token per group, uniform across all 9 groups)
bytes per token   = 9 x 27,000,832 / 1,648 = 147,456 B   (144 KiB)
```

Two consequences, both of which invalidate earlier sizing:

**1. Offloaded KV costs 3.6x what GPU KV costs.** GPU KV is ~40 KiB/token (228,737 tokens in
the pool); offloaded KV is 144 KiB/token. The gap is not compression — it is that 6 of the 9
KV groups are fixed-size Mamba/GDN state, which the GPU keeps once per sequence but the
offload path must write once per *block*.

**2. The CPU tier holds 115,360 tokens, not 472,057.** The tier reports "636 blocks" and
those are per-`(hash, group)` **file slots**, not logical blocks: 636 x 27,000,832 B =
15.99 GiB, i.e. exactly the 16 GiB shm region. Logical capacity is 636 ÷ 9 = 70 blocks =
**115,360 tokens — 0.50x the GPU cache.**

That second point is structural, not a tuning problem:

> **The CPU tier is smaller than the GPU cache, and both are LRU over the same stream.
> Anything the GPU evicts was pushed out of the CPU tier strictly earlier. A CPU-tier hit
> on a prefix that was ever GPU-resident is therefore impossible by construction.**

The CPU tier is not a cache for this model. It is a staging buffer for the disk tier. Sizing
it up to be a real cache would need >32 GiB of shm on a 31 GiB host — not available. The
disk tier is where retention has to come from: 512 GB ≈ 3.5M tokens ≈ **15x the GPU cache**.

*Closed off, so neither of us re-proposes it:* offloading only the 3 attention groups to cut
the 3.6x cost would not work. GDN/Mamba state cannot be reconstructed from a partial cache —
it is a recurrence, so recovering it means replaying the sequence, which is the recompute we
are trying to avoid. All 9 groups are required.

### R3.9.4 Four stacked problems, in the order they bite

The zero-hit result is not one bug. It is four, and fixing any one alone changes nothing.

1. **The CPU tier is structurally too small to serve hits** (R3.9.3). Not fixable by tuning.
2. ~~**The fs cascade does not run under load**~~ → **the store path starves it of work**
   (R3.9.1 as corrected by R3.10.2). The cascade keeps up; blocks never reach the CPU tier
   for it to drain, so the disk tier is usually empty of anything recent.
3. **`patch_offload_mixed_hit.py` declines every external hit on a request that has any GPU
   hit.** This is our own patch, added 2026-09-06 after a crash. `external_prefix_cache_hits_total`
   ran 3–19.7% before it and has been **exactly 0** since. It is the production hit-killer.
4. **All nine KV groups are misclassified as draft groups.** The boot log says
   `EAGLE/MTP draft attention groups [0,1,2,3,4,5,6,7,8] detected. The trailing chunk of
   these groups will be excluded from offloading due to volatility.` Only group 8
   (`model.layers.64–68.self_attn.attn`) is the MTP draft head; groups 0–7 are
   `language_model.model.layers.*`. The blanket DFlash2 fallback marks all nine, so the
   newest chunk of every conversation is never offloaded — exactly the part a growing
   conversation needs next turn.

The decisive number tying (3) to the observed behaviour:
`external_prefix_cache_queries_total` = 1,525,385 with `external_prefix_cache_hits_total` = 0,
and 1,612,729 − 87,344 = 1,525,385 exactly. Every query the GPU did not serve went to the
external tier and every one was declined.

### R3.9.5 Revised build order (supersedes R3.6)

**Phase 1 — is a disk hit achievable at all?** No engine code change; harness only.
`tierbench.py` gains (a) a `drain()` that waits for `kv_offload_fs_inflight_jobs` to reach 0
*and* the on-disk block count to stop moving, run both before and after the eviction, so the
probe is issued against a settled tier rather than into a write backlog; (b) the corrected
geometric tier sizing above; (c) an explicit note that the `cpu` phase cannot pass. **This is
the experiment that decides everything else.** If it produces a hit, the read path is sound
and every remaining problem is about *when* data lands. If it misses with blocks provably on
disk, there is a key-matching or load-path bug and the model in R3.9.4 is wrong.

**Phase 2 — make hits legal again** (needs Phase 1 to have passed):
* Fix the group misclassification so only group 8 is treated as a draft group.
* Replace the blanket mixed-hit decline with a targeted boundary guard. All 9 groups share
  `tokens_per_block = 1648`, so the alignment arithmetic is uniform across them — the
  assertion that crashed EngineCore is checkable directly rather than avoided wholesale.

**Phase 3 — make it work under load:**
* Fix the retry livelock (`continue` without `advance_stored_idx`, R3.3).
* ~~Make the cascade drain while the engine serves (R3.9.1).~~ **Closed by R3.10.2 — it already does.**
* Flush the cascade backlog on shutdown, so a restart does not discard un-written blocks.

**Not doing:** disks (R3.9.2), CPU-tier resizing (R3.9.3), attention-only offload (R3.9.3).

---

## R3.10 — PHASE 1 PASSED: the disk tier served a real 85,696-token hit

**Run `9e4d2b`, GPU window 2026-09-07, hand-launched container on port 5804.**
Artefacts: `baselines/REPORT-9e4d2b.md`, `baselines/RESULTS-9e4d2b.json`.

This is the first external cache hit in the entire investigation. Every prior run recorded
`external_prefix_cache_hits_total = 0`. **The read path is sound.** Everything that remains
is about *what gets written* and *when*, not about whether a disk hit can be served.

### R3.10.1 The result

Prefix 90,045 tokens (90,085 prompt tokens), 4 phases, one settled tier between each:

| phase | expected | served | wall | ttft | `cached_tokens` |
|---|---|---|---|---|---|
| cold | RECOMPUTE | RECOMPUTE ✓ | 45.20 s | 43.53 s | 0 |
| gpu | GPU | GPU ✓ | 2.25 s | 1.89 s | 87,344 |
| cpu | *(impossible by construction, R3.9.3)* | RECOMPUTE ✓ | 42.87 s | 42.50 s | 0 |
| fs | OFFLOAD | **OFFLOAD ✓** | 94.88 s | 94.53 s | **85,696** |

fs-phase metric deltas — the ones that matter:

```
external_prefix_cache_hits_total        85,696      (was exactly 0 in every prior run)
external_prefix_cache_queries_total     90,085
kv_offload_load_bytes_total          3,002,662,912  (3.00 GB — first non-zero read-back ever)
kv_offload_load_size_count                   1      (one single load job)
kv_offload_load_time_total               0.254 s    -> 11.8 GB/s CPU->GPU
kv_offload_tiering_lookup_async_delay     91.758 s
kv_offload_tiering_lookup_sync_delay       0.0057 s
kv_offload_allocation_failure_total           0
prefix_cache_hits_total                       0     <- see below
```

Two things follow immediately:

* **The mixed-hit patch did not fire.** `prefix_cache_hits_total` moved by 0, so
  `num_computed_tokens == 0` and `patch_offload_mixed_hit.py`'s blanket decline was never
  reached. R3.9.4 item 3 is still the production hit-killer, but it is *not* what was
  producing zeros in the bench: the bench evicts the GPU cache first, so its requests have
  no GPU hit to be "mixed" with. Both statements are true and they are about different
  workloads. In production, where a conversation always has a GPU-resident head, item 3
  bites on every turn.
* **A hit is currently 2.1x slower than recomputing** (94.88 s vs 45.20 s cold). Correctness
  first was the right call, but this is not shippable as-is. R3.10.3 says where the time goes.

### R3.10.2 The cascade keeps up under load (corrects R3.9.1)

See the correction block in R3.9.1. Instrument: `drain()` in `tierbench.py` polls
`kv_offload_fs_inflight_jobs` and a `df --output=used`-derived block count every 10 s and
declares the tier settled after 3 consecutive quiet polls.

Two implementation notes for whoever reads that code:

* The block count comes from `df`, not `find`. One `find` walk of 12.5k block files against
  a cold page cache took **over two minutes** — unusable inside a poll loop. Every block file
  is exactly 27,000,832 B, so `used_bytes // 27,000,832` is exact for deltas and O(1).
* The quiet test is `n <= last`, not `n == last`. `kvcache-reap.timer` fires every 5 minutes
  and deleted 484, then 976, then 224 blocks mid-run; a strict-equality test would never
  settle. The reaper's `MIN_AGE_MIN=90` hard floor means it cannot touch anything the run
  itself wrote, so the experiment stands — it only makes the count non-monotonic.

### R3.10.3 The 94.88 s is all staging, not transfer

The hit moved 3.00 GB CPU→GPU in **0.254 s (11.8 GB/s)**. The other **91.76 s** is booked to
`tiering_lookup_async_delay` — a name that suggests key lookup, but R3.5 already established
it is wall time from "lookup issued" to "blocks resident in the CPU tier". It is the
**fs → CPU stage**, and it is the entire cost of the hit.

12.51 GB (what the cold phase wrote for this prefix) ÷ 91.76 s = **136 MB/s** — squarely in
the 101–175 MB/s band the cascade writes at and inside `fio` test D's 204.9 MB/s ceiling for
this write shape (R3.9.2). So the stage is reading back roughly everything that was written,
at about device speed, one 27 MB file at a time, and there is no mystery in the rate.

The lever is not the rate. It is the **volume** — R3.10.4.

### R3.10.4 The store path writes ~4x more than any read can use

This is the highest-value finding of the run and it resolves a discrepancy that looked like a
bug: the store path wrote **12.51 GB** for this prefix, the read path pulled back **3.00 GB**.
Ratio 4.17x. It is not padding and it is not a bug — the block files are 97.8–100% non-zero
right to the last byte (checked directly, all 9 groups of one shared block hash).

The geometry, now derived rather than guessed:

```
attention page/layer/block = 2 x 1648 tok x 4 kv-heads x 256 head-dim x 1 B (fp8) = 3,375,104 B
block file                 = 8 layers x 3,375,104                              = 27,000,832 B
```

so 27,000,832 B is *exactly right* for an 8-layer group, and vLLM's hybrid allocator pads
every group to that same page size (boot log: `Add 3 padding layers, may waste at most
60.00% KV cache memory` — group 8 has 5 real layers in 8 slots). Per token, across 9 groups:

| groups | layers | role | B/token | share |
|---|---|---|---|---|
| 0–5 | 48 | Mamba/GDN recurrent state | 98,304 | 66.7% |
| 6–7 | 16 | full attention | 32,768 | 22.2% |
| 8 | 5 real + 3 padding | MTP draft head | 16,384 | 11.1% |
| | | **total** | **147,456** | |

Now the part that matters. `MambaSpec.max_memory_usage_bytes` in
`v1/kv_cache_interface.py:729-737`, under our `mamba_cache_mode=align`:

```python
elif vllm_config.cache_config.mamba_cache_mode == "align":
    return self.page_size_bytes * (2 + self.num_speculative_blocks)
```

and `max_num_blocks_per_req` explains why:

> *"only 2 + num_speculative_blocks state blocks are resident at a time (earlier states are
> nulled out by `remove_skipped_blocks`)"*

**The GPU keeps two Mamba state blocks per request. The offload store path writes one per
block, for every block.** That is where the 4x lives, and it is why the numbers reconcile:

* GPU pool: 9,300,000,000 B pinned (`KV_MEM`), 8.66 GiB reserved, = 342 per-group blocks =
  38 logical blocks. At 147,456 B/token that is only 62,624 tokens — yet the boot log says
  `GPU KV cache size: 228,737 tokens`, 3.65x more. Both are right: the log figure is
  `max_concurrency x max_model_len` and correctly credits the Mamba groups with needing
  2 blocks per request instead of one per block.
* A load only needs the Mamba state **at the resume boundary**, not the 50 stale snapshots
  behind it. So the 3.00 GB read-back is not a short read — it is the right amount.

Consequences, in order of value:

1. **Two thirds of every byte written to disk is a Mamba snapshot that nothing will ever read
   back.** Storing GDN state at coarser granularity — the last block of a stored run, or every
   Nth block — cuts write volume ~3x, raises effective disk capacity from ~3.5M to ~10M tokens,
   and cuts the 91.76 s stage proportionally. The cost is that a resume landing between
   snapshots must replay the 48 linear-attention layers over up to N-1 blocks (N=4 → ≤4,944
   tokens) with the attention KV still cached. **This is the change to make next.**
2. It is *not* the same as the attention-only-offload idea closed off in R3.9.3. That one
   dropped GDN state entirely and cannot work — the recurrence has to start somewhere. This
   one keeps it, just not 50 copies of it.
3. The 3 padding layers in group 8 waste 6,144 B/token (4.2% of the footprint) for nothing.
   Low value, but free if the group geometry is touched anyway.

### R3.10.5 The reaper shreds logical blocks

`kvcache-reap.sh` deletes oldest-mtime-first at **file** granularity. A logical block is only
usable if all 9 of its group files survive, and the 9 files of one block are written by
different cascade threads at slightly different times. Deleting a single one of them makes
the other 8 dead weight — still on disk, still counted against `TARGET_PCT`, never a hit.

Not urgent while nothing hits, but it must be fixed before retention policy means anything:
reap by logical block (delete all 9 group files for a hash together), not by file.

### R3.10.6 Confirmed from the boot log, not inferred

R3.9.4 item 4 (all nine groups misclassified as draft groups) now has its source line.
`offloading/scheduler.py:220-231`:

```python
eagle_groups = {idx for idx, g in enumerate(...) if g.is_eagle_group}
use_eagle = (vllm_config.speculative_config is not None
             and vllm_config.speculative_config.use_eagle())
if use_eagle and not eagle_groups:
    eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))
```

DFlash2 sets `use_eagle()` but no group carries `is_eagle_group`, so the fallback marks all
nine and the trailing chunk of every group is excluded from offloading. Boot log line:
`KV offloading: EAGLE/MTP draft attention groups [0,1,2,3,4,5,6,7,8] detected.`

### R3.10.7 Build order (supersedes R3.9.5)

**Phase 1 — CLOSED, passed.** The read path works; a settled disk tier serves hits.

**Phase 2 — cut the write volume** (new, and now ahead of everything else):
* Store Mamba/GDN groups at coarse granularity under `mamba_cache_mode=align` (R3.10.4).
  Behind an env flag, default off, like every other radiance patch.
* Fix the group misclassification so only group 8 is treated as a draft group (R3.10.6).

**Phase 3 — make hits legal in production:**
* Replace `patch_offload_mixed_hit.py`'s blanket decline with a targeted boundary guard.
  All 9 groups share `tokens_per_block = 1648`, so the alignment arithmetic is uniform and
  the assertion that crashed EngineCore is checkable directly rather than avoided wholesale.
  Until this lands, production hit rate stays 0 no matter what the tiers do.

**Phase 4 — housekeeping:**
* Fix the retry livelock (`continue` without `advance_stored_idx`, R3.3) — this is what
  starves the cascade (R3.10.2), not cascade scheduling.
* Reap by logical block, not by file (R3.10.5).
* Flush the cascade backlog on shutdown, so a restart does not discard un-written blocks.

**Not doing:** disks (R3.9.2), CPU-tier resizing (R3.9.3), attention-only offload (R3.9.3),
cascade scheduling changes (R3.10.2).

### R3.10.8 Still unmeasured

* ~~Whether a hit is *correct*.~~ **Answered in R3.11** — run `9e6993`, a disk hit is
  bit-identical to a cold recompute, 128/128 tokens and every logprob exact, against a
  cold-vs-cold noise floor that is itself zero. What is still unmeasured is correctness of a
  **mixed** hit (partial GPU hit plus an external hit), which is the case Phase 3 has to fix
  and the case no bench has yet entered (R3.11.4).
* The `cpu` phase can never pass while the CPU tier is 0.50x the GPU cache (R3.9.3), so the
  tier that serves a hit is still inferred from `load_bytes` rather than labelled. R2.9.2's
  per-tier counters would settle it.

---

## R3.11 — PHASE 1 CONFIRMED CORRECT: a disk hit is bit-identical to a cold recompute

R3.10 proved the disk tier can serve a hit. It did not prove the hit was *right*: `tierbench`
throws the completion text away, so "the read path is sound" rested on the volume of bytes
moved and on the phase reaching the tier it was aimed at. R3.10.8 listed this as the one thing
still unmeasured and said to settle it before shipping anything. It is now settled.

Instrument: `equivbench.py`, run `9e6993`, prompt 90,032 tokens, `max_tokens=128`,
`temperature=0`, `logprobs: true, top_logprobs: 3`. Artefacts in
`baselines/EQUIV-9e6993.{md,json}` and `baselines/equivbench-9e6993.log`.

### R3.11.1 The method — how to get a cold-vs-cold control without an eviction

"Same answer" means nothing without a noise floor. Two things could make two runs of one
prompt disagree with no cache bug at all: DFlash2 speculative decoding (MTP measured +15.8σ on
this box; DFlash2 measured clean at +1.4σ, which is a statistical claim, not a promise of
bit-identical greedy decode) and batch-shape-dependent kernel reduction order.

The control that removes both is the *same prompt text recomputed twice from scratch*, and it
looked expensive — an eviction cycle between the two runs. It is not, because this build
accepts `cache_salt` on `/v1/chat/completions` and vLLM folds the salt into the block-0 extra
keys (`v1/core/kv_cache_utils.py:579-580`). Block hashes chain, so salting the first block
invalidates the entire chain: identical text under a fresh salt is a guaranteed full recompute
at the cost of one prefill and no eviction traffic at all.

That is confirmed in the run itself, not assumed: `coldA` cost 45.82 s and `coldB` 45.85 s,
both with `cached_tokens = 0` and both classified RECOMPUTE. A partial hit would have shown up
as a shorter wall.

### R3.11.2 The result

Four measurements of one prompt, all four valid:

| phase | cache_salt | expected | served | wall | cached_tokens |
|---|---|---|---|---|---|
| coldA | A | RECOMPUTE | RECOMPUTE ✓ | 45.82 s | 0 |
| coldB | B | RECOMPUTE | RECOMPUTE ✓ | 45.85 s | 0 |
| gpu   | B | GPU | GPU ✓ | 4.42 s | 87,344 |
| fs    | B | OFFLOAD | **OFFLOAD ✓** | 88.62 s | **85,696** |

and four comparisons, token-level rather than string-level:

| pair | what it tests | tokens identical | logprobs bit-identical |
|---|---|---|---|
| coldA vs coldB | the **noise floor** — two independent recomputes | yes, 128/128 | yes, 128/128 exact |
| coldB vs gpu | does resuming from GPU-cached KV change the answer? | yes, 128/128 | yes, 128/128 exact |
| coldB vs fs | **the question** — disk hit vs same-chain recompute | yes, 128/128 | yes, 128/128 exact |
| coldA vs fs | disk hit vs an independent recompute | yes, 128/128 | yes, 128/128 exact |

`max |Δlogprob| = 0.000e+00` on all four pairs. Not "close" — every one of the 128 emitted
tokens carries the identical logprob to the last bit, in a full recompute and in a resume that
pulled 3.00 GB back off `/dev/vda`.

Two things follow, and the order matters:

1. The noise floor is **zero**. Greedy decode through DFlash2 at 90k context is bit-reproducible
   on this box. That is what makes the other three rows evidence instead of anecdote — had
   `coldA vs coldB` drifted, a matching `coldB vs fs` would have proved nothing.
2. The offload read path is **numerically exact**, not merely plausible. fp8 KV written to disk,
   read back, and resumed reproduces the same logits as recomputing from the tokens. There is
   no accuracy argument against disk caching here.

### R3.11.3 The fs phase really was disk, not the CPU tier

Same ambiguity as R3.10 (R2.9.2's per-tier counters still do not exist), settled the same way
plus one extra: the eviction pushed 270,000 tokens through a CPU tier that holds 115,360
(R3.10.4), so the prefix cannot have survived there. The metric deltas:

```
external_prefix_cache_hits_total                   85,696
external_prefix_cache_queries_total                90,072
kv_offload_load_bytes_total                 3,002,662,912
kv_offload_load_time_total                       0.253148 s   -> 11.9 GB/s
kv_offload_tiering_lookup_async_delay_seconds_sum  83.391 s
kv_offload_tiering_lookup_sync_delay_seconds_sum    0.0056 s
kv_offload_allocation_failure_total                     0
```

83.39 s of async lookup delay against 0.25 s of CPU→GPU transfer. A CPU-tier hit has no
staging step and would show an async delay near zero. This is R3.10.3's split reproduced
(88.62 s wall this time against 94.88 s, same shape), and it is the fs→CPU stage.

### R3.11.4 What this does and does not license

It licenses the Phase 2 work in R3.10.7: the read path is correct, so the remaining problem is
purely economic — the store path writes ~4x more than any read can use (R3.10.4). Cutting the
Mamba snapshot rate changes *what is written*, and R3.11 is the baseline any such patch must
reproduce. Re-run `equivbench.py` after Phase 2 and after Phase 3; a coarser GDN snapshot is
exactly the kind of change that can stay fast and stop being correct, and this bench is now
the test that would catch it.

It does not license anything about the mixed-hit path. Every phase here had
`prefix_cache_hits_total` delta 0, so `num_computed_tokens == 0` and
`patch_offload_mixed_hit.py`'s blanket decline never fired (R3.10.1). The case where a request
holds a partial GPU hit *and* an external hit — the case production is actually in, and the one
Phase 3 has to fix — remains untested for correctness. That is the next equivalence question,
not a solved one.

## R3.12 — Store less, and the RAM tier becomes the point (reframes R3.10.4)

Question that prompted this: *if we are planning on storing less data on disk, can we store
the same data in RAM and make that cache useful?* Answer: yes — the RAM tier is the better
half of the same change. But two things I asserted on a first reading of the source were
wrong, and the direct measurement below replaces them. **Read R3.12.2 before acting on
anything in R3.10.4.**

### R3.12.1 Two thirds of the disk is Mamba state — now measured directly, not fitted

`MambaSpec.page_size_bytes` (`v1/kv_cache_interface.py:719-722`) is
`sum(prod(shape) * get_dtype_size(dtype))` over the spec's shapes — **there is no `block_size`
term.** The GDN recurrent state is `linear_num_value_heads * linear_key_head_dim *
linear_value_head_dim * 4 B` = 48 x 128 x 128 x 4 = 3,145,728 B per layer, plus conv state,
giving the 3,375,104 B/layer behind a 27,000,832 B eight-layer group file — *whatever the
chunk covers.*

The store path writes one such file per chunk **for every group, Mamba included.** Counted on
the live tier (13,028 chunk directories, `df` cross-check 13,030 — layout is
`blocks/<model>_r0/<hash[0:3]>/b<blk>_g<grp>/<hash>.bin`, one 27,000,832 B file each):

| group | kind | chunk dirs stored |
|---|---|---|
| g0–g5 | Mamba/GDN, 8 layers each | 1,420 / 1,420 / 1,418 / 1,422 / 1,424 / 1,424 |
| g6, g7 | full attention, 8 layers each | 1,522 / 1,518 |
| g8 | MTP draft head | 1,460 |

**Six of nine groups — 66.7% of every byte on this device — are Mamba state snapshots, one
per 1,648 tokens.** A resume needs exactly one of them: the state at the hit boundary. That is
the whole of R3.10.4's 4.17x, and it is now a direct count rather than a number fitted to the
read-back. (The fit still corroborates it: 111 useful group-files x 27,000,832 =
2,997,092,352 B against the measured `kv_offload_load_bytes_total` 3,002,662,912 B, 0.19%
apart on a padded-vs-unpadded basis.)

### R3.12.2 CORRECTION — upstream's skip logic is *inapplicable* to us, not disabled

I first read `is_store_reachable_swa_chunk` + `get_sliding_window_size_in_chunks` (which does
return **1** for a `MambaSpec`, commented *"Mamba depends on a single state"*) as a ready-made
fix switched off by a gate, and said so. **That was wrong, and the reason matters.**

The gate is `_alignment_chunk_count` (`offloading/scheduler.py:201-212`), and
`alignment_tokens` is built (:186-196) only from groups where
`get_sliding_window_size_in_chunks(...) is None` — i.e. the *full-attention* groups. It then
bails on `alignment_tokens <= tokens_per_chunk`. For us those are equal: the tier's own
`config.json` reports `"tokens_per_block": 1648` for **all nine groups** and
`"blocks_per_file": 1`, so every group's chunk is 1,648 tokens.

The optimization exists because in DeepSeek V4 the SWA groups have *smaller* chunks than the
MLA group, so a load hit — which can only land on a full-attention boundary — can never
reach the SWA chunks earlier in each segment. Those are unreachable *by construction*.
**Ours are not.** With every group on the same 1,648-token grid, every stored Mamba chunk sits
exactly on a hit boundary and is a perfectly valid resume point. Upstream is behaving
correctly for our geometry; there is nothing here to switch on.

So our waste is a different thing, and needs saying precisely:

> we are not storing unreachable state — we are storing **1,420 valid resume points where a
> real workload only ever resumes at a handful.**

That makes it a *policy* change (store one Mamba state per N chunks, and round the hit window
down to that coarser grid), not a flag flip. Cost and risk both go up accordingly.

### R3.12.3 The RAM threshold is a cliff at 1.0x the GPU cache

R3.9.3: a CPU-tier hit is impossible while the tier is smaller than the GPU cache, because
both are LRU over the same stream — anything the GPU evicted left RAM strictly earlier.
Today 115,360 vs 228,737 tokens = **0.50x**.

The CPU tier's 636 slots are per-`(hash, group)`. Storing Mamba once per N chunks makes the
slots per N-chunk segment `2N (attention) + 6 (one Mamba checkpoint) + N (MTP)` = `3N+6`,
covering 1648N tokens:

| Mamba every N chunks | CPU tier capacity | vs GPU cache |
|---|---|---|
| 1 (today) | 116,459 tok | 0.51x — **impossible** |
| 2 | 174,688 | 0.76x — still impossible |
| 4 | 232,917 | 1.02x — on the line, no margin |
| 8 | 279,501 | 1.22x |
| 8, MTP group dropped | 381,137 | **1.67x** |
| attention only (limit) | 524,064 | 2.29x |

(N=1 computes to 116,459 against the measured 115,360; the gap is 636/9 = 70.67 truncated to
70. The model is right.)

**N=4 is not enough** — it lands on the line. N=8 plus dropping group 8 is what buys usable
margin, and even then a prefix evicted from the GPU survives in RAM only for about the time it
takes to push another ~150,000 tokens through. A real cache, but a shallow one. Note what this
retires: "a real cache needs >32 GiB shm on a 31 GiB host" was sizing *today's* density, not a
law. 16 GiB is enough at N=8.

### R3.12.4 Why RAM, and not just cheaper disk

Measured in R3.11: CPU→GPU transfer **0.253 s**; fs→CPU staging **83.39 s**. The RAM tier is
not merely another place to hit — it is the only tier that hits at GPU-like speed.

| | today | with N=8 |
|---|---|---|
| cold recompute | 45.8 s | 45.8 s |
| GPU hit | 4.4 s | 4.4 s |
| **RAM hit** | impossible | **~1–2 s** |
| disk hit | 88.6 s | ~35 s |

Disk improves too, but only to rough parity with recomputing, because the fs→CPU stage pulls
everything *stored* for the range, not everything *needed*. The disk tier's job is the long
tail; the turn-to-turn win has to come from RAM.

### R3.12.5 The cheap experiment to run first: `blocks_per_chunk`

`blocks_per_chunk` is a **supported connector config key** (`offloading/config.py:67-96`,
accepting either `blocks_per_chunk` or `block_size` in `kv_connector_extra_config`, not both).
Our live `--kv-transfer-config` sets **neither**, so it is at its default of **1** — one chunk
per 1,648-token block, which is what `"blocks_per_file": 1` on the tier reflects.

Raising it is interesting because of how the store path enumerates blocks
(`scheduler.py:1112-1118`): *"For each chunk, take the last corresponding GPU block"* — **one
GPU block id per chunk per group.** And `resolve_mamba_align_size` (:144-161) derives the
hit-window rounding from `tokens_per_block * blocks_per_chunk` automatically, so the align
grid follows the chunk size with no code change.

**The open question I could not settle by reading:** whether a Mamba group's per-chunk payload
then stays *one* state (27,000,832 B, an N-fold volume cut for free) or becomes *N* states
(same volume, merely bundled into bigger files). It depends on the sub-block expansion in
`compute_sub_block_ptrs` (`v1/kv_offload/cpu/gpu_worker.py:72-119`), which slices a row into
`blocks_per_chunk` sub-pages — well-defined for an attention page, meaningless for a
fixed-size recurrent state, and I could not trace which tensor shape wins for a MambaSpec
group without running it.

**So measure it, do not reason about it.** Set `"blocks_per_chunk": 2` in
`kv_connector_extra_config`, restart, push one long prefix, then re-run the count from R3.12.1
and `stat` a `g0` file against a `g6` file:

* `g0` count halves and its file stays 27,000,832 B → **volume cut for free, config only.**
  Go straight to N=8 and re-run `equivbench.py`.
* `g0` count halves and its file is 54,001,664 B → bundling only, no volume change. The
  policy patch in R3.12.2 is required after all.

Either way the price of a coarser grid is tail rounding on every turn: today 1,664–3,327
tokens ([[mamba-align-cache-shortfall-open]], closed as by-design), rising to 0–11,536 at N=8,
averaging ~5,800 tokens ≈ **3 s** at the measured 1,966 tok/s prefill rate. Whether that is
free or expensive depends on one more unknown: **does the align rounding apply to the GPU
prefix-cache path, or only to the external-hit window?** `resolve_mamba_align_size` is called
from the offloading scheduler and used only at `scheduler.py:653-656` to clamp
`max_hit_size_tokens`, which reads external-only — in which case you pay the 3 s exactly when
you would otherwise have recomputed for 45.8 s, and it is free. Confirm that before N=8.

This experiment needs an engine restart, so it does not belong inside a held GPU window that
is keeping a model hot.

### R3.12.5a RESULT (measured 2026-09-07, probe `bpcprobe-1788772843`) — **BUNDLING ONLY**

Run for real, not reasoned about. Method: two new launcher knobs
(`KVOFF_BLOCKS_PER_CHUNK`, `KVOFF_DISK_SUBDIR`, both defaulting to today's behaviour — with
the first unset the generated `--kv-transfer-config` JSON is byte-identical to the live
container's) armed the same entry with `blocks_per_chunk=2` writing to a **fresh, empty**
`/kvcache/blocks-bpc2`. A separate root matters twice over: it isolates the measurement from
the 328 GB production tree, and it keeps `kvcache-reap.sh` (`ROOT=/kvcache/blocks`) from
deleting anything mid-count. Boot confirmed the arm:

```
[kv-offload] L3 disk tier: /kvcache -> /kvcache/blocks-bpc2 (PYTHONHASHSEED=0,
[kv-offload]   NON-DEFAULT blocks_per_chunk=2
```

GPU KV cache came up at **228,737 tokens — identical to production**, so the knob does not
touch GPU geometry.

One 40,062-token prefix, then settle. Files reached disk immediately; no eviction round was
needed, which incidentally re-confirms R3.10.2 (the cascade keeps up). Settled layout:

| group | kind | chunk dirs | file size |
|---|---|---|---|
| g0–g5 | Mamba / GDN | 12 each | 54,001,664 B |
| g6–g7 | full attention | 12 each | 54,001,664 B |
| g8 | MTP draft head | 12 | 54,001,664 B |

**Perfectly uniform.** 54,001,664 = exactly 2 x 27,000,832. Every group, including the six
Mamba groups, writes `blocks_per_chunk` copies of the page. 12 chunks x 3,296 tokens = 39,552
tokens covering the 40,062-token prompt, so the grid is exactly as expected.

Total 5,832,179,712 B for 40,062 tokens = **145,579 B/token, against the `blocks_per_chunk=1`
reference of 147,456 B/token — 98.7% of it.** The 1.3% is tail rounding on the last partial
chunk, not a saving.

**Verdict: `blocks_per_chunk` bundles, it does not deduplicate.** The second branch of R3.12.5
is what happened, so:

* **There is no config-only win.** N=8 via `blocks_per_chunk` alone buys nothing — not on
  disk, and not in RAM either: the CPU primary is byte-sized (16 GiB / 27,000,832 = 636
  slots), so bundling into 54 MB slots gives 318 slots covering 318/9 x 3,296 = 116,480
  tokens — the same ~115k measured at N=1. Bundling is byte-neutral everywhere, which is the
  one number that decides whether the RAM tier can hit at all (R3.12.3).
* **The store-policy patch of R3.12.2 is required.** Cutting Mamba writes to 1-in-N has to be
  done in the store path, by skipping the Mamba groups' chunks that no resume will ever ask
  for, not by re-chunking.
* The align-rounding question at the end of R3.12.5 (GPU prefix path vs external-hit window)
  is **not urgent any more** — nothing is going to raise `blocks_per_chunk`. It comes back
  only if the store-policy patch chooses to move the align grid as part of its own design.

One thing bundling might still be worth, separately and later: the 83 s fs->CPU staging is
3.0 GB across ~111 files at ~36 MB/s, well under what `fio` says the device can do (R3.9.2).
If that gap is per-file overhead rather than bandwidth, fewer/larger files would close it.
Not measured, not on the critical path, and it does not change the volume problem.

### R3.12.6 CORRECTION — the mixed-hit guard is OFF in production, and hits are still ~0

The note that "production hit rate stays 0 until `patch_offload_mixed_hit.py` is replaced" is
**wrong as a statement about the current server.** `KVOFF_MIXED_HIT` defaults to **1**
(`llama-swap-ggz14-27b.sh:709`, comment: *"DEFAULT IS 1 — deliberately the crashing
behaviour"*), and the live container carries `RADIANCE_OFFLOAD_MIXED_HIT=1`. The guard is not
active. Upstream mixed-hit behaviour is what production has been running.

And external hits are still essentially zero. Live counters:

| | queries | hits | rate |
|---|---|---|---|
| GPU prefix cache | 1,586,008 | 182,928 | 11.5% |
| external (CPU+fs) | 1,403,080 | 171,392 | 12.2% |

171,392 is **exactly 2 x 85,696** — the two bench hits of runs `9e4d2b` and `9e6993` and
nothing else. Production contributed no measurable external hit across ~1.4M queried tokens
*with the guard off.*

**So Phase 3 is not the gate.** Something else keeps production from hitting: the newest chunk
of every group excluded by the group misclassification (R3.10.6), the retry livelock (R3.3),
or simply that a turn returns before the cascade has settled the chunks it needs — the 83 s
async delay is longer than a user's think time. That has to be diagnosed on its own evidence,
not assumed to be the guard. Caveat: this is a cumulative counter over one container lifetime
and the attribution rests on the 2x coincidence being exactly that; a clean window with
production traffic and no bench would settle it properly.

### R3.12.7 Unchanged

* The group misclassification (R3.10.6) still has to be fixed first: with all nine groups
  forced to `is_eagle_group = True`, `reachable_tail` gains 1 everywhere and the trailing chunk
  of every group stays unofflodable — the newest chunk is exactly what the next turn needs.
* Correctness must be re-proven with `equivbench.py` (R3.11) after any of this. A coarser GDN
  snapshot is exactly the class of change that stays fast and stops being exact.

## R3.13 — Build sheet: the store-policy patch (now mandatory, after R3.12.5a)

R3.12.5a closed the only config-only route, so this is the build. It is smaller than R3.12.2
implied, for a reason worth stating plainly: **upstream already wrote this mechanism.** It was
written for DeepSeek V4, where the SWA groups sit on a *finer* chunk grid than the MLA group,
so most SWA chunks are unreachable by construction and get skipped at store time. Every piece
of that machinery is present in our container and simply never fires, because our nine groups
all share one 1,648-token grid. The patch is to make it fire for Mamba groups on purpose,
rather than only when the geometry happens to trigger it.

### R3.13.1 The single choke point

`_build_store_jobs` (`offloading/scheduler.py:1122-1136`) decides, one chunk at a time, whether
a chunk is written at all:

```python
if not is_store_reachable_swa_chunk(
        abs_chunk_idx, num_chunks,
        group_config.alignment_chunk_count,
        group_config.sliding_window_size_in_chunks,
        group_config.is_eagle_group):
    continue
```

and `is_store_reachable_swa_chunk` (:124-142) keeps, within each `alignment_chunk_count`-long
segment, only the trailing `sliding_window_chunks + int(is_eagle_group)` chunks. For a Mamba
group `get_sliding_window_size_in_chunks` already returns **1** — *"Mamba depends on a single
state"*. So the store side of "one Mamba snapshot per N chunks" is **already implemented and
already correct**; it is one chunk in every `alignment_chunk_count`.

The reason it does nothing for us is the first line of the function: `alignment_chunk_count is
None → return True`. And it is None because `_alignment_chunk_count` (:201-212) bails on
`alignment_tokens <= tokens_per_chunk`, and for us both are 1,648.

### R3.13.2 The change

Three edits, all in `offloading/scheduler.py`, none of them new logic:

1. **Set `alignment_chunk_count = N` for `MambaSpec` groups** in
   `SchedulerOffloadConfig.from_spec` (:186-231), from a new connector config key
   (`mamba_store_stride`, default 1 = today's behaviour), instead of letting
   `_alignment_chunk_count` return None. Nothing downstream changes: the store filter is
   already written against this field.
2. **Make `resolve_mamba_align_size` (:144-161) return `tokens_per_block * N`** rather than
   `tokens_per_block * blocks_per_chunk`. This is the load side. It feeds `max_hit_size_tokens`
   at :653-656, which rounds a hit window *down* to a boundary where a stored Mamba state
   actually exists. Without this edit the load path would ask for states that were never
   written — the one way this patch can be wrong, and the one edit that prevents it.
3. **Fix the EAGLE/MTP misclassification** (:220-231). This is now a hard prerequisite, not
   the cosmetic item R3.10.6 filed it as. `kv_cache_groups[i].is_eagle_group` is False for all
   nine groups, so `if use_eagle and not eagle_groups: eagle_groups = set(range(9))` marks
   **every** group as a draft group. That costs twice over: `reachable_tail` becomes
   `1 + 1 = 2`, so we would store 2-in-N and halve the win; and the load path drops each
   group's trailing chunk as a volatile draft tail — the newest chunk, which is exactly what
   the next turn of a conversation needs. Only group 8 is the MTP head.

`blocks_per_chunk` stays at 1. R3.12.5a settled that it is orthogonal and buys nothing.

### R3.13.3 What it is worth

Per 1,648-token chunk, group-files written today vs at N=8:

| | today | N=8 | N=8, MTP dropped |
|---|---|---|---|
| full attention (g6, g7) | 2 | 2 | 2 |
| MTP draft head (g8) | 1 | 1 | 0 |
| Mamba / GDN (g0–g5) | 6 | 6/8 = 0.75 | 0.75 |
| **total per chunk** | **9** | **3.75** | **2.75** |
| **volume** | 100% | **42%** | **31%** |

The disk saving is the lesser half. The point is R3.12.3: the CPU primary is byte-sized
(16 GiB / 27,000,832 = 636 slots), and slots per N-chunk segment fall from 9N to `3N+6`
(`2N` attention + `6` Mamba + `N` MTP). At N=8 that is 30 slots per 13,184 tokens →
**279,501 tokens, 1.22x the GPU cache**; dropping the MTP group as well gives 22 slots →
**381,137 tokens, 1.67x**. Above 1.0x the RAM tier stops being a staging buffer and becomes a
cache that can serve a hit — worth ~1–2 s against 45.8 s cold and 88.6 s from disk.

### R3.13.4 What it costs, and what has to be proven before it ships

* **Tail rounding grows.** The hit window rounds down to N x 1,648. Today's shortfall is
  1,664–3,327 tokens ([[mamba-align-cache-shortfall-open]], closed as by-design); at N=8 it is
  0–13,184, averaging ~6,600 ≈ **3.4 s** at the measured 1,966 tok/s prefill. Paid only on
  turns that would otherwise have recomputed for 45.8 s, so it is cheap — *provided* the next
  point holds.
* **Confirm the align clamp is external-only.** `resolve_mamba_align_size` is called from the
  offloading scheduler and consumed only at :653-656 to clamp `max_hit_size_tokens`, which
  reads external-hit-only. If it also reached the GPU prefix-cache path, that 3.4 s would be
  paid on *every* turn, including ones already served from GPU, and the trade would invert.
  This is the one open question and it is cheap to settle by instrumenting the call site.
* **Re-prove correctness with `equivbench.py` (R3.11).** A coarser GDN snapshot is precisely
  the class of change that stays fast and quietly stops being exact. The R3.11 method gives a
  zero noise floor — a disk hit was bit-identical to a cold recompute, 128/128 tokens and every
  logprob to 0.000e+00 — so any drift at all is a real regression, not measurement slop.
* **Start at N=2 or N=4, not N=8.** The capacity table says N=4 (232,917 tokens, 1.02x) is on
  the line with no margin, so N=8 is the target — but the first run should be the smallest N
  that is observably different from today, to separate "the patch works" from "N is too
  coarse".

### R3.13.5 Order

1. Fix the EAGLE misclassification alone, restart, confirm from the boot log that only group 8
   is listed, and re-check whether external hits move. It is a prerequisite either way, and it
   is also one of the live candidates for why production sits at ~0 hits (R3.12.6) — worth
   isolating before anything else changes underneath it.
2. Add `mamba_store_stride`, default 1, and confirm the generated config is byte-identical
   with it unset (the same discipline that made R3.12.5a trustworthy).
3. N=2, count the tier, confirm 6/2 = 3 Mamba files per 2 chunks. Then `equivbench.py`.
4. Settle the external-only question, then N=8, then re-measure the RAM tier against the
   1.0x threshold.

## R3.14 — CLOSEOUT PLAN (supersedes every earlier build order: R3.6, R3.9.5, R3.10.7, R3.12, R3.13.5)

This is the plan to finish the work and stop. It is written so that each phase has an exit
criterion and a stated condition under which the right answer is to **stop and turn the
feature down**, because that is a legitimate outcome and the document should say so before we
find out rather than after.

### R3.14.0 What is actually established, and what is not

**Established by measurement, not inference:**

* The disk tier can serve a real hit — 85,696 tokens (run `9e4d2b`).
* That hit is **bit-identical** to a cold recompute — 128/128 tokens, every logprob to
  0.000e+00 (run `9e6993`). There is no accuracy objection to any of this.
* The disk itself is not the bottleneck (`fio`, R3.9.2), and the cascade keeps up under load
  (R3.10.2, re-confirmed incidentally by R3.12.5a's writes landing immediately).
* Six of the nine groups are redundant Mamba snapshots, counted directly (R3.12.1).
* `blocks_per_chunk` cannot fix that — it bundles, it does not deduplicate (R3.12.5a).
* The mixed-hit guard is **off** in production and hits are still ~0 (R3.12.6).

**Not established, and it is the only thing that matters:** *why production gets ~0 external
hits when a bench on the same server gets an exact one.* Every optimisation in this document is
worthless until that is answered, because a cheaper cache that still never hits is still never
hitting. **This is the pivot of the whole closeout.**

### R3.14.1 CORRECTION — the group misclassification is not the hit-rate suspect

R3.12.6 listed it first among the candidates and R3.13.5 ordered the build around it. Reading
every `is_eagle_group` use site rather than the two I had seen, that ordering is wrong:

* **Store side.** `storable_chunks` (:344-370) does hold back the trailing chunk, but only
  while `is_decoding`, and `advance_stored_idx` (:372-382) wraps it in a `max()` precisely so
  the index cannot move backwards. The "permanent hole breaks prefix-reuse lookup" failure the
  code comments warn about (:357-361) is a bug upstream already guarded against. It does not
  happen here. The one-chunk lag during decode is recovered on the next turn, when that range
  is prefill.
* **Load side.** There is a real defect: `_lookup` (:702-706) extends `query_max` by one chunk
  to compensate for the eagle pop **only for sliding-window groups**. Our two full-attention
  groups are misclassified as eagle, get no compensating extension, and then have
  `num_hit_chunks -= 1` applied anyway (:740). So every lookup is short by one chunk on g6/g7.

That is a **1,648-token haircut, bounded and constant** — not a mechanism that produces zero.
The misclassification stays on the list as a correctness fix and as a hard prerequisite for
R3.13 (it would otherwise make `reachable_tail` 2 instead of 1 and halve the volume win), but
it is **demoted out of the diagnosis**.

### R3.14.2 PHASE A — the production hit autopsy (do this first; everything else is downstream)

Four mechanisms can produce ~0 external hits, and no measurement so far distinguishes them:

| # | mechanism | what it would look like |
|---|---|---|
| A1 | **Retry livelock** (R3.3) | `_maximal_prefix_lookup` (:551-583) turns any `RETRY` into `defer_lookup`, `_lookup` returns `None`, the request is delayed rather than served. If the fs tier answers `RETRY` while it stages, every lookup defers and none ever lands. |
| A2 | **Cascade latency vs think time** | The 83 s fs→CPU staging is longer than a user's pause between turns, so the chunks the next turn needs are not queryable yet. |
| A3 | **Hit window below one chunk** | `_lookup` returns 0 whenever `max_hit_size_tokens - num_computed_tokens < tokens_per_chunk`. With the GPU prefix cache already serving 11.5%, the *remainder* an external lookup is asked for may routinely be under 1,648 tokens. |
| A4 | **The 1,648-token haircut** (R3.14.1) | Only matters if it is the difference between clearing the one-chunk floor and not — i.e. it is A3's accomplice, not a cause on its own. |

**The measurement.** `_maximal_prefix_lookup` already loops over a four-way
`LookupResult` (`HIT` / `HIT_PENDING` / `RETRY` / `MISS`), and `_lookup` has exactly two early
`return 0` sites, both the "less than a chunk" test. Count them. A counter on each of those six
outcomes, exported like any other offload metric (the five-hop recipe is in R2.9.1), then run
**normal production traffic for a day with no bench touching the box** and read the histogram.

**Exit criterion.** The distribution names the mechanism outright:

* mostly `RETRY` → A1, fix the livelock;
* mostly `MISS` on chunks we know were written → A2, and the fix is a residency/timing change,
  not a volume change;
* mostly the "less than a chunk" `return 0` → A3, and the honest conclusion is that this
  workload's misses are *smaller than the cache's granularity*, which no amount of R3.13
  repairs;
* mostly `HIT` → the hits are real and the ~0 is a **metrics** problem, not a cache problem.

**Cost:** one instrumented restart, no GPU window (production traffic is the experiment).
**This is the cheapest phase in the document and it should have been first.**

#### R3.14.2a STATUS — Phase A is instrumented and running (2026-09-07)

`patch_kv_offload_lookup_outcomes.py` is live, wired into the launcher after the two existing
house patches, and confirmed applied at boot (all 13 hunks `OK`). Production came back up under
llama-swap at 20:21 with the normal configuration: `/health` 200, `GPU KV cache size: 228,737
tokens`, fs tier on `/kvcache/blocks`, zero errors, no orphan container.

**Two corrections to the measurement as specified above.** Writing it revealed that the plan
under-counted the instrumentation in two places, and both mattered:

* `_maximal_prefix_lookup` is *not* the only backend lookup site. `_sliding_window_lookup` calls
  `manager.lookup` too, and that is the path the **six Mamba/GDN groups** take. Counting only the
  prefix site would have measured two groups of nine and reported it as the whole picture. Both
  sites now call one shared `_count_lookup_result` helper.
* `_lookup` has **three** `return 0` sites, not two, and they are three different stories: a
  pre-query bail (`skip_short_window` — the window was under a chunk before we even asked), a
  zero-hit (`skip_zero_hit` — we asked and the backend had nothing), and a post-query bail
  (`skip_short_result` — we asked, got something, and it was still under a chunk). Conflating them
  would have made A3 unreadable, because only the pre-query one means "the request never reached
  the cache" while only the post-query one means "the cache had less than we needed".

Twelve counters in all, unlabelled, plus both `return None` defer sites and the served path, so
every exit from `_lookup` is now accounted for. The invocation is non-fatal by design: a
diagnostic must never keep the engine from serving, and the hunks are ordered so a mid-way
failure still leaves a consistent file.

**Reading it:** `vllm/kv-cache/phasea-read.sh` (add `--save NAME` for a JSON snapshot). The
counters **reset on every container restart**, so the reader prints engine uptime beside them and
declines to call a verdict under 50 lookups. A counter with no series has never fired — that is a
result, not a gap, and the reader prints it as 0.

**The one thing that can spoil this: any bench traffic on the box.** Phase A measures *production*
lookups. A tierbench, an equivbench or a probe run before the reading lands its own lookups in the
same counters and there is no way to separate them afterwards. Leave the box alone until the
histogram is read; if a bench does run, restart the engine first and start the day again.

The warm request at boot already produced `calls=1, skip_short_window=1` — a 10-token prompt
obviously has under 1,648 tokens to fetch, so that is proof of wiring, **not** evidence for A3.

#### R3.14.2b PHASE A RESULT — **A1, and the trigger is HIT_PENDING, not RETRY** (2026-09-07, 49 min of production traffic)

It did not need a day. 11,878 production lookups in 49 minutes were enough, and the distribution
is not close:

| outcome | count | share |
|---|---:|---:|
| **deferred at the backend** | **11,849** | **99.75%** |
| served | 3 | 0.03% |
| skip zero hit | 13 | 0.11% |
| skip short result | 12 | 0.10% |
| skip short window | 1 | 0.01% |

**The three A3 counters together are 26 of 11,878.** A3 is dead. So is A2: only **25** requests
ever engaged a secondary tier at all (`kv_offload_tiering_lookup_*_count = 25`), and when they
did they resolved in ~0.96 s mean. The disk tier is not the problem — it is barely in the story.

**But the label A1 carried in R3.14.2 was wrong, and the correction is the whole finding.** The
per-chunk results:

| chunk result | count |
|---|---:|
| HIT | 575,335 |
| **HIT_PENDING** | **1,579,971** |
| MISS | 41,535 |
| RETRY | 3,869 |

`HIT_PENDING` outnumbers `RETRY` **408 to 1**. This is not a retry livelock. Chasing "location
uncertain" would have been chasing 0.24% of the signal.

**The mechanism, end to end.** `cpu/manager.py:125` returns `HIT_PENDING` when
`not block.is_ready`, and `is_ready` is `ref_cnt != -1`; `policies/base.py:25` initialises every
block to `ref_cnt = -1` and only `complete_store` clears it to 0. So **`HIT_PENDING` means exactly
one thing: a block whose GPU→CPU store transfer has not landed yet.** Then:

1. `tiering/manager.py:321` short-circuits on a primary `HIT_PENDING` and returns immediately —
   it never consults the secondary tiers. One not-yet-written CPU block **masks the whole disk
   tier**.
2. Both lookup loops set `defer_lookup = True` on `HIT_PENDING` (`scheduler.py:593` and `:625`)
   and then **discard their hit count and return `None`** — `return hit_count if not defer_lookup
   else None`. The 575,335 HITs already found are thrown away.
3. `_lookup` returns `None`, the request is delayed, and on the next scheduler step it is looked
   up again — by which time the engine has served other requests and started *new* stores, so
   there are fresh not-ready blocks, so it defers again.

That is the livelock: **production stores continuously, so there is never an instant with no
in-flight write, so no lookup ever comes back clean.** 61.1 GB was stored this boot.

**And this retires the oldest puzzle in the document.** Every bench we ran hit — the 85,696-token
exact hit of R3.10/R3.11, reproducibly. Every one of them ran on a *quiet* box: one request, a
settle, nothing else writing. A quiet box has no in-flight stores, therefore no `HIT_PENDING`,
therefore no defer, therefore a clean serve. **Production never gets a quiet instant.** The bench
and production were never measuring the same system, and no amount of bench work would ever have
found this. Only production traffic could.

**Cross-validation that the instrumentation is sound.** The five terminal counters sum to exactly
`calls` (11,849 + 3 + 13 + 12 + 1 = 11,878), so every exit is accounted for and none double-counts.
Independently, `lookup_served_tokens` matched vLLM's own
`prompt_tokens_by_source{source="external_kv_transfer"}` **to the token**, at two readings taken
an hour apart (113,712, then 168,096). The counters are measuring what they claim to.

**What this means for the rest of the plan.** R3.13's store-policy patch — the 9→3.75 group-files
cut — is **not** what stands between us and a hit rate, and Phase C must not be started on the
old rationale. It would reduce store *volume*, and less volume does mean fewer in-flight writes,
so it is now weakly relevant for a second-order reason rather than the headline one. The primary
fix is in the lookup path itself, and it is small: a lookup that has already found hits should be
able to **serve the ready prefix it found** instead of discarding it because a later chunk is
mid-write. That is Phase B, and it is a much smaller change than anything R3.13 proposed.

**Snapshot:** `$HOME/bench-history/phasea-20260907-46min/counters.json`.

### R3.14.3 PHASE B — BUILT: serve the ready prefix instead of deferring

Phase A named the fix precisely enough that Phase B did not need designing, only writing.
The patch is `<repo>/kv-cache/patch_kv_offload_serve_ready_prefix.py`, wired into the
launcher after the Phase A instrumentation, gated on `RADIANCE_OFFLOAD_PENDING_IS_MISS`
(launcher knob `KVOFF_PENDING_IS_MISS`, default **1** = the new behaviour).

**The change, in one sentence.** `HIT_PENDING` stops meaning "wait for this block" and starts
meaning "this block is not readable yet" — which is what `MISS` already means to both lookup
loops. Nothing else moves.

Concretely, in `offloading/scheduler.py`:

| site | groups | upstream on HIT_PENDING | patched |
|---|---|---|---|
| `_maximal_prefix_lookup` | 6, 7 (full attn) + 8 (MTP) | `hit_count += 1`, `defer_lookup = True`, ultimately `return None` | `break` — the prefix ends here and the confirmed-ready count is returned |
| `_sliding_window_lookup` | 0–5 (Mamba/GDN, window = 1 chunk) | `consecutive_hits += 1`, `defer_lookup = True`, `return None` | `consecutive_hits = 0` — the backwards scan keeps going and finds an **older** state that is ready |

The Mamba half is the one that matters most and it is the less obvious of the two. Those six
groups need one specific state, and they scan **backwards from the newest chunk** — which is
precisely the chunk most likely to have been written seconds ago and still be in flight. Upstream
answers "wait". Truncating instead lets the scan reach an older state that is genuinely readable:
a shorter hit, but a real one, where today we take neither.

**Behaviour verified before deployment, not after.** The two loop bodies were lifted out of the
patched file by `ast` and run against hand-built chunk sequences, so the control flow under test
is the shipped control flow rather than a paraphrase of it:

| chunk sequence | upstream prefix / window(1) | patched prefix / window(1) |
|---|---|---|
| `HIT HIT HIT HIT` | 4 / 4 | 4 / 4 — unchanged |
| `HIT HIT PENDING HIT` | **None** / 4 | **2** / 4 |
| `PENDING HIT HIT HIT` | **None** / 4 | **0** / 4 |
| `HIT HIT HIT PENDING` | **None** / **None** | **3** / **3** |
| `HIT HIT MISS HIT` | 2 / 4 | 2 / 4 — unchanged |
| `HIT HIT RETRY HIT` | None / 4 | None / 4 — unchanged, RETRY still defers |

Row 4 is production's case, and it is the row where upstream loses on both paths at once.

**RETRY is deliberately left alone.** It is a genuine "call me again, I have started async work",
it is 0.24% of chunk results, and killing its defer would break the disk tier's promotion
handshake. Only `HIT_PENDING` changes.

**Why this cannot serve wrong data.** Both loops now report **fewer** chunks than upstream, never
more, and every chunk they do report returned `HIT` — ready, `ref_cnt >= 0`, the same condition
the existing serve path already demands. Upstream never serves a pending chunk either; it waits
for one. So the patch changes how much of a hit we take, not whether the bytes are valid, and
R3.11 already established those bytes are bit-identical to a cold recompute. The one-chunk floor
in `_lookup` is untouched, so a truncation that leaves less than 1,648 tokens still returns 0 and
costs no load.

**One thing that looked like a second bug and is not.** `tiering/manager.py:321` short-circuits on
a primary `HIT_PENDING` and never asks the disk tier, which reads like the disk tier being masked.
It is not worth fixing: a secondary-tier `HIT` does not serve the block, it *starts a promotion*
into the primary tier and returns `RETRY`. The block is already inbound to the primary, so
consulting disk would only schedule a redundant copy of a write already in flight. **Leave it.**
This corrects the second half of R3.14.2b's mechanism description — the short-circuit is real, its
consequences are not.

**Measurement.** `<repo>/kv-cache/phaseb-read.sh` compares the run against the Phase A snapshot
as a *share of lookups*, because the windows differ. It refuses to report anything if
`vllm:kv_offload_lookup_pending_truncated` is not registered — an unapplied patch and an applied
patch that never truncated would otherwise look identical, and they are opposite conclusions.

**Exit criterion.** Deferrals fall well below 99.75% **and** serves rise. If deferrals fall but
serves do not, the ready prefixes are landing under the one-chunk floor — that is R3.14.6's
territory, not more of this. If nothing was truncated, the box was too quiet and the run is not
a result.

**Reverting** is `KVOFF_PENDING_IS_MISS=0` and a restart. The patch stays applied; every branch
returns to upstream behaviour. This is an A/B, not a one-way door.

### R3.14.4 PHASE C — the store-policy patch (R3.13), gated on Phase A

Only worth building if Phase A shows hits are reachable. Full build sheet in R3.13; the order
within it stands, with the misclassification fix first as a prerequisite rather than as a
diagnostic. Target N=8: 9 → 3.75 group-files per chunk (42% of volume) and, the real point,
the CPU primary from 116,459 to 279,501 tokens — **1.22x the GPU cache**, over the threshold
where RAM stops being a staging buffer and starts being a cache (R3.12.3).

**Exit criterion:** the CPU tier serves a hit. Not "writes less" — *serves a hit*, measured the
way R3.10 measured the disk one.

### R3.14.5 PHASE D — prove it and close

1. `equivbench.py` (R3.11) against the final configuration. A coarser GDN snapshot is exactly
   the class of change that stays fast and quietly stops being exact, and the R3.11 method has a
   **zero** noise floor, so any drift at all is a real regression.
2. Settle the one open question from R3.13.4: is the mamba align clamp external-hit-only, or
   does it reach the GPU prefix-cache path? External-only means the ~3.4 s rounding is paid
   only where we would otherwise have recomputed for 45.8 s. If it reaches the GPU path, it is
   paid on every turn and N=8 is the wrong answer.
3. End-to-end: median and p95 time-to-first-token on real multi-turn conversations, against the
   same measurement with offloading off. **That number, not the hit counter, is the deliverable.**

### R3.14.6 The stopping conditions — what "closed out" is allowed to mean

Three legitimate endings, and they should be treated as equally acceptable:

1. **It works.** Phase A finds a fixable mechanism, Phase C lands, the RAM tier serves hits,
   p95 TTFT on long conversations improves. Ship it, update the entry, close the document.
2. **It works but is not worth it.** Hits become real but the end-to-end win is inside the
   noise, or the align rounding turns out to be charged on every turn. Then the correct action
   is to **keep the CPU tier, drop the fs tier**, and say so — the 83 s disk path was never
   competitive with a 45.8 s recompute anyway (R3.12.4).
3. **The workload is wrong for it.** Phase A returns A3: production's misses are smaller than
   one 1,648-token chunk, so the cache can never engage. Then **turn offloading off**, reclaim
   the 16 GiB of `/dev/shm` and the 328 GB of `/kvcache`, and record the geometry finding —
   a 1,648-token block is what makes this architecture's offload granularity coarse, and that
   is a property of the model, not of our configuration.

Ending 3 is not a failure. Ruling a lever out on measurement is the same value as adopting one,
and this document already has five of them ([[batchtok-spectok-drafter-tile-closed]],
[[mrv2-rejected-fastokens-noop]], [[radiance-fast-draft-closed]], and the two corrections in
R3.12).

### R3.14.7 Explicitly not doing

Unchanged from R3.10.7 and restated so it is in one place: **no new disks** (the device is
exonerated, R3.9.2), **no CPU-tier resizing** (16 GiB is already at the host's limit, and
R3.12.3 shows the fix is fewer bytes per token, not more bytes), **no attention-only offload**
(it breaks resume correctness for a hybrid model), **no cascade scheduling work** (R3.10.2
withdrew the premise), and **no `blocks_per_chunk` changes** (R3.12.5a).

## R3.15 — The offload boundary crash: two defects, both stock (2026-09-10)

**Status: fix written and dry-run applied; NOT yet run against traffic.** Supersedes the
"replace the blanket decline with a targeted boundary guard" note in R3.10.7 and the
"production runs upstream until the dump names the group" decision in R3.12.6. The dump
named it.

### R3.15.1 What crashed

Twice, and only twice, both with `RADIANCE_OFFLOAD_MIXED_HIT=1`:

    2026-09-08 07:31:44   group=8  local_tokens=24720  boundary=14  n_blocks=16
    2026-09-10 07:57:10   group=8  local_tokens=11536  boundary=6   n_blocks=8

`AssertionError` in `offloading/scheduler.py:update_state_after_alloc`, raised inside
`Scheduler.schedule()`, fatal to EngineCore. llama-swap reloaded both times.

Every field is invariant across the two: `group=8`, `tokens_per_block=1648`,
`tokens_per_chunk=1648`, `swa_chunks=2`, `align_chunks=None`, `external_tokens=1648`
(exactly one chunk), `local_tokens=(n_blocks-1)*1648`, `boundary=n_blocks-2`, and a block
pattern of `n_blocks-2` nulls followed by two `(is_null=False, block_hash=None)` blocks.

Group 8 is the MTP/DFlash2 draft group and it is a `SlidingWindowSpec` group —
`sliding_window_size_in_chunks=2` can only come from `get_sliding_window_size_in_chunks`'s
`SlidingWindowSpec` arm (mamba returns 1, full attention returns `None`).

### R3.15.2 Why it happens — the chain, all of it stock 0.27.1

1. `Scheduler.schedule()` calls `get_computed_blocks_for_connector()` rather than
   `get_computed_blocks()` whenever a connector is attached and the model has mamba
   layers. That helper **deliberately does not reconcile** the per-group hits: it calls
   `find_longest_cache_hit_per_group()`, reports the **full-attention** group's hit as the
   request's local hit, and returns `hit_diverged = min(per_group_hits) < num_local`. Its
   docstring says why — "the connector transfers the remaining suffix".
2. So `num_locally_computed_tokens` is the full-attention hit (11536 = 7 blocks) while a
   lagging group can be resident for less. `SlidingWindowManager.find_longest_cache_hit`
   pops one more block for the eagle drop, so group 8's own hit is 6 blocks.
3. `hit_diverged` is only reconciled away when the connector finds **no** external tokens.
   It found 1648, so the diverged hit stands. By design.
4. `add_local_computed_blocks` pads group 8 with 6 nulls and adds nothing;
   `allocate_external_computed_blocks` allocates `cdiv(13184,1648) - 6 = 2` fresh blocks at
   indices 6 and 7. That is the dumped pattern exactly.
5. The assertion says "every locally computed token is resident below the boundary". For a
   lagging window group that is false — which is the entire point of step 1.

The assertion is a leftover invariant from before divergent lookups existed. It is correct
for full-attention groups, whose hit *defines* the boundary, and wrong for every group that
is allowed to lag.

### R3.15.3 The second defect — why deleting the assertion is not the fix

`_lookup()` sets, for every group,

    start_chunk_idx = num_computed_tokens // tokens_per_chunk

and only ever confirms chunks at or above it. `update_state_after_alloc` loads from the
group's **own** block boundary:

    start_chunk_idx = num_locally_computed_gpu_blocks // blocks_per_chunk

In the 2026-09-10 crash the lookup confirmed chunks 7 and 8 for group 8; the load would
have asked for chunks 6 and 7. Chunk 6 was never confirmed present.
`OffloadingManager.prepare_load` is documented "callers only pass keys already confirmed
HIT by lookup() earlier this step" and enforces it:

    assert block is not None, f"Block {key!r} not found in cache"

So removing the assertion alone converts one EngineCore kill into another, hit whenever the
tier has evicted the gap chunk. **Checked explicitly, because this rig has been burned by
silent wrong output before** — it is an assert, not a silent read of stale bytes. But it is
still a crash, and it is why the fix has to touch the lookup and not just the assertion.

### R3.15.4 The fix

`patch_offload_mixed_hit.py`, rewritten. Two behaviour hunks:

* **hunk 3 (the real fix)** — for groups with a window (`SlidingWindowSpec`, and mamba,
  which reports a window of 1 chunk), start the suffix scan low enough that a full window
  ending at the top of the query range can be found:

      start_chunk_idx = min(start_chunk_idx, max(0, num_chunks - required_window))

  The run length (`required_window`) is untouched, so hit semantics are unchanged for any
  group that is not lagging, and the returned index still converts to absolute chunks
  through the same `start_chunk_idx`. What changes is that the chunks the connector will
  load are now the chunks it confirmed. If the gap chunk is missing the run resets and the
  request gets no external hit — a lost hit, never an unbacked load.

* **hunk 4** — scope the boundary assertion to full-attention groups, keeping the one-shot
  diagnostic dump so a genuinely new geometry still reports numbers.

Hunk 2 (decline every mixed hit) is retained as a kill switch and is now **off** by
default; `RADIANCE_OFFLOAD_MIXED_HIT` flips from default 0 to default 1.

**Why letting the load proceed is safe.** The destination blocks come from
`allocate_external_computed_blocks`: freshly allocated, refcount 1, owned by this request,
so nothing shared is written. The range is bounded by the assertion immediately below the
one being relaxed —

    num_pending_gpu_blocks <= sliding_window_size_in_chunks * blocks_per_chunk + 1

— and a window group's `get_num_skipped_tokens` guarantees the pending range is at most its
window in chunks, which is exactly the range hunk 3 now confirms. The two bounds are the
same bound, which is what makes this a pairing rather than two independent guesses.

**Blast radius in this configuration.** With `blocks_per_chunk=1` and
`RADIANCE_MAMBA_STORE_STRIDE=8`, the six mamba groups compute
`min(start, max(0, num_chunks - 1))`, which is `start` — unchanged. The two full-attention
groups are unchanged by construction. Only group 8 scans differently, and only by one
chunk.

### R3.15.5 What is NOT yet proven

* No traffic has run against the fix. Dry-run only: all seven house patches apply in boot
  order after `patch_kv_group_size.py`, and `scheduler.py` compiles.
* No stock reproduction exists, so this cannot be filed upstream yet. Both defects are in
  stock 0.27.1 and neither is model-specific — they need a connector, a lagging group, and
  an external hit landing on a request that also hit the GPU prefix cache. What this box
  supplies that upstream CI does not is traffic that actually reaches the read-back path.
  `patch_kv_group_size.py` and `patch_kv_offload_eagle_groups.py` change *which* group lags
  and how often, not whether the invariant holds: the eagle pop in `SlidingWindowManager`
  and the divergent lookup in `get_computed_blocks_for_connector` are both stock.
* Acceptance, when a window is available: (a) no `offload boundary` line in the log across
  a run that previously produced one; (b) external prefix cache hit rate stays in the
  3–19.7% band, i.e. the wider scan did not decline hits it used to serve; (c) a served
  mixed hit is bit-identical to a cold recompute, the R3.11 check, re-run on a request that
  hits both tiers.

### R3.15.6 How it gets tested — `check-r315-boot.sh` then `mixedbench.py`

The patches are applied at container start from the host directory mounted at `/house`, so
a reload picks the fix up with no rebuild — and a stale mount, a failed hunk or a launcher
that skipped a step all look identical from outside. `check-r315-boot.sh` closes that gap
first: it reads the RUNNING engine's `scheduler.py` through `/proc/<pid>/root` and confirms
both hunks are physically present, plus the 9-group geometry, `RADIANCE_OFFLOAD_MIXED_HIT=1`,
and that nothing has fired yet this boot. It was validated against the unfixed engine before
being trusted — both hunk checks correctly reported MISSING, which caught two false passes:
"full-attention group" matches four *stock* upstream comments, and "offload boundary" matches
the patch-status line `OK  offload boundary diagnostics`. Both markers are now anchored on
text that exists only in the patched file.

`equivbench.py` cannot test this fix. Its `fs` phase evicts the entire cache, so the probe
that follows has `num_computed_tokens = 0` — a pure external hit, which never reaches the
divergent-hit path. `mixedbench.py` builds the mixed case on purpose, using upstream's own
reverse-order block free (the tail of a prompt leaves the GPU before its head, to preserve
shared prefixes):

    coldA   fresh cache_salt        -> recompute, reference answer
    coldB   fresh cache_salt        -> recompute, identical text: the noise floor
    warm    coldB's salt again      -> full GPU hit, then drain so the tier holds it too
    evict   novel traffic, stepwise -> eats the free queue front-first, so the prompt's TAIL
    mixed   coldB's salt again      -> head from GPU, tail from the tier

The eviction volume is found, not predicted — it depends on what was in the pool before the
run — so the sweep probes after each step and stops at the first request showing GPU hits and
external hits together. It exits non-zero unless a mixed hit was actually built, the engine
survived it, and the tokens matched both recomputes; "no mixed hit built" is reported as a
failure rather than a pass, because a fix that quietly stopped serving mixed hits would
otherwise look clean.

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
