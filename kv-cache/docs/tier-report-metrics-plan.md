# Tier report — the metrics to patch in

**Goal:** a script anyone can point at a running vLLM server with KV offload enabled, which
prints a table justifying **the size and speed of each cache tier** and answers three questions
in plain language:

1. *Have I allocated too much (or too little) RAM to the CPU tier?*
2. *Is my disk too slow — should I buy NVMe?*
3. *Is turning this layered cache on adding value at all?*

Output shaped like a BetterBench run: `results.json` + `results.md`, scriptable, no human
interpretation required to reach a verdict.

---

## 1. Design principles (these decide the whole patch list)

**1.1 Lifetime-cumulative, read once, from the operator's own traffic.**
The user runs this against their real workload, not a replay. A synthetic corpus cannot answer
"is my RAM the right size" because the answer depends entirely on *their* prefix-reuse pattern.
So the tool scrapes `/metrics` once and reports. No GPU window, no restart, no A/B, no corpus.

**1.2 Therefore: counters and histograms only. No gauges.**
This follows directly from 1.1 and it is the single most important constraint here. A gauge is
an instantaneous sample; scraped once from a server that has been up for three weeks it carries
almost no information. Today's tier instrumentation is mostly gauges and is nearly useless for
this purpose:

| Existing | Type | Why it can't feed the report |
|---|---|---|
| `kv_offload_cpu_cache_free_perc` | gauge | reads `0.0` — says "full now", not "full 90% of the time" |
| `kv_offload_cpu_cache_evictable_perc` | gauge | same |
| `kv_offload_fs_inflight_jobs` | gauge | no bytes, no time, no history |
| `num_requests_waiting_by_reason` | gauge | the reason split is right, the type is wrong |
| `kv_cache_usage_perc` | gauge | occupancy caps at 100%; demand *above* capacity is invisible |

Anything the report depends on must be a monotonic counter or a histogram. Where a level
genuinely matters (occupancy, working set), expose the **distribution over time** as a
histogram, not the current value.

Static configuration (tier capacity) is the one exception — it does not vary, so a gauge or an
`_info` metric is fine. But every **judgment** must come from counters.

**1.3 Optional two-snapshot mode, for people who are testing.**
Our own engine counters are skewed right now because we ran a correctness suite that
deliberately forces eviction. That is a testing artifact, not the normal case, so it must not
drive the design — but a `--since <snapshot.json>` flag that differences two scrapes costs
nothing and makes the tool usable while benchmarking. Default stays lifetime.

**1.4 Every column must be derivable, not hand-assembled.**
If a number in the table cannot be computed from scraped counters, either a patch is missing or
the column goes.

---

## 2. The target table, column by column

| Tier | Size | Hit ratio | Fetch latency | Equivalent tok/s | Capacity (tokens) |
|---|---|---|---|---|---|

| Column | Derivation | Status |
|---|---|---|
| **Tier** | `gpu` / `cpu` / `fs` / `recompute` | — |
| **Hit ratio** | `prompt_tokens_by_source{source}` ÷ `prompt_tokens_total` | **P1** — `external_kv_transfer` is one bucket for CPU+FS |
| **Fetch latency** | `kv_offload_load_seconds{tier}` histogram → p50/p99 | **P2** — no per-tier timing exists |
| **Equivalent tok/s** | `load_tokens{tier}` ÷ `load_seconds{tier}` | **P2** |
| **Size / Capacity** | `tier_capacity_bytes{tier}`; tokens = ÷ measured density | **P3** — fs exposes nothing; CPU only percentages |
| **Recompute baseline** | `request_prefill_kv_computed_tokens` ÷ (prefill − stall) | **P5** — prefill time includes KV-load stalls |

KV density (bytes/token) does **not** need a patch or a hard-coded constant: with P2 it is
`load_bytes{tier} ÷ load_tokens{tier}`, measured per tier, per deployment, on any model.

### What "Fetch latency" and "Equivalent tok/s" actually are on a queued tier

As built, the timing wrapper times each pool *task* and adds the durations up, so for a
tier whose I/O is dispatched to workers — `fs` here, 8 read threads — `load_seconds` is
**thread-time summed, not wall time**. Two consequences, both now stated by the tool
rather than papered over:

- **`Equivalent tok/s` is a lower bound**, understated by up to the pool width. `tierreport.py`
  prints it with a `≥`, brackets it against the p99 batch (the transfer cannot have taken
  less than its slowest single batch), and refuses to answer "too slow?", "actively
  hurting?" or "where is the ceiling?" from it. `--serial-io` turns that off for a backend
  that genuinely reads inline.
- **Latency is a *batch* service time, not a request stall.** The column is named
  `Batch p50` / `Batch p99` for that reason. This tier **queues deliberately rather than
  preempting and evicting**: a batch waiting while its siblings share the device is the
  fanout doing its job. A request waits the *promotion*, which overlaps those batches —
  and that number is measured separately, as `prefill_stall_seconds{tier}`.

The fix on the patch side is to time the whole promotion once instead of each batch. It
needs a model reload, which resets these lifetime counters, so it is queued for the next
natural restart.

---

## 3. The patches

### P1 — split the source label — **HIGHEST**

`vllm:prompt_tokens_by_source_total{source="external_cpu"|"external_fs"}`

Today `source` has three values (`local_compute`, `local_cache_hit`, `external_kv_transfer`)
and they sum exactly to `prompt_tokens_total`. The split is exact and already trustworthy —
cross-checked against byte-division to within 2%. The one flaw is that **both offload tiers
share one bucket**, so the report cannot say whether the RAM tier or the disk tier did the work.

That is the hit-ratio column, and it is the column the entire sizing argument rests on. One
label, and it makes "my disk tier is dead weight" visible for the first time.

**Caveat to encode in the tool:** with a promoting tier design, a block that originated on disk
is staged into CPU and loaded from there. `external_cpu` must mean "served from CPU" and a
separate counter must attribute *staged-from-fs* volume, or disk gets credited as RAM. Emit
both: served-from and originated-from.

### P2 — per-tier bytes, tokens and latency on load/store — **HIGHEST**

```
vllm:kv_offload_load_bytes_total{tier}
vllm:kv_offload_load_tokens_total{tier}
vllm:kv_offload_load_seconds{tier}          # histogram
vllm:kv_offload_store_bytes_total{tier}
vllm:kv_offload_store_tokens_total{tier}
vllm:kv_offload_store_seconds{tier}         # histogram
```

Today `load_bytes_total` and `load_time_total` carry **no labels at all**. Aggregate
bandwidth comes out at 11.8 GB/s, which is CPU-tier speed — the disk contribution is invisible
inside it, so a slow disk cannot be detected from the aggregate.

Histograms rather than counters for the timing, because **the mean hides the tail that actually
hurts**. An operator with a p50 of 40 ms and a p99 of 9 s has a problem the mean will never show.

Unlocks: fetch-latency column, equivalent-tok/s column, KV density, and the store-side tax.

### P3 — tier capacity and occupancy — **HIGH**

```
vllm:kv_offload_tier_capacity_bytes{tier}      # static config
vllm:kv_offload_tier_used_bytes{tier}          # current
vllm:kv_offload_tier_occupancy_ratio{tier}     # HISTOGRAM over time, per 1.2
```

The fs tier currently exposes no size information whatsoever, and the CPU tier exposes only
percentages — which cannot be converted to tokens or GiB without knowing the denominator. The
occupancy *histogram* is what lets the report say "your RAM tier was above 95% full for 80% of
the time", which is a sizing statement; the current gauge cannot.

### P4 — the sizing verdict family — **HIGH, and this is the one that makes the tool worth running**

```
vllm:kv_offload_block_reads_before_evict{tier}   # histogram, buckets 0,1,2,4,8,16+
vllm:kv_offload_evictions_total{tier}
vllm:kv_offload_evicted_bytes_total{tier}
vllm:kv_offload_eviction_to_reuse_seconds{tier}  # histogram
```

These two histograms answer the RAM question *directionally*, which no occupancy number can:

- **`block_reads_before_evict` with mass at 0** — you are storing blocks nobody ever reads back.
  The tier is **oversized**, or the store policy is too eager. Shrink it, or spend the RAM elsewhere.
- **`eviction_to_reuse_seconds` with mass at the low end** — you are evicting blocks that are
  requested again seconds later. The tier is **undersized**. This is thrashing, measured
  rather than inferred.

Both at once means the tier is the wrong *shape*, not the wrong size — churning on cold data
while hot data gets evicted, which points at the eviction policy.

This is the generalisation of the "would-have-hit" counter (`kv_offload_lookup_miss_evicted_total`,
already specified) from the fs tier to every tier. It converts a storage purchase from an
argument into a measurement.

### P5 — separate prefill compute from prefill stall — **MEDIUM-HIGH**

`vllm:kv_offload_prefill_stall_seconds_total{tier}` — time a prefill spent **blocked** waiting
on a KV load.

`request_prefill_time_seconds` today includes both computing and waiting on the tier. That
makes the recompute baseline (the denominator of every break-even calculation) wrong, and it is
also why an earlier analysis here mistook async lookup delay for blocking time. With this,
recompute rate = `computed_tokens ÷ (prefill_time − stall_time)`, honestly.

It also directly prices the tier's downside: **stall time is what a slow disk costs you**, and
it is the number that says "your HDD is not just failing to help, it is actively hurting."

### P6 — would-have-hit — **MEDIUM** (already specified; restated for completeness)

`vllm:kv_offload_lookup_miss_evicted_total{tier}` — a lookup that missed for a block this tier
previously held and evicted. Distinguishes *"never cached"* from *"cached and thrown away too
early"*. Only the second justifies buying capacity. Subsumed in spirit by P4's
`eviction_to_reuse` histogram, but cheaper to implement and worth having on its own.

---

## 4. What the tool computes (no patch needed)

**4.1 Device calibration.** Read a stored block file straight off the fs tier and time it. Gives
the raw device rate independent of the engine, so the report can say "your device does 117 MB/s;
break-even is 101 MB/s" without trusting anything vLLM reports.

**4.2 Break-even per tier.** `tier_rate` vs `recompute_rate`, both measured. A tier only pays if
delivery beats recompute:

```
break_even_ratio(tier) = tier_tok_per_s / recompute_tok_per_s
```

Below ~1.2 the tier is not earning its place; the tool names the device rate that would fix it
and how much headroom NVMe would give.

**4.3 Value of the whole cache, counterfactually.** No A/B and no restart required:

```
saved  = Σ_tier tokens{tier} × (1/recompute_rate − 1/tier_rate)
cost   = Σ_tier store_seconds{tier} + unrecovered stall_seconds
net    = saved − cost
```

Every term is a cumulative counter after P1/P2/P5. This is the "yes, turning this on adds value"
answer, computed on the operator's own traffic, in one scrape.

**4.4 KV density** = `load_bytes{tier} ÷ load_tokens{tier}`. Never hard-code 34 KB/token; it is
model- and config-specific.

---

## 5. The verdicts the tool prints

The point of the exercise. Each is a threshold on a measured distribution, not a judgment call:

| Question | Signal | Verdict |
|---|---|---|
| Too much RAM? | `block_reads_before_evict{cpu}` mass at 0 > ~50% | "≈N GiB of your CPU tier is never read back — shrink it" |
| Too little RAM? | `eviction_to_reuse_seconds{cpu}` p50 < ~60 s | "you evict blocks that come back in N s — grow it" |
| Disk too slow? | `break_even_ratio{fs}` < 1.2 | "fs delivers N tok/s vs recompute M — at best marginal; NVMe would give ≈Z×" — **suppressed on a queued tier**, which gets the bracket and the reason instead |
| Disk actively hurting? | `prefill_stall_seconds{fs}` > time saved by fs hits | "your disk tier is a net loss — disable it or replace the device" — **suppressed on a queued tier**: both sides of the subtraction are thread-summed, so the stall is the only wall-clock figure |
| Is the cache worth it? | `net` from 4.3 | "saved H hours against T minutes of store overhead over this uptime" |
| Right shape? | mass at 0 reads **and** short eviction-to-reuse | "eviction policy problem, not a capacity problem" |

---

## 6. Build order

1. **P1 + P2** together — they are the table. Without them there is no per-tier row at all.
2. **P4** — the sizing verdict; the reason an operator runs this rather than reading a blog post.
3. **P3** — capacity/occupancy, completes the Size column.
4. **P5** — makes the break-even denominator honest and prices the slow-disk downside.
5. **P6** — cheap, overlaps P4.
6. Harness: scrape → derive → verdicts → `results.json` + `results.md`, `--since` for the
   testing case.

Existing code to build on: `tierbench.py` already does before/after metric differencing (that
becomes `--since`), and its `classify()` already reaches for a per-tier load counter that does
not exist yet — P2 makes it work as written.

---

## 7. Why this generalises (upstream framing)

None of the six patches encode anything about this model, this GPU, or this deployment. They
are labels and histograms on paths that already exist. The density constant is measured rather
than assumed, tier names come from the configured tier list, and the verdict thresholds are
ratios. A user on any vLLM + KV-offload deployment gets the same table.

That matters because the destination is upstream: this is instrumentation any operator of a
tiered KV cache needs, and the argument for it does not depend on agreeing with any conclusion
we have drawn from our own numbers.
