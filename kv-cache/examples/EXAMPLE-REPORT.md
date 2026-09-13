# Example: what a working cache looks like

A real report from this box, over ~14 hours of agent coding work on one R9700, on the
**experimental** build (GPU → RAM → disk, instrumented) — the run the README's prompt-token shares
come from. The raw `/metrics` snapshot it was generated from ships beside it as
[`metrics-snapshot-20260912.txt`](metrics-snapshot-20260912.txt), so you can reproduce this exact
output without the hardware:

```bash
python3 kv-cache/tools/kvvalidate.py --markdown --metrics-file kv-cache/examples/metrics-snapshot-20260912.txt
```

Run it against your own endpoint with `python3 kv-cache/tools/kvvalidate.py --markdown` and compare. That
needs the experimental build too: on the default build the counters it compares are not exported,
so it reports FAILs that are not real. Use `watch -n 5 python3 kv-cache/tools/kvwatch.py` there instead.

> **Read the regime line first.** This report says `saturated` — the RAM tier is full and the disk
> tier is serving. A freshly booted engine reports a warm-up regime instead, where the disk tier
> *cannot* serve until the RAM tier fills; for the first ~45 minutes it looks dead when it is
> merely cold. Comparing your warm-up numbers against a saturated report will tell you something
> false. The tool names the regime before it shows you a figure, for exactly this reason.

---

## KV-cache offload report — PASS (2 checks not run — no data)

`2026-09-12T17:41:36+1000` · regime **saturated** (running 1.0, waiting 0.0)

### Invariants

| | check | result |
|---|---|---|
| PASS | `lookup_partition` | served 7686 + zero_hit 429 + short_window 4012 + short_result 0 + deferred_backend 273208 + deferred_loading 30 = 285365 vs lookup_calls 285365 (drift +0, bound 32 for 1 active) |
| PASS | `cpu_equals_external` | cpu_hit_tokens 5570240 == external_kv_transfer 5570240  (external_prefix_cache_hits=5570240) (drift +0, bound 4096 for 1 active) |
| PASS | `promotion_refused` | refused 0 (metric absent = 0) / initiated 5423  [no_evictable=0, protected=0] |
| UNKNOWN | `disk_vs_engine` | not run: metrics were read from a saved snapshot, so the on-disk store it describes is not the live filesystem |
| UNKNOWN | `group_ratio` | not run: metrics were read from a saved snapshot, so the on-disk store it describes is not the live filesystem |

### Where prompt tokens came from

| source | tokens | share |
|---|---:|---:|
| local_cache_hit | 20,550,560 | 64.5% |
| external_kv_transfer | 5,570,240 | 17.5% |
| local_compute | 5,749,824 | 18.0% |
| **total** | **31,870,624** | |

**82.0% of prompt tokens served without recomputation.** A rising `local_cache_hit` share is the GPU cache doing its job, not the tier failing -- read rates, not shares.

### Performance

Markers: `SOLID` directly measured · `REGIME` depends on the cache's current state · `UNRESOLVED` two derivations disagree, do not quote it as fact.

| tier | token rate | bandwidth | |
|---|---:|---:|---|
| fs | 3,868 tok/s | 228 MB/s | `SOLID` |
| cpu | 337,408 tok/s | 11,834 MB/s | `SOLID` |

_as a request sees it (submit->complete, includes the wait)_

- **bytes/token** (measured): fs 58,958 · cpu 35,073 `SOLID` — this is what the sizing guidance is built on.
- **wait fraction** 5.3% of 8 read threads `SOLID` — low wait: device-bound - a faster disk helps; more threads will not
- `REFUSAL` tier_load_latency_seconds_sum is byte-identical to tier_load_thread_seconds_total (thread-summed, 7.58x overstatement of wall clock) - no rate is derived from it

---

Produced by `tools/kvvalidate.py --markdown --metrics-file metrics-snapshot-20260912.txt` — a saved `/metrics` snapshot, so the two on-disk invariants are reported as not run.
