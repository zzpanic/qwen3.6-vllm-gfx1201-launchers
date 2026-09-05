# MXFP4 deep sweep — 2026-09-05

`qwen3.8-27b-mxfp4` (MXFP4 W4A8, radiance kernels, DFlash2 K=7, vision tower
loaded) at `MAXLEN=204800`, taken to nearly the full context window.

Everything here is the **MXFP4** stack. The int4 W4A16 numbers live in
`../int4-20260829/` and are **not** a baseline for these — four axes moved
between them (see that folder's README).

## Files

| file | what it is |
| --- | --- |
| `betterbench-prefill.md` | BetterBench 0.4.0 prefill sweep, 8 depths, 1 warm-up + 5 passes each |
| `betterbench-prefill.json` | the same run, machine-readable, with `sample_gate` |
| `betterbench-prefill-config.json` | the exact config that produced it |
| `betterbench-decode-concurrency.md` | BetterBench 0.4.0 decode-by-category and concurrency sweep, 20 passes per category |
| `betterbench-decode-concurrency.json` | the same, machine-readable |
| `bench-live-deep.tsv` | `bench-live.sh` at the same 8 depths — prefill, decode, acceptance, mean accepted length, step rate |
| `bench-live-deep-env.txt` | provenance for that run |

Reproduce the first with `../run-betterbench-prefill.sh`.

## Read this before comparing the two ladders

**The rows do not line up.** Both harnesses synthesise filler text and both
undershoot the real tokenizer, by different amounts:

- bench-live is **tokenizer-calibrated** here (`calibrate: 1`), so its
  `prompt_tok` column is exact — within 0.30% at every rung, 0.01–0.04% above 32k.
- BetterBench estimates 4 chars/token and does **not** check. Its `200000` rung
  is really **146978** tokens; its `2000` rung is really 1544.

So read BetterBench on its **prompt tokens (med)** column, never on its target
depth, and do not put its `200000` row beside bench-live's `200000` row.

## Prefill (take this from BetterBench)

n=5 per rung, p99−p1 spread under 0.2% from 64k up.

| actual tokens | PP t/s median |
| --: | --: |
| 1544 | 2815 |
| 5948 | **2882** |
| 11823 | 2787 |
| 23573 | 2631 |
| 47085 | 2418 |
| 94095 | 2087 |
| 117594 | 1948 |
| 146978 | 1808 |

Peak at ~6k, not at the shallowest rung: 1544 tokens is too small to fill the
GPU. From there it decays gracefully — **2882 → 1808 t/s across a 25× depth
increase**, i.e. 63% of peak at 147k tokens.

bench-live's own prefill column is n=1 per rung and shows it: its 7976 rung
(2506 t/s) sits *below* its 16k rung, which is impossible on a real curve.
Prefill comes from BetterBench; step rate and acceptance come from bench-live.

## Generation at depth (take this from bench-live)

| actual tokens | steps/s | mean accepted len | decode t/s |
| --: | --: | --: | --: |
| 2000 | 16.99 | 4.20 | 71.3 |
| 7976 | 16.84 | 2.98 | 50.1 |
| 15986 | 16.69 | 3.56 | 59.3 |
| 31988 | 16.35 | 2.81 | 46.0 |
| 63982 | 15.78 | 3.32 | 52.5 |
| 127965 | 14.82 | 3.28 | 48.7 |
| 159977 | 14.42 | 3.28 | 47.3 |
| 199972 | **13.90** | 3.28 | 45.6 |

**`steps/s` is the column to read.** `decode t/s = steps/s × mean_len`, and
`mean_len` is a draw from the speculative-acceptance lottery — on a single
request it swings enough to invert neighbouring rows (see 7976 vs 15986 above,
where the step rate is nearly flat but decode differs by 18%).

Two results:

1. **Generation at a real 200k runs at 82% of its 2k step rate** (13.90 vs
   16.99). A 100× longer prompt costs 18% of the step rate.
2. **Acceptance does not degrade with depth.** `mean_len` is flat at 3.28 from
   64k all the way to 200k. The shallow rungs' 4.20/2.98/3.56/2.81 are n=1
   sampling noise, not a trend.

## Caveats

- bench-live is **n=1 per rung**. Its step rate repeats to ~0.1% across boots,
  but its decode t/s and acceptance columns do not — do not price a change on them.
- Both runs are single-boot. Across boots, 50k prefill spans ~2.8%, well above
  the within-boot ±0.6%.
- BetterBench's p1/p99 columns are flagged `†` in its own report: 5 passes
  cannot support a percentile. Read them as "roughly the worst observed".

## Decode by category, and concurrency

`betterbench-decode-concurrency.md`, 20 passes per category, temp 0.7.

| category | tok/update | decode t/s (med) | CV |
| --- | --: | --: | --: |
| file_edit | 5.67 | **109.1** | 11.5% |
| math | 5.98 | 105.2 | 5.8% |
| code | 4.91 | 96.8 | 15.6% |
| json | 5.15 | 95.7 | 15.2% |
| reasoning | 3.43 | 64.5 | 28.6% |
| chat | 3.64 | 57.6 | 22.5% |
| prose | 2.86 | 50.2 | 8.2% |

**Weighted combined: 83.6 t/s**, update p99 62.7 ms, TTFT p50 ~102 ms.

The spread is almost entirely `tok/update` — the speculative acceptance rate. Update
spacing is flat at 58.0–58.3 ms p50 across every category, so **the engine steps at a
constant rate and the category only decides how many tokens ride each step.** Structured
output (file_edit, math, code, json) is predictable enough for the drafter to land 5–6
tokens per update; prose lands 2.86 and is the floor. That is the same mechanism as the
depth result above, seen along a different axis.

Read `CV` before reading a single row: reasoning at 28.6% and chat at 22.5% are noisy
enough that their ordering against each other is not stable.

### Concurrency

| level | aggregate t/s | TTFT p50 (ms) | per-request decode t/s (med) |
| --: | --: | --: | --: |
| 1 | 74.3 | 102 | 90.9 |
| 2 | **134.5** | 163 | 86.0 |
| 4 | 137.0 | 5104 | 86.7 |
| 8 | 133.8 | 14856 | 86.0 |
| 16 | 139.0 | 15583 | 88.5 |

**All the aggregate throughput this configuration has is available at concurrency 2, and
nothing above 2 buys anything.** 1 → 2 is +81%; 2 → 16 is +3.3% for a **95× worse TTFT**
(163 ms → 15.6 s). Per-request decode is flat throughout, so the queue is where the time
goes, not the GPU.

That is `MAXSEQS=2` working as configured, not a limit being hit: the scheduler has two
slots, so requests 3 and beyond queue. It is the right trade for a single-user agentic box
— see the 2026-09-05 concurrency entry in [`../../TUNING.md`](../../TUNING.md) — but if you
are serving several people, this is the number to change first, and it will cost context.

## A caveat on the headline number

83.6 t/s weighted here, against **85.6** measured on 2026-09-05 during config selection.
Same stack, and the gap is not a regression: the earlier figure was taken at
`MAXLEN=131072` before the context was raised, and there is a real ~5% decode cost to a
bigger KV pool (TUNING.md, 2026-09-05 CHUNK entry). **83.6 with 204800 tokens of context is
the number this repo actually serves**, and the trade was deliberate.

*Note also that 110/160 single-stream runs stopped at `max_tokens`. On a thinking model a
truncated run measures the thinking phase rather than a complete answer, so read the
reasoning/answer split table in the report as indicative.*
