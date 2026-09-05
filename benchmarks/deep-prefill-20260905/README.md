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
