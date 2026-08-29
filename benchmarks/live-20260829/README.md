# The production server as shipped — 2026-08-29

Everything here was captured against the running llama-swap endpoint with no exclusive GPU
window and no configuration changes: this is the box as it serves, not a lab arm. Config is
`vllm-radiance:0.9.3`, int4 W4A16 + DFlash2 ×4, `ATTN=R4D`, `RADIANCE_SKINNY_GEMM=all`,
`BATCHTOK=4854`, `KV_GROUP_SIZE=auto`, `GPUUTIL=0.97`, `MAXLEN=204800`, `MAXSEQS=2`.

Geometry at load: `Model loading took 18.91 GiB` (target 17.93 + draft ~1.08),
`Available KV cache memory: 9.62 GiB`, **`GPU KV cache size: 269,837 tokens`** = 1.32×
concurrency at `MAXLEN=204800`.

## Files

- `results.tsv` / `env.txt` — the depth ladder, three prompt depths, engine counters read
  from vLLM's own `/metrics` rather than the client.
- `betterbench-full.md` / `.json` — a full [BetterBench](https://github.com/GGZ14/BetterBench)
  pass: 8 categories × 4 runs single-stream, a 5-rung prefill sweep, and a concurrency sweep.

## Depth ladder (`results.tsv`)

| prompt tokens | prefill t/s | decode t/s | engine steps/s | accepted length |
| --- | --- | --- | --- | --- |
| 3,183 | 1776.6 | 59.94 | 21.31 | 2.81 |
| 25,565 | 1657.4 | 69.44 | 20.61 | 3.37 |
| 76,552 | 1426.4 | 48.54 | 19.53 | 2.49 |

**Read the step rate, not the tok/s.** Decode tok/s is `steps/s × accepted_length`, and the
step rate falls smoothly with depth (21.31 → 20.61 → 19.53) exactly as attention cost
predicts. Decode tok/s does not, because acceptance moves: the 25,565 rung is *faster* than
the 3,183 rung purely because its prompts drafted better (3.37 vs 2.81 accepted tokens per
step). That is content, not depth. Three depths, one capture each — treat the decode column
as soft and the step column as tight.

## Prefill sweep (BetterBench)

| depth | prompt tokens | prefill t/s (median) |
| --- | --- | --- |
| 2,000 | 1,544 | 1632.4 |
| 8,000 | 5,948 | **1736.8** |
| 16,000 | 11,824 | 1696.8 |
| 32,000 | 23,573 | 1631.5 |
| 64,000 | 47,086 | 1530.7 |

Prefill peaks around 8K and decays gently, holding **94% of peak at 47K tokens**. The 1%-low
to 99%-high span is under one tok/s at every rung, so these are effectively noiseless and
prefill is the right discriminator for any A/B on this box.

## Concurrency

| level | aggregate t/s | per-request decode t/s | TTFT p50 |
| --- | --- | --- | --- |
| 1 | 74.9 | 87.6 | 139.5 ms |
| 2 | 113.9 | 68.4 | 187.5 ms |

Two slots return **1.52× aggregate throughput for 0.78× per-request speed** — sublinear, as
expected on a bandwidth-bound single card where the second sequence shares weight reads but
adds its own decode work. `MAXSEQS=2` is the ceiling here for a KV reason, not a throughput
one: at `MAXLEN=204800` the pool holds 1.32 full-length sequences.

## Single-stream by category

Combined weighted decode **79.9 tok/s**, ITL 1%-low 61.4 tok/s, TTFT p50 144 ms. Category
spread is wide (57.6 prose → 99.4 file_edit) and it is mostly acceptance: predictable output
drafts well and edits are the most predictable thing here. Note the CV column — `chat` 18.0%
and `reasoning` 22.8% over 4 runs, against `file_edit` at 0.8%. **Do not compare two configs
on a 4-run chat number**; the categories that matter most to this workload are also the
noisiest.
