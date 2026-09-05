# int4 W4A16 — the production server as shipped, 2026-08-29

**This folder is the int4 path**, `startup-qwen3.8-27b-int4.sh`. It is not the MXFP4
configuration and its numbers must not be read beside MXFP4 ones without the caveats at
the bottom of this file. The folder was called `live-20260829/` until 2026-09-05; it was
renamed because "live" stopped identifying anything once a second stack existed.

One trap the rename is there to prevent: `env.txt` records `model: qwen3.8-27b-vllm`,
because on 2026-08-29 that llama-swap entry name served the **int4** weights. The same name
now serves MXFP4. Match folders on the quantisation, never on the entry name.

Same reason, same date: the launcher itself was renamed `startup-qwen3.8-27b-vllm.sh` ->
`startup-qwen3.8-27b-int4.sh`, and its default served-model id `qwen3.8-27b-vllm` ->
`qwen3.8-27b-int4`. If you were already running it, either pass `SERVED=qwen3.8-27b-vllm`
or update the model id your client sends. The runner scripts under
`spectok-sweep-20260829/` still name the old path -- they are a record of what was run in
August, and carry a note saying so.


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
- `gsm8k-20260815.json` — the lm-eval GSM8K result for this checkpoint, **94.77%** on both
  strict-match and flexible-extract (±0.61, 1319/1319 samples, 5-shot, temperature 1.0).
  Captured 2026-08-15, so it predates the 2026-08-29 retune in the header above — it was run
  at `MAXLEN=131072` and `SPECTOK=4`. It measures the *production serving path* (fp8 KV,
  speculative decoding, server-default sampler), not the weights in isolation, which is why
  it is not directly comparable to AMD's published Quark numbers. Home paths in the record
  are rewritten to `$HOME`. Two notes that matter if you reproduce it: never pass `min_p`
  (vLLM 400s on it under speculative decoding), and `num_concurrent` must not exceed
  `MAXSEQS`.

  Context for the number: statistically tied with AMD's best Quark checkpoint (Qronos,
  94.62/94.69) and 3.6 points above their AWQ. **Do not read it as a win** — it is a single
  run at temperature 1.0, the ±0.61 is binomial standard error only, and run-to-run sampling
  variance sits on top of that. Scoring above the BF16 base is a known artifact at temp 1.0,
  not evidence that quantisation helped.

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


## Comparing this folder to the MXFP4 numbers

Four things moved between this capture and `deep-prefill-20260905/`, and every one of them
is large enough to swamp the difference you would be trying to measure:

1. **Passes.** This BetterBench pass is `--runs 4`; the 2026-09-05 one is `--passes 20`.
   BetterBench cycles prompts rather than repeating one, so a 4-pass category samples a
   different, smaller prompt set. A 4-vs-20 comparison once manufactured a 48% "regression"
   here that did not exist.
2. **Sampling temperature.** 1.0 here, 0.7 on the later sweep.
3. **Depth ladder.** Three uncalibrated rungs here (`depth_req` 4000/32768/98304 are really
   3183/25565/76552 tokens); eight calibrated rungs to a real 200k there.
4. **The stack itself** — quantisation, SPECTOK (4 vs 7), skinny-GEMM setting, and whether
   the KV pool is pinned.

So this is a correct record of *this* configuration and is not a baseline for the MXFP4
work. A same-day int4 re-run on the current ladder is the way to get a real comparison.

### Correction, 2026-09-05: the ITL columns above are an artifact

`betterbench-full.md` reports ITL (inter-token latency) percentiles, including the
"ITL 1%-low 61.4 tok/s" quoted in the single-stream section. **Discount them.** BetterBench
0.4.0 removed those columns because they are meaningless on a speculative rig: under
DFlash2 the server emits a whole accepted run in one update, so "inter-token latency" was
measuring update spacing divided by a token count that varies with acceptance. The
replacement metrics are `update p50/p99 (ms)` and `tok/update`.

Everything else in that file — `decode t/s`, `total t/s`, `PP t/s`, `TTFT`, the concurrency
sweep — is unaffected and remains comparable across BetterBench versions. The numbers are
left as captured rather than edited, because the file is a verbatim record.
