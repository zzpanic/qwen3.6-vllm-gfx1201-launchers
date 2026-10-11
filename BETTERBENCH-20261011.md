# BetterBench report

```
BetterBench 0.6.0 · corpus v1.0 · 29 prompts in 8 categories
phases: decode, prefill, concurrency
3 warmup + 20 measured passes per category
model max context: 204800 tokens (BetterBench's prefill ceiling; the server runs 262144)
```

- **betterbench**: 0.6.0, [GGZ14/BetterBench](https://github.com/GGZ14/BetterBench) `v0.6.0` (`d00ad5e`) plus one local commit, `9a9e69f` ("corpus: salt nonces per run so prefill runs cannot hit each other in a persistent KV cache"). Same version and corpus as BETTERBENCH-20261008.md; that report ran at temperature 1.0, this one at BetterBench's default sampling (0.7 / top_p 0.95 / top_k 20).
- **launcher**: `startup-qwen3.8-27b-mxfp4.sh` with every default (2026-10-11 release): `MAXSEQS=4`, `CHUNK=3568`, `KV_MEM=12.65e9` (369,614 tokens), CUDA graphs captured up to 32 tokens, DFlash2-W4A16 drafter x7, checkpoint retention, RAM tier 16 GiB + disk tier, JIT warm-up. Boot to bench straight from the launcher, no other GPU work.
- **endpoint**: `http://127.0.0.1:<port>/v1`  ·  **model**: `qwen3.8-27b`  ·  **host**: llama
- **corpus**: v1.0  ·  **sampling**: temp 0.7  ·  **passes/cat**: 20  ·  prefix-cache: cold (nonce)
- **notes**: `config=repo-defaults-maxseqs4-chunk3568-kv12.65e9-capture-auto-warmup`  ·  `sampling=t0.7-p0.95-k20`  ·  `tier=ram16GiB+disk-kvcache`
- **gpu**: amd ['device,Card Series,Card Model,Card Vendor,Card SKU,Subsystem ID,Device Rev,Node ID,GUID,GFX Version', 'card0,AMD Radeon AI PRO R9700,0x7551,Advanced Micro Devices Inc. [AMD/ATI],APM107573,0x5413,0xc0,1,51833,gfx1201']

## Single-stream (batch = 1)

This server packs several tokens into one stream update (speculative decoding), so there is no per-token latency to report — the tokens in an update arrive together. **update p50/p99** is the measured wall-clock gap between updates (p99 is the stutter); **tok/update** is how many tokens land per update. TTFT in ms; decode = per-run tok/s.

| category | passes | TTFT p50 | TTFT p99 | update p50 (ms) | update p99 (ms) | tok/update | decode t/s (med) | ±IQR | CV |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| chat | 20 | 98.2 | 109.4† | 33.8 | 38.7 | 3.57 | 101.5 | 31.4 | 20.2% |
| code | 20 | 96.4 | 98.2† | 33.9 | 36.2 | 4.85 | 149.9 | 15.7 | 10.9% |
| file_edit | 20 | 98.5 | 104.4† | 33.9 | 37.8 | 5.83 | 185.3 | 31.9 | 11.7% |
| json | 20 | 97.2 | 104.4† | 33.9 | 36.1 | 5.22 | 170.5 | 37.5 | 12.9% |
| math | 20 | 96.3 | 100.2† | 33.9 | 36.5 | 5.75 | 172.6 | 19.5 | 9.0% |
| prose | 20 | 97.3 | 101.2† | 33.8 | 36.7 | 2.86 | 86.4 | 16.1 | 11.3% |
| reasoning | 20 | 96.7 | 98.1† | 33.8 | 36.3 | 3.43 | 103.5 | 60.9 | 32.1% |
| summarization | 20 | 99.9 | 105.8† | 33.9 | 36.6 | 4.97 | 152.5 | 15.3 | 12.6% |

**Combined (weighted code:0.3, reasoning:0.2, prose:0.15, json:0.15, file_edit:0.1, summarization:0.1)** — decode t/s median ≈ **138.0**, update p99 ≈ **36.5 ms**, TTFT p50 ≈ **97 ms**

*160 of 160 runs streamed several tokens per update (`chunk_token_mismatch`). Per-token ITL is not reported for them — see METHODOLOGY.md §chunk-token.*

## Reasoning / answer split

A per-token rate cannot see how much of a run was spent thinking. Two configs with identical decode t/s can take very different times to reach an answer. **TTFA** is time to the first *answer* token — the wait a reader actually feels.

| category | runs w/ split | reasoning share (est) | TTFA p50 (ms) | never reached answer |
|---|--:|--:|--:|--:|
| chat | 20/20 | 75% | 1678.0 | 14/20 |
| code | 20/20 | 82% | 1551.2 | 9/20 |
| file_edit | 20/20 | 92% | 592.1 | 10/20 |
| json | 20/20 | 54% | 707.6 | 3/20 |
| math | 20/20 | 68% | 1654.4 | 6/20 |
| prose | 20/20 | 72% | 3190.2 | 8/20 |
| reasoning | 20/20 | 67% | 1381.9 | 10/20 |
| summarization | 20/20 | 53% | 879.7 | 0/20 |

*A `—` means too few runs reached an answer to say (fewer than 5, or under half the passes). Runs cut off before any answer began are counted, not folded in: crediting their output as an answer would flatter the result. Token counts are apportioned by character count, so the share is an estimate — punctuation-dense answers (json, code) are under-counted.*

*Stopped at `max_tokens`: **113/160** runs (71%). On a thinking model a truncated run measures the thinking phase, not a complete answer.*

## Concurrency sweep

| level | ok/req | aggregate t/s | TTFT p50 | TTFT p99 | per-req decode t/s (med) |
|--:|--:|--:|--:|--:|--:|
| 1 | 24/24 | 124.7 | 97.4 | 103.8† | 161.1 |
| 2 | 24/24 | 218.1 | 162.2 | 231.4† | 145.1 |
| 3 | 24/24 | 273.5 | 166.3 | 298.6† | 122.3 |
| 4 | 24/24 | 333.7 | 170.9 | 304.4† | 116.4 |

## Prompt processing (prefill) sweep

Prefill throughput = prompt tokens ÷ TTFT, at increasing input depth (tiny decode, cold prefix cache). PP t/s columns: 1% low / median / 99% high.

| target depth | prompt tokens (med) | TTFT p50 (ms) | PP t/s 1% low | PP t/s median | PP t/s 99% high |
|--:|--:|--:|--:|--:|--:|
| 2000 | 1550 | 530.7 | 2865.3† | 2921.9 | 2964.3† |
| 8000 | 5938 | 1902.2 | 3093.9† | 3123.0 | 3146.6† |
| 16000 | 11839 | 3738.4 | 3120.9† | 3167.2 | 3183.6† |
| 32000 | 23618 | 7728.0 | 3043.6† | 3052.6 | 3071.7† |
| 64000 | 47064 | 16433.3 | 2850.3† | 2862.6 | 2869.1† |
| 128000 | 94192 | 38107.9 | 2468.5† | 2471.2 | 2480.0† |
| 180000 | 132338 | 59547.9 | 2222.2† | 2223.2 | 2229.9† |

## Re-run with the drafter prefill tiles (same day, later)

The report above predates `patch_w4a16_prefill.py`. The same config was re-run through the llama-swap entry on the same card (`benchmarks/betterbench-20261011-prefill-patch/`): decode 136.8 t/s (138.0 above; inside the run-to-run band), concurrency 1 / 2 / 3 / 4 at 123.0 / 217.2 / 273.4 / 346.5 t/s (124.7 / 218.1 / 273.5 / 333.7 above; the four-client figure moves by a few percent between runs), and prefill 0.8-1.1% faster at every depth from 6k up:

| prompt tokens (med) | PP t/s above | PP t/s re-run |
|--:|--:|--:|
| 1552 | 2921.9 | 2921.0 |
| 5940 | 3123.0 | 3150.4 |
| 11834 | 3167.2 | 3192.6 |
| 23568 | 3052.6 | 3085.8 |
| 47098 | 2862.6 | 2885.9 |
| 94070 | 2471.2 | 2494.4 |
| 132366 | 2223.2 | 2240.7 |

The tables above stay as the reference run; the end-to-end prefill gain matches the 0.8-1.4% measured when the patch was shipped (ROADMAP.md section 2, item 13).

---
*† this percentile rests on fewer samples than `n · tail ≥ 5` requires — a p99 needs 500 observations, and 20 passes give 20. Read it as "roughly the worst observed", not as a percentile. The full list is under `sample_gate` in `results.json`.*

*Generated by BetterBench. See METHODOLOGY.md §sample-size.*

