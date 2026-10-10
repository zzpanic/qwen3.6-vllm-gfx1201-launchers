# BetterBench report

```
BetterBench 0.6.0 · corpus v1.0 · 29 prompts in 8 categories
phases: decode, prefill, concurrency
3 warmup + 20 measured passes per category
model max context: 204800 tokens
```

- **betterbench**: 0.6.0, [GGZ14/BetterBench](https://github.com/GGZ14/BetterBench) `v0.6.0` (`d00ad5e`) plus local commit `9a9e69f` (per-run nonce salt). Same build as BETTERBENCH-20261008.md.
- **config**: the house production config of 2026-10-11. CHUNK=3568, MAXSEQS=4, checkpoint retention and the canonical-chunk wait are not yet in this repo's launcher. BetterBench prompts are one-off (nonce), so the prefix-cache changes are not exercised here: this is a no-regression check plus concurrency 3 and 4.
- **vs BETTERBENCH-20261008.md**: combined decode 132.5 -> 135.5 t/s; concurrency 2 aggregate 209.4 -> 211.3 t/s; prefill 128k 2470 -> 2490 t/s. New: concurrency 4 aggregate 326.7 t/s (2.75x concurrency 1), 0 preemptions; prefill at 180k target (132k real tokens) 2235 t/s.
- **endpoint**: `http://127.0.0.1:5801/v1`  ·  **model**: `qwen3.8-27b-vllm`  ·  **host**: llama
- **corpus**: v1.0  ·  **sampling**: temp 1.0  ·  **passes/cat**: 20  ·  prefix-cache: cold (nonce)
- **notes**: `image=vllm-radiance-0.9.3`  ·  `libr4d=rx17`  ·  `ssm=fp16`  ·  `chunk=3568`  ·  `maxseqs=4`  ·  `lazy_gdn=on`  ·  `kv_mem=12.25e9`  ·  `fast_draft=on`  ·  `embed_host=on`  ·  `checkpoint_retention=on`  ·  `canon_wait=on`  ·  `kv_offload=RAM 12 GiB, no disk tier`
- **gpu**: amd ['device,Card Series,Card Model,Card Vendor,Card SKU,Subsystem ID,Device Rev,Node ID,GUID,GFX Version', 'card0,AMD Radeon AI PRO R9700,0x7551,Advanced Micro Devices Inc. [AMD/ATI],APM107573,0x5413,0xc0,1,51833,gfx1201']

## Single-stream (batch = 1)

This server packs several tokens into one stream update (speculative decoding), so there is no per-token latency to report — the tokens in an update arrive together. **update p50/p99** is the measured wall-clock gap between updates (p99 is the stutter); **tok/update** is how many tokens land per update. TTFT in ms; decode = per-run tok/s.

| category | passes | TTFT p50 | TTFT p99 | update p50 (ms) | update p99 (ms) | tok/update | decode t/s (med) | ±IQR | CV |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| chat | 20 | 97.8 | 99.2† | 33.6 | 36.5 | 3.37 | 94.1 | 29.5 | 17.5% |
| code | 20 | 96.4 | 99.7† | 33.8 | 36.4 | 4.59 | 149.5 | 22.9 | 13.1% |
| file_edit | 20 | 98.4 | 400.1† | 33.9 | 36.3 | 5.73 | 179.5 | 27.5 | 9.9% |
| json | 20 | 97.1 | 105.8† | 33.9 | 36.6 | 5.13 | 176.0 | 46.5 | 17.2% |
| math | 20 | 96.6 | 103.7† | 33.9 | 36.4 | 5.76 | 176.1 | 12.2 | 7.2% |
| prose | 20 | 96.4 | 98.5† | 33.8 | 36.2 | 2.70 | 80.5 | 17.2 | 11.4% |
| reasoning | 20 | 96.6 | 98.9† | 33.8 | 36.3 | 3.29 | 97.9 | 57.9 | 31.2% |
| summarization | 20 | 100.0 | 106.7† | 33.9 | 37.1 | 4.79 | 146.1 | 32.5 | 13.6% |

**Combined (weighted code:0.3, reasoning:0.2, prose:0.15, json:0.15, file_edit:0.1, summarization:0.1)** — decode t/s median ≈ **135.5**, update p99 ≈ **36.4 ms**, TTFT p50 ≈ **97 ms**

*160 of 160 runs streamed several tokens per update (`chunk_token_mismatch`). Per-token ITL is not reported for them — see METHODOLOGY.md §chunk-token.*

## Reasoning / answer split

A per-token rate cannot see how much of a run was spent thinking. Two configs with identical decode t/s can take very different times to reach an answer. **TTFA** is time to the first *answer* token — the wait a reader actually feels.

| category | runs w/ split | reasoning share (est) | TTFA p50 (ms) | never reached answer |
|---|--:|--:|--:|--:|
| chat | 20/20 | 79% | 1719.1 | 13/20 |
| code | 20/20 | 73% | 1757.7 | 8/20 |
| file_edit | 20/20 | 92% | 621.6 | 10/20 |
| json | 20/20 | 53% | 639.9 | 3/20 |
| math | 20/20 | 65% | 1384.6 | 5/20 |
| prose | 20/20 | 79% | 2966.6 | 11/20 |
| reasoning | 20/20 | 69% | 2661.8 | 9/20 |
| summarization | 20/20 | 52% | 880.3 | 0/20 |

*A `—` means too few runs reached an answer to say (fewer than 5, or under half the passes). Runs cut off before any answer began are counted, not folded in: crediting their output as an answer would flatter the result. Token counts are apportioned by character count, so the share is an estimate — punctuation-dense answers (json, code) are under-counted.*

*Stopped at `max_tokens`: **116/160** runs (72%). On a thinking model a truncated run measures the thinking phase, not a complete answer.*

## Concurrency sweep

| level | ok/req | aggregate t/s | TTFT p50 | TTFT p99 | per-req decode t/s (med) |
|--:|--:|--:|--:|--:|--:|
| 1 | 24/24 | 118.9 | 97.5 | 103.7† | 155.4 |
| 2 | 24/24 | 211.3 | 163.1 | 218.4† | 133.3 |
| 3 | 24/24 | 262.6 | 167.6 | 298.5† | 112.4 |
| 4 | 24/24 | 326.7 | 172.8 | 305.6† | 106.8 |

## Prompt processing (prefill) sweep

Prefill throughput = prompt tokens ÷ TTFT, at increasing input depth (tiny decode, cold prefix cache). PP t/s columns: 1% low / median / 99% high.

| target depth | prompt tokens (med) | TTFT p50 (ms) | PP t/s 1% low | PP t/s median | PP t/s 99% high |
|--:|--:|--:|--:|--:|--:|
| 2000 | 1551 | 530.2 | 2897.4† | 2927.2 | 2970.6† |
| 8000 | 5942 | 1888.5 | 3128.6† | 3147.5 | 3179.7† |
| 16000 | 11857 | 3693.9 | 3200.5† | 3210.3 | 3219.0† |
| 32000 | 23563 | 7610.9 | 3085.4† | 3091.2 | 3104.0† |
| 64000 | 47105 | 16288.5 | 2878.8† | 2890.9 | 2892.6† |
| 128000 | 94143 | 37798.3 | 2487.2† | 2490.3 | 2492.1† |
| 180000 | 132207 | 59151.4 | 2230.3† | 2234.9 | 2237.5† |

---
*† this percentile rests on fewer samples than `n · tail ≥ 5` requires — a p99 needs 500 observations, and 20 passes give 20. Read it as "roughly the worst observed", not as a percentile. The full list is under `sample_gate` in `results.json`.*

*Generated by BetterBench. See METHODOLOGY.md §sample-size.*