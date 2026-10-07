# BetterBench report

```
BetterBench 0.6.0 · corpus v1.0 · 29 prompts in 8 categories
phases: decode, prefill, concurrency
3 warmup + 20 measured passes per category
model max context: 204800 tokens
```

- **betterbench**: 0.6.0, [GGZ14/BetterBench](https://github.com/GGZ14/BetterBench) `v0.6.0` (`d00ad5e`) plus one local commit, `9a9e69f` ("corpus: salt nonces per run so prefill runs cannot hit each other in a persistent KV cache"). Not the version behind BETTERBENCH-20261006.md (0.4.0): 0.6.0 reshuffles the prefill filler every pass. Like-for-like prefill (0.6.0, previous config fp32/CHUNK=2048 measured the same night): 16k 2970 -> 3190 (+7.4%), 32k 2834 -> 3084 (+8.8%), 64k 2596 -> 2870 (+10.6%), 128k 2208 -> 2470 (+11.9%).
- **endpoint**: `http://127.0.0.1:1234/v1`  ·  **model**: `qwen3.8-27b-vllm`  ·  **host**: llama
- **corpus**: v1.0  ·  **sampling**: temp 1.0  ·  **passes/cat**: 20  ·  prefix-cache: cold (nonce)
- **notes**: `image=vllm-radiance-0.9.3`  ·  `libr4d=rx17`  ·  `ssm=fp16`  ·  `chunk=3532`  ·  `lazy_gdn=on`  ·  `kv_mem=11.5e9`  ·  `fast_draft=on`  ·  `embed_host=on`  ·  `verifyhead=global-256`
- **gpu**: amd ['device,Card Series,Card Model,Card Vendor,Card SKU,Subsystem ID,Device Rev,Node ID,GUID,GFX Version', 'card0,AMD Radeon AI PRO R9700,0x7551,Advanced Micro Devices Inc. [AMD/ATI],APM107573,0x5413,0xc0,1,51833,gfx1201']

## Single-stream (batch = 1)

This server packs several tokens into one stream update (speculative decoding), so there is no per-token latency to report — the tokens in an update arrive together. **update p50/p99** is the measured wall-clock gap between updates (p99 is the stutter); **tok/update** is how many tokens land per update. TTFT in ms; decode = per-run tok/s.

| category | passes | TTFT p50 | TTFT p99 | update p50 (ms) | update p99 (ms) | tok/update | decode t/s (med) | ±IQR | CV |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| chat | 20 | 99.1 | 103.8† | 35.2 | 38.3 | 3.58 | 96.6 | 23.7 | 17.8% |
| code | 20 | 97.8 | 101.1† | 35.4 | 39.0 | 4.59 | 147.7 | 27.8 | 15.8% |
| file_edit | 20 | 99.9 | 107.2† | 35.4 | 38.6 | 5.96 | 174.1 | 28.3 | 8.6% |
| json | 20 | 98.1 | 100.6† | 35.3 | 38.8 | 5.07 | 173.6 | 50.5 | 18.5% |
| math | 20 | 96.0 | 99.4† | 35.4 | 38.4 | 5.80 | 169.2 | 13.9 | 7.2% |
| prose | 20 | 96.7 | 102.8† | 35.3 | 38.4 | 2.76 | 77.9 | 10.4 | 9.2% |
| reasoning | 20 | 97.3 | 100.0† | 35.4 | 38.1 | 3.34 | 99.8 | 58.8 | 31.6% |
| summarization | 20 | 100.4 | 107.4† | 35.3 | 37.6 | 4.79 | 130.8 | 39.5 | 17.6% |

**Combined (weighted code:0.3, reasoning:0.2, prose:0.15, json:0.15, file_edit:0.1, summarization:0.1)** — decode t/s median ≈ **132.5**, update p99 ≈ **38.5 ms**, TTFT p50 ≈ **98 ms**

*160 of 160 runs streamed several tokens per update (`chunk_token_mismatch`). Per-token ITL is not reported for them — see METHODOLOGY.md §chunk-token.*

## Reasoning / answer split

A per-token rate cannot see how much of a run was spent thinking. Two configs with identical decode t/s can take very different times to reach an answer. **TTFA** is time to the first *answer* token — the wait a reader actually feels.

| category | runs w/ split | reasoning share (est) | TTFA p50 (ms) | never reached answer |
|---|--:|--:|--:|--:|
| chat | 20/20 | 76% | 1398.6 | 14/20 |
| code | 20/20 | 75% | 2037.5 | 6/20 |
| file_edit | 20/20 | 89% | 663.4 | 9/20 |
| json | 20/20 | 54% | 663.8 | 3/20 |
| math | 20/20 | 69% | 1525.5 | 8/20 |
| prose | 20/20 | 70% | 3024.1 | 8/20 |
| reasoning | 20/20 | 66% | 2033.7 | 9/20 |
| summarization | 20/20 | 52% | 928.9 | 0/20 |

*A `—` means too few runs reached an answer to say (fewer than 5, or under half the passes). Runs cut off before any answer began are counted, not folded in: crediting their output as an answer would flatter the result. Token counts are apportioned by character count, so the share is an estimate — punctuation-dense answers (json, code) are under-counted.*

*Stopped at `max_tokens`: **110/160** runs (69%). On a thinking model a truncated run measures the thinking phase, not a complete answer.*

## Concurrency sweep

| level | ok/req | aggregate t/s | TTFT p50 | TTFT p99 | per-req decode t/s (med) |
|--:|--:|--:|--:|--:|--:|
| 1 | 24/24 | 116.9 | 99.1 | 106.3† | 141.5 |
| 2 | 24/24 | 209.4 | 166.1 | 568.7† | 140.6 |

## Prompt processing (prefill) sweep

Prefill throughput = prompt tokens ÷ TTFT, at increasing input depth (tiny decode, cold prefix cache). PP t/s columns: 1% low / median / 99% high.

| target depth | prompt tokens (med) | TTFT p50 (ms) | PP t/s 1% low | PP t/s median | PP t/s 99% high |
|--:|--:|--:|--:|--:|--:|
| 2000 | 1556 | 535.1 | 2825.7† | 2896.3 | 2930.5† |
| 8000 | 5939 | 1902.9 | 3108.6† | 3119.3 | 3160.4† |
| 16000 | 11838 | 3704.3 | 3169.9† | 3189.9 | 3220.0† |
| 32000 | 23619 | 7663.3 | 2912.8† | 3083.7 | 3095.7† |
| 64000 | 47184 | 16438.9 | 2863.9† | 2870.4 | 2874.0† |
| 128000 | 94156 | 38140.3 | 2438.6† | 2470.1 | 2485.0† |

---
*† this percentile rests on fewer samples than `n · tail ≥ 5` requires — a p99 needs 500 observations, and 20 passes give 20. Read it as "roughly the worst observed", not as a percentile. The full list is under `sample_gate` in `results.json`.*

*Generated by BetterBench. See METHODOLOGY.md §sample-size.*
