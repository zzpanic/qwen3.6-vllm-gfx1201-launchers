# BetterBench report

```
BetterBench 0.4.0 · corpus v1.0 · 29 prompts in 8 categories
phases: decode, prefill, concurrency
3 warmup + 20 measured passes per category
model max context: 204800 tokens
```

- **betterbench**: 0.4.0, [GGZ14/BetterBench](https://github.com/GGZ14/BetterBench) `main` at `1de941d` plus one local commit, `89bef80` ("corpus: salt nonces per run so prefill runs cannot hit each other in a persistent KV cache")
- **endpoint**: `http://127.0.0.1:1234/v1`  ·  **model**: `qwen3.8-27b-vllm`  ·  **host**: llama
- **corpus**: v1.0  ·  **sampling**: temp 1.0  ·  **passes/cat**: 20  ·  prefix-cache: cold (nonce)
- **notes**: `image=vllm-radiance-0.9.3`  ·  `libr4d=rx16-dsplit-clamp`  ·  `ssm=fp32`  ·  `lazy_gdn=on`  ·  `fast_draft=on`  ·  `verifyhead=global-256`  ·  `embed_host=on`  ·  `kv_mem=11.8e9`  ·  `mxfp4=house-sly-decode-band`  ·  `dflash=dflash2x7-probabilistic`  ·  `max_num_seqs=2`  ·  `nonce=salted-per-run`  ·  `config=config-prod-20260823+prefill128k`
- **gpu**: amd ['device,Card Series,Card Model,Card Vendor,Card SKU,Subsystem ID,Device Rev,Node ID,GUID,GFX Version', 'card0,AMD Radeon AI PRO R9700,0x7551,Advanced Micro Devices Inc. [AMD/ATI],APM107573,0x5413,0xc0,1,27203,gfx1201']

## Single-stream (batch = 1)

This server packs several tokens into one stream update (speculative decoding), so there is no per-token latency to report — the tokens in an update arrive together. **update p50/p99** is the measured wall-clock gap between updates (p99 is the stutter); **tok/update** is how many tokens land per update. TTFT in ms; decode = per-run tok/s.

| category | passes | TTFT p50 | TTFT p99 | update p50 (ms) | update p99 (ms) | tok/update | decode t/s (med) | ±IQR | CV |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| chat | 20 | 101.5 | 107.2† | 35.6 | 40.5 | 3.52 | 93.0 | 19.9 | 19.5% |
| code | 20 | 99.4 | 111.5† | 35.8 | 40.1 | 4.80 | 142.9 | 20.5 | 13.0% |
| file_edit | 20 | 100.8 | 107.1† | 35.8 | 40.7 | 5.92 | 176.4 | 19.6 | 9.3% |
| json | 20 | 99.5 | 105.9† | 35.7 | 40.2 | 5.13 | 165.4 | 44.1 | 17.6% |
| math | 20 | 98.1 | 100.8† | 35.8 | 39.3 | 5.74 | 163.1 | 9.0 | 7.5% |
| prose | 20 | 98.9 | 101.1† | 35.7 | 40.2 | 2.73 | 78.8 | 13.9 | 10.9% |
| reasoning | 20 | 99.2 | 105.1† | 35.8 | 40.2 | 3.39 | 93.0 | 45.9 | 28.4% |
| summarization | 20 | 102.2 | 108.1† | 35.8 | 39.5 | 4.85 | 139.8 | 10.2 | 9.9% |

**Combined (weighted code:0.3, reasoning:0.2, prose:0.15, json:0.15, file_edit:0.1, summarization:0.1)** — decode t/s median ≈ **129.7**, update p99 ≈ **40.1 ms**, TTFT p50 ≈ **100 ms**

*160 of 160 runs streamed several tokens per update (`chunk_token_mismatch`). Per-token ITL is not reported for them — see METHODOLOGY.md §chunk-token.*

## Reasoning / answer split

A per-token rate cannot see how much of a run was spent thinking. Two configs with identical decode t/s can take very different times to reach an answer. **TTFA** is time to the first *answer* token — the wait a reader actually feels.

| category | runs w/ split | reasoning share (est) | TTFA p50 (ms) | never reached answer |
|---|--:|--:|--:|--:|
| chat | 20/20 | 78% | 1382.4 | 15/20 |
| code | 20/20 | 79% | 2064.2 | 7/20 |
| file_edit | 20/20 | 86% | 745.4 | 6/20 |
| json | 20/20 | 53% | 655.3 | 4/20 |
| math | 20/20 | 68% | 1241.3 | 8/20 |
| prose | 20/20 | 70% | 3199.2 | 8/20 |
| reasoning | 20/20 | 74% | 988.1 | 11/20 |
| summarization | 20/20 | 54% | 923.3 | 0/20 |

*A `—` means too few runs reached an answer to say (fewer than 5, or under half the passes). Runs cut off before any answer began are counted, not folded in: crediting their output as an answer would flatter the result. Token counts are apportioned by character count, so the share is an estimate — punctuation-dense answers (json, code) are under-counted.*

*Stopped at `max_tokens`: **113/160** runs (71%). On a thinking model a truncated run measures the thinking phase, not a complete answer.*

## Concurrency sweep

| level | ok/req | aggregate t/s | TTFT p50 | TTFT p99 | per-req decode t/s (med) |
|--:|--:|--:|--:|--:|--:|
| 1 | 24/24 | 114.7 | 99.7 | 107.8† | 144.8 |
| 2 | 24/24 | 202.0 | 167.7 | 224.3† | 138.0 |

## Prompt processing (prefill) sweep

Prefill throughput = prompt tokens ÷ TTFT, at increasing input depth (tiny decode, cold prefix cache). PP t/s columns: 1% low / median / 99% high.

| target depth | prompt tokens (med) | TTFT p50 (ms) | PP t/s 1% low | PP t/s median | PP t/s 99% high |
|--:|--:|--:|--:|--:|--:|
| 2000 | 1551 | 512.3 | 3000.0† | 3025.3 | 3031.0† |
| 8000 | 5954 | 1942.7 | 3056.1† | 3065.3 | 3085.7† |
| 16000 | 11830 | 4079.6 | 2870.5† | 2899.8 | 2902.2† |
| 32000 | 23580 | 8415.9 | 2799.3† | 2801.8 | 2805.7† |
| 64000 | 47092 | 18333.3 | 2561.4† | 2568.6 | 2571.8† |
| 128000 | 94102 | 43071.3 | 2173.6† | 2184.8 | 2187.3† |

---
*† this percentile rests on fewer samples than `n · tail ≥ 5` requires — a p99 needs 500 observations, and 20 passes give 20. Read it as "roughly the worst observed", not as a percentile. The full list is under `sample_gate` in `results.json`.*

*Generated by BetterBench. See METHODOLOGY.md §sample-size.*
