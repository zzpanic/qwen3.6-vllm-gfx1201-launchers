# BetterBench, 2026-10-06: the release configuration

This is the final BetterBench run of the configuration this repository ships. It covers long decode,
prefill to 128k, and concurrency 1 and 2. It is compared with the morning's release on the same box.
The raw output (`results.json`, the charted `results.html`, the config and the log) is in
[`benchmarks/betterbench-20261006/`](benchmarks/betterbench-20261006/).

## Headline

| | Morning release | **This release** | Change |
|---|--:|--:|--:|
| Decode, combined (weighted median) | 110.6 t/s | **129.7 t/s** | **+17%** |
| Decode step, p50 / p99 | 41.7 / 46.8 ms | **35.7 / 40.1 ms** | -14% / -14% |
| Concurrency 1, aggregate | 99.1 t/s | **114.7 t/s** | +16% |
| Concurrency 2, aggregate | 175.5 t/s | **202.0 t/s** | **+15%** |
| Prefill at 32k | 2,775 t/s | **2,802 t/s** | +1% |
| GPU KV pool | 259,011 tokens | **329,035 tokens** | **+27%** |
| Stalls, engine faults | 0, 0 | **0, 0** | |

**What changed between the two runs:**
- the int2 draft head and the global top-256 verify head (`FAST_DRAFT=1` plus vllm-radiance PR #9 and
  the fixes that make it arm on vLLM 0.27.1);
- the input embedding in host RAM (`EMBED_HOST=1`, the KV gain);
- the DFlash2 sampling-RNG fix;
- exact-wide DSPLIT prefill and the block-table clamp (libr4d rx14 -> rx16);
- the MXFP4 decode band for M 9-64.

The decode gain is almost all step time: about 6 ms less per verify step, at the same tokens per
step. The details are in [KERNEL.md](KERNEL.md).

## Setup

| | |
|---|---|
| Hardware | 1 x AMD Radeon AI PRO R9700 (gfx1201, 32 GB), VFIO passthrough to a 4-vCPU / 40 GiB Ubuntu 24.04 VM ([SYSTEM.md](SYSTEM.md)) |
| Image | `stilldeadcode/vllm-radiance:0.9.3` (vLLM 0.27.1, V2 model runner) |
| Model | Qwen3.8-27B MXFP4 W4A8, DFlash2 drafter (FP8), 7 speculative tokens, probabilistic draft sampling |
| Kernels | libr4d v0.5.0 + `r4d_kernels.patch` (rx16: exact GDN scan, DSPLIT, clamp, lazy-GDN invalidation), MXFP4 decode band |
| State, KV | fp32 GDN state, lazy GDN on, fp8 KV, `KV_MEM=11.8e9` (329,035 tokens), `EMBED_HOST=1` |
| Serving | `MAXSEQS=2`, `CHUNK=2048`, context **204,800** (see the note below), KV offload GPU -> RAM (16 GiB) -> disk |
| BetterBench | 0.4.0, corpus v1.0, 3 warmup + **20 measured passes** per category, temperature 1.0, top_p 0.95, top_k 20 |
| Cache | Cold: a salted nonce per run, so no request is served from the prefix cache |
| Load | The LAN was quiet: 0 outside requests during the run (checked from the llama-swap log) |

**Note:** this run used the production entry's 204,800-token context and 16 GiB RAM tier. The
launcher as shipped defaults to the model's full 262,144 tokens and a 19 GiB RAM tier (KV-CACHE.md).
The GPU pool and the kernels are the same, so decode and prefill up to 128k are unaffected. Prefill
beyond 128k at 262,144 has not been benchmarked.

## Single stream

| Category | Decode t/s (median) | ± IQR | Tokens / step | Step p50 / p99 (ms) | TTFT p50 (ms) | vs morning |
|---|--:|--:|--:|--:|--:|--:|
| file_edit | **176.4** | 19.6 | 5.92 | 35.8 / 40.7 | 100.8 | +18% |
| json | **165.4** | 44.1 | 5.13 | 35.7 / 40.2 | 99.5 | +19% |
| math | **163.1** | 9.0 | 5.74 | 35.8 / 39.3 | 98.1 | +13% |
| code | **142.9** | 20.5 | 4.80 | 35.8 / 40.1 | 99.4 | +17% |
| summarization | **139.8** | 10.2 | 4.85 | 35.8 / 39.5 | 102.2 | +26% |
| chat | **93.0** | 19.9 | 3.52 | 35.6 / 40.5 | 101.5 | +24% |
| reasoning | **93.0** | 45.9 | 3.39 | 35.8 / 40.2 | 99.2 | +12% |
| prose | **78.8** | 13.9 | 2.73 | 35.7 / 40.2 | 98.9 | +13% |
| **Combined** | **129.7** | | | **35.7 / 40.1** | **100** | **+17%** |

The combined figure is a weighted median: code 0.30, reasoning 0.20, prose 0.15, json 0.15, file_edit
0.10, summarization 0.10.

**Decode speed follows the tokens accepted per step.** The step time is flat at about 35.7 ms in every
category. The rate differs only in how many of the 7 drafted tokens the target accepts: about 6 for
predictable edits and structured output, under 3 for free prose. All 160 runs streamed several tokens
per update, so per-token latency is not meaningful and BetterBench reports the step gap instead.

**Most runs end at the token cap.** 113 of the 160 runs stopped at `max_tokens`. This is a thinking
model, and many runs were still reasoning when they hit it, so these numbers measure generation speed,
not time to a finished answer. Time to the first answer token (TTFA p50) ranged from 0.66 s (json)
to 3.2 s (prose).

## Concurrency

| Clients | OK / requests | Aggregate t/s | Per-request decode t/s | TTFT p50 (ms) |
|--:|--:|--:|--:|--:|
| 1 | 24/24 | 114.7 | 144.8 | 99.7 |
| 2 | 24/24 | **202.0** | 138.0 | 167.7 |

The second client costs each stream under 5% of its speed (144.8 -> 138.0 t/s) and nearly doubles
throughput (1.76x). `MAXSEQS=2` is the served limit, so concurrency 2 is the top of the sweep.

## Prefill

Cold prefix cache, minimal decode. Throughput = prompt tokens / TTFT.

| Target depth | Prompt tokens | TTFT p50 | Prefill t/s (median) | 1% low | 99% high | vs morning |
|--:|--:|--:|--:|--:|--:|--:|
| 2k | 1,551 | 0.51 s | **3,025** | 3,000 | 3,031 | +0.1% |
| 8k | 5,954 | 1.94 s | **3,065** | 3,056 | 3,086 | +0.6% |
| 16k | 11,830 | 4.08 s | **2,900** | 2,871 | 2,902 | +0.7% |
| 32k | 23,580 | 8.42 s | **2,802** | 2,799 | 2,806 | +1.0% |
| 64k | 47,092 | 18.3 s | **2,569** | 2,561 | 2,572 | +0.7% |
| 128k | 94,102 | 43.1 s | **2,185** | 2,174 | 2,187 | (not run in the morning) |

- Prefill is essentially unchanged, which is expected: the release's changes target decode and
  capacity.
- DSPLIT only speeds up *short final chunks over a deep cache*, as in an agent's follow-up turn. These
  cold, whole-prompt prefills don't exercise that.
- The 1% / 99% band is within 1% at every depth.
- The 128k prompt is 94k tokens of the corpus, so that is the real depth.

## Correctness checks run alongside

These were run in the same session, on the same boot sequence, with the LAN quiet.

| Check | Result |
|---|---|
| Approximate lm_head (int2 verify head) vs exact head, no logprobs (`quality_ab.py`) | Open-ended: 24/24 texts identical. Arithmetic: 40/40 both. Lookups over a 300-record registry: 29/30 both, same question missed. |
| Stress: two 158k sequences resident, KV at 96.6% (317,714 of 329,035 tokens), plus a 2048x2048 image request | 0 preemptions, 0 faults. Peak VRAM 31.23 of 31.86 GiB; host RAM peak 27.0 GB. |
| Soak: two concurrent agents for 40 minutes | 836 turns, 834 correct, 0 engine faults, memory flat. Both misses were the model copying a record wrong, not the engine. |
| Resume from the offload tier vs cold run | Token-identical (`val_resume.py --evict`) |

## Reading the numbers

- **p99 values** rest on 20 passes, so read them as "about the worst seen", not as true percentiles.
  BetterBench marks every such value with † in the raw report.
- **Temperature 1.0** is the model card's preset and what is served. Draft acceptance, and therefore
  decode t/s, depend on it; greedy decoding would score higher.
- **Morning release** is the 2026-10-06 run `betterbench-final-full` on the same box: libr4d rx14,
  `EMBED_HOST` off, `FAST_DRAFT` off. It used the same BetterBench config apart from the
  128k prefill depth.
