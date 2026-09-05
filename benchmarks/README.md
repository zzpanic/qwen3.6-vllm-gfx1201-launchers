# Benchmarks

Measured numbers from the homelab rig, with the scripts that produced them. Everything here
is a single Radeon AI PRO R9700 (gfx1201, 32 GiB), TP=1, vLLM 0.27.1 on the radiance fork,
Qwen3.8-27B with a DFlash2 draft head.

**Folders are named for the quantisation, because there are now two stacks.** `int4-*` is
`startup-qwen3.8-27b-int4.sh` (int4 W4A16); `mxfp4-*` and `deep-prefill-*` are
`startup-qwen3.8-27b-mxfp4.sh` (native MXFP4 W4A8). Do **not** match them up by llama-swap
entry name: the name `qwen3.8-27b-vllm` served int4 in August and serves MXFP4 now. Each
folder's README states which stack it is and what it can honestly be compared against.

| folder | what it measures |
| --- | --- |
| `spectok-sweep-20260829/` | **int4 W4A16.** Speculative-decode depth K ∈ {4,5,6,7} — throughput by category, acceptance, and KV cost per K |
| `deep-prefill-20260905/` | **MXFP4 W4A8.** The near-full-context sweep: BetterBench prefill and bench-live generation at 8 depths up to a real 200k tokens, plus the ladder-misalignment warning that applies to any comparison between the two harnesses |
| `int4-20260829/` | **int4 W4A16.** The production server as shipped on that date: prefill/decode/step-rate at three prompt depths, a BetterBench pass with prefill and concurrency sweeps, and the GSM8K quality result (94.77%) |
| `w4a16-census-20260829/` | **int4 W4A16** (the name says so — MXFP4 has no W4A16 GEMM). Runtime shape census under DFlash2, plus the isolated-kernel tile sweep that closed the last 0.61% of uncovered work |

## How to read these

**Prefer step rate to tokens/sec.** Under speculative decoding, decode tok/s is
`steps/s × accepted_length`, and accepted length carries ~±6% run-to-run noise that has
nothing to do with the configuration being tested. The step rate is counted, not derived.

**Single captures are labelled as such.** Where a number comes from one pass it is called
soft in the text beside it. The only results here treated as tight are the ones with a
control arm or an exact counter behind them.

**Acceptance figures come from vLLM's own `spec_decode` counters**, diffed before and after
each run, not from the benchmark client. Under greedy sampling they are exact rather than
sampled — the control arm of the SPECTOK sweep reproduced them bit-for-bit.

**These are single-card numbers.** Depth and draft-head advice in particular does not
transfer across card count; the upstream recommendation of speculation depth 7 is sound on
2×R9700 and loses here.

**Isolated-kernel wins are not end-to-end wins.** The tile sweep here reports per-GEMM
speedups of tens of percent on shapes that are a fraction of a percent of total work. Those
numbers are real and the end-to-end effect is still nil. A flat benchmark after a tile change
is the expected outcome, not a regression — see the caveats in that folder's notes.
