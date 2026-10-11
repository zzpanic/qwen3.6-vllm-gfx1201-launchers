# Roadmap

Where this stack stands and what is still worth doing, as of 2026-10-11. Everything below was
measured on one AMD Radeon AI PRO R9700 (gfx1201, 32 GB) serving Qwen3.8-27B MXFP4 with a DFlash2 x7
drafter, fp8 KV cache, `MAXSEQS=4`, `vllm-radiance:0.9.3` (vLLM 0.27.1, V2 model runner). The launcher is
`startup-qwen3.8-27b-mxfp4.sh`; KERNEL.md, KV-CACHE.md, SYSTEM.md and BETTERBENCH-20261011.md cover the detail.

## Where we are

The current shape:
- **Kernels and GDN:** libr4d v0.5.0 + `r4d_kernels` (exact-decay GDN chunk scan, exact-wide prefill, block-table clamp),
  fp16 GDN state, lazy GDN (with the rx17 materialize fix), four 880-token blocks per prefill step
  (CHUNK=auto, 3568 at four sequences), the MXFP4 split-K fence fix.
- **Speculative decoding:** the int2 draft head plus the global top-256 target verify head.
- **Memory:** the input embedding in pinned host RAM; the 4-bit DFlash2 drafter (-0.78 GiB) and KV pin 12.65e9;
  checkpoint retention (a 16 GiB RAM tier and a 128 GiB disk at 75% hold what dense storage needed twice that for).
- **Correctness:** the uniform-decode guard, the DFlash2 sampling-RNG fix, canonical prefill chunks
  (cached == uncached at fp16 and fp32, KV-CACHE.md) and the 1-token prefill fix.
- **ROCm allocator:** the load-time allocator split.
- **Startup:** CUDA graphs captured up to `MAXSEQS x 8` tokens, and a JIT warm-up after `/health`.

BetterBench 0.6.0, 20 passes, temperature 0.7 (BETTERBENCH-20261011.md):

| | |
|---|---|
| Decode, combined | 138.0 t/s (update p99 36.5 ms, no stalls) |
| Concurrency 1 / 2 / 3 / 4, aggregate | 124.7 / 218.1 / 273.5 / 333.7 t/s |
| Prefill at 12k / 47k / 132k real tokens | 3167 / 2863 / 2223 t/s |
| KV pool | 369,614 tokens at KV_MEM 12.65e9, MAXLEN 262,144 |

Each item below says what it is, why it is worth doing, what is already known, and how to tell whether it
worked. The tools named live in `kv-cache/tools/`.

---

## 1. FP16 SSM state: DONE (2026-10-08), now the default

The 2026-10-05 "fp16 is broken through the KV offload tier" finding was a misdiagnosis. The fault was in
lazy GDN's materialize kernel: on a prefill step spanning more than one mamba block, a request resuming
from the tier skipped the state migration (it required a stash block even with nothing to replay) and
ran from a stale running block. fp16's 880-token block made `CHUNK=2048` two blocks per step, so fp16
hit it; fp32 hits it too at `CHUNK=4956`. libr4d rx17 (`r4d_kernels.patch`) fixes it.

Measured with rx17 (one R9700, the served stack):
- `val_resume --evict`: cold vs tier-resumed bit-identical at fp16 CHUNK=3532 and fp32 CHUNK=4956
  (rx16: 0/8 prompts identical);
- concurrent deep multi-turn gate (2 requests in flight, 6 x 70k-token conversations, ~3M tokens
  restored from the tier, copy-checked every turn): 48/48 (rx16 control: 7/48, gross garbage);
- drift, fp16+3532 vs fp32+2048, cold and greedy on identical prompts: top-1 agreement 99.75% at ~25k,
  100% at ~70k, 99.91% at ~141k (spec on/off noise floor 99.13%); mean |dlogprob| 0.0032 / 0.0007 /
  0.0012 nats -- it does not grow with depth;
- prefill +7.4..+11.9% at 16k-128k, decode steps +0.6..1.7%, KV pool 331,759 tokens at KV_MEM 11.5e9.

**Still open:** at ~141k a verbatim-copy score fell 14 -> 9 of 28 lines from two near-tie token flips in
two of eight prompts -- one sample, but re-measure with a larger set before calling long-range copying
unaffected. Switching `MAMBA_SSM_FP16` changes the disk tier's geometry (README.md, step 3).

---

## 2. Kernel work

**Kernel audit, 2026-10-11** (torch-profiler traces of the shipped launcher, `PROFILE_DIR=1`: one stream of
decode, four streams of decode, and a 31.5k-token prefill). GPU time by kernel:

| Scenario | Top kernels (share of GPU time) |
|---|---|
| Decode, 1 stream | MXFP4 decode GEMM 75.4%, drafter W4A16 split-K 6.3%, int2 draft head 4.5%, GDN lazy update 2.8%, fused add+RMS+quant 1.5%, GDN conv 1.4%, `torch.topk` 1.5% |
| Decode, 4 streams | MXFP4 decode GEMM 67.9%, GDN lazy update 5.7%, drafter split-K 5.7%, int2 draft head 5.3%, drafter W4A16 (tile table, no split-K) 1.9%, `torch.topk` 2.3% |
| Prefill, 31.5k | MXFP4 tiled GEMM 58.8%, attention prefill 12.9%, MXFP4 folded GEMM 7.6%, fused SiLU+quant 5.1%, GDN chunk scan 2.8%, drafter W4A16 (stock kernel, since fixed) 2.1%, offload-tier copy 1.6% |

The same day the audit was re-checked against the traces' launch grids, the kernel sources and the current
upstream trees (deadcode radiance engine 1.3.0, tcclaviger/vllm 29.08.10, libr4d main). Matrix peaks below are
the radiance engine's gfx1201 measurements (f16 207, fp8 412, iu8 407 TF/s); a streaming kernel reaches about
600-634 GB/s of the card's 640.

Where the big kernels stand:
- **Decode GEMM:** ~544 GB/s, 12.9 GB of 4-bit layer weights in 23.8 ms per verify step, 86-91% of what a
  streaming kernel reaches. At the roof.
- **Prefill GEMM (tiled):** ~243 TF/s on every large projection, 59% of the fp8 peak. The out/o projections
  (K 6144) take the folded kernel instead, at ~175 TF/s (item 5).
- **Attention prefill:** f16 matrix ops for both products, ~153 TF/s, 74% of the f16 peak. Closed (below).
- **Int2 draft head:** ~510 GB/s at one stream counting its per-group scale and zero point, ~80% of DRAM. At
  four streams (M=32) it takes about 1 ms a call, bound by the f16 matrix rate rather than by memory.
- **Drafter split-K GEMMs:** ~515 GB/s.
- **Launches:** ~1,100 kernels per decode step. The GPU idles 15% of decode time, almost all of it in gaps under
  50 µs (~4 µs a kernel, ~4.5 ms a step at one stream). At four streams the GDN and attention kernels run
  outside the CUDA graph (46 of 48 GDN updates a step).
- **Offload-tier copy:** it overlaps no compute (item 1).

All eight kernels that used to compile during the first requests now compile during the boot warm-up
(`JIT_WARMUP`). Several items below are worth 1-2% each; none is quick. In order of expected value:

| # | Item | Source | Expected value | Cost / risk | Status |
|---|---|---|---|---|---|
| 1 | **Offload-tier copies serialise with compute.** The KV store copy (`hipMemcpyBatchAsync`: a blit kernel on its own stream, ~20 ms per 3520-token chunk over PCIe 3.0) overlapped no compute in the prefill trace. The likely cause is `GPU_MAX_HW_QUEUES=1`, which puts every stream on one hardware queue | kernel review 2026-10-11 | 1.5% of a cold 31.5k prefill. Tier loads probably stall other streams' decode the same way (unmeasured) | Low to medium. Try a copy-engine (SDMA) path before the queue count: two queues was a per-boot lottery here (one boot in three ran ~30% slower), so any queue or stream A/B needs three or more boots per arm | Not started |
| 2 | **GDN decode, the parts lazy GDN does not cover.** The radiance engine's update kernel folds in the decode convolution (tap for tap, the same arithmetic) and the gated RMS norm + quantiser, and picks a non-temporal state store per launch by flood size. Its anchor state (one state per sequence plus the replayed accepted prefix, about 14% of a step there) is what lazy GDN already does here | deadcode radiance engine 1.3.0 | Today the separate conv update and gated norm + quant cost 1.4% + 0.6% of a one-stream step plus 96 of its ~1,100 launches: about 2-3% of a step. The non-temporal hint is worth about 3% of a step on the engine's fp32 anchor state; this stack writes less state per layer (fp16), so expect less | Medium to high: a correctness-critical kernel. Outputs must stay bit-identical to today's path, and the stash and tier-materialize layout of lazy GDN must survive | Not started |
| 3 | **Draft and verify heads on the int8 matrix rate** (`r4d_gemm_w2a8_nt`, the same asymmetric group-128 2-bit format) | deadcode radiance engine 1.3.0 | The int2 head is 4.5% / 5.3% of a step (one / four streams) and compute-bound from four streams up. On iu8 it stays a weight stream to M=64: about 0.6% / 2% of a step, more at eight streams | Medium: new packing and an int8 activation quantiser. The verify head's top-256 shortlist must keep its top-20 retention (649 of 650 rows today) | Not started |
| 4 | **Benchmark the radiance engine's MXFP4 tiled GEMM** (`r4d_gemm_mxfp4a8_tiled`) against the one shipped here | deadcode radiance engine 1.3.0 | The tiled GEMM is 59% of prefill at 59% of the fp8 peak, and the new kernel has no published rate. Modest odds, large prize: every 10% is about 6% of prefill | Low to measure: it reads the same fragment-order weight layout (`RADIANCE_MXFP4_WPERM`); one GPU window | Not started |
| 5 | **Out/o projections on the tiled GEMM.** Their inputs come from the GDN gated norm + quant and the attention-output quant, which do not write the fragment-tiled activation layout, so they take the folded kernel (~175 against ~243 TF/s) | kernel review 2026-10-11 (ggz14's kernels) | The folded kernel is 7.6% of prefill. ggz14 measured the tiled kernel 12-16% faster at M >= 2048; at the other projections' rate this shape would take 28% less time: 0.9-2.1% of prefill | Low: add the tiled write that `add_rms_quant` and `silu_mul_quant` already have. The same fp8 bytes in a different order, so exact | Not started |
| 6 | **Fused SiLU + quant at 342 GB/s** (one pass already, one workgroup per row) | kernel review 2026-10-11; tcclaviger/vllm `clav_silu_quant` (the same per-token e4m3 op, wired into its MoE path) | 5.1% of prefill: about 1.9% at ~550 GB/s | Medium: keep the rounding contract (silu and product rounded to bf16, amax over the rounded product) and the tiled write. His kernel can be timed in his image first | Not started |
| 7 | **Launch count at decode**: ~1,100 kernels a step at ~4 µs each, 12-14% of decode time. At four streams the GDN and attention kernels run outside the CUDA graph; non-uniform verify widths falling back to piecewise graphs is the suspect, unverified | kernel review 2026-10-11 | The largest decode pool, drained in pieces: each 100 launches removed is about 0.4 ms a step (~1.2%) | High and spread out: fusions in the vLLM glue. Start with why four streams run eager | Not started |
| 8 | **GDN prefill kernels.** The radiance engine has reworked the chunk scan, conv prep and kkt solve since libr4d v0.5.0 | deadcode radiance engine 1.3.0 | 5.8% of prefill with the gated norm; the gain is unmeasured | High: libr4d is a correctness pin, so the full exactness gates (`val_resume.py --evict`, `turnbench.py`) | Not started |
| 9 | **`torch.topk` in decode.** The verify head scatters its 256 rescored logits into a 248,320-wide row of -inf, then the sampler top-k's the whole row | kernel audit + review 2026-10-11 | 1.5% / 2.3% of a step; about 0.65% / 0.9% is fixable by keeping the 256 candidates compact through the sampler, plus a row top-k (`r4d_rowtopk_bf16`, radiance engine) for the head's own selection | Medium: it touches the sampler; exact | Not started |
| 10 | **Drafter down projection at four streams.** At M=32 one drafter GEMM a layer (most likely down_proj) has a tile-table row but no split-K row, and runs at ~330 GB/s | kernel review 2026-10-11 | 1.85% of a four-stream step: about 0.6%. Nothing at one stream | Low: the same sweep as the prefill tiles; drafter only, so acceptance is the only risk | Not started |
| 11 | **Decode-attention extras**: vectorised split-KV merge (bit-identical), attention output gate and fp8 quantisation fused into the merge epilogue, K prefetch 2 instead of 4 | deadcode engine | Under 2% of a step | Medium: the fusion needs the vLLM output path changed | Not started |
| 12 | **Full exact-wide prefill**: 12 warps, 96-key tiles | Crssz/r4dx | Their 2.0-2.6x on 64-token chunks against our 1.26-1.42x. Only affects final prefill chunks of 256 tokens or fewer. | Medium: the QK8 raw-byte K staging assumes a one-pass wide K fetch, and fewer warps or wider tiles make it multi-pass | DSPLIT-only part shipped |
| 13 | **Drafter W4A16 GEMMs at prefill-sized M**: the drafter's fc and context-KV projections (M = CHUNK) fell through to the stock `_triton_w4a16_skinny_fmt_kernel` heuristic (fc ~30 TFLOP/s) | kernel audit 2026-10-11 | `patch_w4a16_prefill.py`: tcclaviger's BLOCK_K 32 + GROUP_M swizzle, bit-identical; fc x2.08 at M 3568, cold prefill +0.9-1.1% (4.7k-46k tokens). A bf16 dequant + hipBLASLt route is still ~19% faster on fc at M >= 2048, at a ~5e-3 drafter error and a scratch buffer | None: same kernel, same K loop | Done 2026-10-11 |
| 14 | **MXFP4 skinny GEMM launch sweep** (`gemm_mxfp4a8_nt_m64`) | libr4d main | Measured -7.4% decode at the default config. A sweep of `_r4d_cfg` is untried. | Low cost, low odds: radiance's own decode kernel is at the bandwidth roof | Rejected as shipped |
| 15 | **Split-KV prefill** (not exact) | Crssz/r4dx | Larger short-chunk gains than DSPLIT | Changes fp32 summation order, so a resumed final chunk is no longer bit-identical to a cold run. Only worth it with an explicit exactness policy. | Not adopted |
| 16 | **M1/M8 route alignment**: serial decode vs speculative verify | vllm-radiance #11 | Reproducibility, not quality: spec on/off agreement is 99.13% today, with flips only at near-ties | About 33k lines | Not adopted |

The tools that check kernel work are `lenprobe.py`, `val_resume.py`, `quality_ab.py` and `errmargin.py` /
`klstat.py`. Exact-wide changes also need a byte-identity harness against the unsplit kernel, like the
one used for DSPLIT.

---

## 3. Speculative decoding

- **Done 2026-10-11:** drafter attention split-KV (1.37 -> 0.21 ms per step at ~119k context) and
  prompt-lookup drafting (verbatim copies ~4.2 -> ~7.6 tokens per update), both ported from SlyBase with the
  W4A16 drafter (KERNEL.md).
- **Load-adaptive draft depth on the V2 runner.** The old dynamic drafter (`RADIANCE_DYNAMIC_DRAFT`) hooks
  the V1 runner and does nothing on V2. Per-request decode falls from 161 t/s alone to 145 with two clients and 116 with four
  (BETTERBENCH-20261011.md); how much of that a shallower draft would recover is unmeasured.
- **Done 2026-10-11: `MAXSEQS=4` is the default.** At temperature 0.7, 1 / 2 / 3 / 4 clients give 124.7 /
  218.1 / 273.5 / 333.7 t/s aggregate, 161 / 145 / 122 / 116 t/s per request (BETTERBENCH-20261011.md).

---

## 4. Capacity and memory

- **VRAM headroom.** At the default `KV_MEM=12.65e9` the stress peak (`MAXSEQS=4`: three concurrent ~110k
  prefills plus a 2048x2048 image in the fourth slot) is 31.53 / 31.86 GiB, 0.33 GiB spare; the transient over
  idle is +0.23 GiB. `KV_MEM=12.25e9` keeps 0.66 GiB spare for 357,813 tokens. Watch real-world use.
- **Vision tower to host.** The stock V2 runner never installs the UVA offloader. A 0.29 backport exists
  (`patch_tower_offload.py`), but VRAM *rose* when it was tried, so it needs a look at what was allocated.
  mtstanfield reported about +100k tokens with the tower and an fp8 embedding moved together.
- **fp8 KV scales.** The checkpoint ships none, so the per-tensor scale is 1.0. K is QK-normed, so it is
  probably fine. To settle it, log per-layer K/V maxima and the fraction below 2^-6 for a few requests.
- **Short-prompt prefill under lazy GDN.** 2k prefill is about 3% slower with lazy on. The suspect is the
  synchronous stale-counter poll on every prefill invalidation; move it off the prefill path.

---

## 5. Correctness and validation debt

- **The public launcher's own gates.** On 2026-10-11 it was booted cold and warm, benched (BETTERBENCH-20261011.md)
  and VRAM-stressed. The exactness gates (`turnbench.py`, `val_resume.py --evict`, `lenprobe.py`) ran on
  the production twin of this launcher, with the same patches, not on this script itself. Run on this
  script itself on 2026-10-11, on a direct boot of it (port 5899, throwaway disk folder): the built-in reaper deleted
  oldest-first down to its limit and touched nothing else; `lenprobe.py` 0 bad of 15; `val_resume.py --evict 12`
  bit-identical, 8/8 prompts, resumed from the tier (~689 MB loaded). `turnbench.py` was 21/21 exact but never
  left the GPU (the 369,614-token pool outgrew its workload), so it did not exercise the tier. The disk tier's raw read rate on `/kvcache` measured 695 then 531 MB/s
  (`tierreport.py --calibrate`, 80 block files, O_DIRECT; the spread is the virtual disk). Still open: the same gates
  through the llama-swap entry. A BetterBench
  re-run through the llama-swap entry with `patch_w4a16_prefill.py` is in BETTERBENCH-20261011.md.
- **Full-length context is untested.** The default is 262,144 tokens. The stress test held three ~110k
  prefills at once and BetterBench prefilled 132k; a single 262k prefill and resume (GPU pool and the
  16 GiB RAM tier) has not been run.

- **Lazy GDN's original corruption** (ggz14 0cadf57) has never been reproduced. Even a control with the
  invalidation fix off passes the multi-turn gates, and a concurrent soak with a lazy-off control showed
  no lazy-specific failure. Keep soaking (`mt_lazy_gate.py --copy`, concurrent).
- **Uniform-decode bug.** The V2 runner replayed the speculative-decode CUDA graph over a prefill chunk of
  exactly 1 + num_speculative_tokens tokens. It is guarded here (`patch_uniform_decode_guard.py`); check
  current vLLM and report or upstream it.
- **A quality gate for any approximate path.** Requests that ask for logprobs force the exact lm_head,
  so logprob-based drift tools cannot see the int2 head. Use `quality_ab.py`, which asks for none.

---

## 6. Operations

- **Disk-tier reaper alerting.** `KVCACHE_MIN_FREE_GB` sizes the free-space floor to the write rate.
  Write failures (ENOSPC) and the reaper's "volume too small" warning should also reach a notifier, not
  just the journal.
- **Host RAM.** About 27 GB at peak with the default 16 GiB RAM KV tier and the 2.37 GiB pinned embedding:
  size the box at 32 GiB or more.
- **Disk tier volume.** 128 GiB at 75% recommended (KV-CACHE.md); the reference box runs 94 GiB at 70%.

---

## Measured and not pursuing

| Item | Result |
|---|---|
| TunableOp | No gain. lm_head is at the bandwidth roof, and tuning made the small drafter GEMMs slower. |
| libr4d `gemm_mxfp4a8_nt_m64` for the decode band | -7.4% decode |
| Vision tower via `--cpu-offload-params visual` on 0.27.1 | Does nothing (the V2 runner never installs the offloader); with the backport, VRAM rose |
| `RADIANCE_SKINNY_GEMM=all` | A drafter acceptance tax larger than the kernel win (ggz14's measurement) |
| ParoQuant / MXFP6 kernels | Need different checkpoints |
| fp16 SSM state (as of today) | See section 1 |
| Int2 draft head retune at TP=1 | ~510 GB/s at one stream counting its per-group scale and zero point, ~80% of DRAM: about 0.2% left. The four-stream case is item 3 of section 2 |
| MXFP4 decode GEMM | 86-91% of achievable bandwidth. The radiance engine's newest decode kernel (`r4d_gemm_mxfp4a8_decode`) uses the same design (split-K across workgroups, LDS-staged weight, non-temporal loads); unmeasured here |
| Attention prefill | 74% of the f16 matrix peak. libr4d measured an fp8 P leg at 16% faster and 15x less accurate; tcclaviger/vllm also prefers libr4d's paged attention to its own `clav_attn` on this geometry |
