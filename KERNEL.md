# Kernels

This page covers the GPU kernel work in this repository: what changed, why, and what was measured
and turned down. Everything here lives in [`r4d_kernels/`](r4d_kernels/). The launcher applies all of
it by default (`R4D_RX13=1` in `startup-qwen3.8-27b-mxfp4.sh`).

All measurements were taken on one AMD Radeon AI PRO R9700 (gfx1201 / RDNA4, 32 GB) serving
Qwen3.8-27B MXFP4, with:
- a DFlash2 x7 drafter and fp8 KV cache;
- `MAXSEQS=4`;
- `stilldeadcode/vllm-radiance:0.9.3` (vLLM 0.27.1, V2 model runner);
- October 2026.

## The short version

| What | File | Effect | Exactness |
|---|---|---|---|
| Exact-decay GDN chunk scan (libr4d #4 fix) | `r4d_kernels.patch` | Correct prefill on chunks whose gate span exceeds 160 (46-94% output error before). Prefill speed unchanged. | Passes an fp64 reference, 12/12 cases |
| Exact-wide prefill attention (DSPLIT) | `r4d_kernels.patch` | 1.26-1.42x on final prefill chunks of 64 tokens or fewer over a deep cache | Bit-identical to the unsplit kernel |
| Prefill block-table prefetch clamp | `r4d_kernels.patch` | Stops a prefetch one block past the sequence end | No arithmetic change |
| Lazy GDN, prefill invalidation (mode 2) + stale-stash counters | `r4d_kernels.patch` | Makes lazy GDN safe to serve: +13.2% KV, +7% decode | Resume through the offload tier is token-identical to a cold run |
| `_st` chunk-scan binding (state width as an argument) | `r4d_kernels.patch` | Lets an fp16 GDN state be read natively (fp16 is the default since 2026-10-08). | Same arithmetic |
| Lazy-GDN materialize: no stash needed when nothing is replayed (rx17) | `r4d_kernels.patch` | Fixes garbage after an offload-tier resume whenever a prefill step spans more than one block -- what broke fp16 state and any CHUNK above one block. Enables fp16 + CHUNK=3532. | Resume through the tier bit-identical to cold at fp32 4956 and fp16 3532; decode path unchanged |
| MXFP4 W4A8 decode band, M 9-64 | `radiance_mxfp4_fp8.patch` | o_proj/out_proj -5..-6% kernel time with two concurrent sequences | Single-stream (M <= 8) untouched, byte for byte |
| DFlash2-W4A16 drafter path (2026-10-11): deferred-quant load, W4A16 Triton tiles + split-K + tiled layout, split-KV drafter attention, int4 fusions, prompt-lookup draft | `kv-cache/patches/` (six patches + two modules) | Drafter VRAM -0.78 GiB (KV pin 11.5e9 -> 12.65e9); decode step 35.9 -> 33.9 ms | Same acceptance (tokens/update 4.56 vs 4.57); lookup draft is lossless by construction |
| MXFP4 split-K reduction fence (2026-10-11) | `radiance_mxfp4_fp8.patch` | Correctness: the split-K decode reduction could read partial sums before other threads had stored them (rare, load-dependent wrong output). Same fix as SlyBase 1.1.1. | Fixes a race; the arithmetic is unchanged |

**Released configuration** (2026-10-11: everything above, launcher defaults, `MAXSEQS=4`, CHUNK=3568),
BetterBench 0.6.0, 20 passes, at BetterBench's default sampling, temperature 0.7 (BETTERBENCH-20261011.md):

| Metric | Result |
|---|---|
| Decode | 138.0 t/s (update p99 36.5 ms) |
| Concurrency 1 / 2 / 3 / 4 | 124.7 / 218.1 / 273.5 / 333.7 t/s aggregate |
| Prefill at 1.5k / 6k / 12k / 24k / 47k / 94k / 132k real tokens | 2922 / 3123 / 3167 / 3053 / 2863 / 2471 / 2223 t/s |
| Stalls | none |

The 2026-10-08 release (BETTERBENCH-20261008.md) ran at temperature 1.0: 132.5 t/s decode, 116.9 / 209.4 at
1 / 2 clients. Sampling temperature moves decode through acceptance, so the two are not directly
comparable. Its prefill was +7.4% at 16k to +11.9% at 128k over the release before it (fp32, CHUNK=2048),
measured the same night. The 2026-10-06 figures (BETTERBENCH-20261006.md) used BetterBench 0.4.0.

These numbers include the non-kernel work listed further down; the gains are not separable per kernel.

Where decode and prefill time goes today (a torch-profiler audit, 2026-10-11, and its review against the
upstream trees) is in ROADMAP.md section 2. The MXFP4 decode GEMM streams at ~544 GB/s, at the bandwidth roof; the
tiled prefill GEMM runs at ~243 TF/s, 59% of the fp8 matrix peak. Triton kernels make their first calls during a
boot warm-up (`JIT_WARMUP=1`), not on the first real requests.

## Why libr4d at all

On this card the radiance overlay serves the model through **libr4d**, StillDeadcode's HIP kernel
library for RDNA4: GDN, R4D paged attention and the small GEMMs. Stock vLLM does not run this model
on gfx1201. Most of the decode and prefill time is spent in libr4d and in ggz14's MXFP4 W4A8 GEMM.
That is where correctness bugs and speed come from, so that is where this work went.

The build is libr4d **v0.5.0** plus one patch, `r4d_kernels.patch`. The launcher clones libr4d,
applies the patch and builds it inside the serving image once per image and GPU arch. The result is
cached under `~/.cache/radiance-libr4d/v0.5.0-p<patch sha>-<image key>`.

## What is in `r4d_kernels.patch`, and why

### 1. The GDN chunk scan that does not lie (libr4d #4)

libr4d's gated-delta-net prefill scan splits each decay factor e^(g_i - g_j) at the chunk midpoint
and clamps each half at e^80. Any chunk whose gate span exceeds 160 then has its terms silently
attenuated. The output stays **finite but wrong**: 46% error at span 200, and 94% at span 2,000,
against an fp64 sequential recurrence. On this model 51 of 2,304 heads cross that span, and 20 cross
it on every chunk. Only prefill is affected.

deadcode's radiance engine (Apache-2.0) rewrote the scan so that every decay factor it forms is
at most 1, with no clamp. We took that kernel unchanged, removing only the engine's plugin-ABI
wrapper. It passes all 12 cases of the fp64 check (output error 0.23-0.36%, including CUDA-graph
replay), at fp32 and fp16 state. Prefill speed was within noise of the clamped kernel. A separate
run on mtstanfield's stack found the same.

**Choice:** we fixed correctness first, even though it brought no speed. A bug that leaves no trace
in the output is the worst kind for an agent workload, where one wrong recall poisons the rest of the
session.

### 2. Exact-wide prefill attention (DSPLIT)

A short final prefill chunk over a deep cache, such as an agent's follow-up turn on a 100k context,
launches too few workgroups to fill the GPU. Crssz/r4dx (MIT) split each query block's work across
`DSPLIT` workgroups. Each workgroup computes the **full** QK and softmax and **1/DSPLIT of PV**, so
no partial results are ever merged, and the output is bit-identical to the unsplit kernel.

We **merged** the technique into our existing prefill kernel instead of taking r4dx's file. Ours
carries ggz14's 8-bit QK8/PV8 legs (`R4D_ATTN_FP8=3`, the launcher default), and r4dx's file does
not. Swapping files would have dropped them.

The launch law picks DS from the grid:
- DS4 when there are at most 4 workgroups per sequence and the context is 2k or more;
- DS2 when there are at most 16 workgroups and the context is 2k or more;
- `R4D_PREFILL_DSPLIT=1/2/4` forces a value.

The thresholds sit at the measured crossovers.

**Measured, 10 shapes:**
- query 16-2,048 tokens, context 4k-100k;
- fp8 and f16 legs;
- **bit-identical** to DS=1 everywhere;
- 1.26-1.42x for chunks of 64 tokens or fewer;
- no change for large chunks, which already fill the GPU.

The check is reproducible: `r4d_kernels/tests/dsplit_bench.py` and `dsplit_cmp.py`.

**Turned down: r4dx's full exact-wide kernel** (12 warps, 96-key tiles, 2.0-2.6x on the same
chunks). Our QK8 path stages K as raw bytes in a single wide fetch, and fewer warps or wider tiles
make that multi-pass. It is worth porting, but it needs that K path reworked first (see ROADMAP.md).

**Turned down: r4dx's split-KV prefill.** It is faster still, but it merges partial softmax results,
which changes the fp32 summation order. A resumed final chunk would then no longer be bit-identical
to a cold run. This repository's KV-cache work treats bit-identical resume as a hard requirement
(KV-CACHE.md), so a kernel that breaks it only goes in behind an explicit exactness policy.

### 3. Lazy GDN, made safe to serve

ggz14's lazy GDN keeps one base state plus one candidate stash per sequence, instead of a snapshot per
draft token. With fp32 state that is **+13.2% KV (228,737 -> 259,011 tokens) and +7% decode**, for a
1-3% cost on short prefill. ggz14 reported multi-turn corruption from recycled stash blocks (0cadf57).
The fix here is a third `gdn_lazy_materialize` mode that zeroes a prefilling request's stash headers
in every row-split region, plus counters that record stale-stash events. The vLLM side is wired by
`kv-cache/patches/patch_gdn_lazy_invalidate.py`. The launcher refuses to serve lazy GDN if that patch
fails.

**Validation:**
- A concurrent two-stream multi-turn copy soak of 72 rounds (5,472 turns) with lazy on, then 37
  rounds with lazy off as the control. Failures appeared at the same rate in both, and all of them
  were the uniform-decode bug (below), not lazy GDN.
- Lazy drift against eager stays at or below the spec on/off noise floor.
- A resume from the offload tier is token-identical to a cold run.

**Limit:** ggz14's original corruption never reproduced here, even with the fix off. The fix is
therefore unproven rather than disproven, and the soak is what lazy GDN rests on.

### 4. Smaller pieces

- **Block-table prefetch clamp** (deadcode's engine): the prefill kernel's next-tile prefetch reads
  `bt[min(i, last block)]` instead of running one entry past the sequence.
- **`_st` binding**: the chunk scan with the state width as an argument, so an fp16 GDN state is read
  and written natively instead of being widened in Python. It is active with the default fp16 state
  (`patch_gdn_state_fp16.py`); an fp32 state takes the old path.

## The MXFP4 decode band (`radiance_mxfp4_fp8.patch`)

ggz14's MXFP4 W4A8 kernel picks its decode launch (K-split, K-tile) from a table that was tuned
mostly at M <= 8, i.e. one sequence with 7 drafts plus 1. With two agents the verify batch is
M = 16. SlyBase re-measured every cell of M in (8, 64] on these exact shapes and found two things:
- BK=64 loses everywhere in that band;
- o_proj/out_proj (K = 6144) at M 16-24 want a 2-way K split, not 4: -6.3% and -5.5% kernel time.

The patch adds that branch to the shipped launch code and nothing else. **M <= 8 is untouched**, so
single-stream decode runs byte-identical kernels.
- Runtime off switch: `RADIANCE_MXFP4_DECODE_TUNE16=0`.
- Build-time: `MFX_PATCH=` (empty) compiles the clone's kernel unchanged.

### Split-K reduction fence (correctness, 2026-10-11)

The split-K decode path has each K-slice block write its partial sums, bump a per-tile counter, and
the last block to arrive reduces them. ggz14's kernel fenced on thread 0 only before the counter
bump. A fence orders only the stores of the thread that runs it, and `__syncthreads` is workgroup
scope, so the reducing block on another CU could read partials that other threads had not drained
yet. Now every thread fences, the bump is an agent-scope acquire-release atomic, and the reducing
block fences (acquire) before it reads. SlyBase found and fixed the same race in their 1.1.1
(`b37a2e8`). It applies with `R4D_RX13=1` (the default); `MFX_PATCH=` (empty) compiles ggz14's
kernel without it.

The launcher patches a **copy** of `radiance_mxfp4_fp8.hip` from your ggz14 clone, keyed on the content
of both files. Your clone is never edited, and if ggz14 changes the file and the patch stops applying,
the boot stops and says so instead of compiling something stale.

## The 4-bit DFlash2 drafter (default since 2026-10-11)

`syvai/Qwen3.8-27B-DFlash2-W4A16` is the DFlash2 drafter quantised to 4-bit weights (compressed-tensors
W4A16). It is 1.2 GB on disk against the FP8 drafter's 2.0, frees 0.78 GiB of VRAM, and the launcher
gives that to the KV pool (`KV_MEM` 11.5e9 -> 12.65e9: 369,614 tokens at MAXLEN 262144, 0.33 GiB spare at the
stress peak). Six patches make it
load and run fast; each is inert for the FP8 drafter:

| Patch | What it does | Source |
|---|---|---|
| `patch_dflash_deferred_quant.py` | DFlash2's fused context-KV build reads a raw `qkv_proj.weight`, which a W4A16 checkpoint does not have (boot crash). Defers the build until the weights are processed and recovers the dense rows through the layer itself. | this repository |
| `patch_w4a16_tiles.py` | Per-shape Triton tile table, split-K and a tiled weight layout for the drafter GEMMs: 3.29 -> 1.74 ms per decode step. | SlyBase |
| `patch_attn_drafter.py` + `radiance_attn_drafter.py` | Split-KV launch for the drafter's sliding-window attention: 1.37 -> 0.21 ms per step at ~119k context. | SlyBase |
| `patch_w4a16_fuse.py` | gate_up + SiLU fused, context-KV and conv projection on the int4 rows (no bf16 copies). | SlyBase, re-anchored here |
| `patch_w4a16_prefill.py` | Tiles for the drafter's two prefill-sized GEMMs (fc and context-KV, M up to CHUNK): BLOCK_K 32 and a GROUP_M program-id swizzle, from tcclaviger's R9700 configs. Bit-identical to stock; fc 19.7 -> 9.5 ms per 3568-token chunk, about 1% faster prefill end to end. `RADIANCE_W4A16_PREFILL=0` turns it off. | this repository |
| `patch_lookup_draft.py` + `radiance_lookup_draft.py` | When the text being written is a verbatim copy of the prompt, propose the copy instead of the drafter's guess (entry rule 16/6/5, picked by a replay of real agent sessions). Verification is unchanged, so it is lossless. `RADIANCE_LOOKUP_DRAFT=0` turns it off. | SlyBase |

Measured on one R9700, BetterBench 0.6.0 decode, 20 passes, same session, before `patch_w4a16_prefill.py` existed:

| | FP8 drafter | W4A16 drafter + all five |
|---|--:|--:|
| tokens per update (acceptance) | 4.565 | 4.562 |
| decode step | 35.9 ms | 33.9 ms |
| weighted decode | 130.5 t/s | 136.0 t/s |
| VRAM after boot | 33.59 GB | 32.74 GB (W4A16 alone) |

Tokens per update across FP8 runs spans 4.39-4.57 from compile to compile, so "same acceptance" means
inside that band. On verbatim-copy tasks the lookup draft raises it from ~4.2 to ~7.6 tokens per update.
`DRAFTER=<models>/Qwen3.8-27B-DFlash2-FP8` restores the FP8 drafter and its 11.5e9 KV pin.

## Turned down after measurement

| Candidate | Source | Result | Why it stays out |
|---|---|---|---|
| libr4d `gemm_mxfp4a8_nt_m64` for the decode band | libr4d main | **-7.4% decode** | radiance's own decode kernel is already at the memory-bandwidth roof |
| PyTorch TunableOp on the bf16 GEMMs | PyTorch | No gain; the drafter's small GEMMs got slower | lm_head is bandwidth-bound; nothing to tune |
| `RADIANCE_SKINNY_GEMM=all` | ggz14 (paro branch) | Drafter acceptance fell by more than the kernel gained (ggz14's measurement) | Net loss |
| ParoQuant / MXFP6 kernels | ggz14 Codeberg main | Need different checkpoints | Not applicable to this model file |

**The 2026-10-06 sweep.** Every kernel file in 25 repositories in the libr4d / radiance ecosystem was
compared by git blob hash against libr4d v0.5.0, libr4d main, deadcode's engine and ggz14, and every
differing file was diffed against this build. Only the items above and the open ones in ROADMAP.md
apply to this model on this card. The largest open one is deadcode's GDN decode rewrite (anchor
state, conv fold, non-temporal store). Its anchor state is what lazy GDN already does here; the 2026-10-11 review puts the rest (conv fold,
gated norm + quant, non-temporal store) at about 2-3% of a step (ROADMAP.md section 2, item 2). It is also costly and correctness-critical to port.

## Non-kernel work behind the same numbers

These are Python patches to vLLM / radiance, in `kv-cache/patches/` (applied at container start).
They are not kernels, but the benchmark numbers above depend on them.

- **Uniform-decode guard.** vLLM 0.27.1's V2 runner replayed the speculative-decode CUDA graph over
  any prefill chunk of exactly 1 + num_speculative_tokens tokens. That produced an empty reply for
  about 1 prompt in one block size (1,648 tokens when it was found, 880 now). The guard is fatal if it fails to apply. Regression test:
  `kv-cache/tools/lenprobe.py`.
- **DFlash2 sampling RNG** (vllm-radiance PR #8 / vLLM #54282): the drafter's selector gets its own
  RNG stream. Shared noise biased rejection sampling by up to ~2% per position.
- **Global top-256 verify head** (vllm-radiance PR #9) and the **int2 draft head** (`FAST_DRAFT=1`).
  Together with the fixes needed to make them actually arm on 0.27.1, they give **+18% decode**.
  Quality check without logprobs (`kv-cache/tools/quality_ab.py`):
  - open-ended texts: 24/24 identical against the exact head;
  - arithmetic: 40/40 both;
  - lookups: 29/30 both.
- **Input embedding in host RAM** (`EMBED_HOST=1`): the 2.37 GiB table moves to pinned host memory
  behind a UVA view. That is **+27% KV** (259,011 -> 329,035 tokens) at no measured speed cost,
  because lookups are a few rows per token.
- `RADIANCE_W4=0`, which keeps radiance_w4 from freeing the drafter weights under `FAST_DRAFT=1`
  (that was a boot loop). Also a ROCm load-time allocator split and vLLM #55450's align-mode
  retirement fix.

## Reproducing the checks

| Check | Tool |
|---|---|
| DSPLIT bit-identity and speed | `r4d_kernels/tests/dsplit_bench.py`, `dsplit_cmp.py` (inside the serving image, against the built `r4d.so`) |
| GDN scan vs fp64 | `check_gdn_extreme_decay.py` from magiccodingman/vllm-radiance#8 |
| Cold vs offload-tier resume | `kv-cache/tools/val_resume.py --evict 12` |
| Lazy GDN, multi-turn | `kv-cache/tools/mt_lazy_gate.py --copy` (run two at once for concurrency) |
| Uniform-decode regression | `kv-cache/tools/lenprobe.py` |
| Output quality, approximate heads | `kv-cache/tools/quality_ab.py` |
| Drift vs a reference | `kv-cache/tools/klstat.py`, `errmargin.py` |

## Sources and licences

The patch headers carry the full provenance; `r4d_kernels/README.md` has the table. In short:
- deadcode's engine kernels are Apache-2.0.
- Crssz/r4dx's DSPLIT is MIT; its notice ships as `r4d_kernels/LICENSE.r4dx`.
- This repository's own changes are Apache-2.0.
- SlyBase's README states its licence is "same as upstream vllm-radiance" (Apache-2.0) but ships no
  LICENSE file; the SlyBase-derived patches keep their provenance headers.
- tcclaviger/vllm: the prefill tile settings in `patch_w4a16_prefill.py` (BLOCK_K 32, GROUP_M swizzle).
- libr4d v0.5.0 (StillDeadcode), ggz14's radiance extras and MXFP4 kernel, J-Nova's rx12 rebase, and
  SlyBase's decode table publish **no licence**. They are redistributed here with attribution and
  without a grant. If you are one of those authors and want something changed or removed, open an
  issue.
