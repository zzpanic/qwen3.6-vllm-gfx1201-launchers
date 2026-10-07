# Roadmap

Where this stack stands and what is still worth doing, as of 2026-10-08. Everything below was
measured on one AMD Radeon AI PRO R9700 (gfx1201, 32 GB) serving Qwen3.8-27B MXFP4 with a DFlash2 x7
drafter, fp8 KV cache, `MAXSEQS=2`, `vllm-radiance:0.9.3` (vLLM 0.27.1, V2 model runner). The launcher is
`startup-qwen3.8-27b-mxfp4.sh`; KERNEL.md, KV-CACHE.md, SYSTEM.md and BETTERBENCH-20261008.md cover the detail.

## Where we are

The current shape:
- **Kernels and GDN:** libr4d v0.5.0 + `r4d_kernels` (exact-decay GDN chunk scan, exact-wide prefill, block-table clamp),
  fp16 GDN state, lazy GDN (with the rx17 materialize fix), four 880-token blocks per prefill step
  (CHUNK=3532).
- **Speculative decoding:** the int2 draft head plus the global top-256 target verify head.
- **Memory:** the input embedding in pinned host RAM.
- **Correctness:** the uniform-decode guard and the DFlash2 sampling-RNG fix.
- **ROCm allocator:** the load-time allocator split.

BetterBench 0.6.0, 20 passes, temperature 1.0 (BETTERBENCH-20261008.md):

| | |
|---|---|
| Decode, combined | 132.5 t/s (update p99 38.5 ms, no stalls) |
| Concurrency 1 / 2, aggregate | 116.9 / 209.4 t/s |
| Prefill 2k / 8k / 16k / 32k / 64k / 128k | 2896 / 3119 / 3190 / 3084 / 2870 / 2470 t/s |
| KV pool | 331,759 tokens at KV_MEM 11.5e9 (measured at MAXLEN 204,800) |

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
unaffected. Switching `MAMBA_SSM_FP16` changes the disk tier's geometry (README.md, step 4).

---

## 2. Kernel work

| Item | Source | Expected value | Cost / risk | Status |
|---|---|---|---|---|
| **GDN decode rewrite**: anchor state (one state per sequence plus the replayed accepted prefix, single-write, with a shift marker that survives vLLM's page migration), the decode conv folded into the update, a non-temporal state store chosen per launch | deadcode radiance engine 1.0.x | Claimed about 14% of a decode step. It overlaps lazy GDN, which already gave about +7%, so the gain on top of that is unknown. | High: correctness-critical state handling, and it needs matching `radiance_gdn.py` integration | Not started |
| **Full exact-wide prefill**: 12 warps, 96-key tiles | Crssz/r4dx | Their 2.0-2.6x on 64-token chunks against our 1.26-1.42x. Only affects final prefill chunks of 256 tokens or fewer. | Medium: the QK8 raw-byte K staging assumes a one-pass wide K fetch, and fewer warps or wider tiles make it multi-pass | DSPLIT-only part shipped |
| **Decode-attention extras**: vectorised split-KV merge (bit-identical), attention output gate and fp8 quantisation fused into the merge epilogue, K prefetch 2 instead of 4 | deadcode engine | Under 2% of a step | Medium: the fusion needs the vLLM output path changed | Not started |
| **MXFP4 skinny GEMM launch sweep** (`gemm_mxfp4a8_nt_m64`) | libr4d main | Measured -7.4% decode at the default config. A sweep of `_r4d_cfg` is untried. | Low cost, low odds: radiance's own decode kernel is at the bandwidth roof | Rejected as shipped |
| **Split-KV prefill** (not exact) | Crssz/r4dx | Larger short-chunk gains than DSPLIT | Changes fp32 summation order, so a resumed final chunk is no longer bit-identical to a cold run. Only worth it with an explicit exactness policy. | Not adopted |
| **M1/M8 route alignment**: serial decode vs speculative verify | vllm-radiance #11 | Reproducibility, not quality: spec on/off agreement is 99.13% today, with flips only at near-ties | About 33k lines | Not adopted |

The tools that check kernel work are `lenprobe.py`, `val_resume.py`, `quality_ab.py` and `errmargin.py` /
`klstat.py`. Exact-wide changes also need a byte-identity harness against the unsplit kernel, like the
one used for DSPLIT.

---

## 3. Speculative decoding

- **Drafter attention split-KV tuning** (SlyBase `radiance_attn_drafter`). The DFlash2 drafter's
  sliding-window attention (5 layers, window 2048, non-causal) runs on vLLM's stock Triton launch, which
  under-fills the GPU at long context. Value: more draft speed at depth. It is written for vLLM 0.29 and
  needs porting.
- **Prompt-lookup drafting** (SlyBase `radiance_lookup_draft`). Overrides the DFlash2 draft with an n-gram
  match from the context. Agent work (editing code, repeating tool output) is copy-heavy, which is where
  this raises acceptance. Measure it with BetterBench's `file_edit` / `code` categories and real agent traces.
- **Load-adaptive draft depth on the V2 runner.** The old dynamic drafter (`RADIANCE_DYNAMIC_DRAFT`) hooks
  the V1 runner and does nothing on V2. With two concurrent agents, per-stream decode drops from 144.8 to
  138.0 t/s, which is small, so this is low priority unless concurrency goes up.
- **`MAXSEQS=3`.** ggz14's single-card profile runs 3 sequences. The 329k pool now leaves room for that
  at long context. Measure concurrency 3 against per-stream latency.

---

## 4. Capacity and memory

- **VRAM headroom.** The stress peak, with two 158k sequences resident plus a 2048x2048 image, is
  31.23 / 31.86 GiB, leaving 0.62 GiB spare. `KV_MEM=11.3e9` would keep about 1 GiB spare for about 315k
  tokens. Decide after more real-world use.
- **Vision tower to host.** The stock V2 runner never installs the UVA offloader. A 0.29 backport exists
  (`patch_tower_offload.py`), but VRAM *rose* when it was tried, so it needs a look at what was allocated.
  mtstanfield reported about +100k tokens with the tower and an fp8 embedding moved together.
- **fp8 KV scales.** The checkpoint ships none, so the per-tensor scale is 1.0. K is QK-normed, so it is
  probably fine. To settle it, log per-layer K/V maxima and the fraction below 2^-6 for a few requests.
- **Short-prompt prefill under lazy GDN.** 2k prefill is about 3% slower with lazy on. The suspect is the
  synchronous stale-counter poll on every prefill invalidation; move it off the prefill path.

---

## 5. Correctness and validation debt

- **The merged launcher has not been boot-tested.** `startup-qwen3.8-27b-mxfp4.sh` now carries the
  served stack; it dry-runs to the served command, but it has not served yet. Boot it, then run
  `lenprobe.py`, `val_resume.py --evict 9` and a short BetterBench before tagging a release.
- **Full-length context is untested.** The default is now 262,144 tokens. The stress test held two 158k
  sequences and BetterBench prefilled 94k; a single 262k prefill and resume (GPU pool and the 19 GiB
  RAM tier) has not been run.

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
- **Host RAM.** About 27 GB at peak with a 16 GiB RAM KV tier and the 2.37 GiB pinned embedding; the
  shipped default is a 19 GiB tier (one full 262,144-token context), so plan on ~30 GB and size the VM at
  32 GiB or more. A 2x tier (37 GiB, two full-length sessions) would need ~56 GiB.
- **Disk tier volume.** 128 GiB or more recommended (KV-CACHE.md); the reference box runs 94 GiB.

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
