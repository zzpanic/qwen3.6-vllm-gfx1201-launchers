# DFlash2 overlay for vLLM 0.26.0 (`vllm-radiance:0.5.8`)

Ten Python files, bind-mounted read-only over the image's `vllm` package by
`startup-qwen3.8-27b-vllm.sh` whenever `SPEC=dflash2`. Without them the image cannot run
DFlash2 at all — it fails on an unknown model architecture.

## Why an overlay and not a newer image

DFlash2 landed upstream as **vLLM PR #52816**, merged 2026-08-21 as commit
`b389ac29465b33f9e9c534df221ea3c129e9793f` (16 commits, 14 files, +866/−44). The image
this repo targets, `docker.io/stilldeadcode/vllm-radiance:0.5.8`, is built on vLLM
**0.26.0** — its [source](https://codeberg.org/StillDeadcode/vllm-radiance) pins
`ARG VLLM_VERSION=0.26.0` and builds from the `v0.26.0` tag — which predates it. Rebuilding the image was not an option here — it carries a
set of gfx1201-specific kernels (see the main README) that a stock vLLM build does not
have — so the PR was back-ported onto 0.26.0 as files instead.

**This pins the overlay to 0.26.0.** It will not apply cleanly to a different base. If you
change `IMAGE`, re-derive the overlay rather than carrying it across.

## What is in it

```
vllm/config/vllm.py                              speculative config plumbing
vllm/model_executor/models/registry.py           register the DFlash2 architecture
vllm/model_executor/models/qwen3_dflash.py       draft model
vllm/model_executor/models/qwen3_dflash2.py      DFlash2 head
vllm/v1/worker/gpu/spec_decode/__init__.py       speculator dispatch
vllm/v1/worker/gpu/spec_decode/speculator.py     base speculator
vllm/v1/worker/gpu/spec_decode/dflash2/__init__.py
vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py   selector walk (Triton)
vllm/v1/worker/gpu/sample/gumbel.py              gumbel_noised_argmax
vllm/v1/core/kv_cache_utils.py                   KV hash granularity — see below
```

Every file carries a header naming the upstream commit it came from, any deviation from
merged upstream, and any hunk that had to be hand-adapted for 0.26.0 drift. Read the
header before editing the file.

## The one file that is not optional and not obvious

`v1/core/kv_cache_utils.py`. The DFlash2 draft declares `sliding_window: 2048` on all five
of its layers, which creates a sliding-window KV group alongside the target's 1664-token
attention group. Stock vLLM resolves the divisibility guard in
`resolve_kv_cache_block_sizes()` against the **LCM** (1664), which an 832 group cannot
satisfy; the patch changes the Mamba condition from

```python
g.kv_cache_spec.block_size != cache_config.block_size
```

to

```python
g.kv_cache_spec.mamba_cache_mode != "align"
```

which resolves it against the **gcd** (832) instead. Without it DFlash2 does not boot.

The failure is *not* the `ValueError` at that guard — when the stock guard fires it
returns early, so that error is never reached. It surfaces downstream, in
`kv_cache_coordinator.py` and `block_pool.py`, where hash_block_size 1664 meets an 832
group. Worth knowing, because the message you get does not point at this file.

That this is the right fix rather than a hack: the comment stock vLLM already carries
above that guard reads *"Mamba groups with block_size != cache_config.block_size
(mamba_cache_mode != "align") break divisibility"* — upstream treats the two as
equivalent. The patch makes the code do what its own comment says.

**Consequence:** this file is not DFlash2-gated. Mounted, it also moves MTP's prefix-hash
granularity from 1664 to 832. That is why the launcher scopes the whole overlay to
`SPEC=dflash2`, so that `SPEC=mtp` remains exactly the configuration the MTP numbers in
the README were measured on.

## Deliberate deviations from merged upstream

Three, each recorded in the relevant file's header. They are decisions, not oversights.

1. **`speculator.py` — `draft_logits_spec()` returns `torch.float32`** where merged
   upstream returns `head_dtype`. float32 is what stock 0.26.0 hardcodes at the allocation
   site, so every speculator that does *not* override the hook — MTP included — allocates
   a bit-identical tensor to stock. The hook itself is ported in full; skipping it would
   silently make `DFlash2Speculator`'s `(float32, -inf)` override dead code and fill the
   proposal cache with `0.0`. Only reachable under `draft_sample_method: probabilistic`,
   which is the default here.
2. **Candidate top-k stays inlined in `qwen3_dflash2.py`.** Upstream commit `0a03f5ba1b38`
   moved it into `logits_processor.py` (+81 lines). Functionally identical at TP=1 with
   FlashInfer absent, and it avoids patching a file that sits on every request's path.
3. **`gumbel.py` adds `gumbel_noised_argmax` only.** Upstream also rewires the shared
   `gumbel_block_argmax` to call it; that hunk is not applied, so `gumbel_block_argmax` is
   byte-identical to 0.26.0 and the main sampler path is untouched.

Upstream commit `fed7eea63395` ("load the walk's scores at the reduction width") **is**
ported — it is the ROCm/Triton fp32-vs-fp64 type-mismatch fix, and this card needs it.

## One function that is not upstream at all

`_dense_kv_rows()` in `qwen3_dflash.py`. Merged upstream builds the fused context-KV
precompute with

```python
kv_weights = [a.qkv_proj.weight[a.q_size:] for a in layers_attn]
```

A packed compressed-tensors checkpoint has no `.weight`, so **merged upstream cannot load
an int4 DFlash2 draft at all.** `_dense_kv_rows()` unpacks `weight_packed` + `weight_scale`
instead. It comes from the third-party overlay at
[`BMorgan1296/qwen3.6-vllm-gfx1201-launchers`](https://github.com/BMorgan1296/qwen3.6-vllm-gfx1201-launchers),
which is where the int4-draft path was worked out.

Two guards in it are ours, and both close silent-corruption paths that version does not:

- an **fp8 checkpoint** also stores a 2D `weight`, in `float8_e4m3` with the scale held
  separately — the fast path would return raw unscaled rows and nothing would complain.
  Raises instead.
- the dequant is **scale-only, with no zero-point term**, so an asymmetric checkpoint would
  load, run, and draft from a corrupted KV precompute — visible only as unexplained low
  acceptance. Raises if `weight_zero_point` is non-zero. The launcher also checks this up
  front so you find out at second one rather than minute four.

## Deliberately excluded from that overlay

- `input_processor.py`'s `thinking_token_budget` softening (`ValueError` →
  `warning_once`) — unrelated to DFlash2.
- the `qwen3_dflash2.py` lm_head guard loosening and 2D reshape — both exist to support a
  quantized lm_head; this target's `lm_head.weight` is BF16 `[248320, 5120]`, so both are
  inert.
- `sitecustomize-inc-rocm.py` and `prepare_dspark_head.py` — referenced by that launcher
  but never committed to its repo. Not needed here: the first suppresses an INC-backend
  error, and this repo's `config.json` routing patch means INC is never reached.

## Licence

These files are derived works of vLLM and carry vLLM's `SPDX-License-Identifier:
Apache-2.0` and copyright header unchanged. `_dense_kv_rows()` is from the overlay cited
above, which is likewise Apache-2.0 vLLM-derived.
