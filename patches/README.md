# gfx1201 W4A16 Triton tile tuning

`RDNAHybridW4A16LinearKernel` (`vllm/model_executor/kernels/linear/mixed_precision/
rdna_hybrid_w4a16.py`, part of the `stilldeadcode/vllm-radiance:0.5.8` image used by the
startup scripts in this repo) routes int4 W4A16 dense-Linear GEMMs between a HIP "skinny"
custom op and a Triton kernel. That Triton kernel's gfx1201 tile-size heuristic
(`_on_gfx12x()`) is a hardcoded block-size table, commented in the source as tuned on
**Llama-3.1-8B AWQ** shapes — not the shapes or group_size of whatever checkpoint you're
actually running.

This directory replaces that heuristic's decisions for the shapes we measured, and leaves
it untouched for everything else. This only matters for a *dense* model
(`startup-qwen3.6-27b-vllm.sh`) — the 35B-A3B MoE checkpoint quantizes only its routed
experts and never touches this kernel at all (see that script's header).

The table covers **both supported checkpoints**. It is keyed
`(group_size, K, N, M-bucket)`, so the AutoRound rows (`group_size=128`) and the Avesed
rows (`group_size=32`) coexist and neither can be reached by the other.

## Results

Both measured end-to-end against the real launcher — not isolated-kernel timing — with
the mounted file path hard-gated via `podman inspect` and the kernel class confirmed in
the boot log, nonce-salted prompts, and the baseline arm re-run *after* the tuned arm so
drift is visible rather than assumed.

### Intel AutoRound, group_size 128 (the default checkpoint)

Prefill tokens/sec, baseline pooled over the two bracketing loads (n=4) vs tuned (n=2):

| prompt tokens | stock | tuned | delta |
|---|---|---|---|
| 3,936 | 1496.7 | 1639.9 | **+9.56%** |
| 15,623 | 1404.3 | 1510.4 | **+7.55%** |
| 62,269 | 1137.5 | 1204.3 | **+5.87%** |
| 120,520 | 916.5 | 958.6 | **+4.60%** |

Complete separation: every tuned rep beat every matched stock rep, 8/8, no overlap at any
depth. Baseline drift between the two bracketing loads was −0.27 / −0.32 / −0.28 / −0.26%
— an order of magnitude below the effect.

**Decode also moved: +4.26% engine step rate** (95% CI +1.28…+7.24%). Decode t/s cannot
be read directly on an MTP entry because acceptance-length variance swamps it (raw range
41–87 t/s in *both* arms), so this is from `steps/s ~ mean_len + arm + depth`, n=24,
R² 0.944, t=2.86. The control that makes it credible: the two bracketing baseline loads
run the *identical* kernel and differ by only +0.54% (t=0.32), while the tuned arm differs
from each of them separately by ~+4%. Load-level variance does not explain it.

⚠ If you re-run this, **do not omit depth from the model.** Without it R² is 0.217 and the
arm coefficient reads t=0.93 — apparently null. Depth is a large omitted factor (steps/s
falls ~18 → ~12.7 across the range).

Raw data: `ab-results-gs128.tsv`.

### Avesed, group_size 32 (the alternative checkpoint)

Prefill tokens/sec, averaged over 3 reps:

| depth | stock | tuned | delta |
|---|---|---|---|
| 4,000 | 1288 | 1413 | +9.7% |
| 16,000 | 1344 | 1419 | +5.6% |
| 50,000 | 1204 | 1251 | +3.9% |
| 100,000 | 1027 | 1062 | +3.5% |

Every one of the 12 tuned rows beat its matched stock row. Raw data: `ab-results.tsv`.

## Two things this repo got wrong the first time

Both are corrections to earlier versions of this file. They're kept here because the
mistakes are more reusable than the fix.

### 1. Decode DOES reach the Triton kernel

This file used to say "decode always uses the HIP skinny path, so this table cannot affect
decode; ignore `ab-results.tsv`'s decode column". **That is wrong.** It came from reading
`MAX_SKINNY_BATCH_SIZE = 5` alone. The real dispatch is a **conjunction**:

```python
if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS:   # 5, 32768
```

and decode breaks each condition in a different case:

* **M is 10, not 5.** Decode's M is `MAXSEQS × (num_speculative_tokens + 1) = 2 × 5`
  whenever two sequences decode concurrently. 10 > 5 ⇒ every shape takes Triton. M ≤ 5 is
  the *single-sequence* case only.
* **`down_proj` has K=17408**, so `K*M` = 34816 exceeds the 32768-element LDS budget even
  at M=2 — it takes Triton at *every* M, single sequence included.

Measured, not argued: a runtime census over one real load recorded **2728 decode-shaped
Triton calls** (M=2, M=10) against 19417 prefill calls (M=1664…2560).

Consequence for anyone A/B-ing this: **decode is a real axis, not an inert control.** An
experiment that treats it as a falsifier ("if decode moves, something other than the table
changed") is mis-specified.

### 2. Tune the shapes the kernel is CALLED with, not the checkpoint's tensor names

The first sweep read `(K, N)` off the checkpoint's tensor list. **vLLM fuses Linears**, so
most of those shapes never occur at runtime. A runtime census found **5** distinct `(K,N)`,
not 8:

| K × N | what it actually is | fused from |
|---|---|---|
| 5120 × 34816 | `mlp.gate_up_proj` | gate_proj + up_proj (`MergedColumnParallelLinear`) |
| 5120 × 16384 | GDN `in_proj` | `in_proj_qkv` + `in_proj_z` |
| 5120 × 14336 | `self_attn.qkv_proj` | q + k + v (`QKVParallelLinear`) |
| 17408 × 5120 | `mlp.down_proj` | — |
| 6144 × 5120 | `o_proj` / `gdn_out_proj` | — |

**24 of the 34 original group_size=32 rows are dead keys** — they name unfused shapes that
cannot occur, so their lookups always miss. Only `down_proj` and `o_proj` were ever
reachable. That sweep's three biggest headline "wins" (`gate_up_proj` +99.8%, `q_proj`
+144.4%, `gdn_in_proj_qkv` +162.1%) were tuning GEMMs that never run. They are annotated
`[DEAD: ...]` in the kernel rather than deleted, so the next reader doesn't re-derive this.

The GDN fusion (`in_proj_qkv` + `in_proj_z` → N=16384) was **not** predictable from reading
the model source. Only the census caught it.

`make_shapelog_kernel.py` generates an instrumented kernel for this. Run the server once,
then read the census back. **Trap:** vLLM runs ≥2 processes and both import the kernel
module, but only EngineCore ever calls it — so an `atexit` dump from the API-server process
writes `[]` over the real census, and on disk that is indistinguishable from "the kernel
was never called". Refuse to flush an empty census.

## What's here

- `rdna_hybrid_w4a16.orig.py` — the stock kernel as shipped in `vllm-radiance:0.5.8`,
  pulled unmodified from the container, kept for diffing.
- `rdna_hybrid_w4a16.py` — the file the launcher actually mounts. `diff` it against
  `.orig.py` to see everything that changed; it is small and additive.
- `bench_gapfill.py` / `bench_gapfill_results.tsv` — the group_size=32 isolated-kernel
  sweep (correctness-checked against a naive dequant+matmul reference before timing).
- `bench_gapfill_gs128.py` / `bench_gapfill_gs128_results.tsv` — the group_size=128 sweep.
  5 shapes × 5 M values × (120 stage-1 configs + 12 stage-2 `num_stages`), 1447s.
- `census-shapes-gs128.json` — the runtime shape census the gs=128 sweep was driven from.
- `make_shapelog_kernel.py` — generates the instrumented kernel that produces that census.
  Additive and inert unless `RDNA_W4A16_SHAPELOG` names an output path, so it's safe to
  leave bind-mounted.
- `build_patch.py` / `build_patch_gs128.py` — regenerate the table from sweep results.
- `ab-results.tsv` / `ab-results-gs128.tsv` — the raw end-to-end A/B data behind the two
  results tables above.

### Provenance of the shipped kernel

The mounted file is reproducible from the stock kernel plus one hand-applied hunk:

```
./build_patch.py       bench_gapfill_results.tsv       rdna_hybrid_w4a16.orig.py  gs32.py
./build_patch_gs128.py bench_gapfill_gs128_results.tsv gs32.py                    chain.py
diff chain.py rdna_hybrid_w4a16.py     # 17 lines: the symmetric-GPTQ fix below, only
```

`build_patch_gs128.py` is **additive and not idempotent** — it refuses to run against a
kernel that already carries the group_size=128 block rather than duplicating the table.

The one thing no generator produces is the **symmetric-GPTQ zero-point fix**, which is a
genuine bug fix rather than tuning, and which the AutoRound checkpoint cannot run without:

```python
if not c.zero_points:
    w_zp = None
```

GPTQ *always* ships a `qzeros` param — filled with the constant 7 — even when the
checkpoint is symmetric. `_get_weight_params` returns whatever `w_zp_name` points at
without consulting `c.zero_points`, while `process_weights_after_loading` only unpacks that
param when `c.zero_points` is true. So on symmetric GPTQ the raw packed int32
`[K//G, N//8]` reaches the GEMM and dies on `assert zp.shape == (N, num_groups)`. Symmetric
means there is nothing to pass: the `ZP_BIAS=8` branch is already correct, since GPTQ's
stored 7 denotes zero point 7+1=8, which is exactly what `scalar_types.uint4b8` encodes.
Verified against the checkpoint — every `qzeros` nibble in it is 7, no exceptions. The fix
is inert on compressed-tensors symmetric weights, where the param doesn't exist at all.

**This is why `TUNED_TILES=0` is a supported baseline arm on the Avesed checkpoint and a
crash on AutoRound**, and why the startup script refuses that combination.

## Two structural results about the stock heuristic

Worth reusing if you tune this kernel for a different model:

1. **`(BLOCK_M, BLOCK_N, BLOCK_K, num_warps) = (256, 128, 32, 8)` won at large M on all
   five shapes.** The stock heuristic branches three ways on shape class (`tallK` /
   `wideN` / `else`) at M>512, and one config beats all three branches. The *branching*,
   not just its constants, is mistuned for gfx1201.
2. **BLOCK_K is a decode lever, not a prefill one** — the opposite of what we expected.
   At group_size=32, `BLOCK_K = min(BLOCK_K, group_size)` pins it to 32 and the dimension
   doesn't exist; at 128 it opens up. But at M≥1664 the optimum *keeps* BLOCK_K=32 even
   though 128 is now legal; only at M≤20 does it want 64–128. At large M there's enough
   reuse to be compute-bound, so a wider K-tile buys nothing; at small M the GEMM is
   bandwidth-bound on weights and a wider K-tile amortises the per-tile scale load and the
   three-stage `tl.interleave` unpack over more work.

All 25 rows of the gs=128 sweep beat the stock heuristic, minimum +8.1%:

| shape | M=5 | M=10 *(decode)* | M=20 | M=1664 | M=2560 *(prefill)* |
|---|---|---|---|---|---|
| 6144×5120 | +20.3% | +38.7% | +144.4% | +8.8% | +16.9% |
| 5120×34816 | +44.1% | +49.6% | +216.6% | +12.7% | +10.3% |
| 17408×5120 | +34.1% | +38.0% | +179.6% | +13.1% | +27.2% |
| 5120×16384 | +45.4% | +56.5% | +200.3% | +8.6% | +15.7% |
| 5120×14336 | +54.5% | +58.4% | +190.8% | +8.1% | +18.6% |

Bucket stability was checked, not assumed: M=1664 and M=2560 pick the same tile on all 5
shapes (only `num_stages` wobbles), so keying the big bucket off M=2560 isn't overfitting
to one chunk size. M=5 and M=20 were swept as diagnostics and deliberately **not**
installed — they share bucket 32 with M=10, and resolving that collision by lowest absolute
milliseconds would just pick the smallest M and measure nothing about tile quality.

## Usage

`startup-qwen3.6-27b-vllm.sh` applies this by default (`TUNED_TILES=1`), bind-mounting
`patches/rdna_hybrid_w4a16.py` over the kernel's path inside the container. To run the
kernel exactly as the image ships it instead (Avesed checkpoint only, see above):

```
MODEL_DIR=./models/qwen3.6-27b-int4 QUANT=compressed-tensors TUNED_TILES=0 \
  ./startup-qwen3.6-27b-vllm.sh
```

It's a community-measured override, not something upstream has reviewed or blessed —
nothing else about the launch changes either way.

## Caveats

- Measured on one card (a single Radeon AI PRO R9700 / gfx1201) and two checkpoints. The
  table is keyed by `(group_size, K, N, M-bucket)`, so a different checkpoint with
  different Linear shapes will simply never hit any override and falls through to the stock
  heuristic — safe, but it also buys you nothing there without re-running the sweep.
- **Re-run the census, don't reuse this one.** The fusion pattern is a property of the vLLM
  version and the model class, not something you can read off a checkpoint.
- `num_stages` values came out of an autotuner-style sweep, not first principles — no claim
  about *why* `num_stages=2` wins for one shape/bucket and `None` (Triton's own default)
  wins for another.
- The large-M shape-class branching is wrong for gfx1201 generally (result 1 above), but
  this table only bypasses it for the five shapes in it. Any shape not in the table still
  gets the mistuned branch — including every shape in the 35B MoE script, which has never
  been swept.
