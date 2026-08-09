# gfx1201 W4A16 Triton prefill tile tuning

`RDNAHybridW4A16LinearKernel` (`vllm/model_executor/kernels/linear/mixed_precision/
rdna_hybrid_w4a16.py`, part of the `stilldeadcode/vllm-radiance:0.5.8` image used by
the startup scripts in this repo) routes int4 W4A16 dense-Linear GEMMs by batch size:
`M <= 5` (decode) goes to a HIP "skinny" custom op, `M > 5` (prefill) goes to a Triton
kernel. That Triton kernel's gfx1201 tile-size heuristic (`_on_gfx12x()`) is a
hardcoded block-size table, commented in the source as tuned on **Llama-3.1-8B AWQ**
shapes with **group_size=128** — not necessarily the shapes or group_size (32) of
whatever checkpoint you're actually running. This only matters for a *dense* model
(`startup-qwen3.6-27b-vllm.sh`) — the 35B-A3B MoE checkpoint quantizes only its
routed experts and never touches this kernel at all (see that script's header).

## What's here

- `rdna_hybrid_w4a16.orig.py` — the stock kernel file as shipped in
  `vllm-radiance:0.5.8`, pulled unmodified from the container, kept for diffing.
- `rdna_hybrid_w4a16.py` — the same file with a per-`(group_size, K, N, M-bucket)`
  override table added ahead of the stock heuristic (falls through to the original
  heuristic for any shape/bucket not in the table). `diff` the two files to see
  exactly what changed — it's additive: one inserted table + helper, one `if`
  branch given an `override is not None` check ahead of it.
- `bench_gapfill.py` — isolated-kernel microbenchmark, run inside the container
  against the 8 real int4 Linear `(K, N)` shapes in
  `Avesed/Qwen3.6-27B-INT4-W4A16` (group_size=32). Sweeps `BLOCK_M`/`BLOCK_N`/
  `num_warps`/`num_stages` at one M per existing heuristic bucket
  (16/48/96/300/2560), and correctness-checks every shape against a naive
  dequant+matmul reference before timing anything.
- `bench_gapfill_results.tsv` — raw output of the above.
- `build_patch.py` — regenerates `rdna_hybrid_w4a16.py` from
  `bench_gapfill_results.tsv` + `rdna_hybrid_w4a16.orig.py`. Only needed if you
  re-run the sweep yourself (different checkpoint, different card, a Triton/driver
  update) — resolves the one real shape collision in this model (`o_proj` and
  `gdn_out_proj` are both 6144→5120, the identical GEMM) by keeping whichever run
  measured faster, and drops any bucket where the swept config was *slower* than
  the stock heuristic (`kv_proj` at M≤512 was −5.7%; that bucket is intentionally
  absent from the table and falls through to stock).
- `ab-results.tsv` — real end-to-end production A/B (not isolated-kernel timing):
  the real launcher vs. the same launcher with this file bind-mounted over the
  kernel's site-packages path, `bench-live`-style HTTP measurement, nonce-salted
  prompts, 3 reps × depths {4k, 16k, 50k, 100k}.

## Results

Isolated-kernel: stock heuristic measured 5–162% off-optimal depending on
shape/M-bucket (see `bench_gapfill_results.tsv`) — mostly a small-M story (M≤128
sees the largest gains; the M>512 bucket, which dominates deep prefill, gains far
less).

End-to-end (real server, prefill tokens/sec, averaged over 3 reps):

| depth | stock | patched | delta |
|---|---|---|---|
| 4,000 | 1288 | 1413 | +9.7% |
| 16,000 | 1344 | 1419 | +5.6% |
| 50,000 | 1204 | 1251 | +3.9% |
| 100,000 | 1027 | 1062 | +3.5% |

Every one of the 12 patched rows beat its matched stock row — no overlap between
arms at any depth/rep. Decode is unaffected (expected — the kernel routes M<=5 to
the HIP skinny op regardless of this table, and this script's default `MAXSEQS=2`
keeps decode's concurrent-sequence count under 5, so decode never reaches the
Triton path this table changes at all; don't read anything into `ab-results.tsv`'s
decode column, its variance there is ordinary MTP-acceptance noise, not a patch
effect).

## Usage

`startup-qwen3.6-27b-vllm.sh` applies this by default (`TUNED_TILES=1`), bind-mounting
`patches/rdna_hybrid_w4a16.py` over the kernel's path inside the container. To run the
kernel exactly as the image ships it instead:

```
TUNED_TILES=0 ./startup-qwen3.6-27b-vllm.sh
```

It's a community-measured override, not something upstream has reviewed or blessed —
nothing else about the launch changes either way.

## Caveats

- Measured on exactly one checkpoint (`Avesed/Qwen3.6-27B-INT4-W4A16`, group_size
  32) and one card (a single Radeon AI PRO R9700 / gfx1201). The table is keyed by
  `(group_size, K, N, M-bucket)`, so a different checkpoint with different Linear
  shapes will simply never hit any override and silently fall through to the stock
  heuristic — safe, but also means the table buys you nothing on a different model
  without re-running `bench_gapfill.py` against its shapes.
- `num_stages` values came out of a Triton autotuner-style sweep, not first
  principles — no claim about *why* e.g. `num_stages=2` wins for one shape/bucket
  and `None` (Triton's own default) wins for another.
- This overrides only the Triton *prefill* path. Decode (the HIP skinny op) was
  never touched and isn't part of this measurement.
