# r4d_kernels

An opt-in libr4d build for the radiance launchers: **`R4D_RX13=1`**. It is libr4d v0.5.0 plus
`r4d_kernels.patch`, built once per image and GPU arch by the launcher's existing libr4d build step
(cached under `~/.cache/radiance-libr4d/v0.5.0-p<patch sha>-<image key>`). It is not the default yet:
the fix below was validated on one production launcher, and the shipped launchers' own defaults
have not been boot-tested with it.

```bash
R4D_RX13=1 ./startup-qwen3.8-27b-kvcache.sh      # or ./startup-qwen3.8-27b-mxfp4.sh
```

## Why: libr4d issue #4, wrong GDN prefill on large decay spans

libr4d's gated-delta-net chunk scan splits each decay factor e^(g_i - g_j) at the chunk midpoint and
clamps each half at e^80, so that the halves stay inside fp32. Any chunk whose gate span exceeds 160
then has its diagonal terms and its contribution to the carried state silently attenuated: the
output stays **finite but wrong** ([libr4d #4](https://codeberg.org/StillDeadcode/libr4d/issues/4)).
On Qwen3.8-27B MXFP4, 51 of 2,304 heads cross that span and 20 do so on every chunk. Only prefill
is affected; decode runs a different kernel. Every image that ships libr4d v0.5.0 or older has it.

deadcode's radiance engine (codeberg.org/StillDeadcode/radiance, 1.0.x) rewrote the scan so that
every decay factor it forms is <= 1, with no clamp. `r4d_kernels.patch` carries that kernel.

Measured against an independent fp64 sequential recurrence
([check_gdn_extreme_decay.py](https://github.com/magiccodingman/vllm-radiance/pull/8)), one R9700:

| max chunk gate span | libr4d b9e42ab + rx9 (stock pin) | r4d_kernels |
|---|---|---|
| 1.3 - 157.5 (ordinary) | 0.27 - 0.36% output error | 0.25 - 0.36% |
| 201.6 | **46%** output, 100% state | 0.25% |
| 504 | **83%** output, 100% state | 0.23% |
| 2016 | **94%** output, 100% state | 0.23% |
| all 12 cases + CUDA-graph replay within 1% | no (6 fail) | **yes**, fp32 and fp16 state |

Served on that box since 2026-10-05: correct answers, warm restart 144 s (rx9: 151-153 s), and a
cached KV-offload resume token-identical to cold references. mtstanfield/vllm-mxfp4 (r9700-tp1)
independently measured the same kind of exact-decay fix as prefill-speed neutral (within noise at
8k / 48k / 112k) and the clamp as moving code perplexity +0.5-0.6% off the reference.

## What else is in the build

- Everything from the radiance extras rx10 (ggz14): narrow-state GDN decode, the fused GDN update,
  the lazy-snapshot kernels, fp8 prefill-attention legs. `R4D_RX13=1` therefore stands in for
  `R4D_RX9=1`.
- v0.5.0's own additions over the stock pin (small-M GEMMs, `quant_act_i8`, `dflash_conv`).
- `gdn_chunk_scan_k128_v128_c64_bf16_st`: the scan with the state width as an argument. With an fp16
  ssm cache, `kv-cache/patches/patch_gdn_state_fp16.py` lets prefill read and write the state
  natively instead of widening it in Python (same arithmetic, two fewer copies). Dormant on fp32.
- **Lazy GDN (UNVALIDATED -- keep `RADIANCE_GDN_LAZY=0`).** `gdn_lazy_materialize` mode 2 zeroes a
  prefilling request's stash headers in every row-split region (a recycled stash block could
  otherwise replay another request's candidates -- the multi-turn corruption ggz14 reported in
  0cadf57), plus stale-stash counters (`gdn_lazy_stale_counts_<tag>`), wired by
  `kv-cache/patches/patch_gdn_lazy_invalidate.py`. The multi-turn gate
  (`kv-cache/tools/mt_lazy_gate.py`) did NOT reproduce the corruption even with the fix off, so the
  fix is unproven. None of this runs while lazy GDN is off.

## Sources and licence status

| part | author / source | licence |
|---|---|---|
| libr4d v0.5.0 (the base the patch applies to) | StillDeadcode, codeberg.org/StillDeadcode/libr4d | none published -- uncertain |
| radiance extras rx10 | ggz14, codeberg.org/ggz14/radiance-vllm-mxfp4 | none published -- uncertain |
| rx12 rebase onto v0.5.0, bf16-state lazy unit | J-Nova, github.com/J-Nova/mtp-offload-r9700 | none published -- uncertain |
| GDN chunk scan kernel | deadcode, codeberg.org/StillDeadcode/radiance | Apache-2.0 (NOTICE: "radiance Copyright 2026 Deadcode and the radiance contributors"); engine plugin-ABI wrapper removed |
| `_st` binding, lazy mode 2, stale-stash counters, launcher knobs, house patches, tools | this repository | Apache-2.0 |

The first three are redistributed here without a licence grant from their authors; the patch header
says so too. If you are one of them and want this changed or removed, open an issue.

## Tools

- `kv-cache/tools/mt_lazy_gate.py` -- multi-turn, prefix-reuse gate (4 conversations x 10 turns over a
  synthetic fact registry, unique `cache_salt` per run, `--compare`). Defaults point at
  `127.0.0.1:1234`; pass `--base` / `--model` for these launchers.
- `kv-cache/tools/spec_ab.py` -- greedy output with speculative decoding on vs off (the launchers'
  new `SPEC_METHOD=none`), token by token with logprob margins. Measured here: 99.13% per-token
  top-1 agreement, every divergence at a near-tie -- serial decode and speculative verify take
  slightly different arithmetic routes.
