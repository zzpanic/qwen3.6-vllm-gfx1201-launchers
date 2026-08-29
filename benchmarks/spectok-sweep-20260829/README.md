# SPECTOK sweep — 2026-08-29

Speculative-decode depth (`SPECTOK`, i.e. `num_speculative_tokens`) swept over K ∈ {4,5,6,7}
on Qwen3.8-27B int4 W4A16 + dflash2 draft head, single Radeon AI PRO R9700 (gfx1201), TP=1,
vLLM 0.27.1 on the radiance 0.9.3 fork.

**Conclusion: pick K on your workload, not from a benchmark mean.** Math throughput rises
13.9% from K=4 to K=7 while chat falls 8.9%; the aggregate is flat only because they cancel.

See `../../TUNING.md` for the written-up version. `summary.tsv` is the machine-readable table.

## Method

- `MAXLEN=131072`, `MAXSEQS=2`, fp8 KV, `KV_GROUP_SIZE=auto`, `ATTN=R4D`, `ASYNCSCHED=0`.
- Bench is [BetterBench](https://github.com/GGZ14/BetterBench), `--greedy`, 8 runs + 2 warmup
  per category, single stream (`--no-concurrency --no-prefill`).
- **Acceptance is not taken from the benchmark.** It comes from vLLM's own
  `vllm:spec_decode_*` counters, diffed pre/post each arm, so it is exact rather than sampled.
- **`BATCHTOK` is derived per arm, not fixed.** The rule is `n*block + 2*(K-1)`, and the
  attention block size itself moves with K. Each arm boots once to read the block size vLLM
  actually chose, then boots again with the aligned value; the second boot is the one benched.
- **Arm Z re-runs K=4 last** as a same-config control. It came back within 0.25% on decode
  with bit-identical spec counters, so the arms are not drift-contaminated and differences
  above ~1% are real.

## Files

| file | what it is |
| --- | --- |
| `summary.tsv` | the whole result, one row per K |
| `runner.sh` | the sweep itself (5 arms, two boots each, bench, counters) |
| `runner-poolfix.sh` | corrective warm re-boot for the two arms whose pool was read cold |
| `sweep.log` | per-arm boot geometry and BATCHTOK derivation |
| `poolfix.log` | the corrected warm pool readings for K=6 and K=7 |
| `maxctx.py` | KV cost model, calibrated against the measured K=4 pool to 0.05% |
| `*.bb.json` | raw BetterBench output per arm |
| `batchtok.txt` | block size and derived BATCHTOK per arm |

## Two traps, both of which bit during this run

**The attention block size moves with K** — 1616 at K=4 and K=5, 1632 at K=6, 1648 at K=7.
A `BATCHTOK` derived from an assumed 1616 is misaligned at K≥6, and missing by a few tokens
drops a whole block. Read it out of the boot log.

**Read the KV pool from a warm boot only.** Changing K changes the graph; the boot that
compiles it reports ~2.1 GiB less available KV than the same config warm. At K=6 that read
167,480 tokens instead of 218,453 — a 24% understatement, and it would have looked exactly
like a plausible cost of deeper speculation. Booting twice is not sufficient by itself: if
the second boot uses a different `BATCHTOK` than the first it recompiles and is cold again,
which is how the first pass at this table produced two bad rows. `runner-poolfix.sh` exists
because that is what happened here.

## Caveat on the last column of the context table

`max_ctx` in `summary.tsv` is **modelled**, not measured — derived from the measured block
size and pool geometry, calibrated against K=4 to 0.05% and K=5 to 1.2%. Treat it as ~1%
optimistic. Every other number in the file is measured.
