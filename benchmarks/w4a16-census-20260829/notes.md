# W4A16 tile gap-fill from the DFlash2 census — 2026-08-29

Window: llama-swap stopped, GPU verified clear at 57 MB. Sweep 1813 s, exit 0.
Production restored, booted twice, coherence verified (391, Lima).

## What was done

1. **Census** (`census-shapes.json`): 182 distinct keys, 30,720 calls,
   1.0034e16 flop-equiv, under DFlash2 + BATCHTOK=4854 + SPECTOK=4.
2. **Isolated sweep** (`run-gapsweep.sh`, `results.tsv`): 8 distinct (K,N) x
   M in {10,45,64,1616,4848} x 132 timed configs (120 stage-1 tiles + num_stages
   on the top 3). Image pinned to PRODUCTION's 0.9.3 — the 2026-08-14 runner's
   0.5.8 pin was stale.
3. **14 rows installed** into `vllm/patches/rdna_hybrid_w4a16.py`.

## THE MEASUREMENT TRAP — read before reusing these numbers

`bench_gapfill_gs128.py:current_cfg()` replicates the **stock gfx12x heuristic**.
It does NOT consult the installed tile table. So its `pct` column is
"win over stock", which equals "win over production" ONLY for keys that have no
installed row.

Worked example: `K17408xN5120 M=4848` reports **+96.2%**. That key is already
covered at MBIG by `(256,128,32,8,1)`, which already makes the structural jump
from stock's `(128,64,64,8)`. Production banks nearly all of that 96% today; the
open question is only ns=3 vs ns=1. Reporting it as an available win would have
been badly wrong.

=> Rows were installed ONLY where no incumbent existed. The five keys already
tuned at buckets 32/MBIG were left ALONE; deciding those needs a head-to-head
against the installed config, which this sweep never timed.

## What was uncovered, and why

**Bucket 64 was entirely empty.** Before this change the gs=128 table had 5 rows
at bucket 32 and 5 at MBIG, and nothing at 64/128/512 — while DFlash2 puts real
traffic at M=45/59/64. Bucket 64 is now 8/8 keys.

**K25600xN5120 had no row at any bucket** — a DFlash2 drafter GEMM the table
predates. 87% of all uncovered work. MBIG is the prize: stock runs
`(128,64,64,8)` at 31.11 ms where `(256,128,32,8,ns=2)` needs 14.43 ms (+115.6%).

Tie-breaks: bucket-64 rows take the winner measured AT M=64 (the bucket's upper
bound and costliest M). MBIG rows take the M=4848 winner (BATCHTOK 4854 =
3x1616 + 2x(SPECTOK-1)), which is 6.6x the cost of the M=1616 case.

Buckets 128 and 512 deliberately left empty: observed M jumps 64 -> 1616.

## Result

Census coverage **0.61% uncovered -> 0.0000%** (recomputed against the new table).

**This closes COVERAGE, not throughput. Do not expect a benchmark to move.**
Uncovered work was 0.61% of GEMM *flops* against a 0.2-0.6% prefill noise floor.
Revised upward slightly in the honest direction: the census `work` column is
flop-weighted, and a shape running 2.16x slow consumes time in proportion, so
K25600xN5120@MBIG is ~1.1% of GEMM *time* and fixing it returns about half of
that — call the whole change ~0.6% of GEMM time, still inside noise.
Precedent: `ornith35b-w4a16-tile-sweep-closed` — isolated wins of +3.6-86.1% did
not survive end-to-end A/B. A later flat benchmark is EXPECTED, not a regression.

## Boot verification

Boot 1 recompiled ("Source code has changed since the last compilation") — the
documented cold-compile trap, expected after editing the kernel. Boot 2: 0
recompiles, pool 269,837 / 1.32x unchanged (tile configs do not touch KV).
