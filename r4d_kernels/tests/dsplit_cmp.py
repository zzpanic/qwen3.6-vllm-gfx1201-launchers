#!/usr/bin/env python3
"""Compare dsplit_bench.py runs: speedup of DS=2/4 over DS=1 and bit-identity of the outputs.

Expects, in DIR, ds{1,2,4}_fp8{3,0}.pt -- one dsplit_bench.py run per setting:
  for fp8 in 3 0; do for ds in 1 2 4; do
    R4D_PREFILL_DSPLIT=$ds R4D_ATTN_FP8=$fp8 python3 dsplit_bench.py DIR/ds${ds}_fp8${fp8}.pt
  done; done
  python3 dsplit_cmp.py DIR
"""
import sys
import torch

d = sys.argv[1] if len(sys.argv) > 1 else "."
bad = 0
for fp8 in ("3", "0"):
    base = torch.load(f"{d}/ds1_fp8{fp8}.pt")
    print(f"=== R4D_ATTN_FP8={fp8} ({'production QK8+PV8' if fp8 == '3' else 'f16 legs'})")
    print(f"{'q_len':>6} {'ctx':>7} {'DS=1 us':>10} {'DS=2 us':>10} {'DS=4 us':>10}  best-speedup  bit-identical(2,4)")
    other = {s: torch.load(f"{d}/ds{s}_fp8{fp8}.pt") for s in (2, 4)}
    for k, (o1, t1, f1) in base.items():
        t2, t4 = other[2][k][1], other[4][k][1]
        same = [torch.equal(o1.view(torch.int16), other[s][k][0].view(torch.int16)) for s in (2, 4)]
        bad += (not all(same)) + (not f1)
        print(f"{k[0]:6d} {k[1]:7d} {t1:10.1f} {t2:10.1f} {t4:10.1f}  {t1 / min(t2, t4):6.2f}x       {same}")
print("PASS: every split is bit-identical to DS=1" if not bad else f"FAIL: {bad} shape(s) differ or are non-finite")
sys.exit(1 if bad else 0)
