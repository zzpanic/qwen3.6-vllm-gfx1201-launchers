#!/usr/bin/env python3
"""Isolated-kernel tile sweep for RDNAHybridW4A16LinearKernel at group_size=128.

Re-does bench_gapfill.py (the group_size=32 sweep) for the AutoRound
checkpoint. TWO things differ, and both change the answer:

 1. BLOCK_K IS NOW A FREE VARIABLE. The kernel ends its heuristic with
        BLOCK_K = min(BLOCK_K, group_size)
    so at gs=32 every branch collapsed to BLOCK_K=32 and the old sweep correctly held
    it fixed. At gs=128 the clamp binds only on the 128-branches, so BLOCK_K in
    {32, 64, 128} is a real dimension -- it is the whole reason for this run.

 2. THE STOCK HEURISTIC IS NO LONGER HANDICAPPED. Its comment says it was tuned "using
    Llama-3.1-8B AWQ weight shapes with group_size=128", i.e. at gs=32 we were beating
    a heuristic running outside the regime it was tuned for. Expect much SMALLER wins
    here than the old sweep's +5..162%, and be honest when a shape has none: the patch
    builder drops any bucket where the sweep did not beat the heuristic.

Shapes come from a RUNTIME CENSUS (shapes.json, produced by the shapelog kernel), not
from the checkpoint's tensor list. The old sweep read tensor names and so tuned
q_proj (N=12288), k/v_proj (N=1024) and gate_up_proj (N=17408) -- none of which the
kernel is ever called with, because vLLM fuses them into QKVParallelLinear (N=14336)
and MergedColumnParallelLinear (N=34816). Three of its eight shapes were dead keys.

Usage (inside the vLLM image, GPU visible):
    python3 bench_gapfill_gs128.py [shapes.json]
"""

import json
import os
import sys
import time

import torch

from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    _triton_w4a16_skinny_fmt_kernel,
    pack_int4_exllama_shuffle,
    triton_w4a16_skinny_fmt_gemm,
)
from vllm import triton_utils as _tu

triton = _tu.triton
torch.manual_seed(0)
device = "cuda"
dtype = torch.bfloat16
GROUP_SIZE = 128
ZP_BIAS = 8

# THE M VALUES THE KERNEL IS ACTUALLY CALLED WITH, from the phase-1 census -- not the
# bucket midpoints the gs=32 sweep used. The census found only two live buckets:
#     M = 5 and 10  (bucket 32)   90.6% of calls   <- DECODE
#     M = 2560      (bucket big)   9.4% of calls   <- prefill chunks at BATCHTOK=2560
# and nothing in between, because chunked prefill hands the kernel full 2560-token
# chunks while decode hands it max_num_seqs x (num_spec+1) = 2 x 5 = 10.
#
# *** DECODE REACHING THIS PATH CONTRADICTS WHAT THE LAUNCHER HEADER AND THE 2026-08-09
# NOTES BOTH CLAIM *** ("MAXSEQS=2 keeps decode's M under the kernel's M<=5 HIP skinny
# cutoff, so decode never reaches the Triton path"). At two concurrent sequences M=10,
# which is over the cutoff; and down_proj reaches Triton even at M=5, so the HIP skinny
# kernel evidently declines that shape. Measured, not argued -- see shapes.json.
#
# The big bucket's real M range is 1664..2560 -- 1664 is the attention block size the
# hybrid allocator picked, and chunked prefill emits chunks at both sizes. 2560 is the
# BATCHTOK cap and is what the table is keyed from; 1664 is swept to check the tile
# choice is stable across the bucket rather than tuned to one chunk size.
#
# 20 is swept as a stability probe for a hypothetical MAXSEQS=4 (still bucket 32). It is
# a DIAGNOSTIC, not a table source: build_patch_gs128.py installs bucket 32 from M=10
# only. Picking per-bucket by lowest absolute ms across different M would always choose
# the smallest M, which measures nothing.
M_VALUES = [int(x) for x in os.environ.get("M_VALUES", "5,10,20,1664,2560").split(",")]

STAGE1_BM = (32, 64, 128, 256)
STAGE1_BN = (32, 64, 128, 256, 512)
STAGE1_BK = (32, 64, 128)          # NEW vs the gs=32 sweep -- see module docstring
STAGE1_NW = (4, 8)

# Fallback only. Predicted from the model definition inside the image; the census file
# is what should actually be used, and a mismatch between the two is itself a finding.
FALLBACK_SHAPES = [
    ("mlp.gate_up_proj", 5120, 34816),
    ("mlp.down_proj", 17408, 5120),
    ("self_attn.qkv_proj", 5120, 14336),
    ("o_proj/gdn_out_proj", 6144, 5120),
    ("gdn.in_proj_qkv", 5120, 10240),
    ("gdn.in_proj_z", 5120, 6144),
]


def load_shapes():
    path = sys.argv[1] if len(sys.argv) > 1 else "shapes.json"
    if not os.path.exists(path):
        print(f"# WARNING: {path} missing -- falling back to PREDICTED shapes", flush=True)
        return FALLBACK_SHAPES
    rows = json.load(open(path))
    seen, out = set(), []
    for r in rows:
        if r["group_size"] != GROUP_SIZE:
            print(f"# skipping census row at group_size={r['group_size']}", flush=True)
            continue
        kn = (r["K"], r["N"])
        if kn in seen:
            continue
        seen.add(kn)
        out.append((f"K{r['K']}xN{r['N']}", r["K"], r["N"]))
    print(f"# {len(out)} distinct (K,N) from census {path}", flush=True)
    return out


def reference(a, w_uint4, scales, group_size, zp_bias):
    N_, K_ = w_uint4.shape
    w_fp = w_uint4.to(torch.float32) - zp_bias
    w_fp = w_fp.view(N_, K_ // group_size, group_size)
    s = scales.to(torch.float32).view(N_, K_ // group_size, 1)
    w_fp = (w_fp * s).view(N_, K_)
    return (a.to(torch.float32) @ w_fp.t()).to(a.dtype)


def timed(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def run_custom(a, b_q, scales, K, N, num_groups, block_m, block_n, block_k,
               num_warps, num_stages=None):
    M = a.shape[0]
    K8 = K // 8
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    extra = {} if num_stages is None else {"num_stages": num_stages}
    _triton_w4a16_skinny_fmt_kernel[grid](
        a, b_q.view(torch.int32), scales, scales, c,
        M, N, K, K8, num_groups,
        group_size=GROUP_SIZE, ZP_BIAS=ZP_BIAS, HAS_ZP=False,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, **extra,
    )
    return c


def current_cfg(M, K, N):
    """The stock gfx12x heuristic, verbatim from rdna_hybrid_w4a16.py, override branch
    excluded. Returns the RAW BLOCK_K; the caller applies the min(.., group_size) clamp
    exactly as the kernel does."""
    if M <= 32:
        return (16, 16, 128, 4)
    elif M <= 64:
        if K >= 2 * N:
            return (64, 32, 128, 8)
        elif N > K:
            return (64, 32, 64, 8)
        else:
            return (32, 64, 128, 4)
    elif M <= 128:
        if K >= 2 * N:
            return (64, 16, 64, 1)
        elif N >= 2 * K:
            return (64, 128, 64, 8)
        else:
            return (64, 64, 64, 8)
    elif M <= 512:
        if K >= 2 * N:
            return (128, 64, 64, 8)
        elif N >= 4 * K:
            return (128, 128, 64, 8)
        else:
            return (64, 128, 64, 8)
    else:
        if K >= 2 * N:
            return (128, 64, 64, 8)
        elif N >= 4 * K:
            return (256, 64, 64, 8)
        else:
            return (128, 128, 32, 8)


SHAPES = load_shapes()
print(f"# group_size={GROUP_SIZE} M_VALUES={M_VALUES} "
      f"grid={len(STAGE1_BM)}x{len(STAGE1_BN)}x{len(STAGE1_BK)}x{len(STAGE1_NW)}"
      f"={len(STAGE1_BM)*len(STAGE1_BN)*len(STAGE1_BK)*len(STAGE1_NW)} stage-1 configs",
      flush=True)
print("shape\tM\tclass\tcur_cfg\tcur_ms\tbest_cfg\tbest_ms\tpct")

t_start = time.time()
for name, K, N in SHAPES:
    if K >= 2 * N:
        shape_class = "tallK"
    elif N >= 4 * K:
        shape_class = "wideN"
    else:
        shape_class = "else"

    w_uint4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    b_q = pack_int4_exllama_shuffle(w_uint4).contiguous()
    num_groups = K // GROUP_SIZE
    scales = (torch.rand(N, num_groups, device=device, dtype=dtype) * 0.02 + 0.001).contiguous()

    # correctness check once per shape, against a naive dequant+matmul
    a_check = (torch.randn(8, K, device=device, dtype=dtype) * 0.05)
    out_default = triton_w4a16_skinny_fmt_gemm(
        a_check, b_q.view(torch.int32), scales, GROUP_SIZE, zp_bias=ZP_BIAS)
    out_ref = reference(a_check, w_uint4, scales, GROUP_SIZE, ZP_BIAS)
    err = (out_default.float() - out_ref.float()).abs().max().item()
    print(f"# {name} K={K} N={N} class={shape_class} correctness_max_err={err:.4f}", flush=True)

    for M in M_VALUES:
        a = (torch.randn(M, K, device=device, dtype=dtype) * 0.05).contiguous()

        cur_bm, cur_bn, cur_bk_raw, cur_nw = current_cfg(M, K, N)
        cur_bk = min(cur_bk_raw, GROUP_SIZE)
        try:
            cur_ms = timed(lambda: run_custom(a, b_q, scales, K, N, num_groups,
                                              cur_bm, cur_bn, cur_bk, cur_nw))
        except Exception:
            cur_ms = float("inf")

        stage1 = []
        for bm in STAGE1_BM:
            for bn in STAGE1_BN:
                for bk in STAGE1_BK:
                    for nw in STAGE1_NW:
                        try:
                            ms = timed(
                                lambda bm=bm, bn=bn, bk=bk, nw=nw: run_custom(
                                    a, b_q, scales, K, N, num_groups, bm, bn, bk, nw),
                                iters=10, warmup=3)
                        except Exception:
                            ms = float("inf")
                        stage1.append((bm, bn, bk, nw, None, ms))
        stage1.sort(key=lambda r: r[5])

        # stage 2: num_stages sweep on top-3 stage-1 candidates
        stage2 = []
        for (bm, bn, bk, nw, _, _) in stage1[:3]:
            for ns in (1, 2, 3, 4):
                try:
                    ms = timed(
                        lambda bm=bm, bn=bn, bk=bk, nw=nw, ns=ns: run_custom(
                            a, b_q, scales, K, N, num_groups, bm, bn, bk, nw, ns),
                        iters=10, warmup=3)
                except Exception:
                    ms = float("inf")
                stage2.append((bm, bn, bk, nw, ns, ms))

        allc = stage1 + stage2
        allc.sort(key=lambda r: r[5])
        best_bm, best_bn, best_bk, best_nw, best_ns, best_ms = allc[0]

        pct = (cur_ms / best_ms - 1.0) * 100 if best_ms > 0 else float("nan")
        print(f"{name}\t{M}\t{shape_class}\t({cur_bm},{cur_bn},{cur_bk},{cur_nw})\t{cur_ms:.4f}\t"
              f"({best_bm},{best_bn},{best_bk},{best_nw},ns={best_ns})\t{best_ms:.4f}\t{pct:+.1f}%",
              flush=True)

print(f"\n\n=== DONE in {time.time() - t_start:.0f}s ===")
