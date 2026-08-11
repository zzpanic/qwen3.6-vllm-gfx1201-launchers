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
GROUP_SIZE = 32
ZP_BIAS = 8

SHAPES = [
    ("down_proj", 17408, 5120),
    ("gate_up_proj", 5120, 17408),
    ("kv_proj", 5120, 1024),
    ("q_proj", 5120, 12288),
    ("o_proj", 6144, 5120),
    ("gdn_in_proj_qkv", 5120, 10240),
    ("gdn_in_proj_z", 5120, 6144),
    ("gdn_out_proj", 6144, 5120),
]

# midpoints of the existing M-buckets, plus the production ceiling
M_VALUES = [16, 48, 96, 300, 2560]

STAGE1_BM = (32, 64, 128, 256)
STAGE1_BN = (32, 64, 128, 256, 512)
STAGE1_NW = (4, 8)


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


def run_custom(a, b_q, scales, K, N, num_groups, block_m, block_n, block_k, num_warps, num_stages=None):
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


print("shape\tM\tclass\tcur_cfg\tcur_ms\tbest_cfg\tbest_ms\tpct")

results = []
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

    # correctness check once per shape at a small M
    a_check = (torch.randn(8, K, device=device, dtype=dtype) * 0.05)
    out_default = triton_w4a16_skinny_fmt_gemm(a_check, b_q.view(torch.int32), scales, GROUP_SIZE, zp_bias=ZP_BIAS)
    out_ref = reference(a_check, w_uint4, scales, GROUP_SIZE, ZP_BIAS)
    err = (out_default.float() - out_ref.float()).abs().max().item()
    print(f"# {name} K={K} N={N} class={shape_class} correctness_max_err={err:.4f}", flush=True)

    for M in M_VALUES:
        a = (torch.randn(M, K, device=device, dtype=dtype) * 0.05).contiguous()

        cur_bm, cur_bn, cur_bk_raw, cur_nw = current_cfg(M, K, N)
        cur_bk = min(cur_bk_raw, GROUP_SIZE)
        try:
            cur_ms = timed(lambda: run_custom(a, b_q, scales, K, N, num_groups, cur_bm, cur_bn, cur_bk, cur_nw))
        except Exception as e:
            cur_ms = float("inf")

        # stage 1: coarse grid, BLOCK_K fixed at 32 (forced by group_size regardless)
        stage1 = []
        for bm in STAGE1_BM:
            for bn in STAGE1_BN:
                for nw in STAGE1_NW:
                    try:
                        ms = timed(lambda bm=bm, bn=bn, nw=nw: run_custom(a, b_q, scales, K, N, num_groups, bm, bn, GROUP_SIZE, nw), iters=10, warmup=3)
                    except Exception:
                        ms = float("inf")
                    stage1.append((bm, bn, nw, None, ms))
        stage1.sort(key=lambda r: r[4])

        # stage 2: num_stages sweep on top-3 stage1 candidates
        stage2 = []
        for (bm, bn, nw, _, _) in stage1[:3]:
            for ns in (1, 2, 3, 4):
                try:
                    ms = timed(lambda bm=bm, bn=bn, nw=nw, ns=ns: run_custom(a, b_q, scales, K, N, num_groups, bm, bn, GROUP_SIZE, nw, ns), iters=10, warmup=3)
                except Exception:
                    ms = float("inf")
                stage2.append((bm, bn, nw, ns, ms))

        allc = stage1 + stage2
        allc.sort(key=lambda r: r[4])
        best_bm, best_bn, best_nw, best_ns, best_ms = allc[0]

        pct = (cur_ms / best_ms - 1.0) * 100 if best_ms > 0 else float("nan")
        row = f"{name}\t{M}\t{shape_class}\t({cur_bm},{cur_bn},{cur_nw})\t{cur_ms:.4f}\t({best_bm},{best_bn},{best_nw},ns={best_ns})\t{best_ms:.4f}\t{pct:+.1f}%"
        print(row, flush=True)
        results.append((name, K, N, shape_class, M, (cur_bm, cur_bn, cur_bk, cur_nw), cur_ms,
                         (best_bm, best_bn, GROUP_SIZE, best_nw, best_ns), best_ms, pct))

print("\n\n=== DONE ===")
