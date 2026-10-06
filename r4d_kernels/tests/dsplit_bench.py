#!/usr/bin/env python3
"""Exact-wide prefill (DSPLIT) check: bit-identity vs DS=1 and timing, on the production geometry.
Run once per setting:  R4D_PREFILL_DSPLIT=<1|2|4> R4D_ATTN_FP8=<0|3> dsplit_bench.py OUT.pt"""
import os, sys, torch, r4d
torch.manual_seed(1234)
QH, KVH, HD, BS = 24, 4, 256, 16
SHAPES = [(16, 4096), (64, 8192), (64, 32768), (64, 100000), (256, 32768), (512, 32768),
          (512, 100000), (1024, 32768), (2048, 32768), (2048, 100000)]
dev = "cuda"
maxctx = max(c for _, c in SHAPES)
nblk = (maxctx + BS - 1) // BS + 8
kv = torch.randint(0, 256, (nblk, KVH, BS, 2 * HD), dtype=torch.uint8, device=dev)
kv[(kv & 0x7f) == 0x7f] = 0x38          # no fp8 NaN codes
desc = torch.tensor([0.05], device=dev)
res = {}
for q_len, ctx in SHAPES:
    mb = (ctx + BS - 1) // BS
    bt = torch.randperm(nblk - 1, device=dev)[:mb].to(torch.int32).view(1, mb).contiguous()
    sl = torch.tensor([ctx], dtype=torch.int32, device=dev)
    q = (torch.randn(q_len, QH, HD, device=dev) * 0.5).to(torch.bfloat16)
    out = torch.empty_like(q)
    scr = torch.empty(16, device=dev)
    args = lambda: (q.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(),
                    desc.data_ptr(), desc.data_ptr(), scr.data_ptr(), 1, q_len, QH, KVH, HD, BS, mb,
                    KVH * BS * 2 * HD, BS * 2 * HD, HD ** -0.5, 0, ctx,
                    torch.cuda.current_stream().cuda_stream)
    for _ in range(3):
        r4d.attn_prefill_h256_gqa6_fp8kv(*args())
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    it = 20
    s.record()
    for _ in range(it):
        r4d.attn_prefill_h256_gqa6_fp8kv(*args())
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1000 / it
    res[(q_len, ctx)] = (out.clone().cpu(), us, bool(torch.isfinite(out.float()).all()))
    print(f"q_len {q_len:5d} ctx {ctx:6d}: {us:9.1f} us  finite={res[(q_len, ctx)][2]}", flush=True)
torch.save(res, sys.argv[1])
