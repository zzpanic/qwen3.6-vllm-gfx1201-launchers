# House copy 2026-10-09 of SlyBase sly/patch_w4a16_tiles.py (github.com/brianfu9/vllm-sly-radiance @7d2304a),
# unmodified below this header. vLLM 0.29 original; all 13 anchors match 0.27.1 (radiance 0.9.3).
# Adopted 2026-10-09: drafter W4A16 GEMMs 3.29 -> 1.74 ms/step, BetterBench step gap 35.9 -> 34.3 ms
# (the author's bench notes (triangle-20261009, not shipped)). Imports _patchlib from /patches (ggz14, same apply signature).
#!/usr/bin/env python3
"""Per-shape Triton tile overrides for the DFlash2 W4A16 drafter on gfx1201 (vLLM 0.29.0).

vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py routes M <= 5 to the
HIP skinny kernel and everything else to _triton_w4a16_skinny_fmt_kernel. A DFlash drafter
never sees M <= 5 (block_size 8 -> M = 8 x running sequences), so every drafter GEMM takes the
Triton path, and the gfx12x tile heuristic there was tuned on Llama-3.1-8B AWQ shapes: at
M <= 32 it picks BLOCK 16x16x128 / 4 warps -- 2176 tiny workgroups for the 34816-wide gate_up,
267 GB/s on the R9700 (profile D, vllm7, 2026-09-15). The per-shape override table that the
file already carries (_GFX1X_PREFILL_OVERRIDES) is consulted in the gfx1151 branch only.

This adds a gfx12x override table keyed by (group_size, K, N, M bucket) for the four drafter
shapes of syvai/Qwen3.8-27B-DFlash2-W4A16 (hidden 5120, 32x128 q / 8x128 kv, intermediate
17408, gs=128), its fc layer, and the int4 lm_head of sly/mxfp4/radiance_lmhead_int4.py
(N=248320: the stock 16-column tiles mean 15520 workgroups, 250 GB/s -- slower than the fp8
hipBLASLt head it replaces; with the table 502 GB/s at M=8), plus the three target shapes of a
compressed-tensors INT4 Qwen3.8-27B that the drafter does not share (qkvz 16384x5120, attention
qkv 14336x5120, out/o 5120x6144: stock 1.2-2.9x slower), all measured DRAM-cold with
sly/bench_w4a16_tiles.py on the R9700. Any shape or M bucket not in the table falls through
to the stock heuristic unchanged.
RADIANCE_W4A16_TILES=0 disables the table (A/B control, no rebuild).

Split-K (hunks 3 + 4, 0.4.1): the stock kernel runs the whole K loop in one workgroup, so an N=5120
shape at M <= 16 gets 160-320 workgroups of 2 waves for 64 CUs -- latency-bound at 266-331 GB/s
(INT4 target down_proj K=17408: 139 us against 82 us for the split-K MXFP4 decode kernel on the same
shape). _radiance_w4a16_splitk_kernel adds grid axis 2 over group-aligned K ranges; the fp32 partials
go either to a [SPLIT_K, M, N] buffer summed and cast by _radiance_w4a16_splitk_reduce (deterministic,
2 launches) or via tl.atomic_add into a zeroed fp32 [M, N] (3 launches, order not fixed). The same kernel
carries two cheaper dequant schemes (DEQ 1: scale applied to the fp32 tile result; DEQ 2: bf16 magic-number
nibble conversion with zero point and scale folded into the tile result) and an interleave-free unpack
(UNPACK 1: 8 dots against stride-8 activation columns), selectable per table entry. Entries of
_GFX12X_SPLITK take precedence over the tile table; split_k 1 with the stock dequant runs the stock kernel,
so such a row is bit-identical to the tile table. RADIANCE_W4A16_SPLITK=0 disables the table.

RADIANCE_DFLASH_BF16 (default 0): the drafter's own W4A16 row sets are also dequantized once at the
post-load hook into a per-layer bf16 weight (layer._radiance_w_bf16); apply_weights then runs one
plain GEMM instead of unpacking the packed rows on every step. The packed rows stay (the int4
context-KV path reads them), the drafter's silu / conv int4 transforms are skipped in this mode so
all rows keep their natural order, and the lm_head (a ParallelLMHead scheme without .kernel) is
never touched. Cost: +3.2 GiB VRAM for this drafter (context-expansion.md); check_w4a16_fuse.py
covers the expansion, the apply_weights short-circuit and the guards on the CPU path.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"

# --- 1. the table, placed right after _GFX1X_PREFILL_OVERRIDES ---
apply(F,
      'def triton_w4a16_skinny_fmt_gemm(\n',
      '# --- radiance (sly/patch_w4a16_tiles.py): gfx1201 tiles for the DFlash2 W4A16 drafter ---\n'
      '# (group_size, K, N, M bucket) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).\n'
      '# M bucket = smallest of (8, 16, 32, 40, 64) that is >= M; larger M keeps the heuristic.\n'
      '# Measured DRAM-cold on the R9700 (sly/bench_w4a16_tiles.py, 2026-09-15).\n'
      '_GFX12X_DRAFT_OVERRIDES: dict[tuple[int, int, int, int], tuple[int, int, int, int, int | None]] = {\n'
      '    # qkv_proj  N=6144  K=5120  (stock at M=8/16/32: 77/78/137 us -> 52/52/58)\n'
      '    (128, 5120, 6144, 8): (16, 32, 128, 2, None),\n'
      '    (128, 5120, 6144, 16): (16, 32, 128, 2, None),\n'
      '    (128, 5120, 6144, 32): (32, 32, 128, 4, None),\n'
      '    (128, 5120, 6144, 40): (64, 64, 64, 4, None),\n'
      '    (128, 5120, 6144, 64): (64, 64, 64, 4, None),\n'
      '    # o_proj  N=5120  K=4096  (57/57/96 us -> 47/46/47)\n'
      '    (128, 4096, 5120, 8): (16, 16, 128, 2, 1),\n'
      '    (128, 4096, 5120, 16): (16, 16, 128, 2, 3),\n'
      '    (128, 4096, 5120, 32): (32, 32, 128, 4, None),\n'
      '    # gate_up_proj  N=34816  K=5120  (369/454/944/453/516 us -> 219/224/272/347/361)\n'
      '    (128, 5120, 34816, 8): (16, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 16): (16, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 32): (32, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 40): (64, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 64): (64, 64, 128, 4, None),\n'
      '    # down_proj  N=5120  K=17408  (248/212/412/233/245 us -> 139/141/161/208/218)\n'
      '    (128, 17408, 5120, 8): (16, 32, 128, 2, None),\n'
      '    (128, 17408, 5120, 16): (16, 32, 128, 2, None),\n'
      '    (128, 17408, 5120, 32): (32, 32, 128, 4, None),\n'
      '    (128, 17408, 5120, 40): (64, 32, 128, 4, None),\n'
      '    (128, 17408, 5120, 64): (64, 32, 128, 4, None),\n'
      '    # fc (combine_hidden_states, ReplicatedLinear)  N=5120  K=25600  -- runs BEFORE the\n'
      '    # draft padding on the target-scheduled tokens, so M = 8 x seqs at conc <= 4\n'
      '    # (288/293/575/316/326 us -> 205/205/205/286/294; every config bit-identical)\n'
      '    (128, 25600, 5120, 8): (16, 32, 128, 2, None),\n'
      '    (128, 25600, 5120, 16): (16, 32, 128, 2, None),\n'
      '    (128, 25600, 5120, 32): (32, 32, 128, 4, None),\n'
      '    (128, 25600, 5120, 40): (64, 32, 128, 4, None),\n'
      '    (128, 25600, 5120, 64): (64, 32, 128, 4, None),\n'
      '    # lm_head int4 (sly/mxfp4/radiance_lmhead_int4.py, RADIANCE_LMHEAD_INT4=1)  N=248320\n'
      '    # K=5120 -- verify M = 8 x seqs, draft bootstrap M = 7 x seqs; 656 MB per call\n'
      '    # (2623/2709/5327/2715/2824 us -> 1305/1333/1585/2099/2185; fp8 hipBLASLt: 2459-2623)\n'
      '    (128, 5120, 248320, 8): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 248320, 16): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 248320, 32): (32, 128, 64, 8, None),\n'
      '    (128, 5120, 248320, 40): (64, 128, 64, 8, None),\n'
      '    (128, 5120, 248320, 64): (64, 64, 64, 4, 1),\n'
      '    # INT4 target (RedHatAI/Qwen3.8-27B-INT4, compressed-tensors W4A16 g128): the shapes the drafter\n'
      '    # does not share -- GDN in_proj_qkvz, attention qkv (q with gate), GDN out_proj = attention o_proj\n'
      '    # (N 5120 x K 6144); bench_w4a16_tiles.py --target, 2026-09-17\n'
      '    # INT4 target qkvz  N=16384  K=5120  (stock at M=8/16/32/40/64: 183/183/331/184/183 us -> 107/106/119/152/155)\n'
      '    (128, 5120, 16384, 8): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 16384, 16): (16, 32, 128, 2, 1),\n'
      '    (128, 5120, 16384, 32): (32, 32, 128, 4, None),\n'
      '    (128, 5120, 16384, 40): (64, 128, 128, 8, None),\n'
      '    (128, 5120, 16384, 64): (64, 128, 64, 8, None),\n'
      '    # INT4 target out_o  N=5120  K=6144  (stock at M=8/16/32/40/64: 78/80/142/105/97 us -> 61/61/61/79/81)\n'
      '    (128, 6144, 5120, 8): (16, 16, 128, 2, 3),\n'
      '    (128, 6144, 5120, 16): (16, 16, 128, 2, 3),\n'
      '    (128, 6144, 5120, 32): (32, 32, 128, 4, None),\n'
      '    (128, 6144, 5120, 40): (64, 32, 64, 4, None),\n'
      '    (128, 6144, 5120, 64): (64, 32, 64, 4, None),\n'
      '    # INT4 target attn_qkv  N=14336  K=5120  (stock at M=8/16/32/40/64: 157/163/330/169/185 us -> 93/97/116/145/146)\n'
      '    (128, 5120, 14336, 8): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 14336, 16): (16, 64, 128, 2, 1),\n'
      '    (128, 5120, 14336, 32): (32, 64, 128, 4, None),\n'
      '    (128, 5120, 14336, 40): (64, 64, 128, 4, 1),\n'
      '    (128, 5120, 14336, 64): (64, 64, 128, 4, None),\n'
      '}\n'
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n'
      '\n'
      '\n'
      'def _gfx12x_draft_override(group_size, K, N, M):\n'
      '    import os\n'
      '\n'
      '    if os.environ.get("RADIANCE_W4A16_TILES", "1") != "1":\n'
      '        return None\n'
      '    for b in _GFX12X_DRAFT_BUCKETS:\n'
      '        if M <= b:\n'
      '            return _GFX12X_DRAFT_OVERRIDES.get((group_size, K, N, b))\n'
      '    return None\n'
      '\n'
      '\n'
      'def triton_w4a16_skinny_fmt_gemm(\n',
      '_GFX12X_DRAFT_OVERRIDES',
      'rdna_hybrid_w4a16: gfx12x drafter tile table')

# --- 2. consult it at the top of the gfx12x branch ---
apply(F,
      '    if _on_gfx12x():\n'
      '        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)\n'
      '        # using Llama-3.1-8B AWQ weight shapes with group_size=128.\n'
      '        if M <= 32:\n',
      '    if _on_gfx12x():\n'
      '        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)\n'
      '        # using Llama-3.1-8B AWQ weight shapes with group_size=128.\n'
      '        _radiance_override = _gfx12x_draft_override(group_size, K, N, M)\n'
      '        if _radiance_override is not None:\n'
      '            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _radiance_override\n'
      '        elif M <= 32:\n',
      '_radiance_override = _gfx12x_draft_override(',
      'rdna_hybrid_w4a16: gfx12x branch consults the drafter tile table')

# --- 3. split-K / fast-dequant / tiled-layout kernel, epilogues, reduce, table, fused ops, post-load ---
# Plain source (not a chain of string literals) so it reads like the kernel it becomes.
_SPLITK_SRC = r'''

# --- radiance (sly/patch_w4a16_tiles.py): split-K, fast dequant, tiled weights, fused epilogues ---
@triton.jit
def _radiance_w4a16_epilogue(acc, offs_m, pid_n, M, N, N1, c_ptr, c2_ptr,
                             EPI: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # EPI 0: c[M, N].  EPI 1: columns < N1 -> c[M, N1], the rest -> c2[M, N - N1] (merged GDN qkvz + ba,
    # both outputs contiguous).  EPI 2: interleaved gate/up columns -> silu(gate) * up into c[M, N // 2]
    # (merged MLP gate_up, rows interleaved at load by radiance_w4a16_postload).
    if EPI == 2:
        g, u = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2)))
        val = g * tl.sigmoid(g) * u
        offs_h = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        mask = (offs_m[:, None] < M) & (offs_h[None, :] < N // 2)
        tl.store(c_ptr + offs_m[:, None] * (N // 2) + offs_h[None, :], val.to(c_ptr.dtype.element_ty),
                 mask=mask)
    else:
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m[:, None] < M
        if EPI == 1:
            tl.store(c_ptr + offs_m[:, None] * N1 + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
                     mask=mask_m & (offs_n[None, :] < N1))
            tl.store(c2_ptr + offs_m[:, None] * (N - N1) + (offs_n[None, :] - N1),
                     acc.to(c2_ptr.dtype.element_ty),
                     mask=mask_m & (offs_n[None, :] >= N1) & (offs_n[None, :] < N))
        else:
            tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
                     mask=mask_m & (offs_n[None, :] < N))


@triton.jit
def _radiance_w4a16_splitk_kernel(
    a_ptr, b_ptr, scales_ptr, zp_ptr, p_ptr, c_ptr, c2_ptr, lock_ptr,
    M, N, K, K8, num_groups, tiles_per_split, n_split, N1,
    group_size,
    ZP_BIAS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    MODE: tl.constexpr,
    EPI: tl.constexpr,
    DEQ: tl.constexpr,
    UNPACK: tl.constexpr,
    LAYOUT: tl.constexpr,
    KSTEP: tl.constexpr,
    MAGIC: tl.constexpr,
    MAGIC_F: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # _triton_w4a16_skinny_fmt_kernel over the K tiles [k_lo, k_hi) of split pid_k. BLOCK_K <= group_size,
    # so one scale (and zero point) per row covers a whole K tile, which DEQ 1/2 exploit:
    #   DEQ 0  as stock: w = (q - zp) * scale in the activation dtype, one dot per tile
    #   DEQ 1  w = q - zp (exact), the scale multiplies the fp32 tile result instead of every weight
    #   DEQ 2  w = bits(MAGIC | q) = MAGIC_F + q exactly (no int->float convert, no subtract), zero point
    #          and scale folded into the tile result: (dot - rowsum(a) * (MAGIC_F + zp)) * scale
    #   UNPACK 0  3x tl.interleave + per-column shifts back to natural K order (stock)
    #   UNPACK 1  no interleave: nibble j of every packed word is K index 8i+j, so 8 dots over K/8 against
    #             the matching stride-8 columns of A (needs BLOCK_K // 8 >= 16)
    #   LAYOUT 0  weights [N, K/8] int32 (stock);  LAYOUT 1  16-row x 128-K blocks of 1 KB contiguous
    #             ([N/16, K/128, 16, 16] int32): every row group of a tile is one burst
    #   KSTEP     K tiles per loop trip (unrolled; the tail is masked)
    # MODE 0: fp32 partial into slice pid_k of [SPLIT_K, M, N], summed by _radiance_w4a16_splitk_reduce;
    # MODE 1: fp32 atomic_add into [M, N] (order not fixed); MODE 2: partial as MODE 0, then the program
    # that finishes a tile last (per-tile counter, self-resetting) sums all partials in split order and
    # runs the epilogue -- deterministic, no second launch; MODE 3: split_k 1, epilogue directly.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_lo = pid_k * tiles_per_split
    k_hi = tl.minimum(k_lo + tiles_per_split, tl.cdiv(K, BLOCK_K))
    for k_base in range(k_lo, k_hi, KSTEP):
        for k_u in tl.static_range(KSTEP):
            k_start = k_base + k_u
            k_ok = k_start < k_hi  # tail trip of an unrolled loop: A masked to 0 -> contributes nothing

            offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
            if LAYOUT == 1:
                b_ptrs = b_ptr + (((offs_n[:, None] // 16) * (K8 // 16) + offs_k8[None, :] // 16) * 256
                                  + (offs_n[:, None] % 16) * 16 + offs_k8[None, :] % 16)
            else:
                b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]
            mask_b = mask_n[:, None] & (offs_k8[None, :] < K8) & k_ok
            b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

            group_idx = (k_start * BLOCK_K) // group_size
            scales = tl.load(scales_ptr + offs_n * num_groups + group_idx, mask=mask_n & k_ok, other=1.0)
            if HAS_ZP:
                zp_word = tl.load(zp_ptr + (offs_n // 8) * num_groups + group_idx, mask=mask_n & k_ok, other=0)
                zp_raw = (zp_word >> (4 * (offs_n % 8))) & 0xF
                zp_col = zp_raw[:, None]
            else:
                zp_col = ZP_BIAS

            tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            rowsum = tl.zeros((BLOCK_M,), dtype=tl.float32)
            if UNPACK == 0:
                offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
                a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                            mask=mask_m[:, None] & (offs_k[None, :] < K) & k_ok, other=0.0)
                b = tl.interleave(b_packed, b_packed)
                b = tl.interleave(b, b)
                b = tl.interleave(b, b)
                b = (b >> shifts_full) & 0xF
                if DEQ == 0:
                    w = (b - zp_col).to(scales.dtype) * scales[:, None]
                elif DEQ == 1:
                    w = (b - zp_col).to(scales.dtype)
                else:
                    w = (b | MAGIC).to(tl.int16).to(scales.dtype, bitcast=True)
                    rowsum += tl.sum(a.to(tl.float32), axis=1)
                tile += tl.dot(a, tl.trans(w), out_dtype=tl.float32)
            else:
                for j in tl.static_range(8):
                    offs_kj = k_start * BLOCK_K + tl.arange(0, BLOCK_K // 8) * 8 + j
                    a_j = tl.load(a_ptr + offs_m[:, None] * K + offs_kj[None, :],
                                  mask=mask_m[:, None] & (offs_kj[None, :] < K) & k_ok, other=0.0)
                    b_j = (b_packed >> ((j // 2) * 4 + (j % 2) * 16)) & 0xF
                    if DEQ == 0:
                        w_j = (b_j - zp_col).to(scales.dtype) * scales[:, None]
                    elif DEQ == 1:
                        w_j = (b_j - zp_col).to(scales.dtype)
                    else:
                        w_j = (b_j | MAGIC).to(tl.int16).to(scales.dtype, bitcast=True)
                        rowsum += tl.sum(a_j.to(tl.float32), axis=1)
                    tile += tl.dot(a_j, tl.trans(w_j), out_dtype=tl.float32)

            if DEQ == 0:
                accumulator += tile
            elif DEQ == 1:
                accumulator += tile * scales.to(tl.float32)[None, :]
            else:
                if HAS_ZP:
                    zf = MAGIC_F + zp_raw.to(tl.float32)
                    tile = tile - rowsum[:, None] * zf[None, :]
                else:
                    tile = tile - rowsum[:, None] * (MAGIC_F + ZP_BIAS)
                accumulator += tile * scales.to(tl.float32)[None, :]

    mask_c = mask_m[:, None] & mask_n[None, :]
    if MODE == 3:
        _radiance_w4a16_epilogue(accumulator, offs_m, pid_n, M, N, N1, c_ptr, c2_ptr, EPI, BLOCK_M, BLOCK_N)
    elif MODE == 1:
        tl.atomic_add(p_ptr + offs_m[:, None] * N + offs_n[None, :], accumulator, mask=mask_c)
    else:
        tl.store(p_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], accumulator, mask=mask_c)
        if MODE == 2:
            # every wave's partial store has completed before the release on the counter; the acquire of
            # the program that sees n_split - 1 makes all of them visible, its loads bypass L1 (.cg)
            tl.debug_barrier()
            tile_id = pid_m * tl.num_programs(1) + pid_n
            done = tl.atomic_add(lock_ptr + tile_id, 1, sem="acq_rel", scope="gpu")
            if done == n_split - 1:
                total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for s in range(0, n_split):
                    total += tl.load(p_ptr + s * M * N + offs_m[:, None] * N + offs_n[None, :],
                                     mask=mask_c, other=0.0, cache_modifier=".cg")
                _radiance_w4a16_epilogue(total, offs_m, pid_n, M, N, N1, c_ptr, c2_ptr, EPI, BLOCK_M, BLOCK_N)
                tl.atomic_xchg(lock_ptr + tile_id, 0, sem="relaxed", scope="gpu")


@triton.jit
def _radiance_w4a16_splitk_reduce(p_ptr, c_ptr, c2_ptr, M, N, N1,
                                  SPLIT_K: tl.constexpr, EPI: tl.constexpr, BLOCK: tl.constexpr):
    # the SPLIT_K fp32 partials summed in split order, then the epilogue of the GEMM (MODE 0 / 1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    MN = M * N
    if EPI == 2:
        NH = N // 2
        mask = offs < M * NH
        row = offs // NH
        col = offs % NH
        g = tl.zeros((BLOCK,), dtype=tl.float32)
        u = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            g += tl.load(p_ptr + s * MN + row * N + 2 * col, mask=mask, other=0.0)
            u += tl.load(p_ptr + s * MN + row * N + 2 * col + 1, mask=mask, other=0.0)
        tl.store(c_ptr + offs, (g * tl.sigmoid(g) * u).to(c_ptr.dtype.element_ty), mask=mask)
    else:
        mask = offs < MN
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
        if EPI == 1:
            row = offs // N
            col = offs % N
            tl.store(c_ptr + row * N1 + col, acc.to(c_ptr.dtype.element_ty), mask=mask & (col < N1))
            tl.store(c2_ptr + row * (N - N1) + (col - N1), acc.to(c2_ptr.dtype.element_ty),
                     mask=mask & (col >= N1))
        else:
            tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# (group_size, K, N, M bucket) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split_k, mode
# [, deq, unpack[, kstep]]) -- mode 0 partial buffer + reduce kernel, 1 atomic, 2 serial (no second
# launch); deq/unpack default 0 (stock dequant), kstep 1. Consulted before the tile table; from
# sly/bench_w4a16_tiles.py --splitk (R9700, DRAM-cold, "today" = the tile table above).
_GFX12X_SPLITK: dict[tuple[int, int, int, int], tuple] = {
@@SPLITK_TABLE@@}

# Split-K partials are allocated per call (graph-pool friendly) and kept <= 1 MiB by the table: the
# caching allocator serves 1-10 MiB requests from 20 MiB segments, and vLLM charges every byte the trial
# CUDA-graph capture takes (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS) to the KV pool.
_RADIANCE_SK_MAX_PARTIAL = 1 << 20
_RADIANCE_SK_TABLE = None
_RADIANCE_SK_LOCKS: dict = {}
_RADIANCE_SK_LOCK_SLOTS = 1 << 16


def _gfx12x_splitk_table():
    # RADIANCE_W4A16_SPLITK_TABLE=<json> replaces the built-in table ({"gs,K,N,bucket": [cfg...]}, as
    # written by bench_w4a16_tiles.py --splitk --table-out); read once per process.
    global _RADIANCE_SK_TABLE
    if _RADIANCE_SK_TABLE is None:
        import json
        import os

        path = os.environ.get("RADIANCE_W4A16_SPLITK_TABLE")
        if path:
            with open(path) as f:
                _RADIANCE_SK_TABLE = {tuple(int(x) for x in k.split(",")): tuple(v)
                                      for k, v in json.load(f).items()}
        else:
            _RADIANCE_SK_TABLE = _GFX12X_SPLITK
    return _RADIANCE_SK_TABLE


def _gfx12x_splitk_override(group_size, K, N, M):
    import os

    if os.environ.get("RADIANCE_W4A16_SPLITK", "1") != "1":
        return None
    for b in _GFX12X_DRAFT_BUCKETS:
        if M <= b:
            return _gfx12x_splitk_table().get((group_size, K, N, b))
    return None


def _radiance_sk_locks(device):
    # per-tile arrival counters of MODE 2, zero between GEMMs (the last program resets its tile)
    locks = _RADIANCE_SK_LOCKS.get(device)
    if locks is None:
        locks = torch.zeros(_RADIANCE_SK_LOCK_SLOTS, dtype=torch.int32, device=device)
        _RADIANCE_SK_LOCKS[device] = locks
    return locks


def _radiance_w4a16_n(b_q):
    # rows of a packed weight: [N, K/8] int32 (LAYOUT 0) or [N/16, K/128, 16, 16] int32 (LAYOUT 1)
    return b_q.shape[0] * 16 if b_q.dim() == 4 else b_q.shape[0]


def radiance_w4a16_tile(w_q):
    """[N, K/2] int8 (or [N, K/8] int32) ExLlama-shuffled rows -> LAYOUT 1 [N/16, K/128, 16, 64] int8."""
    w32 = w_q.view(torch.int32) if w_q.dtype == torch.int8 else w_q
    n, k8 = w32.shape
    t = w32.view(n // 16, 16, k8 // 16, 16).permute(0, 2, 1, 3).contiguous()
    return t.view(torch.int8)


def radiance_w4a16_untile(w_q):
    """inverse of radiance_w4a16_tile -> [N, K/2] int8; a 2-D weight is returned as is."""
    if w_q.dim() != 4:
        return w_q
    w32 = w_q.view(torch.int32)
    nb, kt = w32.shape[0], w32.shape[1]
    return w32.permute(0, 2, 1, 3).contiguous().view(nb * 16, kt * 16).view(torch.int8)


def _radiance_tiled_enabled(n, k):
    import os

    return os.environ.get("RADIANCE_W4A16_TILED", "1") == "1" and n % 16 == 0 and k % 128 == 0


def triton_w4a16_splitk_gemm(a, b_q, scales, group_size, cfg, zp_bias=8, zp=None, c=None, epi=0, n1=0,
                             c2=None):
    """triton_w4a16_skinny_fmt_gemm with an explicit (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages,
    split_k, mode[, deq, unpack[, kstep]]) config and an epilogue: epi 0 -> c [M, N]; epi 1 -> (c [M, n1],
    c2 [M, N - n1]); epi 2 -> silu(gate) * up of row-interleaved gate/up weights, c [M, N / 2].
    b_q is [N, K/8] int32 (LAYOUT 0) or the [N/16, K/128, 16, 16] int32 view of radiance_w4a16_tile.
    split_k 1 with the stock dequant, no epilogue and kstep 1 runs the stock kernel unchanged."""
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split_k, mode = cfg[:7]
    deq, unpack, kstep = (tuple(cfg[7:10]) + (0, 0, 1)[len(cfg[7:10]):])
    M, K = a.shape
    layout = 1 if b_q.dim() == 4 else 0
    N = _radiance_w4a16_n(b_q)
    K8 = K // 8
    num_groups = K // group_size
    BLOCK_K = min(BLOCK_K, group_size)
    if unpack:
        assert BLOCK_K // 8 >= 16, "UNPACK 1 needs BLOCK_K >= 128 (tl.dot K >= 16)"
    if epi == 2:
        assert BLOCK_N % 2 == 0 and N % 2 == 0
    has_zp = zp is not None
    if c is None:
        c = torch.empty((M, n1 if epi == 1 else (N // 2 if epi == 2 else N)), dtype=a.dtype, device=a.device)
    if epi == 1 and c2 is None:
        c2 = torch.empty((M, N - n1), dtype=a.dtype, device=a.device)
    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}
    k_tiles = triton.cdiv(K, BLOCK_K)
    tiles_per_split = triton.cdiv(k_tiles, max(1, split_k))
    split_k = triton.cdiv(k_tiles, tiles_per_split)  # no empty splits
    grid_m, grid_n = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    if split_k <= 1 and not deq and not unpack and epi == 0 and kstep == 1:
        _triton_w4a16_skinny_fmt_kernel[(grid_m, grid_n)](
            a, b_q, scales, zp if has_zp else scales, c, M, N, K, K8, num_groups,
            group_size=group_size, ZP_BIAS=zp_bias, HAS_ZP=has_zp,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, LAYOUT=layout, num_warps=num_warps,
            **extra_kwargs)
        return c
    if split_k <= 1:
        run_mode, p, locks = 3, c, c
    elif mode == 2 and grid_m * grid_n <= _RADIANCE_SK_LOCK_SLOTS and (
            a.device in _RADIANCE_SK_LOCKS
            or not (a.is_cuda and torch.cuda.is_current_stream_capturing())):
        # counters are allocated at load (radiance_w4a16_postload) or eagerly here, never inside a
        # graph capture (a throwaway profiling pool must not own them); without them: MODE 0
        run_mode, locks = 2, _radiance_sk_locks(a.device)
        p = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    elif mode == 1:
        run_mode, locks = 1, c
        p = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    else:
        run_mode, locks = 0, c
        p = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    _radiance_w4a16_splitk_kernel[(grid_m, grid_n, split_k)](
        a, b_q, scales, zp if has_zp else scales, p, c, c2 if c2 is not None else c, locks,
        M, N, K, K8, num_groups, tiles_per_split, split_k, n1,
        group_size, ZP_BIAS=zp_bias, HAS_ZP=has_zp, MODE=run_mode, EPI=epi,
        DEQ=deq, UNPACK=unpack, LAYOUT=layout, KSTEP=kstep,
        # MAGIC | q is MAGIC_F + q exactly: fp16 1024.0 (10 mantissa bits), bf16 128.0 (7)
        MAGIC=0x6400 if a.dtype == torch.float16 else 0x4300,
        MAGIC_F=1024.0 if a.dtype == torch.float16 else 128.0,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps, **extra_kwargs)
    if run_mode in (0, 1):
        n_out = M * (N // 2 if epi == 2 else N)
        _radiance_w4a16_splitk_reduce[(triton.cdiv(n_out, 1024),)](
            p, c, c2 if c2 is not None else c, M, N, n1,
            SPLIT_K=1 if run_mode == 1 else split_k, EPI=epi, BLOCK=1024, num_warps=4)
    return (c, c2) if epi == 1 else c


def _radiance_default_cfg(M):
    # shapes the sweep has not covered yet, decode M only: the config that won most shapes in the
    # window-A2/D sweeps (magic-number dequant, direct epilogue; scale-after-dot from M 40)
    if M > 64:
        return None
    bm = 16 if M <= 16 else (32 if M <= 32 else 64)
    return (bm, 64, 128, 4, None, 1, 0, 2 if M <= 32 else 1, 0)


def _radiance_fused_gemm(x_2d, w_q, w_s, group_size, epi, n1=0):
    """GEMM for the radiance call sites (fused silu, merged qkvz+ba, drafter context-KV and conv
    projection): the split-K table config, else the tile table, else _radiance_default_cfg at decode M;
    prefill-sized M runs the plain GEMM with the epilogue in torch."""
    b_q = w_q.view(torch.int32)
    N = _radiance_w4a16_n(b_q)
    M, K = x_2d.shape
    cfg = None
    if _on_gfx12x():
        cfg = _gfx12x_splitk_override(group_size, K, N, M)
        if cfg is None:
            cfg = _gfx12x_draft_override(group_size, K, N, M)
            cfg = tuple(cfg) + (1, 0) if cfg is not None else _radiance_default_cfg(M)
    if cfg is not None:
        return triton_w4a16_splitk_gemm(x_2d, b_q, w_s, group_size, cfg, epi=epi, n1=n1)
    out = triton_w4a16_skinny_fmt_gemm(a=x_2d, b_q=b_q, scales=w_s, group_size=group_size)
    if epi == 0:
        return out
    if epi == 1:
        return out[:, :n1].contiguous(), out[:, n1:].contiguous()
    g, u = out[:, 0::2], out[:, 1::2]
    return (torch.nn.functional.silu(g.float()) * u.float()).to(out.dtype)


def _radiance_w4a16_gemm_impl(x_2d: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                              group_size: int) -> torch.Tensor:
    return _radiance_fused_gemm(x_2d, w_q, w_s, group_size, epi=0)


def _radiance_w4a16_gemm_fake(x_2d: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                              group_size: int) -> torch.Tensor:
    n = w_q.shape[0] * 16 if w_q.dim() == 4 else w_q.shape[0]
    return torch.empty((x_2d.size(0), n), dtype=x_2d.dtype, device=x_2d.device)


def _radiance_w4a16_silu_impl(x_2d: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                              group_size: int) -> torch.Tensor:
    return _radiance_fused_gemm(x_2d, w_q, w_s, group_size, epi=2)


def _radiance_w4a16_silu_fake(x_2d: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                              group_size: int) -> torch.Tensor:
    n = w_q.shape[0] * 16 if w_q.dim() == 4 else w_q.shape[0]
    return torch.empty((x_2d.size(0), n // 2), dtype=x_2d.dtype, device=x_2d.device)


def _radiance_w4a16_split_impl(x_2d: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, n1: int,
                               group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _radiance_fused_gemm(x_2d, w_q, w_s, group_size, epi=1, n1=n1)


def _radiance_w4a16_split_fake(x_2d: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, n1: int,
                               group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = w_q.shape[0] * 16 if w_q.dim() == 4 else w_q.shape[0]
    m = x_2d.size(0)
    return (torch.empty((m, n1), dtype=x_2d.dtype, device=x_2d.device),
            torch.empty((m, n - n1), dtype=x_2d.dtype, device=x_2d.device))


# --- post-load transforms (sly/patch_w4a16_fuse.py calls radiance_w4a16_postload after the model and
# the drafter are loaded, before profiling and graph capture). Every transform builds its new tensors
# first and only then swaps them in, so a failure leaves the layer as loaded.
def _radiance_w4_parts(layer):
    kern = getattr(getattr(layer, "scheme", None), "kernel", None)
    if type(kern).__name__ != "RDNAHybridW4A16LinearKernel":
        return None
    w_q, w_s, w_zp = kern._get_weight_params(layer)
    if w_zp is not None:
        return None  # symmetric checkpoints only (RedHatAI INT4, syvai drafter)
    return kern, w_q, w_s


def _radiance_set_w4(layer, kern, w_q, w_s):
    kern._transform_param(layer, kern.w_q_name, lambda x: w_q)
    kern._transform_param(layer, kern.w_s_name, lambda x: w_s)


def _radiance_quant_rows(w, group_size=128):
    """bf16/fp16 [rows, K] -> (int8 [rows, K/2] ExLlama-shuffled, scales [rows, K/G] in w's dtype); symmetric int4,
    per-group MSE clip search (the int4 lm_head recipe, sly/mxfp4/radiance_lmhead_int4.py)."""
    rows, k = w.shape
    ng = k // group_size
    # scales in the activation dtype of the layer (bf16 in production, fp16 models too), rounded to it
    # before q is chosen, so q is optimal for the scale the kernel multiplies with
    sdt = w.dtype if w.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    blk = w.float().view(rows, ng, group_size)
    amax = blk.abs().amax(dim=-1, keepdim=True)
    best_err = best_q = best_s = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8):
        s = (amax * (r / 7)).clamp_(min=1e-8).to(sdt).float()
        q = torch.round(blk / s).clamp_(-8, 7)
        err = (q * s - blk).square_().sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_q, best_s = err, q, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
            best_s = torch.where(better, s, best_s)
    nib = (best_q.view(rows, k) + 8).to(torch.uint8)
    return pack_int4_exllama_shuffle(nib).view(torch.int8), best_s.view(rows, ng).to(sdt)


class RadianceW4Linear(torch.nn.Module):
    """A bf16 linear re-quantized to int4 g128 at load, run on the W4A16 GEMM (drafter kernel_projection)."""

    def __init__(self, weight, group_size=128):
        super().__init__()
        n, k = weight.shape
        w_q, w_s = _radiance_quant_rows(weight.detach(), group_size)
        if _radiance_tiled_enabled(n, k):
            w_q = radiance_w4a16_tile(w_q)
        self.register_buffer("w_q", w_q, persistent=False)
        self.register_buffer("w_s", w_s, persistent=False)
        self.group_size = group_size
        self.out_features = n

    def forward(self, x):
        x_2d = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.radiance_w4a16_gemm(x_2d, self.w_q, self.w_s, self.group_size)
        return out.reshape(*x.shape[:-1], self.out_features)


def _radiance_patched(mod, marker):
    # the model source must carry the fused call site (sly/patch_w4a16_fuse.py sets the marker)
    import sys

    return bool(getattr(sys.modules.get(type(mod).__module__), marker, False))


def _radiance_postload_mlp(mod):
    if not _radiance_patched(mod, "_RADIANCE_W4_SILU"):
        return False
    lin = mod.gate_up_proj
    parts = _radiance_w4_parts(lin)
    if parts is None or getattr(lin, "_radiance_silu", False):
        return False
    kern, w_q, w_s = parts
    rows = radiance_w4a16_untile(w_q)
    n = rows.shape[0]
    if n % 32:
        return False
    half = n // 2
    perm = torch.stack([torch.arange(half), half + torch.arange(half)], dim=1).flatten().to(rows.device)
    new_q = rows[perm].contiguous()
    new_s = w_s[perm].contiguous()
    if w_q.dim() == 4:
        new_q = radiance_w4a16_tile(new_q)
    _radiance_set_w4(lin, kern, new_q, new_s)
    lin._radiance_silu = True
    return True


def _radiance_postload_gdn(mod):
    if not _radiance_patched(mod, "_RADIANCE_W4_QKVZ_BA"):
        return False
    qkvz, ba = mod.in_proj_qkvz, mod.in_proj_ba
    parts = _radiance_w4_parts(qkvz)
    w_ba = getattr(ba, "weight", None)
    if (parts is None or getattr(qkvz, "_radiance_ba", 0) or w_ba is None
            or w_ba.dtype not in (torch.bfloat16, torch.float16) or getattr(ba, "bias", None) is not None):
        return False
    kern, w_q, w_s = parts
    gs = kern.config.group_size
    if w_ba.shape[1] % gs or w_ba.shape[0] % 16:
        return False
    rows = radiance_w4a16_untile(w_q)
    n1 = rows.shape[0]
    ba_q, ba_s = _radiance_quant_rows(w_ba, gs)
    new_q = torch.cat([rows, ba_q.to(rows.device)], dim=0).contiguous()
    new_s = torch.cat([w_s, ba_s.to(w_s.dtype)], dim=0).contiguous()
    if w_q.dim() == 4:
        new_q = radiance_w4a16_tile(new_q)
    _radiance_set_w4(qkvz, kern, new_q, new_s)
    qkvz._radiance_ba = n1  # rows before the ba rows: the split point of the merged output
    # the patched GDN forward never calls in_proj_ba again: give its bf16 weight back (~1 MB/layer)
    ba.weight.data = torch.empty(0, dtype=w_ba.dtype, device=w_ba.device)
    return True


def _radiance_w4_unpack_bf16(w_q, w_s, group_size, chunk_rows=1024):
    """Full dequant of one packed symmetric weight to the scales' dtype, [N, K]: the same elements
    the GEMM kernels unpack per call -- nibble j of word i is K index 8*i + j, shift (j // 2) * 4 +
    (j % 2) * 16, value (nibble - 8) * scale of its group (skinny or tiled w_q, scales [N, K/G])."""
    rows_i32 = radiance_w4a16_untile(w_q).contiguous().view(torch.int32)
    n, k8 = rows_i32.shape
    k = k8 * 8
    shifts = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)], dtype=torch.int32,
                          device=rows_i32.device)
    out = torch.empty((n, k), dtype=w_s.dtype, device=rows_i32.device)
    for r0 in range(0, n, chunk_rows):  # keep the shift / scale transients small at load time
        r1 = min(r0 + chunk_rows, n)
        q = ((rows_i32[r0:r1, :, None] >> shifts[None, None, :]) & 0xF).to(w_s.dtype).reshape(r1 - r0, k)
        out[r0:r1] = q.sub_(8).mul_(w_s[r0:r1].repeat_interleave(group_size, dim=1))
    return out


def _radiance_postload_bf16(layer):
    """RADIANCE_DFLASH_BF16=1: stash the full dequant of one W4A16 layer as _radiance_w_bf16 so
    apply_weights runs a plain GEMM on it. The packed rows and scales stay in place (the int4
    context-KV path of radiance_dflash_kv_project reads them), so the cost is the extra copy.
    Returns True when the layer was expanded."""
    if (getattr(layer, "_radiance_silu", False) or getattr(layer, "_radiance_ba", 0)
            or getattr(layer, "_radiance_w_bf16", None) is not None):
        return False  # rows interleaved for a fused op, or already expanded
    parts = _radiance_w4_parts(layer)
    if parts is None:
        return False
    kern, w_q, w_s = parts
    layer._radiance_w_bf16 = _radiance_w4_unpack_bf16(w_q, w_s, kern.config.group_size)
    return True


def radiance_w4a16_postload(model, drafter=None):
    import os
    import sys

    counts = {"silu": 0, "gdn_ba": 0, "conv_w4": 0, "bf16": 0}
    silu_on = os.environ.get("RADIANCE_W4A16_SILU", "1") == "1"
    # GDN qkvz + ba merge: off by default -- the 96 extra rows add a 129th tile (a nearly empty third wave on
    # 64 CUs): 93.9 us merged vs 79.2 + 3.6 us separate at M = 8 (window E sweep, 2026-09-29)
    ba_on = os.environ.get("RADIANCE_GDN_BA_W4", "0") == "1"
    conv_on = os.environ.get("RADIANCE_DFLASH_CONV_W4", "1") == "1"
    # RADIANCE_DFLASH_BF16: the drafter runs plain GEMMs on load-expanded bf16 weights, so skip
    # its silu / conv int4 transforms and keep every row set in its natural order
    bf16_on = os.environ.get("RADIANCE_DFLASH_BF16", "0") == "1"
    for root in (model, drafter):
        if root is None:
            continue
        if bf16_on and root is drafter and model is not drafter:
            for name, mod in list(root.named_modules()):
                try:
                    counts["bf16"] += _radiance_postload_bf16(mod)
                except Exception as exc:  # never fail a model load on a speed transform
                    sys.stderr.write(f"[radiance.w4a16] post-load bf16 expansion skipped for {name}: {exc!r}\n")
            continue
        for name, mod in list(root.named_modules()):
            try:
                if silu_on and hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj"):
                    counts["silu"] += _radiance_postload_mlp(mod)
                elif ba_on and hasattr(mod, "in_proj_qkvz") and hasattr(mod, "in_proj_ba"):
                    counts["gdn_ba"] += _radiance_postload_gdn(mod)
                elif (conv_on and type(mod).__name__ == "DFlashGroupedConv"
                      and not isinstance(mod.kernel_projection, RadianceW4Linear)):
                    w = getattr(mod.kernel_projection, "weight", None)
                    if w is not None and w.dtype in (torch.bfloat16, torch.float16) and w.shape[1] % 128 == 0:
                        mod.kernel_projection = RadianceW4Linear(w)
                        counts["conv_w4"] += 1
            except Exception as exc:  # never fail a model load on a speed transform
                sys.stderr.write(f"[radiance.w4a16] post-load transform skipped for {name}: {exc!r}\n")
    if _on_gfx12x():
        try:
            _radiance_sk_locks(next(model.parameters()).device)  # MODE 2 counters, outside any capture
        except StopIteration:
            pass
    msg = (f"[radiance.w4a16] post-load: gate_up+silu fused {counts['silu']}, GDN qkvz+ba merged "
           f"{counts['gdn_ba']}, drafter conv projections int4 {counts['conv_w4']}")
    if counts["bf16"]:
        msg += f", drafter W4A16 rows expanded to bf16 {counts['bf16']}"
    sys.stderr.write(msg + "\n")
    return counts


def radiance_mlp_forward(mlp, x):
    """fused silu MLP body for a gate_up with interleaved rows (flag _radiance_silu); None = not fused."""
    lin = mlp.gate_up_proj
    parts = _radiance_w4_parts(lin)
    if parts is None:
        return None
    kern, w_q, w_s = parts
    x_2d = x.reshape(-1, x.shape[-1])
    out = torch.ops.vllm.radiance_w4a16_silu(x_2d, w_q, w_s, kern.config.group_size)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def radiance_qkvz_ba(qkvz_proj, hidden_states):
    """merged GDN input projection: (mixed_qkvz, ba), both contiguous, one GEMM."""
    kern, w_q, w_s = _radiance_w4_parts(qkvz_proj)
    return torch.ops.vllm.radiance_w4a16_split(hidden_states, w_q, w_s, qkvz_proj._radiance_ba,
                                               kern.config.group_size)


def radiance_dflash_kv_project(model, normed):
    """DFlash context-KV projection on the drafter's own int4 rows (exact: the same codes and scales the
    bf16 buffer would dequantize) instead of the 105 MB bf16 fused weight; None -> use the stock path."""
    import os

    st = getattr(model, "_radiance_kv_w4", None)
    if st is None:
        st = False
        if os.environ.get("RADIANCE_DFLASH_KV_W4", "1") == "1" and getattr(model, "_fused_kv_bias", None) is None:
            qs, ss, gs = [], [], None
            for a in model._kv_source_attn:
                parts = _radiance_w4_parts(a.qkv_proj)
                if parts is None:
                    qs = None
                    break
                kern, w_q, w_s = parts
                rows = radiance_w4a16_untile(w_q)
                qs.append(rows[a.q_size:])
                ss.append(w_s[a.q_size:])
                gs = kern.config.group_size
            if qs:
                q = torch.cat(qs, dim=0).contiguous()
                if _radiance_tiled_enabled(q.shape[0], q.shape[1] * 2):
                    q = radiance_w4a16_tile(q)
                st = (q, torch.cat(ss, dim=0).contiguous(), gs)
        model._radiance_kv_w4 = st
    if not st:
        return None
    q, s, gs = st
    return torch.ops.vllm.radiance_w4a16_gemm(normed.contiguous(), q, s, gs)


direct_register_custom_op(
    op_name="radiance_w4a16_gemm",
    op_func=_radiance_w4a16_gemm_impl,
    mutates_args=[],
    fake_impl=_radiance_w4a16_gemm_fake,
)
direct_register_custom_op(
    op_name="radiance_w4a16_silu",
    op_func=_radiance_w4a16_silu_impl,
    mutates_args=[],
    fake_impl=_radiance_w4a16_silu_fake,
)
direct_register_custom_op(
    op_name="radiance_w4a16_split",
    op_func=_radiance_w4a16_split_impl,
    mutates_args=[],
    fake_impl=_radiance_w4a16_split_fake,
)
'''

# window E sweep, 2026-09-29 (R9700, 300 W, DRAM-cold, TILED weights, serial split-K, partials <= 1 MiB):
# entries >= 5 % faster than min(tile table, _radiance_default_cfg). us per call, 0.4.0 -> entry.
_SPLITK_TABLE = [
    # down  N=5120  K=17408: M8 151->88 / M16 149->101
    ((128, 17408, 5120, 8), (16, 128, 128, 4, None, 6, 2, 2, 0)),
    ((128, 17408, 5120, 16), (16, 32, 128, 2, None, 2, 2, 2, 0)),
    # out_o  N=5120  K=6144: M8 70->40 / M16 69->44
    ((128, 6144, 5120, 8), (16, 64, 128, 4, None, 6, 2, 2, 0)),
    ((128, 6144, 5120, 16), (16, 32, 128, 2, None, 2, 2, 2, 0)),
    # fc  N=5120  K=25600: M8 207->126 / M16 206->127 / M32 209->200 / M40 293->283 / M64 303->289
    ((128, 25600, 5120, 8), (16, 64, 128, 4, None, 6, 2, 2, 0)),
    ((128, 25600, 5120, 16), (16, 64, 128, 4, None, 2, 2, 2, 0, 2)),
    ((128, 25600, 5120, 32), (32, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 25600, 5120, 40), (64, 32, 128, 4, None, 1, 0, 1, 0)),
    ((128, 25600, 5120, 64), (64, 32, 128, 4, None, 1, 0, 1, 0)),
    # o  N=5120  K=4096: M8 48->32 / M32 45->40 / M40 63->51 / M64 64->51
    ((128, 4096, 5120, 8), (16, 64, 128, 4, None, 6, 2, 2, 0)),
    ((128, 4096, 5120, 32), (32, 32, 128, 4, None, 1, 0, 1, 0, 2)),
    ((128, 4096, 5120, 40), (64, 32, 128, 4, None, 1, 0, 1, 0)),
    ((128, 4096, 5120, 64), (64, 32, 128, 4, None, 1, 0, 1, 0)),
    # qkvz  N=16384  K=5120: M32 131->95 / M40 176->157
    ((128, 5120, 16384, 32), (32, 128, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 16384, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # qkvz_ba  N=16480  K=5120: M32 361->122 / M40 199->177 / M64 202->179
    ((128, 5120, 16480, 32), (32, 128, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 16480, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 16480, 64), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # attn_qkv  N=14336  K=5120: M32 121->94 / M40 159->140 / M64 168->144
    ((128, 5120, 14336, 32), (32, 128, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 14336, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 14336, 64), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # qkv  N=6144  K=5120: M8 47->37 / M16 52->37 / M32 58->51 / M40 75->68 / M64 76->69
    ((128, 5120, 6144, 8), (16, 128, 128, 8, None, 2, 2, 2, 0)),
    ((128, 5120, 6144, 16), (16, 128, 128, 8, None, 2, 2, 2, 0)),
    ((128, 5120, 6144, 32), (32, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 6144, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 6144, 64), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # gate_up  N=34816  K=5120: M8 206->176 / M32 259->211 / M40 362->341 / M64 376->343
    ((128, 5120, 34816, 8), (16, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 34816, 32), (32, 128, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 34816, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 34816, 64), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # ctx_kv  N=10240  K=5120: M8 126->54 / M16 123->56 / M32 228->75 / M64 122->114
    ((128, 5120, 10240, 8), (16, 64, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 10240, 16), (16, 64, 128, 4, None, 1, 0, 2, 0, 2)),
    ((128, 5120, 10240, 32), (32, 128, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 10240, 64), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # conv  N=1280  K=5120: M8 27->23 / M16 30->23 / M32 40->24 / M40 40->28 / M64 40->29
    ((128, 5120, 1280, 8), (16, 32, 128, 4, None, 8, 2, 2, 0)),
    ((128, 5120, 1280, 16), (16, 32, 128, 4, None, 8, 2, 1, 0)),
    ((128, 5120, 1280, 32), (32, 64, 128, 8, None, 4, 2, 0, 0)),
    ((128, 5120, 1280, 40), (64, 32, 128, 4, None, 4, 2, 0, 0)),
    ((128, 5120, 1280, 64), (32, 64, 128, 4, None, 2, 2, 1, 0)),
]

apply(F,
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n',
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n' + _SPLITK_SRC.replace(
          '@@SPLITK_TABLE@@', ''.join(f'    {k}: {v},\n' for k, v in _SPLITK_TABLE)),
      '_radiance_w4a16_splitk_kernel',
      'rdna_hybrid_w4a16: split-K / fast-dequant / tiled / fused-epilogue kernels + table')

# --- 4. the dispatch takes a split-K entry before the stock launch ---
apply(F,
      '    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n',
      '    if _on_gfx12x():\n'
      '        _radiance_sk = _gfx12x_splitk_override(group_size, K, N, M)\n'
      '        if _radiance_sk is not None:\n'
      '            return triton_w4a16_splitk_gemm(a, b_q, scales, group_size, _radiance_sk,\n'
      '                                            zp_bias=zp_bias, zp=zp, c=c)\n'
      '\n'
      '    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n',
      '_radiance_sk = _gfx12x_splitk_override(',
      'rdna_hybrid_w4a16: dispatch consults the split-K table')

# --- 5. the stock kernel reads LAYOUT 1 (tiled) weights too: prefill and tile-table calls ---
apply(F,
      '    BLOCK_K: tl.constexpr,\n'
      '):\n'
      '    """\n'
      '    Fused W4A16 GEMM reading weights from skinny format [N, K//8].\n',
      '    BLOCK_K: tl.constexpr,\n'
      '    LAYOUT: tl.constexpr = 0,  # radiance: 1 = radiance_w4a16_tile blocks [N/16, K/128, 16, 16]\n'
      '):\n'
      '    """\n'
      '    Fused W4A16 GEMM reading weights from skinny format [N, K//8].\n',
      'LAYOUT: tl.constexpr = 0,  # radiance',
      'rdna_hybrid_w4a16: stock kernel LAYOUT parameter')
apply(F,
      '        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]\n'
      '        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)\n',
      '        if LAYOUT == 1:\n'
      '            b_ptrs = b_ptr + (((offs_n[:, None] // 16) * (K8 // 16) + offs_k8[None, :] // 16) * 256\n'
      '                              + (offs_n[:, None] % 16) * 16 + offs_k8[None, :] % 16)\n'
      '        else:\n'
      '            b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]\n'
      '        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)\n',
      '                              + (offs_n[:, None] % 16) * 16 + offs_k8[None, :] % 16)\n'
      '        else:\n'
      '            b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]\n'
      '        mask_b = (offs_n[:, None] < N)',
      'rdna_hybrid_w4a16: stock kernel tiled addressing')

# --- 6. stock launcher: tiled b_q, pass LAYOUT ---
apply(F,
      '    M, K = a.shape\n'
      '    N = b_q.shape[0]\n'
      '    K8 = K // 8\n'
      '    num_groups = K // group_size\n'
      '\n'
      '    assert b_q.shape == (N, K8), f"b_q shape mismatch: {b_q.shape} vs ({N}, {K8})"\n',
      '    M, K = a.shape\n'
      '    layout = 1 if b_q.dim() == 4 else 0  # radiance: 1 = radiance_w4a16_tile blocks\n'
      '    N = b_q.shape[0] * 16 if layout else b_q.shape[0]\n'
      '    K8 = K // 8\n'
      '    num_groups = K // group_size\n'
      '\n'
      '    assert b_q.numel() == N * K8, f"b_q shape mismatch: {b_q.shape} vs ({N}, {K8})"\n',
      'layout = 1 if b_q.dim() == 4 else 0  # radiance',
      'rdna_hybrid_w4a16: stock launcher accepts tiled weights')
apply(F,
      '        BLOCK_K=BLOCK_K,\n'
      '        num_warps=num_warps,\n'
      '        **extra_kwargs,\n'
      '    )\n'
      '    return c\n',
      '        BLOCK_K=BLOCK_K,\n'
      '        LAYOUT=layout,\n'
      '        num_warps=num_warps,\n'
      '        **extra_kwargs,\n'
      '    )\n'
      '    return c\n',
      '        LAYOUT=layout,\n        num_warps=num_warps,\n        **extra_kwargs,',
      'rdna_hybrid_w4a16: stock launch passes LAYOUT')

# --- 7. apply op: tiled weights have N = 16 * dim 0 and never take the HIP skinny path ---
apply(F,
      '    M = x_2d.shape[0]\n'
      '    K = x_2d.shape[1]\n'
      '    N = w_q.shape[0]\n'
      '\n'
      '    if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS:\n',
      '    M = x_2d.shape[0]\n'
      '    K = x_2d.shape[1]\n'
      '    N = w_q.shape[0] * 16 if w_q.dim() == 4 else w_q.shape[0]  # radiance: tiled\n'
      '\n'
      '    if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS and w_q.dim() == 2:\n',
      'K * M <= LDS_CAPACITY_ELEMENTS and w_q.dim() == 2:',
      'rdna_hybrid_w4a16: apply op handles tiled weights')
apply(F,
      '    M = x_2d.size(0)\n'
      '    N = w_q.size(0)\n',
      '    M = x_2d.size(0)\n'
      '    N = w_q.size(0) * 16 if w_q.dim() == 4 else w_q.size(0)  # radiance: tiled\n',
      'N = w_q.size(0) * 16 if w_q.dim() == 4 else w_q.size(0)  # radiance',
      'rdna_hybrid_w4a16: fake impl handles tiled weights')

# --- 8. load: store W4A16 weights tiled (RADIANCE_W4A16_TILED, default on) ---
apply(F,
      '        w_q_skinny = shuffled.contiguous().view(torch.int8)\n',
      '        w_q_skinny = shuffled.contiguous().view(torch.int8)\n'
      '        if _on_gfx12x() and _radiance_tiled_enabled(shuffled.shape[0], shuffled.shape[1] * 8):\n'
      '            w_q_skinny = radiance_w4a16_tile(w_q_skinny)  # radiance: LAYOUT 1\n',
      'w_q_skinny = radiance_w4a16_tile(w_q_skinny)  # radiance',
      'rdna_hybrid_w4a16: tiled weight layout at load')

# --- 9. plain calls into a layer that the post-load transforms reshaped: correct output either way ---
apply(F,
      '        x_2d = x.reshape(-1, x.shape[-1])\n'
      '        N = w_q.shape[0]\n'
      '        out_shape = x.shape[:-1] + (N,)\n',
      '        _radiance_w = getattr(layer, "_radiance_w_bf16", None)\n'
      '        if _radiance_w is not None:\n'
      '            # radiance (RADIANCE_DFLASH_BF16): the packed rows were dequantized to this bf16\n'
      '            # weight once at load -- one plain GEMM, no per-call nibble unpack / scale fold\n'
      '            return torch.nn.functional.linear(x, _radiance_w, bias)\n'
      '        x_2d = x.reshape(-1, x.shape[-1])\n'
      '        N = w_q.shape[0] * 16 if w_q.dim() == 4 else w_q.shape[0]  # radiance: tiled\n'
      '        out_shape = x.shape[:-1] + (N,)\n',
      'no per-call nibble unpack / scale fold',
      'rdna_hybrid_w4a16: apply_weights handles tiled weights + a load-time bf16 expansion')
apply(F,
      '            cu_count,\n'
      '            c.group_size,\n'
      '        )\n'
      '        return output.reshape(out_shape)\n',
      '            cu_count,\n'
      '            c.group_size,\n'
      '        )\n'
      '        if getattr(layer, "_radiance_silu", False):\n'
      '            # radiance: rows interleaved for the fused MLP; a plain caller gets [gate | up] back\n'
      '            output = output.view(-1, N // 2, 2).transpose(1, 2).reshape(-1, N)\n'
      '        elif getattr(layer, "_radiance_ba", 0):\n'
      '            # radiance: GDN qkvz + ba merged; a plain caller gets the qkvz columns only\n'
      '            output = output[:, : layer._radiance_ba].contiguous()\n'
      '            out_shape = x.shape[:-1] + (layer._radiance_ba,)\n'
      '        return output.reshape(out_shape)\n',
      'rows interleaved for the fused MLP; a plain caller',
      'rdna_hybrid_w4a16: plain calls into fused layers stay correct')
