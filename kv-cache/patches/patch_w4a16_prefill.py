#!/usr/bin/env python3
"""Prefill-sized tiles for the DFlash2-W4A16 drafter's two prefill GEMMs on gfx1201 (house patch, 2026-10-11).

Two drafter GEMMs see M = the target's prefill step (up to CHUNK, 3568 by default): fc
(combine_hidden_states, K 25600 -> N 5120) and the fused context-KV projection of the five drafter layers
(K 5120 -> N 10240). patch_w4a16_tiles.py tables M <= 64 only, so at prefill M both took the stock
heuristic -- fc at 128x64x64 ran at ~30 TF/s, 20 ms per 3568-token chunk.

The tiles are tcclaviger's R9700 recipe (tcclaviger/vllm, mxfp4_16 Triton configs): BLOCK_K 32 with
2 stages and his GROUP_SIZE_M program-id swizzle, which walks GROUP_M rows of output tiles down M before
moving along N so neighbouring workgroups share weight tiles in L2. The kernel is the stock
_triton_w4a16_skinny_fmt_kernel with one new constexpr, GROUP_M: 0 (every existing caller) keeps the 2-D
grid exactly as before; > 0 launches a 1-D grid and remaps the program id. The K loop is untouched, so
the output is bit-identical to the stock kernel (measured: max |diff| 0 at every M, both shapes).

Measured DRAM-cold on the R9700 through the patched entry point, us per call (stock -> table):
  fc      M 3568: 19667 -> 9461 (x2.08)   2048: 7109 -> 5373   1024: 3038 -> 2663   512: 1530 -> 1453   128: 816 -> 486
  ctx_kv  M 3568:  3837 -> 3627 (x1.06)   2048: 2268 -> 2003   1024: 1093 -> 1031   512:  726 ->  530   256: 369 -> 294
Below 512 the winners are BLOCK_K 64 tiles without the swizzle (same kernel, GROUP_M 0).
Shapes and M not in the table keep the stock heuristic. RADIANCE_W4A16_PREFILL=0 disables the table.
Needs patch_w4a16_tiles.py (LAYOUT, _gfx12x_draft_override) applied first.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"

# --- 1. GROUP_M constexpr on the stock kernel (default 0 = the 2-D grid every caller uses today) ---
apply(F,
      '    LAYOUT: tl.constexpr = 0,  # radiance: 1 = radiance_w4a16_tile blocks [N/16, K/128, 16, 16]\n'
      '):\n',
      '    LAYOUT: tl.constexpr = 0,  # radiance: 1 = radiance_w4a16_tile blocks [N/16, K/128, 16, 16]\n'
      '    GROUP_M: tl.constexpr = 0,  # radiance prefill: > 0 = 1-D grid, program ids grouped along M\n'
      '):\n',
      'GROUP_M: tl.constexpr = 0,  # radiance prefill',
      'rdna_hybrid_w4a16: GROUP_M parameter on the skinny kernel')

# --- 2. the swizzled program-id mapping (tcclaviger's GROUP_SIZE_M) ---
apply(F,
      '    When HAS_ZP=False, only the constant ZP_BIAS is subtracted (symmetric).\n'
      '    """\n'
      '    pid_m = tl.program_id(0)\n'
      '    pid_n = tl.program_id(1)\n',
      '    When HAS_ZP=False, only the constant ZP_BIAS is subtracted (symmetric).\n'
      '    """\n'
      '    if GROUP_M > 0:\n'
      '        # radiance (patch_w4a16_prefill.py): GROUP_M rows of tiles down M, then the next N column,\n'
      '        # so concurrent workgroups share weight tiles in L2. Only the tile order changes.\n'
      '        pid = tl.program_id(0)\n'
      '        num_pid_m = tl.cdiv(M, BLOCK_M)\n'
      '        num_pid_n = tl.cdiv(N, BLOCK_N)\n'
      '        num_in_group = GROUP_M * num_pid_n\n'
      '        first_m = (pid // num_in_group) * GROUP_M\n'
      '        gsize = min(num_pid_m - first_m, GROUP_M)\n'
      '        pid_m = first_m + ((pid % num_in_group) % gsize)\n'
      '        pid_n = (pid % num_in_group) // gsize\n'
      '    else:\n'
      '        pid_m = tl.program_id(0)\n'
      '        pid_n = tl.program_id(1)\n',
      'radiance (patch_w4a16_prefill.py): GROUP_M rows',
      'rdna_hybrid_w4a16: GROUP_M program-id swizzle')

# --- 3. the prefill table ---
apply(F,
      'def triton_w4a16_skinny_fmt_gemm(\n',
      '# --- radiance (patch_w4a16_prefill.py): gfx1201 prefill-M tiles for the DFlash2 W4A16 drafter ---\n'
      '# (group_size, K, N) -> ((min M, (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M) or None), ...),\n'
      '# largest min M first, the first row with M >= min M wins; None (or M below the last row) keeps the\n'
      '# heuristic. Every row is bit-identical to stock (same kernel, same K loop).\n'
      '_GFX12X_PREFILL_OVERRIDES = {\n'
      '    # fc  N=5120  K=25600  (stock at M 3568/2048/1024/512/128: 19667/7109/3038/1530/816 us\n'
      '    # -> 9461/5373/2663/1453/486; M 129-511: no tile beat stock by more than 5%)\n'
      '    (128, 25600, 5120): ((1024, (256, 128, 32, 8, 2, 8)), (512, (256, 64, 32, 4, 2, 8)), (129, None),\n'
      '                         (65, (128, 64, 64, 8, 2, 0))),\n'
      '    # fused context-KV  N=10240  K=5120  (stock at M 3568/2048/1024/512/384/256/128:\n'
      '    # 3837/2268/1093/726/566/369/191 us -> 3627/2003/1031/530/470/294/172)\n'
      '    (128, 5120, 10240): ((512, (256, 128, 32, 8, 2, 8)), (65, (128, 128, 64, 8, 2, 0))),\n'
      '}\n'
      '\n'
      '\n'
      'def _gfx12x_prefill_override(group_size, K, N, M):\n'
      '    import os\n'
      '\n'
      '    if M <= 64 or os.environ.get("RADIANCE_W4A16_PREFILL", "1") != "1":\n'
      '        return None\n'
      '    for m_min, cfg in _GFX12X_PREFILL_OVERRIDES.get((group_size, K, N), ()):\n'
      '        if M >= m_min:\n'
      '            return cfg\n'
      '    return None\n'
      '\n'
      '\n'
      'def triton_w4a16_skinny_fmt_gemm(\n',
      '_GFX12X_PREFILL_OVERRIDES',
      'rdna_hybrid_w4a16: gfx12x prefill tile table')

# --- 4. consult it at the launch ---
apply(F,
      '    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n'
      '\n'
      '    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}\n'
      '    _triton_w4a16_skinny_fmt_kernel[grid](\n',
      '    _radiance_gm = 0\n'
      '    if _on_gfx12x():\n'
      '        _radiance_pf = _gfx12x_prefill_override(group_size, K, N, M)\n'
      '        if _radiance_pf is not None:\n'
      '            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, _radiance_gm = _radiance_pf\n'
      '            BLOCK_K = min(BLOCK_K, group_size)\n'
      '    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n'
      '\n'
      '    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}\n'
      '    if _radiance_gm:\n'
      '        grid = (grid[0] * grid[1],)\n'
      '        extra_kwargs["GROUP_M"] = _radiance_gm\n'
      '    _triton_w4a16_skinny_fmt_kernel[grid](\n',
      '_radiance_pf = _gfx12x_prefill_override(',
      'rdna_hybrid_w4a16: prefill tile table consulted at launch')
