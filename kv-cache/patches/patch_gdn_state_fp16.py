#!/usr/bin/env python3
"""Run GDN prefill against an fp16 state cache natively (RADIANCE_GDN_PREFILL_FP16).

Target: site-packages/radiance_gdn.py (copied in by the prelude just before this runs).

With --mamba-ssm-cache-dtype float16 the prefill path widened every state to fp32 in Python,
ran the fp32 chunk_scan, and narrowed the result on write-back: a gather + .float() copy and a
.to(fp16) copy per prefill call. libr4d rx13's chunk_scan is templated on the state width
(deadcode's radiance-engine kernel) and exposes it as gdn_chunk_scan_k128_v128_c64_bf16_st, so
the kernel reads and writes the fp16 state itself. The arithmetic is the same -- an exact
widening on load, fp32 accumulators, one round-to-nearest on store -- so this saves the two
copies, not a different answer.

Active only when BOTH hold: the state cache is fp16 and the loaded libr4d has the _st binding.
An fp32 cache (production) takes exactly the code path it took before. RADIANCE_GDN_PREFILL_FP16=0
forces the old widen/narrow path, for A/B.

Three hunks, all anchored in the 0.13.0 radiance_gdn.py.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

GDN = Path(sysconfig.get_paths()["purelib"]) / "radiance_gdn.py"
SENTINEL = "_CHUNK_SCAN_ST"

# 1. Pick up the state-width binding when this libr4d has it.
apply(GDN,
      '_CHUNK_SCAN = _bind("gdn_chunk_scan", head_k=HEAD_K, head_v=HEAD_V, chunk=CHUNK)\n',
      '_CHUNK_SCAN = _bind("gdn_chunk_scan", head_k=HEAD_K, head_v=HEAD_V, chunk=CHUNK)\n'
      '# House patch (patch_gdn_state_fp16.py): rx13+ libr4d takes the state width as an argument.\n'
      'import os as _gdn_os\n'
      '_CHUNK_SCAN_ST = (getattr(_r4d, "gdn_chunk_scan_k128_v128_c64_bf16_st", None)\n'
      '                  if _r4d is not None and _CHUNK_SCAN is not None\n'
      '                  and _gdn_os.environ.get("RADIANCE_GDN_PREFILL_FP16", "1") == "1" else None)\n',
      SENTINEL, "gdn-state-fp16 (binding)")

# 2. fused_prefill: accept an fp16 initial state when the binding exists, and say so to the kernel.
apply(GDN,
      '    if initial_state.dtype != torch.float32:\n'
      '        return _bail(f"state dtype {initial_state.dtype}")\n',
      '    state_fp16 = initial_state.dtype == torch.float16 and _CHUNK_SCAN_ST is not None\n'
      '    if initial_state.dtype != torch.float32 and not state_fp16:\n'
      '        return _bail(f"state dtype {initial_state.dtype}")\n',
      "state_fp16 = initial_state.dtype", "gdn-state-fp16 (accept)")
apply(GDN,
      '    _CHUNK_SCAN(\n'
      '        q.data_ptr(), k.data_ptr(), v.data_ptr(), A.data_ptr(), g.data_ptr(), beta.data_ptr(),\n'
      '        initial_state.data_ptr(), o.data_ptr(), final_state.data_ptr(), cu_seqlens.data_ptr(),\n'
      '        num_seqs, H, Hg, HEAD_K, HEAD_V, CHUNK, float(scale),\n'
      '        torch.cuda.current_stream().cuda_stream,\n'
      '    )\n',
      '    _scan_args = (\n'
      '        q.data_ptr(), k.data_ptr(), v.data_ptr(), A.data_ptr(), g.data_ptr(), beta.data_ptr(),\n'
      '        initial_state.data_ptr(), o.data_ptr(), final_state.data_ptr(), cu_seqlens.data_ptr(),\n'
      '        num_seqs, H, Hg, HEAD_K, HEAD_V, CHUNK, float(scale),\n'
      '    )\n'
      '    if state_fp16:\n'
      '        _CHUNK_SCAN_ST(*_scan_args, 1, torch.cuda.current_stream().cuda_stream)\n'
      '    else:\n'
      '        _CHUNK_SCAN(*_scan_args, torch.cuda.current_stream().cuda_stream)\n',
      "_scan_args = (", "gdn-state-fp16 (launch)")

# 3. The prefill call site: stop widening an fp16 cache in Python when the kernel can read it.
apply(GDN,
      '    initial_state = ssm_state[md.prefill_state_indices].float()\n',
      '    initial_state = ssm_state[md.prefill_state_indices]\n'
      '    if not (initial_state.dtype == torch.float16 and _CHUNK_SCAN_ST is not None):\n'
      '        initial_state = initial_state.float()\n',
      "if not (initial_state.dtype == torch.float16 and _CHUNK_SCAN_ST", "gdn-state-fp16 (call site)")
