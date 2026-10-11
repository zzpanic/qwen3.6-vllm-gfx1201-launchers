#!/usr/bin/env python3
"""GDN out_proj takes the fragment-tiled activation (RADIANCE_GDN_NORM_QUANT_TILED, default 1).

Target: site-packages/radiance_arnq.py (copied in by the prelude just before this runs).
silu_mul_quant and add_rms_quant already emit their fp8 output in WMMA fragment order at M >= RADIANCE_MXFP4_A_TILED_MIN_M,
so the consuming GEMM takes the A-direct prefill kernel (~245 TF/s) instead of the folded one (~175 TF/s at K 6144). The GDN
gated-norm + quant that feeds out_proj did not, so those 48 projections took the slow kernel. This makes that op emit the
tiled layout too: the same fp8 bytes in another order (r4d_kernels/radiance_mxfp4_fp8.patch adds the kernel template and
a trailing `tiled` argument to launch_gdn_norm_quant), registered for the consumer through radiance_mxfp4.a_tiled_register.

Active only when the loaded radiance_mxfp4_fp8 extension has the `tiled` argument (R4D_RX13=1 builds the patched kernel);
otherwise the op is exactly what it was. RADIANCE_GDN_NORM_QUANT_TILED=0 forces the old path. Non-fatal in the launcher.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

ARNQ = Path(sysconfig.get_paths()["purelib"]) / "radiance_arnq.py"
SENTINEL = "RADIANCE_GDN_NORM_QUANT_TILED"

apply(ARNQ,
      '    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)\n'
      '    sc = torch.empty((M,), dtype=torch.float32, device=x.device)\n'
      '    _ext().launch_gdn_norm_quant(x.data_ptr(), z.data_ptr(), z.stride(0), w.data_ptr(),\n'
      '                                 q.data_ptr(), sc.data_ptr(), M, N, float(eps),\n'
      '                                 torch.cuda.current_stream().cuda_stream)\n'
      '    return q, sc\n',
      '    # House patch (patch_gdn_norm_quant_tiled.py): fragment-tiled output for the A-direct prefill GEMM, as silu_mul_quant.\n'
      '    import radiance_mxfp4 as _rm\n'
      '    tiled = (os.environ.get("RADIANCE_GDN_NORM_QUANT_TILED", "1") == "1" and N % 16 == 0\n'
      '             and "tiled" in (_ext().launch_gdn_norm_quant.__doc__ or "") and _rm.a_tiled_wanted(M))\n'
      '    if tiled:\n'
      '        q = torch.empty(((M + 15) // 16 * 16, N), dtype=torch.float8_e4m3fn, device=x.device)[:M]\n'
      '    else:\n'
      '        q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)\n'
      '    sc = torch.empty((M,), dtype=torch.float32, device=x.device)\n'
      '    if tiled:\n'
      '        _ext().launch_gdn_norm_quant(x.data_ptr(), z.data_ptr(), z.stride(0), w.data_ptr(),\n'
      '                                     q.data_ptr(), sc.data_ptr(), M, N, float(eps),\n'
      '                                     torch.cuda.current_stream().cuda_stream, 1)\n'
      '        _rm.a_tiled_register(q, M, N)\n'
      '    else:\n'
      '        _ext().launch_gdn_norm_quant(x.data_ptr(), z.data_ptr(), z.stride(0), w.data_ptr(),\n'
      '                                     q.data_ptr(), sc.data_ptr(), M, N, float(eps),\n'
      '                                     torch.cuda.current_stream().cuda_stream)\n'
      '    return q, sc\n',
      SENTINEL, "gdn-norm-quant tiled output")
