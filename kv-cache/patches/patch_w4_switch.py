#!/usr/bin/env python3
"""Give radiance_w4 its own switch (RADIANCE_W4, default 1 = upstream behaviour).

radiance_w4 arms whenever RADIANCE_FAST_DRAFT=1 and libr4d has a w4a16 gemm_nt. It converts the
DFlash2 drafter's linears and frees layer.weight to an empty tensor, and DFlash2's fused context-KV
precompute then reads weight.shape[1] -> IndexError in rocm_unquantized_gemm_impl at load (the
launcher's FAST_DRAFT comment, 2026-08-27). The old libr4d pin had no w4a16 kernel, so FAST_DRAFT
was measured (+5.1% decode) with radiance_w4 OFF; the v0.5.0-based rx13/rx14 builds ship the kernel
and the crash returned (2026-10-06, a boot loop). RADIANCE_W4=0 restores the measured combination:
the int2 draft/verify heads on, the w4 drafter conversion off.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

apply(Path(sysconfig.get_paths()["purelib"]) / "radiance_w4.py",
      'ENABLED = USE_R4D and os.environ.get("RADIANCE_FAST_DRAFT", "0") == "1"\n',
      'ENABLED = USE_R4D and os.environ.get("RADIANCE_FAST_DRAFT", "0") == "1"\n'
      '# radiance house patch (patch_w4_switch.py): RADIANCE_W4=0 keeps the int2 heads, drops w4.\n'
      'ENABLED = ENABLED and os.environ.get("RADIANCE_W4", "1") == "1"\n',
      "radiance house patch (patch_w4_switch.py)", "radiance_w4: own switch (RADIANCE_W4)")
