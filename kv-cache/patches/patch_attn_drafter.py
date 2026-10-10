#!/usr/bin/env python3
"""House patch 2026-10-09: SlyBase window-sliced split-KV launch for the DFlash2 drafter attention.

Copies radiance_attn_drafter.py (house copy, unmodified Sly module) into site-packages and calls its
install() right after vllm/v1/attention/backends/triton_attn.py imports unified_attention, so the
backend's call site resolves to the wrapper. The wrapper takes only the drafter's shape (fp8 KV,
head 128, GQA 4, sliding window, 2..16 queries per sequence); everything else, and any error, goes to
vLLM's own launch. RADIANCE_ATTN_DRAFTER_TUNE=0 turns it off (A/B control).
"""
import shutil
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
shutil.copyfile(Path(__file__).with_name("radiance_attn_drafter.py"), SP / "radiance_attn_drafter.py")
F = SP / "vllm/v1/attention/backends/triton_attn.py"
ANCHOR = "from vllm.v1.attention.ops.triton_unified_attention import unified_attention\n"
apply(F,
      ANCHOR,
      ANCHOR +
      "try:  # house patch_attn_drafter.py: split-KV launch for the DFlash2 drafter (falls back to vLLM)\n"
      "    import radiance_attn_drafter as _radiance_attn_drafter\n"
      "    _radiance_attn_drafter.install()\n"
      "except Exception as _e:  # never take the backend down\n"
      "    import sys as _sys\n"
      "    _sys.stderr.write(f\"[radiance] radiance_attn_drafter install failed: {_e!r}\\n\")\n",
      "house patch_attn_drafter.py",
      "triton_attn: drafter split-KV attention (radiance_attn_drafter)")
