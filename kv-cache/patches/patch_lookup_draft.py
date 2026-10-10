#!/usr/bin/env python3
"""House patch 2026-10-09: SlyBase prompt-lookup override of the DFlash2 draft (radiance_lookup_draft).

Copies the house radiance_lookup_draft.py into site-packages and calls its install() at the end of
vllm/v1/worker/gpu/model_runner.py (the V2 runner), which wraps GPUModelRunner.__init__ so a DFlash2
speculator gets its propose() wrapped. Lossless: lookup drafts become point masses for the rejection
sampler. Any error turns the override off for the process. RADIANCE_LOOKUP_DRAFT=0 = control.
"""
import ast
import shutil
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])
shutil.copyfile(Path(__file__).with_name("radiance_lookup_draft.py"), SP / "radiance_lookup_draft.py")
F = SP / "vllm/v1/worker/gpu/model_runner.py"
SENTINEL = "house patch_lookup_draft.py"
s = F.read_text()
if SENTINEL in s:
    print("  NOOP  gpu/model_runner: lookup draft install already applied")
else:
    s += ("\n\ntry:  # " + SENTINEL + ": prompt-lookup override of the DFlash2 draft\n"
          "    import radiance_lookup_draft as _radiance_lookup_draft\n"
          "    _radiance_lookup_draft.install()\n"
          "except Exception as _e:  # never take the runner down\n"
          "    import sys as _sys\n"
          "    _sys.stderr.write(f\"[radiance] radiance_lookup_draft install failed: {_e!r}\\n\")\n")
    ast.parse(s)
    F.write_text(s)
    print("  OK    gpu/model_runner: lookup draft install")
