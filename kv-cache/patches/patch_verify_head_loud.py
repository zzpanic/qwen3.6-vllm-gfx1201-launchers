#!/usr/bin/env python3
"""Make the verify-head hook report its first exception instead of swallowing every one.

patch_verify_head.py (ggz14) wraps radiance_verifyhead.before_compute_logits in
`except Exception: pass`, so any failure inside the verify head silently leaves it unarmed and
every step on the full bf16 head -- found 2026-10-06 when it never armed with FAST_DRAFT=1 and
printed nothing. Still non-fatal (serving continues on the exact head); now it says why, once.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

apply(Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/worker/gpu/model_runner.py",
      "            _radiance_vh.before_compute_logits(self, input_batch, grammar_output)\n"
      "        except Exception:\n"
      "            pass\n",
      "            _radiance_vh.before_compute_logits(self, input_batch, grammar_output)\n"
      "        except Exception as _rvh_e:\n"
      "            # radiance house patch (patch_verify_head_loud.py): say why, once.\n"
      "            if not getattr(self, \"_rvh_err_reported\", False):\n"
      "                self._rvh_err_reported = True\n"
      "                import sys as _rvh_sys, traceback as _rvh_tb\n"
      "                _rvh_sys.stderr.write(\"[radiance.verifyhead] hook FAILED (full bf16 head in use): \"\n"
      "                                      + \"\".join(_rvh_tb.format_exception(_rvh_e))[-1500:] + \"\\n\")\n",
      "radiance house patch (patch_verify_head_loud.py)", "V2 runner: verify-head hook reports its failure")
