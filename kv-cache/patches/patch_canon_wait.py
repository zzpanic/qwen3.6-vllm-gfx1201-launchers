#!/usr/bin/env python3
"""Canonical chunks: a prefill cut short by another prefill waits one step instead of freezing its cache.

WHY (the author's bench notes (concreplay-20261010/notes.md, not shipped), smoke + c4/c3; checkpoint-policy-20261010/PLAN.md step 3)
=========================================================================================================
Hunk 4 caches a request only up to _radiance_canon_end, which advances through FULL grid chunks only. With
CHUNK 3568 a grid chunk fits beside up to three spec decodes (hunk 6), but when ANOTHER request's prefill takes
part of the step budget, this request's chunk is the remainder floored to a block -- an off-grid chunk -- and its
canon_end is frozen for the rest of the request: nothing past that point is cached, in VRAM or in the tiers.
Smoke run: two such overlaps cost 56,320 tokens of follow-up recompute (one turn resumed at 10,560 of 59,840).

WHAT IT CHANGES (gate RADIANCE_CANON_WAIT=1, canonical chunks on; 0/unset = today)
  Scheduler._mamba_block_aligned_split: a prefill chunk that
    - starts on the grid at the request's canon_end (still canonical; a new request counts),
    - could run a full grid chunk (start + S <= num_prompt_tokens),
    - was handed fewer than S tokens (the step budget is the only limit then: S <= the threshold, and the request
      has >= S tokens left; requests with encoder inputs are left alone),
  and would still run a non-empty (off-grid) chunk, is scheduled as 0 tokens. (Handed less than a block, it
  gets 0 already -- upstream's block floor -- and is not counted.) Running queue: upstream's "continue" (the next request is tried); waiting queue:
  upstream's "break" -- both are the existing path for a chunk that cannot be block-aligned. Next step the other
  prefill is done or behind it, and the chunk runs whole: canonical, cached, exact.
  Safety cap: after RADIANCE_CANON_WAIT_MAX (default 8) consecutive waits the chunk runs as today (frozen).
  Counted as "waited" / "wait_cap" in the periodic "radiance canon mixed" log line.
Exact by construction (it only delays work). Cost: <= one step (~1.2 s at 3,520 tokens) per overlap, and the
leftover budget of that step goes unused by this request -- the design aim: wait rather than recompute.
Residual: a full grid chunk that still shares its step with a tiny prefill tail (<= the 36-token headroom) is
refused by hunk 6 and freezes as before.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SCHED = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/core/sched/scheduler.py"

apply(SCHED,
      "import itertools\n",
      "import itertools\n"
      "import os as _radiance_cw_os\n"
      "# radiance: see patch_canon_wait.py. Unset/0 = upstream behaviour.\n"
      "_RADIANCE_CANON_WAIT = _radiance_cw_os.environ.get(\"RADIANCE_CANON_WAIT\", \"0\") == \"1\"\n"
      "_RADIANCE_CANON_WAIT_MAX = int(_radiance_cw_os.environ.get(\"RADIANCE_CANON_WAIT_MAX\", \"8\"))\n",
      "_RADIANCE_CANON_WAIT = ",
      "canon wait: gate constant")

apply(SCHED,
      "        end = min((s for s in stops if start < s < end), default=end)\n"
      "        return max(end - start, 0)\n",
      "        end = min((s for s in stops if start < s < end), default=end)\n"
      "        if _RADIANCE_CANON_WAIT and _RADIANCE_CANONICAL_CHUNKS and not request.has_encoder_inputs:\n"
      "            # radiance (patch_canon_wait.py): a canonical prefill cut below one grid chunk by the\n"
      "            # step budget waits a step rather than running off-grid and freezing its cache.\n"
      "            _rad_g = self._radiance_cold_grid_stop(0, block_size)\n"
      "            if (_rad_g and num_new_tokens < _rad_g and start < end and start % _rad_g == 0\n"
      "                    and start + _rad_g <= request.num_prompt_tokens\n"
      "                    and getattr(request, \"_radiance_canon_end\", start) == start):\n"
      "                _rad_st = getattr(self.kv_cache_manager, \"_radiance_canon_stats\", None)\n"
      "                _rad_w = getattr(request, \"_radiance_waits\", 0)\n"
      "                if _rad_w < _RADIANCE_CANON_WAIT_MAX:\n"
      "                    request._radiance_waits = _rad_w + 1\n"
      "                    if _rad_st is not None:\n"
      "                        _rad_st[\"waited\"] = _rad_st.get(\"waited\", 0) + 1\n"
      "                    return 0\n"
      "                if _rad_st is not None:\n"
      "                    _rad_st[\"wait_cap\"] = _rad_st.get(\"wait_cap\", 0) + 1\n"
      "            request._radiance_waits = 0\n"
      "        return max(end - start, 0)\n",
      "radiance (patch_canon_wait.py): a canonical prefill cut",
      "canon wait: wait instead of an off-grid chunk")
