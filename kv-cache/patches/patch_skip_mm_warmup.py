#!/usr/bin/env python3
"""Skip the startup multi-modal processor warmup (RADIANCE_SKIP_MM_WARMUP).

Target: vllm/renderers/base.py, Renderer.warmup() -> _warmup_mm_processor().

The OpenAI api server calls state.online_renderer.warmup() on every boot, which runs
processor.apply() over a DUMMY input sized to max_model_len with one image of the
maximum feature size. For a text-first deployment (this one: --limit-mm-per-prompt
image:999 is headroom, image requests are rare) that is ~21.5 s of one-shot CPU
work (HF image processor + a max_model_len-token placeholder pass) on every boot,
and nothing of it persists: the mm cache is cleared in the warmup's finally clause,
so the next boot pays it again.

The hunk makes _warmup_mm_processor() return early when RADIANCE_SKIP_MM_WARMUP=1.
Cost: the first real image request of the boot pays the ~20 s once, on its own
request. Everything else in warmup() (chat template, jinja) still runs, so the first
TEXT request is untouched. Default is 0 (stock behaviour); the launcher sets 1.
"""
import os
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

VLLM = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"] + "/vllm"))
BASE = VLLM / "renderers" / "base.py"

ANCHOR = (
    "    ) -> None:\n"
    "        from vllm.multimodal.processing import TimingContext\n"
)
NEW = (
    "    ) -> None:\n"
    "        import os as _radiance_os\n"
    "        if _radiance_os.environ.get(\"RADIANCE_SKIP_MM_WARMUP\", \"0\") == \"1\":\n"
    "            logger.info(\"%s warmup skipped (RADIANCE_SKIP_MM_WARMUP=1)\", log_prefix)\n"
    "            return\n"
    "        from vllm.multimodal.processing import TimingContext\n"
)
SENTINEL = "RADIANCE_SKIP_MM_WARMUP"

apply(BASE, ANCHOR, NEW, SENTINEL, "skip-mm-warmup")
