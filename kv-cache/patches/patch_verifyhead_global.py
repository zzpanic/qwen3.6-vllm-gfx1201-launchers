#!/usr/bin/env python3
"""Install the PR #9 target verify head (global top-256 candidates) over the vendored one.

The block shortlist keeps 8 candidates per 64-token vocabulary tile, so with top_k=20 sampling the
true top-20 survived in only ~82% of rows (PR #9's 120k-row measurement); global top-256 selection
keeps 99.83% at the same speed. Replaces site-packages/radiance_verifyhead.py ONLY if it is the
exact base the port was made against (sha256 prefix below); otherwise it changes nothing and says so.
"""
import hashlib
import shutil
import sys
import sysconfig
from pathlib import Path

BASE_SHA = "7fc05ceb76672aa9"     # ggz14 980f891 radiance_verifyhead.py == vllm-radiance 2d0ffaff
dst = Path(sysconfig.get_paths()["purelib"]) / "radiance_verifyhead.py"
src = Path(__file__).resolve().parent / "radiance_verifyhead_global.py"
cur = hashlib.sha256(dst.read_bytes()).hexdigest()
if dst.read_text().startswith("# radiance house port 2026-10-06"):
    print("  NOOP  verify head: global top-k already installed")
elif cur.startswith(BASE_SHA):
    shutil.copyfile(src, dst)
    print("  OK    verify head: global top-256 candidate selection (vllm-radiance PR #9)")
else:
    sys.exit(f"  FAIL  verify head: base changed (sha {cur[:16]} != {BASE_SHA}); port PR #9 again")
