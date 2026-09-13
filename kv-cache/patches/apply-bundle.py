#!/usr/bin/env python3
"""Apply the radiance KV-offload bundle (task 20) in the correct order.

Order, and why:
  1. patch_promotion_refusal_instrumentation.py   (00 Phase A: the counters.
     The counter reading zero is the EVIDENCE the fix was unnecessary, so the
     fix patch (00 Phase B: B2 headroom + B1 bounded retry) was removed in
     task 37. This patch runs first because the wall-clock patch's hunk 2
     anchors on the CONN_METRICS definitions dict tail this patch leaves there
     (the PROMOTION_INITIATED entry).)
  2. patch_kv_offload_wallclock_reanchored.py    (15: wall-clock whole-job
     timing. Its hunk 2 is re-anchored to the CONN_METRICS definitions dict
     tail that patch 1 leaves there, so it must run AFTER patch 1. Do not swap
     in the out/15 original: its anchors collided with patch 1 and fail with
     "anchor matched 0x" on the instrumented tree.)

Both require the house patches 1-8 (..patch_kv_offload_tier_report.py) to be
applied already: they reference TierReportMetrics, TieringOffloadingMetrics
and _tr_tokens_per_hash, which the tier-report patch creates. This wrapper does
NOT re-apply the house patches; it assumes they are done and applies the two
bundle patches on top, in the order above, halting on the first failure.

Each patch is a standalone script that imports _patchlib and edits the target
venv's site-packages (SP = sysconfig purelib). The wrapper runs them in order
with the bundle directory on PYTHONPATH (so `_patchlib` resolves), and prints a
per-patch status. A mid-bundle failure stops here -- the file that failed was
not written, so the state is the state after the last fully-applied patch,
which is a consistent, revertible point.
"""
import os
import subprocess
import sys

BUNDLE = os.path.dirname(os.path.abspath(__file__))
PATCHES = [
    "patch_promotion_refusal_instrumentation.py",
    "patch_kv_offload_wallclock_reanchored.py",
]


def main() -> int:
    env = dict(os.environ)
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = BUNDLE + (os.pathsep + prev if prev else "")

    for name in PATCHES:
        path = os.path.join(BUNDLE, name)
        print(f"[bundle] === {name} ===", flush=True)
        r = subprocess.run([sys.executable, path], env=env)
        if r.returncode != 0:
            print(
                f"[bundle] FAILED at {name} (exit {r.returncode}) -- halting. "
                "The state is the state after the last fully-applied patch.",
                flush=True,
            )
            return r.returncode

    print(
        "[bundle] all 2 patches applied: instrumentation -> wall-clock.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
