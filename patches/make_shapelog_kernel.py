#!/usr/bin/env python3
"""Generate an instrumented copy of the production W4A16 kernel that records the
(group_size, K, N, M) tuples it is actually called with.

WHY THIS EXISTS. The 2026-08-09 tile sweep tuned 8 shapes read off the CHECKPOINT's
tensor list. Three of them cannot occur at runtime, because vLLM fuses:
    qwen3_next.py:267        self.qkv_proj  = QKVParallelLinear(...)
    qwen2_moe.py (MLP)       self.gate_up_proj = MergedColumnParallelLinear(...)
so q_proj(N=12288) + k/v_proj(N=1024) become ONE GEMM at N=14336, and
gate_proj/up_proj (N=17408 each) become ONE GEMM at N=34816. Any override row keyed
on the unfused N is dead code.

That is a static reading, i.e. a hypothesis. This module turns it into a measurement:
run the real server for one request and read back exactly which shapes the kernel saw.

Purely additive and OFF unless RDNA_W4A16_SHAPELOG is set to an output path, so the
same file is safe to leave bind-mounted.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# usage: make_shapelog_kernel.py [DST] [SRC]
# SRC defaults to the tuned kernel next to this script; pass rdna_hybrid_w4a16.orig.py
# instead to instrument the stock kernel.
DST = Path(sys.argv[1] if len(sys.argv) > 1 else _HERE / "rdna_hybrid_w4a16_shapelog.py")
SRC = Path(sys.argv[2] if len(sys.argv) > 2 else _HERE / "rdna_hybrid_w4a16.py")

PREAMBLE = '''
# --- shape census instrumentation (additive; inert unless RDNA_W4A16_SHAPELOG set) ---
import atexit as _sl_atexit
import json as _sl_json
import os as _sl_os
import threading as _sl_threading

_SL_PATH = _sl_os.environ.get("RDNA_W4A16_SHAPELOG")
_SL_LOCK = _sl_threading.Lock()
_SL_SEEN: dict = {}
_SL_DIRTY = False


def _sl_bucket(M: int):
    for b in (32, 64, 128, 512):
        if M <= b:
            return b
    return "big"


def _sl_flush():
    # NEVER write an empty census. vLLM runs at least two processes (APIServer and
    # EngineCore) and BOTH import this module; only EngineCore ever calls the kernel.
    # Without this guard the APIServer process's atexit handler writes [] over the
    # real file at teardown -- which is exactly what happened on the first run, and it
    # looks identical to "the kernel was never called".
    if _SL_PATH is None or not _SL_SEEN:
        return
    with _SL_LOCK:
        rows = [
            {
                "group_size": gs, "K": K, "N": N, "bucket": bucket,
                "calls": v["calls"], "m_min": v["m_min"], "m_max": v["m_max"],
                "m_examples": sorted(v["m_examples"])[:24],
            }
            for (gs, K, N, bucket), v in sorted(_SL_SEEN.items(), key=lambda kv: -kv[1]["calls"])
        ]
        tmp = _SL_PATH + ".tmp"
        with open(tmp, "w") as fh:
            _sl_json.dump(rows, fh, indent=1)
        _sl_os.replace(tmp, _SL_PATH)


def _sl_record(group_size: int, K: int, N: int, M: int):
    global _SL_DIRTY
    key = (group_size, K, N, _sl_bucket(M))
    with _SL_LOCK:
        e = _SL_SEEN.get(key)
        if e is None:
            e = _SL_SEEN[key] = {"calls": 0, "m_min": M, "m_max": M, "m_examples": set()}
            new = True
        else:
            new = False
        e["calls"] += 1
        if M < e["m_min"]:
            e["m_min"] = M
        if M > e["m_max"]:
            e["m_max"] = M
        if len(e["m_examples"]) < 64:
            e["m_examples"].add(M)
        # Flush on every newly-seen key and then every 512 calls: the container is
        # torn down with `podman rm -f`, so an atexit-only dump would be lost.
        due = new or (e["calls"] % 512 == 0)
    if due:
        _sl_flush()


if _SL_PATH is not None:
    _sl_atexit.register(_sl_flush)
# --- end shape census instrumentation ------------------------------------------------

'''

HOOK = """    if _SL_PATH is not None:
        _sl_record(group_size, K, N, M)

"""

src = SRC.read_text()

anchor_fn = "def triton_w4a16_skinny_fmt_gemm("
assert src.count(anchor_fn) == 1, "gemm entry point not found exactly once"
src = src.replace(anchor_fn, PREAMBLE.lstrip("\n") + "\n" + anchor_fn, 1)

# Hook goes AFTER M/K/N are derived and after the shape asserts, so a malformed call
# still fails the same way it would unpatched.
anchor_hook = "    c = torch.empty((M, N), dtype=a.dtype, device=a.device)\n"
assert src.count(anchor_hook) == 1, "output-alloc anchor not found exactly once"
src = src.replace(anchor_hook, HOOK + anchor_hook, 1)

DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines, +{len(src.splitlines()) - len(SRC.read_text().splitlines())})")
