#!/usr/bin/env python3
"""Regenerate rdna_hybrid_w4a16.py from a bench_gapfill.py sweep log.

Usage: python3 build_patch.py [sweep_log] [orig_kernel] [out_kernel]
Defaults: bench_gapfill_results.tsv, rdna_hybrid_w4a16.orig.py, rdna_hybrid_w4a16.py
(all resolved relative to this script's directory).

Only needed if you re-run bench_gapfill.py yourself (different checkpoint shapes,
a driver/Triton update, etc.) and want to regenerate the override table from fresh
data. If you just want to use the existing tuning, rdna_hybrid_w4a16.py in this
directory is the ready-to-use output -- you don't need to run this.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "bench_gapfill_results.tsv"
ORIG = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "rdna_hybrid_w4a16.orig.py"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "rdna_hybrid_w4a16.py"

GROUP_SIZE = 32
SHAPES = {
    "down_proj": (17408, 5120),
    "gate_up_proj": (5120, 17408),
    "kv_proj": (5120, 1024),
    "q_proj": (5120, 12288),
    "o_proj": (6144, 5120),
    "gdn_in_proj_qkv": (5120, 10240),
    "gdn_in_proj_z": (5120, 6144),
    "gdn_out_proj": (6144, 5120),
}

row_re = re.compile(
    r"^(?P<name>\w+)\t(?P<M>\d+)\t\w+\t\([^)]*\)\t[\d.]+\t"
    r"\((?P<bm>\d+),(?P<bn>\d+),(?P<nw>\d+),ns=(?P<ns>None|\d+)\)\t"
    r"(?P<best_ms>[\d.]+)\t(?P<pct>[+-][\d.]+)%$"
)

M_TO_BUCKET = {16: 32, 48: 64, 96: 128, 300: 512, 2560: "big"}

# key: (K, N, bucket) -> (bm, bn, nw, ns, best_ms, pct, name) ; keep the entry with
# the lower measured best_ms when two different shape *labels* collide on the same
# real (K, N) (o_proj and gdn_out_proj are both 6144->5120 -- same GEMM, so whichever
# run actually measured faster is the one to trust, not benchmark-order luck).
# Regressions (pct < 0) are dropped entirely -- fall through to the existing generic
# heuristic for that bucket instead of installing something proven worse.
best_by_key: dict[tuple[int, int, object], tuple] = {}

with open(LOG) as f:
    for line in f:
        m = row_re.match(line.strip())
        if not m:
            continue
        name = m["name"]
        if name not in SHAPES:
            continue
        K, N = SHAPES[name]
        M = int(m["M"])
        bucket = M_TO_BUCKET[M]
        pct = float(m["pct"])
        best_ms = float(m["best_ms"])
        if pct < 0:
            print(f"SKIP (regression): {name} K={K} N={N} bucket={bucket} pct={pct:+.1f}%")
            continue
        key = (K, N, bucket)
        ns = None if m["ns"] == "None" else int(m["ns"])
        entry = (int(m["bm"]), int(m["bn"]), int(m["nw"]), ns, best_ms, pct, name)
        prev = best_by_key.get(key)
        if prev is None or best_ms < prev[4]:
            if prev is not None:
                print(
                    f"COLLISION {key}: {prev[6]}({prev[4]:.4f}ms) vs {name}({best_ms:.4f}ms)"
                    f" -> keeping {name if best_ms < prev[4] else prev[6]}"
                )
            best_by_key[key] = entry

assert len(best_by_key) > 0, "no rows parsed from log -- check the regex against the log format"

lines = []
lines.append("_GFX1201_PREFILL_OVERRIDES: dict[tuple[int, int, int, int], tuple[int, int, int, int, int | None]] = {")
lines.append("    # (group_size, K, N, m_bucket) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)")
lines.append("    # m_bucket is the bucket's own upper M bound (32/64/128/512), or _GFX1201_MBIG for M>512.")
for (K, N, bucket), (bm, bn, nw, ns, best_ms, pct, name) in sorted(
    best_by_key.items(), key=lambda kv: (kv[1][6], kv[0][2] if isinstance(kv[0][2], int) else 10**9)
):
    bucket_lit = "_GFX1201_MBIG" if bucket == "big" else bucket
    lines.append(
        f"    ({GROUP_SIZE}, {K}, {N}, {bucket_lit}): ({bm}, {bn}, {GROUP_SIZE}, {nw}, {ns}),"
        f"  # {name} M<={bucket} ({pct:+.1f}%)"
    )
lines.append("}")
table_src = "\n".join(lines)

helper_src = '''

def _gfx1201_override(
    group_size: int, K: int, N: int, M: int
) -> tuple[int, int, int, int, int | None] | None:
    if M <= 32:
        bucket = 32
    elif M <= 64:
        bucket = 64
    elif M <= 128:
        bucket = 128
    elif M <= 512:
        bucket = 512
    else:
        bucket = _GFX1201_MBIG
    return _GFX1201_PREFILL_OVERRIDES.get((group_size, K, N, bucket))
'''

src = ORIG.read_text()

anchor = "def triton_w4a16_skinny_fmt_gemm("
assert src.count(anchor) == 1
insert = "_GFX1201_MBIG = 1 << 30\n\n" + table_src + "\n" + helper_src + "\n\n"
src = src.replace(anchor, insert + anchor, 1)

old_branch_start = "    if _on_gfx12x():\n        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)\n        # using Llama-3.1-8B AWQ weight shapes with group_size=128.\n        if M <= 32:"
new_branch_start = (
    "    if _on_gfx12x():\n"
    "        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)\n"
    "        # using Llama-3.1-8B AWQ weight shapes with group_size=128.\n"
    "        override = _gfx1201_override(group_size, K, N, M)\n"
    "        if override is not None:\n"
    "            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = override\n"
    "        elif M <= 32:"
)
assert src.count(old_branch_start) == 1, "anchor for gfx12x branch not found -- did the upstream file change?"
src = src.replace(old_branch_start, new_branch_start, 1)

OUT.write_text(src)
print()
print(f"wrote {OUT}: {len(best_by_key)} override entries")
print()
print(table_src)
