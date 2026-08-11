#!/usr/bin/env python3
"""Turn sweep.tsv into group_size=128 rows and splice them into the tuned kernel.

ADDITIVE, not a rewrite: the existing group_size=32 rows stay, so reverting
MODEL_DIR to the old Avesed checkpoint still gets its tuning. The output is written to
a CANDIDATE path, never over the production file -- installing it is a separate,
deliberate step after the A/B says it earned it.

Two filters, both inherited from the gs=32 builder and both load-bearing:
  * a bucket where the sweep did not beat the stock heuristic is DROPPED, so that
    lookup misses and the heuristic runs. Installing a proven-slower tile is worse
    than installing nothing.
  * when two shape labels collide on the same real (K, N), keep the LOWER measured
    time -- same GEMM, so the faster measurement is the honest one, not whichever
    happened to be benchmarked second.

New here: a MIN_PCT floor. At gs=32 the stock heuristic was running outside the
regime it was tuned for (its own comment says group_size=128), so wins were huge and
anything positive was worth taking. At gs=128 it is on home ground and small deltas
are within microbench noise -- taking a +0.4% row costs a permanent maintenance
liability for nothing. Default floor 3%.
"""

import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# usage: build_patch_gs128.py [SWEEP.tsv] [SRC kernel] [DST kernel]
SWEEP = sys.argv[1] if len(sys.argv) > 1 else str(_HERE / "bench_gapfill_gs128_results.tsv")
SRC = sys.argv[2] if len(sys.argv) > 2 else str(_HERE / "rdna_hybrid_w4a16.py")
DST = sys.argv[3] if len(sys.argv) > 3 else str(_HERE / "rdna_hybrid_w4a16_gs128.candidate.py")
MIN_PCT = float(os.environ.get("MIN_PCT", "3.0"))

GROUP_SIZE = 128

_MARKER = "--- group_size 128 (Intel AutoRound checkpoint"

# NOT IDEMPOTENT, so refuse rather than corrupt. Running this over a kernel that
# already carries the gs=128 block appends a second copy of every [DEAD] annotation
# and a second copy of the table. Regenerate from `rdna_hybrid_w4a16.orig.py` (stock,
# pulled from the image) or from the gs=32-only output of `build_patch.py`.
def _refuse_if_already_patched(src_text: str) -> None:
    if _MARKER in src_text:
        sys.exit(
            f"{SRC}\nalready contains the group_size=128 override block. This script is "
            "additive and not\nidempotent -- re-running it here would duplicate the table "
            "and the [DEAD] comments.\nPass a clean source, e.g.:\n"
            "    ./build_patch_gs128.py bench_gapfill_gs128_results.tsv "
            "rdna_hybrid_w4a16.orig.py out.py")

# Only the two M values the census found to carry real traffic become table rows.
# M=5 and M=20 are swept as diagnostics and DELIBERATELY NOT installed: they share
# bucket 32 with M=10, and resolving that collision by lowest absolute ms would just
# pick whichever M is smallest, which measures nothing about tile quality.
M_TO_BUCKET = {10: 32, 2560: "big"}
DIAGNOSTIC_M = {5, 20, 1664}

# bench_gapfill_gs128.py names census shapes "K<K>xN<N>"; accept the fallback labels too.
row_re = re.compile(
    r"^(?P<name>[\w./]+)\t(?P<M>\d+)\t\w+\t\([^)]*\)\t(?P<cur_ms>[\d.]+|inf)\t"
    r"\((?P<bm>\d+),(?P<bn>\d+),(?P<bk>\d+),(?P<nw>\d+),ns=(?P<ns>None|\d+)\)\t"
    r"(?P<best_ms>[\d.]+)\t(?P<pct>[+-][\d.]+)%$"
)
kn_re = re.compile(r"^K(?P<K>\d+)xN(?P<N>\d+)$")

best_by_key: dict[tuple[int, int, object], tuple] = {}
dropped: list[str] = []

for line in open(SWEEP):
    m = row_re.match(line.rstrip("\n"))
    if not m:
        continue
    kn = kn_re.match(m["name"])
    if not kn:
        print(f"SKIP (shape label not K<K>xN<N>, cannot key the table): {m['name']}")
        continue
    K, N = int(kn["K"]), int(kn["N"])
    M = int(m["M"])
    if M in DIAGNOSTIC_M:
        continue
    if M not in M_TO_BUCKET:
        print(f"SKIP (M={M} carries no census traffic): K={K} N={N}")
        continue
    bucket = M_TO_BUCKET[M]
    pct, best_ms = float(m["pct"]), float(m["best_ms"])
    if pct < MIN_PCT:
        dropped.append(f"  K={K} N={N} bucket={bucket}: {pct:+.1f}% < {MIN_PCT}% floor")
        continue
    key = (K, N, bucket)
    entry = (int(m["bm"]), int(m["bn"]), int(m["bk"]), int(m["nw"]),
             None if m["ns"] == "None" else int(m["ns"]), best_ms, pct, m["name"])
    prev = best_by_key.get(key)
    if prev is not None:
        print(f"COLLISION {key}: {prev[7]}({prev[5]:.4f}ms) vs {m['name']}({best_ms:.4f}ms)")
    if prev is None or best_ms < prev[5]:
        best_by_key[key] = entry

print(f"\n{len(best_by_key)} rows kept, {len(dropped)} dropped below the {MIN_PCT}% floor:")
for d in dropped:
    print(d)

if not best_by_key:
    print("\nNOTHING CLEARED THE FLOOR -- the stock heuristic is already at or near "
          "optimal for these shapes at group_size=128. That is a RESULT, not a "
          "failure: do not install a table, and do not run the A/B.")
    sys.exit(3)

lines = []
for (K, N, bucket), (bm, bn, bk, nw, ns, best_ms, pct, name) in sorted(
        best_by_key.items(),
        key=lambda kv: (kv[0][0], kv[0][1],
                        kv[0][2] if isinstance(kv[0][2], int) else 10**9)):
    bucket_lit = "_GFX1201_MBIG" if bucket == "big" else bucket
    lines.append(f"    ({GROUP_SIZE}, {K}, {N}, {bucket_lit}): "
                 f"({bm}, {bn}, {bk}, {nw}, {ns}),  # {name} M<={bucket} ({pct:+.1f}%)")

block = (
    "    # --- group_size 128 (Intel AutoRound checkpoint, PRODUCTION from 2026-08-11) ---\n"
    "    # Swept 2026-08-11 on the SHAPES THE KERNEL IS ACTUALLY CALLED WITH (runtime\n"
    "    # census, patches/census-shapes-gs128.json) rather than the\n"
    "    # checkpoint's tensor names -- vLLM fuses qkv and gate_up, so the two largest\n"
    "    # GEMMs here, (5120,34816) and (5120,14336), have no group_size=32 counterpart\n"
    "    # above and never did. BLOCK_K is a real column in these rows: min(BLOCK_K,\n"
    "    # group_size) no longer pins it to 32.\n"
    + "\n".join(lines) + "\n"
)

src = open(SRC).read()
_refuse_if_already_patched(src)
anchor = "}\n\n\ndef _gfx1201_override("
assert src.count(anchor) == 1, "could not find the end of _GFX1201_PREFILL_OVERRIDES"
src = src.replace(anchor, block + anchor, 1)

# Mark the gs=32 rows that can never be looked up, so the next reader does not spend an
# hour re-deriving why tuning them bought nothing. FIVE of the old sweep's eight shapes
# are dead, not three -- the runtime census caught the GDN pair too, which no amount of
# reading the model file had predicted. Only down_proj (17408,5120) and o_proj /
# gdn_out_proj (6144,5120) were ever really tuned.
for K, N, why in ((5120, 17408, "gate_proj/up_proj are FUSED into gate_up_proj N=34816"),
                  (5120, 12288, "q_proj is FUSED into qkv_proj N=14336"),
                  (5120, 1024, "k_proj/v_proj are FUSED into qkv_proj N=14336"),
                  (5120, 10240, "gdn in_proj_qkv is FUSED with in_proj_z, N=16384"),
                  (5120, 6144, "gdn in_proj_z is FUSED with in_proj_qkv, N=16384")):
    pat = re.compile(rf"^(    \(32, {K}, {N}, [^)]*\): \([^)]*\),)(  #.*)$", re.M)
    n = len(pat.findall(src))
    if n:
        src = pat.sub(rf"\1\2  [DEAD: {why}]", src)
        print(f"marked {n} dead gs=32 row(s) for K={K} N={N}")

open(DST, "w").write(src)
print(f"\nwrote {DST}")
print()
print(block)
