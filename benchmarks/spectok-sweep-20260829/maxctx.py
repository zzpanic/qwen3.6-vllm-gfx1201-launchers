#!/usr/bin/env python3
"""Max-context and pool per SPECTOK, calibrated to the 2026-08-29 measured sweep.

The earlier model (bench-history/kv-group-padding-20260829/kvgroup.py) hardcoded
BLOCK=1616. That is only true for K<=5: the sweep MEASURED the attention block size
moving with K (1616, 1616, 1632, 1648 for K=4,5,6,7), and block size feeds both the
page size and the number of blocks a full-attn layer needs, so it must be per-K.

Calibration: K=4 warm boot measured AVAIL=8.98 GiB -> 231,602 tokens @ MAXLEN=131072.
"""
def cdiv(a, b): return -(-a // b)

BLOCK = {4: 1616, 5: 1616, 6: 1632, 7: 1648}   # MEASURED, not assumed
G = 8                     # KV_GROUP_SIZE=auto resolved to 8
AVAIL = 8.98 * 1024**3    # warm-boot available KV, measured K=4
KVH, HD = 4, 256          # kv heads, head_dim; fp8 -> 1 B/elt; K+V -> x2
BUCKETS = lambda K, blk, maxlen: [
    (48, 2 + K),                       # GDN / mamba: align mode
    (16, cdiv(maxlen, blk)),           # full attention
    (5,  6),                           # dflash2 drafter (sliding_window 2048)
]

def pool(K, maxlen):
    blk = BLOCK[K]
    page = blk * KVH * HD * 2
    blocks = sum(cdiv(n, G) * b for n, b in BUCKETS(K, blk, maxlen))
    per_req = G * blocks * page
    conc = AVAIL / per_req
    return int(conc * maxlen), conc, per_req

def max_ctx(K):
    lo, hi = 1024, 1 << 21
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pool(K, mid)[1] >= 1.0: lo = mid
        else: hi = mid - 1
    return lo

print(f"AVAIL={AVAIL/1024**3:.2f} GiB  G={G}\n")
print(f"{'K':>2} {'block':>6} {'pool@131072':>12} {'conc':>6} {'max ctx (1.0x)':>15}")
for K in (4, 5, 6, 7):
    tok, conc, _ = pool(K, 131072)
    print(f"{K:>2} {BLOCK[K]:>6} {tok:>12,} {conc:>6.2f} {max_ctx(K):>15,}")
print(f"\nCALIBRATION  K=4 predicted {pool(4,131072)[0]:,} vs measured 231,602")
