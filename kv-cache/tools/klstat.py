"""Approximate per-token KL(ref || test) from top-5 logprobs, over positions where both runs still
agree on the prefix (same context). Tokens missing from one side's top-5 get that side's 5th
logprob (an upper bound on its mass), so this slightly UNDER-states KL at very peaked positions
and is a fair comparison across pairs, which is what it is for."""
import json, math, statistics, sys

def kl_at(ref, tst):
    fr, ft = min(ref["top"].values()), min(tst["top"].values())
    keys = set(ref["top"]) | set(tst["top"])
    pr = {k: math.exp(ref["top"].get(k, fr)) for k in keys}
    pt = {k: math.exp(tst["top"].get(k, ft)) for k in keys}
    zr, zt = sum(pr.values()), sum(pt.values())
    return sum(pr[k] / zr * math.log((pr[k] / zr) / (pt[k] / zt)) for k in keys)

def pair(name, ptest, pref):
    T, R = json.load(open(ptest))["results"], json.load(open(pref))["results"]
    kls = []
    for rt, rr in zip(T, R):
        tt, tr = rt["tokens"], rr["tokens"]
        n = min(len(tt), len(tr))
        k = next((j for j in range(n) if tt[j]["t"] != tr[j]["t"]), n)
        kls += [kl_at(tr[j], tt[j]) for j in range(min(k + 1, n))]   # include the divergence position
    s = sorted(kls)
    print(f"{name:44} n={len(s):5d}  mean KL {statistics.fmean(s):.5f}  p99 {s[int(.99*(len(s)-1))]:.4f}  max {s[-1]:.3f} nats  "
          f"positions >0.05: {sum(x > .05 for x in s)}")

# usage: klstat.py NAME TEST.json REF.json [NAME2 ...] -- KL(ref || test)
a = sys.argv[1:]
while a:
    pair(*a[:3]); a = a[3:]
