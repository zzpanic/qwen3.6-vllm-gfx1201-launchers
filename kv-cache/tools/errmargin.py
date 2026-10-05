#!/usr/bin/env python3
"""errmargin -- how far apart are two greedy runs of the same prompts (spec_ab.py / val_long.py format)?

Per pair of runs:
  * prompts token-identical
  * per-token top-1 agreement (tokens compared up to and including each first divergence)
  * |delta logprob| of the chosen token while the two runs still agree (the numeric drift that has
    NOT yet flipped a token): mean, p99, max -- in nats and in 0.125-nat bf16 logprob steps
  * margin at each divergence (larger of the two runs' views), in 0.125-nat steps: a flip at 1-2
    steps is a near-tie decided by rounding, a flip at many steps is a real disagreement
  * answer score, when the runs carry one (val_long.py)
"""
import json
import statistics
import sys
from collections import Counter

STEP = 0.125


def load(p):
    return json.load(open(p))


def pair(pa, pb):
    A, B = load(pa), load(pb)
    same = compared = div = 0
    deltas, steps = [], []
    for ra, rb in zip(A["results"], B["results"]):
        ta, tb = ra["tokens"], rb["tokens"]
        n = min(len(ta), len(tb))
        k = next((j for j in range(n) if ta[j]["t"] != tb[j]["t"]), None)
        upto = n if k is None else k
        deltas += [abs(ta[j]["lp"] - tb[j]["lp"]) for j in range(upto)]
        if k is None:
            compared += n
            same += len(ta) == len(tb)
            continue
        compared += k + 1
        div += 1
        m = max(ta[k]["lp"] - ta[k]["top"].get(tb[k]["t"], -99), tb[k]["lp"] - tb[k]["top"].get(ta[k]["t"], -99))
        steps.append(round(m / STEP))
    ds = sorted(deltas)
    out = {
        "a": A.get("label"), "b": B.get("label"), "prompts": len(A["results"]), "identical": same,
        "compared": compared, "divergences": div,
        "agreement_pct": 100 * (1 - div / compared) if compared else None,
        "dlp_mean": statistics.fmean(ds) if ds else 0.0,
        "dlp_p99": ds[int(0.99 * (len(ds) - 1))] if ds else 0.0,
        "dlp_max": ds[-1] if ds else 0.0,
        "dlp_nonzero_pct": 100 * sum(1 for d in ds if d > 0) / len(ds) if ds else 0.0,
        "margin_steps": dict(sorted(Counter(steps).items())),
        "score_a": A.get("score"), "score_b": B.get("score"),
    }
    return out


def show(o, name):
    print(f"\n{name}: {o['a']} vs {o['b']}")
    print(f"  prompts token-identical {o['identical']}/{o['prompts']}; per-token top-1 agreement "
          f"{o['agreement_pct']:.2f}% ({o['divergences']} flips in {o['compared']} tokens)")
    print(f"  |dlogprob| on agreeing tokens: mean {o['dlp_mean']:.4f}, p99 {o['dlp_p99']:.3f}, max "
          f"{o['dlp_max']:.3f} nats (max = {o['dlp_max'] / STEP:.1f} bf16 steps); "
          f"{o['dlp_nonzero_pct']:.1f}% of tokens differ at all")
    print(f"  margin at each flip, in 0.125-nat steps -> count: {o['margin_steps']}")
    if o["score_a"] is not None:
        print(f"  answer score: {o['score_a']} vs {o['score_b']}")


if __name__ == "__main__":
    # usage: errmargin.py NAME A.json B.json [NAME2 C.json D.json ...]
    args = sys.argv[1:]
    res = []
    while args:
        name, a, b = args[:3]
        args = args[3:]
        o = pair(a, b)
        o["name"] = name
        show(o, name)
        res.append(o)
