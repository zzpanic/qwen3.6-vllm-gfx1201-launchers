#!/usr/bin/env python3
"""quality_ab -- output quality WITHOUT logprobs, for changes that only act when no logprobs are asked.

The int2 target verify head (FAST_DRAFT + RADIANCE_VERIFY_HEAD, PR #9 global top-256) falls back to the
exact bf16 head whenever a request asks for logprobs -- so spec_ab / val_long, which do, cannot see it.
This asks for none. Greedy, batch 1, a fresh cache_salt per run, three sets:

  open   -- spec_ab's 24 open prompts: compared as TEXT between two runs (identical / first divergence)
  arith  -- 40 generated multi-step arithmetic questions with exact integer answers
  lookup -- 30 questions over a 300-record service registry (port / team / region), exact answers

  quality_ab.py --label fast  --out /tmp/q-fast.json        # verify head armed (default serve)
  quality_ab.py --label exact --out /tmp/q-exact.json       # RADIANCE_VERIFY_HEAD=0
  quality_ab.py --compare /tmp/q-exact.json /tmp/q-fast.json
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt_lazy_gate import registry  # noqa: E402
from spec_ab import PROMPTS  # noqa: E402


def ask(base, model, content, salt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
            "top_p": 1.0, "top_k": -1, "seed": 1234, "max_tokens": max_tokens, "cache_salt": salt,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    m = d["choices"][0]["message"]
    return (m.get("reasoning") or m.get("reasoning_content") or ""), (m.get("content") or ""), \
        d["choices"][0]["finish_reason"]


def arith_set():
    r = random.Random(77)
    out = []
    for _ in range(40):
        a, b, c, d = r.randint(12, 99), r.randint(12, 99), r.randint(3, 19), r.randint(100, 999)
        q = (f"Compute ({a} * {b}) + ({d} - {c} * {c}). Give only the final integer on the last line, "
             f"as ANSWER: <number>.")
        out.append((q, str(a * b + (d - c * c))))
    return out


def lookup_set():
    doc, recs = registry(23, 300)
    r = random.Random(5)
    out = []
    for _ in range(30):
        x = r.choice(recs)
        k = r.choice(["port", "team", "region"])
        q = (doc + f"\n\nWhich {k} does service {x['name']} have? Give only the value on the last line, "
             f"as ANSWER: <value>.")
        out.append((q, str(x[k])))
    return out


def got(text, want):
    m = re.findall(r"ANSWER:\s*([^\s*`]+)", text)
    return bool(m) and m[-1].strip(".") == want


def run(a):
    t = int(time.time())
    res = {"label": a.label, "open": [], "arith": [], "lookup": []}
    for i, p in enumerate(PROMPTS):
        rs, ct, fin = ask(a.base, a.model, p, f"q-{a.label}-open{i}-{t}", 300)
        res["open"].append({"prompt": p, "reasoning": rs, "content": ct, "finish": fin})
    for name, items, mt in (("arith", arith_set(), 600), ("lookup", lookup_set(), 400)):
        for i, (q, want) in enumerate(items):
            rs, ct, fin = ask(a.base, a.model, q, f"q-{a.label}-{name}{i}-{t}", mt)
            res[name].append({"want": want, "content": ct, "ok": got(ct, want), "finish": fin})
        ok = sum(x["ok"] for x in res[name])
        print(f"{a.label} {name}: {ok}/{len(items)} correct", flush=True)
    json.dump(res, open(a.out, "w"))
    print(f"wrote {a.out}")


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    same = 0
    firsts = []
    for x, y in zip(A["open"], B["open"]):
        ta, tb = x["reasoning"] + "\x00" + x["content"], y["reasoning"] + "\x00" + y["content"]
        if ta == tb:
            same += 1
        else:
            firsts.append(next(i for i in range(min(len(ta), len(tb)) + 1)
                               if i == min(len(ta), len(tb)) or ta[i] != tb[i]))
    print(f"open set: {same}/{len(A['open'])} texts identical ({A['label']} vs {B['label']})"
          + (f"; first divergence at char {sorted(firsts)}" if firsts else ""))
    for k in ("arith", "lookup"):
        ca, cb = sum(x["ok"] for x in A[k]), sum(x["ok"] for x in B[k])
        both = sum(1 for x, y in zip(A[k], B[k]) if x["ok"] != y["ok"])
        print(f"{k}: {A['label']} {ca}/{len(A[k])}  {B['label']} {cb}/{len(B[k])}  (questions that disagree: {both})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:1234")
    ap.add_argument("--model", default="qwen3.8-27b-vllm")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        if not a.out:
            ap.error("--out required")
        run(a)
