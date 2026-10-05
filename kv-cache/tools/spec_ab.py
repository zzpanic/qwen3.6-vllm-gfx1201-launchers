#!/usr/bin/env python3
"""spec_ab -- greedy output with speculative decoding on vs off, token by token.

WHY. Serial decode (M=1 rows per step) and speculative verify (M=1+k rows) can take different
GEMM / GDN / attention arithmetic routes; magiccodingman/vllm-radiance#11 measured top-1
agreement of 98.38% between them on its stack before aligning the routes. This measures it here.

HOW. The same prompts, greedy (temperature 0), batch size 1, a fresh cache_salt per run (no
prefix-cache crosstalk between runs), top-5 logprobs per generated token. --compare reports, per
prompt, whether the outputs are token-identical and, where they are not, the first differing
position and the logprob margin between the two choices AS EACH RUN SAW IT -- a margin near 0 is
a near-tie decided by arithmetic noise, not a disagreement about the answer.

USAGE
  spec_ab.py --model qwen3.8-27b-vllm   --label spec-a --out /tmp/ab-spec-a.json
  spec_ab.py --model qwen3.8-27b-nospec --label nospec --out /tmp/ab-nospec.json
  spec_ab.py --compare /tmp/ab-spec-a.json /tmp/ab-nospec.json
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in three sentences why the sky is blue.",
    "What is 37 * 43? Show the multiplication step by step.",
    "Write a haiku about a lighthouse in winter.",
    "Convert this to JSON with keys name, age, city: Alice is 31 and lives in Lyon.",
    "List five prime numbers between 50 and 100 and explain how you checked each.",
    "Write a bash one-liner that counts the lines in every .py file under the current directory.",
    "Summarise the plot of Romeo and Juliet in four sentences.",
    "A train leaves at 09:40 and arrives at 13:15. How long is the journey? Explain.",
    "Write a SQL query that returns the top three customers by total order value.",
    "Describe the difference between TCP and UDP for a beginner.",
    "Write a short story opening, two paragraphs, about a robot learning to garden.",
    "Is 2027 a prime number? Reason it out.",
    "Write a Rust function that reverses a string slice and returns a String.",
    "Give a recipe for a simple tomato soup as a numbered list.",
    "Explain what a hash table is and its average lookup complexity.",
    "Translate into French: The meeting has been moved to Thursday afternoon.",
    "Write a regular expression that matches an ISO 8601 date like 2026-10-05, and explain it.",
    "What are the main causes of inflation? Answer in a short paragraph.",
    "Write a JavaScript function debounce(fn, ms) with a short explanation.",
    "If a rectangle has perimeter 30 and area 56, what are its sides? Solve it.",
    "Write a limerick about a cat who learns to code.",
    "Explain the Monty Hall problem and why switching is better.",
    "Produce a YAML config with a server block (host, port) and a list of three users.",
]


def ask(base, model, prompt, salt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "top_p": 1.0, "top_k": -1, "seed": 1234, "max_tokens": max_tokens, "cache_salt": salt,
            "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    ch = d["choices"][0]
    toks = [{"t": e["token"], "lp": e["logprob"],
             "top": {x["token"]: x["logprob"] for x in (e.get("top_logprobs") or [])}}
            for e in ((ch.get("logprobs") or {}).get("content") or [])]
    m = ch["message"]
    return {"reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "content": m.get("content") or "", "finish": ch.get("finish_reason"), "tokens": toks}


def run(a):
    salt = f"specab-{a.label}-{int(time.time())}"
    out = []
    for i, p in enumerate(PROMPTS):
        t0 = time.time()
        r = ask(a.base, a.model, p, salt, a.max_tokens)
        r["prompt"] = p
        out.append(r)
        print(f"{i + 1:2d}/{len(PROMPTS)} {time.time() - t0:5.1f}s {len(r['tokens']):4d} tok  "
              f"{(r['content'] or r['reasoning']).strip()[:60]!r}", flush=True)
    json.dump({"label": a.label, "model": a.model, "salt": salt, "results": out}, open(a.out, "w"))
    print(f"wrote {a.out}")


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    same, firsts, margins, rows = 0, [], [], []
    for ra, rb in zip(A["results"], B["results"]):
        ta, tb = ra["tokens"], rb["tokens"]
        n = min(len(ta), len(tb))
        k = next((i for i in range(n) if ta[i]["t"] != tb[i]["t"]), None)
        if k is None and len(ta) == len(tb):
            same += 1
            rows.append("=")
            continue
        if k is None:
            k = n
            rows.append(f"len@{k}")
            firsts.append(k)
            continue
        a_tok, b_tok = ta[k]["t"], tb[k]["t"]
        ma = ta[k]["lp"] - ta[k]["top"].get(b_tok, float("-inf"))   # A's view of its choice over B's
        mb = tb[k]["lp"] - tb[k]["top"].get(a_tok, float("-inf"))
        m = max(ma, mb)
        margins.append(m)
        firsts.append(k)
        rows.append(f"@{k}({m:.3f})")
    print(f"{A['label']} vs {B['label']}: {same}/{len(A['results'])} prompts token-identical")
    print("per prompt (= identical, @pos(margin nats)): " + " ".join(rows))
    if firsts:
        print(f"first divergence position: median {statistics.median(firsts):.0f}, min {min(firsts)}, max {max(firsts)}")
    if margins:
        fin = [m for m in margins if m != float("inf")]
        tie = sum(1 for m in fin if m < 0.1)
        print(f"margin at divergence (larger of the two views): median {statistics.median(fin):.3f} nats; "
              f"{tie}/{len(fin)} under 0.1 nats (near-ties)")
    return 0 if same == len(A["results"]) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:1234")
    ap.add_argument("--model", default="qwen3.8-27b-vllm")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    if not a.out:
        ap.error("--out is required")
    run(a)
