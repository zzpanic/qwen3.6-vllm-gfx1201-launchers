#!/usr/bin/env python3
"""lenprobe -- regression test for the V2 uniform-decode bug (kv-cache/patches/patch_uniform_decode_guard.py).

Does a prompt whose LAST prefill chunk is exactly 1 + num_speculative_tokens tokens come out as
garbage? Single requests, batch 1, cold (unique cache_salt), greedy. Exit 0 = every length
answered normally (guard working), 1 = at least one 1-2 token reply (guard missing or broken).
Unpatched vLLM 0.27.1 V2 + DFlash x7, block 1,648: 1,656 / 3,304 / 4,952 tokens fail.

For each target length L = k * BLOCK + d, pad a fixed question with filler until the server's
prompt_tokens == L (probing with max_tokens=1), then generate and record the reply. With
block-aligned chunking (align mode), the last chunk is d tokens.
"""
import json
import sys
import time
import urllib.request

import argparse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", default="http://127.0.0.1:1234")
ap.add_argument("--model", default="qwen3.8-27b-vllm")
ap.add_argument("--block", type=int, default=1648,
                help="attention block size; vLLM logs it at boot: 'Setting attention block size to N tokens'")
ap.add_argument("--spec", type=int, default=7, help="num_speculative_tokens (the bad tail is spec + 1)")
ap.add_argument("--out", help="optional JSON of the rows")
A = ap.parse_args()
BASE, MODEL, BLOCK = A.base, A.model, A.block
KS = [1, 2, 3]
DS = [A.spec - 1, A.spec, A.spec + 1, A.spec + 2, A.spec + 3]


def chat(text, salt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "temperature": 0.0,
            "top_p": 1.0, "top_k": -1, "seed": 1234, "max_tokens": max_tokens, "cache_salt": salt,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    m, u = d["choices"][0]["message"], d["usage"]
    return (u["prompt_tokens"], u["completion_tokens"], (m.get("reasoning") or m.get("reasoning_content") or ""),
            m.get("content") or "", d["choices"][0]["finish_reason"],
            (u.get("prompt_tokens_details") or {}).get("cached_tokens"))


QUESTION = "\n\nIgnore the filler above. In two sentences, what is the capital of France and why is it famous?"


def build(n_fill):
    return ("filler " * n_fill) + QUESTION


def fit(target, t):
    # "filler " is one token after the first; search on the filler count
    n = max(1, target - 40)
    for _ in range(12):
        p = chat(build(n), f"lenfit-{target}-{n}-{t}", 1)[0]
        if p == target:
            return n
        n += target - p
    raise RuntimeError(f"could not hit {target}")


t = int(time.time())
rows = []
for k in KS:
    for dd in DS:
        L = k * BLOCK + dd
        n = fit(L, t)
        p, gen, rs, ct, fin, cached = chat(build(n), f"lenprobe-{L}-{t}", 120)
        bad = gen <= 3 or not (rs + ct).strip()
        rows.append({"len": L, "last_chunk": dd, "gen": gen, "finish": fin, "cached": cached, "bad": bad,
                     "text": (rs + ct)[:60]})
        print(f"L={L:6d} (k={k}, last chunk {dd:2d}) gen {gen:4d} {fin:6s} cached {cached} "
              f"{'*** BAD ***' if bad else 'ok':12s} {(rs + ct)[:50]!r}", flush=True)
if A.out:
    json.dump(rows, open(A.out, "w"), indent=1)
bad = [r for r in rows if r["bad"]]
print(f"{len(bad)} BAD of {len(rows)}" + (" -- uniform-decode guard missing or broken" if bad else " -- guard working"))
sys.exit(1 if bad else 0)
