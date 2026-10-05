#!/usr/bin/env python3
"""val_resume -- the same long prompts answered COLD (full prefill) and RESUMED (prefix-cache hit).

reaskbench stops at the first divergent token; this measures the size of the difference over the
whole answer. A ~25k-token registry is shared by 8 questions:
  cold : every question under its own cache_salt -> a full prefill each time
  warm : one cache_salt; a primer question (not compared) caches the registry, then every
         question resumes from the cached GDN state at the last block boundary
Greedy, batch 1, top-5 logprobs. Writes <out>-cold.json and <out>-warm.json in spec_ab format:
  errmargin.py NAME <out>-cold.json <out>-warm.json
At fp32 GDN state the two are expected to be bit-identical; with an fp16 state the resume starts
from the rounded checkpoint while a cold prefill carries the state through that boundary inside
the scan, so a small difference is expected -- this measures how small.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # mt_lazy_gate / val_long live beside this file
from mt_lazy_gate import registry  # noqa: E402
from val_long import ask, line, norm  # noqa: E402

QS = [
    "Summarise what kind of information this registry holds, in three sentences.",
    "Copy records 120 to 126 verbatim, one per line, nothing else.",
    "Which teams appear in the registry? List them, then say which one you think owns the most services and why.",
    "Copy records 377 to 383 verbatim, one per line, nothing else.",
    "Describe the record for service number 250 in plain English.",
    "Which regions appear in the registry, and roughly how are services spread across them?",
    "Write a short Python dict literal for records 440 to 443 with keys name, port, team.",
    "What is the highest record number, and which service is it?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:1234")
    ap.add_argument("--model", default="qwen3.8-27b-vllm")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="prefix; writes <out>-cold.json and <out>-warm.json")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--records", type=int, default=500, help="registry size; moves the cached block's parity")
    ap.add_argument("--evict", type=int, default=0, metavar="N",
                    help="after the primer, prefill N unrelated ~35k-token registries so the primer's blocks "
                         "leave the GPU and the warm turns restore from the offload tier (pool ~270k tokens: N>=9)")
    a = ap.parse_args()
    doc, recs = registry(11, a.records)
    by = {r["id"]: r for r in recs}
    t = int(time.time())
    sysm = {"role": "system", "content": "Be exact. Answer only from the registry."}

    def msgs(q):
        return [sysm, {"role": "user", "content": doc + "\n\n" + q}]

    runs = {"cold": [], "warm": []}
    for i, q in enumerate(QS):
        r = ask(a.base, a.model, msgs(q), f"vres-{a.label}-cold{i}-{t}", a.max_tokens)
        r["prompt"] = q
        runs["cold"].append(r)
        print(f"cold {i + 1}/{len(QS)} cached {r['cached_tokens']} gen {len(r['tokens'])}", flush=True)
    warm_salt = f"vres-{a.label}-warm-{t}"
    ask(a.base, a.model, msgs("How many records are there? One number."), warm_salt, 50)  # primer
    if a.evict:
        t0, tot = time.time(), 0
        for j in range(a.evict):
            fdoc, _ = registry(1000 + j, 690)
            f = ask(a.base, a.model, [sysm, {"role": "user", "content": fdoc + "\n\nHow many records? One number."}],
                    f"vres-{a.label}-evict{j}-{t}", 20)
            tot += f["prompt_tokens"]
        print(f"evict fillers: {a.evict} x ~{tot // a.evict} = {tot} tokens in {time.time() - t0:.0f}s", flush=True)
    for i, q in enumerate(QS):
        r = ask(a.base, a.model, msgs(q), warm_salt, a.max_tokens)
        r["prompt"] = q
        runs["warm"].append(r)
        print(f"warm {i + 1}/{len(QS)} cached {r['cached_tokens']} gen {len(r['tokens'])}", flush=True)
    print(f"resume point {runs['warm'][0]['cached_tokens']} tokens", flush=True)
    for k in ("cold", "warm"):
        exp = [line(by[i]) for i in range(120, 127)] + [line(by[i]) for i in range(377, 384)]
        text = norm(" ".join(r["content"] for r in runs[k]))
        score = f"{sum(1 for e in exp if e in text)}/{len(exp)} copied lines"
        json.dump({"label": f"{a.label}-{k}", "model": a.model, "score": score, "results": runs[k]},
                  open(f"{a.out}-{k}.json", "w"))
        print(f"{k}: {score}")
    print(f"wrote {a.out}-cold.json {a.out}-warm.json")


if __name__ == "__main__":
    main()
