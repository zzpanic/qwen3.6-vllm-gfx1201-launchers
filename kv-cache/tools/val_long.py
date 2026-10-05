#!/usr/bin/env python3
"""val_long -- long-context greedy retrieval with checkable answers, in spec_ab.py's output format.

A deterministic ~23k-token service registry (mt_lazy_gate's generator, more records), then six
questions whose answers can be scored exactly: verbatim copies of record ranges, filtered lists
and single facts deep in the context. Greedy, batch 1, top-5 logprobs, one fresh cache_salt per
run. The six requests share the registry prefix, so requests 2-6 resume from cached GDN state at
block boundaries -- the path an fp16 state or the lazy stash would perturb.

  val_long.py --label P-a --out /tmp/vl-P-a.json
  errmargin.py NAME /tmp/vl-P-a.json /tmp/vl-R-a.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # mt_lazy_gate / val_long live beside this file
from mt_lazy_gate import registry  # noqa: E402


def line(r):
    return (f"Record {r['id']:03d}: service {r['name']} listens on port {r['port']}, is owned by "
            f"team {r['team']}, runs in region {r['region']}, and has been in its current state "
            f"since day {r['day']}.")


def tasks(recs):
    by = {r["id"]: r for r in recs}
    out = []
    for lo in (37, 251, 468):
        ids = range(lo, lo + 8)
        out.append((f"Copy records {lo:03d} to {lo + 7:03d} verbatim, one per line, nothing else.",
                    [line(by[i]) for i in ids]))
    team, region = "Heron", "east-2"
    hits = [r for r in recs if r["team"] == team and r["region"] == region]
    out.append((f"List the name and port of every service owned by team {team} that runs in region "
                f"{region}, one per line as NAME PORT.", [f"{r['name']} {r['port']}" for r in hits]))
    deep = [by[i] for i in (12, 199, 333, 487)]
    out.append(("Answer each on its own line, as NAME PORT: which port does each of these services "
                "listen on? " + ", ".join(r["name"] for r in deep) + ".",
                [f"{r['name']} {r['port']}" for r in deep]))
    out.append((f"Which day has {by[402]['name']} been in its current state since, and which team "
                f"owns it? One sentence.", [str(by[402]["day"]), by[402]["team"]]))
    return out


def ask(base, model, messages, salt, max_tokens):
    body = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "seed": 1234, "max_tokens": max_tokens, "cache_salt": salt, "logprobs": True,
            "top_logprobs": 5, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    ch = d["choices"][0]
    toks = [{"t": e["token"], "lp": e["logprob"],
             "top": {x["token"]: x["logprob"] for x in (e.get("top_logprobs") or [])}}
            for e in ((ch.get("logprobs") or {}).get("content") or [])]
    m = ch["message"]
    u = d.get("usage") or {}
    return {"reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "content": m.get("content") or "", "finish": ch.get("finish_reason"), "tokens": toks,
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


def norm(s):
    return re.sub(r"[`*|]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:1234")
    ap.add_argument("--model", default="qwen3.8-27b-vllm")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=1200)
    a = ap.parse_args()
    doc, recs = registry(11, a.records)
    salt = f"vlong-{a.label}-{int(time.time())}"
    out, got, want = [], 0, 0
    for i, (q, items) in enumerate(tasks(recs)):
        t0 = time.time()
        msgs = [{"role": "system", "content": "Be exact. Answer only from the registry."},
                {"role": "user", "content": doc + "\n\n" + q}]
        r = ask(a.base, a.model, msgs, salt, a.max_tokens)
        text = norm(r["content"])
        hit = sum(1 for it in items if it in text)
        got += hit
        want += len(items)
        r.update({"prompt": q, "expected": items, "hits": hit})
        out.append(r)
        print(f"{i + 1}/6 {time.time() - t0:5.1f}s prompt {r['prompt_tokens']} cached {r['cached_tokens']} "
              f"gen {len(r['tokens'])} finish {r['finish']} score {hit}/{len(items)}", flush=True)
    score = f"{got}/{want}"
    json.dump({"label": a.label, "model": a.model, "salt": salt, "score": score, "results": out},
              open(a.out, "w"))
    print(f"SCORE {score} -> {a.out}")


if __name__ == "__main__":
    main()
