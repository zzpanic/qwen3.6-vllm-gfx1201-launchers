#!/usr/bin/env python3
"""Representative census workload.

The 2026-08-14 lesson (make_shapelog_hist.py docstring): a single cold synthetic
prompt makes small-M buckets look unreachable, because warm multi-turn chat is what
produces short prefill tail chunks. So this drives THREE shape families:
  1. deep single-shot prefills   -> MBIG
  2. warm multi-turn chat        -> small/medium tail chunks (the b=128/512 class)
  3. long decodes                -> the DFlash2 drafter M
Timings here are meaningless (instrumented kernel); only the shape histogram counts.
"""
import json, urllib.request, random, sys

BASE = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen3.8-27b-vllm"
random.seed(7)
WORDS = open("/usr/share/dict/words").read().split() if __import__("os").path.exists("/usr/share/dict/words") else None

def filler(n_words, nonce):
    if WORDS:
        return f"[{nonce}] " + " ".join(random.choice(WORDS) for _ in range(n_words))
    return f"[{nonce}] " + " ".join(f"w{random.randint(1000,9999)}" for _ in range(n_words))

def post(messages, max_tokens):
    req = urllib.request.Request(BASE, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"model": MODEL, "messages": messages,
                         "max_tokens": max_tokens}).encode())
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d["choices"][0]["message"], d["usage"]

# --- 1. deep single-shot prefills (MBIG) ---
for i, words in enumerate([3000, 12000, 40000, 75000]):
    m, u = post([{"role": "user", "content":
        filler(words, f"deep{i}") + "\n\nSummarise the above in one sentence."}], 128)
    print(f"deep{i}: prompt={u['prompt_tokens']} completion={u['completion_tokens']}", flush=True)

# --- 2. warm multi-turn chat (short tail chunks) ---
msgs = [{"role": "system", "content": "You are a concise assistant."}]
for t in range(14):
    msgs.append({"role": "user", "content":
        filler(random.choice([40, 120, 300, 700, 1500]), f"turn{t}") +
        f"\n\nIn one short sentence, what is {t}*7?"})
    m, u = post(msgs, 96)
    msgs.append({"role": "assistant", "content": m.get("content") or "ok"})
    print(f"turn{t}: prompt={u['prompt_tokens']} completion={u['completion_tokens']}", flush=True)

# --- 3. long decode (drafter M) ---
for i in range(3):
    m, u = post([{"role": "user", "content":
        f"[longdec{i}] Write 600 words about heat transfer in dense packed beds."}], 900)
    print(f"longdec{i}: prompt={u['prompt_tokens']} completion={u['completion_tokens']}", flush=True)
print("workload done")
