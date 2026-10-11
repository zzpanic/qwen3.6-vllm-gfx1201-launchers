#!/usr/bin/env python3
"""JIT warm-up: run the serving paths once, right after /health, so Triton kernels compile (cold cache) or load
(warm cache) before the first real request instead of during it.

WHY: vLLM's /health answers before any forward pass. Several Triton kernels are first called by real traffic --
the speculative-decoding sampler (_rejection_kernel, _resample_kernel, _compute_local_logits_stats_kernel), the
drafter's input prep and int2 head, the W4A16 split-K / skinny GEMMs at batch sizes above 1 -- and each first call
costs a compile on a cold cache (seconds) or a cache load on a warm one (~1 s for the set). radiance's jit_monitor
then logs "Triton kernel JIT compilation during inference". This script pays that once, at boot.

WHAT: a few tiny requests, stdlib only. Greedy and sampled (temperature 0.7, top_p 0.95, top_k 20), one prompt
just under a canonical grid chunk (prefill-sized GEMMs, and short enough that nothing reaches the offload tiers),
then 2, 3 and MAXSEQS concurrent sampled requests (decode batch shapes), then a tiny image if the server takes one.
Every step is guarded: a failure prints one line and the next step runs. It changes no state a real request sees
except the prefix cache (short warm-up prompts, evicted like any other).
usage: radiance_jit_warmup.py PORT [MAXSEQS]
"""
import base64
import json
import random
import struct
import sys
import threading
import time
import urllib.request
import zlib

PORT = int(sys.argv[1])
MAXSEQS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
U = f"http://127.0.0.1:{PORT}"
SAMPLED = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}
GREEDY = {"temperature": 0}
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def post(body, timeout=600):
    r = urllib.request.Request(U + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def model():
    return json.load(urllib.request.urlopen(U + "/v1/models", timeout=30))["data"][0]["id"]


def words(seed, n):
    r = random.Random(seed)
    return " ".join(f"{r.choice('bcdfghklmnprstvz')}{r.choice('aeiou')}{r.randrange(10, 99)}" for _ in range(n))


def req(name, content, sampling, max_tokens=48):
    return {"model": name, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            **sampling, **NO_THINK}


def png(w, h):
    raw = b"".join(b"\x00" + bytes((x * 5 + y * 3) % 256 for x in range(w * 3)) for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))


def concurrent(bodies):
    errs = []

    def one(b):
        try:
            post(b)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ths = [threading.Thread(target=one, args=(b,)) for b in bodies]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise errs[0]


def main():
    t0 = time.time()
    try:
        name = model()
    except Exception as e:  # noqa: BLE001
        print(f"[radiance] jit warm-up skipped: no model list ({e!r})", flush=True)
        return
    salt = str(time.time_ns())
    steps = [
        ("greedy", lambda: post(req(name, f"{salt} Count from one to five.", GREEDY))),
        ("sampled", lambda: post(req(name, f"{salt} Name three colours.", SAMPLED))),
        # ~2,500 tokens: prefill-sized GEMMs, below one canonical grid chunk (3,520) so no tier store
        ("prefill", lambda: post(req(name, words(salt, 800) + "\nRepeat the first word.", SAMPLED, 8))),
    ]
    for n in sorted({2, 3, MAXSEQS} - {0, 1}):
        if n <= MAXSEQS:
            steps.append((f"concurrent {n}", lambda n=n: concurrent(
                [req(name, f"{salt} {i} Write a haiku about the number {i}.", SAMPLED) for i in range(n)])))
    img = "data:image/png;base64," + base64.b64encode(png(64, 64)).decode()
    steps.append(("image", lambda: post({"model": name, "max_tokens": 8, **SAMPLED, **NO_THINK, "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                     {"type": "text", "text": "What colour dominates?"}]}]})))
    done, failed = [], []
    for label, fn in steps:
        try:
            fn()
            done.append(label)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{label} ({str(e)[:80]})")
    print(f"[radiance] jit warm-up: {len(done)} of {len(steps)} steps in {time.time() - t0:.1f} s "
          f"({', '.join(done)}){'; failed: ' + ', '.join(failed) if failed else ''}", flush=True)


if __name__ == "__main__":
    main()
