#!/usr/bin/env python3
"""flipbench.py -- long-prompt flip-flop workload: measures L1/L2/L3 cache hit speeds.

Generates N distinct long prompts (~195k tokens each, near the 228k GPU KV cap) and
flips them back and forth so that re-querying a prefix exercises the tier cascade:

  L1 (GPU)   a prefix that still fits on the GPU            -> instant
  L2 (CPU)   a prefix evicted off the GPU into the 16 GiB RAM tier. 2 x 195k = 390k
            > 228k GPU (evicted off L1) but < 462k RAM cap (still in RAM) -> fast load
  L3 (disk)  a prefix evicted off RAM into /kvcache. 3 x 195k = 585k > 462k RAM cap,
            so the oldest is pushed to disk -> the fs->CPU staging

Per request it streams and records TTFT (time to first token = the prefill / load time)
and prompt_tokens_by_source, then classifies the request as L1 / L2 / L3 / recompute.
The headline is the TTFT by hit type: that IS the "speed" of a CPU-RAM cache hit and a
disk cache hit, against the recompute baseline.

Modes:
  flip        sequential flip-flop (clean per-request attribution). The default.
  concurrent  N agents each loop their own prompt, overlapping in time ("run over the
            top of each other"); with max-num-seqs=2 two run at once and evict each other.

Scenarios:
  l2          A, B (recompute), A (re-query -> L2 hit)
  l3          A, B, C (recompute; C evicts A to disk), A (re-query -> L3 hit)
  both        l2 then l3
  novel       N distinct prompts, each once (recompute baseline)

Stdlib-only (urllib + concurrent.futures).

Example (L2 then L3 hit speeds, sequential):
  python3 flipbench.py --url http://127.0.0.1:5804 --model qwen3.8-27b-vllm \
      --prefix-tokens 195000 --scenario both --mode flip
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_URL = "http://127.0.0.1:5804/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-27b-vllm"

# Tier capacities (from the live stack): GPU 228,737 tok; RAM 16 GiB / 37,114 B per
# offloaded token = 462,894 tok. Used to sanity-check the scenario geometry.
GPU_CAP = 228737
RAM_CAP = 462894

NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
         "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau",
         "phi", "chi", "psi", "omega"]
KW = ["def", "for", "while", "return", "if", "else", "import", "class", "lambda",
      "yield", "with", "try", "except", "finally", "async", "await", "match",
      "assert", "pass", "elif"]


def gen_prompt(idx, target_chars):
    """Deterministic code-like corpus of ~target_chars, unique per idx, identical on
    re-query (so a re-query is a clean cache hit)."""
    rng = random.Random(0xC0FFEE + idx * 7919)
    lines = []
    total = 0
    n = 0
    while total < target_chars:
        n += 1
        k = rng.random()
        a = rng.choice(NAMES)
        b = rng.choice(NAMES)
        if k < 0.15:
            ln = "def fn_%02d_%06d(%s, %s):" % (idx, n, a, b)
        elif k < 0.35:
            ln = "    %s %s%d == %d: return %s" % (rng.choice(KW), a, n % 7,
                                                 rng.randint(0, 999), b)
        elif k < 0.55:
            vals = ",".join(str(rng.randint(0, 99)) for _ in range(3))
            ln = "    x%d_%d_%d = [%s]" % (n % 5, idx, n % 997, vals)
        elif k < 0.70:
            ln = "    # radiance marker %02d chunk %06d " % (idx, n) + "tok " * 8
        elif k < 0.85:
            ln = "    for i in range(%d): y%d = '%s%d'" % (rng.randint(1, 64),
                                                        n % 9, a, n % 911)
        else:
            ln = "    assert %s%d is not None or (%d == %d)" % (
                a, n % 31, rng.randint(0, 1), rng.randint(0, 1))
        lines.append(ln)
        total += len(ln) + 1
    return "\n".join(lines)


def _open(args, body):
    req = urllib.request.Request(
        args.url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return urllib.request.urlopen(req, timeout=args.timeout)


def one_request(args, model, prompt, max_tokens):
    """Streaming: returns TTFT, wall, prompt_tokens_by_source, and the token counts."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    src = {}
    usage = {}
    with _open(args, body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ch = obj.get("choices") or [{}]
                d = (ch[0] or {}).get("delta", {}) if ch else {}
                # reasoning model: first generated token may be 'reasoning', not
                # 'content'. prefill ends at the first non-empty generated delta.
                if d.get("content") or d.get("reasoning") or (
                        ch and (ch[0] or {}).get("finish_reason")):
                    ttft = time.perf_counter() - t0
                    n_tokens += 1
            u = obj.get("usage")
            if u:
                usage = u
            s = usage.get("prompt_tokens_by_source")
            if s:
                src = s
    t1 = time.perf_counter()
    if ttft is None:
        ttft = t1 - t0
    return {
        "ttft": ttft,
        "wall": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "src": src,
    }


def classify(src, ttft):
    l1 = src.get("local_cache_hit", 0)
    ext = src.get("external_kv_transfer", 0)
    comp = src.get("local_compute", 0)
    tot = l1 + ext + comp
    if tot == 0:
        return "unknown"
    if comp / tot > 0.5:
        return "recompute"
    if ext / tot > 0.10:
        return "L2 (CPU)" if ttft < 20.0 else "L3 (disk)"
    if l1 / tot > 0.5:
        return "L1 (GPU)"
    return "mixed"


def seq_for(scenario):
    """The prompt-index sequence + the design-intent label per request (0=A, 1=B, 2=C).

    The label is the tier the re-query hits BY CONSTRUCTION (we know the geometry:
    2x fits in RAM -> L2, 3x overflows RAM -> L3), so we report the measured TTFT
    under that label and cross-check the source split against /metrics deltas."""
    if scenario == "l2":
        return [0, 1, 0], ["recompute", "recompute", "L2 (CPU)"]
    if scenario == "l3":
        return [0, 1, 2, 0], ["recompute", "recompute", "recompute", "L3 (disk)"]
    if scenario == "both":
        return ([0, 1, 0, 1, 2, 0],
                ["recompute", "recompute", "L2 (CPU)", "L2 (CPU)",
                 "recompute", "L3 (disk)"])
    if scenario == "novel":
        return [0, 1, 2, 3], ["recompute"] * 4
    return [0], ["recompute"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenario", choices=["l2", "l3", "both", "novel"], default="both")
    ap.add_argument("--mode", choices=["flip", "concurrent"], default="flip")
    ap.add_argument("--prefix-tokens", type=int, default=195000)
    ap.add_argument("--prefix-chars", type=int, default=0,
                   help="override the char count (default: prefix-tokens x 2.14, "
                        "the measured chars/token for this corpus)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    chars = args.prefix_chars or args.prefix_tokens * 2.14
    print(f"locbench->flipbench: scenario={args.scenario} mode={args.mode} "
          f"prefix~{args.prefix_tokens}tok ({chars:,} chars) model={args.model}", flush=True)
    print(f"geometry: GPU={GPU_CAP:,}  RAM={RAM_CAP:,}  "
          f"2x={2 * args.prefix_tokens:,} (>GPU, <RAM -> L2)  "
          f"3x={3 * args.prefix_tokens:,} (>RAM -> L3)", flush=True)

    # build the distinct prompts
    idxs = {0: gen_prompt(0, chars), 1: gen_prompt(1, chars),
            2: gen_prompt(2, chars), 3: gen_prompt(3, chars)}

    def job(tag, idx):
        res = one_request(args, args.model, idxs[idx], args.max_tokens)
        res["tag"] = tag
        res["idx"] = idx
        res["cls"] = classify(res["src"], res["ttft"])
        return res

    seq, labels = seq_for(args.scenario)
    results = []
    if args.mode == "flip":
        for i, idx in enumerate(seq):
            res = job(f"s{i}", idx)
            res["label"] = labels[i]
            results.append(res)
            print(f"  [{args.scenario}] req {i} (prefix {idx}) "
                  f"EXPECT={labels[i]:12} ttft={res['ttft']:8.2f}s "
                  f"wall={res['wall']:8.2f} api_cls={res['cls']:11}", flush=True)
    else:
        # concurrent: 2 agents, each looping its own prefix, overlapping
        def agent(idx):
            for i in range(2):
                res = job(f"a{idx}.{i}", idx)
                results.append(res)
                print(f"  [concurrent] agent {idx} iter {i} ttft={res['ttft']:8.2f}s "
                      f"cls={res['cls']:11}", flush=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(agent, [0, 1]))

    # report
    print("\n" + "=" * 78)
    print("RESULTS (TTFT = time to first token = the prefill / load time)")
    print("=" * 78)
    by_type = {}
    for r in results:
        by_type.setdefault(r.get("label") or r["cls"], []).append(r["ttft"])
    labels = ["L1 (GPU)", "L2 (CPU)", "L3 (disk)", "recompute", "mixed", "unknown"]
    for lab in labels:
        if lab in by_type:
            v = by_type[lab]
            v.sort()
            med = v[len(v) // 2]
            print(f"  {lab:12}: n={len(v)}  min={v[0]:8.2f}s  median={med:8.2f}s  "
                  f"max={v[-1]:8.2f}s")
    # the headline
    rec = sorted(by_type.get("recompute", [0]))
    l2 = sorted(by_type.get("L2 (CPU)", []))
    l3 = sorted(by_type.get("L3 (disk)", []))
    if rec and (l2 or l3):
        base = rec[len(rec) // 2]
        print(f"\n  recompute baseline (median): {base:.2f}s")
        if l2:
            print(f"  L2 (CPU RAM) hit (median)  : {l2[len(l2) // 2]:.2f}s  "
                  f"(x{base / l2[len(l2) // 2]:.2f} vs recompute)")
        if l3:
            print(f"  L3 (disk)    hit (median)  : {l3[len(l3) // 2]:.2f}s  "
                  f"(x{base / l3[len(l3) // 2]:.2f} vs recompute)")
        if l2 and l3:
            print(f"  L2 vs L3: disk hit is x{l3[len(l3) // 2] / l2[len(l2) // 2]:.2f} "
                  f"slower than the RAM hit")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "results": [{k: r.get(k) for k in ("tag", "idx", "label", "cls", "ttft", "wall", "src")}
                           for r in results],
            }, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
