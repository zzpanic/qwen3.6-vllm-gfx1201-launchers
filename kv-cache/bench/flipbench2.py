#!/usr/bin/env python3
"""flipbench2.py -- N concurrent agents, each with a long prompt, 2 rounds.

Tests whether the disk (L3) tier pays off with more parallel agents + the fs
fanout. For each N in [4, 6, 8]:
  round 1: N agents prefill (recompute) -- they fill GPU -> RAM -> disk as they
          evict each other (N x 150k; GPU 228k holds ~1.5, RAM 462k holds ~3,
          the rest spill to disk).
  round 2: N agents re-query (hit) -- each hits whatever tier its prefix is in.
Report per N: the round-2 TTFT distribution, the /metrics delta (CPU_to_GPU
loads = the L2/L3 loads, GPU_to_CPU stores = the evictions), and the per-request
TTFTs. Round 1 = the recompute baseline.

Stdlib-only (urllib + concurrent.futures).

Example:
  python3 flipbench2.py --url http://127.0.0.1:5804 --model qwen3.8-27b-vllm \
      --agents 4,6,8 --prefix-tokens 150000
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_URL = "http://127.0.0.1:5804/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-27b-vllm"
GPU_CAP = 228737
RAM_CAP = 462894

NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
         "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau",
         "phi", "chi", "psi", "omega"]
KW = ["def", "for", "while", "return", "if", "else", "import", "class", "lambda",
      "yield", "with", "try", "except", "finally", "async", "await", "match",
      "assert", "pass", "elif"]


def gen_prompt(idx, target_chars):
    rng = random.Random(0xBEEF + idx * 7919)
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
            ln = "    # marker %02d chunk %06d " % (idx, n) + "tok " * 8
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
                if d.get("content") or d.get("reasoning") or (
                        ch and (ch[0] or {}).get("finish_reason")):
                    ttft = time.perf_counter() - t0
            u = obj.get("usage")
            if u:
                usage = u
    t1 = time.perf_counter()
    if ttft is None:
        ttft = t1 - t0
    return {"ttft": ttft, "wall": t1 - t0,
            "prompt_tokens": usage.get("prompt_tokens", 0)}


def metrics_url(args):
    from urllib.parse import urlparse
    u = urlparse(args.url)
    return u.scheme + "://" + u.netloc + "/metrics"


def mval(path, prefix, tt):
    for ln in open(path):
        if ln.startswith(prefix + "{") and ('transfer_type="%s"' % tt) in ln:
            return float(ln.split()[-1])
    return None


def metrics_snapshot(args, path):
    req = urllib.request.Request(metrics_url(args))
    d = urllib.request.urlopen(req, timeout=30).read().decode()
    open(path, "w").write(d)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--agents", default="4,6,8")
    ap.add_argument("--prefix-tokens", type=int, default=150000)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    chars = args.prefix_tokens * 2.14
    agent_list = [int(x) for x in args.agents.split(",")]
    print("flipbench2: agents=%s prefix~%dtok (%.1fk chars) model=%s"
          % (agent_list, args.prefix_tokens, chars / 1000, args.model), flush=True)
    print("geometry: GPU=%s  RAM=%s  (1 agent=%dk, 2=%dk>GPU, 3=%dk<RAM, "
          "4=%dk>RAM->disk)" % (GPU_CAP, RAM_CAP, args.prefix_tokens // 1000,
                                 2 * args.prefix_tokens // 1000,
                                 3 * args.prefix_tokens // 1000,
                                 4 * args.prefix_tokens // 1000), flush=True)

    all_results = {}
    for N in agent_list:
        print("\n" + "=" * 70)
        print("N=%d agents" % N, flush=True)
        print("=" * 70, flush=True)
        base = "/tmp/opencode/m_base_n%d.txt" % N
        after = "/tmp/opencode/m_after_n%d.txt" % N
        metrics_snapshot(args, base)

        # N fresh prompts
        prompts = [gen_prompt(i, chars) for i in range(N)]
        print("generated %d x ~%dk prompts" % (N, args.prefix_tokens // 1000),
              flush=True)

        def job(i):
            return one_request(args, args.model, prompts[i], args.max_tokens)

        # round 1: N concurrent prefills (recompute)
        print("\n-- round 1 (recompute, N concurrent) --", flush=True)
        t_r1 = time.time()
        with ThreadPoolExecutor(max_workers=N) as ex:
            r1 = list(ex.map(job, range(N)))
        for i, r in enumerate(r1):
            print("  r1 agent %d ttft=%8.2fs" % (i, r["ttft"]), flush=True)
        r1_ttfts = sorted(r["ttft"] for r in r1)

        # round 2: N concurrent re-queries (hit)
        print("\n-- round 2 (re-query / hit, N concurrent) --", flush=True)
        with ThreadPoolExecutor(max_workers=N) as ex:
            r2 = list(ex.map(job, range(N)))
        for i, r in enumerate(r2):
            print("  r2 agent %d ttft=%8.2fs" % (i, r["ttft"]), flush=True)
        r2_ttfts = sorted(r["ttft"] for r in r2)
        metrics_snapshot(args, after)

        # /metrics delta
        cb = mval(after, "vllm:kv_offload_total_bytes_total", "CPU_to_GPU")
        tb = mval(base, "vllm:kv_offload_total_bytes_total", "CPU_to_GPU")
        ct = mval(after, "vllm:kv_offload_total_time_total", "CPU_to_GPU")
        tt = mval(base, "vllm:kv_offload_total_time_total", "CPU_to_GPU")
        cc = mval(after, "vllm:kv_offload_size_count", "CPU_to_GPU")
        tc = mval(base, "vllm:kv_offload_size_count", "CPU_to_GPU")
        d_bytes = (cb - tb) if (cb is not None and tb is not None) else None
        d_time = (ct - tt) if (ct is not None and tt is not None) else None
        d_cnt = (cc - tc) if (cc is not None and tc is not None) else None
        s_bytes = mval(after, "vllm:kv_offload_total_bytes_total", "GPU_to_CPU") \
            - mval(base, "vllm:kv_offload_total_bytes_total", "GPU_to_CPU")

        def med(v):
            v = sorted(v)
            return v[len(v) // 2] if v else 0

        print("\n-- summary N=%d --" % N, flush=True)
        print("  recompute (r1) median: %.2fs  (min %.2f max %.2f)"
              % (med(r1_ttfts), r1_ttfts[0], r1_ttfts[-1]), flush=True)
        print("  hit (r2) median: %.2fs  (min %.2f max %.2f)"
              % (med(r2_ttfts), r2_ttfts[0], r2_ttfts[-1]), flush=True)
        if d_bytes is not None:
            bw = (d_bytes / 1e9 / d_time) if d_time else 0
            print("  RAM->GPU loads: %.2f GB over %.3fs across %d ops "
                  "(%.2f GB/s)" % (d_bytes / 1e9, d_time, d_cnt, bw), flush=True)
        if s_bytes is not None:
            print("  GPU->RAM stores: %.2f GB" % (s_bytes / 1e9), flush=True)

        all_results[N] = {
            "r1_ttft": r1_ttfts, "r2_ttft": r2_ttfts,
            "load_bytes": d_bytes, "load_time": d_time, "load_cnt": d_cnt,
            "store_bytes": s_bytes,
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "results": all_results,
            }, f, indent=2, default=str)
        print("\n  wrote %s" % args.out)


if __name__ == "__main__":
    main()
