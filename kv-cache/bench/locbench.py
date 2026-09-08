#!/usr/bin/env python3
"""locbench.py -- a looping code-generation synthetic workload.

Drives an OpenAI-compatible endpoint with a deterministic stream of code-gen
prompts, measures the per-request rate of lines-of-code per second, and prints
a histogram of that rate. Two modes exercise the two sides of the KV cache:

  --mode repeat   the same N prompts are re-sent every round: round 1 is the
                  recompute baseline and rounds 2+ are cache candidates.
  --mode novel    each request is a distinct task + distinct salt (a full
                  miss), so every request exercises the recompute path.

Metrics per request:
  wall      end-to-end request time (s)
  ttft      time to first token (s) -- only measured with --stream
  lines     non-empty lines of code in the completion
  rate      lines / wall  (the histogrammed metric)

The per-round mean wall (and ttft with --stream) is the cache signal: in
repeat mode a tier that serves hits drives rounds 2+ wall time down. Note the
rate is a *generation* metric, so a cache hit (which shortens TTFT, not
generation) shows up more clearly in wall/ttft than in the rate itself -- we
report both.

Stdlib-only (urllib + concurrent.futures).

Example:
  python3 locbench.py \
      --url http://127.0.0.1:8000/v1/chat/completions \
      --model Qwen3.8-27B-MXFP4-mtpfp8 \
      --n 8 --rounds 3 --mode repeat --stream --bucket 20
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen3.8-27B-MXFP4-mtpfp8"

# A realistic, token-heavy pool of code-generation tasks. Deterministic so a
# given (n, rounds, mode, salt) is fully replayable.
TASKS = [
    "Write a Python function that parses a CSV string into a list of dicts, handling quoted fields, embedded commas, and CRLF line endings. Include type coercion for integers and floats and a strict mode that raises on ragged rows.",
    "Write a C function that implements an LRU cache with a fixed-size doubly-linked list and a hash table, O(1) get/put, and an eviction callback invoked with the evicted key and value. Document the memory lifetime.",
    "Write a Rust function that implements a bounded queue on a ring buffer with a sequence counter for the ABA problem, acquire/release ordering comments, and a drain that returns a Vec preserving order.",
    "Write a TypeScript module that implements a promise-based token-bucket rate limiter supporting concurrent permits, exponential backoff on 429, and a metrics callback reporting the current bucket depth.",
    "Write a Python function that performs a topological sort of a directed graph given as adjacency lists, detects cycles, and returns a stable ordering that breaks ties by node name.",
    "Write a Go function that implements a consistent-hash ring with virtual nodes, add/remove of backends, O(log n) lookup, and power-of-two-choices tie-breaking with a load-balance report.",
    "Write a C++ function that implements a wait-free queue on a Treiber node pool with hazard pointers. Document the reclamation protocol and the memory-ordering choices.",
    "Write a Python function that tokenizes a C source file into a flat list of (token, offset, length) tuples, distinguishing preprocessor lines, string and character literals, and line comments.",
    "Write a JavaScript function that implements a Myers diff algorithm returning a minimal edit script for two strings, with a linear-space optimization for large inputs and an op-count summary.",
    "Write a Python function that implements a B-tree of order m with insert, delete, and range scan, keeping the tree balanced with the standard split and merge rules. Include an invariant checker.",
    "Write a Rust function that implements a slab allocator with size classes, bump allocation within slabs, a free list, and a debug mode that poisons freed memory and verifies on drop.",
    "Write a C function that implements a single-producer single-consumer ring-buffer logger with a sequence counter for the ABA problem and a drain that returns a batch of up to n records.",
    "Write a Python function that computes the transitive reduction of a directed graph and reports whether the input was already a DAG, returning the minimal edge set and a cycle witness if any.",
    "Write a Go function that implements a jittered exponential backoff scheduler that respects a global rate limit and supports context cancellation with a final error summary.",
    "Write a Python function that implements an in-memory key-value store with TTL, LRU eviction, and a snapshot that serializes to a stable byte order for byte-exact comparison.",
    "Write a C function that implements a small-object memory arena with region-based allocation, alignment, and a reset that reuses the arena. Document the memory lifetime and reset semantics.",
    "Write a TypeScript function that implements a declarative state machine with guarded transitions, an event log, and a serialized snapshot that round-trips without loss.",
    "Write a Python function that implements a merge-based external sort for files too large to fit in memory, returning the sorted output path and a per-pass I/O accounting table.",
    "Write a Rust function that implements a fixed-capacity stack that overflows into a heap buffer, with a drain that preserves order and a capacity report showing stack vs heap split.",
    "Write a C++ function that implements a thread-safe reference-counted smart pointer with a custom deleter, weak references, and a thread-safety audit comment explaining the lock discipline.",
    "Write a Python function that implements a sliding-window rate limiter over a time series, supporting reset and returning the current rate as a fraction of the limit with a saturation flag.",
    "Write a Go function that implements a work-stealing scheduler with a local deque and a global queue, where a steal takes from the opposite end of a victim's deque and reports steal success.",
    "Write a Python function that implements a minimal Thompson NFA regex matcher (no backtracking) that compiles a pattern and runs it in O(n + m) over an input, reporting the match span.",
    "Write a Rust function that implements a lock-free stack using a tagged pointer for the ABA problem and a bounded batch pop, with memory-ordering comments and a drop test.",
]


def build_prompt(args, round_idx, slot):
    """repeat -> identical prompt across rounds (cache candidate).
    novel   -> distinct task + distinct salt (full miss)."""
    if args.mode == "repeat":
        base, salt, novel = slot, slot, False
    else:
        base = (round_idx * args.n + slot) % len(TASKS)
        salt = round_idx * args.n + slot
        novel = True
    tail = f"\n\nSalt any generated identifiers with the constant {salt}." if novel else ""
    return f"[Task {base:02d}] {TASKS[base]}{tail}"


def count_code_lines(text):
    """Count non-empty lines of code, stripping code fences if present."""
    if not text:
        return 0
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].splitlines()
        # drop a possible language tag on the first line
        if t and t[0].strip() and not t[0].strip().startswith("#"):
            t = t[1:]
    else:
        t = t.splitlines()
    return sum(1 for ln in t if ln.strip())


def _base_body(model, prompt, max_tokens, stream):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _open(args, body):
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=args.timeout)


def one_nonstream(args, model, prompt, max_tokens):
    t0 = time.perf_counter()
    with _open(args, _base_body(model, prompt, max_tokens, False)) as r:
        data = json.loads(r.read())
    wall = time.perf_counter() - t0
    completion = ""
    try:
        completion = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        pass
    ptok = (data.get("usage") or {}).get("prompt_tokens", 0)
    lines = count_code_lines(completion)
    return {
        "wall": wall, "ttft": None, "gen": wall,
        "lines": lines, "rate": lines / wall if wall > 0 else 0.0,
        "prompt_tokens": ptok, "chars": len(completion), "error": None,
    }


def one_stream(args, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    parts = []
    ptok = 0
    with _open(args, _base_body(model, prompt, max_tokens, True)) as r:
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
                ttft = time.perf_counter() - t0
            choices = obj.get("choices") or [{}]
            delta = (choices[0] or {}).get("delta", {}) if choices else {}
            c = delta.get("content")
            if c:
                parts.append(c)
            usage = obj.get("usage") or {}
            if usage.get("prompt_tokens"):
                ptok = usage["prompt_tokens"]
    t1 = time.perf_counter()
    wall = t1 - t0
    if ttft is None:
        ttft = wall
    completion = "".join(parts)
    lines = count_code_lines(completion)
    return {
        "wall": wall, "ttft": ttft, "gen": wall - ttft,
        "lines": lines, "rate": lines / wall if wall > 0 else 0.0,
        "prompt_tokens": ptok, "chars": len(completion), "error": None,
    }


def make_histogram(values, bucket):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        w = bucket if bucket > 0 else 1.0
        return [(lo, lo + w, len(values))]
    if not bucket or bucket <= 0:
        bucket = max((hi - lo) / 12.0, 1e-9)
    bins = {}
    for v in values:
        b = int((v - lo) / bucket)
        bins[b] = bins.get(b, 0) + 1
    out = []
    for b in sorted(bins):
        out.append((lo + b * bucket, lo + (b + 1) * bucket, bins[b]))
    return out


def fmt_histogram(bins):
    if not bins:
        return "  (no successful requests)"
    mx = max(c for _, _, c in bins)
    width = 40
    lines = []
    for lo, hi, c in bins:
        bar = "#" * (int(width * c / mx) if mx else 0)
        lines.append(f"  {lo:9.2f} - {hi:9.2f}  {c:4d} | {bar}")
    return "\n".join(lines)


def quantile(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=8, help="distinct prompts per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mode", choices=["repeat", "novel"], default="repeat")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--bucket", type=float, default=0.0,
                   help="histogram bucket width (loc/s); 0 = auto (12 bins)")
    ap.add_argument("--stream", action="store_true", help="measure TTFT via SSE")
    ap.add_argument("--warmup", type=int, default=0,
                   help="drop the first N requests from the histogram")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default="", help="write JSON + CSV here (prefix)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args()

    fn = one_stream if args.stream else one_nonstream

    def job(round_idx, slot):
        prompt = build_prompt(args, round_idx, slot)
        for attempt in range(3):
            try:
                res = fn(args, args.model, prompt, args.max_tokens)
                res["round"] = round_idx
                res["slot"] = slot
                return res
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"round": round_idx, "slot": slot,
                            "error": f"{type(e).__name__}: {e}",
                            "wall": 0.0, "ttft": None, "gen": 0.0, "lines": 0,
                            "rate": 0.0, "prompt_tokens": 0, "chars": 0}
                time.sleep(2 * (attempt + 1))

    tasks = [(r, s) for r in range(args.rounds) for s in range(args.n)]
    results = []
    if args.concurrency <= 1:
        for r, s in tasks:
            results.append(job(r, s))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for res in ex.map(lambda t: job(t[0], t[1]), tasks):
                results.append(res)

    if args.warmup:
        results = results[args.warmup:]

    ok = [r for r in results if not r["error"]]
    errs = [r for r in results if r["error"]]
    rates = [r["rate"] for r in ok]

    per_round = []
    for r in range(args.rounds):
        rr = [x for x in ok if x["round"] == r]
        if not rr:
            per_round.append(None)
            continue
        tt = [x["ttft"] for x in rr if x["ttft"] is not None]
        per_round.append({
            "round": r, "n": len(rr),
            "wall": sum(x["wall"] for x in rr) / len(rr),
            "ttft": (sum(tt) / len(tt)) if tt else None,
            "rate": sum(x["rate"] for x in rr) / len(rr),
            "lines": sum(x["lines"] for x in rr) / len(rr),
        })

    if args.json:
        print(json.dumps({
            "config": vars(args),
            "per_round": per_round,
            "errors": [e["error"] for e in errs],
            "results": results,
        }, indent=2))
        return

    print("=" * 78)
    print("locbench -- looping code-generation synthetic workload")
    print("=" * 78)
    print(f"  url          {args.url}")
    print(f"  model        {args.model}")
    print(f"  mode         {args.mode}   (repeat = cache candidates; novel = full miss)")
    print(f"  n x rounds   {args.n} x {args.rounds}   (max_tokens={args.max_tokens}, "
          f"concurrency={args.concurrency}, stream={args.stream})")
    print("-" * 78)

    hdr = f"  {'round':>6} | {'ok':>4} | {'wall s':>8} | {'ttft s':>8} | {'loc/s':>8} | {'loc':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pr in per_round:
        if pr is None:
            continue
        ttft = f"{pr['ttft']:.2f}" if pr["ttft"] is not None else "  -  "
        print(f"  {pr['round']:>6} | {pr['n']:>4} | {pr['wall']:>8.2f} | {ttft:>8} | "
              f"{pr['rate']:>8.2f} | {pr['lines']:>6.1f}")
    if errs:
        print(f"  errors: {len(errs)} of {len(results)}  (first: {errs[0]['error']})")
    print("-" * 78)

    print(f"\n  histogram of lines-of-code per second (n={len(rates)}):")
    print(fmt_histogram(make_histogram(rates, args.bucket)))

    if rates:
        walls = [r["wall"] for r in ok]
        print(f"\n  rate  loc/s : mean {sum(rates)/len(rates):.2f} | "
              f"median {quantile(rates, 0.5):.2f} | min {min(rates):.2f} | "
              f"p95 {quantile(rates, 0.95):.2f} | max {max(rates):.2f}")
        print(f"  wall  s     : mean {sum(walls)/len(walls):.2f} | "
              f"median {quantile(walls, 0.5):.2f} | "
              f"p95 {quantile(walls, 0.95):.2f} | max {max(walls):.2f}")
        ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
        if ttfts:
            print(f"  ttft  s     : mean {sum(ttfts)/len(ttfts):.2f} | "
                  f"median {quantile(ttfts, 0.5):.2f} | "
                  f"p95 {quantile(ttfts, 0.95):.2f}")

    # cache verdict (repeat mode): round 1 vs the mean of rounds 2+
    if args.mode == "repeat" and len(per_round) >= 2 and per_round[0] and any(per_round[1:]):
        base = per_round[0]
        later = [p for p in per_round[1:] if p]
        if later:
            lw = sum(p["wall"] for p in later) / len(later)
            tt0 = base["ttft"]
            tt1 = (sum(p["ttft"] for p in later if p["ttft"] is not None) /
                   len([p for p in later if p["ttft"] is not None])) if args.stream else None
            speedup = base["wall"] / lw if lw > 0 else float("inf")
            print("\n  cache signal (round 1 = recompute baseline):")
            print(f"    wall  base {base['wall']:.2f}s  ->  later {lw:.2f}s  "
                  f"(x{speedup:.2f} faster)")
            if tt0 is not None and tt1 is not None:
                print(f"    ttft  base {tt0:.2f}s  ->  later {tt1:.2f}s  "
                      f"(x{tt0/tt1:.2f} faster)")
            if speedup > 1.5:
                print("    -> later rounds are clearly faster: the tier is serving hits.")
            elif speedup < 0.67:
                print("    -> later rounds are slower: cache is not helping (or is evicting).")
            else:
                print("    -> later rounds about the same: little/no cache effect on wall time.")

    if args.out:
        with open(args.out + ".json", "w") as f:
            json.dump({"config": vars(args), "per_round": per_round,
                      "errors": [e["error"] for e in errs], "results": results}, f, indent=2)
        with open(args.out + ".csv", "w") as f:
            f.write("round,slot,error,wall,ttft,gen,lines,rate,prompt_tokens,chars\n")
            for r in results:
                ttft = "" if r["ttft"] is None else f"{r['ttft']:.4f}"
                f.write(f"{r['round']},{r['slot']},{r['error'] or ''},{r['wall']:.4f},"
                        f"{ttft},{r['gen']:.4f},{r['lines']},{r['rate']:.4f},"
                        f"{r['prompt_tokens']},{r['chars']}\n")
        print(f"\n  wrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
