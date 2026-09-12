#!/usr/bin/env python3
"""equivbench -- does a KV cache hit produce the same answer as a full recompute?

WHY THIS EXISTS
---------------
tierbench (run 9e4d2b) proved the disk tier can serve a real hit: 85,696 cached tokens
read back as 3.00 GB in 0.254 s. What it did NOT prove is that the hit is *correct*.
It threw the completion text away, so "the read path works" rested entirely on the
volume of bytes moved and on the phase reaching the tier it was aimed at. R3.10.8 of
cache-preemption-patch-plan.md lists this as the one thing still unmeasured, and says
to measure it before shipping anything.

WHAT MAKES THIS HARD, AND HOW IT IS HANDLED
-------------------------------------------
"Same answer" is only meaningful against a known noise floor. Two things could make two
runs of the same prompt disagree without any cache bug at all:

  * DFlash2 speculative decoding. `mtp-spec-decoding-not-lossless.md` records MTP at
    +15.8 sigma on this box; DFlash2 measured clean at +1.4 sigma, but "clean" is a
    statistical statement, not a promise of bit-identical greedy decode.
  * Batch-shape-dependent kernel reductions. A prefill split differently across chunks
    can reorder floating-point accumulation.

So the test needs a control: the SAME prompt text recomputed twice from scratch. Getting
that used to require evicting every tier between the two runs. It does not, because this
build accepts `cache_salt` on /v1/chat/completions, and vLLM folds the salt into the
block-0 extra keys (v1/core/kv_cache_utils.py:579). Block hashes chain, so salting the
first block invalidates the whole chain: identical text under a fresh salt is a
guaranteed full recompute, at the cost of one prefill and no eviction traffic.

Four measurements of one prompt, in this order:

    coldA   cache_salt A, never seen        -> recompute, answer A
    coldB   cache_salt B, never seen        -> recompute, answer B   (identical text)
    gpu     cache_salt B again              -> GPU prefix-cache hit
    fs      cache_salt B, after eviction    -> disk hit

and four comparisons:

    coldA vs coldB   the NOISE FLOOR. Two recomputes of the same tokens. Any
                     disagreement here is the decode pipeline, not the cache.
    coldB vs gpu     does resuming from cached KV change the answer at all?
    coldB vs fs      the question pat asked, against the same-chain recompute.
    coldA vs fs      the same question against an independent recompute.

Comparison is token-level, not string-level: `logprobs: true` returns the emitted token
sequence and each token's logprob, so a mismatch can be located exactly and a match can
be graded -- identical tokens with identical logprobs is a bit-exact resume, identical
tokens with drifting logprobs is a numerically-different-but-behaviourally-equal resume,
and a divergence at a near-tie position is nondeterminism rather than corruption.

No min_p is sent anywhere: `vllm-spec-decoding-forbids-min-p.md`, this build 400s.

USAGE
-----
    ./equivbench.py --yes                  full run, ~12-18 min
    ./equivbench.py --yes --phases coldA,coldB,gpu    skip the expensive fs phase
    ./equivbench.py --dry-run              sizing only, no load

THIS EVICTS THE ENTIRE KV CACHE (the fs phase does). Same warning as tierbench: it does
not need a GPU exclusive window, but it does need the box to itself.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tierbench as T                                          # noqa: E402

MAX_TOKENS = int(os.environ.get("EQUIVBENCH_MAX_TOKENS", "128"))
TOP_LOGPROBS = 3
OUT_DIR = os.environ.get("EQUIVBENCH_OUT", T.OUT_DIR)


# ---------------------------------------------------------------- request

def ask(base, text, tag, cache_salt, max_tokens=MAX_TOKENS):
    """Non-streamed on purpose. Streaming buys TTFT, which this bench does not need, and
    it makes the logprobs arrive in fragments that have to be reassembled. Tier
    attribution comes from `cached_tokens` and the metric deltas, not from TTFT."""
    payload = {
        "model": T.MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "cache_salt": cache_salt,
        "stream": False,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=T.PER_REQ_TIMEOUT) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise T.Abort("%s: HTTP %s %s" % (tag, e.code, e.read().decode()[:400]))
    except Exception as e:
        raise T.Abort("%s: %r" % (tag, e))
    wall = time.time() - t0

    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    # This model thinks, and at 128 tokens it is still thinking: the whole completion
    # lands in `reasoning` and `content` stays null. The field is `reasoning`, not
    # `reasoning_content`, on this build (vllm-reasoning-content-renamed.md).
    reasoning = msg.get("reasoning") or ""
    content = msg.get("content") or ""
    lp = ((ch.get("logprobs") or {}).get("content")) or []
    usage = body.get("usage") or {}
    return {
        "tag": tag,
        "cache_salt": cache_salt,
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": ch.get("finish_reason"),
        "reasoning": reasoning,
        "content": content,
        "text": reasoning + content,
        "tokens": [t.get("token") for t in lp],
        "logprobs": [t.get("logprob") for t in lp],
        "top_logprobs": [
            [(c.get("token"), c.get("logprob")) for c in (t.get("top_logprobs") or [])]
            for t in lp
        ],
    }


# ---------------------------------------------------------------- comparison

def compare(a, b):
    """Token-level, with the logprobs used to grade a match and to explain a mismatch."""
    ta, tb = a["tokens"], b["tokens"]
    la, lb = a["logprobs"], b["logprobs"]
    n = min(len(ta), len(tb))
    first = None
    for i in range(n):
        if ta[i] != tb[i]:
            first = i
            break
    if first is None and len(ta) != len(tb):
        first = n                      # one is a strict prefix of the other
    lim = n if first is None else first

    deltas = [abs(la[i] - lb[i]) for i in range(lim)
              if la[i] is not None and lb[i] is not None]
    out = {
        "pair": "%s vs %s" % (a["tag"], b["tag"]),
        "text_identical": a["text"] == b["text"],
        "tokens_identical": ta == tb,
        "n_tokens": [len(ta), len(tb)],
        "matching_prefix_tokens": lim,
        "first_divergent_token_index": first,
        "logprobs_bit_identical": bool(deltas) and max(deltas) == 0.0,
        "n_logprobs_exact": sum(1 for d in deltas if d == 0.0),
        "n_logprobs_compared": len(deltas),
        "max_abs_logprob_delta": max(deltas) if deltas else None,
        "mean_abs_logprob_delta": (sum(deltas) / len(deltas)) if deltas else None,
    }
    if first is not None:
        w = 4
        lo, hi = max(0, first - w), first + w + 1
        out["divergence"] = {
            "index": first,
            "context_a": ta[lo:hi],
            "context_b": tb[lo:hi],
            # A flip between two candidates that were nearly tied is nondeterminism.
            # A flip to a token that was not even in the top-3 is not.
            "top3_a": a["top_logprobs"][first] if first < len(a["top_logprobs"]) else None,
            "top3_b": b["top_logprobs"][first] if first < len(b["top_logprobs"]) else None,
            "margin_a": _margin(a, first),
            "margin_b": _margin(b, first),
        }
    return out


def _margin(r, i):
    """logprob gap between the chosen token and the runner-up at position i. A tiny gap
    means the two runs were choosing between near-equals."""
    try:
        top = r["top_logprobs"][i]
        if len(top) < 2:
            return None
        return round(abs(top[0][1] - top[1][1]), 6)
    except Exception:
        return None


# ---------------------------------------------------------------- report

def write_report(path_md, path_json, meta, phases, comps):
    with open(path_json, "w") as f:
        json.dump({"meta": meta, "phases": phases, "comparisons": comps}, f, indent=1)

    L = []
    L.append("# KV cache hit vs cold recompute -- output equivalence\n")
    L.append("Run `%s`, started %s.\n" % (meta["run_id"], meta["started"]))
    L.append("Prompt %s tokens, max_tokens=%d, temperature=0, logprobs top-%d.\n"
             % ("{:,}".format(meta["prompt_tokens"]), MAX_TOKENS, TOP_LOGPROBS))
    L.append("Identical prompt text in every phase; `cache_salt` is what separates a "
             "fresh chain from a reused one.\n")

    L.append("\n## Phases\n")
    L.append("| phase | salt | expected | served | wall | cached_tokens | valid |")
    L.append("|---|---|---|---|---|---|---|")
    for p in phases:
        r = p["probe"]
        L.append("| %s | %s | %s | %s | %.2fs | %s | %s |"
                 % (p["phase"], r["cache_salt"][:8], p["expected"] or "-",
                    p["served_by"], r["wall_s"],
                    "{:,}".format(r["cached_tokens"] or 0),
                    "yes" if p["valid"] else "**NO**"))

    L.append("\n## Comparisons\n")
    L.append("| pair | tokens identical | text identical | matching prefix | "
             "first divergence | logprobs bit-identical | max abs dlogprob |")
    L.append("|---|---|---|---|---|---|---|")
    for c in comps:
        L.append("| %s | %s | %s | %d/%d | %s | %s | %s |"
                 % (c["pair"],
                    "**yes**" if c["tokens_identical"] else "**NO**",
                    "yes" if c["text_identical"] else "no",
                    c["matching_prefix_tokens"], max(c["n_tokens"]),
                    "-" if c["first_divergent_token_index"] is None
                    else "token %d" % c["first_divergent_token_index"],
                    "yes" if c["logprobs_bit_identical"] else "no",
                    "-" if c["max_abs_logprob_delta"] is None
                    else "%.3e" % c["max_abs_logprob_delta"]))

    for c in comps:
        if "divergence" in c:
            d = c["divergence"]
            L.append("\n### Divergence: %s, token %d\n" % (c["pair"], d["index"]))
            L.append("```")
            L.append("a: %s" % " | ".join(repr(x) for x in d["context_a"]))
            L.append("b: %s" % " | ".join(repr(x) for x in d["context_b"]))
            L.append("top3 a: %s   (margin %s)" % (d["top3_a"], d["margin_a"]))
            L.append("top3 b: %s   (margin %s)" % (d["top3_b"], d["margin_b"]))
            L.append("```")

    L.append("\n## Completions\n")
    for p in phases:
        L.append("\n### %s\n" % p["phase"])
        L.append("```")
        L.append(p["probe"]["text"])
        L.append("```")

    L.append("\n## Metric deltas per phase\n")
    for p in phases:
        L.append("\n### %s\n" % p["phase"])
        L.append("```")
        for k, v in sorted(p["metrics_delta"].items()):
            if v:
                L.append("%-58s %s" % (k, v))
        L.append("```")

    with open(path_md, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                    help="required: the fs phase evicts the entire KV cache")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--salt", default=None, help="replay a run's prompt content")
    ap.add_argument("--phases", default="coldA,coldB,gpu,fs")
    ap.add_argument("--prefix-tokens", type=int, default=T.TARGET_PREFIX_TOKENS)
    args = ap.parse_args()

    run_id = "%x" % (int(time.time()) & 0xFFFFFF)
    content_salt = args.salt or run_id
    want = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Cache salts. These are what make coldA and coldB two independent recomputes of
    # identical text; they must differ between runs too, or run 2's coldB finds run 1's
    # chain still on disk and is not cold at all.
    SALT_A = "equivbench-A-" + run_id
    SALT_B = "equivbench-B-" + run_id

    base = T.detect_endpoint()
    cap = T.detect_capacity()
    T.log("endpoint %s" % base)
    T.log("capacity (%s): GPU %s tokens, CPU primary %d blocks"
          % (cap["source"], "{:,}".format(cap["gpu_tokens"]), cap["cpu_blocks"]))

    sizer = T.Sizer(base)
    geom = T.detect_geometry()
    if geom:
        T._BLOCK_BYTES[0] = geom["block_file_bytes"]
        T.log("geometry (%s): %d groups x %s B per %d-token block"
              % (geom["source"], geom["n_groups"],
                 "{:,}".format(geom["block_file_bytes"]), geom["tokens_per_hash"]))
        cpu_tokens = (cap["cpu_blocks"] // geom["n_groups"]) * geom["tokens_per_hash"]
    else:
        cpu_tokens = 115360
    evict_gpu = int(cap["gpu_tokens"] * T.EVICT_MARGIN)
    evict_cpu = max(int(cpu_tokens * T.EVICT_MARGIN), evict_gpu)
    T.log("plan: prompt %s tok | evict-CPU %s tok | max_tokens %d"
          % ("{:,}".format(args.prefix_tokens), "{:,}".format(evict_cpu), MAX_TOKENS))
    T.log("cache salts: A=%s  B=%s" % (SALT_A, SALT_B))

    if args.dry_run:
        T.log("dry run: nothing sent.")
        return 0
    if not args.yes:
        T.log("refusing to run without --yes (the fs phase evicts the entire KV cache)")
        return 2

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "run_id": run_id,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "content_salt": content_salt,
        "cache_salt_a": SALT_A, "cache_salt_b": SALT_B,
        "endpoint": base, "capacity": cap, "geometry": geom,
        "max_tokens": MAX_TOKENS, "top_logprobs": TOP_LOGPROBS,
        "evict_cpu_tokens": evict_cpu,
        "phases_requested": want,
        "drains": [],
    }
    phases, comps = [], []
    rc = 0

    def run(name, salt, expect, before=None):
        T.budget_check()
        T.log("PHASE %s" % name)
        if before:
            before()
        m0 = T.snapshot(base)
        r = ask(base, P, name, salt)
        m1 = T.snapshot(base)
        d = T.delta(m0, m1)
        served = T.classify(d)
        ok = expect is None or served.startswith(expect)
        row = {"phase": name, "expected": expect, "served_by": served,
               "valid": ok, "probe": r, "metrics_delta": d}
        phases.append(row)
        T.log("  %s: wall %.2fs served_by=%s cached=%s tokens=%d%s"
              % (name, r["wall_s"], served, r["cached_tokens"], len(r["tokens"]),
                 "" if ok else "   <-- INVALID"))
        return row

    try:
        P, ptok = sizer.build(content_salt, "prefix", args.prefix_tokens)
        meta["prompt_tokens"] = ptok
        T.log("prompt built: %s tokens" % "{:,}".format(ptok))

        if "coldA" in want:
            run("coldA", SALT_A, "RECOMPUTE")
        if "coldB" in want:
            run("coldB", SALT_B, "RECOMPUTE")
        if "gpu" in want:
            run("gpu", SALT_B, "GPU")
        if "fs" in want:
            def fs_before():
                meta["drains"].append(T.drain(base, "pre-evict"))
                T.evict(base, sizer, content_salt, evict_cpu, "evictcpu")
                meta["drains"].append(T.drain(base, "post-evict"))
            run("fs", SALT_B, "OFFLOAD", before=fs_before)

        by = {p["phase"]: p["probe"] for p in phases}
        for x, y in (("coldA", "coldB"), ("coldB", "gpu"), ("coldB", "fs"),
                     ("coldA", "fs")):
            if x in by and y in by:
                c = compare(by[x], by[y])
                comps.append(c)
                T.log("  %-16s tokens_identical=%s  prefix=%d/%d  max|dlogprob|=%s"
                      % (c["pair"], c["tokens_identical"],
                         c["matching_prefix_tokens"], max(c["n_tokens"]),
                         "-" if c["max_abs_logprob_delta"] is None
                         else "%.3e" % c["max_abs_logprob_delta"]))
    except T.Abort as e:
        T.log("ABORT: %s" % e)
        meta["aborted"] = str(e)
        rc = 1
    except KeyboardInterrupt:
        T.log("interrupted")
        meta["aborted"] = "KeyboardInterrupt"
        rc = 130

    md = os.path.join(OUT_DIR, "EQUIV-%s.md" % run_id)
    js = os.path.join(OUT_DIR, "EQUIV-%s.json" % run_id)
    if "prompt_tokens" not in meta:
        meta["prompt_tokens"] = args.prefix_tokens
    write_report(md, js, meta, phases, comps)
    T.log("wrote %s" % md)
    T.log("wrote %s" % js)
    return rc


if __name__ == "__main__":
    sys.exit(main())
