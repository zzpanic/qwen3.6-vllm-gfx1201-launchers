#!/usr/bin/env python3
"""mixedbench -- is a MIXED local+external cache hit served correctly?

WHY THIS EXISTS
---------------
equivbench answered "is a disk hit the same answer as a recompute?" and the answer was
yes, bit-identical, 85,696 tokens. It cannot answer the question R3.15 raises, because its
`fs` phase evicts the ENTIRE cache first: the probe that follows has num_computed_tokens=0,
so it is a pure external hit and never touches the code path that crashed.

The crash needed BOTH tiers at once -- a request that hits the GPU prefix cache AND then
asks the connector for more. That is the case where vLLM reports a divergent per-group hit
(get_computed_blocks_for_connector deliberately does not reconcile the groups), which is
what made the boundary assertion fire and what made the lookup confirm a narrower chunk
range than the load reads. See cache-preemption-patch-plan.md R3.15.

So this bench builds a mixed hit on purpose and checks three things:

  1. the engine survives it            (the old failure was an AssertionError that kills
                                        EngineCore, so "still alive" is a real result)
  2. the answer is bit-identical       (the fix widens which chunks a lookup confirms; if
     to a cold recompute                it ever confirmed the WRONG chunk, the tokens
                                        would drift -- this is the test for that)
  3. the hit is not silently declined  (a fix that simply stopped serving mixed hits would
                                        pass 1 and 2 and be worthless)

HOW A MIXED HIT IS BUILT
------------------------
By warming a HEAD prefix into the GPU while the FULL prompt sits on the offload tier:

    coldA     FULL, cache_salt A, never seen  -> recompute, reference answer
    coldB     FULL, cache_salt B, never seen  -> recompute, identical text, noise floor
    drain                                     -> FULL is now resident on the tier
    evictall  ~1.2x the GPU pool of novel     -> the GPU holds none of FULL
              traffic
    head      HEAD, cache_salt B              -> puts ONLY the first HEAD_TOKENS of the
                                                 prompt back in the GPU prefix cache
    mixed     FULL, cache_salt B              -> head from the GPU, tail from the tier

HEAD is a literal text prefix of FULL under the same cache_salt, so their token streams
share a prefix and vLLM's block hash chain matches for the whole of HEAD.

WHY NOT PARTIAL EVICTION. The first version of this bench evicted the prompt's tail and
kept its head, which is what vLLM's own block ordering is built to do -- blocks are freed
via `reversed(...)` precisely so the tail is evicted first and shared prefixes survive
(single_type_kv_cache_manager.py, `free_blocks(reversed(pop_blocks_for_free(...)))`). It
still did not work: run a1f560 went from a full GPU hit straight to gpu_hits=0 with 79,104
external tokens in one 30k step, with no mixed state in between. Aiming at a window whose
position depends on what was in the pool before the run is not a test, it is a coin flip.
The two-prompt construction does not depend on eviction order at all, and it reproduces
the production pattern directly: a conversation head stays hot while longer continuations
come back from disk.

BOX REQUIREMENTS
----------------
This does not need a GPU exclusive window, but it does need the box to itself: concurrent
traffic changes what is in the pool and the eviction sweep stops meaning anything. Same
warning equivbench and tierbench carry.

USAGE
-----
    ./mixedbench.py --yes                 full run
    ./mixedbench.py --dry-run             sizing only, no load
    ./mixedbench.py --yes --head-tokens 20000     smaller GPU-resident head

Exit status is 0 only if a mixed hit was actually built, the engine survived it, and the
tokens matched both recomputes.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tierbench as T                                          # noqa: E402
import equivbench as E                                         # noqa: E402

OUT_DIR = os.environ.get("MIXEDBENCH_OUT", T.OUT_DIR)

# How much of the prompt is warmed back into the GPU before the mixed probe. It only has
# to exceed one chunk (1,648 tokens here) to make the local hit real; a third of the prompt
# leaves an unambiguous tail for the connector to fetch.
HEAD_TOKENS = int(os.environ.get("MIXEDBENCH_HEAD_TOKENS", "30000"))
# The evictall phase must clear the whole pool, not most of it: any surviving block of the
# prompt makes the head phase ambiguous about what it actually put back.
EVICT_ALL_MULT = float(os.environ.get("MIXEDBENCH_EVICT_MULT", "1.25"))

# Lines in the engine log that mean this test found a real failure, not a slow answer.
FATAL_PATTERNS = [
    "offload boundary",              # the assertion R3.15 scoped to full-attention groups
    "not found in cache",            # prepare_load's assert: a key lookup never confirmed
    "AssertionError",
    "EngineCore encountered a fatal error",
    "Engine core proc died",
]


def head_prefix(sizer, full_text, want_tokens):
    """A literal text prefix of `full_text` that tokenizes to about `want_tokens`.

    It must be a text prefix, not separately generated text: vLLM hashes token blocks in
    sequence, so only a shared TOKEN prefix produces a prefix-cache hit, and the cheapest
    way to guarantee one is to cut the same string. Cut on whitespace so the boundary token
    is not split -- an off-by-one token at the very end costs at most the final partial
    block, which is not part of the hit either way."""
    lo, hi = 0, len(full_text)
    cut = min(hi, max(1, int(hi * want_tokens / max(1, sizer.count(full_text)))))
    for _ in range(6):
        sp = full_text.rfind(" ", 0, cut)
        cand = full_text[: sp if sp > 0 else cut]
        got = sizer.count(cand)
        if abs(got - want_tokens) <= max(200, want_tokens * 0.02):
            return cand, got
        if got < want_tokens:
            lo = cut
        else:
            hi = cut
        cut = (lo + hi) // 2 if hi > lo else int(cut * want_tokens / max(1, got))
        cut = max(1, min(len(full_text), cut))
    return cand, got


def classify_mixed(d):
    """tierbench.classify() collapses to OFFLOAD as soon as any external byte moves, which
    hides the case this bench exists for. Report both halves separately."""
    gpu = d.get("vllm:prefix_cache_hits_total", 0.0)
    ext = d.get("vllm:external_prefix_cache_hits_total", 0.0)
    loaded = d.get("vllm:kv_offload_load_bytes_total",
                   d.get("vllm:kv_offload_total_bytes_total|CPU_to_GPU", 0.0))
    fs = d.get("vllm:kv_offload_fs_load_bytes_total", 0.0) or 0.0
    external = (ext > 0 or loaded > 0)
    if gpu > 0 and external:
        kind = "MIXED/FS" if fs > 0 else "MIXED/CPU"
    elif external:
        kind = "OFFLOAD"
    elif gpu > 0:
        kind = "GPU"
    else:
        kind = "RECOMPUTE"
    return kind, {"gpu_hits": gpu, "ext_hits": ext,
                  "load_bytes": loaded, "fs_load_bytes": fs}


def engine_log_since(ts):
    """Engine log since a timestamp. podman, not the vLLM logger, because the failure this
    is watching for KILLS the process that would otherwise report it."""
    try:
        p = subprocess.run(
            ["podman", "logs", "--since", ts, T.CONTAINER],
            capture_output=True, text=True, timeout=60)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return "<could not read container log: %r>" % (e,)


def scan_fatal(text):
    return [ln.strip() for ln in text.splitlines()
            if any(pat in ln for pat in FATAL_PATTERNS)]


def engine_alive(base):
    try:
        T.http(base + "/health", timeout=15, raw=True)
        return True
    except Exception:
        return False


def write_report(path_md, path_json, meta, phases, comps, fatal):
    with open(path_json, "w") as f:
        json.dump({"meta": meta, "phases": phases, "comparisons": comps,
                   "fatal_log_lines": fatal}, f, indent=1)

    L = []
    L.append("# Mixed local+external cache hit -- correctness\n")
    L.append("Run `%s`, started %s.\n" % (meta["run_id"], meta["started"]))
    L.append("Prompt %s tokens, max_tokens=%d, temperature=0.\n"
             % ("{:,}".format(meta.get("prompt_tokens", 0)), E.MAX_TOKENS))
    L.append("Tests the path fixed in R3.15: a request that hits the GPU prefix cache "
             "AND asks the connector for the rest.\n")

    L.append("\n## Verdict\n")
    L.append("- mixed hit built: **%s**" % ("yes" if meta.get("mixed_built") else "**NO**"))
    L.append("- engine alive at end: **%s**" % ("yes" if meta.get("engine_alive") else "**NO**"))
    L.append("- tokens identical to both recomputes: **%s**"
             % ("yes" if meta.get("tokens_ok") else "**NO**"))
    L.append("- fatal log lines: **%d**" % len(fatal))
    if fatal:
        L.append("\n```")
        L.extend(fatal[:40])
        L.append("```")

    L.append("\n## Phases\n")
    L.append("| phase | salt | served | wall | prompt_tokens | cached_tokens | "
             "gpu_hits | ext_hits | load MB |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in phases:
        r, c = p["probe"], p["counters"]
        L.append("| %s | %s | %s | %.2fs | %s | %s | %g | %g | %.1f |"
                 % (p["phase"], r["cache_salt"][:10],
                    p["served_by"], r["wall_s"],
                    "{:,}".format(r["prompt_tokens"] or 0),
                    "{:,}".format(r["cached_tokens"] or 0),
                    c["gpu_hits"], c["ext_hits"], c["load_bytes"] / 1e6))

    L.append("\n## Comparisons\n")
    L.append("| pair | tokens identical | matching prefix | first divergence | "
             "max abs dlogprob |")
    L.append("|---|---|---|---|---|")
    for c in comps:
        L.append("| %s | %s | %d/%d | %s | %s |"
                 % (c["pair"], "**yes**" if c["tokens_identical"] else "**NO**",
                    c["matching_prefix_tokens"], max(c["n_tokens"]),
                    "-" if c["first_divergent_token_index"] is None
                    else "token %d" % c["first_divergent_token_index"],
                    "-" if c["max_abs_logprob_delta"] is None
                    else "%.3e" % c["max_abs_logprob_delta"]))

    for c in comps:
        if "divergence" in c:
            d = c["divergence"]
            L.append("\n### Divergence: %s, token %d\n" % (c["pair"], d["index"]))
            L.append("```")
            L.append("a: %s" % " | ".join(repr(x) for x in d["context_a"]))
            L.append("b: %s" % " | ".join(repr(x) for x in d["context_b"]))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                    help="required: this pushes a lot of eviction traffic")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prefix-tokens", type=int, default=T.TARGET_PREFIX_TOKENS)
    ap.add_argument("--head-tokens", type=int, default=HEAD_TOKENS,
                    help="how much of the prompt is warmed back into the GPU")
    args = ap.parse_args()

    run_id = "%x" % (int(time.time()) & 0xFFFFFF)
    content_salt = run_id
    SALT_A = "mixedbench-A-" + run_id
    SALT_B = "mixedbench-B-" + run_id

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

    evict_all = int(cap["gpu_tokens"] * EVICT_ALL_MULT)
    T.log("plan: prompt %s tok | head %s tok | evictall %s tok | max_tokens %d"
          % ("{:,}".format(args.prefix_tokens), "{:,}".format(args.head_tokens),
             "{:,}".format(evict_all), E.MAX_TOKENS))

    if args.dry_run:
        T.log("dry run: nothing sent.")
        return 0
    if not args.yes:
        T.log("refusing to run without --yes")
        return 2

    os.makedirs(OUT_DIR, exist_ok=True)
    started_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = {
        "run_id": run_id, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_utc": started_ts, "content_salt": content_salt,
        "cache_salt_a": SALT_A, "cache_salt_b": SALT_B,
        "endpoint": base, "capacity": cap, "geometry": geom,
        "max_tokens": E.MAX_TOKENS, "head_tokens_requested": args.head_tokens,
        "evict_all_tokens": evict_all, "drains": [],
    }
    phases, comps = [], []
    rc = 0

    def run(name, salt, expect_kinds, text=None, note=""):
        T.budget_check()
        T.log("PHASE %s" % name)
        m0 = T.snapshot(base)
        r = E.ask(base, text if text is not None else P, name, salt)
        m1 = T.snapshot(base)
        d = T.delta(m0, m1)
        kind, counters = classify_mixed(d)
        ok = expect_kinds is None or any(kind.startswith(k) for k in expect_kinds)
        row = {"phase": name, "expected": expect_kinds, "served_by": kind,
               "valid": ok, "probe": r, "metrics_delta": d, "counters": counters,
               "note": note}
        phases.append(row)
        T.log("  %s: wall %.2fs served_by=%s cached=%s gpu_hits=%g ext_hits=%g "
              "load=%.1f MB%s"
              % (name, r["wall_s"], kind, r["cached_tokens"],
                 counters["gpu_hits"], counters["ext_hits"],
                 counters["load_bytes"] / 1e6, "" if ok else "   <-- unexpected"))
        return row

    try:
        P, ptok = sizer.build(content_salt, "prefix", args.prefix_tokens)
        meta["prompt_tokens"] = ptok
        T.log("prompt built: %s tokens" % "{:,}".format(ptok))

        H, htok = head_prefix(sizer, P, args.head_tokens)
        meta["head_tokens"] = htok
        T.log("head prefix built: %s tokens (%.0f%% of the prompt)"
              % ("{:,}".format(htok), 100.0 * htok / ptok))

        run("coldA", SALT_A, ["RECOMPUTE"])
        run("coldB", SALT_B, ["RECOMPUTE"])

        # FULL must be ON the tier before the GPU is cleared, or the mixed phase degrades
        # into a partial recompute and proves nothing.
        meta["drains"].append(T.drain(base, "post-coldB"))

        T.log("EVICTALL %s tok -- the GPU must hold none of the prompt"
              % "{:,}".format(evict_all))
        T.evict(base, sizer, content_salt, evict_all, "evictall")

        # Puts back ONLY the head. Whether this is served from the tier or recomputed does
        # not matter; what matters is that afterwards the head is GPU-resident and the tail
        # is not.
        run("head", SALT_B, None, text=H,
            note="warms the first %s tokens into the GPU" % "{:,}".format(htok))

        mixed_row = run("mixed", SALT_B, ["MIXED"],
                        note="head from GPU, tail from the offload tier")
        meta["mixed_built"] = mixed_row["served_by"].startswith("MIXED")
        if not meta["mixed_built"]:
            T.log("NO MIXED HIT BUILT (served_by=%s) -- nothing was tested."
                  % mixed_row["served_by"])
            rc = 1

        by = {p["phase"]: p["probe"] for p in phases}
        for x, y in (("coldA", "coldB"), ("coldB", "mixed"), ("coldA", "mixed")):
            if x in by and y in by:
                c = E.compare(by[x], by[y])
                comps.append(c)
                T.log("  %-16s tokens_identical=%s  prefix=%d/%d  max|dlogprob|=%s"
                      % (c["pair"], c["tokens_identical"],
                         c["matching_prefix_tokens"], max(c["n_tokens"]),
                         "-" if c["max_abs_logprob_delta"] is None
                         else "%.3e" % c["max_abs_logprob_delta"]))

        mixed_comps = [c for c in comps if "mixed" in c["pair"]]
        meta["tokens_ok"] = bool(mixed_comps) and all(
            c["tokens_identical"] for c in mixed_comps)
        if not meta["tokens_ok"]:
            rc = 1

    except T.Abort as e:
        T.log("ABORT: %s" % e)
        meta["aborted"] = str(e)
        rc = 1
    except KeyboardInterrupt:
        T.log("interrupted")
        meta["aborted"] = "KeyboardInterrupt"
        rc = 130

    meta["engine_alive"] = engine_alive(base)
    fatal = scan_fatal(engine_log_since(started_ts))
    if fatal or not meta["engine_alive"]:
        rc = 1
    meta.setdefault("prompt_tokens", args.prefix_tokens)
    meta.setdefault("mixed_built", False)
    meta.setdefault("tokens_ok", False)

    md = os.path.join(OUT_DIR, "MIXED-%s.md" % run_id)
    js = os.path.join(OUT_DIR, "MIXED-%s.json" % run_id)
    write_report(md, js, meta, phases, comps, fatal)
    T.log("wrote %s" % md)
    T.log("VERDICT: mixed_built=%s engine_alive=%s tokens_ok=%s fatal_lines=%d -> rc=%d"
          % (meta["mixed_built"], meta["engine_alive"], meta["tokens_ok"],
             len(fatal), rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
