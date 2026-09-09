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
vLLM frees a request's blocks in REVERSE order, so the tail of a prompt is evicted from
the GPU before its head -- that is deliberate upstream behaviour, to preserve shared
prefixes. This bench uses it:

    coldA   cache_salt A, never seen      -> full recompute, reference answer
    coldB   cache_salt B, never seen      -> full recompute, identical text, noise floor
    warm    cache_salt B again            -> full GPU hit; then drain, so the whole prompt
                                             is now resident on the offload tier too
    evict   novel traffic, in increments  -> takes the free queue front-first, which means
                                             the junk eats the TAIL of the warm prompt
    mixed   cache_salt B again            -> head from the GPU prefix cache, tail from the
                                             offload tier: the case that crashed

The eviction volume is not predicted, it is FOUND. Each increment is followed by a full
probe, and the first probe that shows GPU hits AND external hits/loads in the same request
is the measurement. Predicting it would need to know how much of the pool is free, which
depends on what ran before.

BOX REQUIREMENTS
----------------
This does not need a GPU exclusive window, but it does need the box to itself: concurrent
traffic changes what is in the pool and the eviction sweep stops meaning anything. Same
warning equivbench and tierbench carry.

USAGE
-----
    ./mixedbench.py --yes                 full run
    ./mixedbench.py --dry-run             sizing only, no load
    ./mixedbench.py --yes --prefix-tokens 60000   smaller/faster

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

# Eviction sweep. Each step pushes this many novel tokens through the server, then probes.
# Keep it a whole multiple of tierbench.EVICT_CHUNK_TOKENS: evict() rounds DOWN to
# int(want / EVICT_CHUNK_TOKENS) chunks, so 24000 would silently push 30000 and the
# "evicted so far" column would be fiction.
EVICT_STEP_TOKENS = int(os.environ.get("MIXEDBENCH_EVICT_STEP",
                                       str(T.EVICT_CHUNK_TOKENS)))
# Stop rather than evict forever. Past ~1.5x the GPU pool the prompt is gone entirely and
# every further probe is a pure external hit, which is equivbench's test, not this one.
EVICT_CAP_MULT = 1.6

# Lines in the engine log that mean this test found a real failure, not a slow answer.
FATAL_PATTERNS = [
    "offload boundary",              # the assertion R3.15 scoped to full-attention groups
    "not found in cache",            # prepare_load's assert: a key lookup never confirmed
    "AssertionError",
    "EngineCore encountered a fatal error",
    "Engine core proc died",
]


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
    L.append("| phase | salt | evicted so far | served | wall | cached_tokens | "
             "gpu_hits | ext_hits | load MB |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in phases:
        r, c = p["probe"], p["counters"]
        L.append("| %s | %s | %s | %s | %.2fs | %s | %g | %g | %.1f |"
                 % (p["phase"], r["cache_salt"][:10],
                    "{:,}".format(p.get("evicted_tokens", 0)),
                    p["served_by"], r["wall_s"],
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

    evict_cap = int(cap["gpu_tokens"] * EVICT_CAP_MULT)
    T.log("plan: prompt %s tok | evict in %s-tok steps, cap %s tok | max_tokens %d"
          % ("{:,}".format(args.prefix_tokens), "{:,}".format(EVICT_STEP_TOKENS),
             "{:,}".format(evict_cap), E.MAX_TOKENS))

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
        "max_tokens": E.MAX_TOKENS, "evict_step_tokens": EVICT_STEP_TOKENS,
        "evict_cap_tokens": evict_cap, "drains": [],
    }
    phases, comps = [], []
    evicted = 0
    rc = 0

    def run(name, salt, expect_kinds, evicted_tokens=0):
        T.budget_check()
        T.log("PHASE %s" % name)
        m0 = T.snapshot(base)
        r = E.ask(base, P, name, salt)
        m1 = T.snapshot(base)
        d = T.delta(m0, m1)
        kind, counters = classify_mixed(d)
        ok = expect_kinds is None or any(kind.startswith(k) for k in expect_kinds)
        row = {"phase": name, "expected": expect_kinds, "served_by": kind,
               "valid": ok, "probe": r, "metrics_delta": d, "counters": counters,
               "evicted_tokens": evicted_tokens}
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

        run("coldA", SALT_A, ["RECOMPUTE"])
        run("coldB", SALT_B, ["RECOMPUTE"])
        run("warm", SALT_B, ["GPU", "MIXED"])

        # The tail must be ON the tier before it is evicted from the GPU, or the "mixed"
        # phase degrades into a partial recompute and proves nothing.
        meta["drains"].append(T.drain(base, "post-warm"))

        # The pool holds gpu_tokens; our warm prompt occupies prompt_tokens of it and was
        # freed most recently, so everything else in there is ahead of it in the free
        # queue. Clearing that much first puts the sweep directly on the prompt's tail
        # instead of spending five probes evicting coldA and whatever preceded this run.
        pre = max(0, cap["gpu_tokens"] - ptok)
        pre = (pre // T.EVICT_CHUNK_TOKENS) * T.EVICT_CHUNK_TOKENS
        if pre:
            T.log("PRE-EVICT %s tok (pool %s - prompt %s), to reach the prompt's tail"
                  % ("{:,}".format(pre), "{:,}".format(cap["gpu_tokens"]),
                     "{:,}".format(ptok)))
            T.evict(base, sizer, content_salt, pre, "preevict")
            evicted += pre
        meta["pre_evict_tokens"] = pre

        mixed_row = None
        step = 0
        while evicted < evict_cap:
            step += 1
            T.budget_check()
            T.log("EVICT step %d (+%s tok, %s total)"
                  % (step, "{:,}".format(EVICT_STEP_TOKENS), "{:,}".format(
                      evicted + EVICT_STEP_TOKENS)))
            T.evict(base, sizer, content_salt, EVICT_STEP_TOKENS,
                    "evict%d" % step)
            evicted += EVICT_STEP_TOKENS
            row = run("probe%d" % step, SALT_B, None, evicted_tokens=evicted)
            if row["served_by"].startswith("MIXED"):
                row["phase"] = "mixed"
                mixed_row = row
                break
            if row["served_by"] == "OFFLOAD":
                T.log("  overshot: the GPU no longer holds any of the prompt. "
                      "Reduce MIXEDBENCH_EVICT_STEP and rerun.")
                break

        meta["mixed_built"] = mixed_row is not None
        if mixed_row is None:
            T.log("NO MIXED HIT BUILT -- nothing was tested. See the phase table.")
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
