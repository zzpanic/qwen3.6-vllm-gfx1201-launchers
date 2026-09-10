#!/usr/bin/env python3
"""correctbench -- KV-cache correctness suite for the patched offload stack.

Closes out "is the cache correct?" for every path exercisable against the running
endpoint, with NO model reload. See CORRECTNESS.md for the full rationale and the
exact assertion per test.

The property under test (correctness-suite-scope.md):
    Serving any part of a prompt from any cache tier must produce EXACTLY the output a
    full recompute of that same prompt produces -- and a prompt that shares no legitimate
    chained prefix must never be served from another prompt's blocks.

Tests:
    CT5  negative controls              no GPU work; prove the instrument can report failure
    CT1  path equivalence matrix        cold / GPU / OFFLOAD / MIXED, all vs a cold reference
    CT2  salt isolation                 the A1 (R3.15) regression test; counters asserted
                                        separately and both at zero
    CT3  cross-prompt contamination     CT2 + an eviction, so the tier is the only source
    CT4  boundary sweep of the mixed    chunk multiples +/-1 token
         path
    CT6  stride-boundary exactness      MEASURE, DO NOT ASSERT (never fails the run)

Hard rules honoured:
    * imports tierbench / equivbench / mixedbench, never modifies them
    * no reload / restart / unload / systemctl -- endpoint only
    * temperature=0 everywhere; never sends min_p (this build 400s on it under spec decode)
      -- every request goes through E.ask() or T.chat(), neither of which sends min_p
    * T.budget_check() is the wall-clock cap; CT4/CT6 pace their offsets to it

USAGE
    ./correctbench.py --test ct5 --dry-run      print the plan, send nothing
    ./correctbench.py --test ct5 --yes          negative controls only (no GPU work)
    ./correctbench.py --test all  --yes         CT5 first, then CT1-CT4 + CT6 (one cap)
    ./correctbench.py --test ct1  --yes         one test (CT5's gate still runs first)

Exit status: 0 only if CT5's gate passes and CT1-CT4 all PASS with a zero noise floor.
CT6 never affects the exit status.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tierbench as T       # noqa: E402
import equivbench as E      # noqa: E402
import mixedbench as M      # noqa: E402

OUT_DIR = os.environ.get("CORRECTBENCH_OUT", T.OUT_DIR)
PREFIX_TOKENS = int(os.environ.get("CORRECTBENCH_PREFIX_TOKENS", str(T.TARGET_PREFIX_TOKENS)))
HEAD_TOKENS = int(os.environ.get("CORRECTBENCH_HEAD_TOKENS", "30000"))
CT2_BODY_TOKENS = int(os.environ.get("CORRECTBENCH_BODY_TOKENS", "60000"))
CT2_LEAD_TOKENS = int(os.environ.get("CORRECTBENCH_LEAD_TOKENS", "8000"))
EST_OFFSET_S = int(os.environ.get("CORRECTBENCH_EST_OFFSET_S", "240"))

RUN = "%x" % (int(time.time()) & 0xFFFFFF)


# ---------------------------------------------------------------- helpers

def remaining_s():
    return max(0, T.RUNTIME_CAP_S - (time.time() - T._start))


def counters_zero(d):
    g = int(d.get("vllm:prefix_cache_hits_total", 0))
    x = int(d.get("vllm:external_prefix_cache_hits_total", 0))
    return g, x


def contention_guard(m0, m1):
    """Per-probe contention check (T-C). Snapshot a request-count metric immediately before
    and after each probe, and tag the probe CONTENDED if anything else ran. The probe
    itself is one in-flight request, so a co-tenant shows up as: a request already
    running before the probe (num_requests_running>0 at m0), one still running after
    (num_requests_running>0 at m1), or an extra completed request during the window
    (request_success_total delta > 1). Returns the parts plus the boolean.
    CONTENDED probes are recorded, never silently dropped."""
    r0 = m0.get("vllm:num_requests_running", 0.0)
    r1 = m1.get("vllm:num_requests_running", 0.0)
    s0 = m0.get("vllm:request_success_total", 0.0)
    s1 = m1.get("vllm:request_success_total", 0.0)
    extra = (s1 - s0) if (s0 is not None and s1 is not None) else 0.0
    contended = (r0 > 0) or (r1 > 0) or (extra > 1)
    return {"contended": bool(contended), "running_start": r0, "running_end": r1,
            "extra_requests": extra}


def probe(base, text, tag, salt):
    """One E.ask() bracketed by metric snapshots. Returns the record, the counter delta,
    the tier classification, and the per-tier counter split (M.classify_mixed)."""
    T.budget_check()
    m0 = T.snapshot(base)
    r = E.ask(base, text, tag, salt)
    m1 = T.snapshot(base)
    d = T.delta(m0, m1)
    kind, cnt = M.classify_mixed(d)
    return r, d, kind, cnt


def phase(base, text, tag, salt, expect=None):
    T.log("PHASE %s" % tag)
    r, d, k, c = probe(base, text, tag, salt)
    ok = expect is None or k.startswith(expect)
    T.log("  %s: served_by=%s valid=%s gpu_hits=%g ext_hits=%g"
          % (tag, k, ok, c["gpu_hits"], c["ext_hits"]))
    return {"phase": tag, "expected": expect, "served_by": k, "valid": ok,
            "counters": c, "probe": r, "metrics_delta": d}


def fresh_salt(prefix):
    return "%s-%s" % (prefix, RUN)


def compare_vs_ref(ref, phase_rec):
    c = E.compare(ref, phase_rec["probe"])
    ok = c["tokens_identical"] and c["max_abs_logprob_delta"] == 0.0
    return c, ok


def cold_pair(base, sizer, salt_a, salt_b, prefix_tokens):
    """coldA + coldB of one prompt; coldA-vs-coldB is the noise floor. Returns the
    prompt, both phase records, the noise comparison, and whether the floor is zero."""
    P, ptok = sizer.build(RUN, "prefix", prefix_tokens)
    T.log("prompt built: %s tokens" % "{:,}".format(ptok))
    ra = phase(base, P, "coldA", salt_a, "RECOMPUTE")
    rb = phase(base, P, "coldB", salt_b, "RECOMPUTE")
    noise = E.compare(ra["probe"], rb["probe"])
    noise_ok = noise["tokens_identical"] and noise["max_abs_logprob_delta"] == 0.0
    T.log("NOISE FLOOR coldA vs coldB: tokens_identical=%s  max|dlogprob|=%s"
          % (noise["tokens_identical"], noise["max_abs_logprob_delta"]))
    return P, ptok, ra, rb, noise, noise_ok


# ---------------------------------------------------------------- CT5

def ct5(base, sizer):
    """Negative controls. The pure-function controls need no GPU; one short repeated-salt
    request backs the endpoint control. The gate is the in-memory set: it must hold, and
    the discriminator controls (D1/D2) must hold, i.e. the instrument must be selective.
    A control that cannot fail is testing nothing."""
    res = []

    def add(name, held, evidence):
        res.append({"name": name, "held": bool(held) if held is not None else None,
                    "evidence": evidence})

    # A synthetic response record in exactly the shape E.compare consumes.
    r = {
        "tag": "ctrl",
        "tokens": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "logprobs": [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9],
        "top_logprobs": [[(10, -0.0), (9, -1.0)] for _ in range(10)],
        "text": "some text",
    }
    # Mutate: a token flip at index 5 and a logprob drift at index 3 (in the prefix).
    m = dict(r)
    m["tag"] = "ctrl-mut"
    m["tokens"] = r["tokens"][:5] + [999] + r["tokens"][6:]
    m["logprobs"] = r["logprobs"][:4] + [r["logprobs"][3] + 0.5] + r["logprobs"][5:]

    cm = E.compare(r, m)
    ci = E.compare(r, r)
    add("C1 compare(mutated) reports divergence",
        (not cm["tokens_identical"]
         and cm["first_divergent_token_index"] is not None
         and cm["max_abs_logprob_delta"] is not None
         and cm["max_abs_logprob_delta"] > 0.0),
        "tokens_identical=%s first=%s max|dlogprob|=%s"
        % (cm["tokens_identical"], cm["first_divergent_token_index"],
           cm["max_abs_logprob_delta"]))
    add("D1 compare(identical) is NOT flagged (selective)",
        ci["tokens_identical"] is True and ci["first_divergent_token_index"] is None,
        "tokens_identical=%s first=%s"
        % (ci["tokens_identical"], ci["first_divergent_token_index"]))

    # Hand-built metric deltas for classify().
    zero = {"vllm:prefix_cache_hits_total": 0.0,
            "vllm:external_prefix_cache_hits_total": 0.0,
            "vllm:kv_offload_load_bytes_total": 0.0}
    gpu = dict(zero); gpu["vllm:prefix_cache_hits_total"] = 1234.0
    ext = dict(zero)
    ext["vllm:external_prefix_cache_hits_total"] = 5678.0
    ext["vllm:kv_offload_load_bytes_total"] = 999999.0
    ext["vllm:kv_offload_fs_load_bytes_total"] = 12345.0
    cz, cg, ce = T.classify(zero), T.classify(gpu), T.classify(ext)
    add("C2 classify(zero-hit) -> RECOMPUTE", cz == "RECOMPUTE", repr(cz))
    add("C3 classify(gpu-hit) -> GPU", cg == "GPU", repr(cg))
    add("C4 classify(ext-hit) -> OFFLOAD", ce.startswith("OFFLOAD"), repr(ce))
    add("D2 classify is selective (gpu-hit is not RECOMPUTE)", cg != "RECOMPUTE",
        "gpu=%r  zero=%r" % (cg, cz))

    # Endpoint control: a deliberately repeated cache_salt must be detected as a hit.
    if base is not None:
        # > one 1648-token block (the cache only stores complete blocks), but still a few
        # seconds of prefill. 800 tokens = 0 blocks and would never be a hit (a control
        # that can never fail is testing nothing -- the exact trap this test exists for).
        short, _ = sizer.build(fresh_salt("ct5"), "ct5-short", 4000)
        s = fresh_salt("ct5-salt")
        probe(base, short, "ct5-1", s)          # first time
        m0 = T.snapshot(base)
        r2 = E.ask(base, short, "ct5-2", s)      # same salt, repeated
        m1 = T.snapshot(base)
        d2 = T.delta(m0, m1)
        hit = ((r2.get("cached_tokens") or 0) > 0
               or int(d2.get("vllm:prefix_cache_hits_total", 0)) > 0)
        add("C5 repeated cache_salt detected as a hit", hit,
            "cached_tokens=%s  gpu_delta=%d"
            % (r2.get("cached_tokens"), int(d2.get("vllm:prefix_cache_hits_total", 0))))
    else:
        add("C5 repeated cache_salt (endpoint control)", None,
            "endpoint unreachable; pure controls above still stand")

    # Gate = the in-memory controls, including the two discriminators that prove the
    # instrument is selective (and therefore able to fail).
    gate = all(any(x["name"].startswith(n) and x["held"] for x in res)
               for n in ("C1", "C2", "C3", "C4", "D1", "D2"))
    # "Prove it can fail": the equivalence assertion holds for a true hit (D1) but is
    # False for a broken one (C1); classify returns a tier only on a hit delta (C3 vs D2).
    teeth = (any(x["name"].startswith("C1") and x["held"] for x in res)
             and any(x["name"].startswith("D1") and x["held"] for x in res)
             and any(x["name"].startswith("C3") and x["held"] for x in res)
             and any(x["name"].startswith("D2") and x["held"] for x in res))
    return {"status": "PASS" if gate else "FAIL", "gate": gate, "teeth": teeth,
            "controls": res}


# ---------------------------------------------------------------- CT1

def ct1(base, sizer, cap, geom):
    salt_a, salt_b = fresh_salt("ct1-A"), fresh_salt("ct1-B")
    P, ptok, ra, rb, noise, noise_ok = cold_pair(base, sizer, salt_a, salt_b, PREFIX_TOKENS)
    if not noise_ok:
        T.log("CT1 VOID: noise floor is not zero; not a clean pass")
        return {"status": "INCONCLUSIVE", "void": True,
                "reason": "noise floor non-zero (tokens_identical=%s max|dlogprob|=%s)"
                         % (noise["tokens_identical"], noise["max_abs_logprob_delta"]),
                "noise": noise, "phases": [ra, rb]}

    evict_gpu = int(cap["gpu_tokens"] * T.EVICT_MARGIN)
    evict_all = int(cap["gpu_tokens"] * M.EVICT_ALL_MULT)
    phases = [ra, rb]

    gpu = phase(base, P, "gpu", salt_b, "GPU")          # immediate repeat of the B chain
    phases.append(gpu)

    T.drain(base, "ct1-pre-evict")
    T.evict(base, sizer, RUN, evict_gpu, "ct1-evictgpu")
    T.drain(base, "ct1-post-evict")
    off = phase(base, P, "offload", salt_b, "OFFLOAD")   # tier serves
    phases.append(off)

    # mixed: the two-prompt recipe from mixedbench -- evict everything, warm a head.
    T.evict(base, sizer, RUN, evict_all, "ct1-evictall")
    H, htok = M.head_prefix(sizer, P, HEAD_TOKENS)
    T.log("CT1 head: %s tokens warmed into the GPU" % "{:,}".format(htok))
    phase(base, H, "ct1-head-warm", salt_b)              # warms the head; not asserted
    mix = phase(base, P, "mixed", salt_b, "MIXED")
    phases.append(mix)

    # A phase that claims MIXED with one counter at zero did not test the mixed path.
    mixed_valid = mix["counters"]["gpu_hits"] > 0 and mix["counters"]["ext_hits"] > 0
    if not mixed_valid:
        T.log("CT1: MIXED not built (gpu_hits=%g ext_hits=%g) -- INCONCLUSIVE, never a pass"
              % (mix["counters"]["gpu_hits"], mix["counters"]["ext_hits"]))

    comps = {}
    for p in (gpu, off, mix):
        c, ok = compare_vs_ref(ra["probe"], p)
        comps[p["phase"]] = {"compare": c, "ok": ok}

    served_fail = [p["phase"] for p in phases if not p["valid"]]
    token_fail = [n for n, cc in comps.items() if not cc["ok"]]

    if not mixed_valid:
        return {"status": "INCONCLUSIVE",
                "reason": "MIXED not built (gpu_hits=%g ext_hits=%g)"
                         % (mix["counters"]["gpu_hits"], mix["counters"]["ext_hits"]),
                "noise": noise, "phases": phases, "comps": comps,
                "served_fail": served_fail, "token_fail": token_fail}
    if served_fail or token_fail:
        return {"status": "FAIL",
                "reason": "served_by fail=%s  token/logprob fail=%s" % (served_fail, token_fail),
                "noise": noise, "phases": phases, "comps": comps}
    return {"status": "PASS", "noise": noise, "phases": phases, "comps": comps,
            "head_tokens": htok}


# ---------------------------------------------------------------- CT2 / CT3

def _salt_pair(base, sizer, cap, geom, evict_before_b, tag):
    s_main = fresh_salt("%s-main" % tag)    # same salt for A and B-shadow (tests the chain)
    s_cold = fresh_salt("%s-cold" % tag)    # fresh salt for the independent recompute of B
    body = sizer.build(RUN, "%s-body" % tag, CT2_BODY_TOKENS)
    leadA = sizer.build(RUN, "%s-leadA" % tag, CT2_LEAD_TOKENS)
    leadB = sizer.build(RUN, "%s-leadB" % tag, CT2_LEAD_TOKENS)
    A_text = leadA[0] + " " + body[0]
    B_text = leadB[0] + " " + body[0]
    T.log("%s: A/B share a %s-token body, differ only in the leading block; "
          "s_main=%s  s_cold=%s" % (tag, "{:,}".format(CT2_BODY_TOKENS), s_main, s_cold))

    ra = phase(base, A_text, "A", s_main, "RECOMPUTE")
    T.drain(base, "%s-settle" % tag)

    if evict_before_b:
        if geom:
            cpu_tok = (cap["cpu_blocks"] // geom["n_groups"]) * geom["tokens_per_hash"]
        else:
            cpu_tok = 115360
        evict_tier = max(int(cpu_tok * T.EVICT_MARGIN),
                         int(cap["gpu_tokens"] * T.EVICT_MARGIN))
        T.log("%s: evicting past the CPU tier (%s tok) so the tier is the only source"
              % (tag, "{:,}".format(evict_tier)))
        T.evict(base, sizer, RUN, evict_tier, "%s-evict" % tag)
        T.drain(base, "%s-post-evict" % tag)

    rb = phase(base, B_text, "B-shadow", s_main)
    g, x = counters_zero(rb["metrics_delta"])
    counters_clean = (g == 0 and x == 0)
    T.log("%s: B-shadow counters  gpu_delta=%d  ext_delta=%d  -> %s"
          % (tag, g, x, "clean" if counters_clean else "CONTAMINATION"))
    rc = phase(base, B_text, "B-cold", s_cold, "RECOMPUTE")
    c = E.compare(rc["probe"], rb["probe"])
    out_ok = c["tokens_identical"] and c["max_abs_logprob_delta"] == 0.0
    T.log("%s: B-shadow vs B-cold  tokens_identical=%s  max|dlogprob|=%s"
          % (tag, c["tokens_identical"], c["max_abs_logprob_delta"]))

    rows = [ra, rb, rc]
    if not counters_clean:
        return {"status": "FAIL", "phases": rows, "compare": c,
                "gpu_delta": g, "ext_delta": x,
                "reason": "B got cache hits after A: gpu_delta=%d ext_delta=%d "
                         "(R3.15-class cross-prompt contamination)" % (g, x)}
    if not out_ok:
        return {"status": "FAIL", "phases": rows, "compare": c,
                "reason": "B-shadow output diverged from an independent cold B "
                         "(max|dlogprob|=%s)" % c["max_abs_logprob_delta"]}
    return {"status": "PASS", "phases": rows, "compare": c,
            "gpu_delta": g, "ext_delta": x}


def ct2(base, sizer, cap, geom):
    return _salt_pair(base, sizer, cap, geom, evict_before_b=False, tag="ct2")


def ct3(base, sizer, cap, geom):
    return _salt_pair(base, sizer, cap, geom, evict_before_b=True, tag="ct3")


# ---------------------------------------------------------------- CT4

def ct4(base, sizer, cap, geom):
    chunk = geom["tokens_per_hash"] if geom else 1648   # read, do not hard-code
    salt_a, salt_b = fresh_salt("ct4-A"), fresh_salt("ct4-B")
    P, ptok, ra, rb, noise, noise_ok = cold_pair(base, sizer, salt_a, salt_b, PREFIX_TOKENS)
    if not noise_ok:
        return {"status": "INCONCLUSIVE", "void": True, "chunk": chunk,
                "reason": "noise floor non-zero", "noise": noise, "offsets": []}

    evict_all = int(cap["gpu_tokens"] * M.EVICT_ALL_MULT)
    max_k = max(1, (ptok - 1) // chunk)               # largest k that still leaves a tail
    cand = set()
    for k in (1, 2, 3, 4):
        b = k * chunk
        for off in (b, b + 1, b - 1):
            if 1 <= off < ptok:
                cand.add(off)
    for off in (max_k * chunk, max_k * chunk + 1):
        if 1 <= off < ptok:
            cand.add(off)
    offsets = sorted(cand)
    T.log("CT4: chunk=%d  prompt=%s tok  candidate offsets (%d): %s"
          % (chunk, "{:,}".format(ptok), len(offsets),
             ", ".join("{:,}".format(o) for o in offsets)))

    results = []
    for off in offsets:
        if remaining_s() < EST_OFFSET_S:
            results.append({"offset": off, "status": "NOT RUN", "reason": "budget cap"})
            T.log("CT4: budget cap reached before offset %s" % "{:,}".format(off))
            continue
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        T.evict(base, sizer, RUN, evict_all, "ct4-evictall-%d" % off)
        H, htok = M.head_prefix(sizer, P, off)
        probe(base, H, "ct4-head-%d" % off, salt_b)
        m0 = T.snapshot(base)
        r = E.ask(base, P, "ct4-%d" % off, salt_b)
        m1 = T.snapshot(base)
        d = T.delta(m0, m1)
        kind, cnt = M.classify_mixed(d)
        cg = contention_guard(m0, m1)
        c = E.compare(ra["probe"], r)
        tok_ok = c["tokens_identical"] and c["max_abs_logprob_delta"] == 0.0
        alive = M.engine_alive(base)
        new_fatal = M.scan_fatal(M.engine_log_since(ts))
        mixed_ok = cnt["gpu_hits"] > 0 and cnt["ext_hits"] > 0
        ok = tok_ok and alive and not new_fatal and mixed_ok
        if not ok:
            st = "FAIL" if (not tok_ok or not alive or new_fatal) else "INCONCLUSIVE"
        else:
            st = "PASS"
        results.append({
            "offset": off, "head_tokens": htok, "served_by": kind, "status": st,
            "tokens_identical": c["tokens_identical"],
            "max_abs_dlogprob": c["max_abs_logprob_delta"],
            "engine_alive": alive, "new_fatal_lines": new_fatal,
            "gpu_hits": cnt["gpu_hits"], "ext_hits": cnt["ext_hits"],
            "contention": cg})
        T.log("CT4 off=%s: served=%s tok_identical=%s dlog=%s alive=%s fatal=%d "
              "gpu=%g ext=%g contended=%s -> %s" % ("{:,}".format(off), kind,
              c["tokens_identical"], c["max_abs_logprob_delta"], alive, len(new_fatal),
              cnt["gpu_hits"], cnt["ext_hits"], cg["contended"], st))

    reached = [x for x in results if x["status"] != "NOT RUN"]
    any_fail = any(x["status"] == "FAIL" for x in results)
    any_incon = any(x["status"] == "INCONCLUSIVE" for x in results)
    any_notrun = any(x["status"] == "NOT RUN" for x in results)
    if not reached:
        st = "INCONCLUSIVE"
    elif any_fail:
        st = "FAIL"
    elif any_notrun or any_incon:
        st = "INCONCLUSIVE"      # partial: some offsets never ran, or one was not mixed
    else:
        st = "PASS"
    return {"status": st, "chunk": chunk, "offsets_total": len(offsets),
            "offsets_reached": len(reached), "offsets": results}


def ct4_hammer(base, sizer, cap, geom, offsets, reps):
    """T-C: hammer the failed offsets ~reps times each, with a per-probe contention
    guard. Each rep: evict, warm a head, probe, and record the served state, the
    divergence, and whether a co-tenant was in flight. 1,649 was flaky at building a
    mixed state (head_prefix rounds it just under one block); a rep that degrades to
    OFFLOAD (gpu=0) is reported INCONCLUSIVE, not counted.

    Verdict: a divergence while UNCONTENDED is a residual boundary defect; divergences
    that occur only while CONTENDED are explained by batch composition (T-B); no
    divergence at all is consistent with the quiet re-probe."""
    chunk = geom["tokens_per_hash"] if geom else 1648
    salt_a, salt_b = fresh_salt("ham-A"), fresh_salt("ham-B")
    P, ptok, ra, rb, noise, noise_ok = cold_pair(base, sizer, salt_a, salt_b, PREFIX_TOKENS)
    if not noise_ok:
        return {"status": "INCONCLUSIVE", "void": True, "reason": "noise floor non-zero",
                "noise": noise, "offsets": []}
    evict_all = int(cap["gpu_tokens"] * M.EVICT_ALL_MULT)
    T.log("HAMMER: chunk=%d reps=%d offsets=%s" % (chunk, reps,
           ", ".join("{:,}".format(o) for o in offsets)))
    all_reps = []
    for off in offsets:
        off_reps = []
        for rep in range(1, reps + 1):
            if remaining_s() < EST_OFFSET_S:
                off_reps.append({"rep": rep, "status": "NOT RUN", "reason": "budget cap"})
                T.log("HAMMER off=%s rep=%d: budget cap" % ("{:,}".format(off), rep))
                break
            T.evict(base, sizer, RUN, evict_all, "ham-evict-%d-%d" % (off, rep))
            H, htok = M.head_prefix(sizer, P, off)
            probe(base, H, "ham-head-%d-%d" % (off, rep), salt_b)
            m0 = T.snapshot(base)
            r = E.ask(base, P, "ham-%d-%d" % (off, rep), salt_b)
            m1 = T.snapshot(base)
            d = T.delta(m0, m1)
            kind, cnt = M.classify_mixed(d)
            cg = contention_guard(m0, m1)
            c = E.compare(ra["probe"], r)
            diverged = not (c["tokens_identical"] and c["max_abs_logprob_delta"] == 0.0)
            inconc = cnt["gpu_hits"] == 0      # mixed state did not build (degraded)
            off_reps.append({
                "rep": rep, "served_by": kind, "head_tokens": htok,
                "diverged": diverged, "tokens_identical": c["tokens_identical"],
                "max_abs_dlogprob": c["max_abs_logprob_delta"],
                "first_divergent": c["first_divergent_token_index"],
                "gpu_hits": cnt["gpu_hits"], "ext_hits": cnt["ext_hits"],
                "contended": cg["contended"], "running_end": cg["running_end"],
                "inconclusive": inconc,
                "status": "INCONCLUSIVE" if inconc else ("FAIL" if diverged else "PASS")})
            T.log("HAMMER off=%s rep=%d: served=%s diverged=%s first=%s dlog=%s "
                  "gpu=%g ext=%g contended=%s inconc=%s -> %s" % (
                  "{:,}".format(off), rep, kind, diverged,
                  c["first_divergent_token_index"], c["max_abs_logprob_delta"],
                  cnt["gpu_hits"], cnt["ext_hits"], cg["contended"], inconc,
                  off_reps[-1]["status"]))
        all_reps.append({"offset": off, "reps": off_reps})
    any_div_u = any(rp.get("diverged") and not rp.get("contended")
                   for o in all_reps for rp in o["reps"] if "diverged" in rp)
    any_div_c = any(rp.get("diverged") and rp.get("contended")
                   for o in all_reps for rp in o["reps"] if "diverged" in rp)
    st = "RESIDUAL-DEFECT" if any_div_u else ("CONTENDED" if any_div_c else "PASS")
    for o in all_reps:
        ran = [rp for rp in o["reps"] if rp.get("status") != "NOT RUN"]
        div = [rp for rp in ran if rp.get("diverged")]
        o["summary"] = {
            "ran": len(ran), "divergent": len(div),
            "divergent_uncontended": len([rp for rp in div if not rp.get("contended")]),
            "divergent_contended": len([rp for rp in div if rp.get("contended")]),
            "inconclusive": len([rp for rp in ran if rp.get("inconclusive")])}
    return {"status": st, "chunk": chunk, "reps": reps, "offsets": all_reps}


# ---------------------------------------------------------------- CT6 (measure only)

def ct6(base, sizer, cap, geom):
    """MEASURE, DO NOT ASSERT. The N=8 Mamba stride store is approximate by design at
    non-boundary positions; a non-zero divergence here is a finding, not a failure."""
    chunk = geom["tokens_per_hash"] if geom else 1648
    stride = 8 * chunk
    salt_a, salt_b = fresh_salt("ct6-A"), fresh_salt("ct6-B")
    P, ptok, ra, rb, noise, noise_ok = cold_pair(base, sizer, salt_a, salt_b, PREFIX_TOKENS)
    evict_all = int(cap["gpu_tokens"] * M.EVICT_ALL_MULT)
    lens = set()
    for k in (1, 2, 3):
        b = k * stride
        for L in (b, b + 1, b - 1):
            if 1 <= L < ptok:
                lens.add(L)
    lens = sorted(lens)
    T.log("CT6: stride=%s tok  probe lengths (%d): %s"
          % ("{:,}".format(stride), len(lens),
             ", ".join("{:,}".format(L) for L in lens)))

    probes = []
    for L in lens:
        if remaining_s() < EST_OFFSET_S:
            probes.append({"prefix_len": L, "status": "NOT RUN", "reason": "budget cap"})
            continue
        T.evict(base, sizer, RUN, evict_all, "ct6-evictall-%d" % L)
        H, htok = M.head_prefix(sizer, P, L)
        probe(base, H, "ct6-head-%d" % L, salt_b)
        m0 = T.snapshot(base)
        r = E.ask(base, P, "ct6-%d" % L, salt_b)
        m1 = T.snapshot(base)
        d = T.delta(m0, m1)
        kind, cnt = M.classify_mixed(d)
        c = E.compare(ra["probe"], r)
        probes.append({
            "prefix_len": L, "multiple_of_stride": (L % stride == 0),
            "head_tokens": htok, "served_by": kind,
            "tokens_identical": c["tokens_identical"],
            "first_divergent": c["first_divergent_token_index"],
            "max_abs_dlogprob": c["max_abs_logprob_delta"],
            "gpu_hits": cnt["gpu_hits"], "ext_hits": cnt["ext_hits"],
            "status": "measured"})
        T.log("CT6 L=%s (mult=%s): dlog=%s tok_identical=%s"
              % ("{:,}".format(L), (L % stride == 0), c["max_abs_logprob_delta"],
                 c["tokens_identical"]))
    return {"status": "MEASURED", "stride": stride,
            "probes_total": len(lens),
            "probes_reached": sum(1 for p in probes if p["status"] == "measured"),
            "probes": probes, "noise_ok": noise_ok}


# ---------------------------------------------------------------- report

def _dlog(x):
    return "-" if x is None else "%.3e" % x


def write_report(path_md, path_json, meta):
    with open(path_json, "w") as f:
        json.dump(meta, f, indent=1, default=str)

    R = meta["results"]
    L = []
    L.append("# KV-cache correctness suite -- %s\n" % meta["started"])
    L.append("Run `%s`, endpoint `%s`.  \n" % (meta["run_id"], meta["endpoint"]))
    cap = meta["capacity"]; geom = meta.get("geometry")
    L.append("GPU %s tokens, CPU %s blocks%s; token block %s; card usage %s%%.  \n"
             % ("{:,}".format(cap["gpu_tokens"]), "{:,}".format(cap["cpu_blocks"]),
                " (%s)" % cap["source"],
                "{:,}".format(geom["tokens_per_hash"]) if geom else "n/a",
                meta.get("card_quiet_pct")))
    L.append("Prompt %s tokens, head %s tokens, max_tokens=%d, temperature=0, no min_p.  \n"
             % ("{:,}".format(meta["prefix_tokens"]), "{:,}".format(meta["head_tokens"]),
                E.MAX_TOKENS))
    L.append("Wall-clock cap %d s (T.budget_check). Plan: %s.\n"
             % (meta["runtime_cap_s"], ", ".join(meta["plan"])))

    L.append("\n## Verdict\n")
    L.append("| test | status | note |")
    L.append("|---|---|---|")
    for t in ("ct5", "ct1", "ct2", "ct3", "ct4", "ct6"):
        r = R.get(t, {})
        note = r.get("reason") or ("gate=%s teeth=%s" % (r.get("gate"), r.get("teeth"))
                                  if t == "ct5" else "")
        L.append("| %s | **%s** | %s |" % (t.upper(), r.get("status", "NOT RUN"), note))
    if meta.get("fatal_log_lines"):
        L.append("\n**engine fatal log lines: %d**\n" % len(meta["fatal_log_lines"]))
    L.append("- engine alive at end: **%s**\n" % ("yes" if meta.get("engine_alive") else "**NO**"))

    def comp_row(c):
        return ("| %s | %s | %d/%d | %s | %s |"
                % (c["pair"],
                   "**yes**" if c["tokens_identical"] else "**NO**",
                   c["matching_prefix_tokens"], max(c["n_tokens"]),
                   "-" if c["first_divergent_token_index"] is None
                   else "token %d" % c["first_divergent_token_index"],
                   _dlog(c["max_abs_logprob_delta"])))

    # CT5
    c5 = R.get("ct5", {})
    if c5:
        L.append("\n## CT5 -- negative controls\n")
        L.append("| control | held | evidence |")
        L.append("|---|---|---|")
        for x in c5.get("controls", []):
            h = x["held"]
            L.append("| %s | %s | %s |" % (x["name"],
                                         "-" if h is None else ("**yes**" if h else "**NO**"),
                                         x["evidence"]))
        L.append("\nGate: **%s**.  Proves-it-can-fail (discriminating): **%s**.\n"
                 % ("PASS" if c5.get("gate") else "**FAIL**",
                    "yes" if c5.get("teeth") else "**NO**"))
        if c5.get("controls"):
            c1 = next((x for x in c5["controls"] if x["name"].startswith("C1")), None)
            d1 = next((x for x in c5["controls"] if x["name"].startswith("D1")), None)
            L.append("Identical control: %s.  Mutated control: %s.  -> the equivalence "
                    "assertion holds for a true hit and FAILS for a broken one.\n"
                    % (d1["evidence"] if d1 else "-", c1["evidence"] if c1 else "-"))

    # CT1
    c1 = R.get("ct1")
    if c1:
        L.append("\n## CT1 -- path equivalence\n")
        n = c1.get("noise")
        L.append("Noise floor (coldA vs coldB): tokens_identical=%s max|dlogprob|=%s.\n"
                 % (n["tokens_identical"] if n else "-",
                    _dlog(n["max_abs_logprob_delta"]) if n else "-"))
        if c1.get("phases"):
            L.append("\n| phase | expected | served | valid | gpu_hits | ext_hits |")
            L.append("|---|---|---|---|---|---|")
            for p in c1["phases"]:
                if "counters" not in p:
                    continue
                L.append("| %s | %s | %s | %s | %g | %g |"
                         % (p["phase"], p["expected"] or "-", p["served_by"],
                            "yes" if p["valid"] else "**NO**",
                            p["counters"]["gpu_hits"], p["counters"]["ext_hits"]))
        if c1.get("comps"):
            L.append("\n| pair | tokens identical | matching prefix | first divergence | max abs dlogprob |")
            L.append("|---|---|---|---|---|")
            for name, cc in c1["comps"].items():
                L.append(comp_row(cc["compare"]))

    # CT2 / CT3
    for t in ("ct2", "ct3"):
        r = R.get(t)
        if not r:
            continue
        L.append("\n## %s -- salt isolation%s\n" % (t.upper(),
              " (evict first)" if t == "ct3" else ""))
        L.append("B-shadow counters after A: gpu_delta=%s ext_delta=%s.  "
                 "B-shadow vs B-cold: tokens_identical=%s max|dlogprob|=%s.\n"
                 % (r.get("gpu_delta"), r.get("ext_delta"),
                    r.get("compare", {}).get("tokens_identical"),
                    _dlog(r.get("compare", {}).get("max_abs_logprob_delta"))))

    # CT4
    c4 = R.get("ct4")
    if c4:
        L.append("\n## CT4 -- boundary sweep\n")
        L.append("chunk=%s  reached %d/%d offsets.\n"
                 % ("{:,}".format(c4.get("chunk", 0)), c4.get("offsets_reached", 0),
                    c4.get("offsets_total", 0)))
        L.append("| offset | head tok | served | status | tok identical | max dlog | alive | new fatal | gpu | ext |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for x in c4.get("offsets", []):
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % ("{:,}".format(x["offset"]),
                        "{:,}".format(x.get("head_tokens", 0)) if x.get("head_tokens") is not None else "-",
                        x.get("served_by", "-"), x["status"],
                        "-" if x.get("tokens_identical") is None else
                        ("yes" if x["tokens_identical"] else "**NO**"),
                        _dlog(x.get("max_abs_dlogprob")),
                        "-" if x.get("engine_alive") is None else x["engine_alive"],
                        len(x["new_fatal_lines"]) if x.get("new_fatal_lines") is not None else "-",
                        x.get("gpu_hits", "-"), x.get("ext_hits", "-")))

    # CT6
    c6 = R.get("ct6")
    if c6:
        L.append("\n## CT6 -- stride-boundary exactness (MEASURE, DO NOT ASSERT)\n")
        L.append("stride=%s tok  reached %d/%d probe lengths.  A non-zero divergence is a "
                 "finding, not a failure.\n"
                 % ("{:,}".format(c6.get("stride", 0)), c6.get("probes_reached", 0),
                    c6.get("probes_total", 0)))
        L.append("| prefix len | multiple of stride | head tok | served | tok identical | first div | max dlog |")
        L.append("|---|---|---|---|---|---|---|")
        for x in c6.get("probes", []):
            L.append("| %s | %s | %s | %s | %s | %s | %s |"
                     % ("{:,}".format(x["prefix_len"]),
                        "-" if x.get("multiple_of_stride") is None
                        else ("yes" if x["multiple_of_stride"] else "no"),
                        "{:,}".format(x.get("head_tokens", 0)) if x.get("head_tokens") is not None else "-",
                        x.get("served_by", "-"),
                        "-" if x.get("tokens_identical") is None else
                        ("yes" if x["tokens_identical"] else "**NO**"),
                        "-" if x.get("first_divergent") in (None, ) and x.get("status") != "measured"
                        else (x.get("first_divergent") if x.get("first_divergent") is not None else "-"),
                        _dlog(x.get("max_abs_dlogprob"))))

    if meta.get("fatal_log_lines"):
        L.append("\n## Engine fatal log lines\n")
        L.append("```")
        L.extend(meta["fatal_log_lines"][:40])
        L.append("```")

    with open(path_md, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="all",
                    choices=["all", "ct1", "ct2", "ct3", "ct4", "ct5", "ct6", "hammer"])
    ap.add_argument("--reps", type=int, default=20,
                    help="reps per offset for --test hammer (default 20)")
    ap.add_argument("--offset", default="",
                    help="comma-separated offsets for --test hammer "
                         "(default: the two CT4 failures, chunk+1 and 54*chunk)")
    ap.add_argument("--yes", action="store_true",
                    help="required: these phases push big prefills and eviction traffic")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect endpoint/capacity/geometry, print the plan, send nothing")
    args = ap.parse_args()

    base = T.detect_endpoint()
    cap = T.detect_capacity()
    geom = T.detect_geometry()
    if geom:
        T._BLOCK_BYTES[0] = geom["block_file_bytes"]
    sizer = T.Sizer(base)
    quiet = T.snapshot(base).get("vllm:kv_cache_usage_perc")
    T.log("endpoint %s" % base)
    T.log("capacity (%s): GPU %s tokens, CPU %s blocks; kv usage %s%%"
          % (cap["source"], "{:,}".format(cap["gpu_tokens"]),
             "{:,}".format(cap["cpu_blocks"]), quiet))
    T.log("geometry: %s" % (("%d groups x %s B per %d-token block"
          % (geom["n_groups"], "{:,}".format(geom["block_file_bytes"]),
             geom["tokens_per_hash"])) if geom else "unavailable"))
    T.log("card quiet: %s%%" % quiet)

    plan = ["ct5"]
    for t in ("ct1", "ct2", "ct3", "ct4", "ct6"):
        if args.test in ("all", t):
            plan.append(t)
    T.log("plan (in order, under one %d s cap): %s" % (T.RUNTIME_CAP_S, ", ".join(plan)))

    if args.dry_run:
        T.log("dry run: nothing sent.")
        return 0
    if not args.yes:
        T.log("refusing to run without --yes (big prefills + eviction traffic)")
        return 2

    os.makedirs(OUT_DIR, exist_ok=True)
    started_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = {
        "run_id": RUN, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_utc": started_ts, "endpoint": base, "capacity": cap,
        "geometry": geom, "card_quiet_pct": quiet, "prefix_tokens": PREFIX_TOKENS,
        "head_tokens": HEAD_TOKENS, "runtime_cap_s": T.RUNTIME_CAP_S,
        "tests_requested": args.test, "plan": plan, "results": {},
    }
    rc = 0

    def mark_not_run(t, reason):
        if t not in meta["results"]:
            meta["results"][t] = {"status": "NOT RUN", "reason": reason}

    try:
        c5 = ct5(base, sizer)
        meta["results"]["ct5"] = c5
        T.log("CT5: status=%s gate=%s teeth=%s" % (c5["status"], c5["gate"], c5["teeth"]))
        if not c5["gate"]:
            T.log("GATE FAILED: the instrument did not pass its negative controls; "
                 "CT1-CT4 are not trustworthy and are NOT RUN")
            for t in ("ct1", "ct2", "ct3", "ct4"):
                mark_not_run(t, "CT5 gate failed")
            rc = 1
        else:
            for t, fn in (("ct1", ct1), ("ct2", ct2), ("ct3", ct3), ("ct4", ct4)):
                if args.test not in ("all", t):
                    continue
                T.budget_check()
                meta["results"][t] = fn(base, sizer, cap, geom)
                T.log("%s: status=%s" % (t.upper(), meta["results"][t]["status"]))
            for t in ("ct1", "ct2", "ct3", "ct4"):
                if args.test not in ("all", t):
                    continue
                st = meta["results"].get(t, {}).get("status")
                if st in ("FAIL", "INCONCLUSIVE"):
                    rc = 1
            if args.test == "hammer":
                T.budget_check()
                chunk = geom["tokens_per_hash"] if geom else 1648
                # the two CT4 offsets that failed: 1,649 (= chunk+1) and 88,992 (= 54*chunk)
                if args.offset:
                    offsets = sorted(set(int(x) for x in args.offset.split(",") if x.strip()))
                else:
                    offsets = [chunk + 1, 54 * chunk]
                T.log("HAMMER offsets: %s (reps=%d)"
                      % (", ".join("{:,}".format(o) for o in offsets), args.reps))
                meta["results"]["hammer"] = ct4_hammer(
                    base, sizer, cap, geom, offsets, args.reps)
                T.log("HAMMER: status=%s" % meta["results"]["hammer"]["status"])
                if meta["results"]["hammer"]["status"] in ("RESIDUAL-DEFECT", "FAIL"):
                    rc = 1
    except T.Abort as e:
        T.log("ABORT: %s" % e)
        meta["aborted"] = str(e)
        rc = 1
    except KeyboardInterrupt:
        T.log("interrupted")
        meta["aborted"] = "KeyboardInterrupt"
        rc = 130

    if meta.get("aborted"):
        for t in ("ct1", "ct2", "ct3", "ct4"):
            mark_not_run(t, "run aborted: %s" % meta["aborted"])

    # CT6 always runs when in scope and the engine is still alive; never fails the run.
    if args.test in ("all", "ct6") and meta.get("aborted") != "KeyboardInterrupt":
        if M.engine_alive(base):
            try:
                meta["results"]["ct6"] = ct6(base, sizer, cap, geom)
                T.log("CT6: measured (never fails the run)")
            except T.Abort as e:
                T.log("CT6 aborted: %s" % e)
                meta["results"]["ct6"] = {"status": "NOT RUN", "reason": str(e)}
        else:
            meta["results"]["ct6"] = {"status": "NOT RUN",
                                     "reason": "engine not alive at this point"}

    meta["engine_alive"] = M.engine_alive(base)
    meta["fatal_log_lines"] = M.scan_fatal(M.engine_log_since(started_ts))
    if meta["fatal_log_lines"] or not meta["engine_alive"]:
        T.log("!! engine fatal lines=%d alive=%s -- the engine may have been disturbed; "
              "NOT restarting anything" % (len(meta["fatal_log_lines"]), meta["engine_alive"]))
        rc = 1

    md = os.path.join(OUT_DIR, "CORRECT-%s.md" % RUN)
    js = os.path.join(OUT_DIR, "CORRECT-%s.json" % RUN)
    write_report(md, js, meta)
    T.log("wrote %s" % md)
    T.log("wrote %s" % js)
    T.log("VERDICT rc=%d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
