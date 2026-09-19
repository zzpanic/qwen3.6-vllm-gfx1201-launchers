#!/usr/bin/env python3
"""turnbench -- the multi-turn correctness gate, and the three-agent workload measurement.

WHY THIS EXISTS
Every state-cache gate we had was single-turn. Lazy GDN passed reaskbench seven times and still
corrupted chat from turn 5 -- its failure needs MULTI-TURN traffic at a high prefix-cache hit rate
(upstream: 73-77% hits vs ~8% in single-shot gates). And the tier's value has never been measured
on the workload it exists for: pat's "three long context agents at the same time". One harness
closes both holes. It is also the gate any recurrent-state precision change (fp16 ssm) must pass.

WHAT IT DOES
Three sessions (A, B, C), each an agent-shaped conversation that reads code files and answers
questions about them, growing ~14k tokens a turn to ~110k by turn 7. Content is the host's Python
standard library (deterministic; sha256 of every file used is recorded in the report).

  --exact (default)  SEQUENTIAL, one request in flight at a time, so batch shape cannot move
                     numerics.
    CACHED phase     round-robin A1 B1 C1 A2 B2 C2 ... each session under its own cache_salt.
                     Three sessions outgrow the 228,737-token GPU pool by the later rounds, so the
                     early turns are GPU hits and the later ones come back from the CPU tier --
                     both paths, inside real multi-turn conversations.
    COLD phase       every CACHED request is replayed with the IDENTICAL messages under a fresh
                     salt (a true cold prefill), AFTER the cached phase so the replays cannot
                     evict the sessions. Each turn is compared with its own cold twin.
    The history fed to turn k+1 is the CACHED run's own reply -- what an agent really does -- and
    the cold twin gets exactly the same text, so every comparison is prompt-identical.
  --concurrent       the three sessions run AT ONCE (3 > MAXSEQS 2: one queues -- deliberately,
                     that IS the workload). Measures served / tier share / recompute and health.
                     No cold comparison: concurrent batching changes numerics by itself.

PER-TURN RESULT (exact mode)
  path       COLD / GPU / TIER / MIXED, from the prefix-cache metric deltas
  exact      tokens identical AND every logprob bit-identical AND spec-decode counters equal
  drift      first divergent token index and max |dlogprob| when not exact
  health     empty reply, repeat loop (a 6-gram seen >= 5 times), cut off with no content
Verdict lines, kept apart on purpose:
  TIER paths   (TIER + MIXED)  must be EXACT -- a byte copy of an exact state cannot drift
  GPU path                     known OPEN issue at fp32: gpu-partial-hit-divergence.md
  HEALTH                       no empty replies, no loops, in either phase
  DRIFT TREND                  does divergence grow with turn number? (what lazy GDN looked like)

Sends no min_p, temperature 0, thinking OFF (so replies carry content into the history).
USAGE
  ./turnbench.py --dry-run            plan only: files, token targets, nothing sent
  ./turnbench.py --yes                exact gate, ~15 min
  ./turnbench.py --concurrent --yes   workload measurement, ~5 min
Run long modes under systemd-run --user (the harness guard kills background jobs at 10 GB free).
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tierbench as T       # noqa: E402
import equivbench as E      # noqa: E402

BASE = os.environ.get("TURNBENCH_BASE", "http://127.0.0.1:1234/upstream/qwen3.8-27b-vllm")
OUT_DIR = os.environ.get("TURNBENCH_OUT", "$HOME/audit/stress/turnbench")
CORPUS = os.environ.get("TURNBENCH_CORPUS", "/usr/lib/python3.12")
RUN = "%x" % (int(time.time()) & 0xFFFFFF)
SYSTEM = ("You are a careful code assistant working through a codebase one file at a time. "
          "Answer concisely and specifically; quote function names exactly.")
QUESTIONS = [
    "Summarise what these files do, in five bullet points.",
    "Which function in the newest file is the most complex, and why?",
    "Name two edge cases the newest file handles, and one it might miss.",
    "How does the newest file relate to the earlier ones?",
    "Suggest one refactor for the newest file and show the changed lines.",
    "Which module so far would be hardest to unit-test, and why?",
    "List every public function added this turn, one line each.",
]
SPEC = {"drafts": "vllm:spec_decode_num_drafts_total",
        "draft_tok": "vllm:spec_decode_num_draft_tokens_total",
        "accepted": "vllm:spec_decode_num_accepted_tokens_total"}


# ------------------------------------------------------------------------------ requests
def ask(messages, tag, salt, max_tokens):
    """equivbench.ask with a message list and thinking off; same record, so E.compare works."""
    payload = {"model": T.MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0, "logprobs": True, "top_logprobs": E.TOP_LOGPROBS,
               "cache_salt": salt, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=T.PER_REQ_TIMEOUT) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise T.Abort("%s: HTTP %s %s" % (tag, e.code, e.read().decode()[:400]))
    except Exception as e:
        raise T.Abort("%s: %r" % (tag, e))
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    lp = ((ch.get("logprobs") or {}).get("content")) or []
    usage = body.get("usage") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    return {"tag": tag, "cache_salt": salt, "wall_s": round(time.time() - t0, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish_reason": ch.get("finish_reason"), "content": content,
            "reasoning": reasoning, "text": reasoning + content,
            "tokens": [t.get("token") for t in lp], "logprobs": [t.get("logprob") for t in lp],
            "top_logprobs": [[(c.get("token"), c.get("logprob")) for c in (t.get("top_logprobs") or [])]
                             for t in lp]}


def measured(messages, tag, salt, max_tokens):
    """One request bracketed by metric snapshots (sequential mode only)."""
    m0 = T.snapshot(BASE)
    t_start = time.time()
    r = ask(messages, tag, salt, max_tokens)
    r["t_start"], r["t_end"] = t_start, time.time()
    m1 = T.snapshot(BASE)
    d = lambda k: m1.get(k, 0.0) - m0.get(k, 0.0)
    r["gpu_hits"] = d("vllm:prefix_cache_hits_total")
    r["ext_hits"] = d("vllm:external_prefix_cache_hits_total")
    r["computed"] = d("vllm:request_prefill_kv_computed_tokens_sum")
    r["load_bytes"] = d("vllm:kv_offload_load_bytes_total")
    r["spec"] = {s: d(k) for s, k in SPEC.items()}
    r["contended"] = (m0.get("vllm:num_requests_running", 0) > 0
                      or m1.get("vllm:num_requests_running", 0) > 0
                      or d("vllm:request_success_total") > 1)
    r["path"] = ("MIXED" if r["gpu_hits"] > 0 and r["ext_hits"] > 0 else
                 "GPU" if r["gpu_hits"] > 0 else "TIER" if r["ext_hits"] > 0 else "COLD")
    return r


# ------------------------------------------------------------------------------ health
def health(r):
    """Reply-level sickness, judged without a reference. A loop is the TAIL repeating: the last
    48 tokens equal the 48 before them shifted by a period <= 24 -- the runaway shape lazy GDN
    produced. (An n-gram count false-positives on code listings and one-line-each lists.)"""
    toks = r["tokens"]
    flags = []
    if not r["content"].strip():
        flags.append("empty")
    for p in range(1, 25):
        if len(toks) >= 48 + p and toks[-48:] == toks[-48 - p:-p]:
            flags.append("loop(period %d)" % p)
            break
    return flags


# ------------------------------------------------------------------------------ corpus
def plan_turns(sizer, sessions, turns, t1, step):
    """Every turn's file bundle, sized with /tokenize BEFORE anything is timed. Files are dealt
    round-robin in name order so no two sessions share a file (or a prefix), then each turn is
    filled greedily to +-8% of its target. Returns ({session: [(text, tokens, [files])]}, sha)."""
    files = [f for f in sorted(glob.glob(os.path.join(CORPUS, "*.py")))
             if 2000 < os.path.getsize(f) < 60000]
    decks = {s: files[i::len(sessions)] for i, s in enumerate(sessions)}
    text = lambda f: "### %s\n```python\n%s\n```\n" % (os.path.basename(f), open(f).read())
    sha, tok = {}, {}
    for f in files:
        sha[os.path.basename(f)] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
        tok[f] = sizer.count(text(f)) if sizer else os.path.getsize(f) // 4
    out = {}
    for s in sessions:
        deck, out[s] = list(decks[s]), []
        for t in range(1, turns + 1):
            target = t1 if t == 1 else step
            got, pick = 0, []
            while got < target * 0.92:
                fit = [f for f in deck if got + tok[f] <= target * 1.08]
                if not fit and not deck:
                    raise T.Abort("corpus exhausted: session %s turn %d (%d of %d tokens)"
                                  % (s, t, got, target))
                f = fit[0] if fit else min(deck, key=lambda x: tok[x])
                deck.remove(f)
                pick.append(f)
                got += tok[f]
            out[s].append(("".join(text(f) for f in pick), got,
                           [os.path.basename(f) for f in pick]))
    return out, sha


def user_turn(turn, body):
    lead = "Here are some files from the codebase:" if turn == 1 else "Here are the next files:"
    return {"role": "user", "content": "%s\n\n%s\nQuestion: %s"
            % (lead, body, QUESTIONS[(turn - 1) % len(QUESTIONS)])}


# ------------------------------------------------------------------------------ runs
def run_exact(a, sessions, plan, stem):
    if a.resume:
        cached = json.load(open(a.resume))["rows"]
        T.log("PHASE cached: RESUMED %d turns from %s" % (len(cached), a.resume))
        return cold_phase(a, cached)
    conv = {s: [{"role": "system", "content": SYSTEM}] for s in sessions}
    salt = {s: "tb-%s-%s" % (s, RUN) for s in sessions}
    cached = []
    # --replay: send an earlier run's recorded messages verbatim (its own replies as history),
    # so a subset of sessions reproduces the exact prompts of the full run. Without it, a
    # --sessions subset is dealt DIFFERENT files (the deal is round-robin over the sessions).
    rec = None
    if a.replay:
        rec = {(r["session"], r["turn"]): r for r in json.load(open(a.replay))["rows"]}
        T.log("REPLAY messages from %s" % a.replay)
    T.log("PHASE cached: %d sessions x %d turns, round-robin, one request at a time"
          % (len(sessions), a.turns))
    for turn in range(1, a.turns + 1):
        for s in sessions:
            if rec is not None:
                conv[s] = json.loads(json.dumps(rec[(s, turn)]["messages"]))
            else:
                conv[s].append(user_turn(turn, plan[s][turn - 1][0]))
            msgs = json.loads(json.dumps(conv[s]))
            r = measured(msgs, "%s%d" % (s, turn), salt[s], a.max_tokens)
            r.update(session=s, turn=turn, messages=msgs, health=health(r))
            if rec is not None:
                was = rec[(s, turn)]
                r["replay"] = {"same_as_recorded": same(r, was)[0],
                               "recorded_path": was["path"], "recorded_gpu": was["gpu_hits"],
                               "recorded_ext": was["ext_hits"]}
            cached.append(r)
            T.log("  %s%d  %7s tok  %-5s gpu %7s ext %7s recomputed %7s  %5.1fs  %s%s"
                  % (s, turn, "{:,}".format(r["prompt_tokens"] or 0), r["path"],
                     "{:,}".format(int(r["gpu_hits"])), "{:,}".format(int(r["ext_hits"])),
                     "{:,}".format(int(r["computed"])), r["wall_s"],
                     ",".join(r["health"]) or "ok", "  CONTENDED" if r["contended"] else ""))
            conv[s].append({"role": "assistant", "content": r["content"]})
    # the cached phase cannot be re-run with the same salts (its prefixes are now cached), so it
    # is checkpointed: a crash or a question later replays only the cold side (--resume)
    json.dump({"run": RUN, "rows": cached}, open(stem + "-cached.json", "w"))
    T.log("checkpoint %s-cached.json" % stem)
    return cold_phase(a, cached)


def same(x, y):
    """E.compare, reduced to the verdict; never raises."""
    try:
        c = E.compare(x, y)
    except Exception as e:
        return False, {"error": repr(e)}
    ok = c["tokens_identical"] and c["logprobs_bit_identical"]
    return ok, c


def cold_phase(a, cached):
    """Each cached turn against a cold twin. When they differ, a SECOND cold twin decides who
    moved: cold2 == cold1 means the cold side is deterministic and the CACHED path diverged;
    cold2 != cold1 means the noise floor itself moved at this depth, and nothing is proven."""
    T.log("PHASE cold: replay every request, identical messages, fresh salt")
    if a.cold_tags:
        # only these turns get a cold twin (and only they are judged): keeps a byte-compare
        # run's disk writes small enough that the reaper never reaches the cached files
        cached = [r for r in cached if r["tag"] in a.cold_tags.split(",")]
        T.log("  cold twins only for %s" % [r["tag"] for r in cached])
    for r in cached:
        c = measured(r["messages"], r["tag"] + "-cold", "tbcold-%s-%s" % (r["tag"], RUN),
                     a.max_tokens)
        c["health"] = health(c)
        ok, cmp_ = same(c, r)
        spec_eq = r["spec"] == c["spec"]
        r["cold"] = {k: c[k] for k in ("path", "prompt_tokens", "spec", "health", "contended",
                                        "wall_s", "content")}
        r["compare"] = cmp_
        r["exact"] = ok and spec_eq
        r["spec_only"] = ok and not spec_eq
        note = ""
        if not ok:
            c2 = measured(r["messages"], r["tag"] + "-cold2", "tbcold2-%s-%s" % (r["tag"], RUN),
                          a.max_tokens)
            ok2, cmp2 = same(c2, c)
            r["cold2"] = {"path": c2["path"], "contended": c2["contended"], "spec": c2["spec"],
                          "same_as_cold": ok2, "compare": cmp2,
                          "same_as_cached": same(c2, r)[0]}
            note = ("  cold2==cold -> CACHED PATH DIVERGED" if ok2 else
                    "  cold2!=cold -> NOISE FLOOR MOVED (cold2==cached: %s)"
                    % r["cold2"]["same_as_cached"])
        T.log("  %-4s %-5s %s  first-div %s  max|dlp| %s  accepted %d/%d%s%s"
              % (r["tag"], r["path"], "EXACT" if r["exact"] else
                 ("SPEC " if r["spec_only"] else "DIFF "),
                 cmp_.get("first_divergent_token_index"),
                 "%.2e" % (cmp_.get("max_abs_logprob_delta") or 0),
                 r["spec"]["accepted"], c["spec"]["accepted"],
                 "  COLD-NOT-COLD(%s)" % c["path"] if c["path"] != "COLD" else "", note))
    return cached


def run_concurrent(a, sessions, plan):
    conv = {s: [{"role": "system", "content": SYSTEM}] for s in sessions}
    out, lock = [], threading.Lock()

    def agent(s):
        for turn in range(1, a.turns + 1):
            conv[s].append(user_turn(turn, plan[s][turn - 1][0]))
            try:
                r = ask(json.loads(json.dumps(conv[s])), "%s%d" % (s, turn),
                        "tbc-%s-%s" % (s, RUN), a.max_tokens)
            except T.Abort as e:
                T.log("  %s%d ABORT %s" % (s, turn, e))
                return
            r.update(session=s, turn=turn, health=health(r))
            conv[s].append({"role": "assistant", "content": r["content"]})
            with lock:
                out.append(r)
                T.log("  %s%d  %7s tok  cached %7s  %5.1fs  %s"
                      % (s, turn, "{:,}".format(r["prompt_tokens"] or 0),
                         "{:,}".format(r["cached_tokens"] or 0), r["wall_s"],
                         ",".join(r["health"]) or "ok"))

    m0 = T.snapshot(BASE)
    T.log("PHASE concurrent: %d agents at once (MAXSEQS 2 -- one queues, deliberately)"
          % len(sessions))
    ths = [threading.Thread(target=agent, args=(s,)) for s in sessions]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    m1 = T.snapshot(BASE)
    d = lambda k: m1.get(k, 0.0) - m0.get(k, 0.0)
    tot = {"prompt": sum(r["prompt_tokens"] or 0 for r in out),
           "gpu": d("vllm:prefix_cache_hits_total"),
           "ext": d("vllm:external_prefix_cache_hits_total"),
           "load_bytes": d("vllm:kv_offload_load_bytes_total"),
           "wall_s": round(time.time() - t0, 1), "completed": len(out)}
    return out, tot


# ------------------------------------------------------------------------------ verdict
def verdict_exact(rows):
    lines, rc = [], 0
    by = collections.defaultdict(list)
    for r in rows:
        by[r["path"]].append(r)
    contended = [r["tag"] for r in rows if r["contended"] or r["cold"]["contended"]]
    notcold = [r["tag"] for r in rows if r["cold"]["path"] != "COLD"]
    if contended or notcold:
        lines.append("INCONCLUSIVE contended %s, cold twin not cold %s" % (contended, notcold))
        rc = 2

    def judge(name, group, note=""):
        bad = [r["tag"] for r in group if not r["exact"] and not r.get("spec_only")
               and (r.get("cold2") or {}).get("same_as_cold", True)]
        noise = [r["tag"] for r in group if not r["exact"] and r.get("cold2")
                 and not r["cold2"]["same_as_cold"]]
        spec = [r["tag"] for r in group if r.get("spec_only")]
        lines.append("%-12s %s: %d of %d turns exact%s%s%s"
                     % (name, "PASS" if not (bad or spec) else "FAIL" + note,
                        sum(1 for r in group if r["exact"]), len(group),
                        "" if not bad else " -- CACHED PATH DIVERGED %s" % bad,
                        "" if not spec else " -- tokens exact, spec counters differ %s" % spec,
                        "" if not noise else " -- cold not deterministic (unproven) %s" % noise))
        return 1 if (bad or spec) else (2 if noise else 0)

    tier = by["TIER"] + by["MIXED"]
    if tier:
        rc = rc or judge("TIER paths", tier)
    else:
        lines.append("TIER paths   INCONCLUSIVE: no turn was served from the tier")
        rc = rc or 2
    if by["GPU"]:
        rc = rc or judge("GPU path", by["GPU"], " (open fp32 issue: gpu-partial-hit-divergence.md)")
    if by["COLD"]:
        rc = rc or judge("COLD turns", by["COLD"], " (cold vs cold: the NOISE FLOOR moved)")
    # a sick reply only blames the cache when its cold twin, same prompt, is healthy
    cache_sick = [(r["tag"], r["health"]) for r in rows if r["health"] and not r["cold"]["health"]]
    model_sick = [(r["tag"], r["cold"]["health"]) for r in rows if r["cold"]["health"]]
    lines.append("HEALTH       %s: cache-caused %s | both phases (model, not cache) %s"
                 % ("PASS" if not cache_sick else "FAIL", cache_sick or "none",
                    model_sick or "none"))
    rc = rc or (1 if cache_sick else 0)
    trend = [(r["tag"], r["compare"].get("first_divergent_token_index")) for r in rows
             if not r["exact"]]
    lines.append("DRIFT TREND  %s" % (trend or "none -- every turn exact"))
    return lines, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan the turns (sizes them with /tokenize, no inference) and stop")
    ap.add_argument("--concurrent", action="store_true")
    ap.add_argument("--sessions", default="ABC")
    ap.add_argument("--turns", type=int, default=7)
    ap.add_argument("--t1", type=int, default=24000)
    ap.add_argument("--step", type=int, default=14000)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--resume", metavar="TURN-x-cached.json",
                    help="skip the cached phase; replay the cold side of a checkpoint")
    ap.add_argument("--replay", metavar="TURN-x-cached.json",
                    help="send that run's recorded messages (fresh salts) instead of new ones")
    ap.add_argument("--cold-tags", metavar="A5,A7",
                    help="cold twins only for these turns (the rest are sent but not judged)")
    a = ap.parse_args()
    sessions = list(a.sessions)
    T.log("endpoint %s | %s | run %s" % (BASE, "CONCURRENT" if a.concurrent else
                                         "EXACT (sequential)", RUN))
    try:
        plan, sha = plan_turns(T.Sizer(BASE), sessions, a.turns, a.t1, a.step)
    except T.Abort as e:
        T.log("ABORT: %s" % e)
        return 3
    for s in sessions:
        cum = 0
        row = []
        for t, (_, n, names) in enumerate(plan[s], 1):
            cum += n + (a.max_tokens if t > 1 else 0)
            row.append("%d:+%dk=%dk(%d files)" % (t, n // 1000, cum // 1000, len(names)))
        T.log("  session %s  %s" % (s, "  ".join(row)))
    last = sum(sum(n for _, n, _ in plan[s]) for s in sessions)
    T.log("  all sessions at the last turn ~%s tokens (GPU pool 228,737; tier ~488k)"
          % "{:,}".format(last))
    if a.dry_run or not a.yes:
        T.log("dry run -- pass --yes to send")
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, "TURN-%s%s" % (RUN, "-conc" if a.concurrent else ""))
    meta = {"run": RUN, "args": vars(a), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_sha256": sha,
            "plan": {s: [(n, names) for _, n, names in plan[s]] for s in sessions}}
    try:
        if a.concurrent:
            rows, tot = run_concurrent(a, sessions, plan)
            sick = [(r["tag"], r["health"]) for r in rows if r["health"]]
            p = max(tot["prompt"], 1)
            lines = ["COMPLETED    %d of %d turns in %.0f s" % (tot["completed"],
                                                               len(sessions) * a.turns, tot["wall_s"]),
                     "SERVED       GPU %.1f%%, tier %.1f%%, recomputed %.1f%% of %s prompt tokens"
                     % (100 * tot["gpu"] / p, 100 * tot["ext"] / p,
                        100 * (p - tot["gpu"] - tot["ext"]) / p, "{:,}".format(tot["prompt"])),
                     "HEALTH       %s: %s (no cold twin in this mode)"
                     % ("PASS" if not sick else "FLAGGED", sick or "clean")]
            rc = 0 if tot["completed"] == len(sessions) * a.turns and not sick else 1
            meta["totals"] = tot
        else:
            rows = run_exact(a, sessions, plan, stem)
            lines, rc = verdict_exact(rows)
    except T.Abort as e:
        T.log("ABORT: %s" % e)
        return 3
    for r in rows:
        for k in ("messages", "tokens", "logprobs", "top_logprobs"):
            r.pop(k, None)
    json.dump({"meta": meta, "verdict": lines, "rows": rows}, open(stem + ".json", "w"), indent=1)
    with open(stem + ".md", "w") as f:
        f.write("# turnbench %s (%s)\n\n" % (RUN, "concurrent" if a.concurrent else "exact"))
        f.write("\n".join("    " + l for l in lines) + "\n\n")
        if not a.concurrent:
            f.write("| turn | prompt | path | gpu hit | tier hit | recomputed | exact | first div "
                    "| max abs dlogprob | accepted cached/cold | health cached/cold |\n")
            f.write("|---|--:|---|--:|--:|--:|---|--:|--:|---|---|\n")
            for r in rows:
                c = r["compare"]
                f.write("| %s | %s | %s | %s | %s | %s | %s | %s | %.2e | %d/%d | %s/%s |\n"
                        % (r["tag"], "{:,}".format(r["prompt_tokens"] or 0), r["path"],
                           "{:,}".format(int(r["gpu_hits"])), "{:,}".format(int(r["ext_hits"])),
                           "{:,}".format(int(r["computed"])),
                           "yes" if r["exact"] else "**no**", c.get("first_divergent_token_index"),
                           c.get("max_abs_logprob_delta") or 0, r["spec"]["accepted"],
                           r["cold"]["spec"]["accepted"], ",".join(r["health"]) or "ok",
                           ",".join(r["cold"]["health"]) or "ok"))
    for l in lines:
        T.log("VERDICT " + l)
    T.log("wrote %s.md / .json" % stem)
    return rc


if __name__ == "__main__":
    sys.exit(main())
