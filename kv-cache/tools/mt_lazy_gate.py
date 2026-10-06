#!/usr/bin/env python3
"""mt_lazy_gate -- the multi-turn, high-prefix-reuse gate lazy GDN failed (and reaskbench missed).

WHY THIS EXISTS. Lazy GDN corrupted multi-turn chat (ggz14 0cadf57: 35 empty replies and a repeat
loop in 50 turns, first failure at turn 5 with only ~3k tokens of context), and every single-turn
gate read clean -- reaskbench passed seven times. The failure needs (a) multi-turn conversations
whose every turn is a prefix-cache hit on the last, and (b) physical blocks recycled across
requests, so a stash header written by one request can be met by another. This gate builds both.

HOW.
  * A synthetic ~3k-token "registry" of records (codename, port, owner, region, status day), made
    from a fixed seed, so every question has ONE checkable answer: corruption shows as a wrong
    fact, an empty reply or a loop -- not merely "different text".
  * N conversations x T turns, interleaved ONE REQUEST AT A TIME (c1t1 c2t1 ... c1t2 ...): every
    turn extends its conversation's previous prompt (a prefix hit), and blocks freed by one
    request are handed to the next, which is the recycling the bug needs. Batch size stays 1, so
    a clean run is deterministic and two runs can be compared token for token.
  * cache_salt is unique per run: runs cannot hit each other's cached state (GPU, RAM or disk
    tier), but each run still hits its own prefixes every turn. Same tokens, isolated caches.

VERDICT (exit status of a run): 0 if every turn is healthy (non-empty, no loop, finished) and the
fact accuracy is >= --min-accuracy; 1 otherwise. `--compare A.json B.json` reports token-identical
turns and the first divergence of each conversation (exit 0 when all turns are identical).

USAGE
  mt_lazy_gate.py --model qwen3.8-27b-vllm --label lazyoff-a --out /tmp/mt-lazyoff-a.json
  mt_lazy_gate.py --compare /tmp/mt-lazyoff-a.json /tmp/mt-lazyoff-b.json
  mt_lazy_gate.py --copy 8 --max-tokens 1400 --label copy-a --out /tmp/mt-copy-a.json   # long turns
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request

WORDS = ["AZURE", "FALCON", "CEDAR", "QUARTZ", "EMBER", "TUNDRA", "SABLE", "NIMBUS", "ORCHID",
         "COBALT", "VESPER", "JUNIPER", "ONYX", "HALCYON", "MERIDIAN", "PIKE", "LARCH", "SOLACE",
         "TALON", "BRAMBLE", "CINDER", "GLACIER", "HARBOR", "IVORY", "KESTREL", "LUMEN", "MARLIN"]
TEAMS = ["Kestrel", "Osprey", "Heron", "Wren", "Plover", "Curlew", "Merlin", "Avocet"]
REGIONS = ["north-1", "east-2", "south-3", "west-4", "core-5"]


def record_line(r):
    return (f"Record {r['id']:03d}: service {r['name']} listens on port {r['port']}, is owned by "
            f"team {r['team']}, runs in region {r['region']}, and has been in its current state "
            f"since day {r['day']}.")


def registry(seed, n):
    rnd = random.Random(seed)
    names, recs = set(), []
    while len(recs) < n:
        name = f"{rnd.choice(WORDS)}-{rnd.choice(WORDS)}"
        if name in names or name.split("-")[0] == name.split("-")[1]:
            continue
        names.add(name)
        recs.append({"id": len(recs) + 1, "name": name, "port": rnd.randint(2000, 9899),
                     "team": rnd.choice(TEAMS), "region": rnd.choice(REGIONS),
                     "day": rnd.randint(2, 89)})
    lines = [record_line(r) for r in recs]
    doc = "SERVICE REGISTRY (authoritative; answer only from this text)\n\n" + "\n".join(lines)
    return doc, recs


def questions(recs, conv, turns, seed, copy=0):
    rnd = random.Random(seed * 1000 + conv)
    out = []
    if copy:
        # --copy N: every turn copies N records back verbatim, so each answer is several hundred
        # tokens and the conversation crosses GDN cache-block boundaries DURING decode -- where a
        # stale stash would be replayed. Checked exactly: every copied line must be present.
        for _ in range(turns):
            lo = rnd.randint(1, len(recs) - copy + 1)
            q = (f"Copy records {lo:03d} to {lo + copy - 1:03d} from the registry verbatim, one per "
                 f"line, nothing else.")
            out.append((q, [record_line(recs[i - 1]) for i in range(lo, lo + copy)]))
        return out
    for _ in range(turns):
        r = rnd.choice(recs)
        kind = rnd.choice(["port", "team", "region", "day"])
        q = {"port": f"Which port does {r['name']} listen on?",
             "team": f"Which team owns {r['name']}?",
             "region": f"Which region does {r['name']} run in?",
             "day": f"Since which day has {r['name']} been in its current state?"}[kind]
        out.append((q + " Answer in one short sentence.", str(r[kind])))
    return out


def chat(base, model, messages, salt, max_tokens, timeout):
    body = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "seed": 1234, "max_tokens": max_tokens, "cache_salt": salt,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.load(resp)
    ch = d["choices"][0]
    u = d.get("usage") or {}
    return {"content": ch["message"].get("content") or "",
            "reasoning": ch["message"].get("reasoning") or ch["message"].get("reasoning_content") or "",
            "finish": ch.get("finish_reason"), "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "wall_s": round(time.time() - t0, 2)}


def looped(text, n=6, reps=4):
    """A word n-gram repeated `reps`+ times: the repeat-loop signature."""
    w = text.split()
    if len(w) < n * reps:
        return False
    seen = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= reps:
            return True
    return False


def copy_loop(text):
    ls = [x.strip() for x in text.splitlines() if x.strip().startswith("Record ")]
    return len(ls) != len(set(ls))


def run(a):
    doc, recs = registry(a.seed, a.records)
    salt = f"mtgate-{a.label}-{int(time.time())}"
    convs = []
    for c in range(a.convs):
        sysmsg = {"role": "system", "content": f"You are assistant #{c + 1}. Be terse and exact."}
        convs.append({"messages": [sysmsg], "qs": questions(recs, c, a.turns, a.seed, a.copy), "turns": []})
    healthy = correct = total = 0
    for t in range(a.turns):
        for c, cv in enumerate(convs):
            q, want = cv["qs"][t]
            if t == 0:
                cv["messages"].append({"role": "user", "content": doc + "\n\n" + q})
            else:
                cv["messages"].append({"role": "user", "content": q})
            r = chat(a.base, a.model, cv["messages"], salt, a.max_tokens, a.timeout)
            text = r["content"].strip()
            r.update({"conv": c + 1, "turn": t + 1, "question": q, "expected": want,
                      "empty": text == "",
                      # copy turns repeat the registry's phrasing by design, so the n-gram test would
                      # flag every one -- in the content AND in a reasoning pass that rehearses the records
                      # first (2026-10-06 soak: 2 false positives) -- so there a loop is a
                      # copied line that comes back twice
                      "loop": (copy_loop(text) or copy_loop(r["reasoning"])) if a.copy
                              else looped(r["content"] + " " + r["reasoning"]),
                      "correct": (all(x in re.sub(r"[`*|]", "", text) for x in want) if isinstance(want, list)
                                  else re.search(rf"\b{re.escape(want)}\b", text) is not None)})
            r["healthy"] = (not r["empty"]) and (not r["loop"]) and r["finish"] == "stop"
            healthy += r["healthy"]; correct += r["correct"]; total += 1
            cv["turns"].append(r)
            cv["messages"].append({"role": "assistant", "content": r["content"]})
            flag = "" if r["healthy"] else "  <-- UNHEALTHY"
            print(f"c{c + 1} t{t + 1:02d} cached {r['cached_tokens']!s:>6}/{r['prompt_tokens']!s:>6} "
                  f"{r['wall_s']:5.1f}s {'ok ' if r['correct'] else 'BAD'} "
                  f"{text[:70]!r}{flag}", flush=True)
    acc = correct / total
    rep = {"label": a.label, "model": a.model, "salt": salt, "records": a.records,
           "convs": a.convs, "turns_per_conv": a.turns, "healthy": healthy, "total": total,
           "accuracy": acc, "conversations": [cv["turns"] for cv in convs]}
    json.dump(rep, open(a.out, "w"), indent=1)
    ok = healthy == total and acc >= a.min_accuracy
    print(f"VERDICT {'PASS' if ok else 'FAIL'}: healthy {healthy}/{total}, accuracy {acc:.1%} "
          f"(min {a.min_accuracy:.0%}) -> {a.out}")
    return 0 if ok else 1


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    same = tot = 0
    for ca, cb in zip(A["conversations"], B["conversations"]):
        first = None
        for ta, tb in zip(ca, cb):
            tot += 1
            if ta["content"] == tb["content"] and ta["reasoning"] == tb["reasoning"]:
                same += 1
            elif first is None:
                first = ta["turn"]
        print(f"conv {ca[0]['conv']}: first divergence at turn {first if first else '-'}")
    print(f"COMPARE {A['label']} vs {B['label']}: {same}/{tot} turns token-identical "
          f"(accuracy {A['accuracy']:.1%} vs {B['accuracy']:.1%}, healthy {A['healthy']} vs "
          f"{B['healthy']})")
    return 0 if same == tot else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://127.0.0.1:1234")
    p.add_argument("--model", default="qwen3.8-27b-vllm")
    p.add_argument("--label", default="run")
    p.add_argument("--out")
    p.add_argument("--records", type=int, default=80)
    p.add_argument("--convs", type=int, default=4)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--min-accuracy", type=float, default=0.9)
    p.add_argument("--copy", type=int, default=0, metavar="N",
                   help="long turns: copy N records back verbatim each turn (raise --max-tokens)")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = p.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    if not a.out:
        p.error("--out is required for a run")
    sys.exit(run(a))
