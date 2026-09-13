#!/usr/bin/env python3
"""One-screen live KV-cache summary, sized for `watch -n 5`.

    watch -n 5 python3 kv-cache/tools/kvwatch.py

Read-only: two GETs per refresh (the engine /metrics, and llama-swap per-request
activity). It never writes to the engine and names no state change.

WHERE IT READS FROM (environment overrides)
  KVWATCH_METRICS   engine /metrics. Default: the qwen3.8-27b-kvcache entry through
                    llama-swap on :1234. To read vLLM directly, use
                    http://127.0.0.1:<port>/metrics.
  KVWATCH_ACTIVITY  llama-swap's per-request activity feed. Optional: without
                    llama-swap the RECENT REQUESTS table is simply not printed.
  KVWATCH_REQUESTS  how many recent requests to list (default 8).

WHAT IT READS, AND WHY THAT MATTERS
Everything here comes from metrics UPSTREAM vLLM exports, so it keeps working
with the house instrumentation patches absent (KVOFF_MINIMAL=1). It therefore
answers "is the cache working" and never "why did that miss happen" -- the
counters that answer the second question are the ones the minimal set drops.

TWO TRAPS BUILT INTO THE DISPLAY
1. Every total here is a lifetime counter. A lifetime hit rate on a long-lived
   server is dominated by history and barely moves, so the RATE column (delta
   since the previous refresh) is the one to read. Deltas are kept in a state
   file, so the first refresh after starting shows no rate -- that is expected,
   not a fault.
2. `load_time` / `store_time` are engine-side timers whose wall-clock fidelity
   we have NOT verified on this build (a sibling histogram in the house patches
   was thread-summed and overstated wall clock about 7x). The derived MB/s is
   printed with a `~` and must not be quoted as a measured device rate.
"""
import json, os, re, sys, time, urllib.request

METRICS = os.environ.get(
    "KVWATCH_METRICS",
    "http://127.0.0.1:1234/upstream/qwen3.8-27b-kvcache/metrics")
ACTIVITY = os.environ.get("KVWATCH_ACTIVITY",
                          "http://127.0.0.1:1234/api/metrics/activity")
STATE = os.environ.get("KVWATCH_STATE", "/tmp/kvwatch-%d.json" % os.getuid())
NREQ = int(os.environ.get("KVWATCH_REQUESTS", "8"))

SAMPLE = re.compile(r'^(vllm:[a-zA-Z_0-9]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$')


def get(url, timeout=4):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse(body):
    """Sum each family across labels. Absent name -> absent key, never 0:
    a counter that never incremented is never exported, so the caller decides
    whether absence means zero or means not-instrumented."""
    out = {}
    for line in body.splitlines():
        if not line or line[0] == "#":
            continue
        m = SAMPLE.match(line.strip())
        if not m:
            continue
        try:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            pass
    return out


def human(n, unit="B"):
    if n is None:
        return "n/a"
    for s, d in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= d:
            return "%.1f %s%s" % (n / d, s, unit)
    return "%.0f %s" % (n, unit)


def pct(hit, q):
    if not q:
        return "   --  "
    return "%6.1f%%" % (100.0 * hit / q)


def main():
    now = time.time()
    try:
        m = parse(get(METRICS))
    except Exception as e:
        print("kvwatch: cannot read %s\n  %s" % (METRICS, e))
        return 1
    acts = []
    try:
        d = json.loads(get(ACTIVITY))
        acts = d if isinstance(d, list) else (d.get("data") or [])
    except Exception:
        pass

    prev, dt = {}, None
    try:
        with open(STATE) as fh:
            saved = json.load(fh)
        prev, dt = saved.get("m", {}), now - saved.get("t", now)
    except Exception:
        pass

    def v(k):
        return m.get(k)

    def delta(k):
        if k in m and k in prev:
            return m[k] - prev[k]
        return None

    G = lambda k: (v(k) or 0.0)

    # llama-swap URLs name the entry; a direct engine URL has only host:port.
    label = (METRICS.split("/upstream/")[1].split("/")[0] if "/upstream/" in METRICS
             else METRICS.split("://")[-1].split("/")[0])
    print("KV CACHE  %s     %s   %s" % (
        label,
        time.strftime("%H:%M:%S"),
        ("delta over %.0fs" % dt) if dt and dt > 0.5 else "delta: first sample"))
    print("  running %-3.0f waiting %-3.0f  pool %5.1f%%  preemptions %-5.0f" % (
        G("vllm:num_requests_running"), G("vllm:num_requests_waiting"),
        100.0 * G("vllm:kv_cache_usage_perc"), G("vllm:num_preemptions_total")))

    print("\nPREFIX CACHE            lifetime                   this refresh")
    for label, hk, qk in (
            ("GPU          ", "vllm:prefix_cache_hits_total",
             "vllm:prefix_cache_queries_total"),
            ("offload tier ", "vllm:external_prefix_cache_hits_total",
             "vllm:external_prefix_cache_queries_total")):
        h, q = v(hk), v(qk)
        dh, dq = delta(hk), delta(qk)
        rate = pct(dh, dq) if (dh is not None and dq) else "   --  "
        seen = ("+%s tok" % format(int(dq), ",")) if dq else "idle"
        print("  %s %s  (%s / %s)   %s  %s" % (
            label, pct(h, q), human(h or 0, ""), human(q or 0, ""), rate, seen))

    print("\nOFFLOAD TIER")
    cpu = v("vllm:kv_offload_cpu_cache_usage_perc")
    if cpu is not None:
        # These three are INSTANTANEOUS occupancy of the tier transfer buffers,
        # not how full the CPU tier is. They sit near zero between transfers and
        # that is correct, not a broken gauge -- do not read 0.0% as an empty
        # tier. The tier FILL level (free_perc) came from the house tier-report
        # patch and is not exported under KVOFF_MINIMAL=1; judge the tier by the
        # load/store bytes below instead.
        print("  xfer buffers  %5.2f%% busy  read %4.2f%%  write %4.2f%%"
              "   (in-flight, NOT tier fill)" % (
                  100 * cpu,
                  100 * G("vllm:kv_offload_cpu_cache_read_usage_perc"),
                  100 * G("vllm:kv_offload_cpu_cache_write_usage_perc")))
    for label, bk, tk in (("load ", "vllm:kv_offload_load_bytes_total",
                           "vllm:kv_offload_load_time_total"),
                          ("store", "vllm:kv_offload_store_bytes_total",
                           "vllm:kv_offload_store_time_total")):
        b, t, db = v(bk), v(tk), delta(bk)
        if b is None:
            print("  %s        not exported" % label)
            continue
        rate = ("~%s/s" % human(b / t)) if t else "   --  "
        moved = ("+%s" % human(db)) if db else "idle"
        print("  %s  %10s in %7.2fs   %-12s   %s" % (
            label, human(b), t or 0.0, rate, moved))
    n = v("vllm:kv_offload_lookup_async_delay_seconds_count")
    s = v("vllm:kv_offload_lookup_async_delay_seconds_sum")
    if n:
        print("  deferred lookups  n=%-6.0f mean wait %.3fs" % (n, s / n))

    if acts:
        print("\nRECENT REQUESTS (llama-swap; timestamp is the END of the request)")
        print("   ended     input     cached      miss    hit%   ttft*    dur")
        for r in acts[-NREQ:]:
            t = r.get("tokens") or {}
            it, ct = t.get("input_tokens") or 0, t.get("cache_tokens") or 0
            pps = t.get("prompt_per_second") or 0
            ttft = (it - ct) / pps if pps > 0 else 0
            flag = "  <-- MISS" if it > 20000 and ct < it * 0.5 else ""
            print("  %s %9s %9s %9s %6.1f%% %6.1fs %6.1fs%s" % (
                str(r.get("timestamp"))[11:19], format(it, ","),
                format(ct, ","), format(it - ct, ","),
                (100.0 * ct / it) if it else 0.0, ttft,
                (r.get("duration_ms") or 0) / 1000.0, flag))
        print("  * ttft is derived as (input-cached)/prompt_per_second, because"
              " llama-swap divides by the")
        print("    uncached tokens -- it is not an independently reported number.")

    try:
        tmp = STATE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"t": now, "m": m}, fh)
        os.replace(tmp, STATE)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
