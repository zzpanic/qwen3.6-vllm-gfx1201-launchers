#!/usr/bin/env python3
"""kvtable.py -- print the KV-cache tier table live from an engine's /metrics.

Stdlib only. Method (reproduced, not the numbers):
  GPU = prefix_cache_hits/prompt; L2 = tier_hit{fs}/prompt; L1 = (ext - fs)/prompt
  recompute = computed/(prefill_time - stall); rate = served/load_seconds;
  density = load_bytes/served. The fs load_seconds is wall-clock whole-job time
  in the radiance build (wall-clock reanchored patch applied; *_thread_seconds
  carries the thread-summed signal), so the fs rate is a measurement, not a floor.
  Capacities come from the metrics (cache_config_info / capacity_bytes), never
  from constants. Nothing is model- or config-specific."""

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict

GIB = 1024.0 ** 3
KIB = 1024.0

# Declared metric names. prometheus_client appends _total to every Counter's
# wire name, so `sum` tries the bare name then the _total form. Histogram
# components carry _bucket / _sum / _count and are cumulative (differenced).
M = {
    "prompt": "vllm:prompt_tokens",
    "gpu_hits": "vllm:prefix_cache_hits",
    "ext_hits": "vllm:external_prefix_cache_hits",
    "tier_hit": "vllm:kv_offload_tier_hit_tokens",
    "load_bytes": "vllm:kv_offload_tier_load_bytes",
    "load_secs": "vllm:kv_offload_tier_load_seconds",
    "load_lat": "vllm:kv_offload_tier_load_latency_seconds",
    "cap": "vllm:kv_offload_tier_capacity_bytes",
    "used": "vllm:kv_offload_tier_used_bytes",
    "occ": "vllm:kv_offload_tier_occupancy_ratio",
    "stall": "vllm:kv_offload_tier_stall_seconds",
    "prefill_time": "vllm:request_prefill_time_seconds",
    "computed": "vllm:request_prefill_kv_computed_tokens",
    "cfg": "vllm:cache_config_info",
}

LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(.+)$")
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
CUM = ("_total", "_bucket", "_sum", "_count")


def parse_exposition(text):
    """Prometheus text format -> [(name, labels_dict, value)]. Label-aware on
    purpose: summing across labels would collapse the tier dimension."""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "#":
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, lp, val = m.group(1), m.group(2), m.group(3)
        try:
            v = float(val.split()[0])
        except ValueError:
            continue
        labels = {}
        if lp:
            for k, vs in LABEL_RE.findall(lp):
                labels[k] = vs.replace('\\"', '"').replace("\\\\", "\\")
        samples.append((name, labels, v))
    return samples


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


class Scrape:
    """One scrape, queryable by name+label, optionally as a difference.
    `base` is an earlier Scrape (window mode). Counters/histogram components
    are cumulative so they are differenced; gauges are levels so the current
    value stands. The split is done in exactly one place."""

    def __init__(self, samples, base=None, ts=None):
        self.ts = ts or time.time()
        self.idx = defaultdict(list)
        for n, l, v in samples:
            self.idx[n].append((l, v))
        self.base = base
        self.win = (max(0.0, self.ts - base.ts) if base else None)

    def _series(self, name):
        return self.idx.get(name, [])

    def _base(self, name, labels):
        if self.base is None:
            return 0.0
        for l, v in self.base._series(name):
            if l == labels:
                return v
        return 0.0

    def _cum(self, name):
        return name.endswith(CUM)

    def series(self, name, **match):
        out = []
        for l, v in self._series(name):
            if all(l.get(k) == x for k, x in match.items()):
                if self.base is not None and self._cum(name):
                    v = v - self._base(name, l)
                out.append((l, v))
        return out

    def sum(self, name, **match):
        """Sum a series. Tries the bare name then the prometheus _total form."""
        for cand in (name, name + "_total"):
            hits = self.series(cand, **match)
            if hits:
                return sum(v for _, v in hits)
        return 0.0

    def has(self, name):
        return bool(self._series(name) or self._series(name + "_total")
                    or self._series(name + "_bucket"))

    def label_values(self, name, label):
        out = set()
        for cand in (name, name + "_total", name + "_bucket"):
            for l, _ in self._series(cand):
                if label in l:
                    out.add(l[label])
        return out

    def config_labels(self, name):
        for l, _ in self._series(name):
            return l
        return {}

    def histogram(self, name, **match):
        buckets = {}
        for l, v in self.series(name + "_bucket", **match):
            le = l.get("le")
            if le is None:
                continue
            try:
                edge = float(le)
            except ValueError:
                edge = float("inf")
            buckets[edge] = buckets.get(edge, 0.0) + v
        return Histogram(buckets, self.sum(name + "_sum", **match),
                        self.sum(name + "_count", **match))


class Histogram:
    def __init__(self, buckets, total_sum, count):
        self.buckets = dict(sorted(buckets.items()))
        self.sum = total_sum
        self.count = count

    def __bool__(self):
        return self.count > 0

    def quantile(self, q):
        """Interpolated quantile from cumulative buckets (resolution = layout)."""
        if self.count <= 0 or not self.buckets:
            return None
        target = q * self.count
        pe, pc = 0.0, 0.0
        for e, c in self.buckets.items():
            if c >= target:
                if e == float("inf"):
                    return pe
                span = c - pc
                if span <= 0:
                    return e
                return pe + ((target - pc) / span) * (e - pe)
            pe, pc = e, c
        return pe

    def fraction_above(self, edge):
        if self.count <= 0:
            return None
        for e, c in self.buckets.items():
            if abs(e - edge) < 1e-12:
                return 1.0 - (c / self.count)
        return None


def sdiv(a, b):
    return (a / b) if b else None


def f_gib(b):
    return "%.1f GiB" % (b / GIB) if b else "-"


def f_kib(b):
    return "%.0f KB" % (b / KIB) if b else "-"


def f_tok(n):
    if n is None:
        return "-"
    n = float(n)
    if n >= 1e6:
        return "%.1fM" % (n / 1e6)
    if n >= 1e3:
        return "%.0fk" % (n / 1e3)
    return "%.0f" % n


def f_tps(n):
    if n is None:
        return "-"
    n = float(n)
    if n >= 1e6:
        return "%.2fM" % (n / 1e6)
    if n >= 1e3:
        return "%.0fk" % (n / 1e3)
    return "%.0f" % n


def f_rate(bps):
    return "%.1f GB/s" % (bps / 1e9) if bps is not None else "-"


def f_pct(p):
    return "%.1f%%" % (p * 100) if p is not None else "-"


def f_sec(s):
    if s is None:
        return "-"
    return ("%.2f s" % s) if s < 10 else ("%.1f s" % s)


def tier_row(s, tier):
    """One tier's measured cells (L1=cpu / L2=fs). Counters for rates; the fs
    load_seconds is wall-clock in the live build, so the rate is a measurement."""
    served = s.sum(M["tier_hit"], tier=tier)
    lbytes = s.sum(M["load_bytes"], tier=tier)
    lsecs = s.sum(M["load_secs"], tier=tier)
    cap = s.sum(M["cap"], tier=tier)
    used = s.sum(M["used"], tier=tier)
    lat = s.histogram(M["load_lat"], tier=tier)
    occ = s.histogram(M["occ"], tier=tier)
    density = sdiv(lbytes, served)
    return {
        "tier": tier,
        "served": served,
        "load_bytes": lbytes,
        "load_seconds": lsecs,
        "bytes_per_s": sdiv(lbytes, lsecs),
        "tokens_per_s": sdiv(served, lsecs),
        "bytes_per_token": density,
        "cap_bytes": cap,
        "used_bytes": used,
        "lat_p50": lat.quantile(0.5) if lat else None,
        "lat_p99": lat.quantile(0.99) if lat else None,
        "occ_above_95": occ.fraction_above(0.95) if occ else None,
    }


def derive(s, recompute_tps=None):
    prompt = s.sum(M["prompt"])
    gpu = s.sum(M["gpu_hits"])
    ext = s.sum(M["ext_hits"])
    computed = s.sum(M["computed"] + "_sum")
    prefill = s.sum(M["prefill_time"] + "_sum")
    stall = s.sum(M["stall"])

    note = []
    if recompute_tps is None:
        pcompute = prefill - stall
        if pcompute <= 0:
            pcompute = prefill
            if stall > 0:
                note.append("stall exceeded prefill time; recompute baseline "
                           "uses raw prefill wall time (pessimistic).")
        recompute = sdiv(computed, pcompute)
    else:
        recompute = float(recompute_tps)

    tier_names = set()
    for nm in ("tier_hit", "load_bytes", "cap"):
        tier_names |= s.label_values(M[nm], "tier")
    have_tier = s.has(M["tier_hit"]) or s.has(M["load_bytes"])
    has_fs = "fs" in tier_names
    has_cpu = "cpu" in tier_names

    cfg = s.config_labels(M["cfg"])
    try:
        gpu_cap_tok = float(cfg.get("kv_cache_size_tokens", 0)) or None
    except ValueError:
        gpu_cap_tok = None
    try:
        gpu_cap_bytes = float(cfg.get("kv_cache_memory_bytes", 0)) or None
    except ValueError:
        gpu_cap_bytes = None

    fs_hit = s.sum(M["tier_hit"], tier="fs") if has_fs else 0.0
    r = {
        "mode": "window" if s.win else "lifetime",
        "window_s": s.win,
        "prompt": prompt,
        "gpu_share": sdiv(gpu, prompt),
        "tier_total": ext,
        "l2_share": sdiv(fs_hit, prompt),
        "l1_share": sdiv(ext - fs_hit, prompt),
        "recompute_share": sdiv(computed, prompt),
        "recompute_tps": recompute,
        "have_tier": have_tier,
        "has_fs": has_fs,
        "has_cpu": has_cpu,
        "gpu_cap_tok": gpu_cap_tok,
        "gpu_cap_bytes": gpu_cap_bytes,
        "gpu_bpt": sdiv(gpu_cap_bytes, gpu_cap_tok),
        "cpu": tier_row(s, "cpu") if has_cpu else None,
        "fs": tier_row(s, "fs") if has_fs else None,
        "missing": [],
        "notes": note,
    }
    if not have_tier:
        r["missing"] = [k for k in ("tier_hit", "load_bytes", "load_secs",
                                   "load_lat", "cap", "used", "occ")
                       if not s.has(M[k])]
    return r


def build_table(r):
    """Assemble the README tier table. Column order: L0 GPU, then L1 RAM if a
    cpu tier is present, then L2 SSD if an fs tier is present, then recompute.
    Every row is [label] + [gpu] + ([cpu]) + ([fs]) + [rec] in that order."""
    cpu, fs, rec = r["cpu"], r["fs"], r["recompute_tps"]
    hc, hf = r["has_cpu"], r["has_fs"]

    def vs(t):
        return ("%.0fx" % (t / rec)) if (t is not None and rec) else "-"

    def howfull(row):
        parts = []
        if row and row["cap_bytes"] and row["used_bytes"] is not None:
            parts.append("%.0f of %.0f GiB" % (row["used_bytes"] / GIB,
                                              row["cap_bytes"] / GIB))
        if row and row["occ_above_95"]:
            parts.append(">95%% for %.1f%% of the time"
                        % (row["occ_above_95"] * 100))
        return " · ".join(parts) if parts else "-"

    def row(label, gpu, cp, fsv, rec_cell):
        cells = [label, gpu]
        if hc:
            cells.append(cp)
        if hf:
            cells.append(fsv)
        cells.append(rec_cell)
        return cells

    cols = ["L0 GPU"] + (["L1 RAM"] if hc else []) + (["L2 SSD"] if hf else [])
    cols.append("recompute")
    header = [""] + cols

    wl = ("last %.0fs" % r["window_s"]) if r["window_s"] else "lifetime"
    rows = []
    rows.append(row("Served, %s" % wl, f_pct(r["gpu_share"]),
                   f_pct(r["l1_share"]), f_pct(r["l2_share"]),
                   f_pct(r["recompute_share"])))
    rows.append(row("Capacity",
                   "%s tok (%s)" % (f_tok(r["gpu_cap_tok"]),
                                    f_gib(r["gpu_cap_bytes"])),
                   f_gib(cpu["cap_bytes"]) if cpu and cpu["cap_bytes"] else "-",
                   f_gib(fs["cap_bytes"]) if fs and fs["cap_bytes"] else "-",
                   "-"))
    rows.append(row("Bytes per token",
                   f_kib(r["gpu_bpt"]) if r["gpu_bpt"] else "-",
                   f_kib(cpu["bytes_per_token"]) if cpu and cpu["bytes_per_token"] else "-",
                   f_kib(fs["bytes_per_token"]) if fs and fs["bytes_per_token"] else "-",
                   "-"))
    rows.append(row("Moves data", "in place",
                   "RAM -> GPU at " + f_rate(cpu["bytes_per_s"]) if cpu else "-",
                   "SSD -> RAM at " + f_rate(fs["bytes_per_s"]) if fs else "-",
                   "-"))
    rows.append(row("Tokens/s equivalent", "-",
                   "~" + f_tps(cpu["tokens_per_s"]) if cpu else "-",
                   f_tps(fs["tokens_per_s"]) if fs else "-",
                   f_tps(rec)))
    rows.append(row("vs recompute", "-",
                   vs(cpu["tokens_per_s"]) if cpu else "-",
                   vs(fs["tokens_per_s"]) if fs else "-",
                   "1x" if rec else "-"))
    rows.append(row("Batch latency p50 / p99", "-",
                   "%s / %s" % (f_sec(cpu["lat_p50"]), f_sec(cpu["lat_p99"])) if cpu else "-",
                   "%s / %s" % (f_sec(fs["lat_p50"]), f_sec(fs["lat_p99"])) if fs else "-",
                   "-"))
    rows.append(row("How full", "always",
                   howfull(cpu) if cpu else "-",
                   howfull(fs) if fs else "-",
                   "-"))
    return header, rows


def render_md(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Print the KV-cache tier table live from /metrics.")
    ap.add_argument("--url",
                   default="http://127.0.0.1:8000/metrics",
                   help="engine /metrics endpoint (direct engine).")
    ap.add_argument("--window", type=float, default=None,
                   help="window in seconds (two scrapes, deltas); default lifetime.")
    ap.add_argument("--recompute-tps", type=float, default=None,
                   help="override the recompute baseline tok/s; without it the "
                        "vs-recompute row is computed from the metrics "
                        "(computed / (prefill_time - stall)).")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    a = ap.parse_args(argv)

    s0 = Scrape(parse_exposition(fetch(a.url)))
    if a.window:
        time.sleep(a.window)
        s = Scrape(parse_exposition(fetch(a.url)), base=s0, ts=time.time())
    else:
        s = s0

    r = derive(s, a.recompute_tps)

    if a.json:
        print(json.dumps(r, default=float, indent=2))
        return 0

    header, rows = build_table(r)
    print(render_md(header, rows))
    if r["mode"] == "window":
        print()
        print("_Window: last %.0fs (two scrapes, counter deltas)._ " % r["window_s"]
              + "_Lifetime includes all traffic since boot._")
    else:
        print()
        print("_Lifetime since boot; includes all traffic since the engine "
              "started._")
    for n in r["notes"]:
        print("_Note: %s_" % n)
    if r["missing"]:
        print()
        print("_No tier-labelled metrics (public default, KVOFF_MINIMAL=1). "
              "Missing series: %s. The L1/L2 split and per-tier rates are "
              "unavailable; GPU and recompute rows use the standard counters._"
              % ", ".join("vllm:kv_offload_tier_" + m for m in r["missing"]))
    if not r["recompute_tps"] and a.recompute_tps is None:
        print()
        print("_vs recompute unavailable: pass --recompute-tps N (e.g. the "
              "measured recompute tok/s) to fill it._")
    return 0


if __name__ == "__main__":
    sys.exit(main())
