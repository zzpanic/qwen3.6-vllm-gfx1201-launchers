#!/usr/bin/env python3
"""tierreport.py -- justify the size and speed of every KV cache tier, from one scrape.

WHAT THIS ANSWERS

Three questions an operator running a layered KV cache cannot answer today:

  1. Have I allocated too much (or too little) RAM to the CPU tier?
  2. Is my disk too slow -- should I buy NVMe?
  3. Is turning this layered cache on adding value at all?

It answers them from the operator's OWN traffic. There is no corpus, no replay,
no A/B, no restart and no GPU window: the tool scrapes `/metrics` once and
reports. That is deliberate. "Is my RAM the right size" depends entirely on the
prefix-reuse pattern of the real workload, and a synthetic benchmark cannot
reproduce that -- it would measure the benchmark's working set, not yours.

WHY IT IS ALL COUNTERS

Because it reads once, from a server that may have been up for weeks, every
judgment has to rest on a monotonic counter or a histogram. A gauge sampled that
way carries almost no information: `kv_offload_cpu_cache_free_perc` reading 0.0
says "full at this instant", not "full 90% of the time", and only the second is
a sizing statement. The two exceptions are capacity and used bytes, which are
configuration rather than judgment.

WHAT IT NEEDS

The per-tier series added by `patch_kv_offload_tier_report.py` (P1-P6 of
tier-report-metrics-plan.md). Without them the tool still runs and still reports
what it honestly can -- the whole-cache value calculation from
`prompt_tokens_by_source`, which needs no patch -- and says explicitly which
rows are missing and why. It never silently substitutes an aggregate for a
per-tier number: the unlabelled `kv_offload_load_bytes_total` divided by
`load_time_total` comes out at CPU-tier speed with the disk invisible inside it,
and reporting that as "the cache" is exactly the mistake this tool exists to
stop.

USAGE

  tierreport.py                                  # lifetime, table to stdout
  tierreport.py --out ./report                   # + results.json / results.md
  tierreport.py --snapshot before.json           # save a scrape for later
  tierreport.py --since before.json              # window instead of lifetime
  tierreport.py --calibrate /kvcache/blocks      # + measure the raw device rate

Lifetime is the default because that is the mode the questions are asked in.
`--since` exists for people who are benchmarking and want a window; it differs
two scrapes and reports the interval.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import re
import sys
import textwrap
import time
import urllib.request
from collections import defaultdict

# --------------------------------------------------------------------------
# Metric names. Kept in one place so a rename upstream is one edit here.
# --------------------------------------------------------------------------
TR = "vllm:kv_offload_tier_"
M_HIT_BLOCKS = TR + "hit_blocks"
M_HIT_TOKENS = TR + "hit_tokens"
M_LOAD_BYTES = TR + "load_bytes"
M_LOAD_SECONDS = TR + "load_seconds"
M_LOAD_OPS = TR + "load_ops"
M_LOAD_LATENCY = TR + "load_latency_seconds"
M_STORE_BYTES = TR + "store_bytes"
M_STORE_SECONDS = TR + "store_seconds"
M_STORE_OPS = TR + "store_ops"
M_STORE_LATENCY = TR + "store_latency_seconds"
M_CAPACITY = TR + "capacity_bytes"
M_USED = TR + "used_bytes"
M_OCCUPANCY = TR + "occupancy_ratio"
M_READS_BEFORE_EVICT = TR + "reads_before_evict"
M_EVICTIONS = TR + "evictions"
M_EVICTED_BYTES = TR + "evicted_bytes"
M_EVICTION_TO_REUSE = TR + "eviction_to_reuse_seconds"
M_MISS_EVICTED = TR + "lookup_miss_evicted"
M_STALL = TR + "stall_seconds"

M_BY_SOURCE = "vllm:prompt_tokens_by_source"
M_PROMPT_TOKENS = "vllm:prompt_tokens"
M_PREFILL_TIME = "vllm:request_prefill_time_seconds"
M_PREFILL_COMPUTED = "vllm:request_prefill_kv_computed_tokens"

GIB = float(1 << 30)
MIB = float(1 << 20)

# Thresholds. Every verdict is a threshold on a measured distribution rather
# than a judgment call, so they belong in one visible block: an operator who
# disagrees with one can change it and re-run.
T_NEVER_READ_FRACTION = 0.50   # mass at "0 reads" that means oversized
T_REUSE_P50_SECONDS = 60.0     # eviction-to-reuse p50 that means undersized
T_BREAK_EVEN = 1.2             # tier tok/s over recompute tok/s to be worth it

# Tiers whose I/O is dispatched to a worker pool rather than run inline.
#
# This matters more than it looks. For such a tier the per-batch seconds are
# accumulated on several threads at once and then summed, so dividing bytes by
# that sum understates the wall-clock rate by up to the pool width -- and every
# "your disk is too slow" verdict is a division by exactly that number. It also
# changes what the latency histogram MEANS: a batch that waits while its seven
# siblings share the device is not a device that is failing, it is the fanout
# doing precisely what it was built to do. This design queues deliberately
# rather than preempting and evicting, so a long per-batch time is the chosen
# trade, not a symptom.
#
# So a queued tier gets bounds and an explanation, never a verdict computed as
# if the seconds were wall time. Override with --serial-io for a backend that
# really does its reads inline.
QUEUED_IO_TIERS = {"fs"}
T_OCCUPANCY_HIGH = 0.95        # "full" for the occupancy-over-time statement
T_OCCUPANCY_HIGH_FRACTION = 0.80


# --------------------------------------------------------------------------
# Scrape and parse
# --------------------------------------------------------------------------
LINE_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})? +(.+)$')
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_exposition(text):
    """Prometheus text format -> [(name, labels_dict, value)].

    Label-aware on purpose. A parser that sums across labels -- which is what
    the existing bench helpers do, for their own good reasons -- would collapse
    exactly the `tier` dimension this entire report is about.
    """
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "#":
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, labelpart, value = m.group(1), m.group(2), m.group(3)
        try:
            val = float(value.split()[0])
        except ValueError:
            continue
        labels = {}
        if labelpart:
            for k, v in LABEL_RE.findall(labelpart):
                labels[k] = v.replace('\\"', '"').replace("\\\\", "\\")
        samples.append((name, labels, val))
    return samples


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


class Scrape:
    """One scrape, queryable by name and label, optionally as a difference.

    `base` is an earlier Scrape for --since mode. Counters and histogram
    components are cumulative, so they are differenced; gauges are levels, so
    the current value stands. Getting that split wrong is the classic way to
    produce a report that looks plausible and is wrong, so it is done in exactly
    one place.
    """

    CUMULATIVE_SUFFIXES = ("_total", "_bucket", "_sum", "_count")

    def __init__(self, samples, base=None, timestamp=None):
        self.timestamp = timestamp or time.time()
        self.index = defaultdict(list)
        for name, labels, val in samples:
            self.index[name].append((labels, val))
        self.base = None
        if base is not None:
            self.base = base
            self.window_seconds = max(0.0, self.timestamp - base.timestamp)
        else:
            self.window_seconds = None

    # -- raw access ------------------------------------------------------
    def _series(self, name):
        return self.index.get(name, [])

    def _base_value(self, name, labels):
        if self.base is None:
            return 0.0
        for lb, val in self.base._series(name):
            if lb == labels:
                return val
        return 0.0

    def _is_cumulative(self, name):
        return name.endswith(self.CUMULATIVE_SUFFIXES)

    def series(self, name, **match):
        """[(labels, value)] for `name`, filtered by exact label matches."""
        out = []
        for labels, val in self._series(name):
            if all(labels.get(k) == v for k, v in match.items()):
                if self.base is not None and self._is_cumulative(name):
                    val = val - self._base_value(name, labels)
                out.append((labels, val))
        return out

    def sum(self, name, **match):
        """Sum a series. Tries the bare name and the prometheus `_total` form.

        prometheus_client appends `_total` to every Counter's exposed name, so
        the name the patch declares and the name on the wire differ. Trying
        both means the report is written against the declared names and still
        reads a real endpoint.
        """
        for candidate in (name, name + "_total"):
            hits = self.series(candidate, **match)
            if hits:
                return sum(v for _, v in hits)
        return 0.0

    def has(self, name):
        return bool(self._series(name) or self._series(name + "_total")
                    or self._series(name + "_bucket"))

    def has_prefix(self, prefix):
        """True if ANY series starts with `prefix`.

        The presence test that matters is "was the patch applied", and that
        cannot be asked of one named series: prometheus_client registers a
        metric lazily, on first observation, so a tier that has stored but not
        yet been read back exposes its store-side series and none of its
        load-side ones. Asking for a load counter there answers "the patch is
        missing" about an engine that is running the patch, and the remedy that
        answer implies -- apply it and restart -- would destroy the very
        counters the report needs.
        """
        return any(name.startswith(prefix) for name in self.index)

    def label_values(self, name, label):
        out = set()
        for candidate in (name, name + "_total", name + "_bucket"):
            for labels, _ in self._series(candidate):
                if label in labels:
                    out.add(labels[label])
        return out

    # -- histograms ------------------------------------------------------
    def histogram(self, name, **match):
        """Cumulative bucket counts, sum and count for one histogram."""
        buckets = {}
        for labels, val in self.series(name + "_bucket", **match):
            le = labels.get("le")
            if le is None:
                continue
            try:
                edge = float(le)
            except ValueError:
                edge = float("inf")
            buckets[edge] = buckets.get(edge, 0.0) + val
        total = self.sum(name + "_count", **match)
        return Histogram(buckets, self.sum(name + "_sum", **match), total)


class Histogram:
    def __init__(self, buckets, total_sum, count):
        self.buckets = dict(sorted(buckets.items()))
        self.sum = total_sum
        self.count = count

    def __bool__(self):
        return self.count > 0

    def quantile(self, q):
        """Interpolated quantile from cumulative buckets.

        Bucket interpolation, not a real percentile -- the resolution is the
        bucket layout. That is stated wherever a number from here is printed,
        because a p99 quoted to three digits from eight buckets would be a lie
        of precision.
        """
        if self.count <= 0 or not self.buckets:
            return None
        target = q * self.count
        prev_edge, prev_count = 0.0, 0.0
        for edge, cum in self.buckets.items():
            if cum >= target:
                if edge == float("inf"):
                    return prev_edge
                span = cum - prev_count
                if span <= 0:
                    return edge
                frac = (target - prev_count) / span
                return prev_edge + frac * (edge - prev_edge)
            prev_edge, prev_count = edge, cum
        return prev_edge

    def fraction_at_or_below(self, edge):
        """Share of observations in the buckets up to `edge`."""
        if self.count <= 0:
            return None
        for e, cum in self.buckets.items():
            if abs(e - edge) < 1e-12:
                return cum / self.count
        return None

    def fraction_above(self, edge):
        below = self.fraction_at_or_below(edge)
        return None if below is None else 1.0 - below

    @property
    def mean(self):
        return self.sum / self.count if self.count else None


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------
def fmt_bytes(n):
    if n is None:
        return "-"
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= GIB:
        return "%s%.1f GiB" % (sign, n / GIB)
    if n >= MIB:
        return "%s%.0f MiB" % (sign, n / MIB)
    if n >= 1024:
        return "%s%.1f KiB" % (sign, n / 1024.0)
    return "%s%.0f B" % (sign, n)


def fmt_rate(bps):
    if bps is None:
        return "-"
    if bps >= GIB:
        return "%.2f GB/s" % (bps / 1e9)
    return "%.0f MB/s" % (bps / 1e6)


def fmt_secs(s):
    """Human seconds. Negatives matter here -- a net loss is a real answer, and
    a formatter that only handled positives would print the most important
    number in the report as microseconds."""
    if s is None:
        return "-"
    if s < 0:
        return "-" + fmt_secs(-s)
    if s < 1e-3:
        return "%.0f us" % (s * 1e6)
    if s < 1.0:
        return "%.1f ms" % (s * 1e3)
    if s < 120:
        return "%.2f s" % s
    if s < 7200:
        return "%.1f min" % (s / 60)
    return "%.1f h" % (s / 3600)


def fmt_num(n):
    return "-" if n is None else "{:,.0f}".format(n)


def fmt_pct(f):
    return "-" if f is None else "%.1f%%" % (100.0 * f)


def safe_div(a, b):
    if a is None or b in (None, 0) or b == 0.0:
        return None
    return a / b


def derive(scrape, serial_io=False):
    """Turn a scrape into the report structure. No printing, no verdicts.

    `serial_io` says this deployment really does run tier I/O inline, so the
    seconds counters are wall time and the rates can be stated flat. Leave it
    False for the pooled backends in QUEUED_IO_TIERS.
    """
    r = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "window" if scrape.base is not None else "lifetime",
        "window_seconds": scrape.window_seconds,
        # The patch is present if ANY tier-labelled series is; whether the
        # tiers have SERVED anything yet is a separate question, and conflating
        # the two tells a correctly-patched operator to restart.
        "tier_patch_present": scrape.has_prefix(TR),
        "have_tier_metrics": scrape.has(M_HIT_TOKENS) or scrape.has(M_LOAD_BYTES),
        "missing": [],
        "notes": [],
    }

    # ---- the no-patch layer: source attribution and the recompute baseline.
    by_source = {}
    for labels, val in scrape.series(M_BY_SOURCE + "_total"):
        src = labels.get("source")
        if src:
            by_source[src] = by_source.get(src, 0.0) + val
    r["by_source"] = by_source
    r["prompt_tokens"] = scrape.sum(M_PROMPT_TOKENS)

    prefill_time = scrape.sum(M_PREFILL_TIME + "_sum")
    computed_tokens = scrape.sum(M_PREFILL_COMPUTED + "_sum")
    stall_total = scrape.sum(M_STALL)
    # P5: prefill wall time includes waiting on a tier. Subtracting the stall
    # is what makes the recompute baseline -- the denominator of every
    # break-even ratio below -- a measure of COMPUTE rather than of compute
    # plus somebody else's disk.
    prefill_compute_time = prefill_time - stall_total
    if prefill_compute_time <= 0:
        prefill_compute_time = prefill_time
        if stall_total > 0:
            r["notes"].append(
                "Stall time exceeded total prefill time, so the recompute rate "
                "is computed from raw prefill wall time instead. That makes the "
                "recompute baseline pessimistic and every break-even ratio "
                "optimistic; it usually means the stall counter and the prefill "
                "histogram cover different uptimes (a --since window that "
                "starts mid-request will do it)."
            )
    r["recompute_tokens_per_s"] = safe_div(computed_tokens, prefill_compute_time)
    r["prefill_seconds"] = prefill_time
    r["prefill_stall_seconds"] = stall_total
    r["prefill_computed_tokens"] = computed_tokens

    if not r["have_tier_metrics"]:
        if r["tier_patch_present"]:
            r["missing"].append(
                "Per-tier instrumentation is applied, but no tier has served a "
                "read yet, so there are no per-tier rows to show. Do NOT "
                "restart: these counters are lifetime-cumulative and a restart "
                "resets them. Keep serving and re-run."
            )
        else:
            r["missing"].append(
                "Per-tier series absent. Apply patch_kv_offload_tier_report.py "
                "and restart the server. Without it there is no per-tier row at "
                "all: the engine's unlabelled load_bytes/load_time cover the CPU "
                "tier only, so dividing them describes memcpy speed and says "
                "nothing about the disk."
            )
        return r

    # ---- tiers -----------------------------------------------------------
    tier_names = set()
    for name in (M_HIT_TOKENS, M_LOAD_BYTES, M_STORE_BYTES, M_CAPACITY):
        tier_names |= scrape.label_values(name, "tier")
    # cpu first, then the secondaries in a stable order.
    ordered = ([t for t in ("cpu",) if t in tier_names]
               + sorted(t for t in tier_names if t != "cpu"))

    secondary = [t for t in ordered if t != "cpu"]
    hit_tokens = {t: scrape.sum(M_HIT_TOKENS, tier=t) for t in ordered}
    hit_blocks = {t: scrape.sum(M_HIT_BLOCKS, tier=t) for t in ordered}
    promoted_tokens = sum(hit_tokens.get(t, 0.0) for t in secondary)

    tiers = []
    for t in ordered:
        load_bytes = scrape.sum(M_LOAD_BYTES, tier=t)
        load_seconds = scrape.sum(M_LOAD_SECONDS, tier=t)
        store_bytes = scrape.sum(M_STORE_BYTES, tier=t)
        store_seconds = scrape.sum(M_STORE_SECONDS, tier=t)
        lat = scrape.histogram(M_LOAD_LATENCY, tier=t)
        store_lat = scrape.histogram(M_STORE_LATENCY, tier=t)
        occupancy = scrape.histogram(M_OCCUPANCY, tier=t)
        reads = scrape.histogram(M_READS_BEFORE_EVICT, tier=t)
        reuse = scrape.histogram(M_EVICTION_TO_REUSE, tier=t)

        served = hit_tokens.get(t, 0.0)
        # P1's promoting-tier subtraction. The CPU tier is counted at the
        # GPU-facing load, so blocks a secondary tier staged into it are
        # counted there too. Without this subtraction a disk doing all the work
        # reads as dead weight and the RAM tier takes the credit.
        if t == "cpu":
            originated = max(0.0, served - promoted_tokens)
        else:
            originated = served

        capacity = scrape.sum(M_CAPACITY, tier=t) or None
        used = scrape.sum(M_USED, tier=t) or None
        # Density measured, never assumed: it is model- and config-specific,
        # and the whole point of measuring it is that a hard-coded constant
        # would make the tool wrong on anybody else's deployment.
        density = safe_div(load_bytes, served) if served else None

        row = {
            "tier": t,
            "role": "primary" if t == "cpu" else "secondary",
            "hit_tokens_served": served,
            "hit_tokens_originated": originated,
            "hit_blocks": hit_blocks.get(t, 0.0),
            "load_bytes": load_bytes,
            "load_seconds": load_seconds,
            "load_ops": scrape.sum(M_LOAD_OPS, tier=t),
            "store_bytes": store_bytes,
            "store_seconds": store_seconds,
            "store_ops": scrape.sum(M_STORE_OPS, tier=t),
            # Lower bounds, not rates, whenever the seconds were summed
            # across a worker pool -- see QUEUED_IO_TIERS.
            "bytes_per_s": safe_div(load_bytes, load_seconds),
            "store_bytes_per_s": safe_div(store_bytes, store_seconds),
            "tokens_per_s": safe_div(served, load_seconds),
            "bytes_per_token": density,
            "load_latency_p50": lat.quantile(0.5) if lat else None,
            "load_latency_p99": lat.quantile(0.99) if lat else None,
            "store_latency_p50": store_lat.quantile(0.5) if store_lat else None,
            "capacity_bytes": capacity,
            "used_bytes": used,
            "occupancy_p50": occupancy.quantile(0.5) if occupancy else None,
            "occupancy_fraction_above_95": (
                occupancy.fraction_above(T_OCCUPANCY_HIGH) if occupancy else None
            ),
            "never_read_fraction": (
                reads.fraction_at_or_below(0.5) if reads else None
            ),
            "reads_before_evict_count": reads.count,
            "reads_before_evict_mean": reads.mean,
            "eviction_to_reuse_p50": reuse.quantile(0.5) if reuse else None,
            "eviction_to_reuse_count": reuse.count,
            "evictions": scrape.sum(M_EVICTIONS, tier=t),
            "evicted_bytes": scrape.sum(M_EVICTED_BYTES, tier=t),
            "miss_evicted": scrape.sum(M_MISS_EVICTED, tier=t),
            "stall_seconds": scrape.sum(M_STALL, tier=t),
        }
        row["hit_ratio"] = safe_div(originated, r["prompt_tokens"])
        row["break_even_ratio"] = safe_div(
            row["tokens_per_s"], r["recompute_tokens_per_s"]
        )
        # Is this tier's `load_seconds` wall time, or thread-time summed over a
        # pool? If the latter, every rate above is a floor. The upper bound
        # comes from the other end: the whole transfer cannot have taken less
        # than its single slowest batch, which the latency histogram gives us.
        row["queued_io"] = (t in QUEUED_IO_TIERS) and not serial_io
        row["rate_is_lower_bound"] = bool(row["queued_io"] and row["load_ops"] > 1)
        if row["rate_is_lower_bound"] and row["load_latency_p99"]:
            row["bytes_per_s_upper"] = safe_div(load_bytes,
                                                row["load_latency_p99"])
            row["tokens_per_s_upper"] = safe_div(served,
                                                 row["load_latency_p99"])
            row["break_even_upper"] = safe_div(row["tokens_per_s_upper"],
                                               r["recompute_tokens_per_s"])
        else:
            row["bytes_per_s_upper"] = None
            row["tokens_per_s_upper"] = None
            row["break_even_upper"] = None
        tiers.append(row)

    r["tiers"] = tiers

    # ---- consistency check against the unpatched source split ------------
    # `external_kv_transfer` counts tokens the offload path served, which should
    # match the CPU tier's served-from figure. Disagreement means one of the two
    # attributions is wrong, and it is worth saying so out loud rather than
    # picking a favourite.
    ext = by_source.get("external_kv_transfer")
    cpu_served = hit_tokens.get("cpu")
    if ext and cpu_served:
        r["attribution_cross_check"] = {
            "prompt_tokens_by_source_external": ext,
            "tier_hit_tokens_cpu_served": cpu_served,
            "ratio": cpu_served / ext,
        }
        if not 0.8 <= cpu_served / ext <= 1.25:
            r["notes"].append(
                "The per-tier token counts and prompt_tokens_by_source disagree "
                "by more than 25%% (%s vs %s). They are measured at different "
                "points -- the source split counts tokens accepted for a "
                "request, the tier counters count blocks handed to the GPU -- "
                "so some drift is expected, but this much means one of them "
                "should not be trusted for the hit-ratio column."
                % ("{:,.0f}".format(cpu_served), "{:,.0f}".format(ext))
            )

    # ---- the counterfactual: what the whole cache is worth ---------------
    # saved = for every token a tier delivered, the time recompute would have
    # taken minus the time the tier took. cost = the store-side tax plus stall
    # time the tier never earned back. No A/B and no restart: every term is a
    # cumulative counter.
    rc = r["recompute_tokens_per_s"]
    saved = 0.0
    cost = 0.0
    per_tier_value = {}
    for row in tiers:
        t = row["tier"]
        tokens = row["hit_tokens_originated"]
        tps = row["tokens_per_s"]
        tier_saved = 0.0
        if rc and tps and tokens:
            tier_saved = tokens * (1.0 / rc - 1.0 / tps)
        tier_cost = row["store_seconds"] + row["stall_seconds"]
        # On a queued tier both ends of this subtraction are pessimistic: the
        # rate that produces `tier_saved` is a floor (thread-summed seconds),
        # and `store_seconds` is summed over the same pool, so the cost is a
        # ceiling. A negative net here is therefore not evidence of a loss.
        tier_saved_upper = tier_saved
        if rc and tokens and row.get("tokens_per_s_upper"):
            tier_saved_upper = tokens * (1.0 / rc
                                         - 1.0 / row["tokens_per_s_upper"])
        per_tier_value[t] = {
            "saved_seconds": tier_saved,
            "cost_seconds": tier_cost,
            "net_seconds": tier_saved - tier_cost,
            "saved_seconds_upper": tier_saved_upper,
            "net_seconds_upper": tier_saved_upper - tier_cost,
            "bounded": bool(row.get("rate_is_lower_bound")),
        }
        saved += tier_saved
        cost += tier_cost
    r["value"] = {
        "saved_seconds": saved,
        "cost_seconds": cost,
        "net_seconds": saved - cost,
        "per_tier": per_tier_value,
    }
    return r


# --------------------------------------------------------------------------
# Device calibration (no patch, no engine)
# --------------------------------------------------------------------------
def calibrate_device(root, max_files=24, seed=0):
    """Time reads of real stored blocks to get the raw device rate.

    Why this exists even though the engine now reports its own load latency:
    it is an INDEPENDENT number. If the engine says the fs tier delivers
    80 MB/s and the device does 117 MB/s cold, the gap is the engine's; if the
    device itself does 80, no amount of tuning will help and the answer is a
    different disk. A report that can only quote the engine cannot tell those
    two apart.

    O_DIRECT is used where available so the page cache does not make a cold
    disk look like RAM -- which is the single easiest way to get this number
    catastrophically wrong.
    """
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".bin"):
                paths.append(os.path.join(dirpath, fn))
        if len(paths) > 4000:
            break
    if not paths:
        return {"error": "no block files found under %s" % root}
    random.Random(seed).shuffle(paths)
    paths = paths[:max_files]

    # O_DIRECT needs a page-aligned buffer, which a plain bytes object from
    # os.read() is not -- that combination returns EINVAL and reads nothing, so
    # a naive attempt measures 0 B/s and looks like a broken disk. An anonymous
    # mmap IS page-aligned, and os.preadv() will read straight into it.
    CHUNK = 1 << 20
    buf = mmap.mmap(-1, CHUNK)
    o_direct = getattr(os, "O_DIRECT", 0)

    def _open_cold(path):
        """Return (fd, used_direct). Falls back to dropping this file from the
        page cache, so a warm cache cannot make an HDD look like RAM either
        way."""
        if o_direct:
            try:
                return os.open(path, os.O_RDONLY | o_direct), True
            except OSError:
                pass
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError):
            pass
        return fd, False

    total_bytes = 0
    files_read = 0
    used_direct = True
    start = time.monotonic()
    for path in paths:
        try:
            fd, direct = _open_cold(path)
        except OSError:
            continue
        used_direct = used_direct and direct
        offset = 0
        try:
            while True:
                got = os.preadv(fd, [buf], offset)
                if not got:
                    break
                total_bytes += got
                offset += got
                if got < CHUNK:
                    break
        except OSError:
            pass
        finally:
            os.close(fd)
        files_read += 1
    elapsed = time.monotonic() - start
    buf.close()

    if not total_bytes:
        return {"error": "read 0 bytes from %s block files under %s"
                         % (files_read, root)}
    return {
        "root": root,
        "files_read": files_read,
        "bytes_read": total_bytes,
        "seconds": elapsed,
        "bytes_per_s": safe_div(float(total_bytes), elapsed),
        "o_direct": used_direct,
        "caveat": (
            "Sequential reads of whole block files in random file order. It is "
            "the device's read rate for this access pattern, not a synthetic "
            "peak figure. O_DIRECT bypasses this machine's page cache, but it "
            "cannot bypass a cache below the block device -- a hypervisor host "
            "cache, a RAID controller, or a ZFS ARC on the backing store -- so "
            "run it twice: a much faster second pass means something under the "
            "device is caching, and the first number is the honest one."
            + ("" if used_direct else
               " O_DIRECT was unavailable on at least one file, so this is a "
               "page-cache-dropped read rather than a bypassed one -- treat it "
               "as an upper bound.")
        ),
    }


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------
def verdicts(r, calibration=None):
    """The point of the exercise: one plain-language answer per question.

    Each verdict carries the signal it was derived from, so a reader can
    disagree with the threshold rather than with the tool.
    """
    out = []

    def add(question, status, statement, signal):
        out.append({
            "question": question,
            "status": status,          # yes / no / marginal / unknown
            "statement": statement,
            "signal": signal,
        })

    if not r.get("have_tier_metrics"):
        if r.get("tier_patch_present"):
            # The patch is in. The tiers simply have not served anything yet,
            # which on a freshly restarted engine is the expected state rather
            # than a fault: a block has to be stored before it can be read
            # back, and the load-side series do not exist until the first read.
            add("All of them", "unknown",
                "The per-tier instrumentation is present, but no tier has "
                "served a read yet, so there is nothing to size. This is the "
                "normal state of a freshly restarted engine: the store side is "
                "already recording, and the load side appears with the first "
                "cache hit. Leave the server up and run this again after a day "
                "of real traffic -- these counters are lifetime-cumulative and "
                "a restart resets them, so restarting now would throw away the "
                "evidence rather than produce it.",
                "%s* present; no %s or %s yet" % (TR, M_HIT_TOKENS, M_LOAD_BYTES))
        else:
            add("All of them", "unknown",
                "The per-tier metrics are not present on this server, so none "
                "of the sizing questions can be answered. Apply "
                "patch_kv_offload_tier_report.py and restart.",
                "no series matching %s*" % TR)
        return out

    tiers = {row["tier"]: row for row in r["tiers"]}
    rc = r["recompute_tokens_per_s"]
    # In --since mode "nothing has happened yet" means "nothing in the window",
    # which is a different statement about the deployment. Say the right one.
    period = "in this window" if r["mode"] == "window" else "yet"

    # ---- Q1/Q2/Q6: the RAM question -------------------------------------
    cpu = tiers.get("cpu")
    if cpu is None:
        add("Is my CPU tier the right size?", "unknown",
            "No CPU tier row was reported.", "tier=\"cpu\" absent")
    else:
        never = cpu["never_read_fraction"]
        reuse_p50 = cpu["eviction_to_reuse_p50"]
        oversized = never is not None and never > T_NEVER_READ_FRACTION
        undersized = reuse_p50 is not None and reuse_p50 < T_REUSE_P50_SECONDS

        if never is None:
            add("Too much RAM?", "unknown",
                "No blocks have been evicted from the CPU tier %s, so there is "
                "nothing to say about whether stored blocks get read back. That "
                "is itself information: the tier has not filled up." % period,
                "%s has no observations" % M_READS_BEFORE_EVICT)
        elif oversized:
            wasted = None
            if cpu["capacity_bytes"]:
                wasted = never * cpu["capacity_bytes"]
            add("Too much RAM?", "yes",
                "%s of blocks evicted from the CPU tier were never read back%s. "
                "The tier is oversized for this workload, or the store policy "
                "is too eager -- shrink it and spend the memory elsewhere."
                % (fmt_pct(never),
                   " (about %s worth of capacity)" % fmt_bytes(wasted)
                   if wasted else ""),
                "%s mass at 0 reads = %s (> %s)"
                % (M_READS_BEFORE_EVICT, fmt_pct(never),
                   fmt_pct(T_NEVER_READ_FRACTION)))
        else:
            add("Too much RAM?", "no",
                "%s of evicted blocks had been read back at least once "
                "(mean %s reads), so the tier is earning its memory."
                % (fmt_pct(1.0 - never),
                   "-" if cpu["reads_before_evict_mean"] is None
                   else "%.1f" % cpu["reads_before_evict_mean"]),
                "%s mass at 0 reads = %s" % (M_READS_BEFORE_EVICT,
                                             fmt_pct(never)))

        if reuse_p50 is None:
            add("Too little RAM?", "unknown",
                "No evicted block has been asked for again %s, so there is no "
                "evidence of thrashing." % period,
                "%s has no observations" % M_EVICTION_TO_REUSE)
        elif undersized:
            add("Too little RAM?", "yes",
                "Half the blocks the CPU tier evicts are requested again "
                "within %s (%s such re-requests so far). That is cache "
                "thrashing measured rather than inferred -- grow the tier."
                % (fmt_secs(reuse_p50), fmt_num(cpu["eviction_to_reuse_count"])),
                "%s p50 = %s (< %s)"
                % (M_EVICTION_TO_REUSE, fmt_secs(reuse_p50),
                   fmt_secs(T_REUSE_P50_SECONDS)))
        else:
            add("Too little RAM?", "no",
                "Blocks that come back after eviction take %s to do so at the "
                "median, which is long enough that holding them would not have "
                "helped much." % fmt_secs(reuse_p50),
                "%s p50 = %s" % (M_EVICTION_TO_REUSE, fmt_secs(reuse_p50)))

        if oversized and undersized:
            add("Is the tier the right shape?", "no",
                "Both signals fire at once: blocks nobody reads are being "
                "stored while blocks that are wanted again get evicted. That is "
                "an eviction-policy problem, not a capacity problem -- more "
                "memory would buy less than a better replacement decision.",
                "mass at 0 reads = %s AND eviction-to-reuse p50 = %s"
                % (fmt_pct(never), fmt_secs(reuse_p50)))

        if cpu["occupancy_fraction_above_95"] is not None:
            frac = cpu["occupancy_fraction_above_95"]
            add("Does the CPU tier run persistently full?",
                "yes" if frac > T_OCCUPANCY_HIGH_FRACTION else "no",
                "The CPU tier was above %s full for %s of the time (%s of %s "
                "in use at the last sample)."
                % (fmt_pct(T_OCCUPANCY_HIGH), fmt_pct(frac),
                   fmt_bytes(cpu["used_bytes"]),
                   fmt_bytes(cpu["capacity_bytes"])),
                "%s fraction above %.2f = %s"
                % (M_OCCUPANCY, T_OCCUPANCY_HIGH, fmt_pct(frac)))

    # ---- Q3/Q4: the disk question, per secondary tier -------------------
    for row in r["tiers"]:
        t = row["tier"]
        if t == "cpu":
            continue
        ratio = row["break_even_ratio"]
        served = row["hit_tokens_originated"]
        value = r["value"]["per_tier"].get(t, {})
        # A tier whose I/O runs on a worker pool reports thread-summed seconds,
        # so every rate below is a floor and no "too slow" answer can honestly
        # be computed from it. Say what is known -- the bracket, and why the
        # batch latency is long on purpose -- instead of a verdict.
        bounded = bool(row.get("rate_is_lower_bound"))

        if served <= 0 and row["store_seconds"] <= 0:
            add("Is the %s tier pulling its weight?" % t, "unknown",
                "The %s tier neither served nor stored anything %s, so there is "
                "nothing to judge it on." % (t, period),
                "%s and %s{tier=\"%s\"} both 0"
                % (M_HIT_TOKENS, M_STORE_SECONDS, t))
        elif served <= 0:
            add("Is the %s tier pulling its weight?" % t, "no",
                "The %s tier has served nothing, but it has spent %s writing "
                "blocks. Right now it is pure overhead: either the working set "
                "never reaches it, or lookups are not waiting long enough for "
                "it to answer."
                % (t, fmt_secs(row["store_seconds"])),
                "%s{tier=\"%s\"} = 0" % (M_HIT_TOKENS, t))
        elif bounded:
            hi = row.get("break_even_upper")
            add("Is the %s tier too slow?" % t, "unknown",
                "Cannot be answered from these counters, and the number that "
                "looks like an answer is not one. The %s tier dispatches its "
                "I/O to a worker pool, so %s is the SUM of the per-batch times "
                "across every worker, not wall time; dividing by it gives %s "
                "tok/s, which is a floor. The other end comes from the latency "
                "histogram: the whole transfer cannot have taken less than its "
                "single slowest batch (%s), giving at most %s tok/s. So the "
                "tier sits somewhere between %s and %s of recompute -- measure "
                "the device directly (--calibrate) to place it.\n"
                "The long batch time is the design, not a fault: this tier "
                "QUEUES work across the fanout rather than preempting and "
                "evicting, so a batch that waits while its siblings share the "
                "device is the mechanism working. A request waits the "
                "promotion, which overlaps those batches; it does not wait one "
                "batch after another."
                % (t, M_LOAD_SECONDS, fmt_num(row["tokens_per_s"]),
                   fmt_secs(row["load_latency_p99"]),
                   fmt_num(row.get("tokens_per_s_upper")),
                   "-" if ratio is None else "%.2fx" % ratio,
                   "-" if hi is None else "%.2fx" % hi),
                "%s{tier=\"%s\"} is summed across the pool over %s load "
                "batches; break-even is bracketed [%s, %s]"
                % (M_LOAD_SECONDS, t, fmt_num(row["load_ops"]),
                   "-" if ratio is None else "%.2f" % ratio,
                   "-" if hi is None else "%.2f" % hi))
        elif ratio is None:
            add("Is the %s tier too slow?" % t, "unknown",
                "Not enough timing data to compare the %s tier against "
                "recompute." % t,
                "no %s or no recompute baseline" % M_LOAD_SECONDS)
        elif ratio < 1.0:
            need = rc * T_BREAK_EVEN if rc else None
            add("Is the %s tier too slow?" % t, "yes",
                "The %s tier delivers %s tok/s where recompute manages %s "
                "tok/s -- it is SLOWER than just recomputing (%.2fx). Serving "
                "from it costs time instead of saving it. It would need about "
                "%s tok/s, roughly %s at the measured %s per token, to be "
                "worth %sx break-even."
                % (t, fmt_num(row["tokens_per_s"]), fmt_num(rc), ratio,
                   fmt_num(need),
                   fmt_rate((need or 0) * (row["bytes_per_token"] or 0)),
                   fmt_bytes(row["bytes_per_token"]), T_BREAK_EVEN),
                "break-even ratio = %.2f (< 1.0)" % ratio)
        elif ratio < T_BREAK_EVEN:
            need = rc * T_BREAK_EVEN if rc else None
            add("Is the %s tier too slow?" % t, "marginal",
                "The %s tier delivers %s tok/s against a recompute baseline of "
                "%s tok/s -- %.2fx, which is at best marginal. A device fast "
                "enough to reach %sx would need about %s; that is what an NVMe "
                "upgrade buys here."
                % (t, fmt_num(row["tokens_per_s"]), fmt_num(rc), ratio,
                   T_BREAK_EVEN,
                   fmt_rate((need or 0) * (row["bytes_per_token"] or 0))),
                "break-even ratio = %.2f (< %s)" % (ratio, T_BREAK_EVEN))
        else:
            add("Is the %s tier too slow?" % t, "no",
                "The %s tier delivers %s tok/s against %s tok/s for recompute "
                "-- %.2fx. It is faster than recomputing and worth having."
                % (t, fmt_num(row["tokens_per_s"]), fmt_num(rc), ratio),
                "break-even ratio = %.2f" % ratio)

        if row["stall_seconds"] > 0 or served > 0:
            net = value.get("net_seconds")
            saved_s = value.get("saved_seconds")
            if net is not None and net < 0 and bounded:
                # Both halves of this subtraction are thread-summed on a queued
                # tier, so it is pessimistic twice over and cannot be reported
                # as a loss. Give the reader the two facts and the one number
                # that IS wall-clock: the stall.
                up = value.get("net_seconds_upper")
                add("Is the %s tier actively hurting?" % t, "unknown",
                    "Not decidable from these counters, in the direction that "
                    "matters. The saving is computed from a delivery rate that "
                    "is a floor (thread-summed seconds), and the %s of store "
                    "cost on the other side of the subtraction is summed "
                    "across the same pool, so it is a ceiling. Both errors "
                    "push the answer the same way -- towards a loss that may "
                    "not exist. At face value the tier is %s behind; at the "
                    "optimistic end of the delivery bracket alone, still %s. "
                    "Neither is a measurement.\n"
                    "The one wall-clock number here is the stall: %s of "
                    "request time was spent waiting on this tier. That is what "
                    "a caller actually paid, and it is the figure to judge the "
                    "tier by until the seconds counters are wall time."
                    % (fmt_secs(value.get("cost_seconds")), fmt_secs(-net),
                       ("%s behind" % fmt_secs(-up)) if (up or 0) < 0
                       else ("%s ahead" % fmt_secs(up)),
                       fmt_secs(row["stall_seconds"])),
                    "saved and cost are both thread-summed; stall = %s "
                    "(wall clock)" % fmt_secs(row["stall_seconds"]))
            elif net is not None and net < 0:
                saved_phrase = (
                    "saved %s of compute" % fmt_secs(saved_s) if saved_s >= 0
                    else "was %s SLOWER than recomputing the tokens it served"
                         % fmt_secs(-saved_s))
                add("Is the %s tier actively hurting?" % t, "yes",
                    "Over this period the %s tier %s, and cost a further "
                    "%s in stalls and stores -- a net LOSS of %s. Disable it or "
                    "replace the device."
                    % (t, saved_phrase,
                       fmt_secs(value.get("cost_seconds")), fmt_secs(-net)),
                    "saved %s < cost %s" % (fmt_secs(saved_s),
                                            fmt_secs(value.get("cost_seconds"))))
            elif net is not None:
                add("Is the %s tier actively hurting?" % t, "no",
                    "The %s tier is net positive: %s saved against %s spent "
                    "stalling and storing, %s ahead."
                    % (t, fmt_secs(saved_s),
                       fmt_secs(value.get("cost_seconds")), fmt_secs(net)),
                    "net = %s" % fmt_secs(net))

        if bounded and calibration and "bytes_per_s" in calibration:
            add("Where is the %s tier's ceiling?" % t, "unknown",
                "Not comparable as measured. The calibration is wall time for "
                "one stream; the engine's %s is thread-time summed over a "
                "pool, so the two numbers do not describe the same clock and "
                "their ratio means nothing. Fix the instrument (time the whole "
                "promotion once, rather than each batch) before answering this."
                % fmt_rate(row["bytes_per_s"]),
                "engine rate is a thread-summed floor; calibration is "
                "single-stream wall time")
        elif calibration and "bytes_per_s" in calibration and row["bytes_per_s"]:
            dev = calibration["bytes_per_s"]
            eng = row["bytes_per_s"]
            if dev and eng:
                gap = eng / dev
                if gap > 1.15:
                    # The engine cannot really beat its own device. When this
                    # fires the two numbers were not measuring the same thing --
                    # the engine reads many blocks per job and in parallel,
                    # while the calibration reads whole files one at a time --
                    # so the comparison is void rather than surprising. Saying
                    # that is more useful than picking whichever number
                    # supports a purchase.
                    add("Where is the %s tier's ceiling?" % t, "unknown",
                        "The engine reports %s from the %s tier while the raw "
                        "device measured only %s, which cannot both describe "
                        "the same access pattern. The calibration reads whole "
                        "block files one at a time; the engine reads many "
                        "blocks per job, in parallel, at whatever queue depth "
                        "the tier uses. Treat the calibration as a "
                        "single-stream floor, not as this tier's ceiling."
                        % (fmt_rate(eng), t, fmt_rate(dev)),
                        "engine %s exceeds calibrated device %s (%.1fx)"
                        % (fmt_rate(eng), fmt_rate(dev), gap))
                else:
                    add("Where is the %s tier's ceiling?" % t,
                        "no" if gap > 0.7 else "yes",
                        "The engine gets %s from the %s tier; the raw device "
                        "measures %s single-stream. The engine is at %s of "
                        "that, so the limit is %s."
                        % (fmt_rate(eng), t, fmt_rate(dev), fmt_pct(gap),
                           "the hardware -- a faster device is the only lever"
                           if gap > 0.7 else
                           "in the software path, not the disk -- a faster "
                           "device would not be fully used"),
                        "engine %s vs device %s" % (fmt_rate(eng),
                                                    fmt_rate(dev)))

    # ---- Q5: is the whole thing worth it? -------------------------------
    v = r["value"]
    if v["saved_seconds"] or v["cost_seconds"]:
        window = r["window_seconds"]
        span = (" over the %s window" % fmt_secs(window)) if window else ""
        if v["net_seconds"] > 0:
            add("Is the layered cache worth it?", "yes",
                "Yes%s: %s of compute avoided against %s of store overhead and "
                "stalls -- %s net. On this traffic the layered cache is adding "
                "value." % (span, fmt_secs(v["saved_seconds"]),
                            fmt_secs(v["cost_seconds"]),
                            fmt_secs(v["net_seconds"])),
                "net = saved %s - cost %s" % (fmt_secs(v["saved_seconds"]),
                                              fmt_secs(v["cost_seconds"])))
        elif any(row.get("rate_is_lower_bound") for row in r["tiers"]):
            # The roll-up inherits the queued tier's pessimism; it cannot
            # declare a loss the parts could not.
            bounded_names = ", ".join(row["tier"] for row in r["tiers"]
                                      if row.get("rate_is_lower_bound"))
            add("Is the layered cache worth it?", "unknown",
                "The sum says %s saved against %s of overhead%s, short by %s "
                "-- a deficit that inherits the %s tier's thread-summed seconds on "
                "both sides, so it is a pessimistic bound and not a result. "
                "What can be said without that instrument: %s of prompt tokens "
                "came from cache rather than compute, and %s of request time "
                "was spent stalled on a tier."
                % (fmt_secs(v["saved_seconds"]), fmt_secs(v["cost_seconds"]),
                   span, fmt_secs(-v["net_seconds"]), bounded_names,
                   fmt_pct(1.0 - (safe_div(r["by_source"].get("local_compute",
                                                               0.0),
                                            r["prompt_tokens"]) or 0.0)),
                   fmt_secs(r["prefill_stall_seconds"])),
                "net = %s, but %s seconds are thread-summed"
                % (fmt_secs(v["net_seconds"]), bounded_names))
        else:
            add("Is the layered cache worth it?", "no",
                "Not as configured%s: %s saved against %s of store overhead and "
                "stalls, a net loss of %s. The per-tier verdicts above say "
                "which tier is responsible."
                % (span, fmt_secs(v["saved_seconds"]),
                   fmt_secs(v["cost_seconds"]), fmt_secs(-v["net_seconds"])),
                "net = %s" % fmt_secs(v["net_seconds"]))
    return out


# --------------------------------------------------------------------------
# Snapshots (--snapshot / --since)
# --------------------------------------------------------------------------
def save_snapshot(path, samples, timestamp):
    """Store the raw samples, not the derived report.

    Differencing has to happen on the counters themselves; a snapshot of
    computed rates could not be differenced at all. Storing raw also means a
    snapshot taken by an older version of this script still works.
    """
    with open(path, "w") as fh:
        json.dump({
            "version": 1,
            "timestamp": timestamp,
            "samples": [[n, lb, v] for n, lb, v in samples],
        }, fh)


def load_snapshot(path):
    with open(path) as fh:
        blob = json.load(fh)
    samples = [(n, lb, v) for n, lb, v in blob["samples"]]
    return Scrape(samples, timestamp=blob.get("timestamp"))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
STATUS_MARK = {"yes": "YES", "no": "NO", "marginal": "MARGINAL",
               "unknown": "UNKNOWN"}


def _table(headers, rows):
    """Fixed-width table. Same cell text feeds stdout and markdown."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append("  ".join(cell.ljust(widths[i])
                             for i, cell in enumerate(row)).rstrip())
    return out


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def queued_note(r):
    """One paragraph, shared by both renderers, for queued-I/O tiers.

    Without it the table reads as an accusation: a 15-second batch time next to
    a throughput figure that is really a floor. Both come from the same fact --
    the I/O runs on a pool -- and neither is a fault.
    """
    names = [row["tier"] for row in r["tiers"] if row.get("queued_io")]
    if not names:
        return []
    who = " and ".join(names)
    plural = "tiers" if len(names) > 1 else "tier"
    return [
        "The %s %s dispatches its I/O to a worker pool. Two consequences for "
        "this table. Batch p50/p99 are the service times of ONE batch while "
        "its siblings share the device -- a request waits the promotion, which "
        "overlaps them, not one batch after another; this design queues "
        "deliberately rather than preempting and evicting, so a long batch "
        "time is the trade being made and not a stall the caller sees. And "
        "because the seconds counter sums across those threads, Throughput, "
        "Equiv tok/s and vs recompute are LOWER BOUNDS (marked >=), not "
        "measurements. Pass --serial-io if this backend really does read "
        "inline." % (who, plural),
    ]


def tier_table(r):
    # "Batch", not "Fetch": these are the service times of one I/O batch, and
    # on a queued tier several of them are in flight at once. A request does
    # not wait one of these; it waits the promotion, which overlaps them.
    headers = ["Tier", "Role", "Size", "Used", "Hit ratio", "Served tok",
               "Batch p50", "Batch p99", "Throughput", "Equiv tok/s",
               "vs recompute"]
    rows = []
    for row in r["tiers"]:
        rows.append([
            row["tier"],
            row["role"],
            fmt_bytes(row["capacity_bytes"]),
            fmt_bytes(row["used_bytes"]),
            fmt_pct(row["hit_ratio"]),
            fmt_num(row["hit_tokens_originated"]),
            fmt_secs(row["load_latency_p50"]),
            fmt_secs(row["load_latency_p99"]),
            ("\u2265 " if row.get("rate_is_lower_bound") else "")
            + fmt_rate(row["bytes_per_s"]),
            ("\u2265 " if row.get("rate_is_lower_bound") else "")
            + fmt_num(row["tokens_per_s"]),
            "-" if not row["break_even_ratio"]
            else ("\u2265 %.2fx" if row.get("rate_is_lower_bound")
                  else "%.2fx") % row["break_even_ratio"],
        ])
    rows.append([
        "recompute", "fallback", "-", "-",
        fmt_pct(safe_div(r["by_source"].get("local_compute", 0.0),
                         r["prompt_tokens"])),
        fmt_num(r["prefill_computed_tokens"]),
        "-", "-", "-", fmt_num(r["recompute_tokens_per_s"]), "1.00x",
    ])
    return headers, rows


def sizing_table(r):
    headers = ["Tier", "Above 95% full", "Never read back", "Evictions",
               "Evicted", "Reuse-after-evict p50", "Would-have-hit",
               "Stall time"]
    rows = []
    for row in r["tiers"]:
        rows.append([
            row["tier"],
            fmt_pct(row["occupancy_fraction_above_95"]),
            fmt_pct(row["never_read_fraction"]),
            fmt_num(row["evictions"]),
            fmt_bytes(row["evicted_bytes"]),
            fmt_secs(row["eviction_to_reuse_p50"]),
            fmt_num(row["miss_evicted"]),
            fmt_secs(row["stall_seconds"]),
        ])
    return headers, rows


def render_stdout(r, vs, calibration=None):
    lines = []
    lines.append("KV cache tier report -- %s (%s)"
                 % (r["generated_at"], r["mode"]))
    if r["window_seconds"]:
        lines.append("Window: %s" % fmt_secs(r["window_seconds"]))
    lines.append("")

    if r["have_tier_metrics"]:
        lines.append("TIERS")
        h, rows = tier_table(r)
        lines += ["  " + l for l in _table(h, rows)]
        for para in queued_note(r):
            lines.append("")
            lines += ["  " + l for l in textwrap.wrap(para, 74)]
        lines.append("")
        lines.append("SIZING SIGNALS")
        h, rows = sizing_table(r)
        lines += ["  " + l for l in _table(h, rows)]
        lines.append("")
        v = r["value"]
        lines.append("VALUE OF THE LAYERED CACHE")
        lines.append("  compute avoided : %s" % fmt_secs(v["saved_seconds"]))
        lines.append("  overhead paid   : %s (stores + stalls)"
                     % fmt_secs(v["cost_seconds"]))
        lines.append("  net             : %s" % fmt_secs(v["net_seconds"]))
        lines.append("")

    if r["by_source"]:
        lines.append("PROMPT TOKEN ATTRIBUTION")
        h = ["Source", "Tokens", "Share"]
        rows = [[src, fmt_num(val), fmt_pct(safe_div(val, r["prompt_tokens"]))]
                for src, val in sorted(r["by_source"].items(),
                                       key=lambda kv: -kv[1])]
        lines += ["  " + l for l in _table(h, rows)]
        lines.append("")
    lines.append("RECOMPUTE BASELINE")
    lines.append("  %s tok/s (%s computed tokens over %s of prefill compute, "
                 "%s of that removed as tier stall)"
                 % (fmt_num(r["recompute_tokens_per_s"]),
                    fmt_num(r["prefill_computed_tokens"]),
                    fmt_secs(r["prefill_seconds"] - r["prefill_stall_seconds"]),
                    fmt_secs(r["prefill_stall_seconds"])))
    lines.append("")

    if calibration:
        lines.append("DEVICE CALIBRATION")
        if "error" in calibration:
            lines.append("  %s" % calibration["error"])
        else:
            lines.append("  %s: %s over %s files (%s)%s"
                         % (calibration["root"],
                            fmt_rate(calibration["bytes_per_s"]),
                            calibration["files_read"],
                            fmt_bytes(calibration["bytes_read"]),
                            "" if calibration["o_direct"] else " [no O_DIRECT]"))
            lines.append("  %s" % calibration["caveat"])
        lines.append("")

    lines.append("VERDICTS")
    for v in vs:
        lines.append("  [%s] %s" % (STATUS_MARK.get(v["status"], v["status"]),
                                    v["question"]))
        for para in v["statement"].split("\n"):
            lines += ["        " + l for l in textwrap.wrap(para, 72)]
        lines.append("        signal: %s" % v["signal"])
    lines.append("")

    if r["missing"]:
        lines.append("MISSING")
        for m in r["missing"]:
            lines.append("  - %s" % m)
        lines.append("")
    if r["notes"]:
        lines.append("NOTES")
        for n in r["notes"]:
            lines.append("  - %s" % n)
        lines.append("")
    return "\n".join(lines)


def render_markdown(r, vs, calibration=None, source_url=None):
    out = []
    out.append("# KV cache tier report")
    out.append("")
    out.append("- Generated: `%s`" % r["generated_at"])
    out.append("- Mode: **%s**%s"
               % (r["mode"],
                  "" if not r["window_seconds"]
                  else " over %s" % fmt_secs(r["window_seconds"])))
    if source_url:
        out.append("- Source: `%s`" % source_url)
    out.append("")

    out.append("## Verdicts")
    out.append("")
    out.append("| Question | Verdict | Finding | Signal |")
    out.append("|---|---|---|---|")
    for v in vs:
        out.append("| %s | **%s** | %s | `%s` |"
                   % (v["question"], STATUS_MARK.get(v["status"], v["status"]),
                      v["statement"].replace("|", "\\|"),
                      v["signal"].replace("|", "\\|")))
    out.append("")

    if r["have_tier_metrics"]:
        out.append("## Tiers")
        out.append("")
        h, rows = tier_table(r)
        out += _md_table(h, rows)
        out.append("")
        out.append("`Equiv tok/s` is the tier's delivery rate in tokens of "
                   "prefill it replaced, so it is directly comparable to the "
                   "recompute row. `vs recompute` is that ratio: below 1.00x "
                   "the tier is slower than not having it.")
        for para in queued_note(r):
            out.append("")
            out.append(para)
        out.append("")

        out.append("## Sizing signals")
        out.append("")
        h, rows = sizing_table(r)
        out += _md_table(h, rows)
        out.append("")
        out.append("- **Never read back** -- share of evicted blocks that were "
                   "stored and never loaded once. High means oversized, or a "
                   "store policy that is too eager.")
        out.append("- **Reuse-after-evict p50** -- how long after eviction a "
                   "block is asked for again. Short means undersized: that is "
                   "thrashing, measured rather than inferred.")
        out.append("- **Would-have-hit** -- lookups that missed for a block "
                   "this tier used to hold. Only these justify buying "
                   "capacity; a block that was never cached does not.")
        out.append("")

        v = r["value"]
        out.append("## Is the layered cache worth it?")
        out.append("")
        out.append("| Term | Value |")
        out.append("|---|---|")
        out.append("| Compute avoided | %s |" % fmt_secs(v["saved_seconds"]))
        out.append("| Store + stall overhead | %s |" % fmt_secs(v["cost_seconds"]))
        out.append("| **Net** | **%s** |" % fmt_secs(v["net_seconds"]))
        out.append("")
        out.append("Per tier:")
        out.append("")
        out.append("| Tier | Saved | Cost | Net |")
        out.append("|---|---|---|---|")
        for t, pv in v["per_tier"].items():
            out.append("| %s | %s | %s | %s |"
                       % (t, fmt_secs(pv["saved_seconds"]),
                          fmt_secs(pv["cost_seconds"]),
                          fmt_secs(pv["net_seconds"])))
        out.append("")
        out.append("Counterfactual, from cumulative counters only -- no A/B, no "
                   "restart. `saved` credits each tier with the prefill time "
                   "recompute would have spent on the tokens it delivered, "
                   "minus the time it took to deliver them; `cost` charges it "
                   "for stores and for stalls.")
        out.append("")

    if r["by_source"]:
        out.append("## Prompt token attribution")
        out.append("")
        out.append("| Source | Tokens | Share |")
        out.append("|---|---|---|")
        for src, val in sorted(r["by_source"].items(),
                               key=lambda kv: -kv[1]):
            out.append("| `%s` | %s | %s |"
                       % (src, fmt_num(val),
                          fmt_pct(safe_div(val, r["prompt_tokens"]))))
        out.append("")
        if "attribution_cross_check" in r:
            cc = r["attribution_cross_check"]
            out.append("Cross-check: the unlabelled source split says the "
                       "offload path served %s tokens; the per-tier counters "
                       "say %s (%.2fx). They are measured at different points, "
                       "so exact agreement is not expected."
                       % (fmt_num(cc["prompt_tokens_by_source_external"]),
                          fmt_num(cc["tier_hit_tokens_cpu_served"]),
                          cc["ratio"]))
            out.append("")

    if calibration:
        out.append("## Device calibration")
        out.append("")
        if "error" in calibration:
            out.append("- %s" % calibration["error"])
        else:
            out.append("- Root: `%s`" % calibration["root"])
            out.append("- Measured: **%s** (%s across %s files, O_DIRECT %s)"
                       % (fmt_rate(calibration["bytes_per_s"]),
                          fmt_bytes(calibration["bytes_read"]),
                          calibration["files_read"],
                          "yes" if calibration["o_direct"] else "NO"))
            out.append("- %s" % calibration["caveat"])
        out.append("")
        out.append("This number comes from the filesystem, not from vLLM. It is "
                   "the independent check: if the engine is far below the "
                   "device, the limit is in the software path and a faster disk "
                   "will not be used.")
        out.append("")

    if r["missing"]:
        out.append("## Missing inputs")
        out.append("")
        for m in r["missing"]:
            out.append("- %s" % m)
        out.append("")
    if r["notes"]:
        out.append("## Notes")
        out.append("")
        for n in r["notes"]:
            out.append("- %s" % n)
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Justify the size and speed of every KV cache tier from "
                    "one /metrics scrape.")
    ap.add_argument("--base", default="http://127.0.0.1:8000",
                    help="server base URL; /metrics is appended "
                         "(default: %(default)s)")
    ap.add_argument("--url", default=None,
                    help="full metrics URL, overrides --base")
    ap.add_argument("--metrics-file", default=None,
                    help="read exposition text from a file instead of HTTP")
    ap.add_argument("--out", default=None,
                    help="directory for results.md + results.json")
    ap.add_argument("--json", action="store_true",
                    help="print the report as JSON on stdout instead of a table")
    ap.add_argument("--snapshot", default=None,
                    help="write the raw scrape here for a later --since")
    ap.add_argument("--since", default=None,
                    help="difference against a snapshot written by --snapshot")
    ap.add_argument("--calibrate", default=None, metavar="DIR",
                    help="time reads of real block files under DIR to measure "
                         "the raw device rate (e.g. /kvcache/blocks)")
    ap.add_argument("--calibrate-files", type=int, default=24,
                    help="how many block files to read (default: %(default)s)")
    ap.add_argument("--serial-io", action="store_true",
                    help="this deployment runs tier I/O inline, so the seconds "
                         "counters are wall time; report rates flat instead of "
                         "as bounds (default: assume a worker pool for %s)"
                         % ", ".join(sorted(QUEUED_IO_TIERS)))
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    if args.metrics_file:
        with open(args.metrics_file) as fh:
            text = fh.read()
        source = args.metrics_file
    else:
        source = args.url or (args.base.rstrip("/") + "/metrics")
        try:
            text = fetch(source, args.timeout)
        except Exception as exc:
            sys.stderr.write("could not scrape %s: %s\n" % (source, exc))
            return 2

    samples = parse_exposition(text)
    if not samples:
        sys.stderr.write("no metrics parsed from %s -- is that a Prometheus "
                         "exposition endpoint?\n" % source)
        return 2
    now = time.time()

    if args.snapshot:
        save_snapshot(args.snapshot, samples, now)

    base = None
    if args.since:
        try:
            base = load_snapshot(args.since)
        except Exception as exc:
            sys.stderr.write("could not read snapshot %s: %s\n"
                             % (args.since, exc))
            return 2

    scrape = Scrape(samples, base=base, timestamp=now)
    report = derive(scrape, serial_io=args.serial_io)

    calibration = None
    if args.calibrate:
        calibration = calibrate_device(args.calibrate,
                                       max_files=args.calibrate_files)
        report["calibration"] = calibration

    vs = verdicts(report, calibration)
    report["verdicts"] = vs
    report["source"] = source

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(render_stdout(report, vs, calibration))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        md = os.path.join(args.out, "results.md")
        js = os.path.join(args.out, "results.json")
        with open(md, "w") as fh:
            fh.write(render_markdown(report, vs, calibration, source) + "\n")
        with open(js, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
        sys.stderr.write("wrote %s and %s\n" % (md, js))

    # Exit status is a signal, like betterbench: 0 = the cache is earning its
    # place, 1 = at least one verdict says otherwise, 2 = could not measure.
    if not report["have_tier_metrics"]:
        return 2
    if any(v["status"] in ("yes",) and "hurting" in v["question"] for v in vs):
        return 1
    for v in vs:
        if v["question"] == "Is the layered cache worth it?" and v["status"] == "no":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
