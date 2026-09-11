#!/usr/bin/env python3
"""kvvalidate — validate the KV-offload disk tier under LIVE load (not a benchmark).

The premise is that identities, not comparisons, are immune to contamination, so
this runs against a live, busy endpoint — the condition most downstream users
will actually have. It re-establishes five invariants that held on 2026-09-12 and
names the current regime. It never decides for the user: the deceptive (mid-fill)
regime is WARNed, never refused.

Read-only. Stdlib only. No GPU, no engine change. It issues exactly one read
(scrape the Prometheus /metrics endpoint) and walks the fs tier path; it never
issues a state-changing request to the endpoint.

TRAPS this tool encodes as guards (each is also stated in the owning function's
docstring):

  * The `tier_load_latency_seconds` histogram is THREAD-SUMMED — it double-counts
    the fs load pool's workers and overstates wall clock by ~7x. The wall clock
    is `tier_load_seconds_total{fs}` ONLY. Never report the latency histogram sum
    as a duration.
  * Judge tier VALUE by `load_bytes` (bytes actually read from disk), never by the
    chunk/token hit counters: those are per-lookup-attempt and repeat on every
    scheduler step (decode inflation), so a high chunk-hit number says nothing
    about real work.
  * Never print a derived quantity as a finding when it is an algebraic identity
    of the measured ones. Every invariant below is checked against independently
    incremented sources, so a hold is a real result, not a tautology.

Usage:
    kvvalidate.py [--metrics-url URL] [--fs-path DIR] [--history FILE]
                 [--stride N] [--expected-active N] [--engine-pid PID] [--json]

Exit 0 when all five invariants hold (warnings do not affect the exit code);
exit 1 when any invariant fails.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

# The on-disk block size, B (25.75 MB/chunk, uniform across all nine groups).
# Used only as a cross-check on the per-file walk; the walk sums real sizes.
EXPECTED_BLOCK_BYTES = 27_000_832

# A single /metrics scrape is NOT atomic across metric families: Prometheus
# builds the response text as it iterates, so a counter rendered later (the
# authority, which increments first) can be a few increments ahead of the
# counters rendered earlier. Under live traffic that shows up as a small,
# one-directional drift, not a defect. The drift bound therefore scales with
# the in-flight work (active requests), never a flat constant on the total.
# These per-unit headrooms are the worst case of how many of a unit
# (a lookup / a token) a single in-flight request can have pending at the
# render gap: a prefill issues one lookup per scheduler step, and one prefill
# step can have a bounded number of lookups / tokens mid-flight.
LOOKUP_INFLIGHT_HEADROOM = 32   # lookups one active request can hold unclassified
TOKEN_INFLIGHT_HEADROOM = 4096  # tokens one active request can hold unclassified


# --------------------------------------------------------------------------- #
# Metrics scrape
# --------------------------------------------------------------------------- #
def fetch_metrics(url, timeout=30):
    """GET the Prometheus /metrics endpoint (read-only) and return the body."""
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_prom(text):
    """Parse Prometheus text format -> {name: [(labels_dict, value), ...]}."""
    out = {}
    label_re = re.compile(r'(\w+)=("([^"]*)"|\S+)')
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, val = line.rpartition(" ")
        if not _:
            continue
        name, _, labels = head.partition("{")
        name = name.strip()
        if not name:
            continue
        lab = {}
        if labels:
            labels = labels.rstrip("}")
            for m in label_re.finditer(labels):
                lab[m.group(1)] = m.group(3) if m.group(3) is not None else m.group(0)
        try:
            out.setdefault(name, []).append((lab, float(val)))
        except ValueError:
            pass
    return out


def get(m, name, **labels):
    """Value of `name` whose labels are a superset of `labels`; None if absent.

    A prometheus_client Counter that has never been touched for a label combo
    emits no sample, so an absent line genuinely means 0.
    """
    for lab, val in m.get(name, ()):
        if all(lab.get(k) == v for k, v in labels.items()):
            return val
    return None


# --------------------------------------------------------------------------- #
# Invariants — each returns dict(name, pass, detail, values). Each carries the
# trap(s) it guards.
# --------------------------------------------------------------------------- #
def inv_lookup_partition(m):
    """The lookup outcome counters partition `lookup_calls` exactly.

    `served`, `zero_hit`, `short_window` and `deferred` are independently
    incremented counters; their sum equaling `lookup_calls` is a real check, not
    an identity. NOTE: `lookup_chunk_hit_total` is per-CHUNK (decode-inflated)
    and is deliberately NOT part of this partition.
    """
    served = get(m, "vllm:kv_offload_lookup_served_total")
    zero = get(m, "vllm:kv_offload_lookup_skip_zero_hit_total")
    short = get(m, "vllm:kv_offload_lookup_skip_short_window_total")
    deferred = get(m, "vllm:kv_offload_lookup_deferred_backend_total")
    calls = get(m, "vllm:kv_offload_lookup_calls_total")
    if None in (served, zero, short, deferred, calls):
        return _fail("lookup_partition", "missing lookup-outcome metric(s)",
                    {"served": served, "zero_hit": zero, "short_window": short,
                     "deferred": deferred, "lookup_calls": calls})
    total = served + zero + short + deferred
    # lookup_calls is the authority (it increments first); the partition sum may
    # lag it by a small margin under live traffic (non-atomic scrape). The
    # bound is proportional to in-flight work, not a magic constant. A drift
    # that is large — or that grows across two scrapes — IS a real signal.
    tol, marker, active = busy_tolerance(m, LOOKUP_INFLIGHT_HEADROOM)
    drift = calls - total
    ok = abs(drift) <= tol
    # Which DIRECTION the drift goes decides how bad it is.
    #
    #   drift > 0  -- lookup_calls ran ahead of the outcome buckets, i.e. some lookups
    #                 exited without recording any outcome. Measured on this box as a
    #                 CONSTANT +32 across repeated scrapes at zero traffic, which rules
    #                 out the non-atomic-scrape race (a race jitters; a constant cannot).
    #                 These are real uninstrumented exits in vLLM's lookup path -- a gap
    #                 in the engine's accounting, not a fault in the operator's cache,
    #                 and it costs correctness nothing. WARN.
    #
    #   drift < 0  -- the buckets sum to MORE than the calls that produced them, i.e. an
    #                 outcome was counted twice. That is corrupted accounting and every
    #                 rate derived from these counters is then suspect. FAIL.
    severity = "warn" if drift > 0 else "fail"
    detail = (f"served {served:.0f} + zero_hit {zero:.0f} + short_window {short:.0f} "
              f"+ deferred {deferred:.0f} = {total:.0f} vs lookup_calls {calls:.0f} "
              f"(drift {drift:+.0f}, bound {tol:.0f} for {active:.0f} active)")
    if not ok and drift > 0:
        detail += (f" — {drift:+.0f} lookups recorded no outcome ({drift / calls * 100:.3f}% "
                   f"of calls); an engine accounting gap, not a cache fault")
    return _mk("lookup_partition", ok, detail,
                {"served": served, "zero_hit": zero, "short_window": short,
                 "deferred": deferred, "lookup_calls": calls, "sum": total,
                 "drift": drift, "tolerance": tol, "active": active},
                marker=marker, severity=severity)


def inv_cpu_equals_external(m):
    """`cpu_hit_tokens` == `external_kv_transfer` (the external bookkeeping reconciles).

    `tier_hit_tokens{cpu}` counts tokens served from the CPU tier;
    `prompt_tokens_by_source{external_kv_transfer}` counts prompt tokens whose
    source was the external tier. The same event increments both, so they must
    agree. (Cross-check: `external_prefix_cache_hits_total` also equals both.)
    The fs tier serves tokens that were first PROMOTED into the CPU tier, so
    external == the CPU tier hit count even though fs did the disk reads.
    """
    cpu = get(m, "vllm:kv_offload_tier_hit_tokens_total", tier="cpu")
    ext = get(m, "vllm:prompt_tokens_by_source_total", source="external_kv_transfer")
    xpc = get(m, "vllm:external_prefix_cache_hits_total")
    if cpu is None or ext is None:
        return _fail("cpu_equals_external", "missing cpu/external token metric",
                    {"cpu_hit_tokens": cpu, "external_kv_transfer": ext})
    # Same non-atomic-scrape hazard as lookup_partition: both counters increment
    # on the same token-served event but are separate metric families, so under
    # live traffic the later-rendered one can lead by the in-flight work.
    tol, marker, active = busy_tolerance(m, TOKEN_INFLIGHT_HEADROOM)
    drift = cpu - ext
    ok = abs(drift) <= tol
    cross = "" if xpc is None else f"  (external_prefix_cache_hits={xpc:.0f})"
    detail = (f"cpu_hit_tokens {cpu:.0f} == external_kv_transfer {ext:.0f}{cross} "
             f"(drift {drift:+.0f}, bound {tol:.0f} for {active:.0f} active)")
    return _mk("cpu_equals_external", ok, detail,
                {"cpu_hit_tokens": cpu, "external_kv_transfer": ext,
                 "external_prefix_cache_hits": xpc, "drift": drift,
                 "tolerance": tol, "active": active},
                marker=marker)


def inv_promotion_refused(m):
    """`promotion_refused == 0` across the promotions that landed.

    The refused counter (and its no_evictable / protected sub-reasons) is a
    never-touched labelled counter here, so it emits no line; absence == 0.
    The CPU tier was pinned at 100% occupancy for 90+ minutes — exactly the
    condition the 'refusal' thesis predicted would produce refusals — yet the
    count stays 0.

    Atomicity audit (2026-09-12): NOT exposed to the cross-family non-atomic
    scrape hazard — it compares a single counter to the constant 0, not a sum of
    separately-rendered families against a fourth, so there is no drift surface.
    """
    refused = get(m, "vllm:kv_offload_promotion_refused_total", tier="fs")
    refused = 0.0 if refused is None else refused
    noev = get(m, "vllm:kv_offload_promotion_refused_no_evictable_total", tier="fs")
    prot = get(m, "vllm:kv_offload_promotion_refused_protected_total", tier="fs")
    initiated = get(m, "vllm:kv_offload_promotion_initiated_total", tier="fs")
    if initiated is None:
        return _fail("promotion_refused", "missing promotion_initiated metric",
                    {"refused": refused, "initiated": initiated})
    ok = refused == 0
    note = " (metric absent = 0)" if get(m, "vllm:kv_offload_promotion_refused_total",
                                        tier="fs") is None else ""
    detail = (f"refused {refused:.0f}{note} / initiated {initiated:.0f}  "
             f"[no_evictable={0.0 if noev is None else noev:.0f}, "
             f"protected={0.0 if prot is None else prot:.0f}]")
    return _mk("promotion_refused", ok, detail,
               {"refused": refused, "initiated": initiated,
                "no_evictable": noev or 0.0, "protected": prot or 0.0})


def walk_fs(fs_path):
    """Walk the fs tier path; return file count, on-disk bytes, modal size, and
    per-group (by the `_g<N>` dir) file counts. Read-only (stat, no file reads)."""
    total_files = 0
    total_bytes = 0
    sizes = {}
    per_group = {}
    for root, _dirs, files in os.walk(fs_path):
        for f in files:
            if not f.endswith(".bin"):
                continue
            p = os.path.join(root, f)
            try:
                sz = os.lstat(p).st_size
            except OSError:
                continue
            total_files += 1
            total_bytes += sz
            sizes[sz] = sizes.get(sz, 0) + 1
            # group index is in the parent dir name: <hh>_g<N>
            base = os.path.basename(root)
            mm = re.search(r"_g(\d+)$", base)
            if mm:
                g = int(mm.group(1))
                per_group[g] = per_group.get(g, 0) + 1
    modal = max(sizes.items(), key=lambda kv: kv[1])[0] if sizes else 0
    return total_files, total_bytes, modal, per_group


def inv_disk_vs_engine(m, fs_path):
    """On-disk (file count x block size) ~= engine's `tier_used_bytes{fs}`.

    Layout: `<base>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin`, every file a
    uniform 27,000,832 B (25.75 MB/chunk). The check reconciles the engine's
    self-reported used-bytes gauge against a physical walk of the fs tier.

    Atomicity audit (2026-09-12): exposed to write-lag (files landing between
    the walk and the gauge render) — handled by the RELATIVE (0.02) deviation
    allowance below, which is the pattern the counter invariants copy.
    """
    used = get(m, "vllm:kv_offload_tier_used_bytes", tier="fs")
    nfiles, nbytes, modal, _ = walk_fs(fs_path)
    if used is None:
        return _fail("disk_vs_engine", "missing tier_used_bytes{fs}",
                    {"on_disk_bytes": nbytes, "files": nfiles, "engine_used": used})
    cross = nfiles * modal
    ok = abs(nbytes - used) / used < 0.02
    detail = (f"on-disk {nfiles:,} x {modal:,}B = {nbytes/1e9:.1f} GB "
             f"vs engine tier_used_bytes {used/1e9:.1f} GB "
             f"(file_count x block = {cross/1e9:.1f} GB, "
             f"dev {abs(nbytes-used)/used*100:.2f}%)")
    return _mk("disk_vs_engine", ok, detail,
               {"on_disk_bytes": nbytes, "files": nfiles, "block_bytes": modal,
                "file_x_block": cross, "engine_used_bytes": used,
                "deviation_pct": abs(nbytes - used) / used * 100})


def inv_group_ratio(m, fs_path, stride):
    """Per-group file ratio ~= the mamba store stride.

    The six Mamba/GDN groups (g0-g5) store 1-in-`stride` chunks; the full-
    attention (g6,g7) and EAGLE/MTP (g8) groups store every chunk, so the
    dense:sparsE file-count ratio ~= stride. The excess over `stride` is the
    dead-zone sequences that stored zero Mamba snapshots. We take the densest
    group over the sparsest to avoid hard-coding which groups are Mamba.

    Atomicity audit (2026-09-12): NOT exposed — both sides are taken from the
    single os.walk snapshot (self-consistent) against a physical stride
    constant; there is no cross-family scrape drift surface.
    """
    nfiles, _nb, _modal, per_group = walk_fs(fs_path)
    if not per_group:
        return _fail("group_ratio", "no per-group .bin files found", {"groups": per_group})
    dense = max(per_group.values())
    sparse = min(per_group.values())
    ratio = dense / sparse
    ok = abs(ratio - stride) / stride < 0.20
    gd = max(per_group, key=per_group.get)
    gs = min(per_group, key=per_group.get)
    detail = (f"dense g{gd}={dense:,} / sparse g{gs}={sparse:,} = {ratio:.2f} "
             f"vs stride {stride}  (per-group {per_group})")
    return _mk("group_ratio", ok, detail,
               {"per_group": per_group, "dense": dense, "sparse": sparse,
                "ratio": ratio, "stride": stride})


# --------------------------------------------------------------------------- #
# Regime + shared-client detection (WARN, never refuse)
# --------------------------------------------------------------------------- #
def _history_trend(history_path, column, n=2):
    """Delta of `column` between the last n CSV rows, or None if unavailable."""
    if not history_path or not os.path.exists(history_path):
        return None
    try:
        with open(history_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    rows = [r for r in rows if r.get(column) not in (None, "")]
    if len(rows) < n:
        return None
    try:
        vals = [float(r[column]) for r in rows[-n:]]
        return vals[-1] - vals[0]
    except (ValueError, KeyError):
        return None


def detect_regime(m, history_path):
    """Name the regime: saturated / post-boot (hot disk) / mid-fill (the liar).

    * saturated  — cpu_free == 0. Working regime; no warning.
    * post-boot  — cpu_free > 0 with fs_load_ops RISING (fresh traffic hitting a
      cold CPU tier while the disk is hot). fs genuinely serves; no warning.
    * mid-fill   — cpu_free > 0 with fs_load_ops FLAT. THE LIAR: the CPU tier
      holds everything recent, the fs tier is never consulted, and the numbers
      look dead while the tier is merely still warming up. WARN hard.
    """
    cpu_free = get(m, "vllm:kv_offload_cpu_cache_free_perc")
    if cpu_free is None:
        return "unknown", ["cannot determine regime: cpu_cache_free_perc missing"]
    warnings = []
    if cpu_free < 1e-6:
        return "saturated", warnings
    trend = _history_trend(history_path, "fs_load_ops")
    if trend is None:
        # cpu_free>0 but no trend: cannot distinguish post-boot from mid-fill.
        # Default to the safe (warn) reading rather than deciding for the user.
        warnings.append("regime: cpu_free>0 but fs_load_ops trend unavailable "
                       "(no history); cannot distinguish post-boot from mid-fill — "
                       "treating as mid-fill risk")
        return "mid-fill (unverified)", warnings
    if trend > 0:
        return "post-boot / hot disk", warnings
    warnings.append("regime: MID-FILL (the liar) — the fs tier will not serve "
                   "until the CPU tier saturates; these numbers are not "
                   "representative")
    return "mid-fill", warnings


def detect_multiclient(m, expected_active):
    """WARN (never refuse) when more clients than declared are active."""
    running = get(m, "vllm:num_requests_running") or 0.0
    waiting = get(m, "vllm:num_requests_waiting") or 0.0
    active = running + waiting
    warnings = []
    if active > expected_active:
        warnings.append(f"shared endpoint: req_running={running:.0f} + "
                       f"req_waiting={waiting:.0f} = {active:.0f} active "
                       f"(> expected {expected_active:.0f}); numbers may include "
                       "other clients")
    return active, running, waiting, warnings


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def busy_tolerance(m, per_unit):
    """Drift bound for a cross-family identity under LIVE load, proportional to
    the in-flight work (active requests), plus the [SOLID]/[REGIME] marker.

    Returns (tolerance, marker, inflight). When the engine is idle (no active
    requests) the check is exact — a cross-family drift of more than 1.0 is a
    real signal. When busy, the bound scales with the in-flight work, so a
    healthy system does not false-FAIL; the reported drift is the signal.
    """
    active = (get(m, "vllm:num_requests_running") or 0.0) + \
             (get(m, "vllm:num_requests_waiting") or 0.0)
    tol = max(1.0, active * per_unit)
    marker = "SOLID" if active < 0.5 else "REGIME"
    return tol, marker, active


def _mk(name, ok, detail, values, marker="SOLID", severity="fail"):
    """severity: what a violation of THIS invariant means.

    "fail" -- the cache is doing something wrong; the result cannot be trusted.
    "warn" -- something is off in vLLM's own accounting, but the cache is serving
              correctly. A downstream user running this tool to answer "is my cache
              working?" must not be told FAIL for a defect in the engine's counters
              that costs them nothing. Reserve FAIL for incorrectness.
    """
    return {"name": name, "pass": bool(ok), "detail": detail, "values": values,
            "marker": marker, "severity": "fail" if ok else severity}


def _fail(name, detail, values):
    return {"name": name, "pass": False, "detail": f"BLOCKED: {detail}",
            "values": values, "marker": "UNRESOLVED"}


def tier_value_section(m):
    """Report tier VALUE by load_bytes (the trap: never the hit counters for
    value; never the thread-summed latency histogram for wall clock)."""
    load_bytes = get(m, "vllm:kv_offload_tier_load_bytes_total", tier="fs")
    wall = get(m, "vllm:kv_offload_tier_load_seconds_total", tier="fs")
    # thread-summed latency histogram sum — captured ONLY to expose the 7x
    # overstatement, never reported as a duration.
    lat = get(m, "vllm:kv_offload_tier_load_latency_seconds_sum", tier="fs")
    hit_tok = get(m, "vllm:kv_offload_tier_hit_tokens_total", tier="fs")
    hit_blk = get(m, "vllm:kv_offload_tier_hit_blocks_total", tier="fs")
    return {
        "fs_load_bytes": load_bytes,
        "fs_wall_seconds": wall,
        "fs_load_ops": get(m, "vllm:kv_offload_tier_load_ops_total", tier="fs"),
        "fs_hit_tokens_decode_inflated": hit_tok,
        "fs_hit_blocks_decode_inflated": hit_blk,
        "latency_histogram_overstates_wall_by":
            (round(lat / wall, 2) if (lat and wall and wall > 0) else None),
    }


def prompt_source_section(m):
    """The three measured prompt-token sources and their (derived) shares."""
    tot = get(m, "vllm:prompt_tokens_total")
    lc = get(m, "vllm:prompt_tokens_by_source_total", source="local_compute")
    ch = get(m, "vllm:prompt_tokens_by_source_total", source="local_cache_hit")
    ek = get(m, "vllm:prompt_tokens_by_source_total", source="external_kv_transfer")
    s = {}
    for k, v in (("local_compute", lc), ("local_cache_hit", ch),
                 ("external_kv_transfer", ek)):
        s[k] = v
    if tot:
        s["shares"] = {k: (v / tot if v is not None else None)
                      for k, v in (("local_compute", lc),
                                  ("local_cache_hit", ch),
                                  ("external_kv_transfer", ek))}
    s["prompt_tokens_total"] = tot
    return s


def _fanout_marker(thread, wall, n_read_threads):
    """[SOLID] while the measured pool concurrency (thread/wall) is close to the
    configured read-thread count (an evenly fanned, concurrently running pool);
    [REGIME] otherwise. The fanout estimate (and every figure built on it)
    degrades if fanout is disabled or n_read_threads is wrong."""
    if not wall or wall <= 0:
        return "UNRESOLVED", None
    ratio = thread / wall
    close = abs(ratio - n_read_threads) / n_read_threads <= 0.25
    return ("SOLID" if close else "REGIME"), ratio


def performance_section(m, cfg):
    """The PERFORMANCE section: answer 'how well is my cache working?'

    Every figure carries a confidence marker; the tool refuses to print a rate
    without one, and refuses to derive ANY rate from the thread-summed
    `tier_load_latency_seconds` histogram (see the trap note + refusal guard).

    markers:
      SOLID      — wall-clock, single-source, no shared numerator/denominator
      REGIME     — true, but only in the named regime (the fanout assumption)
      UNRESOLVED — the instrument cannot settle it; names the measurement that
                   would
    """
    n_read_threads = cfg["n_read_threads"]
    max_model_len = cfg["max_model_len"]
    maxseqs = cfg["maxseqs"]
    concurrent_agents = cfg["concurrent_agents"]

    fs_wall = get(m, "vllm:kv_offload_tier_load_seconds_total", tier="fs")
    cpu_wall = get(m, "vllm:kv_offload_tier_load_seconds_total", tier="cpu")
    fs_thread = get(m, "vllm:kv_offload_tier_load_thread_seconds_total", tier="fs")
    fs_lat = get(m, "vllm:kv_offload_tier_load_latency_seconds_sum", tier="fs")
    fs_hit = get(m, "vllm:kv_offload_tier_hit_tokens_total", tier="fs")
    cpu_hit = get(m, "vllm:kv_offload_tier_hit_tokens_total", tier="cpu")
    fs_bytes = get(m, "vllm:kv_offload_tier_load_bytes_total", tier="fs")
    cpu_bytes = get(m, "vllm:kv_offload_tier_load_bytes_total", tier="cpu")
    fs_cap = get(m, "vllm:kv_offload_tier_capacity_bytes", tier="fs")
    cpu_cap = get(m, "vllm:kv_offload_tier_capacity_bytes", tier="cpu")
    prefill_time = get(m, "vllm:request_prefill_time_seconds_sum")
    prefill_tok = get(m, "vllm:request_prefill_kv_computed_tokens_sum")

    # --- TRAP GUARD: refuse to derive any rate from the latency histogram ---
    # It is byte-identical to the thread-summed counter (a 7x+ overstatement of
    # wall clock). Detect the identity and state the refusal explicitly.
    lat_refusal = None
    if fs_lat and fs_thread and fs_thread > 0 and abs(fs_lat - fs_thread) / fs_thread < 0.01:
        lat_refusal = {
            "refused": True,
            "reason": "tier_load_latency_seconds_sum is byte-identical to "
                     "tier_load_thread_seconds_total (thread-summed, "
                     f"{(fs_lat / fs_wall if fs_wall else 0):.2f}x overstatement of wall clock) "
                     "- no rate is derived from it",
            "latency_sum": fs_lat, "thread_sum": fs_thread,
            "ratio_to_wall": round(fs_lat / fs_wall, 2) if fs_wall else None,
        }
    else:
        lat_refusal = {"refused": False, "latency_sum": fs_lat,
                        "thread_sum": fs_thread}

    def _bpt(b, t):
        return (b / t) if (b and t and t > 0) else None

    fs_bpt = _bpt(fs_bytes, fs_hit)
    cpu_bpt = _bpt(cpu_bytes, cpu_hit)
    bytes_per_token = {
        "fs": fs_bpt, "cpu": cpu_bpt, "marker": "SOLID",
        "detail": "tier_load_bytes / tier_hit_tokens (model-specific; the figure "
                 "the sizing guidance is built on - handed back measured, not assumed)",
    }

    # Token read speed: request-experienced (wall clock, includes queue wait).
    def _rate(tok, wall):
        return (tok / wall) if (tok is not None and wall and wall > 0) else None
    fs_rate = _rate(fs_hit, fs_wall)
    cpu_rate = _rate(cpu_hit, cpu_wall)
    read_speed = [
        {"tier": "fs", "tok_s": fs_rate,
         "mb_s": (fs_rate * fs_bpt / 1e6) if (fs_rate and fs_bpt) else None,
         "marker": "SOLID", "label": "as a request sees it (submit->complete, includes the wait)"},
        {"tier": "cpu", "tok_s": cpu_rate,
         "mb_s": (cpu_rate * cpu_bpt / 1e6) if (cpu_rate and cpu_bpt) else None,
         "marker": "SOLID", "label": "as a request sees it (submit->complete, includes the wait)"},
    ]

    # Device capability (transfer only) + pool concurrency + wait fraction.
    # transfer_estimate = thread_seconds / n_read_threads (the even-fanout model).
    dev_marker, pool_ratio = _fanout_marker(fs_thread, fs_wall, n_read_threads)
    transfer = (fs_thread / n_read_threads) if (fs_thread and n_read_threads) else None
    fs_dev = _rate(fs_hit, transfer) if transfer else None
    device_capability = [{
        "tier": "fs", "tok_s": fs_dev,
        "mb_s": (fs_dev * fs_bpt / 1e6) if (fs_dev and fs_bpt) else None,
        "marker": dev_marker, "label": "device capability (transfer only)",
        "note": "estimate = thread_seconds / n_read_threads; holds only while the job "
               f"is fanned evenly across {n_read_threads} concurrently-running threads",
    }]
    wait_frac = (1 - transfer / fs_wall) if (transfer and fs_wall and fs_wall > 0) else None

    def _wait_advice(wf):
        if wf is None:
            return "n/a (insufficient data)"
        if wf < 0.10:
            return "low wait: device-bound - a faster disk helps; more threads will not"
        if wf > 0.40:
            return "high wait: queue-bound - more read threads / wider fanout help; a " \
                   "faster disk is wasted money"
        return "mid wait: mixed device- and queue-bound"
    wait_diagnostic = {
        "wait_fraction": wait_frac, "n_read_threads": n_read_threads,
        "pool_concurrency": pool_ratio, "marker": dev_marker,
        "advice": _wait_advice(wait_frac),
    }

    # The prefill comparison - UNRESOLVED: the denominator is a sum over
    # overlapping requests (the same trap as thread_seconds). Print the RANGE,
    # never a point estimate, and name what would settle it.
    prefill_pt = _rate(prefill_tok, prefill_time)
    prefill_lo = prefill_pt if prefill_pt is not None else None
    prefill_hi = (prefill_pt * maxseqs) if prefill_pt is not None else None
    disk_multiple = None
    if fs_rate is not None and prefill_lo and prefill_hi:
        disk_multiple = {"vs_recompute": [round(fs_rate / prefill_hi, 2),
                                          round(fs_rate / prefill_lo, 2)]}
    prefill_comparison = {
        "marker": "UNRESOLVED",
        "point_estimate": prefill_pt, "range": [prefill_lo, prefill_hi],
        "maxseqs": maxseqs, "disk_vs_recompute": disk_multiple,
        "reason": "request_prefill_time_seconds is a sum over overlapping requests "
                 f"(up to MAXSEQS={maxseqs}); the true single-stream prefill is unknown",
        "settled_by": "a controlled single-stream wall-clock prefill run",
        "note": "the older '101 MB/s break-even' is this same number in other units "
               "(prefill_pt x bytes/token) - not independent corroboration",
    }

    # The CPU tier verdict - SOLID: its wall-clock rate beats even the most
    # pessimistic prefill bound. The strongest claim the tool makes.
    cpu_multiple = None
    if cpu_rate is not None and prefill_hi:
        cpu_multiple = {"vs_most_pessimistic_prefill": round(cpu_rate / prefill_hi, 1)}
    cpu_verdict = {
        "marker": "SOLID", "tok_s": cpu_rate,
        "beats_pessimistic_prefill_by": cpu_multiple,
        "detail": "cpu read rate beats even the worst-case prefill estimate",
    }

    # Effective tier capacity (tokens + full contexts) + the concurrency cliff.
    def _cap(cap_bytes, bpt):
        if cap_bytes is None or bpt is None or bpt <= 0 or max_model_len <= 0:
            return {"tokens": None, "contexts": None}
        tok = cap_bytes / bpt
        return {"tokens": tok, "contexts": tok / max_model_len}
    fs_capd = _cap(fs_cap, fs_bpt)
    cpu_capd = _cap(cpu_cap, cpu_bpt)
    cliff = None
    if cpu_capd["contexts"] is not None:
        threshold = concurrent_agents + 1
        cliff = {"cpu_contexts": cpu_capd["contexts"], "threshold": threshold,
                 "hit": cpu_capd["contexts"] < threshold,
                 "detail": f"cpu tier holds {cpu_capd['contexts']:.2f} contexts "
                          f"vs {threshold} needed (concurrent_agents {concurrent_agents} + 1)"
                          " - the measured concurrency cliff"}
    capacity = {"fs": fs_capd, "cpu": cpu_capd, "cliff": cliff, "marker": "SOLID"}

    shares_note = ("prompt-source shares move hard with workload: a RISING gpu "
                   "(local_cache_hit) share is the GPU cache doing its job, NOT the "
                   "tier failing - do not read a falling external share as a regression "
                   "(rates beat shares)")

    return {
        "latency_histogram_refusal": lat_refusal,
        "bytes_per_token": bytes_per_token,
        "read_speed_request_experienced": read_speed,
        "device_capability": device_capability,
        "wait_diagnostic": wait_diagnostic,
        "pool_concurrency": pool_ratio,
        "prefill_comparison": prefill_comparison,
        "cpu_verdict": cpu_verdict,
        "capacity": capacity,
        "shares_note": shares_note,
    }


def detect_engine_pid(arg):
    if arg:
        return arg
    try:
        out = subprocess.run(["pgrep", "-f", "vllm"], capture_output=True,
                            text=True, timeout=10).stdout.split()
        return out[0] if out else "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="validate the KV-offload fs tier")
    ap.add_argument("--metrics-url",
                   default="http://127.0.0.1:1234/upstream/qwen3.8-27b-vllm/metrics")
    ap.add_argument("--fs-path", default="/kvcache/blocks")
    ap.add_argument("--history", default=None,
                   help="metrics-history.csv for regime trend (auto-searched if omitted)")
    ap.add_argument("--stride", type=int, default=8,
                   help="RADIANCE_MAMBA_STORE_STRIDE, for the group-ratio check")
    ap.add_argument("--expected-active", type=int, default=1,
                   help="max active requests the caller declares; more -> warn")
    ap.add_argument("--engine-pid", default=None, help="engine PID for provenance")
    ap.add_argument("--n-read-threads", type=int, default=8,
                    help="fs read-pool fanout (from kv_connector_extra_config); "
                         "the divisor for the transfer estimate")
    ap.add_argument("--max-model-len", type=int, default=204800,
                    help="max_model_len, for the full-contexts capacity figure")
    ap.add_argument("--maxseqs", type=int, default=4,
                    help="max concurrent prefill sequences, bounds the prefill range")
    ap.add_argument("--concurrent-agents", type=int, default=2,
                    help="concurrent agents; the cpu-tier cliff fires below (this+1) contexts")
    ap.add_argument("--json", action="store_true", help="machine output")
    a = ap.parse_args()

    # auto-search for the history csv
    if a.history is None:
        for cand in ("metrics-history.csv",
                     "$HOME/work/kv-queue/metrics-history.csv"):
            if os.path.exists(cand):
                a.history = cand
                break

    t0 = time.time()
    try:
        m = parse_prom(fetch_metrics(a.metrics_url))
    except Exception as e:
        return _emit({"blocked": f"could not scrape metrics at {a.metrics_url}: {e}"},
                     a.json)

    invariants = [
        inv_lookup_partition(m),
        inv_cpu_equals_external(m),
        inv_promotion_refused(m),
        inv_disk_vs_engine(m, a.fs_path),
        inv_group_ratio(m, a.fs_path, a.stride),
    ]
    # A warn-severity violation does not fail the run: the tool's headline answers
    # "is my cache working?", and an engine accounting gap does not make it not work.
    hard_fail = [i for i in invariants if not i["pass"] and i.get("severity") != "warn"]
    soft_warn = [i for i in invariants if not i["pass"] and i.get("severity") == "warn"]
    regime, rwarn = detect_regime(m, a.history)
    active, running, waiting, mwarn = detect_multiclient(m, a.expected_active)
    warnings = rwarn + mwarn + [f"{i['name']}: {i['detail']}" for i in soft_warn]
    all_pass = not hard_fail
    cfg = {"n_read_threads": a.n_read_threads, "max_model_len": a.max_model_len,
           "maxseqs": a.maxseqs, "concurrent_agents": a.concurrent_agents}

    hist_ts = None
    if a.history and os.path.exists(a.history):
        try:
            with open(a.history, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                hist_ts = rows[-1].get("ts")
        except Exception:
            pass

    result = {
        "ok": all_pass,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 2),
        "metrics_url": a.metrics_url,
        "fs_path": a.fs_path,
        "engine_pid": detect_engine_pid(a.engine_pid),
        "history_last_ts": hist_ts,
        "regime": regime,
        "req_running": running,
        "req_waiting": waiting,
        "active": active,
        "invariants": invariants,
        "warnings": warnings,
        "config": cfg,
        "prompt_tokens": prompt_source_section(m),
        "tier_value": tier_value_section(m),
        "performance": performance_section(m, cfg),
    }
    return _emit(result, a.json, all_pass)


def _emit(result, as_json, all_pass=True):
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        okc = "PASS" if result.get("ok") else "FAIL"
        if result.get("ok") and any(i.get("severity") == "warn" and not i["pass"]
                                    for i in result.get("invariants", [])):
            okc = "PASS (with warnings)"
        print(f"kvvalidate — {okc}  ({result.get('timestamp')})")
        print(f"  endpoint: {result.get('metrics_url')}")
        print(f"  fs path : {result.get('fs_path')}   engine pid: "
              f"{result.get('engine_pid')}   history: {result.get('history_last_ts') or 'n/a'}")
        print()
        print("  invariants")
        for i in result.get("invariants", []):
            tag = "PASS" if i["pass"] else ("WARN" if i.get("severity") == "warn" else "FAIL")
            print(f"    [{tag}][{i.get('marker','SOLID'):<10}] "
                  f"{i['name']:<20} {i['detail']}")
        print()
        print(f"  regime: {result.get('regime')}   "
              f"(req_running={result.get('req_running')}, "
              f"req_waiting={result.get('req_waiting')})")
        w = result.get("warnings", [])
        if w:
            print("  warnings:")
            for x in w:
                print(f"    ! {x}")
        else:
            print("  warnings: none")
        print()
        pt = result.get("prompt_tokens", {})
        if pt:
            sh = pt.get("shares", {})
            print(f"  prompt tokens (total {pt.get('prompt_tokens_total') or 0:,.0f}):")
            for k in ("local_compute", "local_cache_hit", "external_kv_transfer"):
                v = pt.get(k)
                print(f"    {k:<22} {v:,.0f}" if v is not None
                      else f"    {k:<22} n/a")
            if sh:
                print(f"    shares (derived): "
                      f"local_compute {sh.get('local_compute', 0)*100:.1f}%  "
                      f"local_cache_hit {sh.get('local_cache_hit', 0)*100:.1f}%  "
                      f"external_kv_transfer {sh.get('external_kv_transfer', 0)*100:.1f}%")
        tv = result.get("tier_value", {})
        if tv:
            print("  tier value (judged by load_bytes; wall clock = "
                  "tier_load_seconds_total{fs} only — the tier_load_latency_seconds "
                  "histogram is thread-summed and overstates wall clock by "
                  f"{tv.get('latency_histogram_overstates_wall_by') or 'n/a'}x):")
            print(f"    fs load_bytes   {tv.get('fs_load_bytes') or 0:,.0f} "
                  f"over {tv.get('fs_load_ops') or 0:,.0f} load ops in "
                  f"{tv.get('fs_wall_seconds') or 0:.2f}s")
            print(f"    (decode-inflated hit counters, for reference only: "
                  f"{tv.get('fs_hit_tokens_decode_inflated') or 0:,.0f} tokens / "
                  f"{tv.get('fs_hit_blocks_decode_inflated') or 0:,.0f} blocks)")
        _emit_performance(result.get("performance", {}),
                         result.get("config", {}))
    return 0 if all_pass else 1


def _fmt(v, spec=",.0f"):
    return format(v, spec) if v is not None else "n/a"


def _emit_performance(p, cfg):
    """Render the PERFORMANCE section; every line carries a confidence marker."""
    print()
    print("  performance (every figure carries a confidence marker)")
    print(f"    config: n_read_threads={cfg.get('n_read_threads')}  "
          f"max_model_len={cfg.get('max_model_len')}  "
          f"maxseqs={cfg.get('maxseqs')}  "
          f"concurrent_agents={cfg.get('concurrent_agents')}")

    lr = p.get("latency_histogram_refusal", {})
    if lr.get("refused"):
        print(f"    [REFUSAL]  {lr.get('reason')}")

    bpt = p.get("bytes_per_token", {})
    print(f"    [{bpt.get('marker')}]  bytes/token (measured)  "
          f"fs {_fmt(bpt.get('fs'))}  cpu {_fmt(bpt.get('cpu'))}")

    print(f"    [SOLID]  token read speed — as a request sees it "
          f"(submit->complete, includes queue wait):")
    for r in p.get("read_speed_request_experienced", []):
        print(f"             {r['tier']:<4} {_fmt(r['tok_s'])} tok/s  "
              f"({_fmt(r['mb_s'])} MB/s)")
    for d in p.get("device_capability", []):
        print(f"    [{d.get('marker')}]  device capability (transfer only)  "
              f"{d['tier']:<4} {_fmt(d['tok_s'])} tok/s  "
              f"({_fmt(d.get('mb_s'))} MB/s)   {d.get('note','')}")

    wd = p.get("wait_diagnostic", {})
    wf = wd.get("wait_fraction")
    print(f"    [{wd.get('marker')}]  promotion: wait+poll lag "
          f"{(f'{wf*100:.1f}%' if wf is not None else 'n/a')}  "
          f"(pool concurrency {_fmt(wd.get('pool_concurrency'), '.2f')} / "
          f"{wd.get('n_read_threads')} threads)  -> {wd.get('advice','')}")

    pc = p.get("prefill_comparison", {})
    rng = pc.get("range") or [None, None]
    print(f"    [{pc.get('marker')}]  prefill (unresolved)  "
          f"{_fmt(rng[0])} - {_fmt(rng[1])} tok/s  "
          f"(point est {_fmt(pc.get('point_estimate'))}, "
          f"denominator is a sum over {pc.get('maxseqs')} overlapping requests)")
    dm = (pc.get("disk_vs_recompute") or {}).get("vs_recompute")
    if dm:
        print(f"             disk tier is {dm[0]}x - {dm[1]}x recompute")
    print(f"             settled by: {pc.get('settled_by','')}")
    print(f"             note: {pc.get('note','')}")

    cv = p.get("cpu_verdict", {})
    mm = (cv.get("beats_pessimistic_prefill_by") or {}).get(
        "vs_most_pessimistic_prefill")
    print(f"    [{cv.get('marker')}]  cpu tier verdict  {_fmt(cv.get('tok_s'))} tok/s "
          f"beats the worst-case prefill by {mm if mm is not None else 'n/a'}x  "
          f"(strongest claim)")

    cap = p.get("capacity", {})
    fsc = cap.get("fs", {})
    cpuc = cap.get("cpu", {})
    print(f"    [{cap.get('marker')}]  effective capacity  "
          f"fs {_fmt(fsc.get('tokens'))} tok / {_fmt(fsc.get('contexts'), '.1f')} ctx   "
          f"cpu {_fmt(cpuc.get('tokens'))} tok / "
          f"{_fmt(cpuc.get('contexts'), '.2f')} ctx")
    cliff = cap.get("cliff")
    if cliff:
        flag = "CLIFF HIT" if cliff.get("hit") else "above cliff"
        print(f"             concurrency cliff: {flag}  ({cliff.get('detail','')})")

    sn = p.get("shares_note")
    if sn:
        print(f"    [note] {sn}")


if __name__ == "__main__":
    sys.exit(main())
