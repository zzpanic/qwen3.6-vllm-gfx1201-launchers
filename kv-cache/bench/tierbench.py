#!/usr/bin/env python3
"""tierbench -- deterministic KV tier attribution bench for qwen3.8-27b-vllm.

WHY THIS EXISTS
---------------
`~/audit/stress/harness.py` established that an offload read-back costs ~78 s against
2.15 s for a GPU hit, but it could not say WHICH tier served the read-back. Its EVICT-1
phase pushed 27.19 GB through a 16 GiB CPU primary tier, so the prefix was always evicted
all the way to the filesystem: its RE-READ-CPU, RE-READ-FS and RE-READ-FS2 phases were
three measurements of the same fs read (77.0 / 78.5 / 78.0 s -- indistinguishable, as they
must be). This bench sizes each eviction so the prefix lands in a KNOWN tier, and refuses
to report a phase whose tier state did not come out as intended.

Four tier states, one prefix, measured the same way:

    COLD  prefix never stored anywhere        -> true recompute cost
    GPU   prefix resident in the GPU pool     -> the 2.15 s floor
    CPU   evicted from GPU, still in CPU tier -> never yet measured on this box
    FS    evicted from CPU, still on disk     -> the ~78 s case

plus an optional CONCURRENT phase that reproduces the multi-client regime (several long
conversations converging on the 138-block pool) which is where the 151 preemptions in
`~/vllm-kv-cache-offload-reuse.md` came from.

WHAT "DETERMINISTIC" MEANS HERE
-------------------------------
Prompt SHAPE is fixed: every run builds the same phases, at the same token counts, in the
same order, with the same sampling parameters (temperature 0, max_tokens 16, no min_p --
this build 400s on min_p under spec decoding).

Prompt CONTENT is salted per run, and it has to be. If the eviction filler were identical
between runs, run 2 would find it already cached and evict nothing; if the measured prefix
were identical, run 2's COLD phase would be served from run 1's fs tier. A run is replayed
exactly by passing its salt back with --salt, which is the right way to re-measure the same
content in a different cache state.

So: timings are comparable across runs, content is not. That is the only definition of
determinism that survives contact with a cache.

ACCEPTANCE TEST FOR THE R2.9.2 INSTRUMENTATION
----------------------------------------------
Run this BEFORE and AFTER the promotion-leg patch (see R2.9.2 of
cache-preemption-patch-plan.md). Before the patch, CPU and FS are indistinguishable in the
metrics -- no counter separates them. After it, the patch is correct if and only if:

    fs_load_seconds delta  ~= 0     in the CPU phase
    fs_load_seconds delta  >  0     in the FS phase
    fs_load_seconds sum    <  the FS phase wall time

and the residue (wall - fs_load - cpu_to_gpu_load) is the part R2.2 says nobody has
measured yet. That residue is the number the whole plan turns on.

USAGE
-----
    ./tierbench.py --yes                     full run, ~15-25 min
    ./tierbench.py --yes --phases cold,gpu   quick subset
    ./tierbench.py --yes --concurrent 3      add the preemption phase
    ./tierbench.py --dry-run                 sizing + detection only, no load
    ./tierbench.py --yes --salt 7a3f9c       replay a previous run's content

THIS EVICTS THE ENTIRE KV CACHE. It is not safe to run against a server someone is using;
every conversation resident at the time gets pushed to disk and will pay a read-back. It
does not need a GPU exclusive window (it drives the HTTP endpoint), but it does need the
box to itself for the duration.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- configuration

MODEL = os.environ.get("TIERBENCH_MODEL", "qwen3.8-27b-vllm")
CONTAINER = os.environ.get("TIERBENCH_CONTAINER", "qwen38-27b-vllm")
OUT_DIR = os.environ.get("TIERBENCH_OUT", "$HOME/audit/stress/tierbench")

# Endpoints tried in order. The llama-swap proxy path is stable across restarts; the
# direct upstream port is not (it moves with the entry), so it is only the fallback.
ENDPOINT_CANDIDATES = [
    "http://127.0.0.1:1234/upstream/" + MODEL,
    "http://127.0.0.1:5804",
]

PER_REQ_TIMEOUT = 900          # a cold 200k-token prefill is minutes, not seconds
RUNTIME_CAP_S = 5400           # hard stop; a hung phase must not run overnight
MAX_TOKENS = 16                # the measurement is prefill; decode is noise
TARGET_PREFIX_TOKENS = 90000   # matches the harness so the numbers are comparable
EVICT_MARGIN = 1.20            # push 1.2x a tier's capacity to be sure it turned over
EVICT_CHUNK_TOKENS = 30000     # eviction traffic is many small requests, not one huge one

# Metrics we care about. Counters are read as _total, histograms as (_sum, _count).
COUNTERS = [
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_load_time_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:kv_offload_store_time_total",
    "vllm:kv_offload_allocation_failure_total",
    # R2.9.2 additions -- absent before the patch, and that is not an error.
    "vllm:kv_offload_fs_load_bytes_total",
    "vllm:kv_offload_tiering_promotion_abandoned_total",
]
HISTOGRAMS = [
    "vllm:kv_offload_tiering_lookup_async_delay_seconds",
    "vllm:kv_offload_tiering_lookup_sync_delay_seconds",
    "vllm:kv_offload_load_size",
    # R2.9.2 additions.
    "vllm:kv_offload_fs_load_seconds",
    "vllm:kv_offload_fs_store_seconds",
    "vllm:kv_offload_fs_retry_rounds",
]
GAUGES = [
    "vllm:kv_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_usage_perc",
]

# Deterministic filler. Distinct sentences so the tokenizer cannot collapse them, and no
# repeated 100-token window that prefix caching could share between two "unique" blocks.
_WORDS = (
    "harbour lantern quartz meadow bramble cinder thistle fathom gallery kestrel "
    "marble nectar obsidian plover ravine sextant tundra vellum walnut zephyr "
    "anvil basalt copper drifts ember fescue granite hollow indigo juniper"
).split()


class Abort(Exception):
    pass


# ---------------------------------------------------------------- plumbing

_start = time.time()
_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        sys.stderr.write("[%7.1fs] %s\n" % (time.time() - _start, msg))
        sys.stderr.flush()


def budget_check():
    if time.time() - _start > RUNTIME_CAP_S:
        raise Abort("runtime cap of %ds exceeded" % RUNTIME_CAP_S)


def http(url, payload=None, timeout=30, raw=False):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return body if raw else json.loads(body)


def detect_endpoint():
    for base in ENDPOINT_CANDIDATES:
        try:
            http(base + "/metrics", timeout=5, raw=True)
            return base
        except Exception:
            continue
    raise Abort("no endpoint responded; tried %s" % ", ".join(ENDPOINT_CANDIDATES))


# ---------------------------------------------------------------- capacity detection

def detect_capacity():
    """Read tier capacities out of the engine's own boot log.

    Two lines carry them:
        kv_cache_utils.py:2296  GPU KV cache size: 228,737 tokens
        spec.py:226             ... primary tier (lru, 636 blocks) and 1 secondary tier(s)

    Both are overridable by env, because a config change moves them and a bench that
    silently sizes its evictions off a stale constant produces confident nonsense.
    """
    gpu_tokens = os.environ.get("TIERBENCH_GPU_TOKENS")
    cpu_blocks = os.environ.get("TIERBENCH_CPU_BLOCKS")
    source = "env"
    if not (gpu_tokens and cpu_blocks):
        source = "boot log (%s)" % CONTAINER
        try:
            r = subprocess.run(
                ["podman", "logs", CONTAINER],
                capture_output=True, text=True, timeout=120,
            )
            out = r.stdout + r.stderr
        except Exception as e:
            raise Abort("cannot read %s boot log (%s); set TIERBENCH_GPU_TOKENS "
                        "and TIERBENCH_CPU_BLOCKS" % (CONTAINER, e))
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", out)
        if m and not gpu_tokens:
            gpu_tokens = m.group(1).replace(",", "")
        m = re.search(r"primary tier \([a-zA-Z0-9_.]+,\s*(\d+)\s*blocks\)", out)
        if m and not cpu_blocks:
            cpu_blocks = m.group(1)
    if not gpu_tokens:
        raise Abort("could not detect GPU KV cache size; set TIERBENCH_GPU_TOKENS")
    if not cpu_blocks:
        raise Abort("could not detect CPU primary tier size; set TIERBENCH_CPU_BLOCKS")
    return {"gpu_tokens": int(gpu_tokens), "cpu_blocks": int(cpu_blocks), "source": source}


# ---------------------------------------------------------------- tier geometry

FS_ROOT = os.environ.get("TIERBENCH_FS_ROOT", "/kvcache/blocks")


def detect_geometry():
    """Block geometry, read off the filesystem tier's own on-disk layout.

    This is exact where the metric-derived figure is not. The fs tier writes one file per
    (block hash, KV group); its config.json carries `tokens_per_hash` and the group list,
    and every block file is the same size. So

        bytes/token = n_groups * block_file_bytes / tokens_per_hash

    which on this build is 9 * 27,000,832 / 1648 = 147,456 B/token exactly -- 16 KiB per
    token per group. The load-leg estimate of ~35 KB/token is wrong by 4.2x: it divides
    bytes by `external_prefix_cache_hits_total`, and that counter is not what it looks
    like. Do not size evictions off it.

    The same geometry corrects the CPU tier, which is the bigger error. Its reported
    "636 blocks" are per-(hash, group) file slots, not logical blocks:
    636 * 27,000,832 = 15.99 GiB, i.e. precisely the 16 GiB shm region. Logical capacity
    is therefore 636 // 9 = 70 blocks = 115,360 tokens -- about HALF the GPU cache, not
    the 472,057 tokens a naive bytes-per-token division reports.

    Returns None if the fs tier is not visible from here; callers fall back to metrics.
    """
    try:
        cfgs = sorted(glob.glob(os.path.join(FS_ROOT, "*", "config.json")))
        if not cfgs:
            return None
        with open(cfgs[0]) as fh:
            cfg = json.load(fh)
        n_groups = len(cfg.get("kv_cache_groups") or [])
        tokens_per_hash = int(cfg.get("tokens_per_hash") or 0)
        if not (n_groups and tokens_per_hash):
            return None
        blk = None
        for f in glob.iglob(os.path.join(FS_ROOT, "*_r*", "*", "*_g*", "*.bin")):
            blk = os.path.getsize(f)
            break
        if not blk:
            return None
        return {
            "n_groups": n_groups,
            "tokens_per_hash": tokens_per_hash,
            "block_file_bytes": blk,
            "blocks_per_file": int(cfg.get("blocks_per_file") or 1),
            "bytes_per_token": n_groups * blk / float(tokens_per_hash),
            "source": os.path.dirname(cfgs[0]),
        }
    except Exception:
        return None


# ---------------------------------------------------------------- sizing

# Sanity band for KV bytes per token. Outside this, something has changed (dtype, group
# layout, block size) and every eviction this bench would size is wrong; better to stop
# than to run for twenty minutes and report a confident number about nothing.
# The upper bound was 128 KB and that was too tight: the measured geometric figure for
# this model is 144 KiB/token, because 6 of its 9 KV groups are fixed-size Mamba/GDN state
# rather than attention KV. A band that excludes the true value is a trap, not a guard.
BPT_MIN, BPT_MAX = 8 * 1024, 512 * 1024
BPT_FALLBACK = 31 * 1024   # ~/audit/stress/results/results.md: 2.796 GB for a ~90k prefix


def derive_bytes_per_token(m, geom=None):
    """KV bytes per token, from the LOAD leg.

    Load side, not store side, because "bytes needed to bring N tokens back" is exactly
    what sizes an eviction. On this boot the store side reads 74 KB/token against the load
    side's 35 KB/token -- stores re-write blocks that were promoted and evicted again, so
    the store counter over-counts tokens. The load figure agrees with the harness's direct
    measurement (2.796 GB for a ~90k prefix, ~31 KB/token); the store figure does not.
    """
    if os.environ.get("TIERBENCH_BYTES_PER_TOKEN"):
        return float(os.environ["TIERBENCH_BYTES_PER_TOKEN"]), "env"
    if geom:
        # Exact, and it does not need a single read-back to have happened. Prefer it.
        return geom["bytes_per_token"], (
            "fs geometry: %d groups x %s B / %d tokens per block"
            % (geom["n_groups"], "{:,}".format(geom["block_file_bytes"]),
               geom["tokens_per_hash"]))
    loaded = m.get("vllm:kv_offload_load_bytes_total",
                   m.get("vllm:kv_offload_total_bytes_total|CPU_to_GPU", 0.0))
    ext_hits = m.get("vllm:external_prefix_cache_hits_total", 0.0)
    if ext_hits >= 100000 and loaded > 0:
        bpt = loaded / ext_hits
        if BPT_MIN <= bpt <= BPT_MAX:
            return bpt, "load leg, %s hit tokens this boot" % "{:,}".format(int(ext_hits))
        raise Abort("derived %.1f KB/token is outside the [%d,%d] KB sanity band; "
                    "set TIERBENCH_BYTES_PER_TOKEN if this is a real config change"
                    % (bpt / 1024.0, BPT_MIN // 1024, BPT_MAX // 1024))
    return BPT_FALLBACK, "fallback constant (too few read-backs this boot to derive one)"


# ---------------------------------------------------------------- metrics

def parse_metrics(raw):
    """Flatten the exposition format. Labelled series are summed across labels, except
    the deprecated transfer_type pair which is kept split -- that label is the only way
    to separate the store leg from the load leg on this build."""
    out = {}
    for line in raw.splitlines():
        if not line or line[0] == "#":
            continue
        try:
            name_part, value = line.rsplit(" ", 1)
            val = float(value)
        except ValueError:
            continue
        if "{" in name_part:
            name, labels = name_part.split("{", 1)
            labels = labels.rstrip("}")
            tt = re.search(r'transfer_type="([^"]+)"', labels)
            le = re.search(r'le="([^"]+)"', labels)
            if tt:
                name = "%s|%s" % (name, tt.group(1))
            elif le:
                name = "%s|le=%s" % (name, le.group(1))
        else:
            name = name_part
        out[name] = out.get(name, 0.0) + val
    return out


def snapshot(base):
    return parse_metrics(http(base + "/metrics", timeout=30, raw=True))


def delta(before, after):
    d = {}
    for k in COUNTERS:
        if k in after or k in before:
            d[k] = after.get(k, 0.0) - before.get(k, 0.0)
    for k in ("vllm:kv_offload_total_bytes_total", "vllm:kv_offload_total_time_total"):
        for tt in ("CPU_to_GPU", "GPU_to_CPU"):
            kk = "%s|%s" % (k, tt)
            if kk in after or kk in before:
                d[kk] = after.get(kk, 0.0) - before.get(kk, 0.0)
    for k in HISTOGRAMS:
        for suffix in ("_sum", "_count"):
            kk = k + suffix
            if kk in after or kk in before:
                d[kk] = after.get(kk, 0.0) - before.get(kk, 0.0)
    for k in GAUGES:
        if k in after:
            d[k + "@end"] = after[k]
    return d


# ---------------------------------------------------------------- prompt construction

def make_text(salt, tag, n_words):
    """Deterministic given (salt, tag, n_words); no RNG, no wall-clock, no hashing that
    varies with PYTHONHASHSEED (the fs tier sets it to 0 and we must not depend on it)."""
    seed = 0
    for ch in "%s/%s" % (salt, tag):
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    words, x = [], seed | 1
    for i in range(n_words):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(_WORDS[x % len(_WORDS)])
        if i % 11 == 10:
            words.append("%s-%s-%d." % (tag, salt, i))
    return " ".join(words)


class Sizer(object):
    """Turns a token target into text via POST /tokenize -- exact, and it never risks the
    synthetic context-limit error that guessing chars-per-token walks into."""

    def __init__(self, base):
        self.base = base
        self.ratio = None      # words per token, learned once

    def count(self, text):
        r = http(self.base + "/tokenize",
                 {"model": MODEL, "prompt": text}, timeout=120)
        return r["count"]

    def build(self, salt, tag, target_tokens):
        if self.ratio is None:
            probe = make_text(salt, "ratio-probe", 4000)
            self.ratio = 4000.0 / self.count(probe)
        n = int(target_tokens * self.ratio)
        for _ in range(4):
            text = make_text(salt, tag, n)
            got = self.count(text)
            if abs(got - target_tokens) <= max(200, target_tokens * 0.01):
                return text, got
            n = max(16, int(n * (float(target_tokens) / got)))
        return text, got


# ---------------------------------------------------------------- request

def chat(base, text, tag, max_tokens=MAX_TOKENS):
    """Streamed so TTFT separates the prefill stall from decode. The whole thesis is that
    the ~78 s is prefill-side; a bench that only reports wall time cannot show that."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft, usage, ntok = None, None, 0
    try:
        with urllib.request.urlopen(req, timeout=PER_REQ_TIMEOUT) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                try:
                    ev = json.loads(body)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    # This model thinks. With max_tokens=16 every emitted token lands in
                    # `reasoning` and `content` stays null -- counting only `content`
                    # reports TTFT None on a request that streamed perfectly well. (And
                    # the field is `reasoning`, not `reasoning_content`, on this build.)
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        ntok += 1
                        if ttft is None:
                            ttft = time.time() - t0
    except urllib.error.HTTPError as e:
        raise Abort("%s: HTTP %s %s" % (tag, e.code, e.read().decode()[:400]))
    except Exception as e:
        raise Abort("%s: %r" % (tag, e))
    wall = time.time() - t0
    cached = None
    if usage:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "tag": tag, "wall_s": round(wall, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "decode_tokens": ntok,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": cached,
    }


# ---------------------------------------------------------------- phases

FS_INFLIGHT = "vllm:kv_offload_fs_inflight_jobs"
_BLOCK_BYTES = [27000832]   # replaced from detect_geometry() at startup


def fs_block_count(block_bytes=None):
    """Number of block files on the fs tier, from `df` rather than by walking the tree.

    Walking is the obvious way and it is unusable here: one pass over the 12.5k block files
    took over two minutes against a cold page cache -- longer than the poll interval, and
    the walk itself competes for the very disk being measured. `df` is O(1) and, because
    every block file is exactly the same size, gives an exact count of the delta, which is
    all the drain needs. /kvcache is a dedicated filesystem holding nothing else.
    """
    blk = block_bytes or _BLOCK_BYTES[0]
    try:
        r = subprocess.run(["df", "--output=used", "-B1", FS_ROOT],
                           capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[-1]) // blk
    except Exception:
        return -1


def drain(base, label, timeout_s=900, quiet_polls=3, poll_s=10.0):
    """Wait for the CPU->fs cascade to finish writing, then confirm it actually wrote.

    This exists because of the central finding of R3.8: the cascade does not run while the
    engine is under load. Block files stop appearing the moment requests arrive and resume
    the minute they stop, so a probe issued straight after an eviction is asking the fs
    tier for data that is still sitting in shm. Every "fs tier serves zero bytes" result so
    far was measured in that window.

    Two conditions, both required:
      * `kv_offload_fs_inflight_jobs` reads 0 -- the write pool has nothing queued;
      * the on-disk block count stops changing for `quiet_polls` consecutive polls.

    The gauge alone is not enough (it can read 0 between batches) and the file count alone
    is not enough (a stalled pool also produces a flat count). Returns a dict for the
    report; never raises -- a drain that times out is a result, not an error.
    """
    t0 = time.time()
    n0 = fs_block_count()
    log("  drain[%s]: %s block files on disk, waiting for the cascade to settle"
        % (label, "{:,}".format(n0)))
    last, quiet, infl = n0, 0, None
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        try:
            infl = snapshot(base).get(FS_INFLIGHT)
        except Exception:
            infl = None
        n = fs_block_count()
        # A gauge that has never been emitted reads None, which is not the same claim
        # as "zero jobs queued" -- it means the offload connector has not reported yet.
        # Treat it as idle (it is, on a fresh boot) but say which one it was in the log.
        idle = (infl is None or infl <= 0)
        # "Did not GROW", not "did not change". `kvcache-reap.timer` fires every 5 minutes
        # and deletes thousands of blocks at a time (it has a hard 90-minute age floor, so
        # it cannot touch anything this run wrote -- but it moves the count downwards). A
        # strict-equality quiet test gets reset by every reap and can never settle.
        quiet = quiet + 1 if (n <= last and idle) else 0
        if n != last:
            log("  drain[%s]: %+d blocks (%s total), inflight=%s%s"
                % (label, n - last, "{:,}".format(n), infl,
                   "  [reaper]" if n < last else ""))
        last = n
        if quiet >= quiet_polls:
            break
    out = {"label": label, "blocks_before": n0, "blocks_after": last,
           "blocks_written": last - n0, "inflight_end": infl,
           "wait_s": round(time.time() - t0, 1),
           "settled": quiet >= quiet_polls}
    log("  drain[%s]: %s after %.0fs -- %+d blocks, %s now on disk, inflight=%s"
        % (label, "settled" if out["settled"] else "TIMED OUT", out["wait_s"],
           out["blocks_written"], "{:,}".format(last),
           "not-yet-emitted" if infl is None else infl))
    return out


def evict(base, sizer, salt, want_tokens, label):
    """Push `want_tokens` of novel context through the server in chunks. Each chunk is a
    separate conversation with no shared prefix, so nothing here can be served from cache
    and every token displaces something."""
    n = max(1, int(want_tokens / EVICT_CHUNK_TOKENS))
    log("  %s: %d chunks x ~%d tokens = ~%d tokens" %
        (label, n, EVICT_CHUNK_TOKENS, n * EVICT_CHUNK_TOKENS))
    for i in range(n):
        budget_check()
        text, _ = sizer.build(salt, "%s-%d" % (label, i), EVICT_CHUNK_TOKENS)
        chat(base, text, "%s-%d" % (label, i), max_tokens=1)


def classify(d):
    """What tier actually served the probe, read off the counters rather than assumed."""
    gpu_hits = d.get("vllm:prefix_cache_hits_total", 0.0)
    ext_hits = d.get("vllm:external_prefix_cache_hits_total", 0.0)
    loaded = d.get("vllm:kv_offload_load_bytes_total",
                   d.get("vllm:kv_offload_total_bytes_total|CPU_to_GPU", 0.0))
    fs_bytes = d.get("vllm:kv_offload_fs_load_bytes_total")
    if ext_hits > 0 or loaded > 0:
        if fs_bytes is None:
            return "OFFLOAD(tier indistinguishable -- pre-R2.9.2)"
        return "OFFLOAD/FS" if fs_bytes > 0 else "OFFLOAD/CPU"
    if gpu_hits > 0:
        return "GPU"
    return "RECOMPUTE"


def run_phase(base, name, before_fn, probe_fn, expect, results):
    budget_check()
    log("PHASE %s" % name)
    if before_fn:
        before_fn()
    m0 = snapshot(base)
    r = probe_fn()
    m1 = snapshot(base)
    d = delta(m0, m1)
    served = classify(d)
    ok = expect is None or served.startswith(expect)
    row = {
        "phase": name, "expected": expect, "served_by": served,
        "valid": ok, "probe": r, "metrics_delta": d,
    }
    results.append(row)
    log("  %s: wall %.2fs ttft %s served_by=%s%s" %
        (name, r["wall_s"], r["ttft_s"], served, "" if ok else "   <-- INVALID"))
    if not ok:
        log("  !! %s did not reach the intended tier state; its timing is not usable."
            % name)
    return row


def concurrent_phase(base, sizer, salt, n_clients, results):
    """The multi-client regime from R2.6: several long conversations that cannot co-fit.
    Timing here is not deterministic -- interleaving is not under our control -- but the
    token counts and the number of clients are, so preemption counts are comparable."""
    budget_check()
    log("PHASE concurrent (%d clients)" % n_clients)
    per = int(TARGET_PREFIX_TOKENS * 1.3)
    texts = []
    for i in range(n_clients):
        t, n = sizer.build(salt, "conc-%d" % i, per)
        texts.append(t)
    m0 = snapshot(base)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_clients) as ex:
        futs = [ex.submit(chat, base, texts[i], "conc-%d" % i) for i in range(n_clients)]
        probes = [f.result() for f in futs]
    wall = time.time() - t0
    m1 = snapshot(base)
    d = delta(m0, m1)
    row = {
        "phase": "concurrent", "expected": None,
        "served_by": "n/a", "valid": True,
        "clients": n_clients, "wall_s": round(wall, 3),
        "probe": probes, "metrics_delta": d,
    }
    results.append(row)
    log("  concurrent: wall %.2fs preemptions=%d waiting_capacity_end=%s" %
        (wall, int(d.get("vllm:num_preemptions_total", 0)),
         d.get("vllm:kv_cache_usage_perc@end")))
    return row


# ---------------------------------------------------------------- report

def gib(x):
    return x / (2.0 ** 30)


def write_report(path_md, path_json, meta, results):
    with open(path_json, "w") as f:
        json.dump({"meta": meta, "phases": results}, f, indent=2, sort_keys=True)

    L = []
    L.append("# tierbench -- %s\n" % meta["started"])
    L.append("Model `%s`, salt `%s`, endpoint `%s`.  \n" % (MODEL, meta["salt"], meta["endpoint"]))
    L.append("Capacities from %s: GPU %s tokens, CPU primary %s blocks.  \n"
             % (meta["capacity"]["source"],
                "{:,}".format(meta["capacity"]["gpu_tokens"]),
                meta["capacity"]["cpu_blocks"]))
    L.append("Prefix %s tokens, max_tokens=%d, temperature=0.\n"
             % ("{:,}".format(meta.get("prefix_tokens") or 0), MAX_TOKENS))
    L.append("\n## Per-phase\n")
    L.append("| phase | expected | served by | valid | wall s | TTFT s | GPU hit tok | ext hit tok | load GB | fs load s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["phase"] == "concurrent":
            continue
        d = r["metrics_delta"]
        fs = d.get("vllm:kv_offload_fs_load_seconds_sum")
        L.append("| %s | %s | %s | %s | %.2f | %s | %s | %s | %.3f | %s |" % (
            r["phase"], r["expected"] or "-", r["served_by"], "yes" if r["valid"] else "**NO**",
            r["probe"]["wall_s"],
            "%.2f" % r["probe"]["ttft_s"] if r["probe"]["ttft_s"] is not None else "-",
            "{:,}".format(int(d.get("vllm:prefix_cache_hits_total", 0))),
            "{:,}".format(int(d.get("vllm:external_prefix_cache_hits_total", 0))),
            gib(d.get("vllm:kv_offload_load_bytes_total",
                      d.get("vllm:kv_offload_total_bytes_total|CPU_to_GPU", 0.0))),
            "%.2f" % fs if fs is not None else "n/a",
        ))
    conc = [r for r in results if r["phase"] == "concurrent"]
    if conc:
        c = conc[0]
        L.append("\n## Concurrent (%d clients)\n" % c["clients"])
        L.append("Wall %.2f s, preemptions +%d, allocation failures +%d, "
                 "async-lookup delay +%.1f s over %d observations.\n" % (
                     c["wall_s"],
                     int(c["metrics_delta"].get("vllm:num_preemptions_total", 0)),
                     int(c["metrics_delta"].get("vllm:kv_offload_allocation_failure_total", 0)),
                     c["metrics_delta"].get(
                         "vllm:kv_offload_tiering_lookup_async_delay_seconds_sum", 0.0),
                     int(c["metrics_delta"].get(
                         "vllm:kv_offload_tiering_lookup_async_delay_seconds_count", 0))))

    # The residue calculation -- the point of the whole exercise.
    L.append("\n## Attribution\n")
    for r in results:
        if r["phase"] not in ("cpu", "fs"):
            continue
        d = r["metrics_delta"]
        wall = r["probe"]["wall_s"]
        cpu_gpu = d.get("vllm:kv_offload_load_time_total",
                        d.get("vllm:kv_offload_total_time_total|CPU_to_GPU", 0.0))
        fs_s = d.get("vllm:kv_offload_fs_load_seconds_sum")
        if fs_s is None:
            L.append("- **%s**: wall %.2f s, CPU->GPU copy %.2f s, fs leg **not instrumented** "
                     "-- %.2f s unattributed. Apply R2.9.2 and re-run.\n"
                     % (r["phase"], wall, cpu_gpu, wall - cpu_gpu))
        else:
            res = wall - cpu_gpu - fs_s
            L.append("- **%s**: wall %.2f s = CPU->GPU %.2f s + fs leg %.2f s + **residue %.2f s** "
                     "(%.0f%%). Retry rounds: %s.\n"
                     % (r["phase"], wall, cpu_gpu, fs_s, res,
                        100.0 * res / wall if wall else 0,
                        int(d.get("vllm:kv_offload_fs_retry_rounds_sum", 0))))
    L.append("\nA large residue is the promotion path (per-step cadence / abandoned "
             "promotions), not the array -- see R2.2 and R2.3 of "
             "cache-preemption-patch-plan.md.\n")
    with open(path_md, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                    help="required: this evicts the entire KV cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect capacities and print the plan, send no load")
    ap.add_argument("--salt", default=None,
                    help="replay a previous run's content (see the module docstring)")
    ap.add_argument("--phases", default="cold,gpu,cpu,fs")
    ap.add_argument("--concurrent", type=int, default=0,
                    help="add the multi-client preemption phase with N clients")
    ap.add_argument("--prefix-tokens", type=int, default=TARGET_PREFIX_TOKENS)
    args = ap.parse_args()

    salt = args.salt or ("%x" % (int(time.time()) & 0xFFFFFF))
    want = [p.strip() for p in args.phases.split(",") if p.strip()]

    base = detect_endpoint()
    cap = detect_capacity()
    log("endpoint %s" % base)
    log("capacity (%s): GPU %s tokens, CPU primary %d blocks"
        % (cap["source"], "{:,}".format(cap["gpu_tokens"]), cap["cpu_blocks"]))

    sizer = Sizer(base)
    geom = detect_geometry()
    if geom:
        _BLOCK_BYTES[0] = geom["block_file_bytes"]
        log("geometry (%s): %d groups x %s B per %d-token block = %s B/token"
            % (geom["source"], geom["n_groups"],
               "{:,}".format(geom["block_file_bytes"]), geom["tokens_per_hash"],
               "{:,}".format(int(geom["bytes_per_token"]))))
    m0 = snapshot(base)
    bytes_per_tok, basis = derive_bytes_per_token(m0, geom)

    # CPU tier capacity, the corrected way. The tier reports its size in *file slots*,
    # one per (block hash, KV group), so logical capacity is slots // groups. Dividing
    # 16 GiB by bytes-per-token instead gives 472,057 tokens and is 4x too generous --
    # that error is why every previous `cpu` phase over-subscribed the tier by 2.4x and
    # flushed the very prefix it was testing.
    if geom:
        cpu_blocks_logical = cap["cpu_blocks"] // geom["n_groups"]
        cpu_tokens = cpu_blocks_logical * geom["tokens_per_hash"]
        cpu_basis = ("%d slots // %d groups = %d blocks x %d tokens"
                     % (cap["cpu_blocks"], geom["n_groups"], cpu_blocks_logical,
                        geom["tokens_per_hash"]))
    else:
        cpu_bytes = float(os.environ.get("TIERBENCH_CPU_GIB", "16")) * (2 ** 30)
        cpu_tokens = int(cpu_bytes / bytes_per_tok)
        cpu_basis = "16 GiB / %.0f B per token (no fs geometry available)" % bytes_per_tok
    log("derived %.1f KB/token (%s)" % (bytes_per_tok / 1024.0, basis))
    log("CPU tier holds ~%s tokens (%s)" % ("{:,}".format(cpu_tokens), cpu_basis))

    evict_gpu_tokens = int(cap["gpu_tokens"] * EVICT_MARGIN)
    # Turning the CPU tier over requires pushing past its capacity; it also has to clear
    # the GPU on the way, so never ask for less than the GPU eviction.
    evict_cpu_tokens = max(int(cpu_tokens * EVICT_MARGIN), evict_gpu_tokens)

    # The CPU tier is smaller than the GPU cache (115,360 vs 228,737 tokens). Both are
    # LRU over the same stream, so anything evicted from the GPU was pushed out of the
    # CPU tier strictly earlier. A CPU-tier hit on a prefix that was GPU-resident is
    # therefore impossible by construction, not merely unlikely, and the `cpu` phase
    # cannot pass however it is tuned. Say so rather than reporting it as a failure.
    cpu_phase_possible = cpu_tokens > cap["gpu_tokens"]
    if not cpu_phase_possible:
        log("NOTE: CPU tier (%s tok) is smaller than the GPU cache (%s tok), so it can "
            "never hold a prefix the GPU has evicted. The `cpu` phase is expected to "
            "recompute; it is a staging buffer for the fs tier, not a cache."
            % ("{:,}".format(cpu_tokens), "{:,}".format(cap["gpu_tokens"])))
    log("plan: prefix %s tok | evict-GPU %s tok | evict-CPU %s tok"
        % ("{:,}".format(args.prefix_tokens),
           "{:,}".format(evict_gpu_tokens), "{:,}".format(evict_cpu_tokens)))

    if args.dry_run:
        log("dry run: nothing sent.")
        return 0
    if not args.yes:
        log("refusing to run without --yes (this evicts the entire KV cache)")
        return 2

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "salt": salt, "endpoint": base, "capacity": cap,
        "bytes_per_token": bytes_per_tok,
        "cpu_tier_tokens": cpu_tokens,
        "prefix_tokens": args.prefix_tokens,
        "evict_gpu_tokens": evict_gpu_tokens,
        "evict_cpu_tokens": evict_cpu_tokens,
        "phases_requested": want,
        "geometry": geom,
        "cpu_phase_possible": cpu_phase_possible,
        "drains": [],
    }
    results = []
    rc = 0
    try:
        P, ptok = sizer.build(salt, "prefix", args.prefix_tokens)
        meta["prefix_tokens"] = ptok
        log("prefix built: %s tokens" % "{:,}".format(ptok))

        if "cold" in want:
            run_phase(base, "cold", None,
                      lambda: chat(base, P, "cold"), "RECOMPUTE", results)
        if "gpu" in want:
            run_phase(base, "gpu", None,
                      lambda: chat(base, P, "gpu"), "GPU", results)
        if "cpu" in want:
            run_phase(base, "cpu",
                      lambda: evict(base, sizer, salt, evict_gpu_tokens, "evictgpu"),
                      lambda: chat(base, P, "cpu"),
                      "OFFLOAD" if cpu_phase_possible else None, results)
        if "fs" in want:
            # Three steps, in this order, and the order is the whole point of the phase:
            #   1. drain -- let the cascade finish writing the prefix to disk. It cannot
            #      do this while requests are in flight, so it has to happen before the
            #      eviction traffic starts, not after.
            #   2. evict -- clear the prefix out of both the GPU and the CPU tier.
            #   3. drain again -- the eviction just queued its own backlog, and a lookup
            #      issued into that backlog waits minutes. We are testing whether an fs
            #      hit is possible at all, not how fast it is under load.
            def fs_before():
                meta["drains"].append(drain(base, "pre-evict"))
                evict(base, sizer, salt, evict_cpu_tokens, "evictcpu")
                meta["drains"].append(drain(base, "post-evict"))
            run_phase(base, "fs", fs_before,
                      lambda: chat(base, P, "fs"), "OFFLOAD", results)
        if args.concurrent:
            concurrent_phase(base, sizer, salt, args.concurrent, results)
    except Abort as e:
        log("ABORT: %s" % e)
        meta["aborted"] = str(e)
        rc = 1
    except KeyboardInterrupt:
        log("interrupted")
        meta["aborted"] = "KeyboardInterrupt"
        rc = 1

    md = os.path.join(OUT_DIR, "REPORT-%s.md" % salt)
    js = os.path.join(OUT_DIR, "RESULTS-%s.json" % salt)
    write_report(md, js, meta, results)
    log("wrote %s" % md)
    log("wrote %s" % js)
    return rc


if __name__ == "__main__":
    sys.exit(main())
