# radiance debug_instrument sink -- write-only, bounded, NO behaviour change.
# Records KV-offload lookup / store / store-drop / store-ready events as JSON
# lines so a miss can be replayed.  Never raises: every public fn is wrapped,
# and a failure is silently swallowed so the engine is never affected.
#
# Output: /tmp/kvinstr/<pid>.jsonl  (container-side).  The host reader tails it
# via /proc/<pid>/root/tmp/kvinstr/<pid>.jsonl.  Bounded + rotated (KVINSTR_MAX_MB).
import json
import os
import time

_LOCK = None  # set in _init (threading imported lazily to avoid import cost if unused)
_MAX_BYTES = int(os.environ.get("KVINSTR_MAX_MB", "50")) * 1024 * 1024
_PATH = os.environ.get("KVINSTR_PATH") or (
    os.path.join(os.environ.get("KVINSTR_DIR", "/tmp/kvinstr"), str(os.getpid()) + ".jsonl")
)
_dir = os.path.dirname(_PATH)
try:
    os.makedirs(_dir, exist_ok=True)
except Exception:
    pass


def _init():
    global _LOCK
    if _LOCK is None:
        import threading
        _LOCK = threading.Lock()
    return _LOCK


def _kser(k):
    try:
        if isinstance(k, (tuple, list)):
            return [int(x) for x in k]
    except Exception:
        pass
    return str(k)


def _emit(rec):
    try:
        rec["ts"] = round(time.time(), 6)
        line = json.dumps(rec, default=str) + "\n"
        lock = _init()
        with lock:
            try:
                if os.path.getsize(_PATH) > _MAX_BYTES:
                    os.replace(_PATH, _PATH + ".1")
            except Exception:
                pass
            with open(_PATH, "a") as f:
                f.write(line)
    except Exception:
        pass


def lookup_terminal(req_id, num_computed_tokens, ret, group_idx=None, num_prompt_tokens=None):
    """Coarse endpoint from the get_num_new_matched_tokens return value.
    ret: None -> deferred (backend or load); 0 -> zero/sub-chunk miss; >0 -> hit.
    Defensively unwraps the (tokens, bool) tuple if one is passed."""
    if isinstance(ret, tuple):
        ret = ret[0]
    if ret is None:
        endpoint = "E_DEFER"
        ext = None
    elif ret == 0:
        endpoint = "E_MISS_ZERO_OR_SUBCHUNK"
        ext = 0
    else:
        endpoint = "E_HIT"
        ext = ret
    _emit({
        "kind": "lookup_terminal",
        "req_id": req_id,
        "num_computed_tokens": num_computed_tokens,
        "endpoint": endpoint,
        "num_external_tokens": ext,
        "group_idx": group_idx,
        "num_prompt_tokens": num_prompt_tokens,
    })


def lookup_key(offload_key, req_id, result):
    """Per-key offload-tier result (the L-gate): HIT / HIT_PENDING / MISS."""
    _emit({
        "kind": "lookup_key",
        "key": _kser(offload_key),
        "req_id": req_id,
        "result": str(result),
    })


def group_counts(keys):
    """{group_idx: count} for OffloadKeys (block_hash + 4-byte big-endian group index)."""
    out = {}
    for k in keys or ():
        try:
            g = int.from_bytes(bytes(k)[-4:], "big")
        except Exception:
            g = -1
        out[g] = out.get(g, 0) + 1
    return out


def store_admission(req_id, dropped, n_new_keys, n_keys, n_evicted=None, evicted_groups=None):
    """prepare_store admission. dropped=True => ALLOCATION_FAILURE (H1).

    n_keys      keys offered for store
    n_new_keys  keys the tier did not already hold (n_keys - n_new_keys were already there)
    n_evicted   keys evicted to make room; evicted_groups = {group_idx: count}
    """
    _emit({
        "kind": "store_admission",
        "req_id": req_id,
        "dropped": dropped,
        "n_new_keys": n_new_keys,
        "n_keys": n_keys,
        "n_evicted": n_evicted,
        "evicted_groups": evicted_groups,
    })


def store_drop(req_id, offload_key, reason, group_type="unknown"):
    """A store candidate dropped before being written. reason: zeroed (H2).
    group_type: 'swa' (mamba/SSM/sliding-window -- the code zeros these by-design)
    or 'full' (full-attention -- the code does NOT zero these, so a 0 is H2/stale)."""
    _emit({
        "kind": "store_drop",
        "req_id": req_id,
        "key": _kser(offload_key),
        "reason": reason,
        "group_type": group_type,
    })


def store_ready(keys):
    """complete_store: the keys are now durably present in the tier."""
    if isinstance(keys, (list, tuple)):
        kser = [_kser(k) for k in keys]
    else:
        kser = _kser(keys)
    _emit({"kind": "store_ready", "keys": kser, "n_keys": len(keys) if isinstance(keys, (list, tuple)) else 1})


def reconcile(rec):
    """hit_diverged fallback in Scheduler.schedule() (patch_offload_reconcile_reask.py):
    what the GPU held per group, what the fallback reconciled to, and what the re-ask did."""
    r = dict(rec)
    r["kind"] = "reconcile"
    _emit(r)


def boot():
    """Canary: write one line on install so the reader can confirm the sink is live."""
    _emit({"kind": "boot", "path": _PATH, "pid": os.getpid()})


def path():
    return _PATH
