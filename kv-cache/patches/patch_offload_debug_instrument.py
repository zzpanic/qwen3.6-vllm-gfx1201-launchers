#!/usr/bin/env python3
# radiance patch: debug_instrument -- record KV-offload lookup/store/store-drop/
# store-ready events to a bounded JSON-line sink so a miss can be replayed.
#
# INSTRUMENTATION ONLY.  No behaviour change, no gate.  Every hook is wrapped in
# try/except and the sink itself never raises, so a failure degrades to "no
# events recorded", never to a broken engine.  Idempotent: re-running is a no-op.
#
# Touches:
#   - writes  vllm/_kvinstr.py                      (the sink; copied from this repo)
#   - appends to vllm/v1/kv_offload/cpu/manager.py (wraps lookup/prepare_store/complete_store)
#   - appends to .../offloading/scheduler.py       (wraps get_num_new_matched_tokens + defines _kvinstr_store_drop)
#   - string-surgery on the `if block_id == 0` drop in scheduler._build_store_jobs (the H2 signal)
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

import sysconfig
SP = Path(
    os.environ.get(
        "RADIANCE_KVOFFLOAD_DIR",
        sysconfig.get_paths()["purelib"] + "/vllm",
    )
)
SINK_SRC = Path(__file__).resolve().parent / "_kvinstr.py"
# The sink must be a TOP-LEVEL module (site-packages/_kvinstr.py), not a vllm
# submodule (site-packages/vllm/_kvinstr.py), so `import _kvinstr` resolves.
SINK_DST = SP.parent / "_kvinstr.py"
MANAGER = SP / "v1" / "kv_offload" / "cpu" / "manager.py"
SCHED = SP / "distributed" / "kv_transfer" / "kv_connector" / "v1" / "offloading" / "scheduler.py"

MARKER = "radiance debug_instrument install"


def append_once(path: Path, block: str, label: str) -> bool:
    """Append `block` to `path` once (idempotent on MARKER). Returns True if written."""
    if not path.exists():
        raise SystemExit(f"patch_debug_instrument: missing {path}")
    cur = path.read_text()
    if MARKER in cur:
        print(f"patch_debug_instrument: {label} already present; skipping (idempotent)")
        return False
    path.write_text(cur.rstrip("\n") + "\n" + block)
    print(f"patch_debug_instrument: appended {label} to {path.name}")
    return True


def main() -> None:
    # 1. install the sink module into site-packages.
    if not SINK_SRC.exists():
        raise SystemExit(f"patch_debug_instrument: missing {SINK_SRC}")
    if SINK_DST.exists() and SINK_DST.read_text() == SINK_SRC.read_text():
        print("patch_debug_instrument: sink already up to date")
    else:
        shutil.copy(SINK_SRC, SINK_DST)
        print(f"patch_debug_instrument: installed sink -> {SINK_DST}")

    # 2. manager.py: wrap lookup / prepare_store / complete_store.
    manager_block = f'''

# --- {MARKER} (manager; idempotent, no behaviour change) ---
try:
    import _kvinstr as _kvs
    _kvs.boot()
    _KManager = CPUOffloadingManager  # module global (defined above)

    _orig_lookup = _KManager.lookup
    def _lookup(self, *a, **k):
        res = _orig_lookup(self, *a, **k)
        try:
            _kvs.lookup_key(a[0], getattr(a[1], "req_id", None), res)
        except Exception:
            pass
        return res
    _KManager.lookup = _lookup

    _orig_ps = _KManager.prepare_store
    def _ps(self, *a, **k):
        out = _orig_ps(self, *a, **k)
        try:
            keys = a[0]
            dropped = out is None
            # PrepareStoreOutput carries keys_to_store (keys the tier did NOT already hold) and
            # evicted_keys. Until 2026-09-17 this read a non-existent new_offload_keys, so
            # n_new_keys was always 0 and evictions were never logged.
            _to_store = () if dropped else (getattr(out, "keys_to_store", None) or ())
            _evicted = () if dropped else (getattr(out, "evicted_keys", None) or ())
            _kvs.store_admission(
                getattr(a[1], "req_id", None), dropped, len(_to_store),
                len(list(keys)) if isinstance(keys, (list, tuple, set)) else 1,
                n_evicted=len(_evicted),
                evicted_groups=_kvs.group_counts(_evicted),
            )
        except Exception:
            pass
        return out
    _KManager.prepare_store = _ps

    _orig_cs = _KManager.complete_store
    def _cs(self, *a, **k):
        r = _orig_cs(self, *a, **k)
        try:
            if k.get("success", True):
                keys = a[0]
                _kvs.store_ready(list(keys) if isinstance(keys, (list, tuple, set)) else [keys])
        except Exception:
            pass
        return r
    _KManager.complete_store = _cs
except Exception:
    pass
'''
    append_once(MANAGER, manager_block, "manager hooks")

    # 3. scheduler.py: wrap get_num_new_matched_tokens + define the drop helper.
    sched_block = f'''

# --- {MARKER} (scheduler; idempotent, no behaviour change) ---
def _kvinstr_store_drop(req_id, offload_key, reason="zeroed"):
    try:
        import _kvinstr as _k
        _k.store_drop(req_id, offload_key, reason)
    except Exception:
        pass

try:
    import _kvinstr as _kvs  # noqa: F401  (the drop helper imports it lazily)
    _kvs.boot()
    _KScheduler = OffloadingConnectorScheduler  # module global (defined above)

    _orig_gm = _KScheduler.get_num_new_matched_tokens
    def _gm(self, *a, **k):
        ret = _orig_gm(self, *a, **k)
        try:
            request = a[0] if len(a) > 0 else k.get("request")
            num_computed = a[1] if len(a) > 1 else k.get("num_computed_tokens")
            ext = ret[0] if isinstance(ret, tuple) else ret
            _kvs.lookup_terminal(getattr(request, "request_id", None), num_computed, ext)
        except Exception:
            pass
        return ret
    _KScheduler.get_num_new_matched_tokens = _gm
except Exception:
    pass
'''
    append_once(SCHED, sched_block, "scheduler hooks")

    # 4. string-surgery the freed/zeroed (H2) drop site in _build_store_jobs.
    anchor = (
        "                    if block_id == 0:\n"
        "                        continue\n"
    )
    new = (
        "                    if block_id == 0:\n"
        "                        _kvinstr_store_drop(req_id, offload_key)\n"
        "                        continue\n"
    )
    sentinel = "_kvinstr_store_drop(req_id, offload_key)"
    v2_sentinel = '"swa" if (group_config.is_mamba_group'
    # Match the v1 call OR the v2 (group_type-bearing) call, so a re-run after the
    # v2 surgery does not try to re-apply the v1 anchor (which no longer matches).
    if (sentinel in SCHED.read_text()) or (v2_sentinel in SCHED.read_text()):
        print("patch_debug_instrument: zeroed-drop hook already present; skipping (idempotent)")
    else:
        apply(SCHED, anchor, new, sentinel, "scheduler zeroed-drop (H2) hook")

    # 5. v2 refinement: separate by-design (SWA/mamba zeroed-out) from H2
    #    (full-attention 0) on the zeroed drop, and record num_prompt_tokens on
    #    lookup_terminal. The block_id is already 0 at the drop site, so the
    #    GROUP TYPE is the clean discriminator: the code (lines 1094-1107) zeros
    #    out only sliding-window/mamba/SSM block_ids by-design; a full-attention
    #    group's block_id==0 is therefore a genuine H2/stale. Idempotent.
    v2_surgery()


def v2_surgery() -> None:
    # 5a. extend the drop helper signature with group_type.
    apply(
        SCHED,
        'def _kvinstr_store_drop(req_id, offload_key, reason="zeroed"):\n',
        'def _kvinstr_store_drop(req_id, offload_key, reason="zeroed", group_type="unknown"):\n',
        'group_type="unknown"',
        "v2: _kvinstr_store_drop group_type param",
    )
    # 5b. pass group_type through to the sink.
    apply(
        SCHED,
        "        _k.store_drop(req_id, offload_key, reason)\n",
        "        _k.store_drop(req_id, offload_key, reason, group_type)\n",
        "_k.store_drop(req_id, offload_key, reason, group_type)",
        "v2: store_drop group_type pass",
    )
    # 5c. at the drop site, classify the group: 'swa' (mamba/SSM/SWA, by-design
    #     zero-out) vs 'full' (full-attention, a genuine H2/stale). Lands in the
    #     3rd positional arg (reason); the reader maps reason->group_type.
    apply(
        SCHED,
        "                        _kvinstr_store_drop(req_id, offload_key)\n",
        "                        _kvinstr_store_drop(req_id, offload_key,\n"
        "                            \"swa\" if (group_config.is_mamba_group\n"
        "                            or (group_config.sliding_window_size_in_chunks is not None))\n"
        "                            else \"full\")\n",
        '"swa" if (group_config.is_mamba_group',
        "v2: drop-site group_type",
    )
    # 5d. record num_prompt_tokens on lookup_terminal (gap size per miss).
    #     Lands in the 4th positional arg (group_idx); the reader reads npt from
    #     group_idx. Fall back through prompt_token_ids.
    apply(
        SCHED,
        "            ext = ret[0] if isinstance(ret, tuple) else ret\n"
        "            _kvs.lookup_terminal(getattr(request, \"request_id\", None), num_computed, ext)\n",
        "            ext = ret[0] if isinstance(ret, tuple) else ret\n"
        "            npt = getattr(request, \"num_prompt_tokens\", None)\n"
        "            if npt is None:\n"
        "                _kvs_pt = getattr(request, \"prompt_token_ids\", None)\n"
        "                npt = len(_kvs_pt) if _kvs_pt is not None else None\n"
        "            _kvs.lookup_terminal(getattr(request, \"request_id\", None), num_computed, ext, npt)\n",
        "npt = getattr(request, \"num_prompt_tokens\", None)",
        "v2: lookup_terminal num_prompt_tokens",
    )


if __name__ == "__main__":
    main()
