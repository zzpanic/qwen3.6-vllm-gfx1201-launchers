#!/usr/bin/env python3
"""Stop a hybrid (Mamba + attention) request from recomputing a prefix the offload tier holds.

THE MISS THIS FIXES (miss-analysis-20260917/README.md, pattern A -- 8 of the 14 real misses
since the 2026-09-15 boot, ~270 s of wasted prefill):

  1. The GPU still holds a long ATTENTION prefix for the prompt, but no MAMBA state at any
     boundary along it.
  2. Scheduler.schedule() calls get_computed_blocks_for_connector(), which reports the
     attention hit as the local hit and sets hit_diverged.
  3. The connector is asked only for chunks BEYOND that attention hit. Those are this
     turn's new tokens, never stored, so it returns 0.
  4. `if hit_diverged and num_external_computed_tokens == 0:` falls back to
     get_computed_blocks(): the boundary every group agrees on ON THE GPU. With no Mamba
     state that boundary is 0, so the whole prompt is recomputed.
  5. The connector is never asked again from that lower boundary -- so the Mamba snapshots
     the CPU tier holds for the same prefix are never used.

Proof it was reachable: requests 46 and 90 ran alone and their full recomputes stored ZERO
new Mamba snapshots (the tier already had all of them); request 79 later loaded, from the
tier, the exact snapshot 46 had recomputed past. All of steps 2-4 are stock vLLM 0.27.1.

WHY THE GPU LOSES THE MAMBA STATE (context, not something this patch changes): in
mamba_cache_mode=align, MambaManager.remove_skipped_blocks frees each passed-over state
block two steps after it is written, while the request keeps its attention blocks pinned
until it finishes. A long request therefore queues its own old Mamba states for eviction
ahead of its attention blocks. The logging hunk below records per-group GPU hits so that
explanation can be confirmed or killed on real traffic.

WHAT THIS PATCH DOES
  hunk 1 (scheduler.py, helper): `_radiance_reconcile_reask`. Called right after the stock
      fallback. If the fallback dropped the local hit by at least
      RADIANCE_REASK_MIN_DROP_BLOCKS blocks (default 2), it asks the connector again with
      the lowered, group-consistent local hit. That is the ordinary "local hit + external
      suffix" path every external hit already takes -- nothing new is served, the tier is
      just asked the right question. It may load chunks the GPU also holds as attention;
      that costs a few seconds of load instead of a minute of prefill, and it is correct.
        - ext2 > 0   -> use it (load_kv_async as the connector says)
        - ext2 == 0  -> stock behaviour (recompute from the reconciled boundary)
        - ext2 None  -> the tier needs time (e.g. a store still landing): skip the request
                        this step, exactly like the stock None branch above it. Bounded by
                        RADIANCE_REASK_MAX_DEFER_S (default 10); past that, give up and
                        recompute, so a pending store can never park a request.
        - exception  -> logged, stock behaviour. A diagnostic path must not kill EngineCore.
      A 1-block drop is the normal per-turn shape (attention hit minus one block) and is left
      alone by default: it is a separate, known shortfall and re-asking there would put a
      load step on every normal turn.
  hunk 2 (scheduler.py, call site): the call, placed inside the stock fallback branch.
  hunk 3 (kv_cache_manager.py, logging): stash per_group_hits on the request so the helper
      can record what the GPU held per group at the moment of the fallback.

KILL SWITCH: RADIANCE_RECONCILE_REASK=0 restores stock behaviour exactly (the helper still
logs, with decision="off", so the counterfactual stays measurable).

LOGGING: one `reconcile` event per fallback in the debug_instrument sink
(/tmp/kvinstr/<pid>.jsonl), when that sink is installed. Absent sink = no events, no change.

ORDER: hunks are applied helper -> call site -> logging, so a failure part-way leaves either
stock behaviour (nothing calls the helper) or the fix without per-group logging. Must run
after patch_offload_mixed_hit.py (it is the lookup this re-ask drives). Touches no file any
other house patch anchors on; patch_dynwidth.py (runs later) anchors elsewhere in
scheduler.py and appends at the end.
"""
import os
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

VLLM = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"] + "/vllm"))
SCHED = VLLM / "v1" / "core" / "sched" / "scheduler.py"
KVCM = VLLM / "v1" / "core" / "kv_cache_manager.py"

# The helper, as a string so the offline test (test_reconcile_reask.py) executes the exact
# source that gets installed.
HELPER = '''

# ---- RADIANCE reconcile re-ask (patch_offload_reconcile_reask.py) ---------------------------
import os as _rad_rr_os
import time as _rad_rr_time

_RAD_REASK = _rad_rr_os.environ.get("RADIANCE_RECONCILE_REASK", "0") == "1"
_RAD_REASK_MIN_DROP_BLOCKS = int(_rad_rr_os.environ.get("RADIANCE_REASK_MIN_DROP_BLOCKS", "2"))
_RAD_REASK_MAX_DEFER_S = float(_rad_rr_os.environ.get("RADIANCE_REASK_MAX_DEFER_S", "10"))
_RAD_RR_SINK = []


def _radiance_rr_emit(rec):
    try:
        if not _RAD_RR_SINK:
            try:
                import _kvinstr
                _RAD_RR_SINK.append(_kvinstr)
            except Exception:
                _RAD_RR_SINK.append(None)
        sink = _RAD_RR_SINK[0]
        if sink is not None and hasattr(sink, "reconcile"):
            sink.reconcile(rec)
    except Exception:
        pass


def _radiance_rr_group_kinds(sched):
    kinds = getattr(sched, "_radiance_rr_kinds", None)
    if kinds is None:
        short = {"MambaSpec": "M", "FullAttentionSpec": "F", "SlidingWindowSpec": "S"}
        try:
            kinds = ",".join(
                short.get(type(g.kv_cache_spec).__name__, type(g.kv_cache_spec).__name__)
                for g in sched.kv_cache_config.kv_cache_groups
            )
        except Exception:
            kinds = "?"
        sched._radiance_rr_kinds = kinds
    return kinds


def _radiance_reconcile_reask(sched, request, fa_local, reconciled_local, ext1):
    """Re-ask the connector after the hit_diverged fallback lowered the local hit.

    Returns (num_external_tokens, load_kv_async), or None to skip the request this step."""
    decision = "off"
    ext2 = None
    load_async = False
    defer_s = None
    block_size = sched.block_size
    drop = fa_local - reconciled_local
    result = (0, False)
    try:
        if not _RAD_REASK:
            decision = "off"
        elif drop < _RAD_REASK_MIN_DROP_BLOCKS * block_size:
            decision = "small_drop"
        elif reconciled_local % block_size:
            decision = "partial_tail"
        else:
            t0 = getattr(request, "_radiance_rr_defer_t0", None)
            if t0 is not None:
                defer_s = _rad_rr_time.monotonic() - t0
            if getattr(request, "_radiance_rr_gave_up", False) or (
                defer_s is not None and defer_s > _RAD_REASK_MAX_DEFER_S
            ):
                # Permanent for this request: once it has waited out the budget it
                # recomputes, and never parks on the tier again.
                decision = "defer_budget_spent"
                request._radiance_rr_gave_up = True
            else:
                ext2, load_async = sched.connector.get_num_new_matched_tokens(
                    request, reconciled_local
                )
                if ext2 is None:
                    decision = "defer"
                    if t0 is None:
                        request._radiance_rr_defer_t0 = _rad_rr_time.monotonic()
                    result = None
                elif ext2 > 0:
                    decision = "hit"
                    result = (ext2, bool(load_async))
                else:
                    decision = "miss"
                    result = (0, False)
            if result is not None:
                request._radiance_rr_defer_t0 = None
    except Exception as e:  # never let the re-ask take down EngineCore
        decision = "error:" + type(e).__name__
        result = (0, False)
    per = getattr(request, "_radiance_per_group_hits", None)
    _radiance_rr_emit({
        "req_id": getattr(request, "request_id", None),
        "num_prompt_tokens": getattr(request, "num_prompt_tokens", None),
        "num_tokens": getattr(request, "num_tokens", None),
        "fa_local": fa_local,
        "reconciled_local": reconciled_local,
        "ext1": ext1,
        "per_group_hits": list(per[0]) if per else None,
        "fa_group": per[1] if per else None,
        "group_kinds": _radiance_rr_group_kinds(sched),
        "decision": decision,
        "ext2": ext2,
        "load_async": bool(load_async),
        "defer_s": round(defer_s, 3) if defer_s is not None else None,
    })
    return result


class Scheduler(SchedulerInterface):
'''

CALL_ANCHOR = (
    "                        if hit_diverged and num_external_computed_tokens == 0:\n"
    "                            # No external tokens back the deeper local hit, so its\n"
    "                            # resume boundary would have no valid Mamba state.\n"
    "                            # Reconcile to the boundary every group agrees on.\n"
    "                            (\n"
    "                                new_computed_blocks,\n"
    "                                num_new_local_computed_tokens,\n"
    "                                request.shared_prefix_boundary,\n"
    "                            ) = self.kv_cache_manager.get_computed_blocks(request)\n"
)
CALL_NEW = CALL_ANCHOR + (
    "                            # radiance reconcile re-ask: the fallback above can drop the\n"
    "                            # local hit to 0 while the offload tier holds this prefix.\n"
    "                            # Ask the connector again from the lowered boundary.\n"
    "                            _rad_reask = _radiance_reconcile_reask(\n"
    "                                self,\n"
    "                                request,\n"
    "                                block_aligned_local,\n"
    "                                num_new_local_computed_tokens,\n"
    "                                ext_tokens,\n"
    "                            )\n"
    "                            if _rad_reask is None:\n"
    "                                request_queue.pop_request()\n"
    "                                step_skipped_waiting.prepend_request(request)\n"
    "                                continue\n"
    "                            num_external_computed_tokens, load_kv_async = _rad_reask\n"
)

STASH_ANCHOR = (
    "        num_local = per_group_hits[fa_group_id]\n"
    "        blocks = self.create_kv_cache_blocks(computed)\n"
)
STASH_NEW = (
    "        num_local = per_group_hits[fa_group_id]\n"
    "        # radiance reconcile re-ask: record what the GPU held per group (logging only).\n"
    "        request._radiance_per_group_hits = (tuple(per_group_hits), fa_group_id)\n"
    "        blocks = self.create_kv_cache_blocks(computed)\n"
)


def main() -> None:
    apply(SCHED, "\n\nclass Scheduler(SchedulerInterface):\n", HELPER,
          "def _radiance_reconcile_reask(", "reconcile re-ask: helper")
    apply(SCHED, CALL_ANCHOR, CALL_NEW,
          "_rad_reask = _radiance_reconcile_reask(", "reconcile re-ask: call site")
    apply(KVCM, STASH_ANCHOR, STASH_NEW,
          "request._radiance_per_group_hits =", "reconcile re-ask: per-group hit logging")


if __name__ == "__main__":
    main()
