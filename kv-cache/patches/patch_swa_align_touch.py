#!/usr/bin/env python3
"""Two offload-scheduler changes for hybrid Mamba + attention (+ EAGLE-style drafter) models.

Both live in distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py. Each has its own
kill switch. Evidence: <repo>/kv-cache/phase2-research/README.md sections 2 and 4.

HUNK 1 -- store sliding-window chunks only where a hit can land (RADIANCE_SWA_STORE_MAMBA_ALIGN)

  Stock vLLM already skips sliding-window chunks that can never serve a hit:
  `is_store_reachable_swa_chunk` keeps only the last `window (+1 for an eagle group)` chunks
  of each alignment segment. The segment is the full-attention chunk size. On a model whose
  Mamba groups run in `align` cache mode, `_lookup` ALSO rounds every hit down to the Mamba
  alignment (`resolve_mamba_align_size`, which radiance R3.13 multiplies by the Mamba store
  stride). So hits land only every stride x chunk tokens. Stock computes
  `alignment_tokens <= tokens_per_chunk -> None` and stores every drafter chunk, including
  ones no hit can reach.

  This hunk sets the alignment to the Mamba alignment when that alignment is a whole multiple
  of the full-attention chunk, and judges reachability on the ABSOLUTE grid. Stock sizes the
  last segment from the chunks storable so far, so during chunked prefill every chunk is judged
  while it is the newest one and gets stored anyway (R3.13 documents the same trap for Mamba). Effects at stride 4, 1,648-token chunks:
    - the drafter group (window 2 chunks + 1 eagle) stores 3 of every 4 chunks: exactly
      the chunks a grid-point hit reads -- positions 2, 3 and the NEXT segment's 0, because the
      eagle lookup queries one chunk past the hit and pops it. (First deploy used the stock
      "trailing window+1" rule on the absolute grid; reaskbench abbda2 showed it misses that
      chunk and the lookup cascades to 0. Stock never shows this: chunked prefill stores all.) Saves 1/4 of the drafter's rows (~4 KB/token, ~6% of
      the tier) with no hit lost.
    - Mamba groups (window 1) store the last chunk of each segment. R3.13's stride filter
      already does exactly that, so there is no change for them.
  Lookup does not read alignment_chunk_count (only the store path and a log line do), so a
  hit is looked up exactly as before.

HUNK 2 -- keep a prefix's Mamba/drafter chunks as fresh as its attention (RADIANCE_TOUCH_ALL_GROUPS)

  Pattern B (miss-analysis-20260917/README.md): `_touch` refreshes EVERY attention chunk of a
  request in the eviction policy, but for sliding-window groups (Mamba window 1, drafter
  window 2) only the chunks at the newest hit. As a conversation grows, its older snapshots age
  out and are evicted while their attention keys survive. A sibling agent or a rewritten turn
  that needs an earlier point then finds attention it cannot use.

  With this hunk, every group's chunks are touched like attention. The stored sliding-window
  chunks are exactly the reachable hit points (hunk 1 + R3.13), so keeping them alive with
  their attention is the intended retention. `touch` ignores keys the tier does not hold, and
  `_touch` runs on a lookup hit and when a request stores a new chunk, not every step.

  Upstream fixed the same asymmetry differently in PR #51787 (merged 2026-09-16:
  request-scoped recency, `_touch` deleted). That PR is written against a newer tree; 24 of
  its 34 source hunks conflict with 0.27.1. This hunk is the minimal equivalent for the
  asymmetry only. It does NOT change the pre-existing "one request counts as many accesses"
  behaviour for ARC, which applies to attention keys today.

  Touch ORDER (RADIANCE_TOUCH_POSITION_ORDER, added 2026-09-18). v1 of this hunk touched group by
  group in index order. CachePolicy.touch applies a list in reverse, so within one refresh every g0
  key ended older than every g1 key, and so on. Under three agents that is exactly what ARC evicted:
  a conversation lost every g0 Mamba snapshot from the tail inward while attention, g1-g5 and the
  drafter stayed held. 4 of 5 big tier_had losses, e.g. a 168k prompt whose recompute re-offered
  243 keys of which 229 were still held (ISSUE-SHAPES.md evict_skew). With the switch on, one touch
  per request carries every group's keys sorted by chunk end position, head to tail. The head is
  most recent, eviction removes whole positions from the tail, and at most one position per
  eviction batch is split. Upstream PR #51787 orders recency the same way (order_request_keys).
  Test: test_touch_order.py (real CPUOffloadingManager + ARC, the patched _touch, _lookup mirror).
"""
import os
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

VLLM = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"] + "/vllm"))
OFFSCHED = VLLM / "distributed" / "kv_transfer" / "kv_connector" / "v1" / "offloading" / "scheduler.py"

MODFLAG_ANCHOR = "\nclass SchedulerOffloadConfig(NamedTuple):\n"
MODFLAG_NEW = (
    "\n# radiance swa-align: set by SchedulerOffloadConfig.from_spec when the sliding-window store\n"
    "# alignment was raised to the Mamba grid. The store path then judges reachability on the\n"
    "# ABSOLUTE grid: a tail-relative (partial-segment) rule stores every chunk as it briefly\n"
    "# becomes the newest one during chunked prefill, the same trap R3.13 documents.\n"
    "_RADIANCE_SWA_GRID_ACTIVE = False\n"
    "\n"
    "\n"
    "def _radiance_store_reachable(abs_idx, storable, align, window, is_eagle):\n"
    "    if not _RADIANCE_SWA_GRID_ACTIVE or align is None or window is None:\n"
    "        return is_store_reachable_swa_chunk(abs_idx, storable, align, window, is_eagle)\n"
    "    # A hit lands on a grid point g (a multiple of `align` chunks). _lookup scans for\n"
    "    # `window` consecutive chunks ending at g; an eagle group queries ONE chunk past g and\n"
    "    # pops it, so it needs chunks g-window .. g. Stock is_store_reachable_swa_chunk keeps\n"
    "    # the trailing window+1 chunks BEFORE g instead, which misses chunk g. That mismatch is\n"
    "    # invisible in stock only because chunked prefill stores every chunk anyway.\n"
    "    pos = abs_idx % align\n"
    "    if pos >= align - window:\n"
    "        return True\n"
    "    return bool(is_eagle) and pos == 0\n"
) + MODFLAG_ANCHOR

ALIGN_ANCHOR = (
    "        alignment_tokens: int | None = None\n"
    "        if len(full_attn_tokens_per_chunk) == 1:\n"
    "            alignment_tokens = full_attn_tokens_per_chunk.pop()\n"
)
ALIGN_NEW = ALIGN_ANCHOR + (
    "        # radiance swa-align: hits are ALSO rounded down to the Mamba alignment (x the\n"
    "        # R3.13 store stride), so a sliding-window chunk earlier in that coarser segment can\n"
    "        # never serve a hit either. Kill switch: RADIANCE_SWA_STORE_MAMBA_ALIGN=0.\n"
    "        if (\n"
    "            os.environ.get(\"RADIANCE_SWA_STORE_MAMBA_ALIGN\", \"0\") == \"1\"\n"
    "            and alignment_tokens is not None\n"
    "        ):\n"
    "            _rad_mamba_align = resolve_mamba_align_size(spec, kv_cache_config)\n"
    "            if (\n"
    "                _rad_mamba_align is not None\n"
    "                and _rad_mamba_align > alignment_tokens\n"
    "                and _rad_mamba_align % alignment_tokens == 0\n"
    "            ):\n"
    "                logger.info(\n"
    "                    \"[radiance] swa-align: sliding-window store alignment %d -> %d tokens \"\n"
    "                    \"(hits land on the Mamba grid)\",\n"
    "                    alignment_tokens,\n"
    "                    _rad_mamba_align,\n"
    "                )\n"
    "                alignment_tokens = _rad_mamba_align\n"
    "                global _RADIANCE_SWA_GRID_ACTIVE\n"
    "                _RADIANCE_SWA_GRID_ACTIVE = True\n"
)

STORE_ANCHOR = (
    "                    if not is_store_reachable_swa_chunk(\n"
    "                        abs_chunk_idx,\n"
    "                        num_chunks,\n"
    "                        group_config.alignment_chunk_count,\n"
)
STORE_NEW = (
    "                    # radiance swa-align: absolute grid, lookup-consistent (see\n"
    "                    # _radiance_store_reachable); stock rule when the grid is not active\n"
    "                    if not _radiance_store_reachable(\n"
    "                        abs_chunk_idx,\n"
    "                        num_chunks,\n"
    "                        group_config.alignment_chunk_count,\n"
)

TOUCH_ANCHOR = (
    "    def _touch(self, req_status: RequestOffloadState):\n"
    "        for group_config, group_state in zip(\n"
    "            self.config.kv_group_configs, req_status.group_states\n"
    "        ):\n"
    "            if group_config.sliding_window_size_in_chunks is None:\n"
)
TOUCH_NEW = (
    "    def _touch(self, req_status: RequestOffloadState):\n"
    "        # radiance touch-all: refresh every group's chunks like attention, so a prefix's\n"
    "        # Mamba/drafter snapshots are not evicted while its attention survives (pattern B).\n"
    "        # Kill switch: RADIANCE_TOUCH_ALL_GROUPS=0.\n"
    "        _rad_touch_all = os.environ.get(\"RADIANCE_TOUCH_ALL_GROUPS\", \"0\") == \"1\"\n"
    "        if _rad_touch_all and os.environ.get(\"RADIANCE_TOUCH_POSITION_ORDER\", \"0\") == \"1\":\n"
    "            # radiance touch-order: ONE touch per request, every group's keys interleaved by\n"
    "            # chunk end position, head to tail. Policies apply a touch list in reverse, so the\n"
    "            # head ends up most recent and eviction takes whole positions from the tail.\n"
    "            # Touching group by group (the loop below) left every g0 key older than every g1\n"
    "            # key: under pressure a conversation lost ALL its g0 Mamba snapshots first while\n"
    "            # attention, g1-g5 and the drafter stayed held and useless (evict_skew). Same\n"
    "            # ordering as upstream PR #51787 order_request_keys.\n"
    "            # Kill switch: RADIANCE_TOUCH_POSITION_ORDER=0.\n"
    "            _rad_keyed = []\n"
    "            for group_config, group_state in zip(\n"
    "                self.config.kv_group_configs, req_status.group_states\n"
    "            ):\n"
    "                _rad_tpc = group_config.tokens_per_chunk\n"
    "                _rad_gidx = group_config.group_idx\n"
    "                for _rad_i, _rad_key in enumerate(group_state.offload_keys):\n"
    "                    _rad_keyed.append(((_rad_i + 1) * _rad_tpc, _rad_gidx, _rad_key))\n"
    "            _rad_keyed.sort(key=lambda t: (t[0], t[1]))\n"
    "            self.manager.touch([t[2] for t in _rad_keyed], req_status.req_context)\n"
    "            return\n"
    "        for group_config, group_state in zip(\n"
    "            self.config.kv_group_configs, req_status.group_states\n"
    "        ):\n"
    "            if group_config.sliding_window_size_in_chunks is None or _rad_touch_all:\n"
)


def main() -> None:
    src = OFFSCHED.read_text() if OFFSCHED.exists() else ""
    if "\nimport os\n" not in src:
        raise SystemExit(f"  FAIL  swa-align/touch-all: {OFFSCHED} has no module-level 'import os'")
    apply(OFFSCHED, MODFLAG_ANCHOR, MODFLAG_NEW, "_RADIANCE_SWA_GRID_ACTIVE = False", "swa-align: module flag")
    apply(OFFSCHED, ALIGN_ANCHOR, ALIGN_NEW, "radiance swa-align: hits are ALSO", "swa-align: store alignment = Mamba grid")
    apply(OFFSCHED, STORE_ANCHOR, STORE_NEW, "radiance swa-align: absolute grid, lookup-consistent", "swa-align: absolute-grid, lookup-consistent reachability")
    apply(OFFSCHED, TOUCH_ANCHOR, TOUCH_NEW, "radiance touch-all:", "touch-all: refresh every group like attention")


if __name__ == "__main__":
    main()
