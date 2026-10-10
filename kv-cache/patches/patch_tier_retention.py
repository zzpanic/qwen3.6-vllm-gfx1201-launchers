#!/usr/bin/env python3
"""Tier checkpoint retention: store Mamba/drafter rows in the offload tiers only where a turn can resume.

WHY (the author's bench notes (checkpoint-policy-20261010/notes.md, not shipped) + PLAN.md; the maintainer, 2026-10-10: "maximise token cache hits while
minimising vram/ram and disk tier storage")
==========================================================================================================
Per 3,520-token grid chunk the tier stores 8 attention rows, 4 drafter rows and (stride 4) 6 Mamba rows: 18 equal
rows, 10 of them checkpoints. A tier hit at grid point E needs attention for every chunk below E but a Mamba state
only AT E (plus the drafter window ending at E and its EAGLE peek chunk). Real resumes land at the previous prompt's
tip, the system-prompt junction, or (7 of 8,268 calls) a few chunks below the tip. Replayed on the real traffic,
keeping the tip + a geometric backoff of 8 + the system prompt + the junction gives the same hits as dense with about
half the RAM + disk (RAM 12 GiB + disk 96 GB: 1/3/4 agents 92.52 / 92.53 / 92.47% served vs ideal 92.56%).

WHAT IT CHANGES (gate RADIANCE_TIER_RETENTION=1; 0/unset = today)
  OffloadingConnectorScheduler._build_store_jobs: per request, the retained grid points R are (grid chunk index i,
  state after (i+1)*S tokens; tip t = the last full grid chunk of the prompt):
    - geometric backoff: d = t - i kept when i is a multiple of 2^floor(log2 d), nearest RADIANCE_TIER_GEO_K (8)
      including d = 0 (the tip) and d = 1 (the resumable tip when the prompt ran < 1 block past T, hunk 5 rule);
    - every grid point <= RADIANCE_VRAM_KEEP_PREFIX (14,080: the system prompt, shared, one copy per prompt);
    - the request's shared-prefix junction floored to the grid (upstream Marconi signal, when set).
  A Mamba chunk is stored only when it ends on a retained point (and on the R3.13 stride, which still applies).
  A sliding-window (drafter) chunk is stored only when it lies in the window ending at a retained point, or is the
  EAGLE peek chunk starting at one. Attention groups are untouched (dense).
  Lookup is unchanged: a Mamba group is a window-1 scan from the end, so it finds the deepest stored state below the
  requested depth, and _lookup re-runs the drafter group at that depth.
Exact: the tier only decides what is stored; a hit still lands on the canonical grid and loads the same bytes.
Composes with patch_vram_retention.py (same KEEP_PREFIX) and the R3.13 stride. Disk is written by the same store
(write-to-all), so the disk tier gets the same thinned rows; restores still go disk -> RAM -> VRAM.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

PUR = Path(sysconfig.get_paths()["purelib"])
OFFS = PUR / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
SCHED = PUR / "vllm/v1/core/sched/scheduler.py"

HELPER = '''_RADIANCE_ASSERT_DUMPED = False

# ---- radiance (patch_tier_retention.py) ----
import functools
_RADIANCE_TIER_RETENTION = os.environ.get("RADIANCE_TIER_RETENTION", "0") == "1"
_RADIANCE_TIER_GEO_K = max(2, int(os.environ.get("RADIANCE_TIER_GEO_K", "8")))
_RADIANCE_TIER_KEEP_PREFIX = int(os.environ.get("RADIANCE_VRAM_KEEP_PREFIX", "14080"))
_RADIANCE_TIER_GRID = [0]   # canonical grid, set by the scheduler on its first schedule()


@functools.lru_cache(maxsize=4096)
def _radiance_tier_retained(num_prompt_tokens, shared_prefix_boundary, grid):
    """Retained grid points as grid chunk indices (i = state after (i + 1) * grid tokens)."""
    t = num_prompt_tokens // grid - 1
    if t < 0:
        return frozenset()
    keep = set()
    for d in range(0, t + 1):  # geometric backoff from the tip, nearest K
        i = t - d
        if d == 0 or i % (1 << (d.bit_length() - 1)) == 0:
            keep.add(i)
            if len(keep) >= _RADIANCE_TIER_GEO_K:
                break
    keep.update(range(0, min(t + 1, _RADIANCE_TIER_KEEP_PREFIX // grid)))
    if shared_prefix_boundary and shared_prefix_boundary // grid >= 1:
        keep.add(min(t, shared_prefix_boundary // grid - 1))
    return frozenset(keep)


def _radiance_tier_keep(req_status, group_config, abs_chunk_idx):
    """False when this Mamba / sliding-window chunk serves no retained grid point."""
    grid = _RADIANCE_TIER_GRID[0]
    is_window = group_config.sliding_window_size_in_chunks is not None
    if not (_RADIANCE_TIER_RETENTION and grid and is_window):
        return True
    req = req_status.req
    rs = _radiance_tier_retained(req.num_prompt_tokens, req.shared_prefix_boundary or 0, grid)
    tpc = group_config.tokens_per_chunk
    start, end = abs_chunk_idx * tpc, (abs_chunk_idx + 1) * tpc
    if group_config.is_mamba_group:
        return end % grid == 0 and end // grid - 1 in rs
    # sliding window (drafter): in the window ending at a retained point, or its EAGLE peek chunk
    span = group_config.sliding_window_size_in_chunks * tpc
    e = -(-end // grid) * grid
    while e - end < span:
        if e // grid - 1 in rs:
            return True
        e += grid
    return bool(group_config.is_eagle_group) and start % grid == 0 and start // grid - 1 in rs'''

apply(OFFS, "_RADIANCE_ASSERT_DUMPED = False\n", HELPER + "\n",
      "def _radiance_tier_retained(", "tier retention: helper")

apply(OFFS,
      "                    # radiance swa-align: absolute grid, lookup-consistent (see\n"
      "                    # _radiance_store_reachable); stock rule when the grid is not active\n",
      "                    # radiance (patch_tier_retention.py): Mamba / drafter rows only where a turn\n"
      "                    # can resume (tip + geometric backoff + system prompt + junction)\n"
      "                    if not _radiance_tier_keep(req_status, group_config, abs_chunk_idx):\n"
      "                        continue\n"
      "                    # radiance swa-align: absolute grid, lookup-consistent (see\n"
      "                    # _radiance_store_reachable); stock rule when the grid is not active\n",
      "if not _radiance_tier_keep(req_status, group_config, abs_chunk_idx):",
      "tier retention: filter in _build_store_jobs")

apply(SCHED,
      "            self.kv_cache_manager._radiance_canon_stats = dict.fromkeys(\n"
      "                (\"alone\", \"spec\", \"refused\", \"dropped\"), 0)\n",
      "            self.kv_cache_manager._radiance_canon_stats = dict.fromkeys(\n"
      "                (\"alone\", \"spec\", \"refused\", \"dropped\"), 0)\n"
      "            # radiance (patch_tier_retention.py): the offload store retains on this grid.\n"
      "            import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as _rad_offs\n"
      "            _rad_offs._RADIANCE_TIER_GRID[0] = self.kv_cache_manager._radiance_grid\n",
      "_rad_offs._RADIANCE_TIER_GRID[0]", "tier retention: grid handoff")
