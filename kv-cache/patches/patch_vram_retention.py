#!/usr/bin/env python3
"""VRAM checkpoint retention on the canonical grid: keep Mamba/drafter states only where a turn can resume.

WHY (the author's bench notes (checkpoint-policy-20261010/notes.md, not shipped); the maintainer, 2026-10-10: "maximise token cache hits while
minimising vram/ram and disk tier storage")
==========================================================================================================
Per 3,520-token grid chunk the GPU caches 8 attention pages AND 10 checkpoint pages (6 Mamba state + 4 drafter
window). A hit needs attention for every chunk below it but a state only AT its resume point, and real agent
turns resume at the previous prompt's tip or at the shared system-prompt junction (opencode.db: 7 of 8,268 calls
resumed anywhere else). Keeping only those states roughly doubles the cached context the pool holds; replayed
on the real traffic it lifts 3 agents VRAM-only 78.5% -> 87.0% served and 4 agents RAM16 59.8% -> 85.8%.

Upstream vLLM 0.27.1 already has the mechanism: VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 makes the Mamba and
sliding-window managers cache only the blocks a "reachable boundary" needs (cache_blocks -> reachable_block_mask:
the state block ending on the boundary; the drafter window ending on it plus the EAGLE peek block). Its boundaries
are wrong for our stack:
  - replay boundary = num_prompt_tokens - 1 floored to the 880 block. Canonical chunks (hunk 4) register states
    only at GRID points (3,520), so that block is never registered and the grid state below it is masked away:
    the next turn would find no state at all.
  - when the prompt ran less than one block past its last grid point T, the next turn resumes at T - grid (the
    EAGLE lookahead block past T is not full -- hunk 5 rule). 24.6% of real prompts.
  - the shared-prefix junction (Marconi) is block-aligned, not grid-aligned.

WHAT IT CHANGES (gate RADIANCE_VRAM_RETENTION=1, and only with VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 set, which is
what switches upstream's mask on; with either unset nothing changes)
  SingleTypeKVCacheManager.cache_blocks: when the canonical grid is known, the reachable boundaries become
    T = the last grid point <= the prompt, the RESUMABLE tip (T, or T - grid when the tail is < one block),
    the shared-prefix junction floored to the grid, and every grid point up to RADIANCE_VRAM_KEEP_PREFIX
    (default 14,080: the system prompt, ~12-15k here; shared, so one copy per distinct prompt -- without it the
    FIRST sibling after a cold start recomputes the system prompt, since upstream only detects the junction on
    the second).
  NOT canon_end: hunk 4 advances it chunk by chunk, so it would retain every state. Cost: a request frozen below T
    (a budget-split chunk under concurrency) keeps only the junction states -- fix the freeze, not this.
  Scheduler: hands the grid to the managers on the first schedule() (beside the hunk 1-6 handoff).
Exact: retention decides which blocks stay cached, never what is computed. Attention groups stay dense.
Masked blocks are freed uncached (reused first); the offload connector already flushes or zeroes stores whose
Mamba/sliding source block was reallocated, so the tier never copies a reused block.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

PUR = Path(sysconfig.get_paths()["purelib"])
STKM = PUR / "vllm/v1/core/single_type_kv_cache_manager.py"
SCHED = PUR / "vllm/v1/core/sched/scheduler.py"

HELPER = '''

# ---- radiance (patch_vram_retention.py) ----
import os as _rad_vr_os
_RADIANCE_VRAM_RETENTION = _rad_vr_os.environ.get("RADIANCE_VRAM_RETENTION", "0") == "1"
_RADIANCE_VRAM_KEEP_PREFIX = int(_rad_vr_os.environ.get("RADIANCE_VRAM_KEEP_PREFIX", "14080"))
_RADIANCE_GRID = [0]   # set by the scheduler on its first schedule()


def _radiance_vram_boundaries(request, grid, block):
    P = request.num_prompt_tokens
    t = P // grid * grid
    import vllm.v1.core.block_pool as _bp
    # hunk 5 receipts on: the hit lands at t when the prompt ran a full block past it; off (the known
    # hunk 4 shortfall) it can land one grid chunk lower either way, so keep both
    out = [t, t if (P - t >= block and getattr(_bp, "_RADIANCE_RECEIPTS", False)) else t - grid]
    spb = request.shared_prefix_boundary
    if spb:
        out.append(spb // grid * grid)
    # the system-prompt region: shared by every conversation, so these dedupe to one copy per prompt
    out += range(grid, min(t, _RADIANCE_VRAM_KEEP_PREFIX) + 1, grid)
    return sorted({b for b in out if b > 0})


class SingleTypeKVCacheManager('''

apply(STKM, "\n\nclass SingleTypeKVCacheManager(", HELPER,
      "def _radiance_vram_boundaries(", "vram retention: boundary helper")

apply(STKM,
      "        reachable_boundaries = [request.num_prompt_tokens - 1]\n"
      "        if request.shared_prefix_boundary:\n"
      "            reachable_boundaries.append(request.shared_prefix_boundary)\n",
      "        reachable_boundaries = [request.num_prompt_tokens - 1]\n"
      "        if request.shared_prefix_boundary:\n"
      "            reachable_boundaries.append(request.shared_prefix_boundary)\n"
      "        if _RADIANCE_VRAM_RETENTION and _RADIANCE_GRID[0]:\n"
      "            # radiance (patch_vram_retention.py): boundaries on the canonical grid\n"
      "            reachable_boundaries = _radiance_vram_boundaries(\n"
      "                request, _RADIANCE_GRID[0], self.scheduler_block_size)\n",
      "_radiance_vram_boundaries(\n                request,", "vram retention: grid boundaries in cache_blocks")

apply(SCHED,
      "            self.kv_cache_manager._radiance_canon_stats = dict.fromkeys(\n"
      "                (\"alone\", \"spec\", \"refused\", \"dropped\"), 0)\n",
      "            self.kv_cache_manager._radiance_canon_stats = dict.fromkeys(\n"
      "                (\"alone\", \"spec\", \"refused\", \"dropped\"), 0)\n"
      "            # radiance (patch_vram_retention.py): the managers retain states on this grid.\n"
      "            import vllm.v1.core.single_type_kv_cache_manager as _rad_stkm\n"
      "            _rad_stkm._RADIANCE_GRID[0] = self.kv_cache_manager._radiance_grid\n",
      "_rad_stkm._RADIANCE_GRID[0]", "vram retention: grid handoff")
