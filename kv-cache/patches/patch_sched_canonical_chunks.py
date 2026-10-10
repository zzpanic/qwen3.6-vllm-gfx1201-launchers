#!/usr/bin/env python3
"""Make a prefix-hit prefill chunk exactly like a cold prefill (align mode): cached == uncached.

WHY (the author's bench notes (divergence-20261010, not shipped), GDN state fingerprints per layer per step)
====================================================================================
A request that resumes from a prefix hit (GPU-local or offload tier) diverged from a cold
prefill of the same prompt: fp32 GPU partial hit, and the fp16 tier hit that kept fp16 ssm out
of production. The fingerprints show the state it resumes from is BIT-IDENTICAL to the cold
run's state at that position (48/48 GDN layers, fp16 and fp32). What differs is the chunking
after the resume:

    cold   ... [24,640, 28,160) [28,160, 29,920) [29,920, 30,800) [30,800, end)
    probe              resume at 28,160: [28,160, 29,040) [29,040, 29,920) ...

The extra stop at 29,040 is stock vLLM 0.27.1's Marconi "shared-prefix junction"
(`request.shared_prefix_boundary`, set by get_computed_blocks to where the longest single-group
hit ends): it ends a chunk there so the junction's Mamba state gets cached for sibling requests.
A cold prefill never stops there. Different chunk lengths change the GEMM M-tiling (a block is
not a multiple of 64 rows: 1,648 = 25x64+48, 880 = 13x64+48) and the GDN chunk-scan's 64-token
chunking, so every token after the junction is computed differently. At fp16 the state is also
rounded at every chunk end, so the difference is larger -- the same mechanism, not an fp16
checkpoint defect.

WHAT IT CHANGES (gate RADIANCE_CANONICAL_CHUNKS, unset/0 = upstream; launcher default 1)
  1. Drops the shared-prefix junction stop. What that costs: a later sibling that shares
     exactly up to the junction recomputes up to one block (<= 1,648 tokens) instead of
     hitting the junction state. That's under a second of prefill, against cached != cold
     on every request that set a junction.
  2. Adds a stop at the cold chunk grid: the next multiple of S from token 0, S = the chunk a
     cold prefill takes (token budget, or long_prefill_token_threshold, floored to a block).
     A request resuming at any block boundary then re-joins the cold grid on its first chunk
     instead of running a full S from wherever it resumed. Cold prefills from 0 already stop
     there, so a solo cold prefill is unchanged.
Only adds/removes stops strictly inside a chunk; every stop stays block-aligned (the slot
invariant _mamba_block_aligned_split exists to keep).

  3. (2026-10-10, turnbench c95de9 FAIL 17/21 at fp16 with 1+2 only) Floors every LOCAL prefix hit to
     the cold grid. Multi-turn: the previous turn leaves Mamba checkpoints at ITS tail stops (eagle /
     last-full-block), e.g. 22,880 with S = 3,520; resuming there runs [22,880, ...) where the cold
     twin runs [21,120, 24,640) in one chunk. Floored, the turn resumes at 21,120 and recomputes the
     same chunks a cold prefill does. Cost: up to S-1 extra recomputed tokens per hit (~1 s). Tier
     hits are already on the grid (the store stride is a multiple of S here), so the connector's
     extension is left alone. Applied in get_computed_blocks (loop: re-query with the floored cap
     until the hit lies on the grid) and get_computed_blocks_for_connector (an off-grid attention
     hit falls back to the floored get_computed_blocks). The grid is handed to the KV cache manager
     by the scheduler on its first schedule() call.

  4. (2026-10-10, divergence-20261010/mt) Caches only blocks computed inside a FULL grid chunk.
     A block's KV depends on the LENGTH of the chunk it was computed in (M-dependent tiling), not
     only on its start: turn N computes its tail in short chunks (eagle / last-block stops) and
     registers those blocks first; turn N+1 recomputes the same tokens in a full chunk, but vLLM
     keeps the first-registered block, so turn N+2 read turn N's tail KV and diverged.
     Each request carries _radiance_canon_end: the end of the contiguous run of full chunks
     [kS, (k+1)S) it computed (starting from its grid-floored hit). GPU prefix-cache registration
     (allocate_slots, cache_blocks) and the offload store (_calc_num_offloadable_tokens) are both
     capped at it. Tail chunks, decode tokens and budget-split chunks are never cached.
     Costs: a prompt shorter than S is never cached; a prefill step whose chunk is cut short by a
     concurrent prefill is not cached.
     Guard: an external (tier) hit whose end is off the grid is refused (recompute) -- can only
     happen if the store stride is not a multiple of S (today 2S at fp32, S at fp16).

LIMIT: this makes a SOLO cached request equal a SOLO cold one. Under concurrency the token
budget is shared and chunk lengths depend on the other requests, cold or cached alike.
REVERT: RADIANCE_CANONICAL_CHUNKS=0, or drop the prelude line.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SCHED = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/core/sched/scheduler.py"

apply(SCHED,
      "import itertools\n",
      "import itertools\n"
      "import os as _radiance_cc_os\n"
      "# radiance: see patch_sched_canonical_chunks.py. Unset/0 = upstream behaviour.\n"
      "_RADIANCE_CANONICAL_CHUNKS = _radiance_cc_os.environ.get(\"RADIANCE_CANONICAL_CHUNKS\", \"0\") == \"1\"\n",
      "_RADIANCE_CANONICAL_CHUNKS = ",
      "canonical chunks: gate constant")

apply(SCHED,
      "            start + (request.shared_prefix_boundary - start) // block_size * block_size\n"
      "            if start < request.shared_prefix_boundary < end\n"
      "            else 0,\n",
      "            start + (request.shared_prefix_boundary - start) // block_size * block_size\n"
      "            if start < request.shared_prefix_boundary < end\n"
      "            and not _RADIANCE_CANONICAL_CHUNKS\n"
      "            else 0,\n"
      "            # radiance (patch_sched_canonical_chunks.py): re-join the cold chunk grid.\n"
      "            self._radiance_cold_grid_stop(start, block_size)\n"
      "            if _RADIANCE_CANONICAL_CHUNKS\n"
      "            else 0,\n",
      "self._radiance_cold_grid_stop(start, block_size)",
      "canonical chunks: drop junction stop, add cold-grid stop")

apply(SCHED,
      "    def _mamba_block_aligned_split(\n",
      "    def _radiance_cold_grid_stop(self, start: int, block_size: int) -> int:\n"
      "        # The next end a cold prefill (chunks of S from token 0) would use after start.\n"
      "        cap = self.max_num_scheduled_tokens\n"
      "        thr = self.scheduler_config.long_prefill_token_threshold\n"
      "        if thr > 0:\n"
      "            cap = min(cap, thr)\n"
      "        S = cap // block_size * block_size\n"
      "        return (start // S + 1) * S if S > 0 else 0\n"
      "\n"
      "    def _mamba_block_aligned_split(\n",
      "def _radiance_cold_grid_stop(self, start: int, block_size: int) -> int:",
      "canonical chunks: cold-grid helper")

# --- 3. floor local prefix hits to the cold grid ---
apply(SCHED,
      "        self.current_step += 1\n",
      "        self.current_step += 1\n"
      "        if _RADIANCE_CANONICAL_CHUNKS and not hasattr(self.kv_cache_manager, \"_radiance_grid\"):\n"
      "            # radiance (patch_sched_canonical_chunks.py): the cold chunk grid, for hit flooring.\n"
      "            self.kv_cache_manager._radiance_grid = (\n"
      "                self._radiance_cold_grid_stop(0, self.cache_config.block_size)\n"
      "                if self.need_mamba_block_aligned_split else 0)\n",
      "self.kv_cache_manager._radiance_grid = (",
      "canonical chunks: hand the grid to the KV cache manager")

KVCM = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/core/kv_cache_manager.py"
apply(KVCM,
      "        computed_blocks, num_new_computed_tokens, num_uncached = (\n"
      "            self.coordinator.find_longest_cache_hit(\n"
      "                request.block_hashes, max_cache_hit_length\n"
      "            )\n"
      "        )\n",
      "        computed_blocks, num_new_computed_tokens, num_uncached = (\n"
      "            self.coordinator.find_longest_cache_hit(\n"
      "                request.block_hashes, max_cache_hit_length\n"
      "            )\n"
      "        )\n"
      "        # radiance (patch_sched_canonical_chunks.py): resume only on the cold chunk grid.\n"
      "        _rad_grid = getattr(self, \"_radiance_grid\", 0)\n"
      "        while _rad_grid and num_new_computed_tokens % _rad_grid:\n"
      "            computed_blocks, num_new_computed_tokens, num_uncached = (\n"
      "                self.coordinator.find_longest_cache_hit(\n"
      "                    request.block_hashes,\n"
      "                    num_new_computed_tokens // _rad_grid * _rad_grid,\n"
      "                )\n"
      "            )\n",
      "_rad_grid = getattr(self, \"_radiance_grid\", 0)\n        while",
      "canonical chunks: floor get_computed_blocks to the grid")
apply(KVCM,
      "        if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):\n",
      "        _rad_grid = getattr(self, \"_radiance_grid\", 0)\n"
      "        if _rad_grid and per_group_hits[fa_group_id] % _rad_grid:\n"
      "            # radiance (patch_sched_canonical_chunks.py): an off-grid local hit resumes mid\n"
      "            # cold chunk; take the floored, group-consistent hit instead.\n"
      "            return *self.get_computed_blocks(request), False\n"
      "        if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):\n",
      "if _rad_grid and per_group_hits[fa_group_id] % _rad_grid:",
      "canonical chunks: floor the connector local hit to the grid")

# --- 4. cache only full-grid-chunk blocks; refuse off-grid external hits ---
apply(KVCM,
      "        # P/D: delay caching blocks if we have to recv from\n"
      "        # remote. Update state for locally cached blocks.\n"
      "        if not self.enable_caching or delay_cache_blocks:\n",
      "        # radiance (patch_sched_canonical_chunks.py): track the full-grid-chunk prefix.\n"
      "        _rad_grid = getattr(self, \"_radiance_grid\", 0)\n"
      "        if _rad_grid:\n"
      "            _rad_ce = getattr(request, \"_radiance_canon_end\", None)\n"
      "            if _rad_ce is None or total_computed_tokens < _rad_ce:  # new, or preempted\n"
      "                _rad_ce = total_computed_tokens // _rad_grid * _rad_grid\n"
      "            if (total_computed_tokens == _rad_ce and num_new_tokens == _rad_grid\n"
      "                    and _rad_ce + _rad_grid <= request.num_prompt_tokens):\n"
      "                _rad_ce += _rad_grid\n"
      "            request._radiance_canon_end = _rad_ce\n"
      "        # P/D: delay caching blocks if we have to recv from\n"
      "        # remote. Update state for locally cached blocks.\n"
      "        if not self.enable_caching or delay_cache_blocks:\n",
      "request._radiance_canon_end = _rad_ce",
      "canonical chunks: track the full-grid-chunk prefix")
apply(KVCM,
      "        self.coordinator.cache_blocks(request, num_tokens_to_cache)\n"
      "\n"
      "        return self.create_kv_cache_blocks(new_blocks)\n",
      "        if _rad_grid:\n"
      "            num_tokens_to_cache = min(num_tokens_to_cache, request._radiance_canon_end)\n"
      "        self.coordinator.cache_blocks(request, num_tokens_to_cache)\n"
      "\n"
      "        return self.create_kv_cache_blocks(new_blocks)\n",
      "num_tokens_to_cache = min(num_tokens_to_cache, request._radiance_canon_end)",
      "canonical chunks: cap GPU registration (allocate_slots)")
apply(KVCM,
      "        if self.enable_caching:\n"
      "            self.coordinator.cache_blocks(request, num_computed_tokens)\n",
      "        if self.enable_caching:\n"
      "            # radiance (patch_sched_canonical_chunks.py): only the full-grid-chunk prefix.\n"
      "            _rad_ce = getattr(request, \"_radiance_canon_end\", None)\n"
      "            if _rad_ce is not None:\n"
      "                num_computed_tokens = min(num_computed_tokens, _rad_ce)\n"
      "            self.coordinator.cache_blocks(request, num_computed_tokens)\n",
      "                num_computed_tokens = min(num_computed_tokens, _rad_ce)\n",
      "canonical chunks: cap GPU registration (cache_blocks)")

OFFS = Path(sysconfig.get_paths()["purelib"]) / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
apply(OFFS,
      "        if self.config.offload_prompt_only:\n"
      "            num = min(num, req_status.req.num_prompt_tokens)\n"
      "        return num\n",
      "        if self.config.offload_prompt_only:\n"
      "            num = min(num, req_status.req.num_prompt_tokens)\n"
      "        # radiance (patch_sched_canonical_chunks.py): store only the full-grid-chunk prefix.\n"
      "        _rad_ce = getattr(req_status.req, \"_radiance_canon_end\", None)\n"
      "        if _rad_ce is not None:\n"
      "            num = min(num, _rad_ce)\n"
      "        return num\n",
      "store only the full-grid-chunk prefix",
      "canonical chunks: cap the offload store")

apply(SCHED,
      "                if self.need_mamba_block_aligned_split else 0)\n",
      "                if self.need_mamba_block_aligned_split else 0)\n"
      "            _rad_g = self.kv_cache_manager._radiance_grid\n"
      "            if _rad_g and self.connector is not None:\n"
      "                _rad_orig = self.connector.get_num_new_matched_tokens\n"
      "\n"
      "                def _rad_matched(request, num_computed_tokens, _o=_rad_orig, _g=_rad_g):\n"
      "                    ext, load_async = _o(request, num_computed_tokens)\n"
      "                    if ext and (num_computed_tokens + ext) % _g:\n"
      "                        # an external hit ending off the cold grid resumes mid chunk\n"
      "                        logger.warning_once(\"radiance canonical chunks: refusing an \"\n"
      "                                            \"off-grid external hit (store stride is \"\n"
      "                                            \"not a multiple of the chunk grid)\")\n"
      "                        return 0, False\n"
      "                    return ext, load_async\n"
      "\n"
      "                self.connector.get_num_new_matched_tokens = _rad_matched\n",
      "def _rad_matched(request, num_computed_tokens",
      "canonical chunks: refuse off-grid external hits")
