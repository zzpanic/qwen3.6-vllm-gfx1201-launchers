#!/usr/bin/env python3
"""Canonical chunks, hunk 5: give the EAGLE/drafter group back its one-block lookahead, as a RECEIPT.

WHY (the author's bench notes (dynamic-mamba-20261010/notes.md, not shipped), "The EAGLE-lookahead regression")
=======================================================================================
The drafter's KV at a block's last position is computed from the NEXT token, so a prefix hit at L
must prove the request's block [L, L+bs) matches before it may reuse the drafter KV ending at L.
Upstream does this with a peek: an EAGLE group matches one block past the candidate and drops it
(SlidingWindowManager / FullAttentionManager find_longest_cache_hit, drop_eagle_block), and
kv_cache_coordinator.cache_blocks caches `aligned + block_size` for EAGLE managers so that the
peeked block exists.

patch_sched_canonical_chunks.py hunk 4 caps caching at request._radiance_canon_end for ALL managers.
That also removes the peeked block at the end of every canonical prefix, so the drafter group fails
there, the hit drops one block and hunk 3 floors it to the grid: every follow-up turn resumes one
whole grid chunk (3,520 tok at fp16) early. Replay of the real opencode trace (sim3.py, full trace,
RAM+disk): served 89.6% vs 93.5%, recomputed 76.7M vs 47.7M tokens (+61%).

WHAT IT CHANGES (gate RADIANCE_CANON_RECEIPTS, unset/0 = hunk 4 behaviour; needs canonical chunks)
  1. EAGLE managers also cache ONE block past canon_end (when it is full): the receipt.
  2. A receipt's KV was computed in a tail chunk, so it is NOT canonical. It is flagged receipt-only
     (BlockPool._radiance_receipts, a set of block ids -- KVCacheBlock has __slots__):
       - a hit may use a receipt only as the block it drops (the peek); if a receipt would be used
         as hit DATA (in a window or a full-attention prefix), the hit is cut before it and floored
         to the grid (_radiance_hit_cap, in get_computed_blocks and the connector lookup);
       - registering a canonical block under the same hash evicts the receipts there, so the exact
         copy is the one found (vLLM keeps duplicates and returns the first);
       - the flag dies with the block's hash (_remove_cached_block_hashes) and follows
         move_block_hashes.
     Whether a receipt can ever sit in a hit's data depends on the geometry: never when
     grid_blocks > cdiv(sliding_window - 1, block_size) for an SWA drafter (fp16 today: 4 > 3), but
     yes at fp32 (2 > 2 false) and for full-attention MTP draft groups. The flag makes it exact in
     every case; it only costs hit length when a receipt would have been read.
     Re-asking for a capped hit leaves one EAGLE block of headroom past the cap (receipts on only):
     find_longest_cache_hit never peeks past its max length, so asking for exactly a grid point
     returns one block short and the floor then drops a further whole chunk. (I first assumed the
     Mamba states snap production's first answer to the grid so this re-ask is rare; the 10-07
     replay says otherwise: 118 follow-ups with no receipt possible resumed TWO chunks early under
     hunk 4 and one chunk early with this patch. Mechanism not pinned beyond that.)
  3. GPU prefix cache only. The offload store stays capped at canon_end (tier hits at the tip keep
     losing one chunk; the GPU serves ~99% of hits on the opencode replay).
Memory: none reserved. The receipt block is already allocated and computed by the request; this
keeps its hash, so it sits in the LRU free queue as a cached block instead of an anonymous one.

GATE (2026-10-10, the author's bench notes (dynamic-mamba-20261010/h5gate, not shipped)): turnbench 21/21 exact at fp16 and fp32
incl. spec counters; reask 120k/180k exact; 10-07 replay served 87.64% -> 91.82% (predicted 91.4..92.7),
recompute -33.8%, 0 calls worse.
REVERT: RADIANCE_CANON_RECEIPTS=0, or drop the prelude line.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

PURE = Path(sysconfig.get_paths()["purelib"])
POOL = PURE / "vllm/v1/core/block_pool.py"
KVCM = PURE / "vllm/v1/core/kv_cache_manager.py"

# ---------------------------------------------------------------- block_pool.py
apply(POOL,
      "\n\nclass BlockHashToBlockMap:\n",
      "\nimport os as _radiance_rc_os\n"
      "# radiance (patch_canon_receipts.py): receipt-only EAGLE lookahead blocks. Unset/0 = off.\n"
      "_RADIANCE_RECEIPTS = _radiance_rc_os.environ.get(\"RADIANCE_CANON_RECEIPTS\", \"0\") == \"1\"\n"
      "\n\nclass BlockHashToBlockMap:\n",
      "_RADIANCE_RECEIPTS = ",
      "canon receipts: gate constant")

apply(POOL,
      "        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()\n"
      "        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}\n",
      "        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()\n"
      "        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}\n"
      "        # radiance (patch_canon_receipts.py): ids of cached receipt-only blocks.\n"
      "        self._radiance_receipts: set[int] = set()\n",
      "self._radiance_receipts: set[int] = set()",
      "canon receipts: receipt set")

apply(POOL,
      "            self._insert_block_hash(\n"
      "                block_hash_with_group_id,\n"
      "                blk,\n"
      "                num_tokens=num_hash_tokens,\n"
      "            )\n",
      "            _rad_ce = (getattr(request, \"_radiance_canon_end\", None)\n"
      "                       if _RADIANCE_RECEIPTS else None)\n"
      "            if _rad_ce is not None:\n"
      "                if num_hash_tokens > _rad_ce:\n"
      "                    # radiance (patch_canon_receipts.py): past the canonical prefix -- the\n"
      "                    # EAGLE lookahead block. Receipt-only: a hit may only drop it.\n"
      "                    self._radiance_receipts.add(blk.block_id)\n"
      "                else:\n"
      "                    self._radiance_drop_receipts(block_hash_with_group_id)\n"
      "            self._insert_block_hash(\n"
      "                block_hash_with_group_id,\n"
      "                blk,\n"
      "                num_tokens=num_hash_tokens,\n"
      "            )\n",
      "self._radiance_drop_receipts(block_hash_with_group_id)",
      "canon receipts: flag receipts, replace them with canonical copies")

apply(POOL,
      "        block.reset_hash()\n"
      "        return removed_hashes\n",
      "        self._radiance_receipts.discard(block.block_id)\n"
      "        block.reset_hash()\n"
      "        return removed_hashes\n"
      "\n"
      "    def _radiance_drop_receipts(self, key: BlockHashWithGroupId) -> None:\n"
      "        # radiance (patch_canon_receipts.py): an exact copy is being registered under key;\n"
      "        # uncache the receipt-only blocks there so lookups find the exact one.\n"
      "        if not self._radiance_receipts:\n"
      "            return\n"
      "        blocks = self.cached_block_hash_to_block._cache.get(key)\n"
      "        if blocks is None:\n"
      "            return\n"
      "        cands = [blocks] if isinstance(blocks, KVCacheBlock) else list(blocks.values())\n"
      "        for b in cands:\n"
      "            if b.block_id in self._radiance_receipts:\n"
      "                self._emit_block_removed_events(self._remove_cached_block_hashes(b))\n",
      "def _radiance_drop_receipts(self, key: BlockHashWithGroupId) -> None:",
      "canon receipts: flag dies with the hash; drop helper")

apply(POOL,
      "        num_tokens = src_block.block_hash_num_tokens\n"
      "        for block_hash in self._remove_cached_block_hashes(src_block):\n"
      "            # `num_tokens` only applies to the first (primary) insertion.\n"
      "            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)\n",
      "        num_tokens = src_block.block_hash_num_tokens\n"
      "        _rad_receipt = src_block.block_id in self._radiance_receipts\n"
      "        for block_hash in self._remove_cached_block_hashes(src_block):\n"
      "            # `num_tokens` only applies to the first (primary) insertion.\n"
      "            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)\n"
      "        if _rad_receipt:  # radiance (patch_canon_receipts.py): the flag follows the hash\n"
      "            self._radiance_receipts.add(dst_block.block_id)\n",
      "_rad_receipt = src_block.block_id in self._radiance_receipts",
      "canon receipts: flag follows move_block_hashes")

apply(POOL,
      "        self.cached_block_hash_to_block = BlockHashToBlockMap()\n"
      "        self.cached_block_hashes_by_block.clear()\n",
      "        self.cached_block_hash_to_block = BlockHashToBlockMap()\n"
      "        self.cached_block_hashes_by_block.clear()\n"
      "        self._radiance_receipts.clear()  # radiance (patch_canon_receipts.py)\n",
      "self._radiance_receipts.clear()",
      "canon receipts: reset_prefix_cache clears the flags")

# ---------------------------------------------------------------- kv_cache_manager.py
apply(KVCM,
      "\n\n@dataclass\nclass KVCacheBlocks:\n",
      "\n\n# ---- radiance (patch_canon_receipts.py) ----\n"
      "def _radiance_cache_receipts(coordinator, request, num_tokens: int) -> None:\n"
      "    # EAGLE managers cache ONE block past the canonical prefix (upstream caches\n"
      "    # aligned + block_size for them) so a hit at canon_end can verify the next token.\n"
      "    # BlockPool.cache_full_blocks flags it receipt-only (num_hash_tokens > canon_end).\n"
      "    import vllm.v1.core.block_pool as _bp\n"
      "    if not _bp._RADIANCE_RECEIPTS:\n"
      "        return\n"
      "    ce = getattr(request, \"_radiance_canon_end\", None)\n"
      "    if not ce:\n"
      "        return\n"
      "    for m in coordinator.single_type_managers:\n"
      "        if m.use_eagle:\n"
      "            n = min(num_tokens, ce + m.block_size)\n"
      "            if n // m.block_size > ce // m.block_size:\n"
      "                m.cache_blocks(request, n, retention_interval=getattr(\n"
      "                    coordinator, \"retention_interval\", None))\n"
      "\n"
      "\n"
      "def _radiance_peek_margin(coordinator) -> int:\n"
      "    # Receipts on: one EAGLE block of headroom when re-asking for a capped hit. Off: 0, so\n"
      "    # canonical-chunks hunk 3 behaves exactly as deployed.\n"
      "    import vllm.v1.core.block_pool as _bp\n"
      "    if not _bp._RADIANCE_RECEIPTS:\n"
      "        return 0\n"
      "    return max((m.block_size for m in coordinator.single_type_managers if m.use_eagle),\n"
      "               default=0)\n"
      "\n"
      "\n"
      "def _radiance_hit_cap(coordinator, blocks_by_group, num_tokens: int, grid: int) -> int:\n"
      "    # The longest allowed hit: on the cold grid (canonical chunks hunk 3) and ending before\n"
      "    # any receipt-only block the hit would use as DATA. A receipt the finder dropped is not\n"
      "    # in blocks_by_group, so only a receipt read as KV caps the hit.\n"
      "    cap = num_tokens // grid * grid\n"
      "    rs = getattr(coordinator.block_pool, \"_radiance_receipts\", None)\n"
      "    if rs:\n"
      "        for g, blocks in enumerate(blocks_by_group):\n"
      "            m = coordinator.single_type_managers[g]\n"
      "            if not m.use_eagle:\n"
      "                continue\n"
      "            for i, b in enumerate(blocks):\n"
      "                if not b.is_null and b.block_id in rs:\n"
      "                    cap = min(cap, i * m.block_size // grid * grid)\n"
      "                    break\n"
      "    return cap\n"
      "\n\n@dataclass\nclass KVCacheBlocks:\n",
      "def _radiance_hit_cap(coordinator, blocks_by_group",
      "canon receipts: helpers")

apply(KVCM,
      "        while _rad_grid and num_new_computed_tokens % _rad_grid:\n"
      "            computed_blocks, num_new_computed_tokens, num_uncached = (\n"
      "                self.coordinator.find_longest_cache_hit(\n"
      "                    request.block_hashes,\n"
      "                    num_new_computed_tokens // _rad_grid * _rad_grid,\n"
      "                )\n"
      "            )\n",
      "        while _rad_grid:\n"
      "            # radiance (patch_canon_receipts.py): ... and never read a receipt as KV.\n"
      "            _rad_cap = _radiance_hit_cap(\n"
      "                self.coordinator, computed_blocks, num_new_computed_tokens, _rad_grid)\n"
      "            if _rad_cap >= num_new_computed_tokens:\n"
      "                break\n"
      "            # Re-ask with room for the EAGLE peek past the cap (find_longest_cache_hit never\n"
      "            # peeks past its max length, so asking for exactly the cap lands one block short).\n"
      "            computed_blocks, num_new_computed_tokens, num_uncached = (\n"
      "                self.coordinator.find_longest_cache_hit(\n"
      "                    request.block_hashes,\n"
      "                    min(max_cache_hit_length,\n"
      "                        _rad_cap + _radiance_peek_margin(self.coordinator)),\n"
      "                )\n"
      "            )\n"
      "            if num_new_computed_tokens > _rad_cap:  # never past the cap: plain re-ask\n"
      "                computed_blocks, num_new_computed_tokens, num_uncached = (\n"
      "                    self.coordinator.find_longest_cache_hit(\n"
      "                        request.block_hashes, _rad_cap))\n",
      "            _rad_cap = _radiance_hit_cap(\n",
      "canon receipts: floor get_computed_blocks below receipts")

apply(KVCM,
      "        if _rad_grid and per_group_hits[fa_group_id] % _rad_grid:\n",
      "        if _rad_grid and _radiance_hit_cap(\n"
      "                coordinator, computed, per_group_hits[fa_group_id], _rad_grid\n"
      "        ) < per_group_hits[fa_group_id]:\n"
      "            # (patch_canon_receipts.py: also when a receipt would be read as KV)\n",
      "coordinator, computed, per_group_hits[fa_group_id], _rad_grid",
      "canon receipts: connector lookup falls back below receipts")

apply(KVCM,
      "        if _rad_grid:\n"
      "            num_tokens_to_cache = min(num_tokens_to_cache, request._radiance_canon_end)\n"
      "        self.coordinator.cache_blocks(request, num_tokens_to_cache)\n",
      "        _rad_uncapped = num_tokens_to_cache\n"
      "        if _rad_grid:\n"
      "            num_tokens_to_cache = min(num_tokens_to_cache, request._radiance_canon_end)\n"
      "        self.coordinator.cache_blocks(request, num_tokens_to_cache)\n"
      "        if _rad_grid:\n"
      "            _radiance_cache_receipts(self.coordinator, request, _rad_uncapped)\n",
      "_radiance_cache_receipts(self.coordinator, request, _rad_uncapped)",
      "canon receipts: cache the receipt (allocate_slots)")

apply(KVCM,
      "            _rad_ce = getattr(request, \"_radiance_canon_end\", None)\n"
      "            if _rad_ce is not None:\n"
      "                num_computed_tokens = min(num_computed_tokens, _rad_ce)\n"
      "            self.coordinator.cache_blocks(request, num_computed_tokens)\n",
      "            _rad_ce = getattr(request, \"_radiance_canon_end\", None)\n"
      "            _rad_uncapped = num_computed_tokens\n"
      "            if _rad_ce is not None:\n"
      "                num_computed_tokens = min(num_computed_tokens, _rad_ce)\n"
      "            self.coordinator.cache_blocks(request, num_computed_tokens)\n"
      "            if _rad_ce is not None:\n"
      "                _radiance_cache_receipts(self.coordinator, request, _rad_uncapped)\n",
      "                _radiance_cache_receipts(self.coordinator, request, _rad_uncapped)\n",
      "canon receipts: cache the receipt (cache_blocks)")
