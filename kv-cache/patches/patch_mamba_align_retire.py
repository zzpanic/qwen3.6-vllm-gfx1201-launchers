#!/usr/bin/env python3
# radiance house port 2026-10-06 from SlyBase/vllm-sly-radiance sly/patch_mamba_align_retire.py (written for vLLM 0.29.0;
# the anchors are identical in 0.27.1). No licence published upstream; vLLM is Apache-2.0.
"""Backport of vllm-project/vllm#55450 (fadfe1c7d, 2026-09-11): align-mode Mamba state retirement
crosses null gaps (vLLM 0.29.0, always on).

Why (vllm7, 2026-09-17, T0b `--max-model-len 262144`): a 258k-token prompt preempted itself twice
during prefill and `vllm:kv_cache_usage_perc` peaked at 97 % where full attention alone needs 68 %.
A debug hook on `KVCacheManager.allocate_slots` showed every gated-delta-net group holding
94 non-null blocks at 156k computed tokens (expected ~9: current + previous state + 7 speculative
blocks), full-attention and sliding-window groups were exact.

Mechanism: with `--enable-prefix-caching` the Mamba groups run in `mamba_cache_mode="align"`,
where prefill leaves non-contiguous blocks (relocated speculative blocks become null gaps between
states). `_remove_blocks_in_range` walks backward and stops at the first null block, so it never
reaches older states; the align branch of `remove_skipped_blocks` frees only
`last_state_block_idx`, and only once it is below the *committed* token count. With async
scheduling the committed count lags one chunk behind (`num_in_flight_tokens` = 1792), the check
fails, and the next `allocate_new_blocks` overwrites `last_state_block_idx` -> one block per GDN
group per prefill chunk stays referenced until the request finishes or is preempted
(6 groups x 1 block per 1792 tokens: a 100k prompt pins ~330 of 929 blocks).

Fix (upstream verbatim, adapted to 0.29.0's `__init__`/`pop_blocks_for_free`): in align mode skip
null blocks instead of breaking, and remember the retired prefix per request so the scan does not
restart at block 0 every step. The freed range is the same `[0, (committed - 1) // block_size)`
the stock code intends; states at and after the last committed token are untouched.
"""

import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/core/single_type_kv_cache_manager.py"
L = "single_type_kv_cache_manager: Mamba align retirement crosses null gaps (vllm#55450)"

apply(F,
      "            self.last_state_block_idx: dict[str, int] = {}\n",
      "            self.last_state_block_idx: dict[str, int] = {}\n"
      "            # radiance (sly/patch_mamba_align_retire.py, vllm#55450): retired block prefix\n"
      "            self._num_retired_blocks: dict[str, int] = {}\n",
      "self._num_retired_blocks: dict[str, int]", L + " [init]")

apply(F,
      "    def remove_skipped_blocks(\n"
      "        self,\n"
      "        request_id: str,\n"
      "        processed_computed_tokens: int,\n"
      "        num_prompt_tokens: int | None = None,\n"
      "    ) -> None:\n"
      "        assert isinstance(self.kv_cache_spec, MambaSpec)\n",
      "    def _remove_blocks_in_range(\n"
      "        self, request_id: str, first_block: int, last_block: int\n"
      "    ) -> None:\n"
      "        # radiance (sly/patch_mamba_align_retire.py, vllm#55450)\n"
      "        if self.mamba_cache_mode != \"align\":\n"
      "            return super()._remove_blocks_in_range(request_id, first_block, last_block)\n"
      "        blocks = self.req_to_blocks.get(request_id, [])\n"
      "        first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))\n"
      "        last_block = min(last_block, len(blocks))\n"
      "        if first_block >= last_block:\n"
      "            return\n"
      "        freed: list[KVCacheBlock] = []\n"
      "        # Mamba prefill leaves null gaps between states awaiting retirement.\n"
      "        for i in range(last_block - 1, first_block - 1, -1):\n"
      "            if blocks[i].is_null:\n"
      "                continue\n"
      "            freed.append(blocks[i])\n"
      "            blocks[i] = self._null_block\n"
      "        if freed:\n"
      "            self.block_pool.free_blocks(freed)\n"
      "        self._num_retired_blocks[request_id] = last_block\n"
      "\n"
      "    def remove_skipped_blocks(\n"
      "        self,\n"
      "        request_id: str,\n"
      "        processed_computed_tokens: int,\n"
      "        num_prompt_tokens: int | None = None,\n"
      "    ) -> None:\n"
      "        assert isinstance(self.kv_cache_spec, MambaSpec)\n",
      "# Mamba prefill leaves null gaps between states awaiting retirement.", L + " [retire]")

apply(F,
      "            self.last_state_block_idx.pop(request_id, None)\n",
      "            self.last_state_block_idx.pop(request_id, None)\n"
      "            self._num_retired_blocks.pop(request_id, None)\n",
      "self._num_retired_blocks.pop(request_id, None)", L + " [free]")
