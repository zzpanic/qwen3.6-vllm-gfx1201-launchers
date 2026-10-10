#!/usr/bin/env python3
"""Canonical chunks, hunk 6: decide whether a grid chunk is canonical AFTER its step is built.
WHY (the author's bench notes (dynamic-mamba-20261010/notes.md, not shipped), "Concurrency freezes canonical caching" and
"Code review of option (a)")
=================================================================================================
Hunk 4 (patch_sched_canonical_chunks.py) caches a request only up to _radiance_canon_end, the end
of its run of FULL grid chunks, and advances it in allocate_slots when `num_new_tokens == grid`.
That test means "this chunk ran exactly like a cold solo chunk" only because the step budget
(max_num_scheduled_tokens) is exactly one grid chunk today: a grid chunk then uses the whole step.
The cost is that a prefill sharing a step with ANY decode is split (a decode takes 1 + drafts
tokens of the budget) and its request is never cached again -- at concurrency >= 2 long prompts
are effectively uncached (live: re-send cached 0 vs 17,600 solo).

The fix is a bigger budget (CHUNK 3568 = grid 3520 + 4x6 spec reserve + 3x8 decode rows), so the
chunk keeps its 3,520 rows next to up to three spec decodes. The code review found every kernel on
the cached path row-invariant for M 3520..3544 (MXFP4 GEMM, R4D prefill attention, drafter
context-KV W4A16, norms), with two exceptions this hunk closes:
  - GDN: radiance_gdn's fused path takes a prefill only beside SPEC decodes. A step whose decodes
    have no drafts runs the whole step -- the prefill too -- on stock FLA, which is not the solo
    numerics. (With a spec decode present, gdn_attn reclassifies a draft-less decode as a 1-token
    prefill and the fused path takes it; that composition was not reviewed, so it is refused too.)
  - Scheduler: with budget > grid, `num_new_tokens == grid` no longer implies "alone", and
    allocate_slots registers blocks before the rest of the step is scheduled.

WHAT IT CHANGES (only when the budget exceeds the grid; at CHUNK 3544 this code never runs)
  1. allocate_slots does not advance canon_end for a full grid chunk. It records the target on the
     request (_radiance_canon_pending) and defers the decision, and it skips the hunk 5 receipt
     for that request (the settle step caches it if the chunk is refused, as a frozen request does).
  2. schedule(), once its loops are done and before the KV-copy, offload-store and output metadata
     are built, settles each deferred chunk:
       accept  -- the chunk ran alone, or RADIANCE_CANON_MIXED=1 and every other request in the step
                  is a speculative decode (exactly 1 + drafts tokens, drafts >= 1): canon_end moves
                  to the target and the blocks are registered (GPU prefix cache; the offload store
                  reads canon_end when the connector metadata is built right after);
       refuse  -- anything else (a draft-less decode, another prefill, a partial spec decode):
                  canon_end stays, so the request is frozen exactly as a budget-split one is today,
                  and only the hunk 5 receipt is registered.
     A request preempted later in the same pass (not in num_scheduled_tokens) is dropped.
  Registration happens at the end of the pass instead of mid-pass: no request in the same pass can
  hit those blocks, which is also what the Mamba manager's cached_blocks_this_step enforces.
GATE RADIANCE_CANON_MIXED: 1 = accept a chunk beside spec decodes (the reviewed composition);
  unset/0 = accept only a chunk that ran alone (the pre-patch guarantee, now enforced explicitly
  rather than by the budget). The deferral itself has no off switch: with budget > grid, caching
  on `num_new_tokens == grid` alone would be wrong.
Counters: kv_cache_manager._radiance_canon_stats {alone, spec, refused, dropped}, logged every 128.
Needs patch_sched_canonical_chunks.py and patch_canon_receipts.py applied first.
REVERT: drop the prelude line (and CHUNK back to 3544).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

PURE = Path(sysconfig.get_paths()["purelib"])
SCHED = PURE / "vllm/v1/core/sched/scheduler.py"
KVCM = PURE / "vllm/v1/core/kv_cache_manager.py"

# ---------------------------------------------------------------- scheduler.py
apply(SCHED,
      "_RADIANCE_CANONICAL_CHUNKS = _radiance_cc_os.environ.get(\"RADIANCE_CANONICAL_CHUNKS\", \"0\") == \"1\"\n",
      "_RADIANCE_CANONICAL_CHUNKS = _radiance_cc_os.environ.get(\"RADIANCE_CANONICAL_CHUNKS\", \"0\") == \"1\"\n"
      "# radiance (patch_canon_mixed.py): 1 = a grid chunk beside spec decodes is canonical.\n"
      "_RADIANCE_CANON_MIXED = _radiance_cc_os.environ.get(\"RADIANCE_CANON_MIXED\", \"0\") == \"1\"\n",
      "_RADIANCE_CANON_MIXED = ",
      "canon mixed: gate constant")

apply(SCHED,
      "            _rad_g = self.kv_cache_manager._radiance_grid\n",
      "            # radiance (patch_canon_mixed.py): with a budget above the grid a grid chunk can share\n"
      "            # its step, so whether it is canonical is settled once the step is built.\n"
      "            self.kv_cache_manager._radiance_defer = bool(\n"
      "                self.kv_cache_manager._radiance_grid\n"
      "                and self.max_num_scheduled_tokens > self.kv_cache_manager._radiance_grid)\n"
      "            self.kv_cache_manager._radiance_pending = []\n"
      "            self.kv_cache_manager._radiance_canon_stats = dict.fromkeys(\n"
      "                (\"alone\", \"spec\", \"refused\", \"dropped\"), 0)\n"
      "            _rad_g = self.kv_cache_manager._radiance_grid\n",
      "self.kv_cache_manager._radiance_defer = bool(",
      "canon mixed: defer flag")

apply(SCHED,
      "        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n"
      "        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens\n",
      "        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n"
      "        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens\n"
      "        if getattr(self.kv_cache_manager, \"_radiance_pending\", None):\n"
      "            # radiance (patch_canon_mixed.py): settle this step's deferred grid chunks before\n"
      "            # the copy / offload-store / output metadata below are built from it.\n"
      "            self.kv_cache_manager._radiance_settle_canon(\n"
      "                num_scheduled_tokens, scheduled_spec_decode_tokens, _RADIANCE_CANON_MIXED)\n",
      "self.kv_cache_manager._radiance_settle_canon(",
      "canon mixed: settle after the scheduling loops")

# ---------------------------------------------------------------- kv_cache_manager.py
apply(KVCM,
      "            if (total_computed_tokens == _rad_ce and num_new_tokens == _rad_grid\n"
      "                    and _rad_ce + _rad_grid <= request.num_prompt_tokens):\n"
      "                _rad_ce += _rad_grid\n",
      "            if (total_computed_tokens == _rad_ce and num_new_tokens == _rad_grid\n"
      "                    and _rad_ce + _rad_grid <= request.num_prompt_tokens):\n"
      "                if getattr(self, \"_radiance_defer\", False):\n"
      "                    # radiance (patch_canon_mixed.py): the step may share this chunk; schedule()\n"
      "                    # settles it once the step is built (_radiance_settle_canon).\n"
      "                    request._radiance_canon_pending = _rad_ce + _rad_grid\n"
      "                    self._radiance_pending.append(request)\n"
      "                else:\n"
      "                    _rad_ce += _rad_grid\n",
      "request._radiance_canon_pending = _rad_ce + _rad_grid",
      "canon mixed: defer the canon_end advance")

apply(KVCM,
      "        if _rad_grid:\n"
      "            _radiance_cache_receipts(self.coordinator, request, _rad_uncapped)\n",
      "        if _rad_grid and getattr(request, \"_radiance_canon_pending\", None) is None:\n"
      "            # (patch_canon_mixed.py: a deferred chunk's receipt is settled with it)\n"
      "            _radiance_cache_receipts(self.coordinator, request, _rad_uncapped)\n",
      "a deferred chunk's receipt is settled with it",
      "canon mixed: no receipt for a deferred chunk at allocation")

apply(KVCM,
      "    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:\n",
      "    def _radiance_settle_canon(self, num_scheduled_tokens, spec_tokens, accept_spec) -> None:\n"
      "        \"\"\"radiance (patch_canon_mixed.py): settle the grid chunks allocate_slots deferred.\n"
      "\n"
      "        Canonical only if nothing in the step can have changed the chunk's numerics: it ran\n"
      "        alone, or (accept_spec) every other request was a speculative decode of exactly\n"
      "        1 + drafts tokens. Otherwise canon_end stays and the request is frozen, as a\n"
      "        budget-split chunk is. Either way cache_blocks registers what canon_end allows plus\n"
      "        the hunk 5 receipt, exactly as allocate_slots would have.\n"
      "        \"\"\"\n"
      "        pend, self._radiance_pending = self._radiance_pending, []\n"
      "        st = self._radiance_canon_stats\n"
      "        grid = self._radiance_grid\n"
      "        for request in pend:\n"
      "            target = request._radiance_canon_pending\n"
      "            request._radiance_canon_pending = None\n"
      "            rid = request.request_id\n"
      "            if num_scheduled_tokens.get(rid) != grid:\n"
      "                st[\"dropped\"] += 1  # preempted later in the same pass\n"
      "                continue\n"
      "            others = [(r, n) for r, n in num_scheduled_tokens.items() if r != rid]\n"
      "            ok = not others or (accept_spec and all(\n"
      "                len(spec_tokens.get(r) or ()) > 0 and n == len(spec_tokens[r]) + 1\n"
      "                for r, n in others))\n"
      "            if ok and request._radiance_canon_end == target - grid:\n"
      "                request._radiance_canon_end = target\n"
      "                st[\"spec\" if others else \"alone\"] += 1\n"
      "            else:\n"
      "                st[\"refused\"] += 1\n"
      "            self.cache_blocks(request, target)\n"
      "            if sum(st.values()) % 128 == 0:\n"
      "                logger.info(\"radiance canon mixed: grid chunks %s\", st)\n"
      "\n"
      "    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:\n",
      "def _radiance_settle_canon(self, num_scheduled_tokens, spec_tokens, accept_spec)",
      "canon mixed: settle helper")
