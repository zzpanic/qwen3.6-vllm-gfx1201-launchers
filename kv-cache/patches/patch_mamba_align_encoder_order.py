#!/usr/bin/env python3
# radiance house backport 2026-10-21 of vllm-project/vllm#51603 (243c63baf523, merged 2026-08-10, Apache-2.0),
# vllm/v1/core/sched/scheduler.py only (the tests/ part of the PR does not apply to this image).
"""Apply Mamba alignment BEFORE the encoder caps, so a multi-image prompt cannot livelock.

Why (llama, 2026-10-21): two 4,096-token images in ONE request hung qwen3.8-27b-vllm forever -- GPU idle,
KV frozen at 3.18%, no error, only a client disconnect frees the slot; one image per request was fine.
Upstream issue vllm#40707 (Qwen3.5-35B-A3B) and #52551 (Qwen3.6-27B "Mode A"). The fix is in v0.28.0+,
not in the 0.27.1 this image ships; the image's scheduler.py is byte-identical to upstream v0.27.1.

Mechanism: the encoder cache holds max(max_num_batched_tokens, max item) = 4,096 tokens here, i.e. ONE
max-size image. Stock order is encoder caps -> `_mamba_block_aligned_split` -> skip on 0. With image 1
still resident, image 2 cannot be admitted, so the chunk is capped at image 2's start (~4,118); the
align split then floors that to the previous 880-token boundary (3,520 = the current position) -> 0
tokens, and every later step repeats it. Upstream's fix picks the aligned endpoint first and lets the
encoder cap shorten it: the request advances sub-block to image 2's start, image 1 is freed, and the
next step's `next_block_boundary` stop restores alignment.

Interaction with the house scheduler patches (none of their hunks touch these call sites):
  * text-only requests are unchanged -- `_try_schedule_encoder_inputs` returns at once without encoder
    inputs, so the align split runs once on the same count; the async-KV-load path still skips it.
  * canonical chunks: the shortened chunk is not a full grid chunk, so `_radiance_canon_end` stops there
    and the rest of that prompt is never cached -- cached == cold holds; the cost is caching for
    multi-image prompts past the first cut. canon wait already leaves encoder requests alone.

All-or-nothing: every anchor is checked in memory before any file is written, so a drifted base leaves
stock behaviour. Applied verbatim with offsets only against 0.27.1.
"""

import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

LIB = Path(sysconfig.get_paths()["purelib"])
L = "mamba align before encoder caps (vllm#51603)"
SCHED = "vllm/v1/core/sched/scheduler.py"
MOVED = "# (radiance backport vllm#51603: Mamba alignment moved above the encoder caps)\n"

HUNKS = [
    # 1/6 running requests: align first ...
    (SCHED,
     '            # Schedule encoder inputs.\n'
     '            encoder_inputs_to_schedule = None\n'
     ,
     '            # Apply Mamba alignment before encoder caps.\n'
     '            if self.need_mamba_block_aligned_split:\n'
     '                num_new_tokens = self._mamba_block_aligned_split(\n'
     '                    request, num_new_tokens\n'
     '                )\n'
     '\n'
     '            # Schedule encoder inputs.\n'
     '            encoder_inputs_to_schedule = None\n'
     ,
     '\n            # Apply Mamba alignment before encoder caps.\n'),
    # 2/6 ... and drop the old split after the encoder caps
    (SCHED,
     '                    shift_computed_tokens=1 if self.use_eagle else 0,\n'
     '                )\n'
     '\n'
     '            if self.need_mamba_block_aligned_split:\n'
     '                num_new_tokens = self._mamba_block_aligned_split(\n'
     '                    request, num_new_tokens\n'
     '                )\n'
     '\n'
     '            if num_new_tokens == 0:\n'
     ,
     '                    shift_computed_tokens=1 if self.use_eagle else 0,\n'
     '                )\n'
     '\n'
     '            ' + MOVED +
     '            if num_new_tokens == 0:\n'
     ,
     '\n            ' + MOVED),
    # 3/6 waiting requests: align first (inside the local-compute branch, so async KV loads still skip it) ...
    (SCHED,
     '                    num_new_tokens = min(num_new_tokens, token_budget)\n'
     '                    assert num_new_tokens > 0\n'
     '\n'
     '                    # Schedule encoder inputs.\n'
     ,
     '                    num_new_tokens = min(num_new_tokens, token_budget)\n'
     '                    assert num_new_tokens > 0\n'
     '\n'
     '                    # Apply Mamba alignment before encoder caps.\n'
     '                    if self.need_mamba_block_aligned_split:\n'
     '                        num_new_tokens = self._mamba_block_aligned_split(\n'
     '                            request,\n'
     '                            num_new_tokens,\n'
     '                            num_new_local_computed_tokens,\n'
     '                            num_external_computed_tokens,\n'
     '                        )\n'
     '                        if num_new_tokens == 0:\n'
     '                            break\n'
     '\n'
     '                    # Schedule encoder inputs.\n'
     ,
     '\n                    # Apply Mamba alignment before encoder caps.\n'),
    # 4/6 ... and drop the old split after them
    (SCHED,
     '                # Skip block alignment when setting up async receive (no local work).\n'
     '                if self.need_mamba_block_aligned_split and not load_kv_async:\n'
     '                    num_new_tokens = self._mamba_block_aligned_split(\n'
     '                        request,\n'
     '                        num_new_tokens,\n'
     '                        num_new_local_computed_tokens,\n'
     '                        num_external_computed_tokens,\n'
     '                    )\n'
     '                    if num_new_tokens == 0:\n'
     '                        break\n'
     '\n'
     ,
     '                ' + MOVED +
     '\n'
     ,
     '\n                ' + MOVED),
    # 5/6 _try_schedule_encoder_inputs: one encoder window, EAGLE shift included ...
    (SCHED,
     '        lo, hi = get_mm_features_in_window(\n'
     '            mm_features,\n'
     '            start=num_computed_tokens,\n'
     '            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,\n'
     '        )\n'
     ,
     '        encoder_window_end = (\n'
     '            num_computed_tokens + num_new_tokens + shift_computed_tokens\n'
     '        )\n'
     '        lo, hi = get_mm_features_in_window(\n'
     '            mm_features,\n'
     '            start=num_computed_tokens,\n'
     '            end=encoder_window_end,\n'
     '        )\n'
     ,
     '        encoder_window_end = (\n'),
    # 6/6 ... used for the embedding range as well
    (SCHED,
     '            end_idx_rel = min(\n'
     '                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos\n'
     '            )\n'
     ,
     '            end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)\n'
     ,
     'end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)'),
]


def precheck():
    """Simulate every hunk in memory; refuse to write anything unless all of them would apply."""
    state = {}
    for rel, anchor, new, sentinel in HUNKS:
        p = LIB / rel
        if not p.exists():
            raise SystemExit(f"  FAIL  {L}: {p} missing -- nothing applied")
        s = state.setdefault(rel, p.read_text())
        if sentinel in s:
            continue
        n = s.count(anchor)
        if n != 1:
            raise SystemExit(f"  FAIL  {L}: {rel} anchor matched {n}x -- nothing applied")
        state[rel] = s.replace(anchor, new, 1)


def main():
    precheck()
    for i, (rel, anchor, new, sentinel) in enumerate(HUNKS, 1):
        apply(LIB / rel, anchor, new, sentinel, f"{L} [{i}/{len(HUNKS)}]")


if __name__ == "__main__":
    main()
