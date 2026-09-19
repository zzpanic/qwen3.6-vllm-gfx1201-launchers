#!/usr/bin/env python3
"""Stop the prompt's final prefill chunk at its last full block boundary (align mode).

WHY
===
`Scheduler._mamba_block_aligned_split` (vllm/v1/core/sched/scheduler.py) keeps every prefill
chunk end block-aligned so Mamba state lands on block boundaries -- except the prompt's LAST
chunk, which it exempts. With Eagle-style drafting (DFlash here) it also stops one block early
(`last_cache_position -= block_size`). Together: when a prompt ends at most ~400 tokens past a
block boundary (the final span fits one 2,048-token step), the final chunk computes the last
full prompt block AND the partial tail in one pass. That block is complete, so the prefix cache
(GPU and the offload tiers) keeps it.

A cold prefill of any longer prompt computes the same block as its own 1,648-token chunk. The
GEMM tiles are 64 rows and 1,648 = 25 x 64 + 48, so in the cold chunk the block's last 48 rows
sit in the partial tail tile, and in the writer's longer chunk they sit in a full tile. Same
tokens, different accumulation -> the cached block's last 48 rows differ from cold in the low
bits, in every attention and drafter layer (fp8 KV turns some of those into whole-ulp flips).
A later turn that loads the block no longer reproduces a cold prefill.

Measured 2026-09-19 (turnbench replay ae4b61, statecmp + realcmp): the only differing loaded
bytes for A5 were block 30 (tokens 49,440-51,088), rows 1,600-1,647, all layers of g6/g7/g8,
written by A3 (prompt 51,252 = 51,088 + 164). The rule "a turn diverges iff it loads a block an
earlier turn computed inside a final chunk" predicts 21/21 turns of run ae43f1 (prompt-only).

WHAT IT CHANGES
===============
Adds one stop to the split's mandatory stops: the prompt's last full block boundary
(prefill_end // block_size * block_size). The final span is then two steps -- the last full
block as its own aligned chunk, then the partial tail -- so every cacheable block is computed
exactly as a cold prefill computes it. Cost: one extra forward step for prompts whose final
span would have crossed a block boundary (tail <= ~400 tokens past it, ~25% of prompts); the
extra step is the <=400-token tail.

GATE (house convention: UNSET = upstream behaviour)
  RADIANCE_ALIGN_PROMPT_LAST_BLOCK  unset/0 = upstream, 1 = stop at the last full block.
  The launcher defaults it to 1 (KVOFF_ALIGN_LAST_BLOCK).

RISK: scheduler thread; adds a stop only when it lies strictly inside the chunk, like the
upstream stops next to it. Chunk ends it adds are block-aligned, which is the invariant the
function exists to keep (slot p = state after (p + 1) * block_size tokens).

REVERT: unset the gate, or drop the prelude line and reload.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
SCHED = SP / "vllm/v1/core/sched/scheduler.py"

print("radiance: scheduler -- stop the prompt's final chunk at its last full block")

apply(
    SCHED,
    "import itertools\nimport time\n",
    "import itertools\nimport os as _radiance_os\nimport time\n"
    "# radiance: see patch_sched_align_last_block.py. Unset/0 = upstream behaviour.\n"
    "_RADIANCE_ALIGN_LAST_BLOCK = (\n"
    '    _radiance_os.environ.get("RADIANCE_ALIGN_PROMPT_LAST_BLOCK", "0") == "1"\n'
    ")\n",
    "_RADIANCE_ALIGN_LAST_BLOCK = (",
    "1  scheduler.py: gate constant",
)

apply(
    SCHED,
    """            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
""",
    """            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # radiance: the prompt's last full block boundary. Upstream exempts the
            # final chunk from alignment, so it can compute the last full block with the
            # partial tail; that block's tail rows then differ from a cold prefill's.
            prefill_end // block_size * block_size if _RADIANCE_ALIGN_LAST_BLOCK else 0,
""",
    "prefill_end // block_size * block_size if _RADIANCE_ALIGN_LAST_BLOCK",
    "2  scheduler.py: last-full-block stop",
)
