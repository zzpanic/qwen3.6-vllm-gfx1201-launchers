#!/usr/bin/env python3
"""DFlash2 selector: give the draft proposal its own RNG stream (vLLM #54282 backport).

Source: magiccodingman/vllm-radiance PR #8 (open, 2026-09-14), patch_dflash_sampling_rng.py, which
adapts vLLM PR #54282 (commit fe755c88995ad468882517b6c4bdd60138d46a3a). Unchanged except for the
import path below. vllm-radiance publishes no licence; vLLM itself is Apache-2.0.

The selector walk's proposal draw and the target's replacement draw both keyed their Gumbel noise
on (seed, position). Sharing it conditions the replacement on the rejected proposal and biases
probabilistic rejection sampling -- measured upstream as up to 0.0195 absolute probability error per
position, 0.0012 after. Applies to DRAFT_SAMPLE=probabilistic at temperature > 0 (greedy unchanged).
Offsetting the local RNG index by 1 << 30 leaves model positions and cache addressing untouched.
Present unfixed in vllm-radiance:0.9.3 even after ggz14's dflash2 patches (checked 2026-10-06).
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

path = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
apply(path,
      "        position = tl.load(sample_pos_ptr + flat) - 1\n",
      "        # vLLM #54282: proposal and target replacement need independent noise.\n"
      "        # This local RNG index does not change model/cache positions.\n"
      "        position = tl.load(sample_pos_ptr + flat) - 1 + (1 << 30)\n",
      "position = tl.load(sample_pos_ptr + flat) - 1 + (1 << 30)",
      "dflash2: separate selector proposal RNG stream")
apply(path,
      "# Candidate ids key the noise, matching the target's own sampling.",
      "# Candidate ids key the proposal's independent sampling noise.",
      "# Candidate ids key the proposal's independent sampling noise.",
      "dflash2: describe independent proposal noise")
