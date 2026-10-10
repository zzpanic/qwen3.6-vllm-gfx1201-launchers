# House copy 2026-10-09 of SlyBase sly/patch_short_prefill.py (github.com/SlyBase/vllm-sly-radiance @b23079a), unmodified
# below this header. Plain vLLM bug: a 1-token prefill takes the GDN decode path. Anchor identical in our 0.27.1.
#!/usr/bin/env python3
"""Route misclassified 1-token prefills off the GDN decode path (vLLM 0.28.0+).

Originally written for vllm5 (vLLM 0.28.0, `patch_short_prefill_028.py` in the
homelab repo's Ansible files/, applied as a runtime bind-mount patch). Re-verified
here for vLLM 0.29.0 (Subagent E) -- anchor matches verbatim, no re-anchor needed --
and moved into the fork's sly/ tree since vllm7 has no runtime-patch bind-mounts,
everything ships baked into the image.

FIX A2 (prefill replayed through decode-shaped FULL cudagraph) is NOT needed:
gather_batch_req_state() computes has_prefill from
num_computed_prefill_tokens < prefill_len and passes it to
get_uniform_decode_token_count(), which returns None when has_prefill=True,
forcing the PIECEWISE path.

FIX A1 -- vllm/v1/attention/backends/gdn_attn.py:
split_decodes_and_prefills() with the default treat_short_extends_as_decodes=True
takes an early return: if max_query_len <= decode_threshold the whole batch is
declared "all decodes" without consulting common_attn_metadata.is_prefilling.
A fresh request whose prompt is exactly one token lands in num_decodes, and the
GDN layer runs its recurrent single-step decode path against uninitialised
state -- corrupted logits, nondeterministic at temperature=0.

The sibling Mamba backend already passes treat_short_extends_as_decodes=False at
the same call site; GDN was the exception. With False, is_prefill |= is_prefilling
routes the 1-token prefill to the chunked prefill path. Genuine 1-token decodes
are unaffected (is_prefilling is False for them).

Plain vLLM bug, not gfx1201-specific and not quantization-specific. Applies to
any GDN/hybrid model including MXFP4-quantised checkpoints.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/backends/gdn_attn.py"

apply(F,
       '                )\n\n        if spec_sequence_masks is None:\n            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n                split_decodes_and_prefills(m, decode_threshold=1)\n            )\n            num_spec_decode_tokens = 0\n            spec_token_indx = None',
       '                )\n\n        if spec_sequence_masks is None:\n            # --- RADIANCE FIX A1 (prefill-of-length-1 misclassified as decode) ---\n            # split_decodes_and_prefills() default treat_short_extends_as_decodes=True\n            # never consults is_prefilling on the early return path. A fresh\n            # 1-token prompt lands in num_decodes -> GDN recurrent decode against\n            # uninitialised state -> corrupted logits. The Mamba backend already\n            # passes False here; this aligns GDN.\n            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n                split_decodes_and_prefills(\n                    m,\n                    decode_threshold=1,\n                    treat_short_extends_as_decodes=False,\n                )\n            )\n            num_spec_decode_tokens = 0\n            spec_token_indx = None',
       '            # --- RADIANCE FIX A1 (prefill-of-length-1 misclassified as decode) ---',
       'A1: 1-token prefill -> prefill path')
