#!/usr/bin/env python3
"""Int2 draft head: always defer quantisation to first use (ggz14 1d76c8269, 2026-09-18).

Source: codeberg.org/ggz14/radiance-vllm-mxfp4 commit 1d76c8269 (after the vendored 980f891),
applied here to the copy the prelude installs in site-packages. No licence published upstream.

_quantize_draft_head decided whether the DFlash2 drafter's lm_head had been shared in yet by
sniffing the parameter for all-zeros, which held only because torch.empty() happened to return
zeroed pages. A loader that dirties the caching allocator first (prefetch is a different loader)
makes the sniff say "populated" and the int2 head is built from garbage: acceptance 7.63 -> 1.01,
decode 320 -> 47 tok/s, with output quality unchanged -- invisible to every accuracy gate. Deferring
unconditionally takes the branch production already used.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

apply(Path(sysconfig.get_paths()["purelib"]) / "radiance_drafthead.py",
      '        return f"unsupported draft-head weight {tuple(w.shape)} {w.dtype}"\n    lp._radiance_topk_only = lp_attr == "candidate_logits_processor"\n    # A drafter whose checkpoint carries no lm_head (DFlash2) gets the target\'s tensor shared in\n    # AFTER load_weights returns, so at this point the parameter is still allocated-but-empty.\n    # Quantising that yields an all-zero head, and the failure is silent and total: the serve comes\n    # up, text stays coherent because the TARGET is fine, and only acceptance collapses to ~1.0 --\n    # which reads as a plausible accuracy verdict on the quantisation. Defer instead.\n    if _head_is_empty(rows, rsc):\n        lp._apply_head = types.MethodType(_apply_head_lazy, lp)\n        return "lm_head empty at load_weights (shared in later); quantising on first use"\n    return _quantize_head_now(lp, lm_head)\n\n\n# int2 buffers keyed by the bf16 weight they were derived from. DFlash2 shares ONE lm_head between\n',
      '        return f"unsupported draft-head weight {tuple(w.shape)} {w.dtype}"\n    lp._radiance_topk_only = lp_attr == "candidate_logits_processor"\n    # A drafter whose checkpoint carries no lm_head (DFlash2) gets the target\'s tensor shared in\n    # AFTER load_weights returns, so at this point the parameter is not yet the weight we want.\n    # Quantising it yields a garbage head, and the failure is silent and total: the serve comes up,\n    # text stays coherent because the TARGET is fine, and only acceptance collapses to ~1.0 --\n    # which reads as a plausible accuracy verdict on the quantisation.\n    #\n    # This deferred only when the parameter sniffed as all-zero, which was load-bearing and wrong:\n    # it held only because a fresh torch.empty() happened to hand back zeroed pages. Measured\n    # 2026-09-18 -- a loader that allocated and freed fp32 temporaries before the drafter loaded\n    # dirtied the caching allocator, the not-yet-shared parameter came back non-zero, the sniff\n    # said "populated", and acceptance fell 7.63 -> 1.01 (draft accept 94.7% -> 0.1%, decode\n    # 320 -> 47 tok/s) with GSM8K unmoved at 97.2%, i.e. invisible to every accuracy gate.\n    #\n    # So defer unconditionally. _apply_head_lazy takes the head as an ARGUMENT and is therefore\n    # guaranteed the populated tensor; it already falls back to the stock GEMM if that is somehow\n    # still empty. Costs one stock-GEMM call on the first draft. This is the branch every dflash\n    # serve already took anyway -- the sniff returned True in production -- so it narrows to the\n    # exercised path rather than adding a new one.\n    lp._apply_head = types.MethodType(_apply_head_lazy, lp)\n    return "deferred to first use (the weight a drafter scores against is shared in later)"\n\n\n# int2 buffers keyed by the bf16 weight they were derived from. DFlash2 shares ONE lm_head between\n',
      "So defer unconditionally. _apply_head_lazy takes the head as an ARGUMENT",
      "drafthead: defer int2 quantisation unconditionally (ggz14 1d76c8269)")
