#!/usr/bin/env python3
"""DFlash2 fused context-KV for a drafter whose qkv_proj has NO usable raw `.weight`.

Two such drafters, one crash. (1) A compressed-tensors W4A16 drafter (syvai/Qwen3.8-27B-DFlash2-W4A16)
ships `weight_packed` + `weight_scale` and the layer has no `.weight` attribute at all
(AttributeError at qwen3_dflash.py _build_context_kv_buffers, 2026-10-09). (2) radiance_w4
(RADIANCE_FAST_DRAFT + RADIANCE_W4=1) packs the FP8 drafter to int4 at load and frees `layer.weight`
to an EMPTY tensor, which the dense-slice branch then reads (IndexError, boot loop, 2026-10-06).

Fix, in the style of SlyBase patch_dflash_w4_packed.py and on top of ggz14 patch_dflash_mxfp4_kv.py:
treat "absent or empty weight" like any other quantized weight. Defer the fused-KV build until the
quant method has processed the weights, then recover the dense K/V rows by pushing an identity
matrix through the layer's own forward (qkv_proj(eye)). The drafter's own forward keeps the 4-bit
kernel; only the small fused K/V matrix is dense.

Inert for the FP8 and bf16 drafters: both have a non-empty `.weight`, so both anchors' new conditions
reduce to the text they replace. Runs AFTER patch_dflash_mxfp4_kv.py (anchors are its output).
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/models/qwen3_dflash.py"

apply(F,
      '    weight = qkv_proj.weight\n'
      '    # radiance (patch_dflash_mxfp4_kv.py): the direct slice is valid only for a dense\n',
      '    weight = getattr(qkv_proj, "weight", None)  # house patch_dflash_deferred_quant.py\n'
      '    # radiance (patch_dflash_mxfp4_kv.py): the direct slice is valid only for a dense\n',
      'house patch_dflash_deferred_quant.py',
      "dflash deferred-quant: _dflash_kv_weight_rows reads .weight with getattr")

apply(F,
      '    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):\n'
      '        return weight[q_size:]\n'
      '    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)\n'
      '    eye = torch.eye(\n'
      '        qkv_proj.input_size_per_partition, dtype=dtype, device=weight.device\n'
      '    )\n',
      '    if (weight is not None and weight.numel() > 0\n'
      '            and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)):\n'
      '        return weight[q_size:]\n'
      '    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)\n'
      '    eye = torch.eye(\n'
      '        qkv_proj.input_size_per_partition, dtype=dtype,\n'
      '        device=next(qkv_proj.parameters()).device,\n'
      '    )\n',
      "next(qkv_proj.parameters()).device",
      "dflash deferred-quant: identity path for absent/empty weight")

apply(F,
      '        if layers_attn[0].qkv_proj.weight.dtype not in (\n'
      '            torch.bfloat16, torch.float16, torch.float32\n'
      '        ):\n',
      '        _w0 = getattr(layers_attn[0].qkv_proj, "weight", None)  # house patch_dflash_deferred_quant.py\n'
      '        if _w0 is None or _w0.numel() == 0 or _w0.dtype not in (\n'
      '            torch.bfloat16, torch.float16, torch.float32\n'
      '        ):\n',
      "_w0 = getattr(layers_attn[0].qkv_proj",
      "dflash deferred-quant: defer when the weight is absent or empty")
