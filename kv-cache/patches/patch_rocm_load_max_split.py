#!/usr/bin/env python3
# radiance house port 2026-10-06 from SlyBase/vllm-sly-radiance sly/patch_rocm_load_max_split.py (written for vLLM 0.29.0;
# the anchors are identical in 0.27.1). No licence published upstream; vLLM is Apache-2.0.
"""Enable vLLM's load-time `max_split_size_mb:20` allocator scope on ROCm too (vLLM 0.29.0, always on).

Why (INT4 G3, 2026-09-17): RedHatAI/Qwen3.8-27B-INT4 + DFlash2 got 313,116 KV tokens at k=7 but
389,054 at k=4 -- 2.2 GiB of "consumed" memory that was neither weights nor activations. A
`torch.cuda.memory` snapshot after `load_model` showed it as allocator fragmentation, not non-torch
memory: reserved 17.42 GiB, allocated 14.52 GiB, 2.90 GiB inactive split blocks that
`empty_cache()` cannot return.

Mechanism: every bf16 `embed_tokens` / `lm_head` table (248320 x 5120 = 2.37 GiB; target and the
DFlash drafter each create one) gets its own 2.37 GiB segment, which is freed again in
`process_weights_after_loading` (int8/int4 embed, int4 lm_head, the drafter shares the target
tables). The caching allocator keeps the freed segment and carves the following W4A16
`process_weights_after_loading` transients (`unpack_quantized_values_into_int32`, 340-680 MB) and
the long-lived packed int8 weights (`pack_int4_exllama_shuffle`, 40-85 MB) out of it. One live
packed weight is enough to pin the whole segment. Whether the last packs land in the drafter's freed
segment depends on the exact allocation sequence, and the k-sized speculator buffers shift it: at
k=4 the segment ends up empty and is released (reserved 15.06 GiB), at k=7 two 85 MB packs pin it.

vLLM already guards against exactly this: `Worker.load_model` runs inside
`_scoped_allocator_max_split(max_split_size_mb=20)`, so a request cannot split a cached block that
is more than 20 MB larger than itself and gets a fresh right-sized segment instead. The guard is
gated on `current_platform.is_cuda()`, which is False on ROCm, so it never ran here. The allocator
setting itself (`torch._C._accelerator_setAllocatorSettings`) is platform-neutral and works on the
HIP build (torch 2.14.0+rocm7.14). Scope is unchanged: load only, restored afterwards; runtime,
CUDA graphs and the KV cache allocation keep the stock allocator settings. The original value is
also looked up in PYTORCH_ALLOC_CONF / PYTORCH_HIP_ALLOC_CONF, the names ROCm reads.

Measured (image 0.2.3 + this change via bind mount, 262144 context, 2026-09-17): INT4 k=7 KV
313,116 -> 386,338 tokens, k=4 389,054 -> 397,377; MXFP4 prod k=7 375,820 -> 385,934
(details in sly/README.md).
"""

import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/worker/gpu_worker.py"
L = "gpu_worker: load-time max_split_size_mb scope on ROCm"

apply(F,
      "        if not current_platform.is_cuda():\n"
      "            yield\n"
      "            return\n"
      "\n"
      "        conf = os.environ.get(\"PYTORCH_CUDA_ALLOC_CONF\", \"\")\n",
      "        # radiance (sly/patch_rocm_load_max_split.py): is_cuda() is False on ROCm, which left the\n"
      "        # freed bf16 embed/lm_head segments open to long-lived packed W4A16 weights (up to 2.4 GiB\n"
      "        # stranded). The allocator setting is the same on the HIP build.\n"
      "        if not current_platform.is_cuda_alike():\n"
      "            yield\n"
      "            return\n"
      "\n"
      "        conf = \",\".join(\n"
      "            os.environ.get(k, \"\")\n"
      "            for k in (\"PYTORCH_ALLOC_CONF\", \"PYTORCH_CUDA_ALLOC_CONF\", \"PYTORCH_HIP_ALLOC_CONF\")\n"
      "        )\n",
      "sly/patch_rocm_load_max_split.py", L)
