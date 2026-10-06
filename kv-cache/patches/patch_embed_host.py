#!/usr/bin/env python3
"""Input embedding table in pinned host memory behind a UVA view (kernel sweep item E).

Idea from mtstanfield/vllm-mxfp4 r9700-tp1 (PQ_EMBED_UVA, 2026-09-29: fp8 table, pool 273k -> 376k
tokens with the vision tower, decode/prefill within noise). Here the table stays bf16, so every value
the GPU reads is the value it read before: greedy output must be bit-identical, which is the test.

Qwen3.8-27B: embed_tokens is 248,320 x 5,120 bf16 = 2.37 GiB, untied from lm_head. A lookup reads
only the rows of the tokens in the step (~10 KB each), over PCIe via UVA, inside CUDA graphs too.
The freed VRAM only becomes KV if KV_MEM is raised to match (it is pinned in the config).

WHERE: V2 runner load_model, AFTER the speculator is loaded, inside its DeviceMemoryProfiler. Not in
the embedding's process_weights_after_loading: a DFlash2 drafter builds its own embed_tokens
UNLOADED and dflash/utils.py swaps in the target's module only after load -- moving at
process-weights time pinned 2.37 GiB of that placeholder on the host. Here the sharing is done, so
each live table is moved exactly once (modules deduplicated by identity).

Acts only when EMBED_HOST=1; only on modules whose type is exactly VocabParallelEmbedding (never
ParallelLMHead, which reads its whole table every step); only at TP=1 and for tables >= 64M
elements; and refuses if any lm_head shares the table's storage (tied embeddings). Costs the
table's size in pinned host RAM.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/worker/gpu/model_runner.py"
apply(F,
      "                eplb_models_added = self.eplb.maybe_register_speculator(\n"
      "                    self.speculator, self.speculative_config, load_dummy_weights\n"
      "                )\n"
      "        time_after_load = time.perf_counter()\n",
      "                eplb_models_added = self.eplb.maybe_register_speculator(\n"
      "                    self.speculator, self.speculative_config, load_dummy_weights\n"
      "                )\n"
      "            # radiance house patch (patch_embed_host.py): input embedding to pinned host memory,\n"
      "            # after the drafter has been wired to the target table (one live table, moved once).\n"
      "            import os as _eh_os\n"
      "            if _eh_os.environ.get(\"EMBED_HOST\", \"0\") == \"1\":\n"
      "                import sys as _eh_sys\n"
      "                from vllm.model_executor.layers.vocab_parallel_embedding import (\n"
      "                    ParallelLMHead as _EhHead, VocabParallelEmbedding as _EhEmb)\n"
      "                from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor\n"
      "                _eh_roots = [self.model]\n"
      "                if getattr(self.speculator, \"model\", None) is not None:\n"
      "                    _eh_roots.append(self.speculator.model)\n"
      "                _eh_mods = {id(m): m for r in _eh_roots for m in r.modules()}.values()\n"
      "                _eh_heads = {m.weight.data_ptr() for m in _eh_mods\n"
      "                             if isinstance(m, _EhHead) and getattr(m, \"weight\", None) is not None}\n"
      "                for _m in _eh_mods:\n"
      "                    _w = getattr(_m, \"weight\", None)\n"
      "                    if (type(_m) is not _EhEmb or _w is None or not _w.is_cuda\n"
      "                            or getattr(_m, \"tp_size\", 1) != 1 or _w.numel() < (64 << 20)\n"
      "                            or getattr(_m, \"_radiance_embed_host\", None) is not None):\n"
      "                        continue\n"
      "                    if _w.data_ptr() in _eh_heads:\n"
      "                        _eh_sys.stderr.write(\"[radiance.embed] table is TIED to an lm_head: left on \"\n"
      "                                             \"the card (moving it would move the head)\\n\")\n"
      "                        continue\n"
      "                    _host = torch.empty(_w.shape, dtype=_w.dtype, device=\"cpu\", pin_memory=True)\n"
      "                    _host.copy_(_w.data)\n"
      "                    _w.data = get_accelerator_view_from_cpu_tensor(_host)\n"
      "                    _m._radiance_embed_host = _host          # keep the pinned storage alive\n"
      "                    _eh_sys.stderr.write(f\"[radiance.embed] input embedding {tuple(_host.shape)} \"\n"
      "                                         f\"{_host.dtype} -> pinned host + UVA view: \"\n"
      "                                         f\"{_host.numel() * _host.element_size() / 2**30:.2f} GiB \"\n"
      "                                         f\"VRAM freed\\n\")\n"
      "                torch.cuda.empty_cache()\n"
      "        time_after_load = time.perf_counter()\n",
      "radiance house patch (patch_embed_host.py)", "V2 runner: input embedding to pinned host (EMBED_HOST)")
