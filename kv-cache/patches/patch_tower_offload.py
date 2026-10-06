#!/usr/bin/env python3
"""Backport (vLLM 0.29): make --cpu-offload-gb work under the V2 runner, and route towers through it.

Target: vllm/model_executor/models/interfaces.py, SupportsMultiModal._mark_tower_model.

vLLM 0.27.1 only offloads what `make_layers` wraps -- the decoder layer stack. A vision tower is
constructed directly, so `--cpu-offload-gb 1 --cpu-offload-params visual` offloads NOTHING on this
version (boot log: no "Total CPU offloaded parameters", model still 20.0 GiB). vLLM 0.29 fixed it
by wrapping each marked tower child at the end of `_mark_tower_model`, with a name prefix so the
parameter filter sees "visual.*" (mtstanfield/vllm-mxfp4 r9700-tp1 round 7 relies on that).

0.27.1's UVAOffloader.wrap_modules takes no prefix, and the parameter names inside a tower are
relative to it ("blocks.0.attn.qkv.weight"), so the filter can never match. The backport instead
wraps a tower child only when its OWN name is listed in cpu_offload_params, with the filter
cleared for that one call: the whole tower is offloaded, up to the --cpu-offload-gb budget.
Done at construction, as in 0.29, so the offloaded weights are loaded straight into pinned host
memory behind a UVA view and are never allocated on the device.

Inert unless --cpu-offload-gb > 0 AND --cpu-offload-params names a tower child (e.g. `visual`).
The decoder stack is unaffected: its parameters do not match `visual`.
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

F = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "model_executor" / "models" / "interfaces.py"

apply(F,
      "        self._tower_model_names = children_names\n",
      "        self._tower_model_names = children_names\n"
      "\n"
      "        # radiance backport of vLLM 0.29 (patch_tower_offload.py): a tower is built directly, so\n"
      "        # make_layers never routes it through the offloader. Wrap each tower child that\n"
      "        # --cpu-offload-params names, with the (relative-name) filter cleared for that call.\n"
      "        from vllm.model_executor.offloader import get_offloader as _rto_get\n"
      "        _rto = _rto_get()\n"
      "        _rto_want = getattr(_rto, \"cpu_offload_params\", None)\n"
      "        if getattr(_rto, \"cpu_offload_max_bytes\", 0) > 0 and _rto_want:\n"
      "            for _rto_name in children_names:\n"
      "                if _rto_name not in _rto_want:\n"
      "                    continue\n"
      "                _rto_saved = _rto.cpu_offload_params\n"
      "                _rto.cpu_offload_params = set()\n"
      "                try:\n"
      "                    _rto.wrap_modules(m for m in [getattr(self, _rto_name)])\n"
      "                finally:\n"
      "                    _rto.cpu_offload_params = _rto_saved\n"
      "            # the tower was built on the device and just moved to pinned host memory: hand the\n"
      "            # freed blocks back to the device now, or they sit in torch's cache where the HIP\n"
      "            # runtime and libr4d scratch (non-torch allocations) cannot use them\n"
      "            import torch as _rto_torch\n"
      "            _rto_torch.cuda.empty_cache()\n",
      "radiance backport of vLLM 0.29 (patch_tower_offload.py)", "interfaces: tower modules through the UVA offloader")

# The V2 model runner (selected automatically on this vLLM) never installed the weight offloader at
# all -- only the V1 runner calls set_offloader(create_offloader(...)) -- so get_offloader() stayed
# the no-op one and --cpu-offload-gb did nothing, for the decoder stack too. Install it the way V1
# does, before the model is constructed. create_offloader returns the no-op offloader when no
# offload flag is set, so this changes nothing without --cpu-offload-gb.
R = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
apply(R,
      "        self.eplb.prepare_load()\n"
      "        eplb_models_added = False\n"
      "        with DeviceMemoryProfiler() as m:\n",
      "        self.eplb.prepare_load()\n"
      "        eplb_models_added = False\n"
      "        # radiance backport (patch_tower_offload.py): install the weight offloader as V1 does.\n"
      "        from vllm.model_executor.offloader import create_offloader as _rto_create\n"
      "        from vllm.model_executor.offloader import set_offloader as _rto_set\n"
      "        _rto_set(_rto_create(self.vllm_config.offload_config))\n"
      "        with DeviceMemoryProfiler() as m:\n",
      "radiance backport (patch_tower_offload.py): install the weight offloader", "V2 runner: install the weight offloader")
