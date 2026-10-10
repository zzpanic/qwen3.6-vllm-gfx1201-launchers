#!/usr/bin/env python3
"""House patch 2026-10-09: the DFlash2-W4A16 drafter parts of SlyBase sly/patch_w4a16_fuse.py
(github.com/brianfu9/vllm-sly-radiance @7d2304a), re-anchored for our vLLM 0.27.1 + radiance 0.9.3.

Needs patch_w4a16_tiles.py first (the kernels, custom ops and post-load transforms live in
rdna_hybrid_w4a16.py). This only wires the call sites, drafter only:

  1. gpu/model_runner.py (V2 runner): radiance_w4a16_postload(model, drafter) right after the drafter is
     loaded and registered, inside the load memory profile. RE-ANCHORED: Sly's anchor sits at a deeper
     indent and right before time_after_load; ours has the GDN merge and EMBED_HOST blocks there.
     On our MXFP4 target it touches nothing (no W4A16 layers); on the drafter it interleaves gate_up rows
     (fused silu) and re-quantizes the DFlash2 grouped-conv kernel_projection (bf16) to int4.
  2. qwen2.py Qwen2MLP (the drafter Qwen3MLP): gate_up GEMM + silu(gate) * up in one kernel. Verbatim.
  3. qwen3_dflash.py _project_context_kv: context K/V on the drafter own int4 rows instead of the 105 MB
     bf16 fused weight (which our patch_dflash_deferred_quant.py leaves to this lazy path). Verbatim.

LEFT OUT: the qwen2_moe / GDN hunks (W4A16 *target* only; ours is MXFP4) and the envs.py
compile_factors hunk (our launcher keys the compile cache by CACHE_SUF instead).
Knobs (Sly defaults): RADIANCE_W4A16_SILU=1, RADIANCE_DFLASH_CONV_W4=1 (changes proposals only: the
target verifies every token), RADIANCE_DFLASH_KV_W4=1 (exact).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
IMPORT = ("try:  # --- radiance (house patch_w4a16_fuse.py, from SlyBase): W4A16 fused call sites ---\n"
          "    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as _radiance_w4a16\n"
          "except Exception:  # never block model import on the fusion module\n"
          "    _radiance_w4a16 = None\n")

# --- 0. (house) our MPLinearKernel._get_weight_params returns (w_q, w_s, w_zp, w_gidx): Sly's helper
# unpacked three (vLLM 0.29), which failed every fusion at load. Only the fusion paths call it. ---
apply(SP / "vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py",
      "    w_q, w_s, w_zp = kern._get_weight_params(layer)\n",
      "    w_q, w_s, w_zp = kern._get_weight_params(layer)[:3]  # house: 0.27.1 also returns w_gidx\n",
      "house: 0.27.1 also returns w_gidx",
      "rdna_hybrid_w4a16: _radiance_w4_parts takes the 4-tuple of 0.27.1")

# --- 1. post-load hook in the V2 model runner (re-anchored) ---
ANCHOR = ("                eplb_models_added = self.eplb.maybe_register_speculator(\n"
          "                    self.speculator, self.speculative_config, load_dummy_weights\n"
          "                )\n")
apply(SP / "vllm/v1/worker/gpu/model_runner.py",
      ANCHOR,
      ANCHOR +
      "            # radiance (house patch_w4a16_fuse.py): W4A16 post-load fusions before profiling / capture\n"
      "            try:\n"
      "                from vllm.model_executor.kernels.linear.mixed_precision import (\n"
      "                    rdna_hybrid_w4a16 as _rw4,\n"
      "                )\n"
      "\n"
      "                _rw4.radiance_w4a16_postload(self.model, getattr(self.speculator, \"model\", None))\n"
      "            except Exception as _rw4_exc:  # a speed transform never fails the load\n"
      "                logger.warning(\"radiance w4a16 post-load skipped: %r\", _rw4_exc)\n",
      "_rw4.radiance_w4a16_postload(self.model",
      "gpu/model_runner: W4A16 post-load hook")

# --- 2. the drafter MLP (qwen3.py / qwen3_dflash.py: Qwen3MLP = Qwen2MLP) ---
F = SP / "vllm/model_executor/models/qwen2.py"
apply(F,
      "class Qwen2MLP(nn.Module):\n",
      IMPORT + "_RADIANCE_W4_SILU = _radiance_w4a16 is not None\n\n\nclass Qwen2MLP(nn.Module):\n",
      "_RADIANCE_W4_SILU = _radiance_w4a16 is not None",
      "qwen2: W4A16 fusion import")
apply(F,
      "    def forward(self, x):\n"
      "        gate_up, _ = self.gate_up_proj(x)\n"
      "        x = self.act_fn(gate_up)\n"
      "        x, _ = self.down_proj(x)\n"
      "        return x\n",
      "    def forward(self, x):\n"
      "        if getattr(self.gate_up_proj, \"_radiance_silu\", False):\n"
      "            # radiance (house patch_w4a16_fuse.py): gate_up GEMM + silu(gate) * up in one kernel\n"
      "            x, _ = self.down_proj(_radiance_w4a16.radiance_mlp_forward(self, x))\n"
      "            return x\n"
      "        gate_up, _ = self.gate_up_proj(x)\n"
      "        x = self.act_fn(gate_up)\n"
      "        x, _ = self.down_proj(x)\n"
      "        return x\n",
      "x, _ = self.down_proj(_radiance_w4a16.radiance_mlp_forward(self, x))",
      "qwen2: Qwen2MLP fused W4A16 silu")

# --- 3. DFlash context-KV on the drafter int4 rows ---
F = SP / "vllm/model_executor/models/qwen3_dflash.py"
apply(F,
      "_DFLASH_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)\n",
      IMPORT + "\n_DFLASH_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)\n",
      "rdna_hybrid_w4a16 as _radiance_w4a16",
      "qwen3_dflash: W4A16 fusion import")
apply(F,
      "        if self._fused_kv_weight is None:\n"
      "            self._fused_kv_weight = torch.cat(\n"
      "                [\n"
      "                    _dflash_kv_weight_rows(a.qkv_proj, a.q_size)\n"
      "                    for a in self._kv_source_attn\n"
      "                ],\n"
      "                dim=0,\n"
      "            )\n"
      "        all_kv_flat = F.linear(\n"
      "            normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
      "        )\n",
      "        all_kv_flat = None\n"
      "        if _radiance_w4a16 is not None:\n"
      "            # radiance (house patch_w4a16_fuse.py): the drafter own int4 K/V rows, no bf16 copy\n"
      "            all_kv_flat = _radiance_w4a16.radiance_dflash_kv_project(\n"
      "                self, normed_context_states\n"
      "            )\n"
      "        if all_kv_flat is None:\n"
      "            if self._fused_kv_weight is None:\n"
      "                self._fused_kv_weight = torch.cat(\n"
      "                    [\n"
      "                        _dflash_kv_weight_rows(a.qkv_proj, a.q_size)\n"
      "                        for a in self._kv_source_attn\n"
      "                    ],\n"
      "                    dim=0,\n"
      "                )\n"
      "            all_kv_flat = F.linear(\n"
      "                normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
      "            )\n",
      "all_kv_flat = _radiance_w4a16.radiance_dflash_kv_project(",
      "qwen3_dflash: int4 context-KV projection")
