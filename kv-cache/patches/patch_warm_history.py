#!/usr/bin/env python3
"""Boot log line: WARM HISTORY -- how many tokens of finished conversations each cache tier can keep ready to resume.

WHY (2026-10-11, the maintainer: "It is hard to sell to users that are used to defined context lengths" -> "Yes" to a defined
number at boot). vLLM reports GPU KV cache size and max context: space for RUNNING requests. Checkpoint retention
(patch_vram_retention.py / patch_tier_retention.py) changes neither; it changes how much FINISHED conversation stays
cached. This prints that as a defined figure per tier, next to the numbers users already know. INSTRUMENTATION ONLY:
one INFO line on the first schedule(), every step guarded; a failure logs nothing and changes nothing.

ARITHMETIC (all from the engine's own kv_cache_config; nothing model-specific)
  Per canonical grid chunk of S tokens (block b, g = S / b blocks per chunk), counted in pages (the hybrid pool's unit;
  every group's page is padded to one size; a pool block = page_size_bytes x the group's layers, and a tier row is
  one block per chunk block):
    attention   n_full * g                          always (attention is dense)
    state       n_mamba * 1 + n_window * min(g, W + 1 if EAGLE else W)    per checkpoint (W = window in blocks)
  Dense (retention off): one checkpoint per grid chunk. Retained: checkpoints only at kept points per conversation --
  VRAM 2 (resumable tip) + 1 junction; tiers RADIANCE_TIER_GEO_K (geometric backoff incl. the tip) + 1 junction --
  plus the system-prompt grid points (KEEP_PREFIX / S), shared by every conversation, counted once.
  Warm history = capacity pages / pages per token for conversations of RADIANCE_WARM_CONV_TOKENS (default 100,000).
  Capacity: VRAM = every GPU page (idle engine); RAM = RADIANCE_WARM_RAM_GIB; disk = RADIANCE_WARM_DISK_BYTES (the
  launcher passes volume size x the reaper target), both / page bytes.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SCHED = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/core/sched/scheduler.py"

apply(SCHED,
      "            self.kv_cache_manager._radiance_canon_stats = dict.fromkeys(\n"
      "                (\"alone\", \"spec\", \"refused\", \"dropped\"), 0)\n",
      "            self.kv_cache_manager._radiance_canon_stats = dict.fromkeys(\n"
      "                (\"alone\", \"spec\", \"refused\", \"dropped\"), 0)\n"
      "            try:  # radiance (patch_warm_history.py): instrumentation only\n"
      "                self._radiance_log_warm_history()\n"
      "            except Exception as _rad_e:  # never let a log line break scheduling\n"
      "                logger.debug(\"radiance warm history: %s\", _rad_e)\n",
      "self._radiance_log_warm_history()",
      "warm history: call on first schedule")

apply(SCHED,
      "    def _radiance_cold_grid_stop(self, start: int, block_size: int) -> int:\n",
      "    def _radiance_log_warm_history(self) -> None:\n"
      "        import os as _o\n"
      "        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec\n"
      "        cfg = self.kv_cache_config\n"
      "        b = self.cache_config.block_size\n"
      "        S = getattr(self.kv_cache_manager, \"_radiance_grid\", 0) or b\n"
      "        g = max(1, S // b)\n"
      "        n_full = n_mamba = 0\n"
      "        window_pages = 0\n"
      "        page = 0\n"
      "        for grp in cfg.kv_cache_groups:\n"
      "            sp = grp.kv_cache_spec\n"
      "            page = max(page, sp.page_size_bytes * max(1, len(grp.layer_names)))  # a block spans the group's layers\n"
      "            if isinstance(sp, MambaSpec):\n"
      "                n_mamba += 1\n"
      "            elif isinstance(sp, SlidingWindowSpec):\n"
      "                w = -(-sp.sliding_window // b)\n"
      "                window_pages += min(g, w + (1 if grp.is_eagle_group else 0))\n"
      "            elif isinstance(sp, FullAttentionSpec):\n"
      "                n_full += 1\n"
      "        attn = n_full * g                      # pages per grid chunk\n"
      "        ckpt = n_mamba + window_pages          # pages per checkpoint\n"
      "        L = int(_o.environ.get(\"RADIANCE_WARM_CONV_TOKENS\", \"100000\"))\n"
      "        chunks = max(1, L // S)\n"
      "        keep_prefix = int(_o.environ.get(\"RADIANCE_VRAM_KEEP_PREFIX\", \"14080\")) // S\n"
      "        def warm(cap_pages, retained, kept):\n"
      "            if cap_pages <= 0:\n"
      "                return 0\n"
      "            if not retained:\n"
      "                return int(cap_pages / (attn + ckpt) * S)\n"
      "            per_conv = chunks * attn + kept * ckpt\n"
      "            cap = cap_pages - keep_prefix * ckpt\n"
      "            return int(max(0, cap) / per_conv * chunks * S)\n"
      "        vram_ret = _o.environ.get(\"RADIANCE_VRAM_RETENTION\", \"0\") == \"1\"\n"
      "        tier_ret = _o.environ.get(\"RADIANCE_TIER_RETENTION\", \"0\") == \"1\"\n"
      "        geo_k = int(_o.environ.get(\"RADIANCE_TIER_GEO_K\", \"8\"))\n"
      "        gpu_pages = self.kv_cache_manager.block_pool.num_gpu_blocks\n"
      "        ram_pages = int(float(_o.environ.get(\"RADIANCE_WARM_RAM_GIB\", \"0\") or 0) * 2**30 / page) if page else 0\n"
      "        disk_pages = int(int(_o.environ.get(\"RADIANCE_WARM_DISK_BYTES\", \"0\") or 0) / page) if page else 0\n"
      "        def fmt(n):\n"
      "            return f\"{n / 1e6:.2f}M\" if n >= 1e6 else f\"{n / 1e3:.0f}k\"\n"
      "        parts = [f\"VRAM ~{fmt(warm(gpu_pages, vram_ret, 3))} (dense {fmt(warm(gpu_pages, False, 0))})\"]\n"
      "        if ram_pages:\n"
      "            parts.append(f\"RAM ~{fmt(warm(ram_pages, tier_ret, geo_k + 1))} \"\n"
      "                         f\"(dense {fmt(warm(ram_pages, False, 0))})\")\n"
      "        if disk_pages:\n"
      "            parts.append(f\"disk ~{fmt(warm(disk_pages, tier_ret, geo_k + 1))} \"\n"
      "                         f\"(dense {fmt(warm(disk_pages, False, 0))})\")\n"
      "        # practical minimum tier sizes (the author's bench notes (checkpoint-policy-20261010/notes.md, not shipped) s12):\n"
      "        # RAM stages a disk hit whole (disk -> RAM -> VRAM): one max-context resume = attention + 1 checkpoint\n"
      "        stage = (-(-self.max_model_len // S) * attn + ckpt) * page\n"
      "        ram_min = -(-stage * 1.15 // 2**30)\n"
      "        # disk: ~RADIANCE_WARM_AGENT_TOKENS of warm history per concurrent agent (350k: the knee on real agent traffic)\n"
      "        agent_tok = int(_o.environ.get(\"RADIANCE_WARM_AGENT_TOKENS\", \"350000\"))\n"
      "        kept_t = geo_k + 1 if tier_ret else chunks\n"
      "        bpt = (chunks * attn + kept_t * ckpt) * page / (chunks * S)\n"
      "        disk_agent = agent_tok * bpt / 1e9\n"
      "        logger.info(\"[radiance] warm history (tokens of finished conversations kept ready to resume, \"\n"
      "                    \"at %s-token conversations): %s. Context per request is unchanged: max_model_len %d. \"\n"
      "                    \"Recommended minimum: RAM tier %d GiB (stages one %d-token resume), disk tier ~%.0f GB \"\n"
      "                    \"per concurrent agent, at least %.0f GB. \"\n"
      "                    \"[grid %d, block %.1f MB, %d attention + %d checkpoint blocks per grid chunk, tier %.1f KB/token]\",\n"
      "                    fmt(L), \", \".join(parts), self.max_model_len, ram_min, self.max_model_len,\n"
      "                    disk_agent, 2 * disk_agent, S, page / 1e6, attn, ckpt, bpt / 1e3)\n"
      "\n"
      "    def _radiance_cold_grid_stop(self, start: int, block_size: int) -> int:\n",
      "def _radiance_log_warm_history(self) -> None:",
      "warm history: helper")
