#!/usr/bin/env python3
"""Lazy GDN: a prefill invalidates its stash, and stale-stash events are counted (needs libr4d rx14).

Why: lazy GDN (RADIANCE_GDN_LAZY=1) replays last step's candidates from a per-request stash block,
trusting a header whose magic and base_slot match. Physical blocks are recycled across requests
and turns, and nothing cleared the stash header when a request was (re)prefilled -- gdn_attn
already carries `radiance_stash_indices` "so a prefill can invalidate its stash", but nothing
consumed it. So the first decode step after a prefill could replay ANOTHER request's candidates
into this one's state: ggz14's controlled A/B (0cadf57) had 35 empty replies and a repeat loop in
50 turns, first failure at turn 5.

What: libr4d rx14 adds mode 2 to gdn_lazy_materialize (zero the stash header in every row-split
region of every head of every temporal state, through the kernel's own per-group block tables)
and per-dtype counters of stale stashes met where a replay was owed. This patch:
  1. radiance_gdn_lazy.py: invalidate() (mode 2, refused on a pre-rx14 libr4d, where mode 2 would
     fall into the checkpoint branch) and a counter poll in materialize() that logs a
     `stale-stash counters` line whenever the counts change.
  2. vllm mamba_hybrid.preprocess_state: invalidate the prefilling rows before the precopy, on
     steps that have any.
Knobs: RADIANCE_GDN_LAZY_INVALIDATE=0 skips the invalidation (control arm: reproduces the bug,
counters still count); RADIANCE_GDN_LAZY_STALE_EVERY=N polls every N materialize calls (default
512; 0 = never). Applied by the launcher only when RADIANCE_GDN_LAZY=1.
"""
import sysconfig
import sys
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

SPP = Path(sysconfig.get_paths()["purelib"])
LZ = SPP / "radiance_gdn_lazy.py"
MH = SPP / "vllm" / "v1" / "worker" / "gpu" / "model_states" / "mamba_hybrid.py"

# ---- 1a. the tables learn the counter reader (absent on a pre-rx14 libr4d) ----
apply(LZ,
      '        self.fn = getattr(r4d, f"gdn_lazy_materialize_k128_v128_bf16_{tag}")\n',
      '        self.fn = getattr(r4d, f"gdn_lazy_materialize_k128_v128_bf16_{tag}")\n'
      '        # rx14: stale-stash counters; their presence also marks a libr4d whose materialize\n'
      '        # kernel knows mode 2 (prefill invalidation). None on anything older.\n'
      '        self.stale = getattr(r4d, f"gdn_lazy_stale_counts_{tag}", None)\n',
      "self.stale = getattr(r4d", "radiance_gdn_lazy: counter reader")

# ---- 1b. invalidate() + the poll ----
apply(LZ,
      'def materialize(ctx, mode, num_reqs, a, b, c, idx_mapping):\n',
      '''INVALIDATE = os.environ.get("RADIANCE_GDN_LAZY_INVALIDATE", "1") == "1"
STALE_EVERY = int(os.environ.get("RADIANCE_GDN_LAZY_STALE_EVERY", "512"))
_poll = {"calls": 0, "last": None, "warned": False}


def _tables(ctx):
    tb = getattr(ctx, "_radiance_lazy_tables", None)
    if tb is None:
        tb = _Tables(ctx, ctx._radiance_kv_cfg, ctx._radiance_fwd_ctx)
        ctx._radiance_lazy_tables = tb
    return tb


def _stale_poll(tb, force=False):
    """Log the rx14 stale-stash counters when they change. Synchronous (hipMemcpyFromSymbol), so
    it runs every STALE_EVERY calls and never while a CUDA graph is being captured."""
    if tb.stale is None or STALE_EVERY <= 0 or torch.cuda.is_current_stream_capturing():
        return
    _poll["calls"] += 1
    if not force and _poll["calls"] % STALE_EVERY:
        return
    c = tuple(tb.stale(False))
    if c != _poll["last"]:
        _poll["last"] = c
        _log(f"stale-stash counters: update={c[0]} migrate={c[1]} checkpoint={c[2]} "
             f"| prefill invalidations={c[3]}")


def invalidate(ctx, num_reqs, state_idx, idx_mapping, mask):
    """rx14 mode 2: zero the stash headers of the batch rows whose int32 `mask` is non-zero (the
    prefilling rows). state_idx is the post-advance running column, as for mode 1."""
    tb = _tables(ctx)
    if tb.stale is None:
        if not _poll["warned"]:
            _poll["warned"] = True
            _log("WARNING: this libr4d has no prefill invalidation (needs rx14); stale stashes can "
                 "replay across requests -- do not serve multi-turn traffic with lazy on")
        return
    if INVALIDATE:
        im = idx_mapping.data_ptr() if idx_mapping is not None else 0
        tb.fn(tb.state_ptrs.data_ptr(), tb.slot_strides.data_ptr(), tb.group_idx.data_ptr(),
              tb.alog_ptrs.data_ptr(), tb.dtb_ptrs.data_ptr(), ctx.block_table_ptrs.data_ptr(),
              int(ctx.block_table_stride_req), im, 2, state_idx.data_ptr(), 0, 0,
              mask.data_ptr(), 0, int(ctx.block_size), int(num_reqs), tb.n, tb.H, tb.Hg,
              tb.K, tb.V, int(tb.st_head), tb.K ** -0.5, 20.0,
              torch.cuda.current_stream().cuda_stream)
    _stale_poll(tb, force=True)          # prefill steps are rare and slow: report promptly


def materialize(ctx, mode, num_reqs, a, b, c, idx_mapping):
''', "def invalidate(ctx, num_reqs, state_idx, idx_mapping, mask)", "radiance_gdn_lazy: invalidate + poll")

apply(LZ,
      '''          int(num_reqs), tb.n, tb.H, tb.Hg, tb.K, tb.V, int(tb.st_head), tb.K ** -0.5, 20.0,
          torch.cuda.current_stream().cuda_stream)
''',
      '''          int(num_reqs), tb.n, tb.H, tb.Hg, tb.K, tb.V, int(tb.st_head), tb.K ** -0.5, 20.0,
          torch.cuda.current_stream().cuda_stream)
    _stale_poll(tb)
''', "    _stale_poll(tb)\n", "radiance_gdn_lazy: poll after materialize")

# ---- 2. the hook: a prefill invalidates its stash before the precopy ----
apply(MH,
      '''            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        ctx.run_fused_precopy(
''',
      '''            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        import os as _rlz_os  # radiance lazy gdn (rx14): a prefill invalidates its stash
        if _rlz_os.environ.get("RADIANCE_GDN_LAZY", "0") == "1":
            _pf = input_batch.is_prefilling_np[:num_reqs]
            if _pf.any():
                import torch as _rlz_t
                import radiance_gdn_lazy as _rlz
                _rlz.invalidate(ctx, num_reqs, self._mamba_state_idx_gpu, input_batch.idx_mapping,
                                _rlz_t.from_numpy(_pf.astype("int32")).to(
                                    self._mamba_state_idx_gpu.device))
        ctx.run_fused_precopy(
''', "radiance lazy gdn (rx14): a prefill invalidates its stash", "mamba_hybrid: prefill invalidation hook")
