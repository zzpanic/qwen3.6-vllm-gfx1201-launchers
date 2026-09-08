#!/usr/bin/env python3
"""R3.13 -- store the Mamba/GDN groups every Nth chunk instead of every chunk.

THE PROBLEM THIS SOLVES

Phase B works: the serve-ready-prefix patch does serve external hits. But all of
its serves landed in a single 60-second window, and the reason is capacity, not
correctness. The CPU tier holds 636 slots = 115,360 tokens, which is 0.50x the
GPU cache (228,737 tokens). A conversation is evicted from the CPU tier before
the follow-up turn arrives, so the lookup falls through to the fs tier, and an
fs->CPU promotion measured 64.26 seconds. Nobody waits 64 seconds.

The CPU tier is too small because of what we write into it. Every 1,648-token
chunk writes all nine groups, each exactly 27,000,832 bytes -- 243,007,488 bytes
per chunk, 147,456 bytes per token. Only ~34,264 bytes per token is ever read
back, a 4.17x amplification.

WHERE THE WASTE IS -- and where it is NOT

It would be convenient if the six Mamba/GDN groups were mostly padding. They are
not: measured directly, g0-g5 are 97.0-99.0% dense, last nonzero byte at
26,976,256. That is real GDN state and none of it can be dropped.

The waste is temporal, not spatial. A Mamba group holds *one recurrent state*,
not a per-token history -- `get_sliding_window_size_in_chunks` already returns 1
for MambaSpec, and the load path already fetches exactly one Mamba chunk per
request (the last one). Yet the store path writes a fresh 27 MB snapshot of all
six Mamba groups at every single chunk boundary. On the GPU the same model keeps
just 2 Mamba pages per request (MambaSpec.max_memory_usage_bytes under "align" is
page_size_bytes * (2 + num_speculative_blocks)). The offload path keeps one per
chunk: ~70 snapshots for a long conversation where the GPU needs 2.

We are paying to cache 70 states so that we can read exactly one.

WHAT THIS CHANGES

Keep every Nth Mamba snapshot (N = RADIANCE_MAMBA_STORE_STRIDE, default 8), and
round the servable hit window down to the same boundary so we only ever ask for a
snapshot we actually kept. Two halves, and they must agree:

  STORE: in _build_store_jobs, a Mamba group's chunk is stored only when
         (absolute_chunk_index + 1) % N == 0.
  LOOKUP: resolve_mamba_align_size returns N * tokens_per_chunk instead of
         tokens_per_chunk, so _lookup's existing round_down() clamps
         max_hit_size_tokens to a multiple of N chunks.

WHY THE TWO HALVES ARE CONSISTENT (the part that makes this safe)

The load path in update_state_after_alloc fetches, for each group, chunks
[start_chunk_idx : cdiv(num_cached_tokens, tokens_per_chunk)]. For a Mamba group
the KV cache manager materialises only the final chunk's blocks (the rest are
null placeholders), which is why the existing assert holds:

    num_pending_gpu_blocks <= sliding_window_size_in_chunks * blocks_per_chunk + 1

So a Mamba group loads exactly one chunk, at index num_chunks - 1, where
num_chunks = num_cached_tokens / tokens_per_chunk. num_cached_tokens is
max_hit_size_tokens, which the LOOKUP half has rounded down to a multiple of
N * tokens_per_chunk. Therefore num_chunks is a multiple of N and the index
requested is num_chunks - 1 == N-1 (mod N) -- exactly the set the STORE half
keeps. Store and load address the same chunks by construction.

Absolute indices, not tail-relative ones, are what make this work. The existing
is_store_reachable_swa_chunk() computes reachability relative to the *current*
tail, so as a request grows each new chunk becomes the tail once and gets stored;
reusing it here would save nothing. `(abs_chunk_idx + 1) % N` is a fixed grid.

WHAT IT COSTS

A hit is truncated down to an N-chunk boundary: up to N * 1,648 = 13,184 tokens
of prefix that we hold but decline to serve, ~6,592 on average. On an 85k-token
conversation that is ~8% of the prefix recomputed. This is not an extra cost on
top of the attention groups -- the Mamba state is required at the boundary
regardless, so the whole hit was already limited by it.

WHAT IT BUYS

Bytes per 8 chunks, in units of 27,000,832: today 8 chunks x 9 groups = 72. With
N=8: 6 Mamba groups x 1 + 2 full-attention x 8 + 1 draft x 8 = 6 + 16 + 8 = 30.
That is 0.417x, i.e. 2.4x more tokens resident per byte. The CPU tier goes from
115,360 tokens (0.50x the GPU cache) to ~276,900 (1.21x). A conversation then
survives in the CPU tier to the next turn, and a CPU hit costs 1-2 s instead of
the 64.26 s fs->CPU promotion. That is the whole point: it is not a bandwidth
optimisation, it is what moves hits from the fs tier to the CPU tier.

PREREQUISITE. Apply patch_kv_offload_eagle_groups.py first. While all nine groups
are mislabelled as EAGLE draft groups, storable_chunks() drops each group's
trailing chunk during decode and _lookup queries-then-pops an extra chunk, so the
store grid and the lookup boundary no longer line up cleanly.

TUNING / REVERTING: RADIANCE_MAMBA_STORE_STRIDE=1 disables the patch completely
(both halves collapse to today's behaviour) without unpatching anything. Larger N
trades more truncated prefix for more CPU-tier residency.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _patchlib import apply  # noqa: E402

VLLM = Path(sys.prefix) / "lib" / f"python3.{sys.version_info.minor}" / "site-packages" / "vllm"
if not VLLM.exists():
    import vllm as _v
    VLLM = Path(_v.__file__).parent

SCHED = VLLM / "distributed" / "kv_transfer" / "kv_connector" / "v1" / "offloading" / "scheduler.py"

print("[radiance] R3.13 mamba store cadence")

# --- 1. the knob ------------------------------------------------------------------------
apply(
    SCHED,
    anchor="_RADIANCE_ASSERT_DUMPED = False",
    new='''_RADIANCE_ASSERT_DUMPED = False

# radiance R3.13: keep every Nth Mamba/GDN snapshot rather than one per chunk. A Mamba group
# holds a single recurrent state and the load path reads exactly one chunk of it, but the
# store path writes 27 MB per group per chunk. 1 disables the patch entirely.
_RADIANCE_MAMBA_STRIDE = max(1, int(os.environ.get("RADIANCE_MAMBA_STORE_STRIDE", "8")))''',
    sentinel="_RADIANCE_MAMBA_STRIDE",
    label="1 scheduler: RADIANCE_MAMBA_STORE_STRIDE knob",
)

# --- 2. mark which groups are Mamba -----------------------------------------------------
apply(
    SCHED,
    anchor="    is_eagle_group: bool = False",
    new='''    is_eagle_group: bool = False
    # radiance R3.13: True for MambaSpec groups. Distinct from
    # sliding_window_size_in_chunks == 1, which a genuinely small attention window
    # would also produce.
    is_mamba_group: bool = False''',
    sentinel="is_mamba_group: bool = False",
    label="2 scheduler: GroupOffloadConfig.is_mamba_group",
)

apply(
    SCHED,
    anchor="                    is_eagle_group=idx in eagle_groups,",
    new='''                    is_eagle_group=idx in eagle_groups,
                    is_mamba_group=isinstance(  # radiance R3.13
                        kv_cache_config.kv_cache_groups[idx].kv_cache_spec, MambaSpec
                    ),''',
    sentinel="is_mamba_group=isinstance(",
    label="3 scheduler: populate is_mamba_group in from_spec",
)

# --- 3. LOOKUP half: round the hit window down to the stride grid -----------------------
apply(
    SCHED,
    anchor='''            mamba_align_size = tokens_per_chunk
    return mamba_align_size''',
    new='''            mamba_align_size = tokens_per_chunk
    if mamba_align_size is not None and _RADIANCE_MAMBA_STRIDE > 1:
        # radiance R3.13: we only keep every Nth Mamba snapshot, so the hit window must land
        # on the same grid -- otherwise _lookup asks for a state we deliberately did not
        # store and _sliding_window_lookup walks backwards probing chunks that cannot exist.
        mamba_align_size *= _RADIANCE_MAMBA_STRIDE
    return mamba_align_size''',
    sentinel="radiance R3.13: we only keep every Nth Mamba snapshot",
    label="4 scheduler: resolve_mamba_align_size honours the stride",
)

# --- 4. STORE half: skip off-grid Mamba chunks ------------------------------------------
apply(
    SCHED,
    anchor="                    abs_chunk_idx = start_chunk_idx + key_idx",
    new='''                    abs_chunk_idx = start_chunk_idx + key_idx
                    # radiance R3.13: keep only every Nth Mamba/GDN snapshot. The grid is
                    # absolute, not relative to the current tail: a tail-relative rule would
                    # store every chunk as it briefly became the newest one.
                    if (
                        group_config.is_mamba_group
                        and _RADIANCE_MAMBA_STRIDE > 1
                        and (abs_chunk_idx + 1) % _RADIANCE_MAMBA_STRIDE != 0
                    ):
                        continue''',
    sentinel="radiance R3.13: keep only every Nth Mamba/GDN snapshot",
    label="5 scheduler: skip off-grid Mamba chunks in _build_store_jobs",
)

print(f"[radiance] R3.13 applied -- stride "
      f"{os.environ.get('RADIANCE_MAMBA_STORE_STRIDE', '8')} (1 = off)")
