#!/usr/bin/env python3
"""Stop the OffloadingConnector killing the engine on a mixed local+external prefix hit.

WHAT CRASHED (2026-09-06 21:32:31, first ever occurrence):

    File ".../kv_connector/v1/offloading/scheduler.py", line 912,
      in update_state_after_alloc
    AssertionError

    assert (num_locally_computed_tokens
            <= num_locally_computed_gpu_blocks * tokens_per_block)

`update_state_after_alloc` returns immediately when `num_external_tokens == 0`, so this
code had never executed on this box until the L3 fs tier started producing real read-back
hits (external prefix cache hit rate went 0% -> 3-19.7% the same afternoon). The crash is
the read-back path, not the store path.

WHAT THE ASSERTION MEANS: the block loop walks the request's blocks for one KV group and
stops at the first non-null block with no `block_hash` -- the boundary between the prefix
already resident on the GPU and the freshly allocated blocks the offload tier is about to
fill. The assertion is the sanity check that the locally computed tokens actually fit
inside that hashed prefix. It fires when a block *inside* the local prefix has no hash,
i.e. the block array and the token count disagree.

WHY THIS CONFIGURATION IS UNUSUAL. Two things put us outside what upstream exercises:

  * patch_kv_group_size.py rewrites the hybrid group layout (9 groups, size 8) instead of
    upstream's min-bucket 15. Groups have different `tokens_per_block`, and the assertion
    is evaluated per group.
  * `SchedulerOffloadConfig.from_spec` contains

        use_eagle = (speculative_config is not None and speculative_config.use_eagle())
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

    DFlash2 makes `use_eagle()` true but sets `is_eagle_group` on no group, so the blanket
    fallback marks ALL NINE groups as draft groups -- including the real full-attention and
    GDN groups. That is what the boot line "EAGLE/MTP draft attention groups [0..8]
    detected" is reporting, and it changes which chunks are storable
    (`is_store_reachable_swa_chunk` adds `int(is_eagle_group)` to the reachable tail).

THE FIX HERE IS DELIBERATELY THE BLUNT, SAFE ONE. Declining an external cache hit is
always correct -- it costs a prefill, it cannot corrupt KV. Every alternative considered
was unsafe or unreachable:

  * Skipping the load for the offending group leaves freshly allocated blocks holding
    garbage while the scheduler counts those tokens as computed => silent wrong output.
    This rig has been burned by a silent-wrong-output bug before; see the `cd /` comment in
    llama-swap-ggz14-27b.sh.
  * Loading the whole range from chunk 0 writes into refcounted blocks other requests
    share => corrupts THEIR KV.
  * Reporting a per-request load failure so the scheduler recomputes is the designed
    fallback, but OffloadingConnector does not implement
    `get_block_ids_with_load_errors()` (only nixl, mooncake, flexkv and lmcache do), so
    `kv_load_failure_policy` never fires for it. Confirmed in the 0.27.1 image.
  * `update_state_after_alloc` runs inside `Scheduler.schedule()`, after allocation, and
    has no way to decline; any exception it raises is fatal to EngineCore.

So: when a request already has locally computed tokens, do not offer it an external hit at
all. With `num_locally_computed_tokens == 0` the assertion reduces to `0 <= 0` and the
whole failure class is unreachable by construction. The case we actually built the disk
tier for -- a conversation whose prefix was evicted entirely, resuming cold -- has no local
computed tokens and still gets its hit.

COST: external hits are lost on requests that also hit the GPU prefix cache. That is a
throughput cost, not a correctness one, and it is measurable: watch "External prefix cache
hit rate" in the serve log against the 3-19.7% band recorded on 2026-09-06.

Knobs:
    RADIANCE_OFFLOAD_MIXED_HIT=1   restore upstream behaviour (crashes again; use only to
                                   reproduce the assertion on purpose)

The second hunk leaves the assertion in place but makes it talk before it dies, so if a
different group geometry trips it we get the numbers instead of a bare AssertionError.
The dump is one-shot per process.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"

# --- hunk 1: module-level flag ------------------------------------------------------
apply(
    TARGET,
    "class SchedulerOffloadConfig(NamedTuple):",
    """_RADIANCE_ALLOW_MIXED_HIT = os.environ.get("RADIANCE_OFFLOAD_MIXED_HIT", "0") == "1"
_RADIANCE_ASSERT_DUMPED = False


class SchedulerOffloadConfig(NamedTuple):""",
    "_RADIANCE_ALLOW_MIXED_HIT",
    "offload mixed-hit flag",
)

# `os` is imported by this module already in 0.27.1, but do not assume it.
_src = TARGET.read_text()
if "\nimport os\n" not in _src:
    _src = _src.replace("\nlogger = init_logger(__name__)", "\nimport os\n\nlogger = init_logger(__name__)", 1)
    TARGET.write_text(_src)
    print("  OK    offload: added missing 'import os'")

# --- hunk 2: decline external hits for requests with a local prefix hit --------------
apply(
    TARGET,
    """        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0""",
    """        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache or (
            num_computed_tokens > 0 and not _RADIANCE_ALLOW_MIXED_HIT
        ):
            # radiance: a mixed local+external hit trips the block/token boundary
            # assertion in update_state_after_alloc and kills EngineCore. Declining
            # the external hit is always safe; see patch_offload_mixed_hit.py.
            num_hit_tokens = 0""",
    "radiance: a mixed local+external hit",
    "offload decline mixed hit",
)

# --- hunk 3: make the assertion report before it dies -------------------------------
apply(
    TARGET,
    """            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * tokens_per_block
            )""",
    """            if (
                num_locally_computed_tokens
                > num_locally_computed_gpu_blocks * tokens_per_block
            ):
                global _RADIANCE_ASSERT_DUMPED
                if not _RADIANCE_ASSERT_DUMPED:
                    _RADIANCE_ASSERT_DUMPED = True
                    logger.error(
                        "[radiance] offload boundary assertion: req=%s group=%d "
                        "tokens_per_block=%d tokens_per_chunk=%d swa_chunks=%s "
                        "align_chunks=%s local_tokens=%d external_tokens=%d "
                        "num_gpu_blocks=%d boundary=%d n_blocks=%d "
                        "blocks(is_null,has_hash)=%s",
                        request.request_id,
                        group_config.group_idx,
                        tokens_per_block,
                        tokens_per_chunk,
                        group_config.sliding_window_size_in_chunks,
                        group_config.alignment_chunk_count,
                        num_locally_computed_tokens,
                        num_external_tokens,
                        num_gpu_blocks,
                        num_locally_computed_gpu_blocks,
                        len(group_blocks),
                        [
                            (b.is_null, b.block_hash is not None)
                            for b in group_blocks[: min(num_gpu_blocks, 24)]
                        ],
                    )
                raise AssertionError(
                    "offload boundary: local_tokens=%d > boundary=%d * tpb=%d "
                    "(group %d)"
                    % (
                        num_locally_computed_tokens,
                        num_locally_computed_gpu_blocks,
                        tokens_per_block,
                        group_config.group_idx,
                    )
                )""",
    "[radiance] offload boundary assertion",
    "offload boundary diagnostics",
)
