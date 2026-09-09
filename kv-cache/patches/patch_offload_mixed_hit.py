#!/usr/bin/env python3
"""Make OffloadingConnector safe when a KV group's prefix hit lags the request's.

WHAT CRASHES (twice on this box: 2026-09-08 07:31:44 and 2026-09-10 07:57:10):

    File ".../kv_connector/v1/offloading/scheduler.py", in update_state_after_alloc
    AssertionError

    assert (num_locally_computed_tokens
            <= num_locally_computed_gpu_blocks * tokens_per_block)

`update_state_after_alloc` returns immediately when `num_external_tokens == 0`, so this
code had never executed on this box until the L3 fs tier started producing real read-back
hits (external prefix cache hit rate went 0% -> 3-19.7% on 2026-09-06). It is the read-back
path, not the store path. `update_state_after_alloc` runs inside `Scheduler.schedule()`,
after allocation, and has no way to decline: any exception it raises kills EngineCore.

Both dumps are the same shape (the second, in full):

    group=8 tokens_per_block=1648 tokens_per_chunk=1648 swa_chunks=2 align_chunks=None
    local_tokens=11536 external_tokens=1648 num_gpu_blocks=8 boundary=6 n_blocks=8
    blocks(is_null,has_hash)=[(T,F) x6, (F,F), (F,F)]

WHY IT HAPPENS -- and it is not a house-patch artefact. Chain, all of it stock 0.27.1:

  1. `Scheduler.schedule()` uses `get_computed_blocks_for_connector()` (not
     `get_computed_blocks()`) whenever a connector is attached and the model has mamba
     layers. That helper deliberately does NOT reconcile the per-group hits. It calls
     `find_longest_cache_hit_per_group()`, reports the FULL-ATTENTION group's hit as the
     request's local hit, and sets `hit_diverged = min(per_group_hits) < num_local`. Its
     own docstring says why: "the connector transfers the remaining suffix".
  2. So `num_locally_computed_tokens` is the full-attention hit (here 11536 = 7 blocks)
     while a lagging group can be resident for less. Group 8 is a SlidingWindowSpec group
     (swa_chunks=2) and is the MTP/DFlash2 draft group, so `SlidingWindowManager.
     find_longest_cache_hit` pops one more block for the eagle drop: its own hit is 6
     blocks (9888 tokens).
  3. `hit_diverged` is only reconciled away when the connector finds NO external tokens.
     Here it found 1648, so the diverged hit stands -- by design.
  4. `add_local_computed_blocks` pads group 8 with 6 nulls and adds nothing;
     `allocate_external_computed_blocks` then allocates `cdiv(13184,1648) - 6 = 2` fresh
     blocks at indices 6 and 7. That is the dumped block pattern exactly.
  5. The assertion says "every locally computed token is resident below the boundary".
     For a lagging window group that is false, which is the whole point of step 1.

So the assertion is a leftover invariant from before divergent lookups existed. It is
correct for full-attention groups -- their hit IS the boundary -- and wrong for every
group that is allowed to lag.

THE SECOND, WORSE DEFECT. Deleting the assertion alone is not the fix. `_lookup()` sets

    start_chunk_idx = num_computed_tokens // tokens_per_chunk

for every group and only ever confirms chunks at or above it, whereas
`update_state_after_alloc` loads from the group's own block boundary:

    start_chunk_idx = num_locally_computed_gpu_blocks // blocks_per_chunk

In the dumped crash the lookup confirmed chunks 7 and 8 for group 8 and the load would
have asked for chunks 6 and 7 -- chunk 6 never confirmed present. `prepare_load` is
documented "callers only pass keys already confirmed HIT by lookup() earlier this step"
and enforces it with `assert block is not None, f"Block {key!r} not found in cache"`.
That is a second EngineCore kill, hit whenever the tier has evicted the gap chunk. It is
NOT silent corruption -- checked, because this rig has been burned by silent wrong output
before (see the `cd /` comment in llama-swap-ggz14-27b.sh) -- but it is still a crash, and
it is the reason the fix has to touch the lookup and not just the assertion.

WHAT THIS PATCH DOES.

  hunk 3 (the real fix): for groups with a sliding window -- SlidingWindowSpec and mamba,
      which reports a window of 1 chunk -- start the suffix scan low enough that a full
      window ending at the top of the query range can be found:

          start_chunk_idx = min(start_chunk_idx, max(0, num_chunks - required_window))

      Nothing else changes: the run length (`required_window`) is untouched, so hit
      semantics are unchanged for any group that is not lagging, and the returned index is
      still converted to absolute chunks through the same `start_chunk_idx`. What changes
      is that the chunks the connector will actually load are now the chunks it confirmed.
      If the gap chunk is missing, the run resets and the request simply gets no external
      hit -- a lost hit, never an unbacked load.

      This is also why it is safe to let the load proceed: the destination blocks are
      freshly allocated by `allocate_external_computed_blocks` (refcount 1, owned by this
      request), so nothing shared is written, and the range is bounded by the assertion
      just below the one being relaxed:
      `num_pending_gpu_blocks <= sliding_window_size_in_chunks * blocks_per_chunk + 1`.
      A window group's `get_num_skipped_tokens` guarantees the pending range is at most
      its window in chunks, which is exactly what the scan now confirms.

  hunk 4: scope the boundary assertion to full-attention groups, where it is true, and
      keep the one-shot diagnostic dump so a genuinely new geometry still reports numbers
      instead of a bare AssertionError.

  hunk 2 (retained, now OFF by default): the original blunt remedy -- decline any external
      hit for a request that already has local computed tokens. Declining is always
      correct: it costs a prefill, it cannot corrupt KV. It is kept as a kill switch.

Knobs:
    RADIANCE_OFFLOAD_MIXED_HIT=0   fall back to declining every mixed local+external hit
                                   (loses external hits on any request that also hit the
                                   GPU prefix cache; watch "External prefix cache hit
                                   rate" in the serve log against the 3-19.7% band).
                                   Default is 1 = serve mixed hits, which is now safe.

UPSTREAM. Both defects are in stock vLLM 0.27.1 and neither is model-specific: they need a
connector, a mamba (or any lagging) group, and an external hit landing on a request that
also hit the GPU prefix cache. What this box supplies that upstream CI does not is traffic
that actually reaches the read-back path. `patch_kv_group_size.py` (9 groups) and
`patch_kv_offload_eagle_groups.py` (draft-group annotation) change WHICH group lags and how
often, not whether the invariant holds -- the eagle pop in `SlidingWindowManager` and the
divergent lookup in `get_computed_blocks_for_connector` are both stock. A stock
reproduction still has to be built before this can be filed; see kv-cache-known-issues.md.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
TARGET = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"

# --- hunk 1: module-level flags -----------------------------------------------------
apply(
    TARGET,
    "class SchedulerOffloadConfig(NamedTuple):",
    """_RADIANCE_ALLOW_MIXED_HIT = os.environ.get("RADIANCE_OFFLOAD_MIXED_HIT", "1") == "1"
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

# --- hunk 2: kill switch -- decline external hits for requests with a local prefix hit
apply(
    TARGET,
    """        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0""",
    """        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache or (
            num_computed_tokens > 0 and not _RADIANCE_ALLOW_MIXED_HIT
        ):
            # radiance kill switch (RADIANCE_OFFLOAD_MIXED_HIT=0): decline every
            # mixed local+external hit. Costs a prefill, cannot corrupt KV.
            # Off by default -- hunk 3 makes the mixed hit safe to serve.
            num_hit_tokens = 0""",
    "radiance kill switch (RADIANCE_OFFLOAD_MIXED_HIT=0)",
    "offload decline mixed hit (kill switch)",
)

# --- hunk 3: confirm the chunks a lagging window group will actually load ------------
apply(
    TARGET,
    """                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )""",
    """                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk

                required_window = sliding_window_size_in_chunks
                if required_window is not None:
                    if is_eagle_unverified:
                        required_window += 1
                    # radiance: num_computed_tokens is the FULL-ATTENTION hit
                    # (get_computed_blocks_for_connector reports a diverged hit
                    # when a connector is attached), so a window group can be
                    # resident for less than that and update_state_after_alloc
                    # will load chunks from BELOW start_chunk_idx. Scan low
                    # enough to confirm them; otherwise prepare_load is handed
                    # keys lookup() never confirmed. Run length is unchanged, so
                    # a non-lagging group behaves exactly as before.
                    start_chunk_idx = min(
                        start_chunk_idx, max(0, num_chunks - required_window)
                    )

                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if required_window is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )""",
    "radiance: num_computed_tokens is the FULL-ATTENTION hit",
    "offload window-group lookback",
)

# --- hunk 4: the boundary assertion holds for full attention only --------------------
apply(
    TARGET,
    """            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * tokens_per_block
            )""",
    """            # radiance: this invariant belongs to full-attention groups only.
            # They define num_locally_computed_tokens, so their resident prefix
            # always reaches it. A window group (sliding-window or mamba) is
            # allowed to lag -- see patch_offload_mixed_hit.py -- and the load
            # below fills the gap into freshly allocated blocks. The pending
            # range stays bounded by the sliding-window assertion that follows.
            if group_config.sliding_window_size_in_chunks is None and (
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
                    "(full-attention group %d)"
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
