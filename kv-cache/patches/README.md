# The house patches

These are **not standalone scripts.** Read this before you try to run one.

## Which build applies which

 The **default** build applies six: `patch_offload_mixed_hit.py`, `patch_eagle_groups.py`,
 `patch_mamba_stride.py`, `patch_reconcile_reask.py`, `patch_swa_align_touch.py` and
 `patch_sched_align_last_block.py`. They are the ones that change what is served — a crash
 guard plus the behavioural fixes: the draft-group annotation this model needs, the
 recurrent-state store stride that makes the RAM tier hold enough, reconcile re-ask (stop
 recomputing prefixes the tier holds), the swa-align/touch-all eviction fix, and the
 last-block align (cached == cold). Everything else here is instrumentation or fs-tier-only
 and is applied only by the **disk** build (`KVCACHE_DISK_TIER=1`).
 `APPLY-ORDER.txt` marks the six.

## What they are

Anchored string surgery against an *installed* vLLM tree. Each one opens a file
under `/opt/vllm/lib/python3.12/site-packages/vllm/...`, finds a literal anchor
string, and replaces it. There is no `.patch` file, no `git apply`, no fuzz
factor: if the anchor is not found byte-for-byte, the script raises.

That makes them precise and makes them brittle in exactly the same way. They are
written against **vLLM 0.27.1 + radiance 0.9.3**, the versions pinned by the
image in `launcher/`. Against any other tree, expect them to fail loudly — which
is the intended failure mode, not a bug.

## They need `_patchlib`

Every `patch_*.py` begins:

```python
from _patchlib import apply
```

`_patchlib` is **not in this repository.** It comes from the ggz14 /
`radiance-vllm-mxfp4` repo, which the launcher bind-mounts at `/patches`. That is
why every invocation in `launcher/serve-mxfp4-kvcache-base.sh` looks like:

```
PYTHONPATH=/patches python3 /house/patch_offload_mixed_hit.py
```

`PYTHONPATH=/patches` is what makes the import resolve — Python puts the
*script's* directory on `sys.path[0]`, not the working directory, and these
scripts live in `/house`.

If you want to run one outside the launcher you need three things: the target
tree on disk, `_patchlib` importable, and `SP` pointing at the site-packages
root the script expects. The launcher does all three for you; nothing else does.

## Apply order is a dependency

 See `APPLY-ORDER.txt`. Several patches are genuinely ordered:
 
 - `promotion_wallclock` (12) re-anchors timing that `instrumentation` (2) installs, and its
   hunks reference names `tier_report` (11) creates, so it must run after 11.
 - `mamba_stride` (5) requires `eagle_groups` (4): while every group is flagged as
   an EAGLE/MTP draft group, `storable_chunks()` drops each group's trailing chunk
   during decode and the store grid stops lining up with the hit window. It also
   anchors on `offload_mixed_hit` (1)'s `_RADIANCE_ASSERT_DUMPED` line.
 
 The 1-anchors are trivially satisfied — 1 is FATAL and runs first.
 
 `lookup_invalidate` (13) and `fs_failed_load` (14) stand on `fs_fanout` (10)'s
 `v1/kv_offload/tiering/fs/manager.py` anchors, and 14 also stands on 13. `fs_fanout` (10)
 itself is order-independent with the rest. Three patches touch that file — `instrumentation`
 (2), `fs_fanout` (10) and `tier_report` (11) — and the 2/10/11 trio is order-independent
 because they anchor at different sites in it, not because any one is the sole editor.

## Two of them are fatal; the rest only warn

 The container block runs under `set -e`. Patches 1 and 2 have **no `|| echo`
 fallback**, so a failure there is a hard boot failure, not a warning. The default
 build applies only 1 of the two (2 is disk-build only):
 
 | # | Patch | On failure |
 |---|---|---|
 | 1 | `patch_offload_mixed_hit.py` | **FATAL** — engine does not boot |
 | 2 | `patch_offload_instrumentation.py` | **FATAL** — engine does not boot (disk build only) |
 | 3 | `patch_offload_lookup_metrics.py` | warns; Phase A metrics absent |
 | 4 | `patch_eagle_groups.py` | warns; all nine KV groups treated as draft groups |
 | 5 | `patch_mamba_stride.py` | warns; every chunk stores all six Mamba groups |
 | 6 | `patch_reconcile_reask.py` | warns; a GPU attention hit with no Mamba state recomputes the whole prompt |
 | 7 | `patch_swa_align_touch.py` | warns; every drafter chunk stored, stock eviction touch |
 | 8 | `patch_sched_align_last_block.py` | warns; cached turns can differ from cold in the low bits |
 | 9 | `patch_offload_debug_instrument.py` | warns; miss playback events absent |
 | 10 | `patch_offload_fs_fanout.py` | warns; one fs job per promotion |
 | 11 | `patch_offload_tier_report.py` | warns; no per-tier metrics, so `tools/tierreport.py` has no rows |
 | 12 | `patch_offload_promotion_wallclock.py` | warns; no promotion_* counters, timing stays per-batch |
 | 13 | `patch_lookup_invalidate.py` | warns; a just-stored block can read as absent (long-context hits drop) |
 | 14 | `patch_fs_failed_load.py` | warns; a reaped/unreadable disk block can hang the request that needs it |
 | 15 | `patch_offload_miss_deferral_metrics.py` | warns; a deferral that never resolves stays invisible |
 
 That split is deliberate. 1 and 2 are load-bearing — without patch 1 the engine
asserts and dies the first time an external hit lands on a request that also hit
the GPU prefix cache, and without the instrumentation an allocation failure is
unattributable. The rest degrade to defined, previously-shipped behaviour.

Patch 1 is also the one exception to "every behaviour change is gated" below. Its
gate, `RADIANCE_OFFLOAD_MIXED_HIT=0`, selects a conservative fallback (decline the
external hit) rather than upstream, because upstream's behaviour in this case is
the crash. Both halves of the fix — scoping the boundary assertion to
full-attention groups, and widening the lookup so a window group confirms the
chunks it will actually load — are unconditional when mixed hits are served.

Note what the warnings mean in practice: **a patch that fails silently leaves the
environment variables lying.** Setting a gate for a patch you have not applied reads
as "that behaviour is disabled" when in fact the code has never heard of the flag,
and the two are indistinguishable from the outside. If you are A/B-ing, confirm the
patch's metric series actually exists in `/metrics` before you believe a result;
`tools/kvvalidate.py` reports which series are present.

## Every behaviour change is gated

 No patch changes behaviour unconditionally. The gates (the unset state is the
 launcher's own default, which is the production value for the re-baselined ones):
 
 | Patch | Environment variable(s) |
 |---|---|
 | `offload_mixed_hit` | `RADIANCE_OFFLOAD_MIXED_HIT`, `RADIANCE_ALLOW_MIXED_HIT`, `RADIANCE_ASSERT_DUMPED` |
 | `promotion_wallclock` (counters) | none — counters only, always on |
 | `eagle_groups` | `RADIANCE_OFFLOAD_EAGLE_GROUPS` |
 | `mamba_stride` | `RADIANCE_MAMBA_STORE_STRIDE`, `RADIANCE_ASSERT_DUMPED` |
 | `reconcile_reask` | `RADIANCE_RECONCILE_REASK` |
 | `swa_align_touch` | `RADIANCE_SWA_STORE_MAMBA_ALIGN`, `RADIANCE_TOUCH_ALL_GROUPS`, `RADIANCE_TOUCH_POSITION_ORDER` |
 | `sched_align_last_block` | `RADIANCE_ALIGN_PROMPT_LAST_BLOCK` |
 | `fs_failed_load` | `RADIANCE_FS_FAILED_LOAD_FORGET` |
 | `lookup_invalidate` | `RADIANCE_LOOKUP_INVALIDATE` |
 | `fs_fanout` | `RADIANCE_FS_FANOUT_MAX`, `RADIANCE_FS_FANOUT_TARGET_MB` |
 | `instrumentation`, `lookup_metrics`, `debug_instrument`, `tier_report`, `miss_deferral_metrics` | none — metrics only, no behaviour change |
 
 Patching is therefore reversible without rebuilding: unset the gate and you get
 upstream behaviour with the patch still in place.

## Before you use any of these

Check each against current vLLM/radiance HEAD and delete it in favour of the
upstream implementation wherever one now exists. Two already measure as directly
applicable to this tree — PR #54327 (`tiering/fs/manager.py`, 100%) would retire
the external reaper entirely, and PR #54743 supplies the filtered-group primitive
patch 4 reinvents by hand.

A house patch that duplicates merged upstream work is a liability, not an asset.
