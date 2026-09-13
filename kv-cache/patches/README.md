# The house patches

These are **not standalone scripts.** Read this before you try to run one.

## Which build applies which

The **default** build applies three: `patch_offload_mixed_hit.py`,
`patch_kv_offload_eagle_groups.py` and `patch_kv_offload_mamba_stride.py`. They are the
ones that change what is served — a crash guard plus a cross-prompt correctness fix, the
draft-group annotation this model needs, and the recurrent-state store stride that makes
the RAM tier hold enough. Everything else here is instrumentation or fs-tier-only and is
applied only by the **experimental** build (`KVCACHE_EXPERIMENTAL=1`). `APPLY-ORDER.txt`
marks the three.

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

See `APPLY-ORDER.txt`. Two pairs are genuinely ordered:

- `wallclock_reanchored` (in 8) re-anchors timing that `instrumentation` (2) installs, and on
  `offload_mixed_hit` (1)'s `_RADIANCE_ALLOW_MIXED_HIT` line.
- `mamba_stride` (5) requires `eagle_groups` (4): while every group is flagged as
  an EAGLE/MTP draft group, `storable_chunks()` drops each group's trailing chunk
  during decode and the store grid stops lining up with the hit window. It also
  anchors on `offload_mixed_hit` (1)'s `_RADIANCE_ASSERT_DUMPED` line.

Both 1-anchors are trivially satisfied — 1 is FATAL and runs first.

`fs_fanout` (6) is order-independent with the rest. Three patches touch
`v1/kv_offload/tiering/fs/manager.py` — `instrumentation` (2), `fs_fanout` (6), and
`tier_report` (7) — and they are order-independent because they anchor at different
sites in it, not because any one is the sole editor.

## Two of them are fatal; the rest only warn

The container block runs under `set -e`. Patches 1 and 2 have **no `|| echo`
fallback**, so a failure there is a hard boot failure, not a warning. The default
build applies only 1 of the two:

| # | Patch | On failure |
|---|---|---|
| 1 | `patch_offload_mixed_hit.py` | **FATAL** — engine does not boot |
| 2 | `patch_kv_offload_instrumentation.py` | **FATAL** — engine does not boot |
| 3 | `patch_kv_offload_lookup_outcomes.py` | warns; Phase A metrics absent |
| 4 | `patch_kv_offload_eagle_groups.py` | warns; all nine KV groups treated as draft groups |
| 5 | `patch_kv_offload_mamba_stride.py` | warns; every chunk stores all six Mamba groups |
| 6 | `patch_kv_offload_fs_fanout.py` | warns; one fs job per promotion |
| 7 | `patch_kv_offload_tier_report.py` | warns; no per-tier metrics, so `tools/tierreport.py` has no rows |
| 8 | `apply-bundle.py` | warns; no promotion_* counters, timing stays per-batch |
| 9–11 | lookup invalidation, deferral outcomes, miss reason | warns; each names what is missing |

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

No patch changes behaviour unconditionally. The gates:

| Patch | Environment variable(s) |
|---|---|
| `offload_mixed_hit` | `RADIANCE_OFFLOAD_MIXED_HIT`, `RADIANCE_ALLOW_MIXED_HIT`, `RADIANCE_ASSERT_DUMPED` |
| `promotion_refusal_instrumentation` | none -- counters only, always on |
| `eagle_groups` | `RADIANCE_OFFLOAD_EAGLE_GROUPS` |
| `mamba_stride` | `RADIANCE_MAMBA_STORE_STRIDE`, `RADIANCE_MAMBA_STRIDE`, `RADIANCE_ASSERT_DUMPED` |
| `fs_fanout` | `RADIANCE_FS_FANOUT_MAX`, `RADIANCE_FS_FANOUT_TARGET_MB` |
| `instrumentation`, `lookup_outcomes` | none — metrics only, no behaviour change |

Patching is therefore reversible without rebuilding: unset the gate and you get
upstream behaviour with the patch still in place.

## Before you use any of these

Check each against current vLLM/radiance HEAD and delete it in favour of the
upstream implementation wherever one now exists. Two already measure as directly
applicable to this tree — PR #54327 (`tiering/fs/manager.py`, 100%) would retire
the external reaper entirely, and PR #54743 supplies the filtered-group primitive
patch 4 reinvents by hand.

A house patch that duplicates merged upstream work is a liability, not an asset.
