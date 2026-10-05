# startup-cache

Build caches for the container launchers, keyed so that they invalidate by themselves. Every cache
key is a fingerprint of everything its artifact was built from: source content, image **ID** (not
tag), GPU arch, and build flags. When any of those changes the key changes and the cache misses.
A miss is always safe: the slow path rebuilds.

| File | Side | What it does |
|---|---|---|
| `startup-cache.sh` | host, sourced | image ID + arch → `STARTUP_CACHE_KEY`; checks (and on a miss builds) the boot overlay; builds the container run args; persists the JIT caches that live under the container's HOME |
| `boot-overlay-meta.sh` | host, sourced | the one formula for the overlay's identity (shared by the check and the builder) |
| `build-boot-overlay.sh` | host | snapshots what the patch prelude changes in site-packages; run for you on a miss |
| `overlay-apply.sh` | container, sourced | copies the verified overlay in, or leaves the prelude to run |
| `hip-so-cache.sh` | container | compiles a `.hip` to a `.so` once per key |
| `cache-reap.sh` | host | removes stale sibling cache trees when the disk fills; run for you at launch |

## Wiring a launcher

**1. Host, once IMAGE is set, before anything is built or keyed** (required; it pulls the image if missing):

```bash
# Build caches: image-ID + arch keys, boot overlay (see startup-cache/README.md).
STARTUP_CACHE=${STARTUP_CACHE:-$(dirname "$(realpath -m "$0")")/startup-cache}
. "$STARTUP_CACHE/startup-cache.sh"
```

Put `$STARTUP_CACHE_KEY` into every cache directory or key the launcher owns. Add
`"${STARTUP_CACHE_RUN_ARGS[@]}"` to the `podman run` / `docker run` command.

Then, once the launcher's own cache dir is set (it should carry `$STARTUP_CACHE_KEY`), persist the
three JIT caches that live under the container's HOME instead of under any `*_CACHE_DIR`:

```bash
CACHE=${CACHE:-$HOME/.my-launcher-cache-$STARTUP_CACHE_KEY}
startup_cache_jit_mounts "$CACHE"                       # comgr, tvm-ffi, tilelang
startup_cache_reap "$CACHE" "$HOME/.my-launcher-cache-"  # stale sibling trees
```

Without the first line a `--rm` container rebuilds all three JIT caches on every boot; tvm-ffi's
torch DLPack addon alone is about 22 s. Set `STARTUP_CACHE_CONTAINER_HOME` if the image's HOME is
not `/root`. The second line keeps old trees from piling up (see "Pruning stale trees").

**2. Container, around the patch prelude** (optional, for the boot overlay):

```bash
. /startup-cache/overlay-apply.sh "$SP"
if [ -z "$BOOT_OVERLAY_APPLIED" ]; then
# BEGIN patch prelude
python3 patch_a.py
cp foo.py "$SP"/
# END patch prelude
fi
```

The builder extracts and hashes the lines between the markers. Only put site-packages patching
there: no `hipcc`, no `/cache`, no `exec`. The builder refuses anything else, and it also refuses to
publish if any patch prints `WARN`, `WARNING` or `FAIL`.

If the prelude branches on env vars, or reads patches from a second mounted dir, declare them
before the overlay check. They go into the overlay's key and are set the same way in the builder:

```bash
STARTUP_CACHE_OVERLAY_ENV="KVOFF_MINIMAL=$KVOFF_MINIMAL"   # exactly as the container gets them
STARTUP_CACHE_OVERLAY_MOUNTS="$HOUSE:/house"
```

Those knobs may only be resolved after the cache keys are needed. In that case, set
`STARTUP_CACHE_DEFER_OVERLAY=1` before sourcing, and call `startup_cache_overlay` yourself once
they are final. `kv-cache/launcher/serve-mxfp4-kvcache-base.sh` does this.

**3. Container, for a `.hip` compiled at boot** (optional):

```bash
bash /startup-cache/hip-so-cache.sh foo.hip "$SP"/foo.so -O3 -fPIC -shared $(python3 -m pybind11 --includes)
```

## What it saves

Measured 2026-10-05 on one R9700 (gfx1201): radiance 0.9.3, Qwen3.8-27B MXFP4 with DFlash2 ×7, a
16 GiB KV offload tier. Launch to first reply:

| Boot | `startup-qwen3.8-27b-kvcache.sh` |
|---|---|
| Cold: no caches, the overlay built on this boot, page cache dropped | 527 s |
| Cached, page cache dropped (what a host reboot looks like) | 196 s |
| Cached, page cache warm (a restart) | 161 s |

The gap between the last two rows is reading 20 GiB of weights from disk, which no cache here helps
with.

Where the cold boot's extra time goes, by phase. These come from the production launcher on the
same image and model (cold 470 s, cached 183 s), attributed from log timestamps, so treat them as
approximate:

| Cache | Saved per boot |
|---|---|
| vLLM's torch.compile / inductor / Triton caches under `/cache` | ~140 s |
| engine start: Triton kernels and comgr code objects | ~50 s |
| API-server start, including the tvm-ffi addon | ~30 s |
| `radiance_mxfp4_fp8.so` (`hip-so-cache.sh`) | ~30 s |
| aiter JIT (`module_aiter_core`) | ~30 s |
| boot overlay instead of the patch prelude | ~5 s |

The comgr / tvm-ffi / tilelang mounts on their own, measured as an A/B with every other cache warm:
189 s → 151 s, almost all of it the tvm-ffi addon no longer being rebuilt.

What a cached boot still spends is fixed cost: Python imports in vLLM's two processes (~50 s), the
image processor set up in both (~25 s, only for a model that takes images), the weights (~20–40 s)
and loading the compiled graphs (~15 s).

## Pruning stale trees

Every new image, card or flag combination gets a new keyed tree, and the old ones stay behind:
on the box this was written on, nine stale trees held 17 GiB. `startup_cache_reap <dir> <prefix>`
runs `cache-reap.sh` over the siblings `<prefix>*` before each boot. The launchers pass
`$HOME/.radiance-cache-w4a8-` or `./vllm-cache/`, so trees left by an old image are covered too.

The rule, in order:

1. Nothing happens below **65% disk use**. The trigger is space, not age.
2. The live tree is never touched, and neither is a tree with the same flag suffix.
3. A tree with a **stamp** (`.last-boot-ok`) from the last 24 h is kept. Every launch stamps its
   tree, and the kvcache launcher's container stamps it again once the server answers.
4. Otherwise a tree goes only if **no file in it was read in the last 24 h**. The stamp file is
   left out of that check: reading the stamp refreshes its own access time, and counting it once
   made every tree look in use, so the pruner removed nothing for weeks without saying so.
5. Whole trees, oldest first, at most 8 a run, stopping as soon as the disk is back under 65%.

Getting it wrong costs one slow boot (everything recompiles), never a wrong answer, so it is
deliberately simple. Its result is in the boot log as `[cache-reap]` lines. A custom `CACHE` or
`CACHE_DIR` outside those prefixes is never pruned, and neither are its neighbours.

To see what it would do: `startup-cache/cache-reap.sh --live <tree> --base <prefix>` (a dry run
unless you add `--apply`). Knobs: `CACHE_REAP_TARGET_PCT` (65), `CACHE_REAP_MIN_AGE_HOURS` (24),
`CACHE_REAP_MAX_DELETE` (8).

## Knobs

- `ARCH=<gfx...>` overrides the arch read from `/sys/class/kfd`.
- `BOOT_OVERLAY=<dir>|0` sets where this launcher's overlay lives. The default is
  `~/.cache/startup-cache/boot-overlay/<launcher name>`. `0` means never use one.
- There is nothing to rebuild by hand. On a miss (a changed patch, prelude, knob or image) the boot
  builds the overlay in a throwaway container (no GPU) and uses it straight away. That costs about
  the same as the prelude run it replaces. If the builder refuses (a patch warned), that exact input
  set is remembered in `<overlay>.refused` and is not retried until something changes.
  A dry run (`DRY_RUN`) reports and never builds.

## Scope: which launchers this works for

Any vLLM launcher that runs in a container (podman/docker) on AMD/ROCm. Nothing here names a model,
a patch or a knob, and every launcher wires it in with the same lines.

It is not fully generic yet:

1. **Containers only.** The identity is the image ID. A vLLM installed with pip or in a virtualenv
   has no image ID and would need a different fingerprint, such as a hash of the installed packages.
2. **AMD only.** The arch comes from `/sys/class/kfd`, which only exists with the amdgpu driver. On
   a host with more than one GPU family it prefers gfx12 (RDNA4). NVIDIA would need its own
   detection, for example the compute capability from `nvidia-smi`.
3. **One image layout.** site-packages defaults to `/opt/vllm/lib/python3.12/site-packages`, which
   is the radiance image. Set `STARTUP_CACHE_SP` for any other image.
4. **Overlay rule.** The boot overlay is only safe when the patches do not branch on env vars
   *while patching*. That was checked for `startup-qwen3.8-27b-mxfp4.sh`; check it for every new
   launcher (see `boot-overlay-meta.sh`).
5. **Linux tools.** It needs bash and GNU `realpath`.

How much time it saves depends on how much work the launcher does at boot (patching, `hipcc`,
aiter JIT). A stock `vllm serve` in a stock image gains correct keys and the JIT-cache mounts (in
the radiance image the tvm-ffi addon alone is ~22 s a boot), but nothing from the overlay or the
`.so` cache. vLLM already
fingerprints its own compile cache, so for that cache the image-ID key is insurance against
reusing a stale graph, not a speedup.

## Limits

A key only covers inputs it can see. If a build step silently does nothing (for example, a patch
installer that skips itself), the key still says "configured" and not "happened". An artifact that
can be produced by a no-op needs a check of the artifact itself, or a documented `rm` of its cache
dir. The overlay builder's warning refusal is one such check.

Only the launcher cache trees are pruned (see "Pruning stale trees"). The libr4d builds under
`~/.cache/radiance-libr4d/` are keyed per image too and are not, so clear old ones by hand when you
rotate images. Boot overlays are replaced in place and do not pile up.
