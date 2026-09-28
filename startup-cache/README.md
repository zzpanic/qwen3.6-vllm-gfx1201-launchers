# startup-cache

Build caches for the container launchers, keyed so that they invalidate by themselves. Every cache
key is a fingerprint of everything its artifact was built from: source content, image **ID** (not
tag), GPU arch, and build flags. When any of those changes the key changes and the cache misses.
A miss is always safe: the slow path rebuilds.

| File | Side | What it does |
|---|---|---|
| `startup-cache.sh` | host, sourced | image ID + arch → `STARTUP_CACHE_KEY`; checks (and on a miss builds) the boot overlay; builds the container run args |
| `boot-overlay-meta.sh` | host, sourced | the one formula for the overlay's identity (shared by the check and the builder) |
| `build-boot-overlay.sh` | host | snapshots what the patch prelude changes in site-packages; run for you on a miss |
| `overlay-apply.sh` | container, sourced | copies the verified overlay in, or leaves the prelude to run |
| `hip-so-cache.sh` | container | compiles a `.hip` to a `.so` once per key |

## Wiring a launcher

**1. Host, once IMAGE is set, before anything is built or keyed** (required; it pulls the image if missing):

```bash
# Build caches: image-ID + arch keys, boot overlay (see startup-cache/README.md).
STARTUP_CACHE=${STARTUP_CACHE:-$(dirname "$(realpath -m "$0")")/startup-cache}
. "$STARTUP_CACHE/startup-cache.sh"
```

Put `$STARTUP_CACHE_KEY` into every cache directory or key the launcher owns. Add
`"${STARTUP_CACHE_RUN_ARGS[@]}"` to the `podman run` / `docker run` command.

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
aiter JIT). A stock `vllm serve` in a stock image gains correct keys but no speed. vLLM already
fingerprints its own compile cache, so for that cache the image-ID key is insurance against
reusing a stale graph, not a speedup.

## Limits

A key only covers inputs it can see. If a build step silently does nothing (for example, a patch
installer that skips itself), the key still says "configured" and not "happened". An artifact that
can be produced by a no-op needs a check of the artifact itself, or a documented `rm` of its cache
dir. The overlay builder's warning refusal is one such check.

Nothing is pruned automatically. Each new image leaves its old keyed directories behind, so remove
them by hand when you rotate images.
