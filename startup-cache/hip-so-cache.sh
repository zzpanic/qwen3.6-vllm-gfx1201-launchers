#!/bin/bash
# hip-so-cache.sh <src.hip> <out.so> [hipcc flags...] -- CONTAINER side: compile a HIP source
# to a shared object once, then reuse it on every later boot.
#
#   bash /startup-cache/hip-so-cache.sh foo.hip "$SP"/foo.so -O3 -fPIC -shared ...
#
# --offload-arch=$LAUNCH_GFX_ARCH is appended; LAUNCH_GFX_ARCH and LAUNCH_IMG_KEY come from
# startup-cache.sh through the run args. The key is everything the .so is built from:
#   the source, plus every #include "..." file that sits next to it (one level, by content);
#   the full flag list (including the arch); the image ID (hipcc, headers, pybind11).
# NOT covered: a quoted include that is not next to the source, or a nested include inside
# one -- if a source grows those, the key cannot see them; delete the cache dir after an edit.
# Cache: ${HIP_SO_CACHE:-/cache/hipso}/<name>-<key>/. Published by rename, so a boot killed
# mid-build or mid-copy never leaves a truncated .so under a key.
set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: $0 <src.hip> <out.so> [hipcc flags...]" >&2; exit 2; }
src=$1 out=$2; shift 2
: "${LAUNCH_GFX_ARCH:?not set -- add the startup-cache run args}" "${LAUNCH_IMG_KEY:?not set -- add the startup-cache run args}"
flags=("$@" --offload-arch="$LAUNCH_GFX_ARCH")

srcdir=$(dirname "$src")
src_sha=$( {
  sha256sum < "$src"
  { grep -oE '^[[:space:]]*#[[:space:]]*include[[:space:]]*"[^"]+"' "$src" || true; } | sed 's/.*"\(.*\)"/\1/' | sort -u |
    while read -r inc; do if [ -f "$srcdir/$inc" ]; then echo "$inc"; sha256sum < "$srcdir/$inc"; fi; done
} | sha256sum | cut -c1-16)
flag_sha=$(printf '%s\n' "${flags[@]}" | sha256sum | cut -c1-8)
name=$(basename "$src"); name=${name%.*}
key="$name-$src_sha-$flag_sha-$LAUNCH_GFX_ARCH-$LAUNCH_IMG_KEY"
dir="${HIP_SO_CACHE:-/cache/hipso}/$key"
so=$(basename "$out")

if [ -f "$dir/$so" ]; then
  cp "$dir/$so" "$out"
  echo "[startup-cache] $so from cache ($key)"
else
  hipcc "${flags[@]}" "$src" -o "$out"
  mkdir -p "$dir"
  cp "$out" "$dir/.$so.tmp"
  mv "$dir/.$so.tmp" "$dir/$so"
  echo "[startup-cache] $so built and cached ($key)"
fi
