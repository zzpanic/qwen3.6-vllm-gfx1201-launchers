#!/bin/bash
# build-boot-overlay.sh <launcher> <patch-dir> -- pre-apply a launcher's patch prelude once and
# snapshot the site-packages files it changes, so the launcher can copy them in instead of
# re-running every patch script on every boot.
#
# Normally run for you: startup-cache.sh calls it on a boot whose overlay is missing or out of
# date, with the launcher's own IMAGE, patch dir, knobs and mounts.
# By hand:  IMAGE=... OUT=... ./build-boot-overlay.sh ../startup-x.sh <patch-dir>
#
# A speed device only. A stale overlay costs boot time, never correctness: the launcher
# recomputes the meta line (boot-overlay-meta.sh) every boot and, on any mismatch, ignores the
# overlay and runs the full prelude.
#
# How it works (one throwaway container, no GPU, no network):
#   1. the patch block is EXTRACTED from the launcher (between its BEGIN/END patch prelude
#      markers), so this can never drift from what the launcher would run;
#   2. inside the container: stamp, run the block, list every non-.pyc file in site-packages
#      newer than the stamp (patches rewrite files, cp copies them; neither keeps mtimes),
#      refuse if the prelude removed any file, copy the changed files out;
#   3. refuse if any patch printed a WARN/WARNING/FAIL (baked into an overlay, that message
#      would never print again), write meta.env and publish atomically: a failed build never
#      replaces a good overlay.
#
# Env: IMAGE (required) OUT (required) RUNTIME STARTUP_CACHE_SP (site-packages in the image,
#      default /opt/vllm/lib/python3.12/site-packages -- the prelude sees it as $SP),
#      STARTUP_CACHE_OVERLAY_ENV ("NAME=value ..." set in the container, as the launcher sets
#      them), STARTUP_CACHE_OVERLAY_MOUNTS ("host:/container ..." extra patch dirs).
set -euo pipefail

[ "$#" -eq 2 ] || { echo "usage: IMAGE=... OUT=... $0 <launcher> <patch-dir>" >&2; exit 2; }
LAUNCHER="$(realpath -m "$1")"
PATCHES_DIR="$(realpath -m "$2")"
: "${IMAGE:?IMAGE must be set}" "${OUT:?OUT must be set}"
SP_IN=${STARTUP_CACHE_SP:-/opt/vllm/lib/python3.12/site-packages}
RUNTIME=${RUNTIME:-}
if [ -z "$RUNTIME" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else echo "no container runtime found" >&2; exit 1; fi
fi
[ -d "$PATCHES_DIR" ] || { echo "FATAL: patch dir $PATCHES_DIR does not exist" >&2; exit 1; }
"$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: $IMAGE is not pulled" >&2; exit 1; }
# shellcheck source=boot-overlay-meta.sh
. "$(dirname "$(realpath -m "$0")")/boot-overlay-meta.sh"

# --- 1. extract the patch block from the launcher ---------------------------------------
PRELUDE=$(_boot_prelude "$LAUNCHER")
[ -n "$PRELUDE" ] || { echo "FATAL: no BEGIN/END patch prelude markers in $LAUNCHER" >&2; exit 1; }
# The prelude may only patch site-packages. Anything that compiles, reads the run-time mounts
# or execs the server is not something a snapshot of site-packages can stand in for.
# Comment lines are skipped: prose may mention these words, only commands matter.
if printf '%s\n' "$PRELUDE" | grep -vE '^[[:space:]]*#' \
     | grep -qE 'hipcc|(^|[[:space:]"=])/(r4d|bo|cache|startup-cache)(/|[[:space:]"]|$)|(^|[^_])exec '; then
  echo "FATAL: extracted prelude reaches past the patch block (hipcc/r4d/bo/cache/exec)" >&2; exit 1
fi

# --- 2. apply in a throwaway container and copy out the delta ---------------------------
mkdir -p "$(dirname "$OUT")"
TMP=$(mktemp -d "$(dirname "$OUT")/.boot-overlay-build.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/patched"
printf '%s\n' "$PRELUDE" > "$TMP/prelude.sh"

EXTRA=()
for kv in ${STARTUP_CACHE_OVERLAY_ENV:-}; do EXTRA+=(-e "$kv"); done
for m in ${STARTUP_CACHE_OVERLAY_MOUNTS:-}; do
  [ -d "${m%%:*}" ] || { echo "FATAL: overlay mount source ${m%%:*} does not exist" >&2; exit 1; }
  EXTRA+=(-v "$m:ro,z")
done
echo "[boot-overlay] applying the patch prelude of $(basename "$LAUNCHER") in a throwaway container ($IMAGE${STARTUP_CACHE_OVERLAY_ENV:+, $STARTUP_CACHE_OVERLAY_ENV})"
"$RUNTIME" run --rm --network=none --entrypoint bash \
  -v "$PATCHES_DIR":/patches:z -v "$TMP":/out:z -e SP="$SP_IN" ${EXTRA[@]+"${EXTRA[@]}"} \
  "$IMAGE" -c '
    set -e -o pipefail
    find "$SP" -type f ! -name "*.pyc" | sort > /tmp/before
    touch /tmp/stamp; sleep 1
    ( set -e; cd /patches; . /out/prelude.sh ) 2>&1 | tee /out/prelude.log
    find "$SP" -type f ! -name "*.pyc" | sort > /tmp/after
    if comm -23 /tmp/before /tmp/after | grep -q .; then
      echo "FATAL: the prelude removed files; an overlay cannot express that:" >&2
      comm -23 /tmp/before /tmp/after >&2; exit 1
    fi
    cd "$SP"
    find . -type f ! -name "*.pyc" -newer /tmp/stamp | sed "s|^\./||" | sort > /out/files.txt
    xargs -r -d "\n" cp --parents -t /out/patched < /out/files.txt
  '

if grep -qE '\bWARN(ING)?\b|\bFAIL\b' "$TMP/prelude.log"; then
  echo "FATAL: a patch printed a warning or failure; not publishing an overlay that would hide it:" >&2
  grep -E '\bWARN(ING)?\b|\bFAIL\b' "$TMP/prelude.log" >&2; exit 1
fi
n=$(wc -l < "$TMP/files.txt")
echo "[boot-overlay] $n changed/new files"
if [ "$n" -eq 0 ] || [ "$n" -gt 200 ]; then
  echo "FATAL: $n changed files is outside the sane range (1-200); not publishing" >&2; exit 1
fi
[ "$(find "$TMP/patched" -type f | wc -l)" -eq "$n" ] || { echo "FATAL: copied file count != list" >&2; exit 1; }

# --- 3. meta + atomic publish ------------------------------------------------------------
_boot_meta "$LAUNCHER" "$PATCHES_DIR" > "$TMP/meta.env"
rm -f "$TMP/prelude.sh"
chmod 755 "$TMP"
rm -rf "$OUT.old"
if [ -e "$OUT" ]; then mv "$OUT" "$OUT.old"; fi
mv "$TMP" "$OUT"
trap - EXIT
rm -rf "$OUT.old"
echo "[boot-overlay] published $OUT"
echo "[boot-overlay] meta: $(cat "$OUT/meta.env")"
