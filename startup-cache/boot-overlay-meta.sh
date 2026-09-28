# boot-overlay-meta.sh -- sourced by startup-cache.sh AND build-boot-overlay.sh.
#
# The one definition of the boot overlay's identity. The builder writes this line to meta.env;
# startup-cache.sh recomputes it every boot and the overlay is used only on an exact match.
# One file, so the two can never disagree about the formula.
#
# Fields: <image tag> <image id, 12 hex> <patch-set sha16> <prelude sha16> <builder sha16> <env>
#   * image id, not just the tag: a re-pulled image under the same tag must not reuse an
#     overlay taken from the old one (the overlay replaces whole site-packages files).
#   * patch set: every *.py and *.json in the patch dir and one level below it, by content,
#     keyed by relative path -- and the same for every STARTUP_CACHE_OVERLAY_MOUNTS dir,
#     labelled by its container path. Deliberately wider than what a given prelude reads: an extra
#     file can only cause a miss, a missing one could cause a false hit.
#   * prelude: the text of the patch block itself, extracted from the launcher. The patch set
#     alone would miss an edit to the block that leaves every patch FILE unchanged -- a script
#     added, dropped or reordered, or a copy retargeted -- and that is a false hit.
#   * builder: build-boot-overlay.sh itself, so a change to how overlays are taken (what counts
#     as changed, what is refused) retires every overlay taken the old way.
# Not in the line, deliberately:
#   * GPU arch. The overlay is Python source and JSON; nothing in it is compiled.
#   * serve knobs in general. Only STARTUP_CACHE_OVERLAY_ENV reaches the builder, and it is the
#     <env> field (commas for spaces; "-" when empty). A launcher must list there every value its
#     prelude or its patch scripts branch on at PATCH time. Audited 2026-09-28: the mxfp4 prelude
#     branches on none; the kvcache prelude on KVOFF_MINIMAL and RADIANCE_GDN_LAZY; every other
#     os.environ read in either set of patch scripts is inside injected code (read at serve time),
#     a log message, or a test-only path override the launchers never set.

_SC_BUILDER="$(dirname "$(realpath -m "${BASH_SOURCE[0]}")")/build-boot-overlay.sh"

# _boot_prelude <launcher>: the lines strictly between the BEGIN/END patch prelude markers.
_boot_prelude() {
  awk '/^[[:space:]]*# BEGIN patch prelude$/ { on = 1; next }
       /^[[:space:]]*# END patch prelude$/   { exit }
       on' "$1" 2>/dev/null
}

# _boot_meta <launcher> <patch-dir>   (needs RUNTIME IMAGE; reads STARTUP_CACHE_OVERLAY_ENV/_MOUNTS)
_boot_tree_sha() {   # _boot_tree_sha <dir> <label>
  echo "== $2"
  (cd "$1" 2>/dev/null && find . -maxdepth 2 \( -name .git -o -name __pycache__ \) -prune -o \
     -type f \( -name '*.py' -o -name '*.json' \) -print | LC_ALL=C sort | xargs -r -d '\n' sha256sum)
}
_boot_meta() {
  local _bo_id _bo_sha _bo_pre _bo_bld _bo_env _bo_m
  _bo_id=$("$RUNTIME" image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) || true
  _bo_id=${_bo_id#sha256:}
  _bo_sha=$( { _boot_tree_sha "$2" /patches
               for _bo_m in ${STARTUP_CACHE_OVERLAY_MOUNTS:-}; do _boot_tree_sha "${_bo_m%%:*}" "${_bo_m#*:}"; done
             } | sha256sum | cut -c1-16)
  _bo_pre=$(_boot_prelude "$1" | sha256sum | cut -c1-16)
  _bo_bld=$(sha256sum "$_SC_BUILDER" 2>/dev/null | cut -c1-16)
  _bo_env=$(echo ${STARTUP_CACHE_OVERLAY_ENV:-} | tr ' ' ',')
  echo "${IMAGE##*:} ${_bo_id:0:12} ${_bo_sha} ${_bo_pre} ${_bo_bld:-nobuilder} ${_bo_env:--}"
}
