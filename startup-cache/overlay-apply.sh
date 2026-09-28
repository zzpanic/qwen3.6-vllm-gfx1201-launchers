# overlay-apply.sh <site-packages> -- CONTAINER side, SOURCED (not run) from the launcher body:
#
#   . /startup-cache/overlay-apply.sh "$SP"
#   if [ -z "$BOOT_OVERLAY_APPLIED" ]; then
#   # BEGIN patch prelude
#   ...
#   # END patch prelude
#   fi
#
# Copies the host-verified boot overlay (mounted at /bo) into site-packages and sets
# BOOT_OVERLAY_APPLIED=1; otherwise leaves it empty and the prelude runs as normal. Sourced so
# that, under the body's set -e, a copy that fails part-way stops the container instead of
# serving a half-patched site-packages.
BOOT_OVERLAY_APPLIED=""
if [ -n "${BOOT_OVERLAY_OK:-}" ] && [ -d /bo/patched ]; then
  cp -r /bo/patched/. "$1"/
  BOOT_OVERLAY_APPLIED=1
  echo "[startup-cache] boot overlay applied (meta verified on host)"
fi
