#!/usr/bin/env bash
# clear-cache.sh -- wipe the KV-offload disk tier and/or the startup caches, from a menu.
#
#   ./clear-cache.sh             interactive menu
#   ./clear-cache.sh --dry-run   the same menu; prints what it would delete, deletes nothing
#
# Everything here is CACHE: deleting it never loses data. The cost is time --
#   * KV disk tier: conversations stored there are recomputed the next time they are sent.
#   * startup caches: the next boot recompiles (a few minutes instead of ~2).
#
# What it looks at (the launcher's own defaults; the same environment variables override them):
#   KV disk tier   ${KVCACHE_DISK:-/kvcache}/blocks/<entry>     one entry per model checkpoint and numerics
#                  (e.g. Qwen3.8-27B-MXFP4-mtpfp8-f16ssm-canon). The blocks/ directory itself is kept:
#                  the reaper's unit checks for it.
#   startup caches $HOME/.radiance-cache-w4a8-*                 compile caches, one per image + flag set
#                  ${R4D_CACHE:-$HOME/.cache/radiance-libr4d}/*  libr4d builds, one per patch + image
#                  $HOME/.cache/startup-cache/boot-overlay/*    pre-patched site-packages, one per launcher
#
# Refuses to delete anything a RUNNING container has mounted (podman or docker): stop the server first.
set -uo pipefail

DRY=0
case "${1:-}" in
  --dry-run|-n) DRY=1 ;;
  -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

KV_BLOCKS="${KVCACHE_DISK:-/kvcache}/blocks"
COMPILE_PREFIX="$HOME/.radiance-cache-w4a8-"
R4D_DIR="${R4D_CACHE:-$HOME/.cache/radiance-libr4d}"
OVERLAY_DIR="$HOME/.cache/startup-cache/boot-overlay"

# ---------------------------------------------------------------- helpers
human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0} B"; }
size_of() { du -sb -- "$1" 2>/dev/null | awk '{print $1}'; }
# newest mtime among the entry and its first two levels (a full walk of a KV tree is slow)
last_used() {
  find "$1" -maxdepth 2 -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1 |
    { read -r t; [ -n "$t" ] && date -d "@$t" '+%Y-%m-%d %H:%M' || echo "?"; }
}

# Mount sources of every running container (podman and docker, whichever exist).
MOUNTS=""
for rt in podman docker; do
  command -v "$rt" >/dev/null 2>&1 || continue
  ids=$("$rt" ps -q 2>/dev/null) || continue
  [ -n "$ids" ] || continue
  # shellcheck disable=SC2086
  MOUNTS+=$("$rt" inspect --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' $ids 2>/dev/null)$'\n'
done
# in_use PATH: true if a running container mounts PATH, something inside it, or a parent of it.
in_use() {
  local p m
  p=$(realpath -m -- "$1")
  while IFS= read -r m; do
    [ -n "$m" ] || continue
    m=$(realpath -m -- "$m")
    case "$m/" in "$p/"*) return 0 ;; esac
    case "$p/" in "$m/"*) return 0 ;; esac
  done <<<"$MOUNTS"
  return 1
}

# ---------------------------------------------------------------- entry lists
# Each list fills ENTRY_PATH[] and ENTRY_DESC[].
ENTRY_PATH=(); ENTRY_DESC=()
add_entry() {
  local path=$1 kind=$2 sz
  sz=$(size_of "$path")
  ENTRY_PATH+=("$path")
  ENTRY_DESC+=("$(printf '%-9s %8s  last used %s  %s' "$kind" "$(human "$sz")" "$(last_used "$path")" "$(basename "$path")")")
}

list_kv() {
  ENTRY_PATH=(); ENTRY_DESC=()
  [ -d "$KV_BLOCKS" ] || return 0
  local d models
  for d in "$KV_BLOCKS"/*/; do
    [ -d "$d" ] || continue
    d=${d%/}
    add_entry "$d" "kv-model"
    # name the checkpoint(s) vLLM stored under it (its own <model>_<hash> directories)
    models=$(find "$d" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sed 's/_r[0-9]*$//; s/_[0-9a-f]\{12\}$//; s/^_models_//' | sort -u | paste -sd' ')
    [ -n "$models" ] && ENTRY_DESC[-1]+="  [$models]"
  done
}

list_startup() {
  ENTRY_PATH=(); ENTRY_DESC=()
  local d
  for d in "$COMPILE_PREFIX"*; do [ -d "$d" ] && add_entry "$d" "compile"; done
  for d in "$R4D_DIR"/*/; do
    d=${d%/}; [ -d "$d" ] || continue
    case "$(basename "$d")" in .build) continue ;; esac
    add_entry "$d" "libr4d"
  done
  for d in "$OVERLAY_DIR"/*/; do
    d=${d%/}; [ -d "$d" ] && add_entry "$d" "overlay"
  done
}

# ---------------------------------------------------------------- selection
# parse_sel "1 3,5-7" N -> the chosen indices (1-based), one per line; returns 1 on bad input
parse_sel() {
  local tok a b i
  for tok in ${1//,/ }; do
    if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then a=${BASH_REMATCH[1]}; b=${BASH_REMATCH[2]}
    elif [[ "$tok" =~ ^[0-9]+$ ]]; then a=$tok; b=$tok
    else return 1; fi
    (( a >= 1 && b <= $2 && a <= b )) || return 1
    for ((i = a; i <= b; i++)); do echo "$i"; done
  done | sort -un
}

wipe() {   # wipe PATH...
  local p total=0 sz failed=0
  echo
  for p in "$@"; do
    if in_use "$p"; then
      echo "  SKIPPED (a running container uses it -- stop the server first): $p"
      continue
    fi
    sz=$(size_of "$p"); total=$((total + ${sz:-0}))
    if [ "$DRY" = 1 ]; then
      echo "  [dry-run] would delete $p ($(human "$sz"))"
    elif rm -rf -- "$p" 2>/dev/null; then
      rm -f -- "$p.refused" 2>/dev/null   # a boot overlay's "refused" marker sits beside it
      echo "  deleted $p ($(human "$sz"))"
    else
      echo "  FAILED to delete $p (permission?) -- re-run with sudo, or delete it by hand"
      failed=1
    fi
  done
  if [ "$DRY" = 1 ]; then echo "  [dry-run] total $(human "$total"); nothing was deleted"
  else echo "  freed about $(human "$total")"; fi
  return "$failed"
}

confirm() {   # confirm PATH...
  local p
  echo
  echo "About to delete:"
  for p in "$@"; do echo "  $p"; done
  [ "$DRY" = 1 ] && return 0
  read -r -p "Type yes to delete: " ans
  [ "$ans" = yes ]
}

# category NAME LISTFN
category() {
  local name=$1 listfn=$2 choice sel i n keep picked
  while true; do
    echo; echo "== $name"
    echo "   (sizing...)"
    "$listfn"
    n=${#ENTRY_PATH[@]}
    if (( n == 0 )); then echo "   nothing found"; return; fi
    for ((i = 0; i < n; i++)); do printf '  %2d) %s\n' $((i + 1)) "${ENTRY_DESC[i]}"; done
    echo
    echo "  a) wipe ALL of the above"
    echo "  w) choose entries to WIPE"
    echo "  k) choose entries to KEEP, wipe the rest"
    echo "  b) back"
    read -r -p "> " choice || return
    picked=()
    case "$choice" in
      a) picked=("${ENTRY_PATH[@]}") ;;
      w|k)
        read -r -p "Numbers (e.g. 1 3 5-7): " sel || return
        if ! mapfile -t keep < <(parse_sel "$sel" "$n") || [ ${#keep[@]} -eq 0 ]; then
          echo "  not a valid selection"; continue
        fi
        for ((i = 1; i <= n; i++)); do
          if printf '%s\n' "${keep[@]}" | grep -qx "$i"; then
            [ "$choice" = w ] && picked+=("${ENTRY_PATH[i-1]}")
          else
            [ "$choice" = k ] && picked+=("${ENTRY_PATH[i-1]}")
          fi
        done ;;
      b|q) return ;;
      *) echo "  ?"; continue ;;
    esac
    if (( ${#picked[@]} == 0 )); then echo "  nothing to wipe"; continue; fi
    if confirm "${picked[@]}"; then wipe "${picked[@]}"; else echo "  cancelled"; fi
  done
}

# ---------------------------------------------------------------- main menu
[ "$DRY" = 1 ] && echo "DRY RUN: nothing will be deleted."
[ -n "$MOUNTS" ] && echo "Note: containers are running; anything they mount will be skipped."
while true; do
  echo
  echo "clear-cache -- everything here is cache; deleting costs recompute time, never data"
  echo "  1) KV-offload disk tier   ($KV_BLOCKS)"
  echo "  2) startup caches         (compile trees, libr4d builds, boot overlays)"
  echo "  3) BOTH, everything in 1 and 2"
  echo "  q) quit"
  read -r -p "> " top || exit 0
  case "$top" in
    1) category "KV-offload disk tier: one entry per model / numerics" list_kv ;;
    2) category "startup caches: one entry per build" list_startup ;;
    3) list_kv; all=("${ENTRY_PATH[@]}"); list_startup; all+=("${ENTRY_PATH[@]}")
       if (( ${#all[@]} == 0 )); then echo "  nothing found"
       elif confirm "${all[@]}"; then wipe "${all[@]}"; else echo "  cancelled"; fi ;;
    q|"") exit 0 ;;
    *) echo "  ?" ;;
  esac
done
