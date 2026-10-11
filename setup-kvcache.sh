#!/usr/bin/env bash
# setup-kvcache.sh -- set up (or change) the KV-cache offload tiers, interactively.
#
#   ./setup-kvcache.sh             menu: RAM tier, disk tier + reaper, wipe caches, status
#   ./setup-kvcache.sh --dry-run   the same questions; prints every root command instead of running it
#
# What it can do, each step asking first and showing the commands it will run as root:
#   RAM tier   measures RAM, suggests a tier size, resizes /dev/shm in /etc/fstab (backed up) and live.
#   Disk tier  measures the filesystem's direct-I/O speed, then uses an existing mount, formats a spare
#              disk, or creates an image file; installs the reaper (kv-cache/ops) with a drop-in that
#              points it at the folder you chose, and enables its timer.
#   Wipe       runs ./clear-cache.sh (KV disk tier and startup caches, per model).
# The choices go to kvcache.env next to the launcher, which reads it at start. Anything set in the
# environment (llama-swap env:, the shell) still wins over kvcache.env.
# Sizing background: KV-CACHE.md -> "Sizing", kv-cache/docs/SETUP.md.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ENVF="$HERE/kvcache.env"
OPS="$HERE/kv-cache/ops"
DRY=
case "${1:-}" in
  --dry-run|-n) DRY=1 ;;
  -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
  "") ;;
  *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
esac

RESERVE_GIB=15        # the launcher keeps this much RAM for the engine, drafter, pinned embedding and OS
TIER_DEFAULT=16       # the launcher's default RAM tier
TIER_MIN=10           # stages one 262,144-token resume (smaller works, the longest resumes cannot come back)
SHM_HEADROOM=4        # /dev/shm = tier + this
DISK_DEFAULT_GIB=128  # recommended disk-tier volume
REAP_PCT_DEFAULT=75   # the reaper keeps the volume at or below this
BREAK_EVEN_MBS=100    # direct-read speed at which restoring from disk ties recomputing (~2,900 tok/s x ~35 KB)

say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n== %s ==\n' "$*"; }
warn() { printf '  ! %s\n' "$*"; }
ask() {  # ask VAR "question" default
  local __v; read -r -p "$2 [${3}]: " __v || exit 1
  printf -v "$1" '%s' "${__v:-$3}"
}
yes_no() {  # yes_no "question" default(y|n)
  local a; read -r -p "$1 [$( [ "$2" = y ] && echo Y/n || echo y/N )]: " a || exit 1
  a=${a:-$2}; [[ "$a" =~ ^[Yy] ]]
}
root() {  # run (or, with --dry-run, print) one command as root
  say "  # $*"
  [ -n "$DRY" ] && return 0
  if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi
}
root_sh() {  # same, through sh -c (redirections)
  say "  # $1"
  [ -n "$DRY" ] && return 0
  if [ "$(id -u)" = 0 ]; then sh -c "$1"; else sudo sh -c "$1"; fi
}
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# ---------------------------------------------------------------- kvcache.env
env_get() { [ -f "$ENVF" ] && sed -n "s/^: \"\${$1:=\(.*\)}\"$/\1/p" "$ENVF" | tail -1; }
env_set() {  # env_set NAME VALUE -- kvcache.env holds `: "${NAME:=VALUE}"` lines, so the environment wins
  local tmp; tmp=$(mktemp)
  if [ -f "$ENVF" ]; then grep -v "^: \"\${$1:=" "$ENVF" > "$tmp"; else
    printf '# Written by setup-kvcache.sh; read by the launcher at start. Environment variables win.\n' > "$tmp"; fi
  printf ': "${%s:=%s}"\n' "$1" "$2" >> "$tmp"
  if [ -n "$DRY" ]; then say "  (dry run) kvcache.env: $1=$2"; rm -f "$tmp"; else mv "$tmp" "$ENVF"; say "  kvcache.env: $1=$2"; fi
}

# ---------------------------------------------------------------- measurements
gib_of_kb() { awk -v k="$1" 'BEGIN{printf "%.1f", k/1048576}'; }
mem_kb() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo; }
shm_gib() { python3 -c "import os;s=os.statvfs('/dev/shm');print('%.1f'%(s.f_blocks*s.f_frsize/2**30))"; }

# direct-I/O throughput of a folder, the way the tier uses it: 64 files of 16 MiB (one KV block file is
# ~14 MB) written by 4 writers, then read back by 8 readers (the tier's thread counts), page cache bypassed
# (the tier uses O_DIRECT, so a buffered test would flatter a slow disk). Prints "WRITE READ" MB/s.
disk_speed() {
  local dir=$1 t0 t1 n=64 i w r
  local tdir; tdir=$(mktemp -d "$dir/.kvcache-speedtest.XXXX" 2>/dev/null) || { echo "0 0"; return; }
  t0=$(date +%s.%N)
  for j in 0 1 2 3; do ( for i in $(seq $j 4 $((n - 1))); do
    dd if=/dev/urandom of="$tdir/f$i" bs=16M count=1 oflag=direct conv=fsync status=none || exit 1; done ) & done
  wait; t1=$(date +%s.%N)
  w=$(awk -v a="$t0" -v b="$t1" -v m=$((16 * n)) 'BEGIN{printf "%.0f", m*1.048576/(b-a)}')
  t0=$(date +%s.%N)
  for j in 0 1 2 3 4 5 6 7; do ( for i in $(seq $j 8 $((n - 1))); do
    dd if="$tdir/f$i" of=/dev/null bs=16M iflag=direct status=none || exit 1; done ) & done
  wait; t1=$(date +%s.%N)
  r=$(awk -v a="$t0" -v b="$t1" -v m=$((16 * n)) 'BEGIN{printf "%.0f", m*1.048576/(b-a)}')
  [ -s "$tdir/f$((n - 1))" ] || { w=0; r=0; }
  rm -rf "$tdir"
  echo "$w $r"
}
speed_verdict() {  # speed_verdict READ_MBS
  if [ "$1" -gt 2500 ]; then
    echo "faster than most SSDs read cold: probably a cache below this filesystem (a VM host, a RAID card). The tier's real rate on a long session will be lower; judge by the disk underneath"
  elif [ "$1" -lt "$BREAK_EVEN_MBS" ]; then
    echo "SLOWER than recomputing (break-even ~${BREAK_EVEN_MBS} MB/s): a disk tier here would cost more than it saves"
  elif [ "$1" -lt 250 ]; then
    echo "about $(( $1 / BREAK_EVEN_MBS ))x recompute speed: worth having, a faster disk would help"
  else
    echo "about $(( $1 / BREAK_EVEN_MBS ))x recompute speed: a good disk tier"
  fi
}
test_dir_for() {  # the nearest existing folder at or above a path (for speed tests before it exists)
  local p=$1; while [ ! -d "$p" ]; do p=$(dirname "$p"); done; echo "$p"
}

# ---------------------------------------------------------------- RAM tier
step_ram() {
  hdr "RAM tier (/dev/shm)"
  local total avail max shm cur sugg tier need
  total=$(mem_kb MemTotal); avail=$(mem_kb MemAvailable)
  max=$(( total / 1048576 - RESERVE_GIB ))
  shm=$(shm_gib); cur=$(env_get KVCACHE_TIER_GIB)
  say "  RAM: $(gib_of_kb "$total") GiB total, $(gib_of_kb "$avail") GiB available now; /dev/shm is $shm GiB."
  say "  The tier is pinned: it is taken from the host for as long as the server runs. The launcher keeps"
  say "  $RESERVE_GIB GiB for the engine and OS, so the largest tier this box can hold is ${max} GiB."
  [ -n "$cur" ] && say "  Currently configured: KVCACHE_TIER_GIB=$cur"
  if [ "$max" -lt 4 ]; then
    warn "Not enough RAM for a useful tier (need $((RESERVE_GIB + TIER_MIN)) GiB or more). Offload will be off."
    yes_no "Turn the RAM tier (and with it offload) off in kvcache.env?" y && env_set KVCACHE_TIER_GIB 0
    return
  fi
  # suggestion: the default where it fits; more when RAM is plentiful and there will be no disk tier
  if [ "$max" -ge "$TIER_DEFAULT" ]; then sugg=$TIER_DEFAULT; else sugg=$max; fi
  if [ "$(env_get KVCACHE_DISK_TIER)" = 0 ] && [ "$max" -ge 24 ]; then
    sugg=$(( max < 32 ? max : 32 ))
    say "  No disk tier is configured, so a bigger RAM tier is worth it: suggesting $sugg GiB."
  fi
  [ "$sugg" -lt "$TIER_MIN" ] && warn "$sugg GiB is below $TIER_MIN GiB: the longest (262k-token) conversations cannot be restored from disk."
  [ $(( avail / 1048576 )) -lt "$sugg" ] && warn "only $(gib_of_kb "$avail") GiB is free right now; something else is using RAM (fine if it is a running server)."
  while :; do
    ask tier "RAM tier size in GiB (0 = offload off)" "$sugg"
    is_int "$tier" && [ "$tier" -le "$max" ] && break
    warn "enter a whole number from 0 to $max"
  done
  env_set KVCACHE_TIER_GIB "$tier"
  [ "$tier" = 0 ] && return
  need=$(( tier + SHM_HEADROOM ))
  if awk -v s="$shm" -v n="$need" 'BEGIN{exit !(s >= n)}'; then
    say "  /dev/shm ($shm GiB) already holds a $tier GiB tier. Nothing to change."
    return
  fi
  say "  /dev/shm must be at least ${need}G for a $tier GiB tier (it is $shm GiB)."
  yes_no "Resize /dev/shm to ${need}G now and in /etc/fstab (a backup is kept)?" y || { warn "skipped: the launcher will turn offload off until /dev/shm is big enough"; return; }
  local line="tmpfs /dev/shm tmpfs rw,nosuid,nodev,inode64,size=${need}G 0 0" bak
  bak="/etc/fstab.bak-kvcache-$(date +%Y%m%d-%H%M%S)"
  root cp -a /etc/fstab "$bak"
  if grep -qE '^[^#]*[[:space:]]/dev/shm[[:space:]]' /etc/fstab; then
    root_sh "sed -i -E 's|^([^#]*[[:space:]]/dev/shm[[:space:]].*)$|# \1   (replaced by setup-kvcache.sh)\n$line|' /etc/fstab"
  else
    root_sh "printf '%s\n' '# /dev/shm holds the KV-offload RAM tier (setup-kvcache.sh)' '$line' >> /etc/fstab"
  fi
  if [ -z "$DRY" ] && ! findmnt --verify --tab-file /etc/fstab >/dev/null 2>&1; then
    warn "/etc/fstab no longer verifies; restoring the backup"; root cp -a "$bak" /etc/fstab; return
  fi
  root mount -o remount,size="${need}G" /dev/shm
  say "  /dev/shm is now $( [ -n "$DRY" ] && echo "${need}G (dry run)" || shm_gib ) GiB. fstab backup: $bak"
}

# ---------------------------------------------------------------- disk tier
reap_drop_in() {  # reap_drop_in DIR MAX_GB PCT MIN_FREE
  local d=/etc/systemd/system/kvcache-reap.service.d
  root mkdir -p "$d"
  root_sh "printf '%s\n' '# Written by setup-kvcache.sh' '[Unit]' 'ConditionPathIsDirectory=' 'ConditionPathIsDirectory=$1/blocks' '[Service]' 'Environment=KVCACHE_ROOT=$1/blocks' 'Environment=KVCACHE_MAX_GB=$2' 'Environment=KVCACHE_TARGET_PCT=$3' 'Environment=KVCACHE_MIN_FREE_GB=$4' > $d/10-setup-kvcache.conf"
}
gb_size() { df -P -B1000000000 "$1" 2>/dev/null | awk 'NR==2{print $2}'; }   # filesystem size, GB
gb_free() { df -P -B1000000000 "$1" 2>/dev/null | awk 'NR==2{print $4}'; }   # free, GB
install_reaper() {  # install_reaper DIR DEDICATED(1|0)
  local dir=$1 dedicated=$2 vol free used max sugg pct minfree headroom
  vol=$(gb_size "$dir"); free=$(gb_free "$dir")
  used=$(du -s -B1000000000 "$dir/blocks" 2>/dev/null | cut -f1); used=${used:-0}
  hdr "Size limit and reaper"
  say "  The disk tier never deletes on its own. The reaper (a systemd timer, every minute) enforces a"
  say "  maximum size for $dir/blocks: it deletes the oldest blocks down to that maximum minus a"
  say "  headroom (a minute of peak writes), so the folder does not pass the maximum between runs."
  say "  The filesystem is ${vol} GB, ${free} GB free; the cache holds ${used} GB now."
  # dedicated: 90% of the volume as the ceiling puts the after-reap level (ceiling - headroom) near 75%
  if [ "$dedicated" = 1 ]; then sugg=$(( vol * 90 / 100 )); else sugg=$(( (free + used) * 60 / 100 )); fi
  [ "$sugg" -gt 96 ] && [ "$dedicated" = 0 ] && sugg=96          # 96 GB: the recommended cache (128 GiB volume at 75%)
  while :; do
    ask max "Maximum disk cache size in GB (96 recommended, at least 32)" "$sugg"
    is_int "$max" && [ "$max" -ge 32 ] && [ "$max" -le $(( free + used )) ] && break
    warn "enter a whole number from 32 to $(( free + used )) (free space plus what the cache already holds)"
  done
  headroom=$(( max / 5 < 24 ? max / 5 : 24 ))
  # the volume guards stay on underneath: never let the filesystem itself fill, whoever else writes to it
  pct=95; minfree=$(( vol / 10 < 24 ? vol / 10 : 24 ))
  [ "$dedicated" = 1 ] && [ $(( max + minfree )) -gt "$vol" ] && minfree=$(( vol - max > 0 ? vol - max : 0 ))
  say "  Limit ${max} GB: the reaper keeps $dir/blocks at or below $(( max - headroom )) GB after each run"
  say "  (headroom ${headroom} GB), and the filesystem at or below ${pct}% with ${minfree} GB free."
  root install -m 0755 "$OPS/kvcache-reap.sh" /usr/local/bin/kvcache-reap.sh
  root install -m 0644 "$OPS/kvcache-reap.service" "$OPS/kvcache-reap.timer" /etc/systemd/system/
  reap_drop_in "$dir" "$max" "$pct" "$minfree"
  root systemctl daemon-reload
  root systemctl enable --now kvcache-reap.timer
  [ -z "$DRY" ] && say "  timer: $(systemctl is-enabled kvcache-reap.timer 2>/dev/null) / $(systemctl is-active kvcache-reap.timer 2>/dev/null)"
  [ -z "$DRY" ] && { say "  first run:"; root systemctl start kvcache-reap.service; journalctl -u kvcache-reap.service -n 3 --no-pager -o cat 2>/dev/null | sed 's/^/    /'; }
}
own_blocks() {  # the container writes as the invoking user (rootless podman); docker users may need root
  local dir=$1 u=${SUDO_USER:-$(id -un)}
  root mkdir -p "$dir/blocks"
  root chown "$u:" "$dir" "$dir/blocks"
}
fstab_add() {  # fstab_add LINE (backed up, verified, restored on failure)
  local bak; bak="/etc/fstab.bak-kvcache-$(date +%Y%m%d-%H%M%S)"
  root cp -a /etc/fstab "$bak"
  root_sh "printf '%s\n' '# KV-offload disk tier (setup-kvcache.sh)' '$1' >> /etc/fstab"
  if [ -z "$DRY" ] && ! findmnt --verify --tab-file /etc/fstab >/dev/null 2>&1; then
    warn "/etc/fstab no longer verifies; restoring the backup"; root cp -a "$bak" /etc/fstab; return 1
  fi
}
step_disk() {
  hdr "Disk tier"
  say "  A disk tier keeps finished conversations after the RAM tier is full: ~2.2M tokens in 96 GB,"
  say "  restored at disk speed instead of recomputed. The reaper holds it to a size you choose."
  say "  Without one, make the RAM tier larger."
  if ! yes_no "Use a disk tier?" y; then
    env_set KVCACHE_DISK_TIER 0
    if systemctl is-enabled kvcache-reap.timer >/dev/null 2>&1 && yes_no "Disable the installed reaper timer too?" n; then
      root systemctl disable --now kvcache-reap.timer
    fi
    return
  fi
  local dir cur dedicated=1; cur=$(env_get KVCACHE_DISK)
  ask dir "Folder for the disk tier (inside the container it is always /kvcache)" "${cur:-/kvcache}"
  dir=${dir%/}
  if mountpoint -q "$dir"; then
    local src fs size used
    read -r src fs size used < <(findmnt -n -o SOURCE,FSTYPE,SIZE,USE% "$dir")
    say "  $dir is a mounted $fs filesystem ($src), $size, $used used."
    local other; other=$(find "$dir" -mindepth 1 -maxdepth 1 ! -name blocks ! -name lost+found ! -name '.kvcache-speedtest.*' | head -3)
    [ -n "$other" ] && { dedicated=0; warn "it holds other files ($(echo $other | tr '\n' ' ')); the size limit below keeps the cache to its share."; }
  else
    say "  $dir is not a mount point. How should the disk tier be stored?"
    say "    1) an image file on an existing filesystem, mounted at $dir (no partitioning; the simple choice)"
    say "    2) a spare disk or partition, formatted ext4 (ERASES it)"
    say "    3) a plain folder on the filesystem $dir would be on (shares that disk; the size limit below"
    say "       still holds, and the reaper never lets the filesystem itself fill)"
    local how; ask how "Choice" 1
    case "$how" in
      1) make_image "$dir" || return ;;
      2) make_device "$dir" || return ;;
      3) root mkdir -p "$dir"; dedicated=0 ;;
      *) warn "no change"; return ;;
    esac
  fi
  hdr "Speed test ($dir)"
  if [ -n "$DRY" ] && [ ! -d "$dir" ]; then
    say "  (dry run: $dir does not exist yet; testing $(test_dir_for "$dir") instead)"
  fi
  local tdir; tdir=$(test_dir_for "$dir"); local w r
  if [ -w "$tdir" ] || [ "$(id -u)" = 0 ]; then
    say "  writing and reading 1 GiB with direct I/O (bypassing the page cache, as the tier does)..."
    read -r w r < <(disk_speed "$tdir")
    if [ "${r:-0}" -gt 0 ]; then
      say "  write ${w} MB/s, read ${r} MB/s -> $(speed_verdict "$r")"
      say "  (read back right after writing: a cache under the filesystem, e.g. a VM host's, can flatter it)"
      if [ "$r" -lt "$BREAK_EVEN_MBS" ] && yes_no "Turn the disk tier off and use RAM only instead?" y; then
        env_set KVCACHE_DISK_TIER 0; return
      fi
    else
      warn "the test could not write to $tdir (direct I/O unsupported, or no space); skipped"
    fi
  else
    warn "$tdir is not writable by $(id -un); speed test skipped"
  fi
  own_blocks "$dir"
  install_reaper "$dir" "$dedicated"
  env_set KVCACHE_DISK_TIER 1
  env_set KVCACHE_DISK "$dir"
}
make_image() {  # make_image MOUNTPOINT
  local mnt=$1 img size host free
  ask img "Image file path" "/var/lib/kvcache.img"
  [ -e "$img" ] && { warn "$img already exists; choose another path or remove it first"; return 1; }
  host=$(test_dir_for "$(dirname "$img")")
  free=$(df -P -B1G "$host" | awk 'NR==2{print $4}')
  say "  $host has ${free} GiB free."
  local w r
  read -r w r < <(disk_speed "$host")
  [ "${r:-0}" -gt 0 ] && say "  its direct-I/O speed: write ${w} MB/s, read ${r} MB/s -> $(speed_verdict "$r")"
  local sugg=$DISK_DEFAULT_GIB
  [ $(( free * 8 / 10 )) -lt "$sugg" ] && sugg=$(( free * 8 / 10 ))
  ask size "Image size in GiB (128 recommended, for a 96 GB cache; at least 40)" "$sugg"
  is_int "$size" && [ "$size" -ge 40 ] && [ "$size" -lt "$free" ] || { warn "need a whole number from 40 to $((free - 1))"; return 1; }
  yes_no "Create a ${size} GiB ext4 image at $img and mount it at $mnt (and in /etc/fstab)?" y || return 1
  root mkdir -p "$(dirname "$img")" "$mnt"
  root fallocate -l "${size}G" "$img"
  root chmod 600 "$img"
  root mkfs.ext4 -q -F -L kvcache -m 0 "$img"
  fstab_add "$img $mnt ext4 loop,noatime,nodiratime,nofail 0 2" || return 1
  root systemctl daemon-reload
  root mount "$mnt"
}
make_device() {  # make_device MOUNTPOINT
  local mnt=$1 dev
  say "  Block devices with no mounted filesystem:"
  lsblk -dpno NAME,SIZE,TYPE,FSTYPE,MODEL | awk '$3=="disk"||$3=="part"' | while read -r n rest; do
    findmnt -rno TARGET -S "$n" >/dev/null 2>&1 && continue
    lsblk -no MOUNTPOINT "$n" | grep -q . && continue
    say "    $n $rest"
  done
  ask dev "Device to FORMAT (everything on it is erased)" ""
  [ -b "$dev" ] || { warn "$dev is not a block device"; return 1; }
  if lsblk -no MOUNTPOINT "$dev" | grep -q . || findmnt -rno TARGET -S "$dev" >/dev/null 2>&1; then
    warn "$dev (or a partition on it) is mounted; refusing"; return 1
  fi
  local fs; fs=$(lsblk -dno FSTYPE "$dev")
  [ -n "$fs" ] && warn "$dev holds a $fs filesystem. It will be destroyed."
  local typed; read -r -p "Type the device name again to erase it ($dev): " typed || exit 1
  [ "$typed" = "$dev" ] || { warn "did not match; nothing done"; return 1; }
  root mkfs.ext4 -q -F -L kvcache -m 0 "$dev"
  root mkdir -p "$mnt"
  # by LABEL, never /dev/sdX: adding a disk can rename devices and a stale name can stop the boot
  fstab_add "LABEL=kvcache $mnt ext4 noatime,nodiratime,nofail 0 2" || return 1
  root systemctl daemon-reload
  root mount "$mnt"
}

# ---------------------------------------------------------------- wipe / status
step_wipe() {
  hdr "Wipe caches"
  local d; d=$(env_get KVCACHE_DISK)
  KVCACHE_DISK=${KVCACHE_DISK:-${d:-/kvcache}} "$HERE/clear-cache.sh" ${DRY:+--dry-run}
}
step_status() {
  hdr "Status"
  local total; total=$(mem_kb MemTotal)
  say "  RAM $(gib_of_kb "$total") GiB, available $(gib_of_kb "$(mem_kb MemAvailable)") GiB; /dev/shm $(shm_gib) GiB"
  if [ -f "$ENVF" ]; then say "  kvcache.env:"; grep -v '^#' "$ENVF" | sed 's/^/    /'; else say "  kvcache.env: none (launcher defaults: 16 GiB RAM tier, disk tier at /kvcache)"; fi
  local d; d=$(env_get KVCACHE_DISK); d=${d:-/kvcache}
  if [ "$(env_get KVCACHE_DISK_TIER)" = 0 ]; then say "  disk tier: off"
  elif [ -d "$d" ]; then
    say "  disk tier: $d -- $(findmnt -n -o SOURCE,FSTYPE,SIZE,USE% --target "$d" | tr -s ' ')$(mountpoint -q "$d" || echo '  (not its own mount)')"
    [ -d "$d/blocks" ] && du -sh "$d/blocks"/* 2>/dev/null | sed 's/^/    /'
  else say "  disk tier: $d does not exist (the launcher will refuse to start with the disk tier on)"; fi
  say "  reaper timer: $(systemctl is-enabled kvcache-reap.timer 2>/dev/null || echo not installed) / $(systemctl is-active kvcache-reap.timer 2>/dev/null)"
  local dropin=/etc/systemd/system/kvcache-reap.service.d/10-setup-kvcache.conf
  [ -f "$dropin" ] && grep -E 'KVCACHE_(ROOT|TARGET_PCT|MIN_FREE_GB)' "$dropin" | sed 's/^Environment=/    /'
}

# ---------------------------------------------------------------- menu
[ -n "$DRY" ] && say "DRY RUN: nothing is changed; root commands are printed with a leading #."
while :; do
  hdr "KV-cache offload setup"
  say "  1) RAM tier       size it, resize /dev/shm"
  say "  2) Disk tier      speed test, filesystem, reaper"
  say "  3) Both           1 then 2"
  say "  4) Wipe caches    ./clear-cache.sh"
  say "  5) Status"
  say "  q) Quit"
  ask c "Choice" 5
  case "$c" in
    1) step_ram ;;
    2) step_disk ;;
    3) step_disk; step_ram ;;
    4) step_wipe ;;
    5) step_status ;;
    q|Q) break ;;
    *) warn "pick 1-5 or q" ;;
  esac
done
[ -f "$ENVF" ] && say "Settings are in $ENVF; restart the server (llama-swap: restart the entry) to use them."
