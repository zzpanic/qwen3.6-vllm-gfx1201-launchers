# System

This page describes the machine every number in this repository was measured on: the hypervisor host,
the GPU passthrough, the guest that serves the model, and the GPU power policy. Nothing here is required
to run the launcher. It is here so results can be compared like for like, and so the settings that
matter can be copied.

## Layout

| Layer | What |
|---|---|
| Host | ASUS P10S WS (Intel C236, LGA1151), Intel Skylake-family CPU (the guest sees `Intel Core Processor (Skylake, IBRS)`); TrueNAS SCALE 25.10.7, kernel `6.12.105-production+truenas` |
| GPU | AMD Radeon AI PRO R9700 (`1002:7551`, gfx1201 / RDNA4, 32 GB, 300 W), passed through whole to one VM (VFIO). **Physical link: PCIe 3.0 x16** (the board's limit; the card is Gen5) |
| Guest | Ubuntu 24.04.4 LTS, kernel `6.8.0-138-generic`, `amdgpu-dkms` 6.19.14 (out-of-tree driver from AMD's ROCm repository), 4 vCPUs, 40 GiB RAM |
| Serving | Podman, `stilldeadcode/vllm-radiance:0.9.3`, launched by llama-swap through `startup-qwen3.8-27b-mxfp4.sh` |

### PCIe 3.0 x16, and what it limits

The R9700 is a PCIe 5.0 card on a PCIe 3.0 board: about 15.75 GB/s per direction instead of about 63.
Decode and prefill run from VRAM and do not notice. Three things cross the bus and do:

- **The RAM KV tier.** Restores measured 12.0 GB/s, which is the practical ceiling of a 3.0 x16 link,
  so the tier's speed here is set by the slot, not the software. A 100k-token restore (about 7 GB)
  takes about 0.6 s. On a Gen4 or Gen5 board expect roughly 2x or 4x that rate.
- **Weight loading** at boot: 18 GB of checkpoint, a few seconds of the boot either way.
- **`EMBED_HOST=1`**: the input embedding is read across the bus, but only a few rows (10 KB each)
  per token, so it is not measurable.

`lspci` *inside the VM* reports the virtual root port's link (32 GT/s here), not the physical one.
Read the real link on the host: `sudo lspci -vv -d 1002:7551 | grep LnkSta`.

## Host kernel command line (TrueNAS)

```
libata.allow_tpm=1 amd_iommu=on iommu=pt kvm_amd.npt=1 kvm_amd.avic=1 intel_iommu=on zfsforce=1
nvme_core.multipath=N mitigations=off iommu=pt pcie_aspm=off
```

TrueNAS writes this line itself. Extra arguments go in through **System -> Advanced -> Kernel**
(`midclt call system.advanced.config` shows `kernel_extra_options`), not by editing GRUB, which
TrueNAS regenerates. `libata.allow_tpm=1`, `zfsforce=1` and `nvme_core.multipath=N` are TrueNAS's own
defaults.

| Argument | Effect here |
|---|---|
| `intel_iommu=on` | Turns on the IOMMU (VT-d). Required for passing the GPU through. |
| `iommu=pt` | Passthrough mode for devices the host keeps, so its own DMA skips translation. Standard for VFIO hosts. |
| `pcie_aspm=off` | No PCIe link power saving. The guest cannot control the passed-through link's ASPM, so if it is wanted off it has to be off here. |
| `mitigations=off` | Disables CPU side-channel mitigations. See the trade-off below. |
| `amd_iommu=on`, `kvm_amd.npt=1`, `kvm_amd.avic=1` | **No effect on this Intel host**; these only apply to AMD CPUs. Harmless; they can be dropped. |
| `iommu=pt` (second copy) | Duplicate; harmless. |

## Guest kernel command line (the serving VM)

```
amdgpu.ppfeaturemask=0xffffbfff mitigations=off amdgpu.ras_enable=0
```

Set in `/etc/default/grub` (`GRUB_CMDLINE_LINUX_DEFAULT`), then `sudo update-grub`.

| Argument | Effect | Measured |
|---|---|---|
| `amdgpu.ppfeaturemask=0xffffbfff` | The driver default is `0xfff7bfff`. This mask adds bit 19, `PP_GFX_DCS_MASK` (GFX async DCS). Overdrive, bit 14, stays **off**, so there is no `pp_od_clk_voltage` and no under- or over-volting. | Not isolated; the setting predates every benchmark here. |
| `mitigations=off` | Disables CPU side-channel mitigations in the guest. | +32% at concurrency 4 and +19% at 8 (174 -> 207 t/s); concurrency 1 within noise. Measured 2026-07-17 on an earlier vLLM V1 stack, not re-measured on this one. |
| `amdgpu.ras_enable=0` | Disables the driver's RAS (reliability / error-reporting) features. | **No effect** (llama.cpp pp512 867 vs 871 t/s, tg128 26.7 vs 26.8, within noise). |

**The `mitigations=off` trade-off.** It lets processes in the same kernel read each other's memory
through speculative-execution attacks; on the host, that extends to VMs reading each other. That buys a
measurable gain under concurrency on a CPU-bound stack. It only makes sense on a single-purpose box that
runs nothing untrusted, on a LAN, as here. On anything shared or internet-facing, leave mitigations on.
The guest's single-stream decode is GPU-bound and gained nothing.

**`ras_enable=0` buys nothing measurable** and turns off the driver's GPU error reporting, so you could
equally drop it and keep the default.

## GPU power policy (guest)

RDNA4 under-ramps its clocks under inference load on the default `auto` policy: it idles at about
44 MHz and oscillates during decode. A oneshot unit pins it at boot:

| sysfs (`/sys/class/drm/card*/device/`) | Value |
|---|---|
| `power_dpm_force_performance_level` | `manual` |
| `pp_power_profile_mode` | `COMPUTE` |
| `hwmon*/power1_cap` | 300 W (the card's default and its maximum; range 210-300 W) |

The COMPUTE profile holds the memory clock at its top state (decode is memory-bandwidth-bound), while
the core clock still ramps on demand and idles down. The alternative, `high`, pins every clock at its
maximum: there is no ramp latency, but idle power is higher. `rocm-smi`'s set-clock commands do not
work inside this passthrough guest, but sysfs writes do.

The script (`set-gpu-performance.sh compute`; `status` prints the current state):

```bash
#!/usr/bin/env bash
# Pin the R9700 (1002:7551) to manual + COMPUTE. Matches the card by PCI ID, and looks the COMPUTE
# index up by name because the numeric index can change between driver versions.
set -euo pipefail
for dev in /sys/class/drm/card*/device; do
  [ -f "$dev/power_dpm_force_performance_level" ] || continue
  [ "$(cat "$dev/vendor")" = 0x1002 ] && [ "$(cat "$dev/device")" = 0x7551 ] || continue
  echo manual > "$dev/power_dpm_force_performance_level"
  idx=$(awk '/COMPUTE/ {print $1; exit}' "$dev/pp_power_profile_mode")
  echo "$idx" > "$dev/pp_power_profile_mode"
  echo "$(basename "$(dirname "$dev")"): manual + COMPUTE ($idx)"
done
```

The unit, `/etc/systemd/system/gpu-performance.service`, enabled with `systemctl enable --now`:

```ini
[Unit]
Description=Set AMD R9700 GPU power policy (manual + COMPUTE profile) for inference
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/set-gpu-performance.sh compute
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Check it with:

```bash
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
grep '\*' /sys/class/drm/card*/device/pp_power_profile_mode
```

LACT (`lactd`) is installed for monitoring, with fan control off and no profile, so it does not fight
the unit. Its shipped unit's `After=multi-user.target` formed an ordering cycle with llama-swap, and
systemd silently dropped llama-swap's start job. The local copy of the unit orders on `basic.target`
instead.

## Guest memory and filesystems

| Setting | Value | Why |
|---|---|---|
| `/dev/shm` | `tmpfs size=28G` in `/etc/fstab` | The RAM KV tier lives here. It must hold the tier (19 GiB for one full 262,144-token context) plus a 0.25 GiB margin. The default (50% of RAM) is too small on a 40 GiB VM. |
| `/kvcache` | dedicated ext4, `noatime,nodiratime,nofail` | The disk KV tier. **128 GiB or more recommended**; this box's 94 GiB volume is below that (KV-CACHE.md). |
| `vm.swappiness` | 10 | Evict disposable page cache before swapping anonymous memory. |
| `vm.max_map_count` | 1048576 | Large mmap counts (Ubuntu bug 2057792). |
| Transparent huge pages | `madvise` (default) | Unchanged. |

**Host RAM.** The peak during the stress run was about 27 GB: a 16 GiB RAM tier, the 2.37 GiB pinned
embedding, and the engine. At the 19 GiB tier, plan on about 30 GB, so 40 GiB leaves room. Do not size
the VM below 32 GiB with this configuration.

## Reading the same settings on your machine

On the guest:

```bash
cat /proc/cmdline
cat /sys/module/amdgpu/parameters/ppfeaturemask
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
grep '\*' /sys/class/drm/card*/device/pp_power_profile_mode
cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap
findmnt /dev/shm /kvcache
sudo lspci -vv -d 1002:7551 | grep 'Region 0'      # BAR size; the link speed here is the VM's virtual port
```

On a TrueNAS host:

```bash
cat /proc/cmdline
sudo midclt call system.advanced.config | jq '{kernel_extra_options, isolated_gpu_pci_ids}'
sudo lspci -vv -d 1002:7551 | grep -E 'LnkCap:|LnkSta:'   # the physical link
```
