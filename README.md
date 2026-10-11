# Qwen3.8-27B on one AMD Radeon AI PRO R9700

One launcher, `startup-qwen3.8-27b-mxfp4.sh`, serves Qwen3.8-27B in native MXFP4 with a DFlash2
speculative drafter on a single R9700 (gfx1201 / RDNA4, 32 GB). It has a 262,144-token context, and
its prefix cache spills from the GPU to RAM to disk, so a returning conversation is restored rather
than recomputed.

**Decode 138.0 t/s · 4 clients 333.7 t/s aggregate · prefill 3,167 t/s at 12k, 2,223 t/s at 132k** —
[BETTERBENCH-20261011.md](BETTERBENCH-20261011.md)

## Where it comes from

- **Built on ggz14's [radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4)** and the
  `stilldeadcode/vllm-radiance:0.9.3` image. Those provide the MXFP4 GEMM, the R4D attention path and
  the DFlash2 integration.
- **Kernel changes are pulled from several forks**: deadcode's radiance engine, Crssz/r4dx, J-Nova,
  SlyBase, tcclaviger and vllm-radiance pull requests. They are gathered in [`r4d_kernels/`](r4d_kernels/)
  and [`kv-cache/patches/`](kv-cache/patches/);
  [KERNEL.md](KERNEL.md) explains each one and its licence.
- **The startup cache ([`startup-cache/`](startup-cache/)) and the KV-cache offload
  ([`kv-cache/`](kv-cache/), [KV-CACHE.md](KV-CACHE.md)) are this repository's own work.**

The rest of the documentation:
- [SYSTEM.md](SYSTEM.md): the reference machine, kernel parameters and GPU power settings.
- [ROADMAP.md](ROADMAP.md): what is still open.
- [`old_work/`](old_work/): earlier launchers and notes.

## Requirements

| What | Needed |
|---|---|
| GPU | One AMD Radeon AI PRO R9700 |
| Linux | `amdgpu` loaded, `/dev/kfd` and `/dev/dri` present |
| Container runtime | Podman or Docker |
| Host RAM | 32 GiB: the RAM tier is 16 GiB, pinned, plus ~15 GiB for the engine and OS |
| `/dev/shm` | At least 20 GiB |
| Disk tier | **About 96 GB of cache**, held there by the launcher's built-in reaper: a folder on any fast disk with a size limit (`KVCACHE_MAX_GB`), or a 128 GiB filesystem or image file of its own (`setup-kvcache.sh` makes either). On by default: if the folder is missing the launcher refuses to start; `KVCACHE_DISK_TIER=0` runs RAM-only. |
| Downloads | About 40 GiB for the checkpoints |

## Setup

**1. Clone the repositories.** ggz14's tree goes next to the launcher, under exactly this name, at the
commit everything here was measured on (the GitHub mirror does not carry it).

```bash
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers
git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4
git -C radiance-vllm-mxfp4 checkout 980f891
```

**2. Download the models into `~/models`.** ggz14's `setup-mxfp4.sh` fetches AMD's MXFP4 checkpoint,
rewrites its MTP head to fp8, and fetches the FP8 DFlash2 drafter. The launcher's default drafter is the
4-bit one (`syvai/Qwen3.8-27B-DFlash2-W4A16`, 0.78 GiB less VRAM at the same acceptance; KERNEL.md), so
fetch that too, or set `DRAFTER=~/models/Qwen3.8-27B-DFlash2-FP8`. The uncensored model is optional: a
sibling of the same architecture that ships its MTP head already in fp8.

```bash
(cd radiance-vllm-mxfp4 && ./setup-mxfp4.sh)
hf download syvai/Qwen3.8-27B-DFlash2-W4A16 --local-dir ~/models/Qwen3.8-27B-DFlash2-W4A16
hf download just1moremodel/Qwen3.8-27B-Uncensored-MXFP4-awq \
  --local-dir ~/models/Qwen3.8-27B-Uncensored-MXFP4-awq
```

**3. Set up the KV-cache tiers.** `./setup-kvcache.sh` asks a few questions and does the rest:
- it measures your RAM and suggests a RAM-tier size (16 GiB by default), then resizes `/dev/shm` to match;
- it measures your disk's direct-I/O speed, then uses an existing mount, creates an image file, or formats a
  spare disk for the disk tier;
- it sets the size limit for the disk cache (96 GB recommended). The tier never deletes on its own; the
  launcher runs a reaper inside the container that holds it to that limit, so nothing is installed for it.

It shows every command it runs as root and asks first; `--dry-run` only prints them. Your choices go to
`kvcache.env`, which the launcher reads. Or skip the disk tier and run RAM-only with `KVCACHE_DISK_TIER=0`.
kv-cache/docs/SETUP.md has the same steps by hand.

```bash
./setup-kvcache.sh
```

> **Upgrading from a release before 2026-10-08: the fp16 SSM default invalidates your disk tier.**
> fp16 GDN state (now the default, `MAMBA_SSM_FP16=1`) changes the KV block geometry (1,648 -> 880
> tokens), and the tier's on-disk layout is fixed when it is first written. The launcher therefore keeps
> fp16 blocks in a separate tree (`blocks/<model>-f16ssm`), and the old fp32 tree (`blocks/<model>`) is
> never read again. You may wish to wipe it to get the space back:
>
> ```bash
> rm -rf /kvcache/blocks/Qwen3.8-27B-MXFP4-mtpfp8      # the old fp32 tree; cache only, nothing is lost
> ```
>
> If you pin `KVOFF_DISK_SUBDIR` / `KVCACHE_DISK_SUBDIR` yourself, the launcher cannot separate them: wipe
> that directory whenever you switch `MAMBA_SSM_FP16`, or the tier will be read with the wrong geometry.
>
> **Upgrading from a release before 2026-10-11:** canonical prefill chunks (cached == uncached) write a
> new tree, `blocks/<model>-f16ssm-canon`. Rows in the old `blocks/<model>-f16ssm` were computed on a
> different chunk grid and are never read again; wipe it to get the space back.

**4. Check the configuration.** This prints the container command and runs nothing. The log lines
tagged `[kvcache]` show the RAM tier and disk tier the launcher chose.

```bash
DRY_RUN=1 ./startup-qwen3.8-27b-mxfp4.sh
```

## Serving both models with llama-swap

```yaml
healthCheckTimeout: 900   # the first boot builds kernels and caches, about 7.5 minutes; later boots about 2.5

models:
  "qwen3.8-27b":
    cmd: /path/to/qwen3.6-vllm-gfx1201-launchers/startup-qwen3.8-27b-mxfp4.sh --port ${PORT}
    cmdStop: podman stop -t 30 qwen38-27b
    env:
      - "NAME=qwen38-27b"
      - "SERVED=qwen3.8-27b"
    proxy: "http://127.0.0.1:${PORT}"
    checkEndpoint: "/health"
    ttl: 0

  "qwen3.8-27b-heretic":
    cmd: /path/to/qwen3.6-vllm-gfx1201-launchers/startup-qwen3.8-27b-mxfp4.sh --port ${PORT}
    cmdStop: podman stop -t 30 qwen38-27b-heretic
    env:
      - "NAME=qwen38-27b-heretic"
      - "SERVED=qwen3.8-27b-heretic"
      - "SNAP=/home/you/models/Qwen3.8-27B-Uncensored-MXFP4-awq"
    proxy: "http://127.0.0.1:${PORT}"
    checkEndpoint: "/health"
    ttl: 0
```

How the two entries fit together:
- **`cmdStop`** stops the container by name, so the name must match `NAME`. Otherwise a killed launcher
  can leave the container holding the GPU and the 16 GiB `/dev/shm` region.
- **One model at a time.** The models swap; only one fits on the card.
- **Separate disk caches.** Each model gets its own disk-tier directory under `/kvcache/blocks/`, named for the
  checkpoint and its numerics, so the two can never load each other's cached KV.

Everything else is a launcher default, and the defaults are the configuration measured here.
`./startup-qwen3.8-27b-mxfp4.sh -h` lists every setting.

## Checking it works

```bash
curl -s localhost:<port>/v1/models | python3 -m json.tool        # max_model_len: 262144
python3 kv-cache/tools/lenprobe.py --base http://127.0.0.1:<port> --model qwen3.8-27b --block 880 --spec 7
python3 kv-cache/tools/kvtable.py --url http://127.0.0.1:<port>/metrics   # hits per tier
```

`logs/bootlog-first.txt` and `logs/bootlog-second.txt` show what a first (cold) boot and a second
(warm) boot print. About 20-30 seconds after `/health` answers, the log shows one `[radiance] jit warm-up` line:
a few tiny requests that compile or load the remaining Triton kernels before real traffic (`JIT_WARMUP=0` turns
it off).

## Clearing the caches

`./clear-cache.sh` (also option 4 of `./setup-kvcache.sh`) lists the KV-offload disk tier (one entry per model and numerics) and the startup
caches (compile trees, libr4d builds, boot overlays) with their sizes, and wipes all of them, the ones
you pick, or everything except the ones you pick. It is all cache: the cost is recompute time, never
data. It skips anything a running container has mounted; `--dry-run` deletes nothing.

## Licence

Apache-2.0 for this repository's own work ([LICENSE](LICENSE)). Some third-party parts publish no
licence and are redistributed with attribution; [KERNEL.md](KERNEL.md) and the patch headers say which.
