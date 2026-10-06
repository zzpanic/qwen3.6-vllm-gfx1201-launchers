# Qwen3.8-27B on one AMD Radeon AI PRO R9700

One launcher, `startup-qwen3.8-27b-mxfp4.sh`, serves Qwen3.8-27B in native MXFP4 with a DFlash2
speculative drafter on a single R9700 (gfx1201 / RDNA4, 32 GB). It has a 262,144-token context, and
its prefix cache spills from the GPU to RAM to disk, so a returning conversation is restored rather
than recomputed.

**Decode 129.7 t/s · 2 clients 202.0 t/s aggregate · prefill 3,025 t/s at 2k, 2,185 t/s at 128k** —
[BETTERBENCH-20261006.md](BETTERBENCH-20261006.md)

## Where it comes from

- **Built on ggz14's [radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4)** (GitHub
  mirror [GGZ14/vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4)) and the
  `stilldeadcode/vllm-radiance:0.9.3` image. Those provide the MXFP4 GEMM, the R4D attention path and
  the DFlash2 integration.
- **Kernel changes are pulled from several forks**: deadcode's radiance engine, Crssz/r4dx, J-Nova,
  SlyBase and vllm-radiance pull requests. They are gathered in [`r4d_kernels/`](r4d_kernels/);
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
| Host RAM | 40 GiB recommended, 32 GiB minimum: the RAM tier is 19 GiB, pinned |
| `/dev/shm` | At least 20 GiB |
| Disk tier | A dedicated filesystem of **128 GiB or more**. Optional: without it the launcher serves RAM-only. |
| Downloads | About 40 GiB for the checkpoints |

## Setup

**1. Clone the repositories.** ggz14's tree goes next to the launcher, under exactly this name.

```bash
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers
git clone https://github.com/GGZ14/vllm-mxfp4 radiance-vllm-mxfp4
```

**2. Download the models into `~/models`.** ggz14's `setup-mxfp4.sh` fetches AMD's MXFP4 checkpoint,
rewrites its MTP head to fp8, and fetches the DFlash2 drafter. The second model is optional: it is an
uncensored sibling of the same architecture that ships its MTP head already in fp8.

```bash
(cd radiance-vllm-mxfp4 && ./setup-mxfp4.sh)
hf download just1moremodel/Qwen3.8-27B-Uncensored-MXFP4-awq \
  --local-dir ~/models/Qwen3.8-27B-Uncensored-MXFP4-awq
```

**3. Size `/dev/shm` for the RAM tier.**

```bash
echo 'tmpfs /dev/shm tmpfs rw,nosuid,nodev,inode64,size=28G 0 0' | sudo tee -a /etc/fstab
sudo mount -o remount,size=28G /dev/shm
```

**4. Set up the disk tier and its reaper.** The reaper is required: the tier never deletes on its own.
Skip this step and set `KVCACHE_DISK_TIER=0` to run RAM-only.

```bash
# a dedicated filesystem of 128 GiB or more, mounted at /kvcache (kv-cache/docs/SETUP.md has fstab lines)
sudo cp kv-cache/ops/kvcache-reap.sh /usr/local/bin/
sudo cp kv-cache/ops/kvcache-reap.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now kvcache-reap.timer
```

**5. Check the configuration.** This prints the container command and runs nothing. The log lines
tagged `[kvcache]` show the RAM tier and disk tier the launcher chose.

```bash
DRY_RUN=1 ./startup-qwen3.8-27b-mxfp4.sh
```

## Serving both models with llama-swap

```yaml
healthCheckTimeout: 900   # the first boot builds kernels and caches, about 8 minutes; later boots about 3

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
  can leave the container holding the GPU and the 19 GiB `/dev/shm` region.
- **One model at a time.** The models swap; only one fits on the card.
- **Separate disk caches.** Each model gets its own disk-tier directory (`/kvcache/blocks/<model>`), so
  the two can never load each other's cached KV.

Everything else is a launcher default, and the defaults are the configuration measured here.
`./startup-qwen3.8-27b-mxfp4.sh -h` lists every setting.

## Checking it works

```bash
curl -s localhost:<port>/v1/models | python3 -m json.tool        # max_model_len: 262144
python3 kv-cache/tools/lenprobe.py --base http://127.0.0.1:<port> --model qwen3.8-27b --block 1648 --spec 7
python3 kv-cache/tools/kvtable.py --url http://127.0.0.1:<port>/metrics   # hits per tier
```

`logs/bootlog-first.txt` and `logs/bootlog-second.txt` show what a first (cold) boot and a second
(warm) boot print.

## Licence

Apache-2.0 for this repository's own work ([LICENSE](LICENSE)). Some third-party parts publish no
licence and are redistributed with attribution; [KERNEL.md](KERNEL.md) and the patch headers say which.
