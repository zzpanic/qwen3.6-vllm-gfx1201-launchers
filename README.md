# Qwen3.6 vLLM launchers for AMD gfx1201 (Radeon AI PRO R9700)

Standalone podman/vLLM startup scripts for serving Qwen3.6-27B (dense) and
Qwen3.6-35B-A3B (MoE) int4 on a single 32 GiB AMD Radeon AI PRO R9700 (gfx1201, RDNA4),
using the `docker.io/stilldeadcode/vllm-radiance:0.5.8` image (ROCm + AITER + a few
gfx1201-specific perf hooks).

Extracted from a homelab multi-model setup normally driven by
[llama-swap](https://github.com/mostlygeek/llama-swap); these scripts have no dependency
on it and run standalone. Every config decision baked into them (kernel gate to check in
the boot log, KV cache budgeting, why MTP is on for one model and off for the other,
measured speculative-decode tuning, a checkpoint defect that silently breaks vision) is
explained inline in each script's header comment, since most of it isn't documented
anywhere else.

## Contents

- `startup-qwen3.6-27b-vllm.sh` — dense 27B, int4 W4A16, MTP speculative decoding on,
  vision, 131072 context.
- `startup-qwen3.6-35b-vllm.sh` — MoE 35B-A3B (256 experts), int4 W4A16, MTP off
  (KV cost doesn't pay off at this size), vision, 262144 context.
- `logs/boot-qwen3.6-27b-vllm.log` / `logs/boot-qwen3.6-35b-vllm.log` — real boot logs
  from container start through `Application startup complete`, captured 2026-08-09.

## Usage

```
hf download Avesed/Qwen3.6-27B-INT4-W4A16 --local-dir ./models/qwen3.6-27b-int4
./startup-qwen3.6-27b-vllm.sh
```

Both scripts are configured entirely through environment variables (see each header) with
working defaults; nothing needs to be edited to run as-is once weights are downloaded.

## Requirements

- rootless podman with ROCm/kfd device access
- an AMD gfx1201 card (this is tuned for the R9700 specifically — attention backend,
  MoE/kernel gates, and KV budgeting numbers are card-shape-specific and won't transfer
  as-is to other GPUs)
