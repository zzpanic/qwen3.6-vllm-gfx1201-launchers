# Qwen3.6 / Qwen3.8 vLLM launchers for AMD gfx1201 (Radeon AI PRO R9700)

Standalone podman/vLLM startup scripts for serving Qwen3.8-27B and Qwen3.6-27B (dense) and
Qwen3.6-35B-A3B (MoE) int4 on a single 32 GiB AMD Radeon AI PRO R9700 (gfx1201, RDNA4),
using the `docker.io/stilldeadcode/vllm-radiance` image (ROCm + AITER + a few
gfx1201-specific perf hooks) — `:0.9.3` for the 3.8 script, `:0.5.8` for the 3.6 ones. See
[BACKGROUND.md](BACKGROUND.md) for what that image is, which of the decisions here are
ours rather than its defaults, and the measurements behind each of them.

Extracted from a homelab multi-model setup normally driven by
[llama-swap](https://github.com/mostlygeek/llama-swap); these scripts have no dependency
on it and run standalone. Every config decision baked into them (kernel gate to check in
the boot log, KV cache budgeting, why MTP is on for one model and off for the other,
measured speculative-decode tuning, checkpoint defects that silently break things) is
explained inline in each script's header comment, since most of it isn't documented
anywhere else.

## Qwen3.8-27B: what it actually does on this card

Qwen3.8-27B is a **retrained release of the Qwen3.6-27B architecture**, not a new one —
same `model_type qwen3_5`, same 64 layers / 5120 hidden / 17408 intermediate / 24:4 heads /
head_dim 256 / vocab 248320. So every tuning decision here transfers unchanged, including
the W4A16 tile table in `patches/` (keyed on GEMM shape, not model name — see below).

Checkpoint: [`devan-carlin/Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound),
17.93 GiB, int4 W4A16 gs=128, **int4 MTP draft head**.

### Throughput

`MAXLEN=204800`, `MAXSEQS=2`, **DFlash2 ×4**, fp8 KV, single stream. Re-captured
2026-08-29 against the live server, after the `KV_GROUP_SIZE=auto` group-padding fix and
the W4A16 tile gap-fill.

| prompt depth | prefill tok/s | decode tok/s | accepted draft len | engine steps/s |
|---|---|---|---|---|
| 3,183 | 1776.6 | 59.94 | 2.81 | 21.31 |
| 25,565 | 1657.4 | 69.44 | 3.37 | 20.61 |
| 76,552 | 1426.4 | 48.54 | 2.49 | 19.53 |

`steps/s` is the low-noise metric — it's counted, not derived. Decode and accepted-length
carry roughly ±6% run-to-run noise, and the 25,565 rung above is a good illustration:
its decode reads *higher* than the shallow rung, which is not a real depth effect, it is
acceptance variance (3.37 vs 2.81) moving tokens-per-step around. **Read the step rate,
not tokens/sec** — the step rate declines monotonically with depth, as it should.

Against the same benchmark on 2026-08-22 (pre group-padding, depths 3,183 / 12,520 /
38,997 → 1685.2 / 1605.8 / 1378.1 prefill, 20.72 / 20.05 / 18.60 steps/s): prefill is up
~5% at the shallow rung and the step rate up ~3%, at depths that are deeper rather than
shallower. Both passes are single captures, so treat the direction as sound and the
magnitude as soft.

Load-time facts to check against your own boot log: **`Model loading took 18.91 GiB`**
(target 17.93 + draft ~1.08, reported as one figure) and **`GPU KV cache size: 269,837
tokens`** = 1.32× concurrency at `MAXLEN=204800`, from `Available KV cache memory:
9.62 GiB`. Both lines are in `logs/boot-qwen3.8-27b-vllm-0.9.3.log`. If your own boot reads
~250k instead, you are running `KV_GROUP_SIZE=5` and leaving ~8% of the pool on the floor;
see TUNING.md.

**First boot after any change to the image, `MAXLEN`, or the model graph may fail**, with
`ValueError: ... estimated maximum model length is 198016`. torch.compile runs cold on
that boot and transiently costs ~2.28 GiB of the pool. Run it again — the second boot
loads the compiled graph from cache and succeeds. Nothing needs changing.

#### BetterBench, production config, live endpoint

[BetterBench](https://github.com/GGZ14/BetterBench) run against the running server with the
production sampler, 4 runs per category, cold prefix cache. Full report and raw JSON in
`benchmarks/live-20260829/`.

Single stream, by workload category. ITL columns are tokens/sec: **1% low** is the stutter
floor, median is the typical rate.

| category | TTFT p50 (ms) | prefill t/s (med) | ITL 1% low | ITL median | decode t/s (med) | CV |
|---|--:|--:|--:|--:|--:|--:|
| chat | 166.4 | 852.0 | 45.8 | 71.7 | 68.7 | 18.0% |
| code | 137.7 | 716.8 | 68.6 | 84.6 | 85.7 | 3.0% |
| file_edit | 169.4 | 797.1 | 86.5 | 98.4 | 99.4 | 0.8% |
| json | 140.2 | 748.9 | 57.3 | 80.3 | 89.7 | 12.0% |
| math | 136.4 | 668.7 | 74.9 | 91.9 | 93.9 | 5.9% |
| prose | 136.5 | 700.9 | 47.1 | 55.9 | 57.6 | 4.8% |
| reasoning | 137.5 | 719.9 | 47.1 | 60.1 | 70.9 | 22.8% |
| summarization | 171.2 | 826.7 | 70.6 | 79.7 | 79.9 | 7.0% |

Weighted combined: **decode ≈ 79.9 tok/s**, ITL 1%-low ≈ 61.4 tok/s, **TTFT p50 ≈ 144 ms**.
The prefill column here is lower than the depth ladder above because these prompts are short
— prefill rate climbs with depth until the pool starts to bite, which is what the sweep below
shows.

Prefill against input depth (tiny decode, cold cache):

| target depth | prompt tokens (med) | TTFT p50 (ms) | prefill t/s (median) |
|--:|--:|--:|--:|
| 2,000 | 1,544 | 946.0 | 1632.4 |
| 8,000 | 5,948 | 3,424.8 | **1736.8** |
| 16,000 | 11,824 | 6,967.7 | 1696.8 |
| 32,000 | 23,573 | 14,448.7 | 1631.5 |
| 64,000 | 47,086 | 30,759.9 | 1530.7 |

Prefill peaks near 8k and has lost only **12%** by 47k tokens of context. The 1%-low to
99%-high spread is under one tok/s at every rung — prefill on this card is essentially
noiseless, unlike decode.

Concurrency, at the shipped `MAXSEQS=2`:

| concurrent requests | aggregate t/s | per-request decode t/s | TTFT p50 (ms) |
|--:|--:|--:|--:|
| 1 | 74.9 | 87.6 | 139.5 |
| 2 | 113.9 | 68.4 | 187.5 |

**1.52× aggregate throughput for 0.78× per-request speed.** Two streams is worth it if you
have two streams; it is not free.

### Speculative decoding, in one paragraph

The default is **DFlash2 ×4** with the int4 draft checkpoint sampled probabilistically. That
combination decodes **1.44× faster than the target's own MTP head** (19.07 vs 14.03 engine
steps/s at matched depth) and costs **8.2% of the KV pool**, because the draft model is
resident. `SPEC=mtp` remains supported and drops the draft model entirely. `SPECTOK=4` is a
measured **workload** choice, not a universal one: over K ∈ {4,5,6,7} aggregate decode barely
moves, but math gains 13.9% and chat loses 8.9% going to K=7. A math- or code-heavy server
should run K=7. Draft-checkpoint comparison, the K tables, and the retracted "K≥7 doesn't
fit" claim are in [BACKGROUND.md](BACKGROUND.md#dflash2-versus-mtp) and TUNING.md.

### Quality

Measured against the **live server through this exact config** — fp8 KV, speculative
decoding, thinking template — not against the weights in isolation.

| | **this checkpoint** | AMD Quark-Qronos | AMD Quark-AWQ | Qwen BF16 base |
|---|---|---|---|---|
| GSM8K 5-shot (thinking), DFlash2 | **93.71%** | — | — | — |
| GSM8K 5-shot (thinking), MTP | **94.77%** | 94.62% | 91.21% | 93.33% |
| BFCL v4 overall (single_turn) | **25.29%** | 23.80% | 24.06% | 24.38% |

Single runs, no seeds, and not a like-for-like reproduction of AMD's harness — the 0.15 pp
over Qronos is not a win, and both GSM8K figures are temperature-1.0 (greedy scores ~97.6%
on the same harness). The DFlash2-vs-MTP gap is 1.17σ on 1319 problems, i.e. not
distinguishable. Per-category BFCL, the caveats in full, and the one generalisable takeaway
— **the standard AutoRound recipe transfers across uploaders**, beating two AMD checkpoints
built with more sophisticated algorithms — are in [BACKGROUND.md](BACKGROUND.md#quality-in-full).

### Checkpoint choice, and the image

There is **no Intel AutoRound release for 3.8**; every other int4 candidate was rejected for
a specific measured reason (BF16 MTP heads, gs=32 off the tuned table, or simply not fitting
the card). Verify you got the right build at load: `Model loading took 17.93 GiB`, or
19.01 GiB with the DFlash2 draft included. The rejection table is in
[BACKGROUND.md](BACKGROUND.md#why-not-the-other-38-int4-checkpoints).

These scripts run a third-party ROCm image (`vllm-radiance`) whose own tested envelope is
FP8 weights on two cards; this repo runs int4 on one, i.e. **outside it** — so nothing here
should be read as a defect report against that image. The full divergence table, one
divergence per row with the measurement behind it, is in
[BACKGROUND.md](BACKGROUND.md#the-image-these-build-on-and-where-this-repo-diverges).

## Contents

- `startup-qwen3.8-27b-vllm.sh` — dense 27B, int4 W4A16, **DFlash2 speculative decoding**
  on (with `SPEC=mtp` as a supported fallback), vision, 204800 context,
  `REASONING_EFFORT` knob.
- `startup-qwen3.6-27b-vllm.sh` — dense 27B, int4 W4A16, MTP speculative decoding on,
  vision (images only -- video untested, see TUNING.md), 131072 context.
- `startup-qwen3.6-35b-vllm.sh` — MoE 35B-A3B (256 experts), int4 W4A16, MTP off
  (KV cost doesn't pay off at this size), vision, 262144 context.
- `logs/boot-qwen3.8-27b-vllm-0.9.3.log` — **the current configuration**: `:0.9.3`, R4D
  attention, `KV_GROUP_SIZE=auto`, `BATCHTOK=4854`, `GPUUTIL=0.97`, DFlash2 ×4 with the int4
  draft. Container start through `Application startup complete` plus the first inference JIT
  warnings and the first `SpecDecoding metrics` line, captured 2026-08-29. Worth grepping
  for: `Using RDNAHybridW4A16LinearKernel` (twice — once for the target's AutoGPTQ path,
  once for the draft's compressed-tensors path), `Using R4D backend`, `libr4d 0.5.0, 19
  kernels built, 12 of 12 queries resolved`, `Model loading took 18.91 GiB`,
  `KV_GROUP_SIZE=auto -> group_size 8 for layer buckets [48, 16, 5]`, and
  `GPU KV cache size: 269,837 tokens`.
- `logs/boot-qwen3.8-27b-vllm-dflash2.log` — the same for the **previous** `:0.5.8` +
  back-ported-DFlash2 configuration, captured 2026-08-22, before the group-padding fix:
  `Model loading took 19.01 GiB`, `GPU KV cache size: 250,148 tokens`, `Capturing dflash2
  CUDA graphs`, `Mean acceptance length: 3.00`.
- `logs/boot-qwen3.8-27b-vllm.log` — the same for the **MTP** configuration, captured
  2026-08-15: `Model loading took 17.93 GiB`, `GPU KV cache size: 249,982 tokens`.
  All three 3.8 logs were captured from the homelab llama-swap launcher rather than these
  scripts, so their `non-default args` line carries three extra metrics flags
  (`enable_prompt_tokens_details`, `enable_per_request_metrics`,
  `enable_force_include_usage`) that these scripts don't set, and client IPs are redacted to
  `CLIENT`. Nothing else differs.
- `logs/boot-qwen3.6-27b-vllm.log` / `logs/boot-qwen3.6-35b-vllm.log` — same, for the 3.6
  models, captured 2026-08-09.
- `patches/` — a per-shape override table for the 27B scripts' W4A16 kernel, whose stock
  gfx1201 Triton tile heuristic is tuned on a different model's shapes and group_size.
  Measured **+4.6% to +9.6% real end-to-end prefill tokens/sec** and **+4.3% decode step
  rate**. Applied by default (`TUNED_TILES=1`). Also carries a symmetric-GPTQ zero-point
  bug fix the AutoRound checkpoints need to run at all. Full sweep data, runtime shape
  census, generator scripts, and methodology in `patches/README.md`.

  **The same table serves 3.6 and 3.8.** It is keyed `(group_size, K, N, M-bucket)` — GEMM
  shape, not model name — and 3.8 is the same architecture at the same group_size, so every
  fused shape is identical. Re-confirmed by a runtime kernel census on a 3.8 load under
  DFlash2, then gap-filled on 2026-08-29: **100.00% of kernel work lands on tuned rows**
  (0.61% uncovered before). Bucket 64 had been entirely empty and the DFlash2 drafter GEMM
  `K25600xN5120` had no row at any bucket. Census, sweep and the measurement trap are in
  `benchmarks/w4a16-census-20260829/`. **This closed coverage, not throughput** — the
  uncovered work was inside the prefill noise floor, so expect benchmarks not to move.

- `patches/radiance-0.9.3/` — two patched **copies of the default image's own files**,
  mounted read-only on every 3.8 boot: `v1/core/kv_cache_utils.py` adds the `KV_GROUP_SIZE`
  knob (+21.2% KV pool, see the tuning log) and `model_executor/models/qwen3_dflash.py`
  lets merged-upstream DFlash2 load a *packed int4* draft checkpoint. Every local hunk is
  marked in-file. These are copies, so they are **version-bound**: bumping the image tag
  silently reverts that image's own fixes in these two files. Rebase by diffing against the
  new image's stock file, never by replaying the patch.

- `patches/dflash2/` — a ten-file back-port of vLLM **PR #52816** (DFlash2 speculative
  decoding) onto the vLLM 0.26.0 inside `vllm-radiance:0.5.8`. **Only needed on that
  image**, which the 3.8 script no longer defaults to: on the default `:0.9.3` DFlash2 is
  native, `DFLASH2_PATCH` defaults to 0, and the launcher refuses the combination outright
  because this port would land on top of theirs. Every file names
  the upstream commit it came from and any deviation. Provenance, the one mandatory
  non-obvious file, the three deliberate deviations, and the one function that is not
  upstream at all are all in `patches/dflash2/README.md`.

- `benchmarks/` — the raw numbers and the scripts that produced them: the production server
  as shipped (depth ladder, prefill sweep, concurrency, per-category BetterBench), the
  speculative-depth sweep over K ∈ {4,5,6,7}, and the W4A16 shape census. `benchmarks/README.md`
  explains how to read them and which ones are soft.

See [BACKGROUND.md](BACKGROUND.md) for the long-form reasoning: the decoder and
draft-checkpoint comparisons, the quality evals in full, the rejected checkpoints, and the
divergence table against the underlying image.

See [TUNING.md](TUNING.md) for the dated tuning log: what each non-obvious default
is set to, what it was measured against, and the traps found along the way.

## Running the 3.8

```
hf download devan-carlin/Qwen3.8-27B-int4-AutoRound --local-dir ./models/qwen3.8-27b-autoround
hf download syvai/Qwen3.8-27B-DFlash2-W4A16      --local-dir ./models/qwen3.8-27b-dflash2-int4
./startup-qwen3.8-27b-vllm.sh
```

The second download is the DFlash2 draft model. To run without it:

```
SPEC=mtp MAXLEN=131072 ./startup-qwen3.8-27b-vllm.sh
```

which is the configuration every MTP number in [BACKGROUND.md](BACKGROUND.md) was
measured on.

The DFlash2 knobs are `SPEC` (`dflash2` | `mtp` | `off` | `ngram`), `DRAFT_DIR`,
`DRAFT_SAMPLE_METHOD` (`probabilistic` | `greedy`), `DRAFT_ATTN`, and `DFLASH2_PATCH`.
Two of them have non-obvious reasons behind their defaults:

- **`DRAFT_ATTN=TRITON_ATTN`, which is deliberately not the target's backend.** DFlash2's
  draft attention is **non-causal** — it attends across all K mask rows — and
  `ROCM_AITER_UNIFIED_ATTN` refuses a non-causal mask. `TRITON_ATTN` takes it. The target
  keeps AITER; the two backends are set independently and both are in the boot log.
- **`DFLASH2_PATCH=1` is scoped to `SPEC=dflash2` on purpose.** One file in the overlay
  (`v1/core/kv_cache_utils.py`) is not DFlash2-gated and would also move MTP's prefix-hash
  granularity from 1664 to 832. Scoping it keeps `SPEC=mtp` byte-for-byte the
  configuration the MTP numbers describe.

The launcher validates the draft checkpoint before it starts the container: that the
directory exists, that its `config.json` declares `DFlash2DraftModel` (a plain Qwen3 draft
in that slot loads as a *generic* draft model, quietly, and drafts badly), and that a
quantized draft is **symmetric** (the context-KV precompute dequantizes scale-only, so an
asymmetric draft would corrupt silently and show up only as unexplained low acceptance).

The only knob that is new versus the 3.6 script is `REASONING_EFFORT` (default `low`;
`xhigh` | `medium` | `low` | empty). Three things about it are not obvious:

- It is a **prompt steer, not a decode cap**. `low` injects "keep your thinking brief…"
  into the system prompt. Nothing enforces it and it does not bound output length.
- `medium` sets **no instruction at all** — the template branches on `xhigh` and `low` only,
  so `medium` is identical to unsteered.
- It is a **server default only**: vLLM merges `--default-chat-template-kwargs` *under*
  request values, so a client sending its own `chat_template_kwargs` still wins.

The script hard-fails rather than warns if you point it at a checkpoint whose chat template
has no `reasoning_effort` variable (every 3.6), because the flag would otherwise be accepted
and silently do nothing — and it also validates the *value* against the template's own
accepted set at boot, since the template raises on an unknown one and that would be a 500 on
every chat request from a server that started up perfectly.

The AutoRound `config.json` routing patch described below applies verbatim to the 3.8 —
its `modules_in_block_to_quantize` list is byte-identical to Intel's — and is applied
automatically. The 3.8 checkpoint does **not** ship the tokenizer truncation defect; that
guard stays in place because the failure is silent, but it won't fire.

## Which 27B checkpoint (Qwen3.6)

Two are supported and both work. **Intel AutoRound is the default and the one to use.**

```
# recommended
hf download Intel/Qwen3.6-27B-int4-AutoRound --local-dir ./models/qwen3.6-27b-autoround
./startup-qwen3.6-27b-vllm.sh

# alternative (compressed-tensors, group_size 32)
hf download Avesed/Qwen3.6-27B-INT4-W4A16 --local-dir ./models/qwen3.6-27b-int4
MODEL_DIR=./models/qwen3.6-27b-int4 QUANT=compressed-tensors ./startup-qwen3.6-27b-vllm.sh
```

AutoRound uses sign-gradient-descent rounding rather than round-to-nearest, which is the
reason to prefer it — but **no quality benchmark was run here on the 3.6**, so treat that as
the published claim rather than a measurement made in this repo. (The 3.8 section above is
the first quality measurement in this repo, and it is on a different checkpoint.) What *was*
measured on this card, same launcher, everything else held constant:

| | AutoRound vs Avesed |
|---|---|
| KV cache pool | **+17.4%** — 255,387 vs 217,552 tokens at `MAXLEN=131072` (1.95× vs 1.66× concurrency) |
| prefill | +0.6% at ≥20k depth, +2.1% at 5k |
| engine steps/s | +6.3% (acceptance-controlled) |
| accepted draft length | −4.9% — its MTP head is int4 where Avesed's was BF16 |
| net decode wall-clock | a wash at ≥20k |

So the concrete, non-speculative reason to prefer it is the **materially larger KV pool**;
speed is roughly unchanged either way.

**The catch, and it is a real one:** AutoRound's shipped `config.json` needs a three-key
patch on this stack, and without it the model loads **silently wrong** rather than failing
— vLLM 0.26's INC backend hijacks any checkpoint declaring `quant_method: "auto-round"`,
ignores `--quantization` entirely, and lands on the wrong kernel; and a missing
`modules_in_block_to_quantize` makes the whole model load *unquantized*, also silently.
The startup scripts detect this and fix it for you (`CONFIG_FIX=1`, idempotent, keeps a
`config.json.autoround-orig` backup). **A re-download reinstates all three defects**, which
is exactly when that guard earns its keep. Full mechanism in each script's ROUTING GUARD
comment.

Whichever you use, confirm `RDNAHybridW4A16LinearKernel` appears in the boot log. On
AutoRound its absence means the config patch didn't take.


## Credit

Almost none of the code here was written for this repo. The launchers are shell wrappers;
the substance — the ROCm/gfx1201 kernel work, DFlash2, vLLM itself — is other people's, and
what this repo adds on top is measurement, configuration, and documentation of both.

- **`stilldeadcode`** — the [`vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance)
  image, and the gfx1201 kernel work in it. Bug reports about the *image* belong on its issue
  tracker there, not here.
- **The vLLM project** — everything under `patches/` is a derived work of vLLM,
  Apache-2.0, SPDX headers intact. None of it is original code. `patches/dflash2/` is
  upstream **PR #52816** back-ported; `patches/radiance-0.9.3/` is two of the radiance
  image's own vLLM 0.27.1 files with a marked local hunk each; `patches/rdna_hybrid_w4a16.py`
  is vLLM's own kernel with a per-shape override table added. Every file names the base it
  was copied from and comments each deviation in place.
- **The checkpoints** — [`devan-carlin/Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound)
  (the served weights) and [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
  (the DFlash2 draft model). Neither was quantized here; both are other people's work and
  carry their own licences on the Hub.
- **[BetterBench](https://github.com/GGZ14/BetterBench)** (Apache-2.0) — the benchmark
  harness behind `benchmarks/live-20260829/betterbench-full.*`. Not vendored; run against
  the live endpoint, output published verbatim.
- **[`BMorgan1296/qwen3.6-vllm-gfx1201-launchers`](https://github.com/BMorgan1296/qwen3.6-vllm-gfx1201-launchers)**
  — an independent adaptation of these launchers that got DFlash2 running on this card
  first. Its `_dense_kv_rows()` is the piece merged upstream doesn't have and without
  which no int4 DFlash2 draft loads at all; it's carried here, with two added guards, and
  attributed in the file. The rest of the overlay here was re-derived from the merged PR
  rather than taken from that snapshot, which predates the merge.

## Usage

All scripts are configured entirely through environment variables (see each header) with
working defaults; nothing needs to be edited to run as-is once weights are downloaded.

## Requirements

- rootless podman with ROCm/kfd device access
- an AMD gfx1201 card (this is tuned for the R9700 specifically — attention backend,
  MoE/kernel gates, and KV budgeting numbers are card-shape-specific and won't transfer
  as-is to other GPUs)
