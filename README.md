# Qwen3.6 / Qwen3.8 vLLM launchers for AMD gfx1201 (Radeon AI PRO R9700)

Standalone podman/vLLM startup scripts for serving Qwen3.8-27B and Qwen3.6-27B (dense) and
Qwen3.6-35B-A3B (MoE) int4 on a single 32 GiB AMD Radeon AI PRO R9700 (gfx1201, RDNA4),
using the `docker.io/stilldeadcode/vllm-radiance:0.5.8` image (ROCm + AITER + a few
gfx1201-specific perf hooks).

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

`MAXLEN=131072`, `MAXSEQS=2`, MTP ×4, fp8 KV, single stream. Two passes × three depths.

| prompt depth | prefill tok/s | decode tok/s | accepted draft len | engine steps/s |
|---|---|---|---|---|
| 3,183 | 1618.9 | 44.34 | 3.03 | 14.64 |
| 12,520 | 1546.7 | 39.81 | 2.77 | 14.40 |
| 38,997 | 1335.1 | 43.91 | 3.18 | 13.80 |
| **mean** | **1500.2** | **42.69** | **2.99** | **14.28** |

`steps/s` is the low-noise metric — it's counted, not derived. Decode and accepted-length
carry roughly ±6% run-to-run noise, so don't read them to three significant figures.

Load-time facts to check against your own boot log: **weights 17.93 GiB**, **GPU KV cache
249,982 tokens** = 1.91× concurrency at `MAXLEN=131072`.

**Against Qwen3.6-27B on the same launcher and the same rungs** (the 3.6 arm was captured
2026-08-11, not re-baselined in the same window as the 3.8 — treat the magnitude as soft):

| | 3.6-27B | 3.8-27B | ratio |
|---|---|---|---|
| prefill tok/s | 1513.2 | 1500.2 | 0.99 — a wash |
| engine steps/s | 16.61 | 14.28 | 0.86 |
| accepted draft len | 4.25 | 2.99 | 0.70 |
| decode tok/s | 71.2 | 42.7 | **0.60** |

**The 3.8 decodes materially slower, and it is not a missing optimisation.** All 13
performance env vars are byte-identical between the two launchers; the difference is
speculative-decode *acceptance* — the 3.8 checkpoint's MTP head drafts worse. The two
`mean_len` distributions don't overlap (3.8's best rung, 3.24, is below the 3.6's worst,
3.28), so the gap is real even where its size isn't precise. Whether that is Qwen 3.8 or
that particular quantizer's head is untested.

### Quality

Measured against the **live server through this exact config** (fp8 KV + MTP + thinking
template at `REASONING_EFFORT=low`), not against the weights in isolation.

| | **this checkpoint** | AMD Quark-Qronos | AMD Quark-AWQ | Qwen BF16 base |
|---|---|---|---|---|
| GSM8K 5-shot (thinking) | **94.77%** | 94.62% | 91.21% | 93.33% |
| BFCL v4 overall (single_turn) | **25.29%** | 23.80% | 24.06% | 24.38% |
| BFCL Non-Live AST | 87.46% | 85.17% | 86.58% | **88.52%** |
| BFCL Live AST | 81.87% | **82.09%** | 81.57% | **83.05%** |
| BFCL Relevance Detection | **75.00%** | 62.50% | **75.00%** | **75.00%** |
| BFCL Irrelevance Detection | **83.62%** | 70.79% | 72.47% | 72.22% |

BFCL per-category, ours: `simple_python 95.00 · simple_java 63.00 · simple_javascript
68.00 · multiple 95.50 · parallel 90.50 · parallel_multiple 88.50 · irrelevance 86.25 ·
live_simple 87.60 · live_multiple 80.44 · live_parallel 93.75 · live_parallel_multiple
75.00 · live_irrelevance 81.00 · live_relevance 75.00`. 3641 entries, `--temperature 0.001`,
`--num-threads 2`, 3 h 02 m. "Overall Acc" is ~25% rather than ~85% because this is
single-turn only — the multi-turn and agentic categories, which AMD also omits, drag the
aggregate down for everyone.

Read these honestly:

- **Not a like-for-like reproduction of AMD's harness.** They ran `--model vllm` in-process
  with `enforce_eager`, BF16 KV and no speculative decoding. These numbers went through the
  full production serving stack. That measures *the deployed path*, which is the useful
  thing here, but it is not the same experiment.
- **Single run, no seeds, no error bars.** GSM8K's ±0.61 is binomial SE only; the 0.15 pp
  over Qronos is not a win. Scoring above the BF16 base is a known temperature-1.0 artifact
  (AMD's Qronos shows it too), not evidence that quantization helped.
- **Distrust the irrelevance row specifically.** +11.4 pp over the BF16 base is not
  something int4 does. The likely cause is the thinking template talking the model out of
  spurious calls, i.e. a harness difference rather than a checkpoint difference.
- **The losses are real and they're Java/JavaScript typed literals** — `simple_java` 63.00%
  against `simple_python` 95.00%, dominated by `type_error:simple`. That is the whole of
  the gap to the BF16 base.
- Neither eval says anything about the acceptance deficit above. Both measure target-model
  output, and MTP rejection sampling is distributionally neutral.

The takeaway worth generalising: **the AutoRound recipe transfers across uploaders.** This
is an unknown author's checkpoint using AutoRound's standard settings (gs=128, symmetric,
`auto_round:auto_gptq`, the default `linear_attn.in_proj_a/b` fp16 exclusions), and it beats
two AMD-authored checkpoints built with more sophisticated algorithms — Qronos is
Hessian-based PTQ with a paper behind it. Prefer standard AutoRound over a clever algorithm
with a bespoke module list.

### Why not the other 3.8 int4 checkpoints

There is **no Intel AutoRound release for 3.8** (they published three for the 3.6). Of what
exists:

| candidate | why not |
|---|---|
| `amd/…Quark-Qronos-INT4-W4A16` | **BF16 MTP head** (all 15 `mtp.*` tensors plain BF16). A sibling 3.8 checkpoint with the same BF16 head was measured here at +0.51 GiB and ~9% slower decode; AMD's own were not downloaded, so that cost is inferred from the shape, not measured on theirs |
| `amd/…Quark-AWQ-INT4-W4A16` | same, BF16 MTP head |
| `goldhub/…INT4-W4A16-AutoRound` | gs=32 → clamps `BLOCK_K` to 32, off the tuned gs=128 table; also 26.37 GiB |
| `Israeli-AI/…MTP-W4A16` | 25.77 GiB, mostly unquantized, BF16 MTP head |
| `Qwen/Qwen3.8-27B-FP8` (official) | 28.75 GiB + 3.29 GiB measured overhead = **overruns the 31.86 GiB card before a single KV byte** |
| NVFP4 family | no gfx1201 kernel |

Verify you got the int4-head build at load: `Model loading took 17.93 GiB`. An 18.4-ish GiB
figure means a BF16-head checkpoint.

## Contents

- `startup-qwen3.8-27b-vllm.sh` — dense 27B, int4 W4A16, MTP speculative decoding on,
  vision, 131072 context, `REASONING_EFFORT` knob.
- `startup-qwen3.6-27b-vllm.sh` — dense 27B, int4 W4A16, MTP speculative decoding on,
  vision, 131072 context.
- `startup-qwen3.6-35b-vllm.sh` — MoE 35B-A3B (256 experts), int4 W4A16, MTP off
  (KV cost doesn't pay off at this size), vision, 262144 context.
- `logs/boot-qwen3.6-27b-vllm.log` / `logs/boot-qwen3.6-35b-vllm.log` — real boot logs
  from container start through `Application startup complete`, captured 2026-08-09.
- `patches/` — a per-shape override table for the 27B scripts' W4A16 kernel, whose stock
  gfx1201 Triton tile heuristic is tuned on a different model's shapes and group_size.
  Measured **+4.6% to +9.6% real end-to-end prefill tokens/sec** and **+4.3% decode step
  rate**. Applied by default (`TUNED_TILES=1`). Also carries a symmetric-GPTQ zero-point
  bug fix the AutoRound checkpoints need to run at all. Full sweep data, runtime shape
  census, generator scripts, and methodology in `patches/README.md`.

  **The same table serves 3.6 and 3.8.** It is keyed `(group_size, K, N, M-bucket)` — GEMM
  shape, not model name — and 3.8 is the same architecture at the same group_size, so every
  fused shape is identical. Re-confirmed by a runtime kernel census on a 3.8 load: **99.45%
  of kernel work lands on tuned rows.** Nothing to re-sweep.

## Running the 3.8

```
hf download devan-carlin/Qwen3.8-27B-int4-AutoRound --local-dir ./models/qwen3.8-27b-autoround
./startup-qwen3.8-27b-vllm.sh
```

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

## Usage

All scripts are configured entirely through environment variables (see each header) with
working defaults; nothing needs to be edited to run as-is once weights are downloaded.

## Requirements

- rootless podman with ROCm/kfd device access
- an AMD gfx1201 card (this is tuned for the R9700 specifically — attention backend,
  MoE/kernel gates, and KV budgeting numbers are card-shape-specific and won't transfer
  as-is to other GPUs)
