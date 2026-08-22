# Qwen3.6 / Qwen3.8 vLLM launchers for AMD gfx1201 (Radeon AI PRO R9700)

Standalone podman/vLLM startup scripts for serving Qwen3.8-27B and Qwen3.6-27B (dense) and
Qwen3.6-35B-A3B (MoE) int4 on a single 32 GiB AMD Radeon AI PRO R9700 (gfx1201, RDNA4),
using the `docker.io/stilldeadcode/vllm-radiance:0.5.8` image (ROCm + AITER + a few
gfx1201-specific perf hooks). See
[The image these build on, and where this repo diverges](#the-image-these-build-on-and-where-this-repo-diverges)
for what that image is and which of the decisions here are ours rather than its defaults.

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

`MAXLEN=204800`, `MAXSEQS=2`, **DFlash2 ×4**, fp8 KV, single stream. Captured 2026-08-22.

| prompt depth | prefill tok/s | decode tok/s | accepted draft len | engine steps/s |
|---|---|---|---|---|
| 3,183 | 1685.2 | 62.40 | 3.01 | 20.72 |
| 12,520 | 1605.8 | 61.10 | 3.05 | 20.05 |
| 38,997 | 1378.1 | 56.68 | 3.05 | 18.60 |
| 77,851 | 1151.3 | 47.62 | 2.84 | 16.74 |

`steps/s` is the low-noise metric — it's counted, not derived. Decode and accepted-length
carry roughly ±6% run-to-run noise: a second pass of this identical config read
`56.42 / 54.50 / 52.78 / 51.71` decode against step rates that matched to three figures.
**Read the step rate, not tokens/sec.**

Load-time facts to check against your own boot log: **`Model loading took 19.01 GiB`**
(target 17.93 + draft ~1.08, reported as one figure) and **`GPU KV cache size: 250,148
tokens`** = 1.22× concurrency at `MAXLEN=204800`. Compare against
`logs/boot-qwen3.8-27b-vllm-dflash2.log`, which is a real boot of this configuration.

**First boot after any change to the image, `MAXLEN`, or the model graph may fail**, with
`ValueError: ... estimated maximum model length is 198016`. torch.compile runs cold on
that boot and transiently costs ~2.28 GiB of the pool. Run it again — the second boot
loads the compiled graph from cache and succeeds. Nothing needs changing.

#### DFlash2 versus MTP

Both are speculative decoding and both go through the same rejection sampler. MTP uses a
draft head inside the target checkpoint and walks a **sequential** chain of K forwards;
DFlash2 is a **separate draft model** that proposes all K tokens in **one parallel**
forward. Measured at matched depth, same window, same card, both at `MAXLEN=131072`:

| | MTP ×4 | DFlash2 ×4 | ratio |
|---|---|---|---|
| engine steps/s | 14.03 | 19.07 | 1.36 |
| accepted draft len | 2.83 | 3.01 | 1.07 |
| decode tok/s | 39.74 | 57.31 | **1.44** |

The cheaper step is most of it; the slightly longer accepted draft is the rest. Prefill
isn't in that table because speculation only runs in decode — the two arms' prefill agreed
inside noise.

**What it costs is 8.2% of the KV pool**, because the draft model is resident and its own
KV comes out of the same budget: 250,148 tokens against MTP's 272,585 at `MAXLEN=204800`.
At `MAXSEQS=2` that's 1.22× concurrency instead of 1.33×. If the second full-length slot
is worth more to you than the decode speed, `SPEC=mtp` is still supported and still
correct — it's one environment variable, and the MTP numbers below describe it.

`SPECTOK=4` is measured, not assumed: K was swept over {4, 6, 7, 8} and 4 wins, with
K ≥ 7 unable to hold `MAXLEN=204800` at all (the draft's own KV grows with K).

#### Which DFlash2 draft checkpoint

Two exist. Both were measured here at `MAXLEN=131072`, everything else held constant:

| draft checkpoint | added weights | steps/s | acc. len | decode tok/s |
|---|---|---|---|---|
| *(none — MTP head in the target)* | — | 14.03 | 2.83 | 39.74 |
| [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (bf16), greedy | +3.47 GiB | 17.58 | 2.68 | 47.19 |
| [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16) (int4), greedy | +1.08 GiB | 18.74 | 2.61 | 48.80 |
| **`syvai/…-W4A16` (int4), probabilistic** | **+1.08 GiB** | **19.07** | **3.01** | **57.31** |

"Added weights" is the measured `Model loading took` figure minus the 17.93 GiB target,
which is what actually comes out of the KV pool.

**Take the int4 one and sample it probabilistically.** int4 is not a quality compromise
here — it is *faster* than the bf16 draft and 2.39 GiB lighter, and that 2.39 GiB goes
straight back into the KV pool.

`draft_sample_method` is the larger of the two levers and it is one word.
`greedy` makes the drafter commit to its argmax; `probabilistic` samples from its
distribution, and the target then accepts more of it — 2.61 → 3.01 tokens per step, +15%,
worth +17% decode on its own. It's the single largest tuning win in this configuration.

The int4 draft is a **packed compressed-tensors** checkpoint, which merged upstream vLLM
cannot load at all — see `patches/dflash2/README.md`.

<details>
<summary>The MTP numbers this replaced (2026-08-15, <code>MAXLEN=131072</code>, MTP ×4)</summary>

| prompt depth | prefill tok/s | decode tok/s | accepted draft len | engine steps/s |
|---|---|---|---|---|
| 3,183 | 1618.9 | 44.34 | 3.03 | 14.64 |
| 12,520 | 1546.7 | 39.81 | 2.77 | 14.40 |
| 38,997 | 1335.1 | 43.91 | 3.18 | 13.80 |
| **mean** | **1500.2** | **42.69** | **2.99** | **14.28** |

Weights 17.93 GiB, GPU KV cache 249,982 tokens = 1.91× concurrency at `MAXLEN=131072`.
This is what `SPEC=mtp` still gives you, and it is what `logs/boot-qwen3.8-27b-vllm.log`
was captured on.

</details>

**Against Qwen3.6-27B**, which still runs MTP. The 3.6 arm was captured 2026-08-11 and not
re-baselined in the same window, so treat the magnitudes as soft:

| | 3.6-27B (MTP) | 3.8-27B (MTP) | 3.8-27B (DFlash2) |
|---|---|---|---|
| prefill tok/s | 1513.2 | 1500.2 | ~1500 |
| engine steps/s | 16.61 | 14.28 | 19.07 |
| accepted draft len | 4.25 | 2.99 | 3.01 |
| decode tok/s | 71.2 | 42.7 | 57.3 |

**On MTP the 3.8 decoded materially slower than the 3.6, and it was not a missing
optimisation.** All 13 performance env vars are byte-identical between the two launchers;
the difference was speculative-decode *acceptance* — the 3.8 checkpoint's MTP head drafts
worse, and the two `mean_len` distributions didn't overlap. Whether that is Qwen 3.8 or
that particular quantizer's head was never tested, and with DFlash2 it stops mattering:
swapping the drafter recovers most of the gap without touching the target checkpoint. The
3.6 has not been re-measured on DFlash2 — no DFlash2 draft has been published for it.

### Quality

Measured against the **live server through this exact config** (fp8 KV + speculative
decoding + thinking template at `REASONING_EFFORT=low`), not against the weights in
isolation.

| | **this checkpoint** | AMD Quark-Qronos | AMD Quark-AWQ | Qwen BF16 base |
|---|---|---|---|---|
| GSM8K 5-shot (thinking), DFlash2 | **93.71%** | — | — | — |
| GSM8K 5-shot (thinking), MTP | **94.77%** | 94.62% | 91.21% | 93.33% |
| BFCL v4 overall (single_turn) | **25.29%** | 23.80% | 24.06% | 24.38% |
| BFCL Non-Live AST | 87.46% | 85.17% | 86.58% | **88.52%** |
| BFCL Live AST | 81.87% | **82.09%** | 81.57% | **83.05%** |
| BFCL Relevance Detection | **75.00%** | 62.50% | **75.00%** | **75.00%** |
| BFCL Irrelevance Detection | **83.62%** | 70.79% | 72.47% | 72.22% |

**The BFCL row was measured on the MTP config and has not been re-scored on DFlash2.**
The GSM8K pair above is the like-for-like decoder comparison: same harness, same config,
one variable. The 1.06 pp gap is **1.17σ on 1319 problems** — not a distinguishable
difference at this sample size, in either direction. Both are temperature-1.0 numbers;
greedy scores ~97.6% on the same harness, which is worth knowing before reading either
figure as a quality ceiling.

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
- Neither eval says anything about the drafter comparison above. Both measure
  target-model output.

The takeaway worth generalising: **the AutoRound recipe transfers across uploaders.** This
is an unknown author's checkpoint using AutoRound's standard settings (gs=128, symmetric,
`auto_round:auto_gptq`, the default `linear_attn.in_proj_a/b` fp16 exclusions), and it beats
two AMD-authored checkpoints built with more sophisticated algorithms — Qronos is
Hessian-based PTQ with a paper behind it. Prefer standard AutoRound over a clever algorithm
with a bespoke module list.

### Why not the other 3.8 int4 checkpoints

There is **no Intel AutoRound release for 3.8** (they published three for the 3.6). Of what
exists:

**One caveat on this whole table, added 2026-08-22.** It ranks candidates largely on their
MTP draft head, and with `SPEC=dflash2` — now the default — the target's MTP head is never
called. It is then simply 0.51 GiB of resident dead weight, so the argument weakens from
"the int4 head drafts better" to "the int4 head is smaller". It still selects the same
checkpoint, for a weaker reason. The argument returns in full under `SPEC=mtp`.

| candidate | why not |
|---|---|
| `amd/…Quark-Qronos-INT4-W4A16` | **BF16 MTP head** (all 15 `mtp.*` tensors plain BF16). A sibling 3.8 checkpoint with the same BF16 head was measured here at +0.51 GiB and ~9% slower decode; AMD's own were not downloaded, so that cost is inferred from the shape, not measured on theirs |
| `amd/…Quark-AWQ-INT4-W4A16` | same, BF16 MTP head |
| `goldhub/…INT4-W4A16-AutoRound` | gs=32 → clamps `BLOCK_K` to 32, off the tuned gs=128 table; also 26.37 GiB |
| `Israeli-AI/…MTP-W4A16` | 25.77 GiB, mostly unquantized, BF16 MTP head |
| `Qwen/Qwen3.8-27B-FP8` (official) | 28.75 GiB + 3.29 GiB measured overhead = **overruns the 31.86 GiB card before a single KV byte** |
| NVFP4 family | no gfx1201 kernel |

Verify you got the int4-head build at load: `Model loading took 17.93 GiB`. With
`SPEC=dflash2` that line reports target *and* draft as one figure — 19.01 GiB — so subtract
~1.08 GiB for the int4 draft before comparing. An 18.4-ish GiB figure (19.5-ish with the
draft) means a BF16-head checkpoint — that was measured here at 236,470 KV tokens against
249,982, i.e. the head costs you 0.51 GiB of weights and ~13,500 tokens of context.
Compare your own boot against `logs/boot-qwen3.8-27b-vllm.log`.

## Contents

- `startup-qwen3.8-27b-vllm.sh` — dense 27B, int4 W4A16, **DFlash2 speculative decoding**
  on (with `SPEC=mtp` as a supported fallback), vision, 204800 context,
  `REASONING_EFFORT` knob.
- `startup-qwen3.6-27b-vllm.sh` — dense 27B, int4 W4A16, MTP speculative decoding on,
  vision, 131072 context.
- `startup-qwen3.6-35b-vllm.sh` — MoE 35B-A3B (256 experts), int4 W4A16, MTP off
  (KV cost doesn't pay off at this size), vision, 262144 context.
- `logs/boot-qwen3.8-27b-vllm-dflash2.log` — real boot log of the **shipped DFlash2
  configuration**, container start through `Application startup complete` plus the first
  inference JIT warnings and the first `SpecDecoding metrics` line, captured 2026-08-22.
  Worth grepping for: `Using RDNAHybridW4A16LinearKernel` (twice — once for the target's
  AutoGPTQ path, once for the draft's compressed-tensors path), `Model loading took 19.01
  GiB`, `GPU KV cache size: 250,148 tokens`, `Capturing dflash2 CUDA graphs`, and
  `Mean acceptance length: 3.00`.
- `logs/boot-qwen3.8-27b-vllm.log` — the same for the **MTP** configuration, captured
  2026-08-15: `Model loading took 17.93 GiB`, `GPU KV cache size: 249,982 tokens`.
  Both were captured from the homelab llama-swap launcher rather than these scripts, so
  their `non-default args` line carries three extra metrics flags
  (`enable_prompt_tokens_details`, `enable_per_request_metrics`,
  `enable_force_include_usage`) that these scripts don't set. Nothing else differs.
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
  fused shape is identical. Re-confirmed by a runtime kernel census on a 3.8 load: **99.45%
  of kernel work lands on tuned rows.** Nothing to re-sweep.

- `patches/dflash2/` — a ten-file back-port of vLLM **PR #52816** (DFlash2 speculative
  decoding) onto the vLLM 0.26.0 inside the image, bind-mounted read-only over the image's
  `vllm` package whenever `SPEC=dflash2`. **The 3.8 script cannot run DFlash2 without it** —
  the image predates the PR and fails on an unknown model architecture. Every file names
  the upstream commit it came from and any deviation. Provenance, the one mandatory
  non-obvious file, the three deliberate deviations, and the one function that is not
  upstream at all are all in `patches/dflash2/README.md`.

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

which is the configuration every MTP number in this README was measured on.

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

## The image these build on, and where this repo diverges

Every script here runs **`docker.io/stilldeadcode/vllm-radiance:0.5.8`** — a third-party
ROCm build of vLLM carrying gfx1201/RDNA4-specific work that stock vLLM does not have.
None of these launchers would reach these numbers on an upstream vLLM image, and the
credit for that half belongs to its author, not to this repo.

The image is built from a public source repository:
**[codeberg.org/StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance)**.
Read that before this section — it is the authority on what the image is, and its
`DOCKERHUB.md` carries the full environment-variable reference.

Two things to know about it before you depend on it:

- **Pin it by commit, not by tag.** The repository publishes no git tags and no releases; the
  version lives in a single `VERSION` file on `main`, which read `0.5.8` at the time of writing
  (commit `a55f1bc`, 2026-08-14). `main` moving does not change the Docker tag it produces, so a
  future `0.5.8` build need not be this one — if you need to reproduce these numbers exactly, pin
  the commit as well as the image tag. The repository also carries no `LICENSE` file at the time
  of writing, which is worth knowing before you vendor any of it.
- **Its own centre of gravity is not this configuration**, by its author's own statement. Its
  README gives the tested envelope as three FP8 models "all with fp8 (or bf16/`auto`) KV cache on
  **two R9700 GPUs (tensor parallel)**", and says in as many words that "**non-FP8 weights, single
  or 3+ GPUs** … are untested". This repo sits squarely in that untested region: **int4 W4A16 on a
  single card at TP=1**, every model. The practical read is that the fp8-weight GEMM path the image
  *owns* is not a path any of these scripts take, and neither is the TP=2 collective work; what they
  do use — attention and the fp8 **KV cache** — sits much closer to stock vLLM. That is the good
  news, because it means upstream fixes to those reach us through a version bump rather than needing
  the fork to carry them. It is also why nothing here should be read as a defect report against that
  image: it is being run well outside the envelope its author tested.

### Where this repo diverges, and why

| divergence | why |
|---|---|
| **int4 W4A16, not FP8** | The 27B in FP8 is 28.75 GiB of weights plus 3.29 GiB measured overhead, which overruns a 31.86 GiB card before a single KV byte. FP8 is the configuration this card *wants* on models that fit; these don't. |
| **TP=1, single card** | One R9700, not two. Everything here — KV budgeting, `MAXSEQS`, the concurrency arithmetic — is sized for one. |
| **Tuned W4A16 tile table** (`patches/rdna_hybrid_w4a16.py`, `TUNED_TILES=1`) | The image's own gfx1201 Triton tile heuristic is tuned on a different model's shapes and group_size. Re-sweeping it for these shapes at gs=128 measured **+4.6–9.6% prefill and +4.3% decode step rate**. It also carries a symmetric-GPTQ zero-point fix the AutoRound checkpoints need to run at all. |
| **`--no-async-scheduling`** (`ASYNCSCHED=0`) | vLLM's default is async scheduling on. On this host (a 4-vCPU guest) enabling it measured **42–45% slower decode** — the per-step CPU-side scheduling work is the bottleneck, so overlapping it with GPU execution hurts. Likely host-specific; it's a knob. |
| **`ROCM_AITER_UNIFIED_ATTN`** for the target | Swept against `TRITON_ATTN` and `ROCM_ATTN` on this card; AITER unified won by 88–92% on deep prefill and 97–99% on decode against the respective runners-up. The cost is that CPU KV offload, which needs `TRITON_ATTN`, is off the table — a real trade. |
| **`TRITON_ATTN` for the DFlash2 draft** | Not a preference — DFlash2's draft attention is non-causal and AITER refuses a non-causal mask. The two backends are set independently. |
| **fp8 KV cache** | Doubles the KV pool on a card where the pool is the binding constraint. |
| **The AutoRound `config.json` routing patch** (`CONFIG_FIX=1`) | vLLM 0.26's INC backend hijacks any checkpoint declaring `quant_method: "auto-round"` and lands on the wrong kernel, and a missing `modules_in_block_to_quantize` loads the model **fully unquantized** — both silently. Not an image issue; it bites any stack at this vLLM version. |
| **`MAXLEN=204800`** on the 3.8 | Sized to the measured KV pool, not to the model's 262144 `max_position_embeddings`. vLLM refuses to start when `max-model-len` exceeds the pool. |
| **DFlash2 back-port** (`patches/dflash2/`) | The image is built on vLLM **0.26.0** (its Dockerfile pins `ARG VLLM_VERSION=0.26.0`), which predates DFlash2 (upstream PR #52816, merged 2026-08-21). Rebuilding the image to get it would have cost the gfx1201 work above, so the PR was back-ported onto 0.26.0 as ten bind-mounted files instead. Full provenance in `patches/dflash2/README.md`. |

One image behaviour is left at its default and is worth naming because it is
**fork-specific rather than upstream vLLM**: `RADIANCE_DYNAMIC_DRAFT=1` enables a per-request
draft-depth controller that lives in the image, not in the `vllm` package (source:
`radiance_draft.py`; the image's `DOCKERHUB.md` documents the knob, and `=0` restores stock
behaviour). It applies to `method=mtp` only, so it is inert on the DFlash2 default here and
live under `SPEC=mtp`. These scripts leave it alone.

### Credit

- **`stilldeadcode`** — the [`vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance)
  image, and the gfx1201 kernel work in it. Bug reports about the *image* belong on its issue
  tracker there, not here.
- **vLLM PR #52816** — DFlash2 itself. Everything in `patches/dflash2/` is a derived work
  of vLLM, Apache-2.0, headers intact.
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
