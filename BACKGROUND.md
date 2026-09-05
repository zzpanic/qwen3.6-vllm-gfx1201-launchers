# Background: how these numbers were arrived at

The long-form reasoning behind the 3.8 launcher's defaults — the decoder comparison, the
draft-checkpoint selection, the quality evals in full, the rejected checkpoints, and what
the underlying image is and where this repo departs from it. [README.md](README.md) is the
executive summary; [TUNING.md](TUNING.md) is the dated tuning log; `benchmarks/` holds the
raw numbers.

**Scope: this file is the int4 W4A16 story** (`startup-qwen3.8-27b-int4.sh`). It was
written when that was the only 3.8 path, and the decoder comparison, the draft-checkpoint
selection, the quality evals and the rejected checkpoints below are all measured on it.
Two things carry across to the MXFP4 path unchanged, because they are properties of the
method and the model rather than of the quantisation: **DFlash2 over MTP** (MTP is not
lossless — +15.8 sigma against the model's own logits, where DFlash2 measures clean at
+1.4), and the *shape* of the quality argument — measure the serving path, not the weights,
and mind the error bar on a difference. The quality **numbers** do not carry across: they
are int4 figures, and the MXFP4 stack is evaluated separately (see the Quality section of
[README.md](README.md#quality)). Everything else numeric does **not** carry across either. The MXFP4
configuration is in [README.md](README.md#quick-start) and its decisions are in the
2026-09-05 entries of [TUNING.md](TUNING.md).

## Contents

- [DFlash2 versus MTP](#dflash2-versus-mtp)
- [Which DFlash2 draft checkpoint](#which-dflash2-draft-checkpoint)
- [Quality, in full](#quality-in-full)
- [Why not the other 3.8 int4 checkpoints](#why-not-the-other-38-int4-checkpoints)
- [The image these build on, and where this repo diverges](#the-image-these-build-on-and-where-this-repo-diverges)

## DFlash2 versus MTP

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
At `MAXSEQS=2` that's 1.22× concurrency instead of 1.33×. (Both figures predate
`KV_GROUP_SIZE=auto`, which since put DFlash2 at 269,837 / 1.32×; the *ratio* between the
two arms is the point here, and it was measured with the same group size on both sides.)
If the second full-length slot
is worth more to you than the decode speed, `SPEC=mtp` is still supported and still
correct — it's one environment variable, and the MTP numbers below describe it.

`SPECTOK=4` is measured, not assumed — and it is a **workload** choice. K was swept over
{4, 5, 6, 7} and aggregate decode barely moves (79.0 / 69.5 / 76.3 / 78.6 tok/s), but the
categories move in opposite directions: **math +13.9% and chat −8.9% going from K=4 to
K=7**. A math- or code-heavy server should run K=7 — **and that is what the MXFP4 script
ships**. The K=4 default below survives only on the int4 script: it was chosen against a
prose-weighted corpus, and re-run against the agentic code/reasoning corpus this box
actually serves, K 4 → 7 moved the weighted score +35.1%. See the retraction on the
2026-08-29 `SPECTOK` entry in TUNING.md. It also costs context (231,602 → 212,098 tokens at `MAXLEN=131072`). The
older claim that K ≥ 7 could not hold `MAXLEN=204800` is **retracted** — that was an
artifact of `KV_GROUP_SIZE=5` and died with the group-padding fix. Full tables, the
unexplained K=5 dip, and the control arm are in TUNING.md.


## Which DFlash2 draft checkpoint

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


## Quality, in full

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

## Why not the other 3.8 int4 checkpoints

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
Compare your own boot against `logs/boot-qwen3.8-27b-vllm.log` (MTP) or
`logs/boot-qwen3.8-27b-vllm-0.9.3.log` (the current DFlash2 default).

## The image these build on, and where this repo diverges

Every script here runs a **`docker.io/stilldeadcode/vllm-radiance`** image — a third-party
ROCm build of vLLM carrying gfx1201/RDNA4-specific work that stock vLLM does not have. The
3.8 script defaults to **`:0.9.3`** (vLLM 0.27.1); the 3.6 scripts still run **`:0.5.8`**
(vLLM 0.26.0). The tag is the only reliable identifier — image `0.7.4` reported
`RADIANCE_VERSION=0.6.2` in its own environment.

One pin inside the image matters more than the image version: the kernels live in a
separate repository (`libr4d`) pinned by `R4D_VERSION`, and that is a **correctness** pin,
not a performance one. The version 0.7.4 pinned had three exponent overflows in the
gated-delta-net kernels producing `0 * INF = NaN` across an entire head — WikiText-2
perplexity **653,586** against 8.37 once fixed. 0.9.3 pins the fixed version. Check it on
every bump.
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
| **`R4D`** for the target on `:0.9.3` (`ROCM_AITER_UNIFIED_ATTN` on `:0.5.8`) | AITER unified was first swept against `TRITON_ATTN` and `ROCM_ATTN` and won by 88–92% on deep prefill and 97–99% on decode. R4D — radiance's own gfx1201 attention, which only exists on 0.9.x — then beat *that* on deep prefill: 2K +0.4%, 8K −1.8%, 16K +1.0%, 32K +2.9%, **64K +7.2%**, with decode and acceptance unchanged within noise and the KV pool slightly *up*. Either way CPU KV offload, which needs `TRITON_ATTN`, is off the table — a real trade. |
| **`RADIANCE_SKINNY_GEMM=all`** | The image ships `1`, which routes only the bf16 projections that win outright. `all` adds shapes that differ from rocBLAS at a bf16 ULP on a few elements in ten thousand — notably the gated-delta-net `in_proj_ba`, 480 KiB run **48 times per step**, 28.5 µs → 3.6 µs. Those are opt-in upstream because under speculative decoding a ULP can move acceptance; measured here under DFlash2 it did not. Set `RADIANCE_SKINNY_GEMM=1` to take the image default. |
| **`KV_GROUP_SIZE=auto`** (`patches/radiance-0.9.3/`) | vLLM sizes KV cache groups at `min()` of the layer-bucket sizes — 48 GDN + 16 full-attn + 5 drafter → 5, which divides neither 48 nor 16, padding 69 real layers into 75 slots. `auto` picks 8 instead: **222,639 → 269,837 tokens, +21.2%**, at identical VRAM and unchanged speed. Unset is byte-identical to stock. |
| **`TRITON_ATTN` for the DFlash2 draft** | Not a preference — DFlash2's draft attention is non-causal and AITER refuses a non-causal mask. The two backends are set independently. |
| **fp8 KV cache** | Doubles the KV pool on a card where the pool is the binding constraint. |
| **The AutoRound `config.json` routing patch** (`CONFIG_FIX=1`) | vLLM 0.26's INC backend hijacks any checkpoint declaring `quant_method: "auto-round"` and lands on the wrong kernel, and a missing `modules_in_block_to_quantize` loads the model **fully unquantized** — both silently. Not an image issue; it bites any stack at this vLLM version. |
| **`MAXLEN=204800`** on the 3.8 | Sized to the measured KV pool, not to the model's 262144 `max_position_embeddings`. vLLM refuses to start when `max-model-len` exceeds the pool. |
| **DFlash2 back-port** (`patches/dflash2/`) | The image is built on vLLM **0.26.0** (its Dockerfile pins `ARG VLLM_VERSION=0.26.0`), which predates DFlash2 (upstream PR #52816, merged 2026-08-21). Rebuilding the image to get it would have cost the gfx1201 work above, so the PR was back-ported onto 0.26.0 as ten bind-mounted files instead. Full provenance in `patches/dflash2/README.md`. |

One image behaviour is left at its default and is worth naming because it is
**fork-specific rather than upstream vLLM**: `RADIANCE_DYNAMIC_DRAFT=1` enables a per-request
draft-depth controller that lives in the image, not in the `vllm` package (source:
`radiance_draft.py`; the image's `DOCKERHUB.md` documents the knob, and `=0` restores stock
behaviour). Its baked schedule is `1:8,2:7,4:6,8:5,16:4` — propose 8 tokens at batch size 1,
tapering to 4 at batch 16 — which reads as though a single-stream request drafts 8 regardless
of `SPECTOK`.

**It cannot.** The schedule is a ceiling applied as `if 0 < batch_ceil < num_speculative_tokens`
— it only ever clamps *downward*, and `num_speculative_tokens` is `SPECTOK`. Raising `SPECTOK`
is the only way to lengthen a draft on this box. The K sweep is consistent with that: the
engine counters read exactly `SPECTOK` draft tokens per draft (4.000 / 5.000 / 6.000 / 7.000)
at a `MAXSEQS` where the schedule would have permitted 7–8.

**It is not MTP-only, despite its own documentation.** Its module docstring and every knob
name say MTP, but on `:0.9.3` it patches `SpecDecodeBaseProposer` — the shared base that
`DFlashProposer` inherits without overriding either patched method — so it is loaded on the
DFlash2 default here too. What makes it *mostly* inert under this configuration is something
else: the depth controller lives inside `_greedy_sample`, and vLLM only calls that when the
request is all-greedy or probabilistic draft probs are off. This repo's default is
`DRAFT_SAMPLE_METHOD=probabilistic`, so ordinary sampled traffic takes the probabilistic
path and never reaches the controller; greedy requests do reach it. Either way the ceiling
above holds. These scripts leave the controller alone.
