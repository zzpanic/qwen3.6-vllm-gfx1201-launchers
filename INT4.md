# The int4 fallback, and the historical 3.6 scripts

**This page is the int4 W4A16 path** — `startup-qwen3.8-27b-int4.sh` — plus the two
Qwen3.6 scripts kept for their reasoning. It was split out of [README.md](README.md), which
is now about the current MXFP4 configuration only.

**Nothing on this page is comparable to a number in README.md.** Different checkpoint,
different speculative depth (K=4 against 7), different KV pool, different sampling
temperature — and a pool-size change alone is worth ~5% decode on this card. Read each
stack against itself.

**Why this path is kept.** It is the maintained fallback: same model, a better-understood
kernel path, measurably slower. It is what to fall back to when a radiance image bump
breaks the MXFP4 stack, which has happened. It is not abandonware and it is not the place
the tuning work is going.

- [What it does](#what-it-does)
- [Running it](#running-it)
- [Which 27B checkpoint (Qwen3.6) — historical](#which-27b-checkpoint-qwen36--historical)

## What it does
Checkpoint: [`devan-carlin/Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound),
17.93 GiB, int4 W4A16 gs=128, **int4 MTP draft head**.

### Throughput

`MAXLEN=204800`, `MAXSEQS=2`, **DFlash2 ×4**, fp8 KV, single stream, **int4**. Re-captured
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
`benchmarks/int4-20260829/`.

Single stream, by workload category.

> **The ITL columns below are an artifact and BetterBench 0.4.0 removed them.** Under
> speculative decoding one stream update carries a whole accepted run, so "inter-token
> latency" was really update spacing divided by a variable token count. They are left here
> verbatim because this is a published record; read `decode t/s`, `TTFT` and `CV` instead.

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

Concurrency, at the shipped `MAXSEQS=2` (int4 — the MXFP4 sweep above goes to 16):

| concurrent requests | aggregate t/s | per-request decode t/s | TTFT p50 (ms) |
|--:|--:|--:|--:|
| 1 | 74.9 | 87.6 | 139.5 |
| 2 | 113.9 | 68.4 | 187.5 |

**1.52× aggregate throughput for 0.78× per-request speed.** Two streams is worth it if you
have two streams; it is not free.

### Speculative decoding, in one paragraph

The int4 default is **DFlash2 ×4** with the int4 draft checkpoint sampled probabilistically;
**the MXFP4 script ships K=7** (see the retraction in TUNING.md — the K=4 "peak" was an
artifact of a prose-weighted corpus). That
combination decodes **1.44× faster than the target's own MTP head** (19.07 vs 14.03 engine
steps/s at matched depth) and costs **8.2% of the KV pool**, because the draft model is
resident. `SPEC=mtp` remains supported and drops the draft model entirely. `SPECTOK=4` is a
measured **workload** choice, not a universal one: over K ∈ {4,5,6,7} aggregate decode barely
moves, but math gains 13.9% and chat loses 8.9% going to K=7. A math- or code-heavy server
should run K=7 — the MXFP4 script does. Draft-checkpoint comparison, the K tables, and the
retracted "K≥7 doesn't fit" claim are in
[BACKGROUND.md](BACKGROUND.md#dflash2-versus-mtp) and TUNING.md.

"Sampled probabilistically" is `draft_sample_method=probabilistic`, and it is worth keeping
even though it looks like it should cost you something. Isolated over four alternating boots
on 2026-09-05 it gave **about +10% accepted tokens per update for no measurable step cost**
— `steps/s` was flat to 0.1% on every boot, so the full draft-logits head it needs is free
here. The edge grows with sampling temperature. Numbers and method are in TUNING.md.

### Quality

Measured against the **live server through this exact config** — fp8 KV, speculative
decoding, thinking template — not against the weights in isolation. **These are the int4
figures**; the MXFP4 checkpoint's own GSM8K result is in
[`benchmarks/deep-prefill-20260905/`](benchmarks/deep-prefill-20260905/).

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

## Running it

For the current MXFP4 configuration see [README.md](README.md#quick-start) instead.


```
hf download devan-carlin/Qwen3.8-27B-int4-AutoRound --local-dir ./models/qwen3.8-27b-autoround
hf download syvai/Qwen3.8-27B-DFlash2-W4A16      --local-dir ./models/qwen3.8-27b-dflash2-int4
./startup-qwen3.8-27b-int4.sh
```

The second download is the DFlash2 draft model. To run without it:

```
SPEC=mtp MAXLEN=131072 ./startup-qwen3.8-27b-int4.sh
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

## Which 27B checkpoint (Qwen3.6) — historical

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
the published claim rather than a measurement made in this repo. (The 3.8 quality results — on
[this page](#quality) for int4 and in [README.md](README.md#quality) for MXFP4 — are the
first quality measurements in this repo, and they are on different checkpoints.) What *was*
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
