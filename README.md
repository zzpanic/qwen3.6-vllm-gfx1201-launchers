# Qwen3.6 / Qwen3.8 vLLM launchers for AMD gfx1201 (Radeon AI PRO R9700)

Standalone podman/vLLM startup scripts for serving Qwen3.8-27B and Qwen3.6-27B (dense) and
Qwen3.6-35B-A3B (MoE) on a single 32 GiB AMD Radeon AI PRO R9700 (gfx1201, RDNA4),
using the `docker.io/stilldeadcode/vllm-radiance` image (ROCm + AITER + a few
gfx1201-specific perf hooks) — `:0.9.3` for the 3.8 scripts, `:0.5.8` for the 3.6 ones.

**Start with [Quick start](#quick-start).** The current, actively tuned configuration is
`startup-qwen3.8-27b-mxfp4.sh` (native MXFP4 W4A8). That path is **not this repo's work** —
it is downstream of three R9700 community projects.
[**ggz14**](https://github.com/GGZ14/vllm-mxfp4) (`brian_launch80`) wrote the MXFP4 kernels
and the image; [**malicz**](https://github.com/malicz/vllm-gfx1201-launchers)'s launchers are how that work was picked up here; and
[**hifi/vllm-radlight**](https://codeberg.org/hifi/vllm-radlight) published the numbers this was measured against and eventually
passed. [Where the MXFP4 path came from](#where-the-mxfp4-path-came-from) says what each
one contributed. To *run* it you must fetch three pieces:
[**ggz14**](https://github.com/GGZ14/vllm-mxfp4)'s MXFP4 kernels, DFlash2 integration and
setup script; [**AMD**](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4)'s Quark
MXFP4 weights; and
[**tcclaviger**](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8)'s FP8 DFlash2
draft head — all running inside `stilldeadcode`'s `vllm-radiance` image. What this repo
adds is the arrangement, the tuning and the measurement. The int4 script beside it is the
maintained fallback; the two Qwen3.6 scripts are historical and kept for the reasoning
rather than to be served. The status table under [Contents](#contents) says which is which.

See
[BACKGROUND.md](BACKGROUND.md) for what that image is, which of the decisions here are
ours rather than its defaults, and the measurements behind each of them.

Extracted from a homelab multi-model setup normally driven by
[llama-swap](https://github.com/mostlygeek/llama-swap); these scripts have no dependency
on it and run standalone. Every config decision baked into them (kernel gate to check in
the boot log, KV cache budgeting, why MTP is on for one model and off for the other,
measured speculative-decode tuning, checkpoint defects that silently break things) is
explained inline in each script's header comment, since most of it isn't documented
anywhere else.

## Quick start

The fastest configuration in this repo is the MXFP4 one. End to end on a clean box:

```bash
# 1. the launchers
git clone https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers
cd qwen3.6-vllm-gfx1201-launchers

# 2. ggz14's radiance repo, NEXT TO the script and under this exact directory name.
#    The MXFP4 launcher sources gpu-detect.sh from it at start-up and mounts it as
#    /patches, so it is a hard dependency, not an optional extra. The GitHub repo is
#    named vllm-mxfp4; the directory must be radiance-vllm-mxfp4, hence the argument.
git clone https://github.com/GGZ14/vllm-mxfp4 radiance-vllm-mxfp4

# 3. the two checkpoints (~21 GiB). setup-mxfp4.sh is ggz14's, in the repo you just
#    cloned; it downloads AMD's Quark MXFP4 weights and runs fp8_mtp.py over the MTP
#    head, which is mandatory for that checkpoint (see below).
cd radiance-vllm-mxfp4 && ./setup-mxfp4.sh && cd ..

# 4. serve
./startup-qwen3.8-27b-mxfp4.sh
```

Then check it is alive, and that it is serving what you think it is:

```bash
curl -s localhost:8080/v1/models | python3 -m json.tool     # max_model_len should be 204800
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-mxfp4","messages":[{"role":"user","content":"hello"}]}'
```

`./startup-qwen3.8-27b-mxfp4.sh -h` prints every knob, its default, and what it does.
Nothing needs editing to run: the defaults **are** the measured production configuration.

### Four things that go wrong

- **`gpu-detect.sh: No such file`** — step 2 was skipped, or the clone landed under
  `vllm-mxfp4` instead of `radiance-vllm-mxfp4`. Pass `REPO=/path/to/it` if you want it
  somewhere else.
- **An assertion about a half-width parameter during load** — the MTP head was not
  requantised. AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` names its `mtp.*` layers as *tensor*
  names inside a list of *module* names, so Quark's exclusion never fires and vLLM applies
  the mxfp4 scheme to a bf16 head. `fp8_mtp.py` fixes it; `setup-mxfp4.sh` runs it for you.
  A checkpoint that declares `mtp.*` in `layer_quant_config` needs neither — point `SNAP`
  at it directly.
- **"current platform does not support native MXFP4/MXFP6" in the log** — a false alarm.
  It comes from a separate `supports_mx()` call, not the kernel gate. The line that matters
  is `Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM`.
- **It starts, then dies on the first long prompt** — the KV pin is sized for a 32 GiB
  card. See "The KV pin, which is the one that will bite you" below; it is the one default
  here that is genuinely hardware-specific.

### Requirements

- An **AMD gfx1201** card with **32 GiB** (Radeon AI PRO R9700). This is tuned for that
  card specifically — attention backend, kernel gates and KV budgeting are card-shape
  specific and will not transfer as-is.
- **rootless podman** (or docker) with ROCm/kfd device access. On a systemd box that means
  `loginctl enable-linger $USER`, or a cold boot fails with "crun not found".
- **~25 GiB of disk** for the two checkpoints, plus room for the image (~20 GiB).
- **Host RAM**: the weights are sharded on disk into four parts, so loading needs roughly
  the size of the largest part rather than the whole checkpoint.
- No Resizable BAR requirement — this was developed on a board without it.

## What this card actually does — MXFP4, the current configuration

Qwen3.8-27B is a **retrained release of the Qwen3.6-27B architecture**, not a new one —
same `model_type qwen3_5`, same 64 layers / 5120 hidden / 17408 intermediate / 24:4 heads /
head_dim 256 / vocab 248320, which is why tuning done on 3.6 transfers to it unchanged.

Everything in this section is `startup-qwen3.8-27b-mxfp4.sh` at its shipped defaults:
`MAXLEN=204800`, `KV_MEM=9300000000`, `MAXSEQS=2`, DFlash2 **K=7** sampled
probabilistically, fp8 KV, vision tower loaded. Raw files in
[`benchmarks/deep-prefill-20260905/`](benchmarks/deep-prefill-20260905/).

**Check your own boot against these four lines** (from `logs/boot-qwen3.8-27b-mxfp4.log`):

```
Setting attention block size to 1648 tokens
max_num_scheduled_tokens is set to 2036
GPU KV cache size: 228,737 tokens
Maximum concurrency for 204,800 tokens per request: 1.12x
```

There is deliberately **no** `Available KV cache memory` line. An explicit
`--kv-cache-memory` makes vLLM skip memory profiling, which is what prints it. Its absence
is the pin working, not a truncated log.

### Generation, by depth

Tokenizer-calibrated depths, so the token counts are exact rather than estimates.

| prompt tokens | prefill t/s | decode t/s | accepted len | **engine steps/s** |
|--:|--:|--:|--:|--:|
| 2,000 | 3070 | 71.3 | 4.20 | **16.99** |
| 15,986 | 2708 | 59.3 | 3.56 | **16.69** |
| 63,982 | 2291 | 52.5 | 3.32 | **15.78** |
| 127,965 | 1895 | 48.7 | 3.28 | **14.82** |
| 199,972 | 1590 | 45.6 | 3.28 | **13.90** |

**Read `steps/s`, not decode t/s.** `decode t/s = steps/s × accepted len`, and accepted
length is a draw from the speculative-acceptance lottery — it swings enough on a single
request to invert neighbouring rows. The step rate is counted, not derived, and repeats to
~0.1% across boots.

Two things worth knowing before you size a workload:

- **Generation at a real 200k runs at 82% of its 2k step rate.** A 100× longer prompt costs
  18% of the step rate.
- **Acceptance does not degrade with depth** — accepted length is flat at 3.28 from 64k
  through 200k.

### Prefill, by depth

Taken separately at n=5 per rung (the ladder above is n=1 and its prefill column shows it).
Peak is at ~6k, not at the shallowest rung: 1544 tokens is too small to fill the GPU.

| prompt tokens | prefill t/s (median) |
|--:|--:|
| 1,544 | 2815 |
| 5,948 | **2882** |
| 23,573 | 2631 |
| 47,085 | 2418 |
| 94,095 | 2087 |
| 146,978 | 1808 |

**2882 → 1808 t/s across a 25× depth increase** — 63% of peak at 147k tokens. The
1%-low-to-99%-high spread is under 0.2% from 64k up; prefill on this card is essentially
noiseless, unlike decode.

### Generation, by workload

20 passes per category, temp 0.7. **Weighted combined ≈ 83.6 tok/s**, TTFT p50 ≈ 102 ms.

| category | tok/update | decode t/s (med) | CV |
|---|--:|--:|--:|
| file_edit | 5.67 | **109.1** | 11.5% |
| math | 5.98 | 105.2 | 5.8% |
| code | 4.91 | 96.8 | 15.6% |
| json | 5.15 | 95.7 | 15.2% |
| reasoning | 3.43 | 64.5 | 28.6% |
| chat | 3.64 | 57.6 | 22.5% |
| prose | 2.86 | 50.2 | 8.2% |

The spread is **entirely acceptance, not speed**. Update spacing is flat at 58.0–58.3 ms
p50 across every category, so the engine steps at a constant rate and the category only
decides how many tokens ride each step. Structured output is predictable enough for the
drafter to land 5–6 tokens per update; prose lands 2.86 and is the floor. Check `CV` before
trusting a single row — reasoning (28.6%) and chat (22.5%) are not stably ordered against
each other.

### Concurrency

| concurrent requests | aggregate t/s | TTFT p50 | per-request decode t/s |
|--:|--:|--:|--:|
| 1 | 74.3 | 102 ms | 90.9 |
| 2 | **134.5** | 163 ms | 86.0 |
| 4 | 137.0 | 5.1 s | 86.7 |
| 8 | 133.8 | 14.9 s | 86.0 |
| 16 | 139.0 | 15.6 s | 88.5 |

**All the aggregate throughput this configuration has is available at 2, and nothing above
2 buys anything.** 1 → 2 is +81%; 2 → 16 is +3.3% for a 95× worse TTFT. Per-request decode
is flat throughout, so above 2 the time goes to the queue, not the GPU.

That is `MAXSEQS=2` working as configured, not a limit being hit — the right trade for a
single-user agentic box. **If you are serving several people, this is the first thing to
change**, and it will cost context.

### Quality

GSM8K, 5-shot, thinking template, measured **through the live server in this exact
configuration** — MXFP4 weights, fp8 KV, DFlash2 K=7, production sampler — rather than
against the weights in isolation. That is the number that matters for serving, and it is
not the number a weights-only eval reports.

> **MXFP4 result: measurement in progress (2026-09-05).** The run is under way against the
> live endpoint; this paragraph carries the figure once it completes. It is stated here as
> pending rather than omitted so that a missing number reads as unfinished work rather than
> as a result nobody wanted to publish.

**The int4 path scores 94.77%** on the same harness (93.71% with DFlash2 rather than MTP).
Read the MXFP4 figure against that one, and only with these four caveats, which apply
whichever way the comparison lands:

- **It is a temperature-1.0 number**, because that is what these scripts serve. Greedy
  scores about 97.6% on the same harness, so ~5 points of the headline are the sampler, not
  the quantisation. GSM8K at temp 1.0 is a poor discriminator between decoders.
- **Single run, no seeds.** The binomial standard error on 1319 problems at ~95% is
  ±0.6 pp, so the standard error on a *difference* between two single runs is ~0.9 pp.
  **Anything inside about ±1.7 pp is not a distinguishable difference**, in either
  direction. Do not read a gap that size as MXFP4 costing or saving quality.
- **It measures the whole serving path**, so a change to the draft head, the KV dtype or
  the chat template moves it. It is not a property of the checkpoint alone.
- **GSM8K is grade-school arithmetic.** It is a regression check against silent numerical
  damage — the kind a bad quantisation causes — not evidence about coding or agentic work,
  which is what this box actually does.

The int4 quality table, the BFCL v4 tool-calling results and the full caveats are in the
[int4 section below](#quality-1) and in
[BACKGROUND.md](BACKGROUND.md#quality-in-full).

---

## The int4 fallback: what it does

> **These are int4 W4A16 numbers**, from `startup-qwen3.8-27b-int4.sh` — *not* the MXFP4
> configuration above, and not what you get if you followed the Quick start. They are kept
> because that script is the maintained fallback and these are the numbers it was tuned to.
> **Do not compare them to the MXFP4 table above**: the checkpoint, the speculative depth
> (K=4 vs 7), the KV pool and the sampling temperature all differ. A different pool size
> alone is worth ~5% decode on this card.

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

## Contents

**Status, so you pick the right one.** Only the MXFP4 script is actively tuned. The
others are kept deliberately and are not abandonware, but they are not where the work is
going:

| script | status |
| --- | --- |
| `startup-qwen3.8-27b-mxfp4.sh` | **current.** Every measurement in `benchmarks/` dated 2026-09-05 is this one. |
| `startup-qwen3.8-27b-int4.sh` | **maintained fallback** (int4 W4A16). Same model, better-understood path, measurably slower. Kept because it is what to fall back to when a radiance bump breaks the MXFP4 stack — that has happened. |
| `startup-qwen3.6-27b-vllm.sh` | **historical.** Qwen3.6 is superseded by Qwen3.8 on the same architecture; kept for the reasoning and the tile table, not because you should serve it. |
| `startup-qwen3.6-35b-vllm.sh` | **historical.** As above, plus the MoE-specific findings (why MTP is off at that size). |

The two 3.6 scripts pin `:0.5.8` rather than `:0.9.3` and have not been re-measured since
2026-08-09. Their numbers are correct **for what they measured** and are not comparable to
anything in `benchmarks/` dated later. Treat them as a record of how this card was tuned,
which is genuinely the useful part — the W4A16 tile table in `patches/` came out of that
work and still serves the 3.8, because it is keyed on GEMM shape rather than model name.

- `startup-qwen3.8-27b-mxfp4.sh` — dense 27B, **native MXFP4 W4A8**, DFlash2 ×7 with an fp8
  drafter, R4D attention, vision, 204800 context on a **pinned** KV cache. The fastest
  configuration in this repo; see "MXFP4 vs int4" below for when to prefer it.
- `startup-qwen3.8-27b-int4.sh` — dense 27B, int4 W4A16, **DFlash2 speculative decoding**
  on (with `SPEC=mtp` as a supported fallback), vision, 204800 context,
  `REASONING_EFFORT` knob.
- `startup-qwen3.6-27b-vllm.sh` — *(historical)* dense 27B, int4 W4A16, MTP speculative decoding on,
  vision (images only -- video untested, see TUNING.md), 131072 context.
- `startup-qwen3.6-35b-vllm.sh` — *(historical)* MoE 35B-A3B (256 experts), int4 W4A16, MTP off
  (KV cost doesn't pay off at this size), vision, 262144 context.
- `logs/boot-qwen3.8-27b-mxfp4.log` — the boot for `startup-qwen3.8-27b-mxfp4.sh`, captured
  2026-09-05: `:0.9.3`, R4D attention, `max_model_len 204800`, `kv_cache_memory_bytes
  9300000000` (the pin), `max_num_batched_tokens 2048`, DFlash2 K=7 with
  `draft_sample_method: 'probabilistic'`, and the vision tower loaded. Worth grepping for
  the `non-default args` line (it is the whole configuration on one line) and
  `GPU KV cache size: 228,737 tokens`.
- `logs/boot-qwen3.8-27b-vllm-0.9.3.log` — **the current *int4* configuration** (the MXFP4
  boot is the bullet above): `:0.9.3`, R4D
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
  Those three 3.8 int4 logs were captured from the homelab llama-swap launcher rather than these
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
  speculative-depth sweep over K ∈ {4,5,6,7}, and the W4A16 shape census. Folders are named
  for the quantisation — `int4-*` is the fallback script, `deep-prefill-*` is MXFP4 — because
  the llama-swap entry name is *not* a reliable guide (`qwen3.8-27b-vllm` served int4 in
  August and MXFP4 now). Three runner scripts reproduce the quality and prefill numbers
  against a live endpoint with no exclusive GPU window: `run-betterbench-prefill.sh`,
  `run-gsm8k.sh` and `run-bfcl.sh`. `benchmarks/README.md` explains how to read them all and
  which ones are soft.

See [BACKGROUND.md](BACKGROUND.md) for the long-form reasoning: the decoder and
draft-checkpoint comparisons, the quality evals in full, the rejected checkpoints, and the
divergence table against the underlying image.

See [TUNING.md](TUNING.md) for the dated tuning log: what each non-obvious default
is set to, what it was measured against, and the traps found along the way.

## Running the 3.8 int4 fallback

This section is the **int4 W4A16** path (`startup-qwen3.8-27b-int4.sh`). For the current
MXFP4 configuration see [Quick start](#quick-start) above.


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

## MXFP4 vs int4, and the flags that actually move this card

Two of the scripts here serve the same model on the same card by different routes. Short
version: **MXFP4 is faster, int4 is the better-understood fallback.** Both are maintained.

The MXFP4 path serves `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (with its MTP head requantised to
fp8) through ggz14's radiance kernels in the `stilldeadcode/vllm-radiance:0.9.3` image. None
of those kernels are ours. What this repo adds is the arrangement — and the arrangement is
most of the difference between a stock R9700 and a tuned one.

### The flags, ranked by what they are worth

| flag | effect | notes |
|---|---|---|
| `R4D_ATTN=1` | **+1.7 / +5.6 / +24.9% prefill** at 4k / 16k / 64k; +1.3/+1.5/+2.9% decode | radiance's attention instead of `ROCM_AITER_UNIFIED_ATTN`. **The win grows with depth** — benchmark it at 2k and you will conclude it does nothing. Also yields a slightly larger KV pool. |
| `SPEC=7` (DFlash2) | **+35.1%** weighted decode over `SPEC=4` | The old "K=4 is the peak" result was a prose-corpus artifact. Match the corpus to the axis you are being compared on. |
| fusions: `RADIANCE_NORMQUANT_FUSION=1`, `RADIANCE_FP8_STREAM=1`, `RADIANCE_SKINNY_GEMM=1` | material, not yet isolated per-fusion | Check the boot log actually says `fuse_norm_quant: True` — ours was silently **off** for weeks. |
| `KV_MEM=<bytes>` | 0% speed, but it is what makes a 204800 context *safe* | See below. |
| `GPU_MAX_HW_QUEUES=1`, `HSA_ENABLE_MWAITX=1` | small, free | ROCm runtime, not radiance. Read inside the container, so they do nothing unless forwarded with `-e`. |
| `CHUNK` alignment | **0% under R4D** | The `n*block_size` alignment rule is an AITER property. A full 2×2 against KV pool size found chunk did nothing at any size. We use 2048 for the compile transient, not for speed. |

### Measured and rejected — so you don't repeat them

`HSA_ENABLE_INTERRUPT=1` (flat: −0.06/−0.18/−0.31% decode steps/s; it fights `MWAITX`, which
is the polling path) · MRv2 (−29% steps/s; `MAXSEQS=2` leaves it nothing to amortise) ·
`COMPILE_SIZES`/`COOP_RED` (the static specializations change numerics enough to cost the
drafter its acceptance) · `RADIANCE_FAST_DRAFT=1` (3.8 GiB of KV pool for +2.3% decode; the
vendor's +16.6% is a TP=2 number).

### The KV pin, which is the one that will bite you

vLLM sizes its KV pool from the free memory it profiles at boot. A **cold** boot — empty
`torch.compile` cache — sees ~2.3 GiB less than a warm one, because compile scratch is counted
as permanent. So a context sized against the warm pool works for weeks and then refuses to
start the first time the compile cache is invalidated:

```
7.8 GiB KV cache is needed, which is larger than the available KV cache memory (7.08 GiB)
```

`--kv-cache-memory` (the script's `KV_MEM`) skips profiling entirely, so the pool is identical
cold and warm. It is safe because compile peaks *before* the pool is allocated — the two peaks
never coexist. vLLM prints the value it would "fully utilize" on its own boot line; set yours
just under it. **Do not size a pin by arithmetic from another `max_model_len`**: tokens/GiB is
context-length-dependent (218 blocks/request at 131072, 308 at 204800).

### Benchmarking notes that cost us real time

- Read **`steps/s`**, not decode tok/s, for anything acceptance-neutral: it repeats to ~0.1%
  across separate cold boots, while raw decode swings ±6%. For an acceptance-*sensitive* knob
  (drafting method, `SPEC`) you need `mean_len`/accept as well, since decode = steps/s × tok/update.
- **A bigger KV pool is ~5% SLOWER on decode** (prefill unaffected). This is why a cold boot
  looks fast, and it silently inflated one of our A/B results by 4.7pp. Never compare a cold
  boot against a warm one.
- The ±0.6% prefill noise band is a **within-boot** figure. Across boots, 50k prefill spans
  ~2.8% here. Two agreeing samples from one boot are *one* sample — re-measure the control on
  its own boot, in the same session, or you will publish a phantom win. We nearly did.

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

### Where the MXFP4 path came from

**This configuration is downstream of three other people's projects.** None of the MXFP4
work is ours; the chain is worth stating explicitly because two of the three are easy to
miss when you arrive at this repo from a search engine.

**1. `ggz14` (`brian_launch80`) — the origin, and the only hard dependency.**
[codeberg.org/ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4),
mirrored to [github.com/GGZ14/vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4) (the Quick
start clones the mirror; the trees are identical). The MXFP4 kernels, the W4A8 fp8-WMMA HIP
kernel, the DFlash2 integration and `setup-mxfp4.sh` are all theirs. That repo and the
`stilldeadcode/vllm-radiance` image are **one project, not two** — its `DOCKERHUB.md` is
that image's Docker Hub page, and MXFP4 is a runtime flag in the image (`RADIANCE_MXFP4=1`,
plus `RADIANCE_MXFP4_W4A8=1`), not a separate build. The launcher here sources
`gpu-detect.sh` from that clone and mounts it as `/patches`, so it does not start without
it. Same author as [BetterBench](https://github.com/GGZ14/BetterBench), the harness every
number in `benchmarks/` comes from. Markedly understated about what is a full vendor-grade
RDNA4 stack.

**2. [`malicz/vllm-gfx1201-launchers`](https://github.com/malicz/vllm-gfx1201-launchers) — the intermediary we actually picked this up
through.** A thin config layer over ggz14's work: it names both upstreams in its README and
`curl`s ggz14's files from a pinned commit at build time rather than vendoring them, plus
one original file. The arrangement in this repo was arrived at by working through theirs,
and it is the reason the pinned-commit approach is used here at all. Credited here because
it would otherwise be invisible — nothing in the running system carries its name.

**3. [`hifi/vllm-radlight`](https://codeberg.org/hifi/vllm-radlight) — the yardstick, not a dependency.** The published R9700 MXFP4
throughput numbers this configuration was measured against and eventually passed — **83.6
vs 81.7 t/s** weighted decode, and with the vision tower **loaded**, which those numbers
drop via `--language-model-only`. (During config selection this stack measured 85.6 at
`MAXLEN=131072`; 83.6 is the same stack at the shipped 204800, and the ~2 t/s is the
KV-pool cost, not a regression — see the caveat in
[`benchmarks/deep-prefill-20260905/`](benchmarks/deep-prefill-20260905/). The number this
repo publishes is the one it actually serves.) Two ROCm runtime settings here were
taken from it and are marked as such in the launcher: `GPU_MAX_HW_QUEUES=1` and
`HSA_ENABLE_MWAITX=1`. A third, `HSA_ENABLE_INTERRUPT=1`, was tried and measured flat, and
is documented as not adopted. Reproducing someone else's published numbers before trying to
beat them is most of why the decode gap here closed at all.

### The wider gfx1201 ecosystem — the Launch80 Discord

Nearly all serious R9700 work is happening in one place: the **Launch80 Discord**. It is
where the projects above are developed and where several others are shared that have no
public repo at all. If you are running this card and hit something this README does not
cover, that community is the answer far more often than a search engine is.

Beyond the three projects this repo descends from:

| project | scope |
| --- | --- |
| [`stew675/llama-cpp-rdna-boosts`](https://github.com/stew675/llama-cpp-rdna-boosts) | llama.cpp patches for RDNA 3, 3.5 and 4 — the Vulkan side, relevant if you serve GGUF rather than vLLM |
| [`tcclaviger`](https://huggingface.co/tcclaviger) | an RDNA4-targeted vLLM fork, plus [`Qwen3.8-Flash-Next-MXFP4-FP8`](https://huggingface.co/tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8) for 4x R9700 |
| [`Dyluhn/R9V`](https://github.com/Dyluhn/R9V) | Qwen3.8-Flash-Next on 2x R9700, with [IQ4_XS weights](https://huggingface.co/Dyluhn/Qwen3.8-Flash-Next-R9V-IQ4_XS) |
| `stilldeadcode` | vLLM patches for 2x R9700, and Deepseek v4 Flash for 1-2x — shared in the Discord, no public repo |

**Card count is the thing to check first.** This repo is single-card throughout, and so is
`hifi/vllm-radlight`. Several of the projects above target 2x or 4x, where the tuning
questions are different ones — tensor parallelism, interconnect, and a memory budget that
stops being the binding constraint. Do not carry a number across a card count.

### The weights, and the rest

- **AMD** — [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4),
  the served body weights, quantized with AMD's Quark. Used as published; the only change
  is that its MTP head is requantised to fp8 by ggz14's `fp8_mtp.py`, because that
  checkpoint names its `mtp.*` layers as *tensor* names inside a list of *module* names, so
  Quark's own exclusion never fires. That is a bug worked around, not a criticism of the
  weights.
- **`tcclaviger`** — [`Qwen3.8-27B-DFlash2-FP8`](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8),
  the DFlash2 draft head this path speculates with. Not trained or quantized here. They also
  maintain an RDNA4-targeted vLLM fork of their own, and MXFP4-FP8 Qwen3.8-Flash weights for
  multi-card R9700 setups — neither is used here, but both are worth knowing about if your
  card count is not one.

### Shared, and the int4 path

- **`stilldeadcode`** — the [`vllm-radiance`](https://codeberg.org/StillDeadcode/vllm-radiance)
  image both paths run in, the gfx1201 kernel work in it, and
  [`libr4d`](https://codeberg.org/StillDeadcode/libr4d), whose version the MXFP4 launcher
  pins for **correctness** rather than speed. Bug reports about the *image* belong on its
  issue tracker there, not here.
- **The vLLM project** — everything under `patches/` is a derived work of vLLM,
  Apache-2.0, SPDX headers intact. None of it is original code. `patches/dflash2/` is
  upstream **PR #52816** back-ported; `patches/radiance-0.9.3/` is two of the radiance
  image's own vLLM 0.27.1 files with a marked local hunk each; `patches/rdna_hybrid_w4a16.py`
  is vLLM's own kernel with a per-shape override table added. Every file names the base it
  was copied from and comments each deviation in place.
- **The int4 checkpoints** — [`devan-carlin/Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound)
  (the served weights) and [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
  (the DFlash2 draft model). Neither was quantized here; both are other people's work and
  carry their own licences on the Hub.
- **[BetterBench](https://github.com/GGZ14/BetterBench)** (Apache-2.0) — the benchmark
  harness behind `benchmarks/int4-20260829/betterbench-full.*`. Not vendored; run against
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
Full requirements and the install walk-through are under [Quick start](#quick-start).
