# Tuning log — Qwen3.8-27B on gfx1201

Dated notes on what was changed and why, so a default that looks arbitrary can be traced to
the measurement that set it. Split out of the README to keep that file about *running* the
launchers.

Short dated notes on what was changed and why, so a default that looks arbitrary can be
traced to the measurement that set it. Everything here was measured on **1x R9700**
(gfx1201, TP=1). Benchmarks are llama-benchy, `--pp 2048 --tg 256 --runs 3 --concurrency 1
--no-cache`, on the second boot after any graph change (see the cold-compile trap below).

### 2026-08-29 — `BATCHTOK` default 2560 -> 3240, and the alignment rule

**The rule: `BATCHTOK = n * block_size + 2 * (K-1)`**, where `block_size` is the align-mode
attention block (1616 on the 27B; it is set by the GDN state size, printed at boot as
`Setting attention block size to N tokens`) and `K` is `SPECTOK` (that term is 0 with
speculation off).

`--mamba-cache-mode align` floors every mid-prefill chunk to a block multiple
(`vllm/v1/core/sched/scheduler.py`: `aligned_end = end // block_size * block_size`), because
GDN state is only written at chunk ends. Speculative decoding takes `2*(K-1)` slots off the
top first (`max_num_scheduled_tokens = BATCHTOK - 2*(K-1)`). So a budget that clears the
block multiple by fewer than `2*(K-1)` tokens **silently drops an entire block per step** —
no warning, no error, just a slower prefill and activation memory spent on tokens that are
never dispatched.

The old 2560 default sat between multiples: it dispatched 1616 tokens per step and paid for
944 (37%) of nothing.

| BATCHTOK | tokens/step | KV pool @ 204800 | pp @ 32K | pp @ 98K |
| --- | --- | --- | --- | --- |
| 1624 | 1616 | 220,911 | 1512.9 | 1246.8 |
| 2560 (old default) | 1616 | 218,763 | 1517.8 | 1247.4 |
| **3240 (new default)** | **3232** | 215,879 | **1582.3 (+4.3%)** | **1315.8 (+5.5%)** |

Confidence intervals on those two prefill figures are +-1.05 and +-0.69 tok/s. Decode was
flat across all three (see the noise-floor note below). 1624 and 2560 dispatch the same
1616 tokens per step and measure within 0.3% of each other at depth — which is the direct
evidence that the wasted budget was doing nothing.

**Caution for anyone porting this:** `n * block_size` exactly *looks* aligned and is not,
under speculation. That mistake cost us a whole benchmark window once already — a 3328
value on a block-1664 model is exactly 2x the block, but the 6-token spec reserve left 3322,
which floors back to one block. It benched flat and got written up as a dead lever.
Subtract the reserve before you trust a value.

### 2026-08-29 — `--block-size` left at the floor (1616), and why bigger is a trap

In align mode vLLM sets `mamba_block_size = cache_config.block_size` and only ever *raises*
`block_size` to the mamba-page floor, so any value >= the floor is honoured verbatim and
anything below is silently raised. The floor is by construction the block that minimises GDN
page padding: **0.62%** at 1616, **101.25%** at 3232, **201.87%** at 4848.

Larger blocks are **faster** — this was worth measuring and had been assumed otherwise:

| block | KV pool @ 65536 | KiB/token | pp @ 32K | longest context it can serve |
| --- | --- | --- | --- | --- |
| 1616 | 175,624 | 56.2 | 1519.8 | 204800 |
| 3232 | 135,976 (-22.6%) | 72.6 | 1579.9 (+4.0%) | 193,920 |
| 4848 | 112,744 (-35.8%) | 87.6 | 1619.3 (+6.5%) | 169,680 |

But the pool cost is severe and the same speed is available far cheaper: block 3232 at
BATCHTOK 2560 measures 1579.9 at 32K, and block 1616 at BATCHTOK 3240 measures 1582.3 —
**0.15% apart**, because both push 3232 tokens through a prefill step. Tokens-per-step is
the lever; the block is just the expensive way to move it. Raise BATCHTOK, leave the block
alone.

### 2026-08-29 — `SPECTOK` stays 4 under dflash2 speculative decoding

Re-tested because the upstream kernel author recommends depth 7. K=7 drafts genuinely
better — acceptance chain reaches position 6, mean accepted length +18.3% — and serves
worse: decode -1.0% / -2.5% / **-6.4%** at 0 / 32K / 98K depth, KV pool -5.6%, max
concurrency down to 1.01x. The loss grows with depth, which is the depth this config exists
to serve.

The vendor recommendation is sound on **2x R9700 (TP=2)** and paired with a low-latency
draft head that is not viable at TP=1 here. Depth advice does not transfer across card
count — measure it on your own rig before adopting.

### 2026-08-23 — images capped to stop ViT OOM crashes; video untested

**Video has never been tested here, at all.** Not "tested and slow", not "works with
caveats" — never run. The homelab config passes `--limit-mm-per-prompt {"image":2,
"video":0}`, so video is switched off outright, which also skips vLLM 0.27.1's video
encoder profiling at startup. If you enable it you are the first one trying it; nothing in
this repo's numbers covers that path.

**Images need a pixel cap or the vision tower will OOM the engine.** Unpatched, this
checkpoint's `processor_config.json` caps at `size.longest_edge = 16777216`, i.e. **16,384
visual tokens for a single image**. A full-resolution image at that setting killed the
engine outright — ViT OOM at `qwen3_vl.py:2187`, `EngineDeadError`, server gone. Not a
rejected request; a dead server.

Two independent limits are worth setting, because they fail differently:

| limit | what it bounds | why it alone is not enough |
| --- | --- | --- |
| max pixels per image (4,194,304 = 4,096 visual tokens) | one image's size | a request with several legal-sized images still spikes |
| `--limit-mm-per-prompt {"image":N}` | images per request | doesn't stop one huge image |

The ViT activation spike scales with the *number* of images in a request as well as their
pixels, so the per-request limit is a second line of defence, not a duplicate of the first.

**The pixel cap has to be applied to `processor_config.json`, not by a flag.** Passing
`--mm-processor-kwargs` looks like it works and is silently inert on this stack:
`Qwen3VLProcessor.__init__` in this transformers build accepts `**kwargs` and discards
everything except `chat_template`. Patching the JSON is the only route the processor
actually reads. Re-downloading the checkpoint reinstates the original value, so the patch
needs to be idempotent and re-applied on every boot.

**The scripts in this repo do not implement either cap** — they are in the homelab launcher
these were extracted from. If you serve images with them, set the cap yourself before you
put them in front of anything real.

### Method notes worth stealing

- **The cold-compile trap.** The first boot after *any* graph change (block size, spec
  depth, batch budget) reports ~7.0 GiB of available KV instead of ~9.4 and may refuse to
  start at full context. It is not a real memory shortfall — it is the compile cache. Boot
  twice and read the second boot. If the first boot dies before it can warm its own cache,
  warm it at a shorter `--max-model-len` first, then retry at the real one.
- **`N GiB KV cache is needed` is not the pool accounting.** That figure in vLLM's startup
  error is a projection for one request at max length. Deriving bytes-per-token from it
  understated the real cost by 2x in our case (10.3% vs a measured 22.6%). Compare two
  matched warm boots instead.
- **Decode noise floor is about +-10%** at `--runs 3 --concurrency 1`. Two boots of the
  same configuration measured 69.59 and 62.98 tok/s. Prefill at depth is far tighter
  (+-1 tok/s), so prefer prefill as the discriminator and don't read decode deltas below
  the floor. None of the conclusions above rest on a decode delta.

