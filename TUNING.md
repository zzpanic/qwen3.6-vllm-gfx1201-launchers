# Tuning log — Qwen3.8-27B on gfx1201

Dated notes on what was changed and why, so a default that looks arbitrary can be traced to
the measurement that set it. Split out of the README to keep that file about *running* the
launchers.

Short dated notes on what was changed and why, so a default that looks arbitrary can be
traced to the measurement that set it. Everything here was measured on **1x R9700**
(gfx1201, TP=1). Benchmarks are llama-benchy, `--pp 2048 --tg 256 --runs 3 --concurrency 1
--no-cache`, on the second boot after any graph change (see the cold-compile trap below).

### 2026-08-29 — `BATCHTOK` default 2560 -> 3240 -> 4854, and the alignment rule

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

**The shipped default is now 4854, not 3240** — `3 * 1616 + 2 * (4-1)`, i.e. the same rule
one block further out, matching what this rig actually runs. Be clear about the evidence:
the table above is a measured A/B and it is what establishes the *rule*; the 3240 -> 4854
step itself was not independently benched, it is the rule extrapolated by one block. If you
are porting this, the rule is the transferable part — recompute it for your block size and
your `K` rather than copying 4854, which is only correct at block 1616 with `SPECTOK=4`.

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

### 2026-08-29 — `SPECTOK` stays 4, and the reason is workload, not speed

Swept K in {4,5,6,7} at `MAXLEN=131072`, dflash2 speculative decoding, `MAXSEQS=2`, greedy,
single-stream, 8 runs per category. Earlier testing here compared only 4 against 7 at
204800 and concluded "4 wins". The full sweep says something more useful: **in aggregate K
barely matters, and the choice should be made on what you actually run.**

| category | K=4 | K=5 | K=6 | K=7 | K=4 -> K=7 |
| --- | --- | --- | --- | --- | --- |
| math | 97.0 | 92.3 | 102.1 | **110.5** | **+13.9%** |
| code | 84.5 | 74.8 | 82.8 | **85.6** | +1.3% |
| reasoning | **70.6** | 61.5 | 66.6 | 68.1 | -3.5% |
| chat | **65.5** | 59.6 | 60.2 | 59.7 | **-8.9%** |
| combined | **79.0** | 69.5 | 76.3 | 78.6 | -0.5% |

decode t/s, median of 8 runs.

Math and chat move in **opposite directions**, and the aggregate looks flat only because
they cancel. Math output is predictable enough that the drafter's 6th and 7th tokens land
often enough to pay for themselves; chat is the least predictable, so those tokens get
generated, verified, and thrown away. **A math- or code-heavy server should run K=7; a
chat-heavy one should run K=4.** This homelab is chat- and reasoning-heavy and both peak
at K=4, so K=4 stays. That is a workload decision, not a hardware one — yours may differ.

Acceptance decays monotonically, which is the part that generalises:

| K | acceptance | mean accepted length | marginal gain |
| --- | --- | --- | --- |
| 4 | 63.9% | 3.556 | — |
| 5 | 58.1% | 3.905 | +0.349 |
| 6 | 52.4% | 4.142 | +0.237 |
| 7 | 48.1% | 4.369 | +0.227 |

Each extra draft token is accepted less often than the last; by K=7 the marginal token is
accepted under half the time.

**K=5 is a real dip, not noise** — -12% against neighbours that sit within 0.5% of each
other. Its acceptance curve is perfectly ordinary, so this is not a speculation-quality
effect; it is scheduling or kernel shape. Unexplained, and recorded because it reproduces,
not because it is understood. Don't pick 5.

The control arm re-ran K=4 last, 50 minutes and four container restarts after the first:
decode within **0.25%**, and the speculative-decoding counters **bit-identical** (5099
drafts / 20396 draft tokens / 13031 accepted). Under greedy sampling these arms are fully
deterministic, so acceptance here is exact rather than sampled, and differences above ~1%
are real.

#### `SPECTOK` is the only lever on draft length

The image bakes a dynamic draft-length controller —
`RADIANCE_DYNAMIC_DRAFT=1` with `RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4`, i.e. propose
8 tokens at batch size 1, tapering to 4 at batch 16. Read naively that says a single-stream
request drafts 8 tokens regardless of what `SPECTOK` says.

It does not. **The schedule only ever clamps downward.** In the image's own
`radiance_draft.py` the batch ceiling is applied as
`if 0 < batch_ceil < num_speculative_tokens`, and `num_speculative_tokens` is `SPECTOK` — so
it can lower a draft below K and can never raise one above it. Every arm of this sweep is
consistent: counters at exactly K draft tokens per draft
(`num_draft_tokens_total / num_drafts_total` = 4.000, 5.000, 6.000, 7.000) at `MAXSEQS=2`,
where the schedule would have permitted 7–8. **Raising `SPECTOK` is the only way to lengthen
a draft.**

**Does the controller only fire under MTP?** Not by class, despite what it calls itself.
Its docstring and every knob name say MTP, but on `:0.9.3` it patches
`SpecDecodeBaseProposer`, and `DFlashProposer` inherits from that base without overriding
either of the two patched methods — so it is installed on the DFlash2 path as well. What
keeps it *mostly* out of the way here is the sampling method, not the speculation method:
the depth controller runs inside `_greedy_sample`, which vLLM calls only when the request is
all-greedy or probabilistic draft probs are disabled. Production runs
`DRAFT_SAMPLE_METHOD=probabilistic`, so ordinary sampled traffic never reaches it; greedy
traffic — including every arm of this sweep — does. So the counters above establish the
*ceiling*, not that the controller sat idle: it deliberately keeps the draft at full width
and fills positions it declines to draft with an n-gram tail, which a draft-token count
cannot distinguish from ordinary drafting. Read that table as "K is the ceiling", and
nothing more.

The corollary is a retraction worth keeping. An earlier note here argued from source that a
larger K was "free at the scheduler" — `max_num_new_slots_for_drafting()` returns 0 for this
config, `max_num_scheduled_tokens` equals `BATCHTOK` exactly in the log, and the schedule
clamps only downward. Every one of those observations is still true and **the conclusion was
still wrong**: the schedule governs how many draft tokens are *allowed*, not what they
*cost*. Free of batch budget is not free of time — the extra draft forwards halve the step
rate. Source reading gives a hypothesis, not a result.

#### Context cost per K

| K | block size | KV pool @ 131072 | max concurrency | longest context it can serve |
| --- | --- | --- | --- | --- |
| 4 | 1616 | 231,602 | 1.77x | 260,176 |
| 5 | 1616 | 222,575 | 1.72x | 255,328 |
| 6 | 1632 | 218,453 | 1.67x | 249,696 |
| 7 | 1648 | 212,098 | 1.62x | 243,904 |

Pool and concurrency are measured. The last column is **modelled** from the measured
geometry, calibrated against K=4 to 0.05% and K=5 to 1.2% — treat it as ~1% optimistic and
as a different class of number from the measured columns beside it.

**Every K from 4 to 7 clears 204800.** The old "K>=7 does not fit" rule was an artifact of
`KV_GROUP_SIZE=5` and died with the group-padding fix; it was a context constraint that had
been mistaken for a speed finding.

The vendor recommendation of depth 7 is sound on **2x R9700 (TP=2)** paired with a
low-latency draft head that is not viable at TP=1 here. Depth advice does not transfer
across card count.

#### Two traps this sweep walked into

**The attention block size moves with K** — 1616 at K=4 and K=5, 1632 at K=6, 1648 at K=7.
Since `BATCHTOK = n*block + 2*(K-1)`, a BATCHTOK derived from an assumed 1616 is misaligned
at K>=6, and missing by a few tokens drops a whole block. Read the block size out of the
boot log (`Setting attention block size to N tokens`) and derive BATCHTOK from that.

**Read the KV pool from a warm boot only.** Changing K changes the graph, and the boot that
compiles it reports ~2.1 GiB less available KV than the same config on a warm cache — at
K=6 that read 167,480 tokens instead of 218,453, a 24% understatement. Boot twice with the
*final* BATCHTOK and take the second number. Booting twice is not sufficient on its own: if
the second boot uses a different BATCHTOK than the first, it recompiles and is cold again,
which is exactly how the first pass at this table produced two bad rows.

### 2026-08-29 — KV cache group padding: `KV_GROUP_SIZE=auto`, +21.2% of the pool

The largest single win in this file, and it is a scheduler accounting bug, not a kernel.

vLLM groups KV layers before it sizes the cache, in
`_get_kv_cache_groups_uniform_page_size`, and it picks the group size as **`min()` of the
layer-bucket sizes**. Its own `FIXME(Chen)` calls that rule a placeholder. Under DFlash2 this
model's buckets are **48 gated-delta-net + 16 full-attention + 5 drafter**, so the rule picks
**5** — which divides neither 48 nor 16. Every bucket rounds up: 50 + 20 + 5 = **75 layer
slots for 69 real layers**.

The cost is not spread evenly, and the boot warnings actively mislead about where it is.
Per-request budget is `group_size * sum(cdiv(bytes, page))`, and the per-group block need is
wildly uneven: a full-attention group needs `cdiv(204800, 1616)` = **127 blocks**, a GDN
group in align mode needs `page*(2+SPECTOK)` = **6**. So the 16→20 full-attention padding is
about **88% of the whole cost**, and the 48→50 GDN padding is nearly free — the opposite of
what the two warnings' own percentages ("25.00%" vs "4.17%") suggest. **Weight those warnings
by the bucket's block need, not by the percentage they print.**

`KV_GROUP_SIZE=auto` (a +43-line local patch, `patches/radiance-0.9.3/vllm/v1/core/
kv_cache_utils.py`) picks the size that minimises padded slots instead, tie-breaking to fewer
groups and capped by `KV_GROUP_MAX_GROUPS` (32). Here that is **8** — 72 slots, 9 groups.

| `KV_GROUP_SIZE` | slots | KV pool @204800 | concurrency |
| --- | --- | --- | --- |
| unset (stock, `min()`=5) | 75 | 222,639 | 1.09x |
| `auto` → 8 | 72 | **269,837** | **1.32x** |

**+21.2% of the pool at byte-identical VRAM and unchanged speed** (pp2048 @ d98304 measured
1338.6 → 1338.48). Leaving it unset is byte-for-byte stock upstream, so the backout is
deleting one environment variable.

Two things worth carrying away. First, the patch is opt-in *because* the rule it replaces is
right for most models — it only pays where the layer buckets are coprime-ish, which is what a
hybrid attention stack plus a separate drafter produces. Second, a prediction model for this
(`benchmarks/spectok-sweep-20260829/maxctx.py`) matched both boots to within 0.05%, so future
`g` questions do not need a GPU window.

### 2026-08-29 — the W4A16 tile table reaches 100% census coverage (and it changes nothing)

The tuned Triton tile table in `patches/rdna_hybrid_w4a16.py` was built against an MTP-era
shape census. DFlash2 changed the shapes it sees, so a fresh runtime census was taken under
the production configuration (DFlash2, `BATCHTOK=4854`, `SPECTOK=4`): 182 distinct keys,
30,720 calls, in `benchmarks/w4a16-census-20260829/`.

Two gaps showed up, both structural rather than marginal:

- **Bucket 64 was entirely empty.** The table had 5 rows at bucket 32 and 5 at MBIG and
  nothing between, while DFlash2 puts real traffic at M = 45/59/64. It is now 8/8 keys.
- **`K25600xN5120` had no row at any bucket** — the DFlash2 drafter GEMM, which the table
  predates entirely, and **87% of all uncovered work**. 25600 is `num_hidden_layers(5) x
  hidden_size(5120)`, confirmed against the checkpoint's `fc.weight_packed [5120, 3200]`.
  Stock runs it at 31.11 ms where a tuned `(256,128,32,8,ns=2)` needs 14.43 ms.

14 rows were installed and census coverage went **0.61% uncovered → 0.0000%**.

**This closes coverage, not throughput. Do not expect a benchmark to move, and do not read a
flat result afterwards as a regression.** The uncovered work was 0.61% of GEMM *flops*
against a 0.2–0.6% prefill noise floor; flop-weighting understates it slightly, since a shape
running 2.16x slow eats time in proportion, so call the whole change ~0.6% of GEMM time with
about half of that recoverable. Precedent: the Ornith 35B sweep produced isolated-kernel wins
of +3.6% to +86.1% that did not survive an end-to-end A/B.

**The measurement trap, if you re-run the sweep.** `bench_gapfill_gs128.py:current_cfg()`
replicates the *stock* gfx12x heuristic — it does **not** consult the installed table. Its
`pct` column therefore means "win over stock", which equals "win over what you are running"
only for keys that have no installed row. `K17408xN5120 M=4848` reports +96.2%, and
production already banks nearly all of it. **Rows were installed only where no incumbent
existed**; the five already-tuned keys were left alone, because deciding those needs a
head-to-head against the installed config that this sweep never timed. Buckets 128 and 512
are still deliberately empty: observed M jumps straight from 64 to 1616.

### 2026-08-29 — fp8 KV is on; KV *scales* are a separate thing, and we have none

`--kv-cache-dtype fp8` is hardcoded in the launcher and is where the context budget comes
from. What it does **not** imply is per-tensor or per-head KV scale factors, and the two get
conflated:

- **This checkpoint ships zero KV-scale tensors** — 0 of 2,013 — so K and V are stored fp8
  against an **implicit scale of 1.0**. Nothing is being calibrated.
- **The finer-grained option is unreachable on our backend.** `fp8_per_token_head` is
  `TRITON_ATTN`-only; radiance's R4D attention advertises exactly
  `["auto","bfloat16","fp8","fp8_e4m3"]`. Taking it would mean giving up R4D, i.e. paying the
  +7.2% deep-prefill win for it.
- **There is no VRAM in it either way.** Scales are per-tensor or per-head metadata, not a
  change in how the cache is stored, so this is a quality question and not a capacity one.

Closed as a dead item, recorded here so it does not get re-opened as an obvious-looking win.

The one fp8-KV fact that *is* load-bearing: the fp8 KV path is **stock upstream** inside
radiance (see the next section), so upstream fp8-KV fixes arrive with an image bump rather
than needing a patch here.

### 2026-08-28 — the image, the fork, and every deviation this repo adds on top

Three layers run here, and it is worth being explicit about which is whose, because "it's
slow / it's wrong" has a different owner in each.

**1. vLLM.** The base. `vllm-radiance:0.9.3` is built on **vLLM 0.27.1**; the older
`:0.5.8` is **0.26.0**.

**2. radiance.** Not a repackage — a real out-of-tree gfx1201 fork. It applies idempotent
string patches to a stock vLLM install and ships its own compiled modules, and the split
decides whether an upstream fix ever reaches this card:

| area | who owns it | consequence |
| --- | --- | --- |
| fp8 **weight** GEMM | radiance (`block_scaled_mm`, returns before upstream code runs) | upstream improvements land in a function this card no longer calls. Moot for us: we are int4 W4A16, so the path is inert. |
| attention, fp8 **KV** | stock upstream | upstream fp8-KV fixes arrive with an image bump. |
| `rdna_hybrid_w4a16.py` (the file this repo's tile table edits) | **untouched by the fork** | 0.9.3's copy is byte-identical to 0.27.1's, so the local tile edit carries across image bumps with no rebase work. |

Two further pins matter. The **kernels are a separate repo** (`libr4d`) pinned by
`R4D_VERSION` — 0.9.3 pins v0.5.0 — and that pin is a **correctness pin, not a performance
pin**: v0.4.0 had three exponent overflows in the GDN kernels that produced `0 * INF = NaN`
across a whole head, WikiText-2 perplexity **653,586** against 8.37 fixed. Check it on every
radiance bump. And the tag lies about itself: image `0.7.4` reported `RADIANCE_VERSION=0.6.2`
in its own environment, so the **image tag is the only reliable identifier**.

**3. this repo.** No file in `patches/` is original work. Every one of them is a copy of an
Apache-2.0 vLLM source file — sometimes as the radiance image ships it — with a small number
of local hunks added and marked in place. SPDX headers are intact, each file names the exact
base it was copied from, and the deviations are individually commented. What this repo
contributes is the *measurement and the configuration*, not the code. Two kinds of local
change, kept apart on purpose:

- `patches/dflash2/` — a 10-file **backport** of upstream PR #52816 onto vLLM 0.26.0.
  Needed *only* on `:0.5.8`. On 0.9.3 DFlash2 is native and mounting this would lay a second
  copy of the PR on top of the image's own; the launcher now refuses that combination outright.
- `patches/radiance-0.9.3/` — patched **copies of 0.9.3's own files**, two of them:
  `v1/core/kv_cache_utils.py` (the `KV_GROUP_SIZE` knob above) and
  `model_executor/models/qwen3_dflash.py` (loads a *packed int4* draft checkpoint —
  merged-upstream DFlash2 reads `.weight` directly and dies with `AttributeError:
  'QKVParallelLinear' object has no attribute 'weight'`; the fix routes to the identity-matrix
  path fp8 already used). Every local hunk is marked in-file.
- `patches/rdna_hybrid_w4a16.py` — the tuned tile table, orthogonal to both.

**An overlay directory is version-bound, and that is the trap.** These are copies, not a
diff: bumping the image tag silently reverts the image's own fixes in exactly those files
back to their 0.9.3 state, with no error. **Rebase by diffing this overlay against the new
image's stock file**, never by replaying the patch blind.

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

