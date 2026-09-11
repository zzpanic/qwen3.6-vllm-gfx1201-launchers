# KV-Cache Future Work

Future-work plan for the two-tier KV offload + GDN (Gated DeltaNet) recurrent-state reuse, built on the N=8 stride store. Grounded in the DASC/DAMP investigation and the ReplaySSM analysis. See `status-2026-09-10.md` for the current snapshot (`status-2026-09-09.md` is superseded and kept only as history) and `cache-preemption-patch-plan.md` for the existing patch plan.

> **This is the plan, not the record.** The mechanism in §2–§3 is what the work is *for*;
> the numbers scattered through it are the state of the box at the time of writing. Anything
> already decided — and several things here have been — is in
> [`kv-cache-closed-decisions.md`](kv-cache-closed-decisions.md), including a table of the
> claims this project has withdrawn. Check it before acting on a number.

---

## 0. Where we are
- **Two-tier offload (RAM + disk)** for KV + the linear-attention state: **built and measured.** The tier earns its keep: ~2.45M tokens served from offload ≈ **~26 min of prefill avoided** at the honest cold rate (~1,555 t/s).
- **Stride-N=8 state store** (`RADIANCE_MAMBA_STORE_STRIDE=8`, 0.417x bytes/token): **applied.** Keep only when `(abs_chunk_idx + 1) % 8 == 0`; lookup rounds to `N×tokens_per_chunk` (i.e. the nearest kept boundary *at or before* the requested position).
- **No number here taken before 2026-09-10 is citable**, including the "tier earned its keep" line above. The offload tier was serving blocks under keys it had not confirmed (R3.15, fixed); the harness was sound, the engine under it was not. Re-measurement is `status-2026-09-10.md` §4.
- **Model shape:** `qwen3.8-27b-vllm`, **MXFP4 weights (AMD Quark) + FP8 KV**, GDN hybrid (64 layers, 16 Gated-Attention + 48 Gated-DeltaNet, 3:1), AMD R9700 / gfx1201.
- **DASC / DAMP:** DASC (arXiv 2608.30386, Meituan) = state *compression* via weight-derived, per-unit **retention-horizon selection** + **suffix refresh**; run on **unquantized BF16**. DAMP (arXiv 2608.27513, same group) = state *quantization* (mixed-precision). No clean public unmerged DASC PR found — most likely in a private/internal branch or a fork not yet publicly PR'd (paper is ~9 days old).

---

## 1. The bar (re-stated)
The contributor is **not an AI researcher or a software developer.** The deliverable is a **best-effort reimplementation with clearly-stated limitations** — a **working, measured baseline** with an **explicit, single, well-described gap** to DASC's full method — not a research-grade DASC reproduction. It must be something others can pick up and extend.

**Shippable best-effort (feasible for a non-researcher):**
1. Two-tier offload (RAM + disk) — already built & measured.
2. Stride-N=8 state coarsening — already applied.
3. **Reuse suffix refresh** — the work in §2–§3.

---

## 2. The reuse suffix-refresh mechanism (the key finding)

### 2.1 Two different "L"s — do not conflate
- **ReplaySSM's ring (L = 16, default).** This is a **decode-latency optimization** (speculative-decode verify + rollback), *not* a reuse-correctness number. It holds the last 16 decode steps' `(d, k, g)` inputs so each step can *replay* instead of writing back the full state.
  - SGLang: `--enable-gdn-replayssm-spec` (default off), ring length `--linear-replayssm-cache-len`, **default 16** (PR #28695, closed/merged).
  - vLLM: `--replayssm-buffer-len` (B), sized `B + T`, flushes when `history + T > B` (T = `num_spec_tokens + 1`).
  - It is a **performance knob** (buffer sizing: how sensitive are the gains, what's a safe default). 16 ≈ "the last few decode steps."
- **The reuse suffix refresh (DASC-style).** Replaying a bounded suffix to re-establish *omitted* units when a prefix is reused. **No single fixed number** in general — it is a bounded window sized to the **retention horizon of the omitted units** (short-horizon / fast-decaying units need only a few tokens to re-establish — that is *why* you only omit short-horizon units). DASC's exact value is a tunable bounded parameter (in the paper's method section).

### 2.2 The stride-gap insight: for the stride store, L is *derived*, not tuned
Because the N=8 store keeps a **full (exact) state at each kept boundary**, serving a position P means: take the checkpoint at the nearest kept boundary *at or before* P (≤ 8 chunks behind P), and **replay the tokens from that boundary to P** using the exact `(d, k, g)` inputs.

- **L ≤ one block = 8 chunks = 8 × tokens_per_chunk tokens.** The "approximately" is the round-down: the gap is **0 to 8 chunks** (avg ~4).
- This is an **EXACT reconstruction**, not an approximation: exact checkpoint + exact inputs → exact state at P. (Assuming the recurrent update is deterministic and the conv window is included — it is; the GDN checkpoint bundles the recurrent state **and** the convolution windows.)
- **Therefore the reuse L is not a heuristic.** It is **set by the stride itself** (≤ one block). This is more defensible than a hand-picked window — the bound is *derived*, not tuned.

### 2.3 Checkpoint-precision nuance (state this in the deliverable)
The **gap replay is exact *given* the checkpoint**; the **result's precision = the checkpoint's precision**:
- Offload tier stores states **exactly (bf16)** → the reconstructed state is **exact**.
- Offload tier **quantizes (int8, cf. PR #28185)** → the result carries **quantization error** (not gap error). Either way, **one block is the correct suffix length.**

> Consequence: the *length* of the refresh is independent of checkpoint precision; only the *error floor* depends on it.

---

## 3. The plan (what to do)
1. **Wire in the reuse suffix refresh at `L = the stride gap ≈ one mamba block (8 chunks = 8 × tokens_per_chunk tokens).`** Use the existing ReplaySSM "replay raw `(d,k,g)` to advance/reconstruct state" primitive (the closed-loop exact fold), applied at **reuse time** (prefix lookup) rather than decode time. The checkpoint already bundles recurrent + conv state, so one replay pass covers both.
2. **Confirm `tokens_per_chunk`** for the live config so `L` is a concrete token count (currently expressed as `8 × tokens_per_chunk`; e.g. chunk=64 → up to ~512 tokens).
3. **Verify exactness** (bf16 checkpoint): reconstructed state at P should match the un-strided state at P to within fp rounding. (This is the parity check that proves "sufficient in general" is actually *exact*.)
4. **Optionally set a watch** on sglang/vllm PRs for the DASC terms (retention-horizon / ragged / suffix-refresh) so the Meituan PR is folded in if/when it lands. Low priority; the best-effort stands on its own.

**What we deliberately skip (the single named gap to DASC):** DASC's **weight-derived, per-unit retention-horizon selection** (the decay-aware, per-head/per-channel omission) — the research-core behind DASC's 2.63× / 42.6% / 68.4%. We approximate with the **uniform stride** instead. The named gap to close: *"derive retention horizons from the gate weights, per the DASC method."*

---

## 4. Limitations to state up front (so it is not mistaken for a DASC benchmark)
1. **Uniform coarsening (stride), not decay-aware per-unit selection** → our compression / TTFT numbers are **not comparable** to DASC's 2.63× / 42.6% / 68.4%.
2. **Quantized config** (MXFP4 weights, FP8 KV; and int8 state if the tier quantizes) vs **DASC's unquantized BF16** → deltas are **not directly transferable**.
3. **Single model** (Qwen3.8-27b) on **single HW** (AMD R9700), not DASC's test set (e.g. Qwen3-Next 80B) or benchmark mix.
4. **Reuse L is the stride gap (≤ one block)** — derived, exact for bf16 checkpoints, but *not* DASC's tuned bounded suffix.
5. **The BetterBench numbers are invalid** (polluted) and are not cited; the quantitative claim is the "tier earned its keep" measurement (~2.45M tokens served ≈ ~26 min prefill avoided).

---

## 5. Open items / decisions
- **The §5 "what was 'there'" question** (from `status-2026-09-09.md`): what the MXFP4/FP8 config the user saw was presented as (eval setup vs target/deployment note) — determines whether DASC-style numbers are directly ours or a BF16 baseline to re-baseline onto. Still open.
- **Confirm `tokens_per_chunk`** → make `L` a concrete token count.
- **Go / go-not** on wiring in the reuse refresh (item 3.1).

---

## 6. Measured metrics & benchmarks (the baseline)
> The **valid, measured** numbers for the current implementation. The **BetterBench numbers are invalid (polluted)** — shown in the last table only as the *evidence* of the pollution, never as results.

### Tier capacities
| Tier | Size | Tokens | Notes |
|---|---|---|---|
| **L1 GPU** | 32 GB VRAM, `--kv-cache-dtype fp8` | **~228,737** | store path reads 74 KB/token vs the load path's 35 KB/token |
| **L2 RAM** (`/dev/shm`) | **24 GiB** | **~419,000** | at 61,440 B/token as stored; confirmed live by `kv_offload_tier_capacity_bytes{tier="cpu"}` |
| **L3 disk** (fs) | 512 GB zvol | — | store rate **44.6 MB/s** |

### Tier transition / I/O latencies
| Transition | Time | Note |
|---|---|---|
| **L2 (RAM) hit** | **1–2 s** | the fast follow-up path the stride buys |
| **L3→L2 (fs→CPU) promotion** | **64.26 s** (pre-fanout) | single-thread serial read of ~200 × 27 MB files at queue depth 1; **fixed by the fs-fanout patch** (256 MiB budget → 8 batches per promotion, 4 per store) |
| **fs read** (512 GB zvol, O_DIRECT) | **~117 MB/s** | device-limited, and unchanged by fanout. Break-even against recompute is ~101 MB/s, so the tier runs at **1.16×** — see `kv-cache-closed-decisions.md` §3 |
| **fs write** | **1,100 MB/s** (1 thr) | writes never the limit — the tier only stores 44.6 MB/s |

### The "tier earned its keep" headline — method only, not citable (pre-2026-09-10)
- **~2.45M tokens** served from the RAM/disk offload tier instead of recomputed.
- ≈ **~26 min of prefill avoided** at the honest cold rate (**1,555 t/s**). (2.45M ÷ 1,555 ≈ 1,576 s ≈ 26.3 min.)
- Internally consistent: `prompt_tokens_total` climbed ~1.85M while ~4.7M tokens were sent; the **~2.45M gap = tokens the tier served**.

### The N=8 stride effect (the capacity lever)
- **0.417× bytes** per 8 chunks (72 → 30 units).
- L2 capacity at that density: **24 GiB ≈ 419,000 tokens**, ~1.83× the GPU cache.
- Mamba store cost: **27 MB / group / chunk**; a long conversation = **~70 snapshots** (the GPU itself keeps 2).
- Cost: prefix hits truncate down to an **N-chunk (13,184-token)** boundary.

### Why the reaper is mandatory (fs-tier growth)
- Under agentic load: `/kvcache` went **55 GB → 186 GB in 39 min (~3.4 GB/min)**; worst-case store ~45 MB/s.
- Reaper: 5-min cadence, target 65%, max-age 8 h, min-age 90-min floor, max-delete 4000/run.

### Pollution evidence (why the BetterBench numbers are invalid — for the record, NOT as results)
| Body at 32k | Time | t/s |
|---|---|---|
| repeating `_PARA×n` (self-caches) | 9.1 s | 3,520 |
| unique non-repeating (true cold) | 20.6 s | 1,555 |
- **2.3× ratio**; 2.15M GPU + 2.45M external prefix-cache hits observed during the run. Root cause: the offload tier's mixed-hit path served keys it had not confirmed (R3.15, fixed). The A/B that produced the 9.1 s / 20.6 s pair was itself taken on defective code, so it shows the *direction* of the pollution and not its size; re-running it is `status-2026-09-10.md` §4 item 2.

### Reference numbers (DASC / others — NOT ours, for the gap)
- **DASC** (run on unquantized BF16): 2.63× compression, 42.6% lower TTFT, 68.4% higher input throughput.
- **vLLM #38230**: 97% same-session hit rate (offload plumbing).

---

## 7. References
- **ReplaySSM** — Dao AI Lab 2026 (blog + `github.com/Johnny-Liou/ReplaySSM`, based on vLLM `193ce8812`); SGLang RFC **#28511**, PR **#28695** (ring spec-verify, `--linear-replayssm-cache-len` default 16), **#28451** (buffered output-only decode); vLLM RFC **#49232** / **#46187** (`--replayssm-buffer-len`, sized `B+T`, flush when `history+T>B`).
- **DASC** — arXiv **2608.30386** (Meituan): retention-horizon unit selection + suffix refresh; 2.63× / 42.6% / 68.4%; run on unquantized BF16.
- **DAMP** — arXiv **2608.27513** (same group): mixed-precision recurrent-state quantization.
- **vLLM #38230** — hybrid KV-offload planner + MultiConnector for the Qwen3.5 family (offload plumbing, 97% same-session hit rate).
- **PR #28185** — int8 checkpoint pool for the linear-attn prefix cache (state quantization, DAMP-side).
- **LMCache hybrid-models** (shipped) — supports Qwen3.5/3.6 GDN series; reinterprets recurrent-state caches as opaque pages.
- **Qwen3.8 day-0 (LMSYS, 2026-08-12)** — GDN checkpoint bundles recurrent state + convolution windows; ReplaySSM fold kernel for MTP.
- Local: `patch_kv_offload_mamba_stride.py` (the N=8 stride patch), `status-2026-09-09.md`, `cache-preemption-patch-plan.md`.
