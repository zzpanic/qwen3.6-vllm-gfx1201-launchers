# KV-Cache References

Index of every PR, paper, blog, and local artifact pulled & reviewed in the DASC/DAMP + ReplaySSM + two-tier KV-offload investigation. For the plan see `kv-cache-future-work.md`; for the snapshot see `status-2026-09-09.md`.

Legend — status: `open` / `closed` (merged or superseded) / `commit` / `docs`. "Ruling" = why we reviewed it and the conclusion.

---

## 1. Papers (the core research)
| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [DASC — arXiv 2608.30386](https://arxiv.org/abs/2608.30386) | *Decay-Aware State Compression for Hybrid Linear-Attention Serving* — Meituan, submitted 31 Aug 2026. Authors: Yanqi Yu, Pingwei Sun, Jianchao Tan, Tao Zhang, Yuchen Xie, Xunliang Cai, Yao Liu (Ziqian Zeng, SCUT). Two mitigations: **retention-horizon unit selection** + **suffix refresh**. | paper | The headline method. 2.63× compression / 42.6% lower TTFT / 68.4% higher input throughput. **Run on unquantized BF16** — not our config. Implemented in SGLang. |
| [DAMP — arXiv 2608.27513](https://arxiv.org/abs/2608.27513) | *Decay-Aware Mixed-Precision Recurrent-State Quantization* — same Meituan group, same SGLang implementation. | paper | The **state-precision** (quantization) sibling. The closer match to our FP8 state/KV shape. |

---

## 2. ReplaySSM (the reuse / decode-replay mechanism)
| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [Dao AI Lab — ReplaySSM blog](https://dao-lab.ai/blog/2026/replayssm/) | "Cache SSM inputs, not state." Checkpoint `S0` + ring buffer of last L steps' `(d, k, g)`; output read without materializing the full state on non-flush steps. | docs | Primary source for the replay-from-raw-inputs primitive we reuse. |
| [Tri Dao — ReplaySSM blog](https://tridao.me/blog/2026/replayssm/) | Same post (author's site). | docs | — |
| [ReplaySSM research fork](https://github.com/Johnny-Liou/ReplaySSM) | Research fork based on vLLM `193ce8812`; opt-in Triton kernels (`--use-replayssm`, `--use-replayssm-spec`, `--replayssm-buffer-len`, `--replayssm-route`). | repo | Origin of the kernels; upstreaming wants unification behind `ssu_dispatch.py`. |
| [vLLM commit 3c85112](https://github.com/vllm-project/vllm/commit/3c85112) | The original ReplaySSM commit SGLang ports. | commit | The exact commit `#28511` ports. |
| [vLLM RFC #49232](https://github.com/vllm-project/vllm/issues/49232) | ReplaySSM tracking hub across vLLM / FlashInfer / TensorRT-LLM (supersedes #47572, #46187). Ring sized `B+T`, flush when `history+T>B`. | RFC (open) | Canonical vLLM design; buffer-sizing note ("how sensitive are gains to `--replayssm-buffer-len`, safe default?"). |
| [vLLM RFC #46187](https://github.com/vllm-project/vllm/issues/46187) | "Faster hybrid-SSM decode via input replay ring buffer." | RFC | Early design; folded into #49232. |
| [SGLang RFC #28511](https://github.com/sgl-project/sglang/issues/28511) | "Porting ReplaySSM to SGLang" — faster decode + spec decode for hybrid (GDN/KDA). | RFC (open) | The SGLang port umbrella (Parts A/B). |
| [SGLang PR #28695](https://github.com/sgl-project/sglang/pull/28695) | [GDN] ReplaySSM Ring Spec-Verify. Replaces per-draft full-state snapshot with a per-slot circular ring + frozen checkpoint; closed-loop exact fold; rollback = cursor move. `--enable-gdn-replayssm-spec`, `--linear-replayssm-cache-len` **default 16**. | closed (merged) | **Source of the L=16 default** (the *decode* ring — a perf knob, not a reuse number). Measures accuracy/throughput parity, −11.5 GB spec scratch (TP1). |
| [SGLang PR #28451](https://github.com/sgl-project/sglang/pull/28451) | ReplaySSM buffered **output-only** decode for linear attention. | PR | The non-spec decode path. |
| [SGLang PR #36821](https://github.com/sgl-project/sglang/pull/36821) | [KDA] ReplaySSM ring-write in the fused chain-verify kernel. | open (yuan-luo) | Extends the ring to the KDA fused verify. |
| [SGLang PR #34059](https://github.com/sgl-project/sglang/pull/34059) | [KDA] End-to-end context-parallel prefill for hybrid linear models. | open (yuan-luo) | Prefill-side CP for GDN/KDA (complementary, not the reuse refresh). |

---

## 3. KV offload (the two-tier plumbing)
| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [vLLM Issue #38230](https://github.com/vllm-project/vllm/issues/38230) | "Hybrid KV offload: MultiConnector + planner for mamba+attention models." | issue (open) | The **offload enabler** for the GDN family; 97% same-session hit rate. The plumbing our tier stands on. |
| [LMCache — hybrid models](https://docs.lmcache.ai/mp/hybrid_models.html) | LMCache hybrid-model support; reinterprets recurrent-state caches as opaque pages. | docs | **Shipped** support for the Qwen3.5/3.6 GDN series. |
| [LMCache — Qwen3.5 recipe](https://docs.lmcache.ai/recipes/qwen3_5.html) | Qwen3.5 (GDN) serving recipe. | docs | Reference config for the GDN family. |

---

## 4. State quantization / compression (reviewed)
| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [SGLang PR #28185](https://github.com/sgl-project/sglang/pull/28185) | [GDN][KDA][mem_cache] int8 checkpoint pool for the linear-attn prefix cache. | closed (yuan-luo) | The **DAMP-side state quantization** (int8 state). Relevant to the int8-checkpoint branch of §2.3. |
| [SGLang PR #31967](https://github.com/sgl-project/sglang/pull/31967) | KVarN: KV-cache compression for hybrid linear-attention models (per-tile Hadamard + Sinkhorn + int4, dual-pool compressed/tail). | open (jtabet) | **Ruled out** as the DASC PR — it compresses the **full-attention KV** (TurboQuant-family), not the recurrent state. (Sister closed PR #31676.) |
| [vLLM TPU PR #2416](https://github.com/vllm-project/vllm/pull/2416) | Compact mamba KV cache + GDN op req-slot indexing; includes a **stale-state guard**. | PR | Relevant guard pattern for serving coarse/quantized mamba states. |

---

## 5. Related state-management PRs (reviewed)
| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [SGLang PR #38000](https://github.com/sgl-project/sglang/pull/38000) | [Mamba] Thin cached states **by coverage** instead of evicting the LRU tail. | open (alphabetc1, refs #36935/#30314) | Close-but-not-it: an **eviction-policy** change (which states to evict), not unit-selection **compression**. |
| [SGLang PR #38344](https://github.com/sgl-project/sglang/pull/38344) | [Feature] Prefill context parallelism for linear attention (GDN/KDA). | open | Complementary prefill-side work. |
| SGLang yuan-luo GDN/KDA cluster (e.g. [#36696](https://github.com/sgl-project/sglang/pull/36696) mamba radix split node, [#36014](https://github.com/sgl-project/sglang/pull/36014) GDN verify beta semantics, [#36096](https://github.com/sgl-project/sglang/pull/36096) NPU SSM state stability) | Active GDN/KDA contributor work. | mixed | Context on the current GDN state-management surface; none is the DASC PR. |

---

## 6. Serving context (Qwen3.8 day-0)
| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [LMSYS — SGLang & Miles day-0 Qwen3.8 support](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support) | Qwen3.8-2.4T-A95B day-0. **Each GDN checkpoint bundles the recurrent state + convolution windows**; ReplaySSM fold kernel for MTP; Unified Radix Cache (FULL = full-attn KV, MAMBA = GDN checkpoints). | docs | Confirms the checkpoint includes the conv window (so one replay pass covers both — §2.2) and that Qwen3.8 already applies ReplaySSM. |

---

## 7. Search dead-ends (documented so we don't re-search)
- **Meituan authors as GitHub users:** `zixujiang` (DAMP co-author) → **0 sglang PRs**; `yuan-luo` (active GDN/KDA, no DASC PR); other Meituan authors not resolvable to clean GitHub handles.
- **sglang forks** (mamba/dasc/compression): `Clarit-AI/Engram`, `my-user-open/sglang-mamba-ssd--kernels`, `empty-quiver/sglang-turboquant`, `architehc/sglang_attentio` — none is a DASC implementation.
- **PR keyword sweeps** (sglang + vllm): `DASC` / `decay-aware` / `ragged state` / `retention horizon` / `suffix refresh` / `omission` / `head-aware` / `recurrent-state` — **no clean DASC PR hit.**
- **Conclusion:** no public unmerged DASC PR found → most likely a **private/internal branch or a not-yet-PR'd fork** (paper is ~9 days old).

---

## 8. Local artifacts
| Path | What it is |
|---|---|
| `<repo>/kv-cache/patch_kv_offload_mamba_stride.py` | The N=8 stride patch (the mechanism behind "cache past the nonce"). |
| `<repo>/kv-cache/cache-preemption-patch-plan.md` | The existing patch plan. |
| `<repo>/kv-cache/status-2026-09-09.md` | Investigation snapshot (this thread). |
| `<repo>/kv-cache/kv-cache-future-work.md` | The future-work plan (this thread). |
| `/tmp/opencode/poc_summary.txt` | The publishable markdown report (BetterBench section known polluted). |
| `<repo>/benchmarks/betterbench-full-20260908/` | BetterBench run output (results.json 589KB, results.html, results.md). **Data invalid (polluted).** |
| `$HOME/eval/betterbench/` | BetterBench v0.4.0 git checkout (up-to-date; `prefill.py` = the `_PARA×n` self-caching body, `corpus.py` = `with_nonce`, `runner.py` = prompt cycling). |
