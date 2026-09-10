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

## 3a. Upstream applicability sweep (measured against the installed tree)

The first two sweeps of this ground found nothing, because they judged candidates by
**merge status** and looked only at marquee RFCs. Redone as an *applicability* test — fetch
each PR's diff and count how many of its removed lines still exist verbatim in the installed
tree — the picture inverted. Percentages below are that measure, per file.

| PR | What it is | Applies | Ruling |
|---|---|---|---|
| [vLLM #54743](https://github.com/vllm-project/vllm/pull/54743) | [KV Offload] Scope offload group configs to prefix-cacheable KV cache groups | `offloading/config.py` **100%**, `offloading/scheduler.py` **80%** | **The one that matters for the stride work.** Not a fix for us as-is (all nine of our groups *are* prefix-cacheable, so it changes nothing we can feel), but it adds exactly the primitive the stride patch needs: `config.groups` becomes a *filtered* tuple and `OffloadingGroupConfig.group_idx` preserves the original index, so `GPULoadStoreSpec.group_sizes` and the offload keys stay correctly indexed after filtering. "Offload group g only every Nth chunk" then becomes a predicate on a reviewed data structure instead of an invasive rewrite. **Build on it, not beside it.** |
| [vLLM #54327](https://github.com/vllm-project/vllm/pull/54327) | [Feature][KV Offload] Bounded capacity and LRU eviction for the filesystem tier | `tiering/fs/manager.py` **100%** (+255/−7) | **Independently deployable, and it would delete a component of this proof of concept.** The fs tier here is unbounded, which is why `kvcache-reap.sh` and its timer are a *mandatory* external dependency. This adds capacity and LRU eviction inside the tier. Test it early; if it holds, the reaper becomes legacy. |
| [vLLM #49225](https://github.com/vllm-project/vllm/pull/49225) | [KV-offload][FS] Batching for read/write threads | manager 29%, thread_pool 65% | **Ported by hand** as `patch_kv_offload_fs_fanout.py` (manager-only, five edits). Mechanism exact, gain nil on this device — see `kv-cache-historical.md` §5. Note upstream's own 32 MiB budget would give a fanout of 2 here, so the budget was made a knob. |
| [vLLM #55390](https://github.com/vllm-project/vllm/pull/55390) | Annotate MTP draft KV cache groups positionally on the hybrid grouping path | `kv_cache_utils.py` 11% | **Our live eagle-group bug.** Little of the diff applies (radiance has its own `_annotate_eagle_groups_deepseek_v4`), but it is small (+45/−18) and confirms the fix shape. **Port the idea, not the diff.** |
| [vLLM #55519](https://github.com/vllm-project/vllm/pull/55519) | Don't warn that prefix reuse is disabled when the EAGLE block drop is off | `kv_cache_utils.py` 100%, scheduler 0% | Cosmetic here; applies cleanly. |
| [vLLM #54756](https://github.com/vllm-project/vllm/pull/54756) | Register mixed page sizes in one cache group | 5% | Targets `v1/simple_kv_offload/`, absent in this tree. No. |
| [vLLM #38261](https://github.com/vllm-project/vllm/pull/38261) | (earlier offload branch) | `offloading/scheduler.py` 10%, `cpu/spec.py` 0% | **Out of date against everyone**, not just us: it targets `kv_offload/{spec.py, mediums.py, worker/cpu_gpu.py}`, a subtree since refactored to `{base.py, config.py, cpu/, tiering/, factory.py, file_mapper.py}` — and our layout now matches upstream main exactly. Porting it forward is most of the work of the stride revision without doing what the stride revision does. (An earlier note recorded this PR as "261 files"; that was a branch-vs-main compare. It is **17 files**.) |

**The lesson, stated once because it generalises:** scan upstream by **applicability against
the installed tree**, not by merge status or issue prominence. The useful work was in small,
open `[Bugfix][KV Offload]` PRs updated within the last week — not in the headline RFCs.

---

## 3b. Scheduling, admission control, and KV-cache thrashing (the literature)

Added 2026-09-10, prompted by a measured result: on this box, **the third concurrent deep-context
agent halves effective prefill throughput and triples recompute work**, while average cache coverage
barely moves (96% -> 93%). Full measurement in `$HOME/audit/concurrency-and-maxseqs-20260910.md`.

Pat's read at the time — *"my gut tells me that this is a researched topic with good papers available
to explain the maths and breakeven points"* — was correct. It is a named, actively-published phenomenon:
**"KVCache thrashing"**. The offload tier is only half the problem; the other half is *admission and
eviction order*, and that half is where the recent work is.

Scored by **applicability to this deployment**, not by venue or citation count.

| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [CacheWise — arXiv 2606.16824](https://arxiv.org/abs/2606.16824) | *Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents* (Tiwari, Chugh, Rickert, Peter, Mahajan, Shen; 15 Jun 2026). Real coding-assistant traces (CATraces); prefix-aware scheduling + reuse-aware eviction predicted from tool-call metadata. Implemented **in vLLM**. | **VERIFIED against the paper's full HTML text.** | **The closest match to this workload that exists, and the top read.** Formalises our pool arithmetic exactly: working set `W(t) = Σ d_i` over live sessions, eviction-free only while it fits the budget. Names our mechanism: FCFS *"expands the active KVCache working set, increasing the likelihood of evicting prefixes that will soon be needed again"*; LRU *"cannot distinguish"* a session about to resume from one that will idle. Reproduces our load curve: *"token goodput decreases with increasing load... more sessions compete for GPU memory, causing more KVCache evictions."* Gains: **1.38x–1.64x goodput from prefix-aware scheduling alone**, 2–2.6x fewer evictions, up to 3.5x faster session completion. Two things to take: (a) the term **"token goodput"** — throughput net of recompute and KV movement — which is the correct name for what our harness calls effective prefill tok/s, and (b) the scale-transfer insight: they hit this at N=30–40 sessions on datacenter hardware, we hit it at **N=3**, because the governing ratio is `W(t)`/capacity, not session count. Their conclusion applies unchanged; only the axis label differs. **Caveat on provenance:** an automated PDF summary of this paper produced four numbers (a "2.5–3x working set" sizing rule, thrashing at "8–16 sessions", "90% of hits in the most recent 30% of tokens", a "40% miss rate" break-even). **None appear in the paper.** All four were fabricated by the summariser; do not quote them. |
| [PEEK — arXiv 2607.02525](https://arxiv.org/html/2607.02525) | Radix tree maintained over the **waiting queue**, driving cluster-aware admission plus an eviction hook that protects blocks queued requests will reuse. | Search-result summary; not read in full. | **The most directly implementable idea for us.** Our failure is precisely that eviction is blind to what is *queued*: we evict a live agent's prefix to seat a request that will itself be evicted. A queue-aware eviction hook is a smaller change than full predictive eviction and needs no tool-call metadata. **Read next after CacheWise.** |
| [Prediction-based KV-Cache Management for Dynamic Agent Workflows — arXiv 2605.06472](https://arxiv.org/pdf/2605.06472) | Predicts reuse distance in multi-agent workflows to drive retention. | Search-result summary. | Same family as CacheWise's eviction half, framed for agent workflows rather than coding specifically. Worth reading for the prediction signal — ours would have to come from the agent harness, since we have no tool-call metadata at the server. |
| [TokenDance — arXiv 2604.03143](https://arxiv.org/pdf/2604.03143) | Agentic serving; scheduling around tool-call gaps. | Search-result summary. | The tool-call gap **is** our eviction window — work is lost between turns while an agent holds no in-flight request. Relevant to why "let the in-flight request finish" is the right instinct. |
| [ThunderAgent — arXiv 2602.13692](https://arxiv.org/pdf/2602.13692) | Agentic-workload serving system. | Search-result summary. | Same cluster; lower priority than the three above. |
| [CacheWise's baseline: InferCept](https://arxiv.org/abs/2606.16824) | Interception-aware serving, used as CacheWise's comparison point. | Named in CacheWise. | Useful only as the baseline CacheWise beats; not a candidate here. |
| [PrefixShield — arXiv 2608.01657](https://arxiv.org/abs/2608.01657) | Admission/eviction protection for shared prefixes. | Search-result summary. | Adjacent; read if PEEK's approach looks promising. |
| [CacheRoute — arXiv 2608.19677](https://arxiv.org/html/2608.19677) | Routing requests by cache locality. | Search-result summary. | **Multi-replica.** Assumes a choice of where to send a request; we have one GPU. **No applicability** until there is a second node. |
| [CacheSolidarity — arXiv 2603.10726](https://arxiv.org/html/2603.10726v1) | Cross-request/cross-tenant cache sharing. | Search-result summary. | Multi-tenant framing; single-user box. Low. |
| [SAECache — arXiv 2605.18825](https://arxiv.org/pdf/2605.18825) | Semantic/approximate cache reuse. | Search-result summary. | Approximate reuse changes outputs. Given the correctness work already done here (see `CORRECTNESS.md`), this is **out of scope on accuracy grounds** unless explicitly revisited. |

### Break-even and offload-economics papers

Directly relevant to the standing NVMe question — the fs tier is at **1.16x** break-even
(101 MB/s to tie recompute, device does 117), which is why nothing serves from it.

| Ref | What it is | Status | Ruling |
|---|---|---|---|
| [CacheFlow — arXiv 2604.25080](https://arxiv.org/html/2604.25080v1) | Offload scheduling and tier economics. | Search-result summary. | Read for the break-even formalism — we derived ours empirically and should check it against a published one. |
| [KVServe — arXiv 2605.13734](https://arxiv.org/pdf/2605.13734) | KV serving with tiered storage. | Search-result summary. | Same. |
| [Adaptive KV Cache Reuse — arXiv 2605.24022](https://arxiv.org/pdf/2605.24022) | Adaptive reuse-vs-recompute decisions. | Search-result summary. | This is the decision our tier makes implicitly and badly (`pending-is-miss` drops 61% of found hits). Worth reading against that defect. |
| [HERALD — arXiv 2606.21633](https://arxiv.org/pdf/2606.21633) | Hierarchical KV management. | Search-result summary. | General tiering; medium. |
| [OrbitFlow — arXiv 2601.10729](https://arxiv.org/pdf/2601.10729) | Serving-system scheduling. | Search-result summary. | Medium. |
| [TokenCake — arXiv 2510.18586](https://arxiv.org/pdf/2510.18586) | KV/memory partitioning. | Search-result summary. | Medium. |
| [backend.ai — KV cache offloading](https://www.backend.ai/blog/2026-04-how-to-save-gpu-memory-in-llm-serving-kv-cache-offloading) | Vendor engineering blog on offload economics. | Read. | Carries one very useful datapoint: **128K TTFT 11 s -> 1.5 s** with a precomputed cache — on a DGX SuperPOD with 400 Gbps RDMA. Ours is **44.9 s -> 1.7 s** on a consumer card. The *cached* path is essentially the same absolute time on both; the difference is entirely in what a miss costs. That is the clearest available argument that our effort belongs on **hit rate and residency**, not on transfer speed. |
| [vLLM — KV offloading connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) | Upstream's own writeup of the connector we run. | Read. | Background for the installed design; no new lever. |

**What this literature changes about our plan:**

1. **Admission control is a first-class lever we have not pulled.** Every measured intervention so far
   has been on the offload path (stride, fanout, batching, `pending-is-miss`). CacheWise gets
   1.38x–1.64x from **scheduling alone**, without touching the tier. Our equivalent first move is
   `--max-num-seqs 4 -> 2`, which costs nothing to try.
2. **The metric to report is token goodput**, not raw prefill throughput — it is the one that makes
   thrashing visible. Fold this into the metrics spec (the author-local
   `kv-offload-metrics-spec.md`, listed in the file table below; its shipped successor is
   `tier-report-metrics-plan.md`).
3. **Queue-aware eviction (PEEK) is the cheapest real improvement**, because it needs no workload
   prediction — only visibility into the waiting queue, which the scheduler already has.
4. **The upstream knob we would want does not exist in this build.** `max_long_partial_prefills` /
   `max_num_partial_prefills` are absent; only `long_prefill_token_threshold` exists, and it caps
   per-step tokens for a long request rather than limiting how many long requests are admitted. A
   token-aware admission limit therefore needs a patch, not a config change.

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
- **NOT a dead end (2026-09-10):** scheduling / admission-control / thrashing literature was searched
  for the first time and is **rich** — see §3b. The search term that unlocks it is **"KV cache thrashing"**;
  "agentic KV cache" and "coding agent serving" also work. Earlier sweeps missed it because they searched
  the *offload* vocabulary only.
- **Conclusion:** no public unmerged DASC PR found → most likely a **private/internal branch or a not-yet-PR'd fork** (paper is ~9 days old).

---

## 8. Local artifacts
| Path | What it is |
|---|---|
| `<repo>/kv-cache/patch_kv_offload_mamba_stride.py` | The N=8 stride patch (the mechanism behind "cache past the nonce"). |
| `<repo>/kv-cache/cache-preemption-patch-plan.md` | The existing patch plan. |
| `<repo>/kv-cache/status-2026-09-09.md` | Investigation snapshot (this thread). |
| `<repo>/kv-cache/kv-cache-future-work.md` | The future-work plan (this thread). |
| `$HOME/audit/concurrency-and-maxseqs-20260910.md` | **The measurement behind §3b**: the third deep-context agent halves effective prefill and triples recompute; pool holds 2.5 deep contexts vs `--max-num-seqs 4`; 63% of wall time is queue+prefill; mechanism is eviction+re-prefill, **not** preemption (2 lifetime). |
| `$HOME/audit/kv-offload-metrics-spec.md` | The BetterBench-style metrics spec (four questions, seven panels, nine traps, five counters to add). Should gain **token goodput** per §3b. |
| `$HOME/audit/kv-offload-value-CORRECTED-20260910.md` | The authoritative offload verdict; supersedes two retracted files. Carries the `prompt_per_second` metric trap. |
| `/tmp/opencode/poc_summary.txt` | The publishable markdown report (BetterBench section known polluted). |
| `<repo>/benchmarks/betterbench-full-20260908/` | BetterBench run output (results.json 589KB, results.html, results.md). **Data invalid (polluted).** |
| `$HOME/eval/betterbench/` | BetterBench v0.4.0 git checkout (up-to-date; `prefill.py` = the `_PARA×n` self-caching body, `corpus.py` = `with_nonce`, `runner.py` = prompt cycling). |
