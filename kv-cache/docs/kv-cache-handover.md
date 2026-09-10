# KV-Cache Handover

> **Read this first.** It orients you on the whole project, where we are, and how to resume. The detailed knowledge is split across the sibling docs in §10 — this file is the map + the current-state summary, so you do not have to re-derive context. Every section below is a pointer with enough inline fact to act without opening the target.

> **A note for readers of the release.** This is a *working* document — it was
> written for the author's own resumption, not for publication, and it is shipped
> unedited because its inline facts are worth more than its tidiness. Two
> consequences. **First: §5 and §6 are a snapshot of where the author was paused
> when this was packaged, not the project's roadmap.** The roadmap is the six
> stages in [`../README.md`](../README.md), in the author's stated order, and
> stage 1 (the metrics harness) is the authoritative next step. Where §6 and the
> README disagree, the README wins. **Second: two of §5's open items are now
> settled** — the polluted BetterBench numbers (§5.3) were dropped rather than
> published, which is why `bench/` ships harnesses and no result files, and the
> best-effort build (§5.2) went ahead, which is what this repository is.
>
> Paths: this document was written against a flat working tree and says
> `<repo>/kv-cache/`. In the release the docs are in `docs/`, the patches in
> `patches/`, the harnesses in `bench/`. [`../MANIFEST.md`](../MANIFEST.md)
> translates every one of them.

---

## 0. TL;DR (orient in 30 seconds)
> **STATUS: PROOF OF CONCEPT.** It works and it is measured. The correctness defects that were found have been **fixed and then tested** (R3.15, and the CT suite behind it); what remains is one *deliberate* approximation (the mamba N=8 stride) and a named list of validation still owed — see `README.md` and `kv-cache-closed-decisions.md`, not this line, for the current state. The roadmap out of "proof of concept", in the author's intended order: **(1) benchmark hooks, metrics and a reproducible cache-metrics harness** — first, because every later claim is unfalsifiable without it, and it must not repeat BetterBench's content-hash pollution. *Partly done:* patch 8 (`patch_kv_offload_tier_report.py`) plus `tierreport.py` deliver the per-tier sizing/speed report from one lifetime scrape; the controlled A/B arm is still outstanding. **(2) continue the review of existing work and write the implementation plan** — including reconciling every house patch against current HEAD and deleting the ones upstream has since implemented; *avoid reimplementation*, and score upstream work by applicability, not merge status. **(3) tunable accuracy toggles + a PoC of the most promising quality options** — the mamba stride `N` as config rather than constant, then the exactness fix (§4). **(4) refactoring** — eight string-surgery monkey-patches with an implicit dependency order want to be one module with an explicit interface. **(5) speed optimisation** — nothing here is tuned, only made to work. **(6) normal software engineering** — review, tests, CI, packaging; none of it has happened. Full statement at the top of `README.md`.

We are building and documenting a **two-tier (three-tier) KV-cache offload** for a **GDN hybrid model** (Qwen3.8-27B: 16 full-attention + 48 Gated-DeltaNet layers, 3:1) served on a **single AMD R9700 (gfx1201, 32 GB, TP=1)** under **vLLM 0.27.1 / radiance 0.9.3**, with **MXFP4 weights + FP8 KV**. The offload (GPU → RAM → disk) is **built, patched, and measured** (the RAM/disk tier served ~2.45M tokens ≈ ~26 min of prefill avoided — **a pre-2026-09-10 figure and therefore uncitable**; it is the shape of the result, not a result). The offload tier **writes and never deletes**, so it needs an **external reaper** (a systemd timer). The **contributor is not an AI researcher or a software developer** — the bar is a **best-effort reimplementation + clearly-stated limitations**, not a research-grade reproduction. We are currently **paused at a decision point** (§6): a best-effort build is the default, gated on a few open items.

---

## 1. What this project is
- **The thing being served:** `qwen3.8-27b-vllm` (container `qwen38-27b-vllm`) — the entry being measured. A sibling **MXFP4** entry (reference launcher `llama-swap-ggz14-27b.sh`, container `qwen38-27b-ggz14`) shares the same house implementation. **The two are whole-GPU and cannot run together** (`podman stop` the other first).
- **The architecture that matters:** a **GDN hybrid** — most layers are **Gated DeltaNet (linear-attention / recurrent-state)**, a few are full-attention. The recurrent state is large and is what the offload tier caches for the linear-attention layers; full-attention uses the paged KV.
- **The offload:** three tiers — **L1 GPU** (fp8 KV, ~228k tokens) → **L2 RAM** (`/dev/shm`, pre-faulted + pinned, **24 GiB ≈ 762,000 tokens** — confirmed live: `kv_offload_tier_capacity_bytes{tier="cpu"}` reads 25.76 GB) → **L3 disk** (a dedicated fs, `O_DIRECT`, 8r/4w threads). Wired via `--kv-transfer-config` (`TieringOffloadingSpec` / `OffloadingConnector`).
- **The house code lives in `<repo>/kv-cache/`** and rides its own bind mount (`/house`) into the container; the upstream ggz14 clone stays pristine. The **launcher** wires in **8 house patches** at container start + the **2-layer GC**.
- **The bar (calibrate to this):** a **proof of concept** — a working, measured baseline with clearly-stated limitations and a single well-described gap to the full DASC method — *not* a research-grade DASC reproduction, and not production code. It is published at this maturity deliberately, because others want the capability and the author cannot carry it to completion alone. Suggested to an LLM: keep proposals in this lane; do not reach for the research-core (weight-derived retention-horizon selection) unless explicitly asked.

---

## 2. The environment (how to inspect it)
- **Do NOT `podman exec` into `qwen38-27b-vllm`** — it fails with `crun: setrlimit RLIMIT_MEMLOCK: Operation not permitted` (known issue). **But you can still read the running engine's own files, read-only, through `/proc`:**
  `/proc/$(podman inspect qwen38-27b-vllm --format '{{.State.Pid}}')/root/opt/vllm/lib/python3.12/site-packages/vllm/...`
  That is how a patch set can be pre-flighted against the *live* engine without restarting it, and it supersedes the older advice to work from `podman inspect` alone. Also useful: **`podman inspect`** + the **applied patches** + the **boot log** (`<repo>/logs/boot-qwen3.8-27b-vllm*.log`).
- **API ports:** raw `127.0.0.1:5804`; llama-swap proxy `<server-ip>:1234`.
- **No cache-flush endpoint** (404 on both) — you cannot force a cold run via the API.
- **Key env in the container:** `RADIANCE_MAMBA_STORE_STRIDE=8`, `RADIANCE_OFFLOAD_EAGLE_GROUPS=1`, `RADIANCE_OFFLOAD_PENDING_IS_MISS=1`, `PYTHONHASHSEED=0` (pinned — required for the on-disk cache to survive a restart).
- **Launcher:** `llama-swap-ggz14-27b.sh` (house copy of ggz14's `serve-mxfp4.sh`, 2026-09-05). All `KVOFF_*` defaults are in `kv-cache-current-implementation.md` §4.

---

## 3. State of the work
**Done (built, patched, measured, documented):**
- Three-tier offload is live and **measured** — the tier **earned its keep**: ~2.45M tokens served from RAM/disk ≈ **~26 min prefill avoided** at the honest cold rate (~1,555 t/s). ⚠️ **Pre-2026-09-10 and uncitable** pending re-measurement on the fixed engine (`status-2026-09-10.md` §4 items 2–3). Per-tier numbers from the *current* engine come from `tierreport.py` instead.
- **8 house patches** applied in dependency order (mixed-hit guard, instrumentation, lookup-outcomes, serve-ready-prefix, eagle-groups, **mamba-stride N=8**, fs-fanout, **tier-report metrics**) — see `kv-cache-current-implementation.md` §3.
- **N=8 stride** is the capacity lever: 0.417× bytes, lifts the RAM tier to ~276,900 tokens (1.21× GPU) from 115,360 (0.50× GPU).
- **GC** in two layers: startup (orphan container report + `fuser`-gated `/dev/shm` region reap) + runtime fs reaper (`kvcache-reap.sh`, systemd timer, 5-min cadence, 2-stage age/capacity, `MIN_AGE=90min` hard safety floor). See `kv-cache-current-implementation.md` §5.
- **Docs** (all in `<repo>/kv-cache/`): current-implementation, future-work, references, known-issues, operations, historical, the closed-decisions register, `CORRECTNESS.md`, the tier-report metrics plan, the dated statuses, and this handover. The full map is §10.

**The one big finding (know this before trusting any number):**
- **The BetterBench run is INVALID (polluted).** vLLM radiance matches the prefix cache by block **content**, not **chained prefix**, and BetterBench's "cold (nonce)" only varies **block 0** → the body self-caches. Evidence: 2.15M GPU + 2.45M external hits; A/B at 32k repeating body 9.1s (3,520 t/s) vs unique 20.6s (1,555 t/s). **Do not cite the BetterBench numbers.** The only valid quantitative claim is the "tier earned its keep" measurement above.

---

## 4. Key technical facts to keep in mind
- **The N=8 stride is approximate by design, and "good" because of the GDN gate.** The store keeps only every Nth chunk's recurrent state; a lookup rounds **down** to the nearest kept boundary, so the served state is ≤ 8 chunks behind the requested position. The Gated-DeltaNet gate gives the state **finite effective memory**, so a coarser state is a good approximation (recent tokens dominate). It is *less sensitive to the nonce* than an exact state — this is the root of the BetterBench pollution.
- **Exactness claim (the defensible one):** the stride checkpoint is an **exact state at kept boundaries**. Serving an arbitrary position P = take the boundary checkpoint ≤ P and **replay the ≤ one-block gap** with exact `(d,k,g)` inputs → **exact** state at P (the GDN checkpoint bundles recurrent + conv state, so one replay covers both). **Reuse L = the stride gap ≤ one block = 8 chunks = 8 × `tokens_per_chunk` tokens — derived, not tuned.**
- **Checkpoint-precision nuance:** the gap replay is exact *given* the checkpoint, but the result's precision = the checkpoint's precision. **bf16 checkpoint → exact result; int8 (quantized) checkpoint → quantization-error floor.** The *length* of the refresh is independent of precision; only the *error floor* depends on it.
- **`tokens_per_chunk` is NOT yet confirmed** (can't `podman exec` to read it). So the reuse L is currently `8 × tokens_per_chunk` (e.g. chunk=64 → up to ~512 tokens). Confirm it to make L a concrete number.
- **ReplaySSM's L=16 is NOT a reuse number.** It is the **decode-latency** ring (spec-decode verify/rollback; `--linear-replayssm-cache-len` default 16). The **reuse** refresh is a different, bounded, derived L (§4 above). Do not conflate them.
- **NIAH impact of the stride:** modest increase in failure rate; scales with the Mamba:full-attention reliance ratio and haystack length. Full-attention is a precise, stride-unaffected fallback.
- **DASC / DAMP (the research context):** DASC (arXiv 2608.30386, Meituan) = state *compression* via **weight-derived per-unit retention-horizon selection** + suffix refresh; **run on unquantized BF16** → its 2.63×/42.6%/68.4% are **not ours** (we are uniform-stride + quantized MXFP4/FP8). DAMP (2608.27513) = the state-*quantization* sibling (closer to our FP8 shape). **No clean public unmerged DASC PR found** — most likely a private/internal branch or not-yet-PR'd fork.
- **A failed offload load kills EngineCore** (`assert transfer_result.success`; `OffloadingConnector` has no `get_block_ids_with_load_errors()`), which is why the reaper's `MIN_AGE` floor is a hard safety property and `kv_load_failure_policy=recompute` is inert here.

---

## 5. The open items / decisions (where we are paused)
1. **The "what was 'there'" question (unresolved).** The user saw an MXFP4/FP8 config "somewhere" (DASC paper / DAMP paper / a specific PR/table) but the exact document and whether MXFP4/FP8 was the **eval setup** or a **target/deployment note** is unconfirmed. This determines whether DASC-style numbers are directly ours or a BF16 baseline to re-baseline onto.
2. **Build vs watch (the DASC work).** The default is the **best-effort build**: offload (measured) + stride N=8 (applied) + **wire in the reuse suffix refresh at L = one block** via the existing ReplaySSM "replay raw (d,k,g)" primitive, applied at **reuse time**. Optionally set a watch on sglang/vllm PRs for the DASC terms to fold in the Meituan PR if it lands.
3. **The fate of the polluted BetterBench numbers** in the publishable report: replace with honest (cold) numbers, add a caveat, or drop the section. (The "tier earned its keep" framing is the strongest honest story.)
4. **Confirm `tokens_per_chunk`** so the reuse L is a concrete token count.

---

## 6. How to resume (concrete next steps, in order)
1. **Read the docs in this order** if you need depth: `kv-cache-current-implementation.md` (what's running) → `kv-cache-known-issues.md` (what's broken/limited) → `kv-cache-closed-decisions.md` (what is already settled, and what would reopen it — read this before proposing anything) → `kv-cache-future-work.md` (the plan) → `kv-cache-references.md` (links) → `status-2026-09-09.md` (the snapshot).
2. **Resolve §5.1** — pin down what "there" was (the document) and whether MXFP4/FP8 was the eval setup or a target note.
3. **Confirm `tokens_per_chunk`** (from the model config / boot log, since `podman exec` is out) → make the reuse L concrete.
4. **Get the go / go-not** for the best-effort build (§5.2), then **wire in the reuse suffix refresh at L = one block** (the one new piece; everything else is already live).
5. **Verify exactness** (bf16 checkpoint): reconstructed state at P should match the un-strided state at P within fp rounding.
6. **Decide the BetterBench numbers' fate** (§5.3) and update the report.

---

## 7. What NOT to do (the hard don'ts — full list in `kv-cache-known-issues.md` §E)
- **Never cite the BetterBench numbers** — polluted.
- **Never assume an exact state at an arbitrary position** — the stride store is exact only at kept boundaries; replay the gap.
- **Never treat DASC's 2.63×/42.6%/68.4% as ours** — uniform stride + quantized config + single model/HW.
- **Never flush the cache via the API** — no endpoint.
- **Never `podman exec` into `qwen38-27b-vllm`** — fails on RLIMIT_MEMLOCK.
- **Never delete a young fs block** — the reaper's `MIN_AGE=90min` floor is a safety property (a reaped in-flight block kills EngineCore).
- **Never change `PYTHONHASHSEED`** without invalidating the on-disk cache deliberately.

---

## 8. Verify it is still as documented (quick checks)
- `podman inspect qwen38-27b-vllm` (or `qwen38-27b-ggz14`) → confirm env `RADIANCE_MAMBA_STORE_STRIDE=8`, `RADIANCE_OFFLOAD_EAGLE_GROUPS=1`, `PYTHONHASHSEED=0`.
- Boot log `logs/boot-qwen3.8-27b-vllm*.log` → the **eagle-group line must read `[8]`** (not all nine flagged); the `Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM` line.
- `systemctl status kvcache-reap` / `kvcache-reap.timer` (or the system-level equivalent) → the reaper is active (it is *mandatory* — the fs tier writes and never deletes).
- `df -h /kvcache` → usage should hover ≤ ~65% (the reaper's target).

---

## 9. The contributor (calibrate your suggestions)
**Not an AI researcher, not a software developer.** The deliverable is a **proof of concept with clearly-stated limitations** — a working measured baseline with a single, well-described gap to the full DASC method, that others can pick up and extend. Keep proposals in this lane: lean on what is already live (offload + stride + the ReplaySSM primitive), skip the research-core (weight-derived retention-horizon selection), and state the limitations up front.

---

## 10. The map of the docs (the working tree is flat; in the release these are `docs/` — see ../MANIFEST.md)
| Doc | What it has |
|---|---|
| **`kv-cache-handover.md`** (this file) | The map + current-state summary + resume path. **Read first.** |
| `kv-cache-current-implementation.md` | What is actually built and running: the 3 tiers, the 8 house patches (gates + order), the 2-layer GC, the serve invocation, the design gotchas. |
| `kv-cache-operations.md` | **The runbook.** Turn the disk (L3) tier off; resize `/dev/shm`; resize the KV tier. Each with the command, the verification, the undo — and the OOM cliff (§3.0) before you enlarge the tier. |
| `kv-cache-known-issues.md` | Every problem/gotcha/limitation with Symptom/Root cause/Impact/Status, grouped by severity, + the hard "never do X" list. |
| `kv-cache-future-work.md` | The plan: the best-effort bar, the reuse-refresh mechanism (the L analysis), the 5 limitations to state up front, the open items. |
| `kv-cache-references.md` | Links to every PR / paper / blog / local artifact pulled & reviewed, each with a status + ruling. |
| **`kv-cache-closed-decisions.md`** | **The register of what is settled**, each row with its evidence and the one condition that would reopen it. Consult this *before* proposing work — it exists so the narrative docs do not have to be read end to end to find out whether a question is already answered. |
| `CORRECTNESS.md` | The correctness suite's specification: CT1–CT6, the CT5 negative-control gate that runs first, and the contention asymmetry the whole method rests on. |
| `tier-report-metrics-plan.md` | The design note behind patch 8 and `tierreport.py`: which metrics, why counters and histograms rather than gauges, and the verdict each one supports. |
| `status-2026-09-10.md` | **The current dated snapshot** — the R3.15 defect, the controlled reverse test, and §4's list of validation still owed. Carries a box at the top listing what has moved since. |
| `status-2026-09-09.md` | The earlier investigation snapshot (the BetterBench pollution finding, the DASC/DAMP findings, the PR-search dead-ends, the environment). **Superseded** — its §3 mechanism is wrong; kept as history. |
| `kv-cache-historical.md` | The experimental record: what was tried, what it showed, and what was later retracted. The reasoning behind the register. |
| `kv-cache-results-preliminary.md` | The preliminary result, and the critique of the third-party table it was compared against. Superseded pending re-measurement. |
| `cache-preemption-patch-plan.md` | The deeper patch plan (Revision 2 supersedes parts of the original). |
| `README.md` | **The public front door.** The proof-of-concept status statement, the roadmap above, the architecture, the hard requirements, the one honest number. |
| `../startup-qwen3.8-27b-kvcache.sh` | The `qwen3.8-27b-kvcache` entry: the KV-cache configuration in one file, every knob documented inline with its measured cost. A thin wrapper around the base launcher. `-h` for the list. |
| `make-dist.sh` | Assembles the public release tree on demand from the real files (no drift), including the three that live outside this directory. Ships method, not result files. |
| `README.house.md` | How the house patch is wired in (`/house` mount, `PYTHONPATH=/patches`), the `tierbench` acceptance test. |

**Related home docs:** `$HOME/vllm-kv-cache-offload-reuse.md` (the GC units' `Documentation=` target), `$HOME/known_good.md` (the pinned config the `upstream-updates` skill checks against).
