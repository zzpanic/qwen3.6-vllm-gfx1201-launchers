# KV-Cache Known Issues

Known problems, gotchas, and limitations discovered in the two-tier KV-offload + GDN stride-store + BetterBench investigation. Each entry: **Symptom / Root cause / Impact / Status or workaround.** Severity: **Critical** (blocks valid results), **Design** (inherent — must be documented), **Blocker** (environmental/tooling), **Open** (unresolved).

See `kv-cache-references.md` for links, `kv-cache-future-work.md` for the plan, `status-2026-09-10.md` for the current snapshot (`status-2026-09-09.md` is kept as history and is superseded).

---

## A. Critical — blocks valid measurement

### A1. The BetterBench run is polluted (prefix-cache hits on "cold" prompts) — ROOT-CAUSED and FIXED by R3.15; revalidation required
- **Symptom:** BetterBench "cold" prefill runs were ~2.3× faster than a true cold run; the full run (all 3 phases) is **invalid**.
- **Root cause.** The **offload tier's** lookup: pre-R3.15 the mixed-hit path could hand `prepare_load` a key it had not confirmed, so the tier matched blocks whose *content* was identical while their *chained prefix* was not. (The GPU prefix cache chains correctly and always did; an earlier revision blamed it — see `kv-cache-closed-decisions.md` §5.) The nonce varies only block 0, so blocks 1..N are byte-identical text whose KV was computed under a **different preceding context** — exactly what the chained hash exists to poison.
  - **Controlled reverse test, one variable** (`patch_offload_mixed_hit.py`, fixed vs `a48e3a7^`; `KVOFF_PENDING_IS_MISS=0` held identical and confirmed live in both arms):

    | arm | GPU prefix-cache hits | external (tier) hits |
    |---|--:|--:|
    | R3.15 present | 0 of 188,192 queried | 0 — new nonce recomputes in full |
    | R3.15 absent | 0 of 188,192 queried | **184,576 of 188,192 = 98.1%**, 6.44 GB read in 60 s |

  - Both arms looked at the same prompts in the same window. The GPU cache refused all of them in both. Only the tier differed. That asymmetry **is** the bug.
  - With the fix in place the prefill sweep matched an independent cold baseline within **1.3% at all eight depths**, monotonic in depth, TTFT spread **0.05–0.35%** (it had been 14–80% on the affected depths).
- **Impact:** This was never only a benchmark-pollution issue. The engine was substituting KV belonging to a different prompt into real answers, silently. Treat any output produced before 2026-09-10 accordingly, not just any number.
- **Status:** **Fixed** by R3.15 (`patch_offload_mixed_hit.py`, hunks 3 and 4) and confirmed live by `check-r315-boot.sh` section 5. **Not yet fully validated** — `status-2026-09-10.md` §4 lists what is owed, with §4 item 4 already met by **CT5's gate** in `CORRECTNESS.md`. In particular the A/B that produced the 9.1 s / 20.6 s figures above was itself taken on defective code and must be re-run; until then the *size* of the historical pollution is unknown, only its direction.
- **Measurement trap:** a probe that replicated the harness prompt byte-for-byte "cleared" the nonce — but it was run on the **fixed** engine, where there is nothing to find. This class of bug is invisible to any probe run on patched code. **Arm the defect before concluding that a cache respects a salt.**

### A2. The N=8 stride store truncates the hit to a boundary — CORRECTED 2026-09-11, NOT an accuracy defect
> **This entry was wrong and is corrected in place.** It described a substitution the
> implementation does not perform. The stride is **truncate-and-recompute, not
> serve-an-approximate-state**: `resolve_mamba_align_size` (`scheduler.py:157`, resolved
> `:518`) makes `max_hit_size_tokens = round_down(max_hit_size_tokens, mamba_align_size)`
> at `:722`, applied to the *whole* hit window, so the engine only ever requests a snapshot
> it actually kept. What is served is **exact**; the tokens between the boundary and the
> requested position are recomputed by the normal prefill path. Corroborated independently:
> the 85,696-token disk hit was bit-identical to a cold recompute. The cost is **compute,
> not accuracy** — up to 13,184 tokens, ~6,592 on average, ~8% of an 85k prefix, and not
> additive since the Mamba state bounded the hit regardless. **D4 below inherits this
> correction.** The residual risk is a store/lookup N mismatch across a restart, which
> yields a MISS rather than wrong data.

- **Symptom:** A hit is truncated down to an N-chunk boundary; up to 8 chunks of prefix that
  is held but declined, and recomputed.
- **Root cause:** `patch_kv_offload_mamba_stride.py` keeps only when `(abs_chunk_idx + 1) % 8 == 0`; lookup rounds **down** to `N×tokens_per_chunk`. The 7-of-8 intermediate states are discarded (never read).
- **Impact:** The state is **inherently approximate by the stride design** — it is *less sensitive to the nonce* than an exact state. It is independent of A1 and untouched by R3.15. It is what makes the GDN finite-memory approximation work (the gate makes the coarsening a *good* approximation), but it means the store is not exact at arbitrary positions.
- **Status/workaround:** **Exact only at kept boundaries.** To get the exact state at an arbitrary position P, replay the ≤ one-block gap (see `kv-cache-future-work.md` §2.2) — which is exact *given* the checkpoint.

---

## B. Blockers — environmental / tooling

### B1. No cache-flush endpoint — BLOCKER
- **Symptom:** Cannot flush the prefix/KV cache.
- **Root cause:** `/flush_cache` (and equivalents) return **404** on both raw `127.0.0.1:5804` and the llama-swap proxy `<server-ip>:1234`.
- **Impact:** Cannot force a true cold run; directly blocks the A1 workaround (b).
- **Status/workaround:** Unresolved. Options: patch in a flush endpoint, restart the container between runs, or make prompts fully unique (A1 workaround a).

### B2. `podman exec` into the model container fails — BLOCKER
- **Symptom:** `podman exec qwen38-27b-vllm …` fails with `crun: setrlimit RLIMIT_MEMLOCK: Operation not permitted`.
- **Impact:** Cannot run *commands* inside the container.
- **Status/workaround.** Inspecting live state does not need `exec`: the container's whole filesystem is readable, read-only, through the host's `/proc`:
  ```
  /proc/$(podman inspect qwen38-27b-vllm --format '{{.State.Pid}}')/root/opt/vllm/lib/python3.12/site-packages/vllm/...
  ```
  This is how all 31 hunks of patch 8 were pre-flighted against the **running** engine before a reload (31/31 clean). Applied patches, `podman inspect` and the boot log remain useful alongside it.

---

## C. Open — unresolved questions

### C1. `tokens_per_chunk` not confirmed → the reuse L is only `8 × tokens_per_chunk` — OPEN, not blocked
- **Symptom:** The reuse refresh length is expressed as `≤ one block = 8 chunks = 8 × tokens_per_chunk tokens`, but `tokens_per_chunk` is not pinned to a concrete value.
- **Root cause:** Nobody has read it off the running engine. It is not blocked: B2's `/proc` route reads the live config directly.
- **Impact:** `L` is not a concrete token count yet (e.g. chunk=64 → up to ~512 tokens).
- **Status/workaround:** Pull `tokens_per_chunk` from the running container (or the model config) to make `L` concrete.

### C2. The "what was 'there'" question is unresolved — OPEN
- **Symptom:** The user saw an MXFP4/FP8 config "somewhere," but the exact document (DASC paper / DAMP paper / a specific PR/table) and whether MXFP4/FP8 was presented as the **eval setup** or a **target/deployment note** is unconfirmed.
- **Impact:** Determines whether DASC-style numbers are directly ours or a **BF16 baseline to re-baseline onto** (see **D2**).
- **Status/workaround:** Resolve by identifying the document.

---

## D. Design limitations — inherent, must be documented in the deliverable

### D1. We use uniform stride coarsening, not DASC's decay-aware per-unit selection — DESIGN
- **Root cause:** DASC's headline win (2.63× / 42.6% / 68.4%) comes from **weight-derived, per-unit retention-horizon selection** (the research-core). We skip it and use a **uniform** stride.
- **Impact:** Our compression / TTFT numbers are **not comparable** to DASC's.
- **Status:** Intentional (best-effort bar). The single named gap to close: *"derive retention horizons from the gate weights, per the DASC method."*

### D2. Quantized config vs DASC's unquantized BF16 — DESIGN
- **Root cause:** We run **MXFP4 weights + FP8 KV** (and int8 state if the tier quantizes, cf. PR #28185); DASC's numbers are **unquantized BF16**.
- **Impact:** Deltas are **not directly transferable** to our shape. The checkpoint-precision nuance: the gap replay is exact *given* the checkpoint, but the result's precision = the checkpoint's precision (bf16 → exact; int8 → quantization error floor).
- **Status:** Must be stated up front (see `kv-cache-future-work.md` §4).

### D3. Single model / single hardware — DESIGN
- **Root cause:** Measured on **Qwen3.8-27b** on **AMD R9700 / gfx1201** only.
- **Impact:** Not general to DASC's test set (e.g. Qwen3-Next 80B) or other benchmark mixes / hardware.
- **Status:** State up front.

### D4. NIAH accuracy impact of the stride — SUPERSEDED by A2's correction (2026-09-11)
> The premise below — that a *coarser state is served* — is false; see A2. The served state
> is exact, so there is no stride-induced retrieval degradation to characterise. Retained
> only so the claim is not rediscovered from an old copy.
- **Root cause:** The coarser (stride) state is less sensitive to the nonce, so needle-in-a-haystack retrieval can degrade.
- **Impact:** **Modest** increase in NIAH failure rate; **scales with** the Mamba:full-attention reliance ratio and haystack length. Full-attention is a precise, stride-unaffected fallback; effect is larger for **shorter** haystacks and **Mamba-heavy** models.
- **Status:** Characterized, not eliminated. Document it.

### D5. DASC/DAMP implementation not publicly available to verify against — DESIGN
- **Root cause:** No clean public unmerged DASC PR found (most likely a private/internal branch or not-yet-PR'd fork; paper is ~9 days old). Author handle `zixujiang` (DAMP co-author) has 0 sglang PRs.
- **Impact:** Cannot verify our best-effort against the actual Meituan implementation.
- **Status/workaround:** Optionally set a watch on sglang/vllm PRs for the DASC terms and fold in when it lands.

---

## E. Quick reference — what to never do

> These are the operationally dangerous ones — the mistakes that cost a run or a number.
> For questions that are simply *settled*, and what would unsettle each,
> see [`kv-cache-closed-decisions.md`](kv-cache-closed-decisions.md).
1. **Never cite any benchmark number taken before 2026-09-10** — the engine under the harness was serving another prompt's KV (A1). The harness itself was sound.
2. **Never assume an exact state at an arbitrary position** — the stride store is exact only at kept boundaries (A2); replay the gap to get the exact state.
3. **Never treat DASC's 2.63×/42.6%/68.4% as ours** — uniform coarsening + quantized config + single model/HW (D1/D2/D3).
4. **Never try to flush the cache via the API** — no endpoint exists (B1).
5. **Never `podman exec` into `qwen38-27b-vllm`** — it fails on RLIMIT_MEMLOCK (B2); read the live container through `/proc/<pid>/root/...` instead (B2), or use `podman inspect` + applied patches.
6. **Never validate a cache-correctness fix on the fixed build alone** — arm the defect and show the instrument catches it (A1).
