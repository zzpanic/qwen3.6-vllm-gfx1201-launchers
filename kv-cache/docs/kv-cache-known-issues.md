# KV-Cache Known Issues

Known problems, gotchas, and limitations discovered in the two-tier KV-offload + GDN stride-store + BetterBench investigation. Each entry: **Symptom / Root cause / Impact / Status or workaround.** Severity: **Critical** (blocks valid results), **Design** (inherent — must be documented), **Blocker** (environmental/tooling), **Open** (unresolved).

See `kv-cache-references.md` for links, `kv-cache-future-work.md` for the plan, `status-2026-09-09.md` for the snapshot.

---

## A. Critical — blocks valid measurement

### A1. The BetterBench run is polluted (prefix-cache hits on "cold" prompts) — CRITICAL
- **Symptom:** BetterBench "cold" prefill runs were ~2.3× faster than a true cold run; the full run (all 3 phases) is **invalid**.
- **Root cause:** vLLM radiance matches the prefix cache by block **content**, not by **chained prefix**. BetterBench's "cold (nonce)" design only varies **block 0** (`corpus.py` `with_nonce`), so the body is identical across requests → it self-caches.
  - A/B at 32k: repeating `_PARA×n` body = **9.1s (3,520 t/s)** vs unique non-repeating = **20.6s (1,555 t/s)**.
  - 2.15M GPU prefix-cache hits + 2.45M external (offload) hits during the run.
  - Decode/concurrency reuse the same handful of prompts (20 runs cycling over 3–4 prompts; 48 req cycling over 29 prompts; 3 levels reusing the same 48).
- **Impact:** Do **not** cite the BetterBench numbers. The only valid quantitative claim is the "tier earned its keep" measurement (~2.45M tokens served ≈ ~26 min prefill avoided).
- **Status/workaround:** To get honest numbers, either (a) make the body fully unique per request (not just block 0), or (b) flush the cache between requests — which is currently impossible (see **B1**).

### A2. The N=8 stride store serves a coarser state than the exact position — CRITICAL (for exactness claims)
- **Symptom:** A reused checkpoint is **≤ 8 chunks behind** the requested position; the served state is `S_boundary`, not the exact per-position `S_M`.
- **Root cause:** `patch_kv_offload_mamba_stride.py` keeps only when `(abs_chunk_idx + 1) % 8 == 0`; lookup rounds **down** to `N×tokens_per_chunk`. The 7-of-8 intermediate states are discarded (never read).
- **Impact:** The state is **inherently approximate by the stride design** — it is *less sensitive to the nonce* than an exact state. This is the mechanism behind A1. It is what makes the GDN finite-memory approximation work (the gate makes the coarsening a *good* approximation), but it means the store is not exact at arbitrary positions.
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
- **Impact:** Cannot run commands inside the container to inspect live state (e.g. confirm `tokens_per_chunk`, dump the stride config).
- **Status/workaround:** Work from **applied patches + `podman inspect`** instead. (This is also why `tokens_per_chunk` is not yet confirmed — see **C1**.)

---

## C. Open — unresolved questions

### C1. `tokens_per_chunk` not confirmed → the reuse L is only `8 × tokens_per_chunk` — OPEN
- **Symptom:** The reuse refresh length is expressed as `≤ one block = 8 chunks = 8 × tokens_per_chunk tokens`, but `tokens_per_chunk` is not pinned to a concrete value.
- **Root cause:** Cannot inspect the live config (see **B2**).
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

### D4. NIAH accuracy impact of the stride — DESIGN
- **Root cause:** The coarser (stride) state is less sensitive to the nonce, so needle-in-a-haystack retrieval can degrade.
- **Impact:** **Modest** increase in NIAH failure rate; **scales with** the Mamba:full-attention reliance ratio and haystack length. Full-attention is a precise, stride-unaffected fallback; effect is larger for **shorter** haystacks and **Mamba-heavy** models.
- **Status:** Characterized, not eliminated. Document it.

### D5. DASC/DAMP implementation not publicly available to verify against — DESIGN
- **Root cause:** No clean public unmerged DASC PR found (most likely a private/internal branch or not-yet-PR'd fork; paper is ~9 days old). Author handle `zixujiang` (DAMP co-author) has 0 sglang PRs.
- **Impact:** Cannot verify our best-effort against the actual Meituan implementation.
- **Status/workaround:** Optionally set a watch on sglang/vllm PRs for the DASC terms and fold in when it lands.

---

## E. Quick reference — what to never do
1. **Never cite the BetterBench numbers** — they are polluted (A1).
2. **Never assume an exact state at an arbitrary position** — the stride store is exact only at kept boundaries (A2); replay the gap to get the exact state.
3. **Never treat DASC's 2.63×/42.6%/68.4% as ours** — uniform coarsening + quantized config + single model/HW (D1/D2/D3).
4. **Never try to flush the cache via the API** — no endpoint exists (B1).
5. **Never `podman exec` into `qwen38-27b-vllm`** — it fails on RLIMIT_MEMLOCK (B2); use `podman inspect` + applied patches.
