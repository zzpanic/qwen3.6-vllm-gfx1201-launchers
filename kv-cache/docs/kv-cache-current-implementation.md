# KV-Cache Current Implementation

What is actually built and running today for the two-tier (three-tier) KV offload on this box. Reference launcher: `llama-swap-ggz14-27b.sh`. For the plan see `kv-cache-future-work.md`, for links see `kv-cache-references.md`, for problems see `kv-cache-known-issues.md`.

> This box has **two whole-GPU vLLM entries that share this house implementation** and cannot run together: the **MXFP4** one (reference launcher above → container `qwen38-27b-ggz14`, served `qwen3.8-27b-ggz14`) and the **vllm** one (container `qwen38-27b-vllm`, served `qwen3.8-27b-vllm` — the one being measured). Switching = `podman stop` the other first. Everything below is the shared house KV code.

---

## 1. What is running
- **Hardware:** single AMD R9700 (RDNA4, gfx1201, 32 GB), **TP=1** (auto-detected by `gpu-detect.sh`).
- **Model:** Qwen3.8-27B, native **MXFP4 (4-bit) weights** + **FP8 DFlash2 drafter** (block-diffusion, `SPEC_METHOD=dflash`, depth 7; `SPEC_METHOD=mtp` uses the head inside the target, no drafter).
- **Image / engine:** `stilldeadcode/vllm-radiance:0.9.3`, **vLLM 0.27.1**. Host: `--ipc=host --network=host`, `--ulimit memlock=-1`, `/dev/kfd` + `/dev/dri`, `--cap-add SYS_PTRACE`.
- **KV cache:** `--kv-cache-dtype fp8`, `--enable-prefix-caching`, `--mamba-cache-mode align`, `--gpu-memory-utilization 0.98` (the measured ceiling on a 32 GB card).
- **House launcher** = a copy of ggz14's `serve-mxfp4.sh` (2026-09-05) + six llama-swap edits (`--port`, `NAME`/`SERVED` split, `MODELS` default, `--rm`, `REPO` knob). The upstream clone stays pristine; the house code rides its own bind mount.

---

## 2. The offload tiers
| Tier | Where | Sized by | Notes |
|---|---|---|---|
| **L1 GPU** | VRAM | `--kv-cache-memory` pin (measured per hardware+batch shape via `rad_kv_lookup`) or vLLM profiling | `--kv-cache-dtype fp8`; `~228,737 tokens` at this boot. |
| **L2 CPU (primary)** | `/dev/shm` region (host tmpfs, `--ipc=host`) | `KV_OFFLOAD` → GiB via `kvoff_resolve`, **clamped to both** the tmpfs cap (`statvfs`, minus 256 MiB margin) **and** the RAM cap (`MemAvailable − 3 GiB keep-free`); min 4 GiB, else off | **Pre-faulted (`MADV_POPULATE_WRITE`) and pinned (`cudaHostRegister`) — never swaps.** **24 GiB ≈ 762,000 tokens** at this boot, confirmed live by `kv_offload_tier_capacity_bytes{tier="cpu"}` = 25.76 GB. Eviction: `KVOFF_POLICY` (default `arc`, scan-resistant; `lru` available). Admission: `KVOFF_STORE_THRESHOLD` (default 0 = admit on first sight; 2 = admit on second sighting). |
| **L3 disk (fs)** | dedicated filesystem (`KVOFF_DISK`), container path `/kvcache` | `--kv-transfer-config` (`TieringOffloadingSpec`, `OffloadingConnector`, `kv_both`, `kv_load_failure_policy=recompute`) | **8 read / 4 write threads** (tuned: 512 GB zvol, O_DIRECT, reads peak at 8 thr 719 MB/s, writes 4 thr already 24× the 44.6 MB/s stored). **`O_DIRECT`** → bypasses page cache; spare RAM can only help as the primary tier. **NO built-in eviction** → the reaper is mandatory (§5). **Requires the L2 primary tier** to reach the GPU. |

**`kv_load_failure_policy` is INERT here** (measured): the recompute path is driven by `get_block_ids_with_load_errors()`, which only nixl/mooncake/flexkv/lmcache implement — `OffloadingConnector` returns an empty set, and `offloading/worker.py:361` is a **bare `assert transfer_result.success`**. A failed load kills EngineCore outright. See §7.

---

## 3. The patches
Applied at container start inside the entrypoint `bash -lc '…'`, in **this order** (dependency matters).

### In-repo (from `/patches` = the ggz14 repo)
`patch_quark_mxfp4.py`, `patch_ar_maxbytes.py`, `patch_topk_triton_rows.py`, `patch_dflash_calib.py`, `patch_dflash_mxfp4_kv.py`, `patch_rmsquant_fusion.py`, `patch_verify_head.py`, `patch_kv_group_size.py`, `patch_topk_composite.py`, `patch_gdn_shared_build.py`, `patch_dflash_selector_topk.py`, `patch_gdn_merge_inproj.py`, `patch_dynwidth.py`, `patch_ar_geometry.py`, `patch_gdn_glue.py`, `patch_qwen3_thinkoff.py` (non-fatal). Plus: copy the `radiance_*.py` files + `mxfp4-configs/*.json`, **compile `radiance_mxfp4_fp8.so` with `hipcc -O3 … --offload-arch=gfx1201`**, and (if `R4D_SO` set) drop in a patched `r4d.so`. `cd /` before exec so a stale `/patches/*.so` can't shadow the freshly compiled one.

### House (from `/house` = `<repo>/kv-cache`; `PYTHONPATH=/patches python3 /house/<patch>`)
| # | Patch | What it does | Gate (default) | Order / notes |
|---|---|---|---|---|
| 1 | `patch_offload_mixed_hit.py` | Stops `OffloadingConnector` killing the engine on a mixed local+external prefix hit | `RADIANCE_OFFLOAD_MIXED_HIT` / `KVOFF_MIXED_HIT` (**1** = serve mixed hits = default = safe; 0 = decline = retained kill switch) | first |
| 2 | `patch_kv_offload_instrumentation.py` | Adds 3 gauges (`cpu_cache_evictable_perc`, `cpu_cache_free_perc`, `fs_inflight_jobs`) + widens the `lookup_async_delay` histogram 10 s→600 s. **Instrumentation only.** | — | after 1 |
| 3 | `patch_kv_offload_lookup_outcomes.py` | Counts every terminal branch of the lookup path (diagnostic: why production gets ~0 external hits when the bench gets an exact 85,696-token one). **Non-fatal.** | — | after 2 |
| 4 | `patch_kv_offload_serve_ready_prefix.py` | Serve the ready prefix instead of deferring on an in-flight store (the Phase A fix — the only behaviour change in the block). **Non-fatal, but not harmless if it fails.** | `RADIANCE_OFFLOAD_PENDING_IS_MISS` / `KVOFF_PENDING_IS_MISS` (**0**; 1 = the truncating serve-ready-prefix, opt-in) | must run after 3 |
| 5 | `patch_kv_offload_eagle_groups.py` | Annotate the EAGLE/MTP draft KV group positionally so the scheduler stops flagging **all nine** groups as draft (prerequisite for #6). **Non-fatal.** | `RADIANCE_OFFLOAD_EAGLE_GROUPS` / `KVOFF_EAGLE_GROUPS` (**1**) | must run before 6 |
| 6 | `patch_kv_offload_mamba_stride.py` | **R3.13 Mamba store cadence — the capacity lever (the N=8 stride).** Keep every Nth Mamba/GDN snapshot instead of one per chunk. | `RADIANCE_MAMBA_STORE_STRIDE` / `KVOFF_MAMBA_STRIDE` (**8**; 1 = upstream) | must run after 5 |
| 7 | `patch_kv_offload_fs_fanout.py` | **R3.14 fs-tier job fanout** (port of upstream PR #49225): one fs job split across the thread pool instead of a single serial task over every block file. | `RADIANCE_FS_FANOUT_TARGET_MB` / `KVOFF_FS_FANOUT_MB` (**256**), `KVOFF_FS_FANOUT_MAX` (**0** = byte budget) | order-independent (different file) |
| 8 | `patch_kv_offload_tier_report.py` | **Tier report metrics (P1-P6 of `tier-report-metrics-plan.md`).** Adds 19 `tier`-labelled series: hit blocks/tokens per tier, load/store bytes, seconds, ops and latency histograms, capacity and used bytes, an occupancy *histogram* (a distribution over time, not a level), reads-before-evict, evictions, evicted bytes, eviction-to-reuse, would-have-hit, and prefill stall seconds. Feeds `tierreport.py`. **Instrumentation only.** | — | **must run after 2** — it extends the `FileSystemTierManager.get_stats()` that 2 creates, and refuses to apply (with a named error) if 2 has not run |

**The N=8 stride in numbers** (from the launcher): a Mamba group holds one recurrent state and the load path reads exactly one chunk, but the store path writes **27 MB per group per chunk** — ~70 snapshots for a long conversation where the GPU keeps 2. At N=8 the bytes per 8 chunks fall 72→30 (**0.417×**), so the tier holds proportionally more — at the measured 33,808 B/token the 24 GiB tier holds **~762,000 tokens, 3.3× the GPU cache**. That is what keeps a conversation in RAM on the follow-up turn; a CPU hit costs 1–2 s vs the measured **64.26 s** fs→CPU promotion. Cost: prefix hits truncate down to an N-chunk (13,184-token) boundary.

---

## 4. Config knobs (the `KVOFF_*` defaults)
| Knob | Default | Meaning |
|---|---|---|
| `KV_OFFLOAD` | `off` (this launcher; the vllm entry runs it on via config) | `off` / `auto` / `<pct>` / `<GiB>`. Resolved+clamped by `kvoff_resolve`. |
| `KVOFF_MIN_GIB` / `KVOFF_KEEP_FREE_GIB` / `KVOFF_SHM_MARGIN_MIB` | 4 / 3 / 256 | min useful tier / RAM floor left for the rest / tmpfs margin. |
| `KVOFF_BACKEND` | `native` | offload backend (`native` \| `lmcache`). |
| `KVOFF_DISK` / `KVOFF_DISK_MNT` / `KVOFF_DISK_SUBDIR` | `""` / `/kvcache` / `blocks` | enable L3 (host path) / container path / subdirectory. |
| `KVOFF_DISK_RTHREADS` / `KVOFF_DISK_WTHREADS` | 8 / 4 | fs-tier read / write threads (vLLM default 16). |
| `KVOFF_HASHSEED` | 0 | **pinned `PYTHONHASHSEED`** — block filenames are content hashes chained from `NONE_HASH`; if unset, `kv_cache_utils.py:112` seeds from `os.urandom(32)` and every restart hashes the same tokens to different filenames, orphaning the whole on-disk cache (100% miss, no error). Changing it invalidates the cache. |
| `KVOFF_MIXED_HIT` / `KVOFF_PENDING_IS_MISS` / `KVOFF_EAGLE_GROUPS` / `KVOFF_MAMBA_STRIDE` | 1 / 0 / 1 / 8 | the four offload behaviour gates (see §3). |
| `KVOFF_FS_FANOUT_MB` / `KVOFF_FS_FANOUT_MAX` | 256 / 0 | fs-job fanout budget (MiB) / max (0 = byte budget). `KVOFF_FS_FANOUT_MAX=1` restores upstream one-task-per-job. |
| `KVOFF_BLOCKS_PER_CHUNK` | `""` (empty = vLLM default 1) | KV blocks per offloaded chunk; raises it coarsens the offload grid **and** the external hit-window rounding. Leave empty for production. |
| `KVOFF_POLICY` / `KVOFF_STORE_THRESHOLD` | `arc` / 0 | L2 eviction policy / admission filter. |
| `KVOFF_RAM_RESERVE_GIB` | 15 | everything that is not the tier + headroom, for the explicit-size RAM clamp. |
| `GC_ORPHANS` | `on` | startup GC (§5). |

---

## 5. Garbage collection (two layers)
### 5.1 Startup GC (in the launcher; runs every start, regardless of `KV_OFFLOAD`; `DRY_RUN` stays side-effect free)
- **`gc_report_orphan_container`** — *reports only* if our own `$NAME` container is already `Up` (an orphan from a crash/killed stop holding VRAM). The launch already reclaims the name (podman `--replace` / docker `rm -f`); this just tells you it happened. Scoped strictly to `$NAME`.
- **`kvoff_reap_orphans`** — a restart orphans the container **but not its `/dev/shm` region**; the stale mmap keeps its full size, so the next boot finds the tmpfs full. Reaps `/dev/shm/vllm_offload_*.mmap` regions that no live process still holds — `fuser` (or `lsof`) is the precise test, so only genuinely unreferenced regions are deleted and a concurrently running second model keeps its buffer.

### 5.2 Runtime GC — the fs-tier reaper (`kvcache-reap.sh` + `.service` + `.timer`)
**Why it exists:** `vllm/v1/kv_offload/tiering/fs/manager.py` has **no capacity, quota or TTL** and `SecondaryTierManager` exposes **no eviction hook** — the tier **writes and never deletes**. Without the reaper the filesystem fills and every subsequent store fails. **This is the eviction policy, not a tuning script.**

**Schedule:** `kvcache-reap.service` (Type=oneshot, `Nice=10`, `IOSchedulingClass=idle`, `ConditionPathIsDirectory=/kvcache/blocks`) driven by `kvcache-reap.timer` — **`OnBootSec=5min`, `OnUnitActiveSec=5min`, `AccuracySec=1min`**. (5-min cadence keeps each reap small and its I/O idle-class rather than one large stall; the tier writes ~45 MB/s worst case, so 512 GB is hours of headroom.)

**Policy (age-based, changed 2026-09-06):**
- **The safety property is `MIN_AGE`, not the schedule.** Deleting a block a load is about to read is an **engine killer**, not a missed hit: `lookup` stats the filesystem, but between lookup and load there is a window and `offloading/worker.py:361` is a bare `assert` with no `get_block_ids_with_load_errors()`, so `recompute` can't catch it. That window can't be closed from here (needs upstream). The reaper's job is to **never delete a block the engine plausibly still wants** — `MIN_AGE_MIN` is a hard floor **no rule, capacity pressure included, may cross**. If the target can't be met without crossing it, the reaper says so and stops rather than delete young blocks.
- **Ordering = oldest-mtime-first.** The tier writes each block once and never modifies it (`io.py` short-circuits on `os.path.exists`), so mtime is insertion time → **FIFO ≈ LRU at zero I/O cost.** (True atime-LRU would need `relatime`; the volume is `noatime` today, so atime is meaningless.)
- **Stage A (age)** — every cycle, regardless of %use: delete any `*.bin` block older than `MAX_AGE_HOURS=8`. This is what stops the volume building toward a panic purge.
- **Stage B (capacity)** — only if age alone didn't keep it under `TARGET_PCT=65%`: delete oldest-first, **never below the `MIN_AGE_MIN=90 min` floor** (expressed in the `find`, not in a check a later edit could drop).
- Also deletes **orphaned `*.tmp` older than `TMP_AGE_MIN=60` min** (a crashed store leaves `<name>.bin.tmp<suffix>` behind; `io.py` writes to a temp path then `os.replace()`s it — never read, never reaped by vLLM).
- **`MAX_DELETE=4000` per-run cap** (so no cycle runs long; the next cycle continues where it left off).
- **Known anti-correlation (accepted with eyes open):** a prefix shared by every conversation (a system prompt) is written once, early, and stays hot forever — exactly the shape mtime-FIFO deletes first. It is a handful of blocks out of thousands, and losing one costs a single recompute before the tier writes it back with a fresh mtime.

**Knobs:** `KVCACHE_ROOT=/kvcache/blocks`, `KVCACHE_MIN_AGE_MIN=90`, `KVCACHE_MAX_AGE_HOURS=8`, `KVCACHE_TARGET_PCT=65`, `KVCACHE_MAX_DELETE=4000`, `KVCACHE_TMP_AGE_MIN=60`, `DRY_RUN`.

---

## 6. The `vllm serve` invocation (the exec)
```
/opt/radiance_entrypoint.sh <SNAP> --served-model-name <SERVED> --host 0.0.0.0 --port <PORT>
  --kv-cache-dtype fp8 --tensor-parallel-size <TP> --gpu-memory-utilization <GPU_UTIL>
  [--kv-cache-memory <KV_MEM>]
  [--kv-transfer-config <KVOFF_TIER_ARG>]            # L3 wiring (only if KVOFF_DISK set + L2 present)
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192
  --attention-backend R4D
  --speculative-config <SPEC_CFG>                    # dflash (default) or mtp
  <ASYNC_FLAG> <EXTRA> <MMIMG_ARG> <SKIPMM_ARG>
  --enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice
  --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --enable-per-request-metrics --enable-force-include-usage --enable-prompt-tokens-details
  --override-generation-config {"temperature":1.0,"top_p":0.95,"top_k":20}
  --chat-template <CT_PATH> [reasoning-effort]
```
- **`--max-num-seqs` = 4** is the shipped value (every dated snapshot agrees); the register's open experiment **4 → 2** (costs nothing to try, never tried under a controlled replay) is noted, not adopted.
- The `KVOFF_TIER_ARG` JSON is **space-free** (word-split at the call site) and deliberately **omits `cpu_bytes_to_use`** — `config/vllm.py:933` unconditionally `.update()`s it from `--kv-offloading-size`, so a copy here would just lose.
- The three per-request-metrics flags are **all required** (llama-swap computes prompt t/s as `(prompt_tokens − cached)/ttft`, so `--enable-prompt-tokens-details` is what stops every prefix-cache hit from being miscounted as real prefill).
- **Check in the log:** `"Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM"`, `"[radiance] native MXFP4 enabled on gfx12x"`, the R4D selections table. After a restart, the eagle-group line must read **`[8]`** (not all nine flagged).

---

## 7. Gotchas baked into the design (from the launcher comments)
1. **`PYTHONHASHSEED` pinned** (`KVOFF_HASHSEED=0`) — otherwise every restart orphans the on-disk cache (100% miss, silent).
2. **The fs tier has no eviction** — the reaper is mandatory, not advisory.
3. **`O_DIRECT` on the fs tier** — bypasses page cache; spare RAM can't act as a read cache in front of it (only the primary tier is a productive home for RAM).
4. **A failed load kills EngineCore** (`assert transfer_result.success`, no `get_block_ids_with_load_errors()` for `OffloadingConnector`) — `kv_load_failure_policy=recompute` is inert here.
5. **The offload region is pre-faulted + pinned** — overshooting `/dev/shm` fails the *start* with no log line (0.27.1); hence the mandatory clamp (and a second RAM clamp, since the tmpfs is only a limit).
6. **A restart orphans the `/dev/shm` region, not just the container** — hence the startup `kvoff_reap_orphans` (`fuser`-gated).
7. **`--ipc=host`** — the tmpfs that matters is the host's; size against `statvfs`, never `df` (it rounds up).
8. **The two whole-GPU entries can't coexist** — `podman stop` the other first.
