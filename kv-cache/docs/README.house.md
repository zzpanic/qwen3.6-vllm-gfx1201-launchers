# KV cache work — house files

Our own KV-cache code and documents for the `qwen3.8-27b-vllm` entry.

These used to live loose inside `<repo>/ggz14-mxfp4/`, which is ggz14's upstream clone
and is excluded from this repo by `.gitignore:74`. That meant they were tracked by nothing:
a `git clean` in that checkout, or a fresh clone of the upstream tree, would have deleted
them with no copy anywhere. They now live here and ride their own bind mount.

| file | what it is |
|---|---|
| `cache-preemption-patch-plan.md` | The plan for the "retain, don't recompute" work. Read **Revision 2** first — it supersedes parts of the original and reorders the build. |
| `patch_offload_mixed_hit.py` | Stops `OffloadingConnector` killing the engine on a mixed local+external prefix hit. Applied at every boot. |
| `tierbench.py` | Deterministic KV tier attribution bench. Also the acceptance test for the R2.9.2 instrumentation. |

## How the patch is wired in

`llama-swap-ggz14-27b.sh` resolves `HOUSE` (default: this directory), bind-mounts it at
`/house`, and runs the patch from there:

```
PYTHONPATH=/patches python3 /house/patch_offload_mixed_hit.py
```

`PYTHONPATH=/patches` is what lets it import ggz14's `_patchlib` the way the in-repo
patches do — Python puts the *script's* directory on `sys.path[0]`, not the working
directory, so `cd /patches` alone would not be enough. The upstream clone stays pristine.

Behaviour is still gated at runtime by `RADIANCE_OFFLOAD_MIXED_HIT` (`KVOFF_MIXED_HIT` in
the launcher): `1` = upstream behaviour, `0` = the patch's decline-the-hit path.

## tierbench

```
./tierbench.py --dry-run                 # detect capacities, print the plan, send nothing
./tierbench.py --yes                     # full run, ~20 min
./tierbench.py --yes --concurrent 3      # add the multi-client preemption phase
./tierbench.py --yes --salt 7a3f9c       # replay a previous run's content
```

It measures one 90k-token prefix in four known cache states — `cold` (recompute),
`gpu`, `cpu` (evicted from GPU only), `fs` (evicted from CPU too) — sizing each eviction
from the tier capacities it reads out of the engine's boot log, and **refusing to report a
phase whose tier state did not come out as intended**. That last part is the reason it
exists: `~/audit/stress/harness.py` pushed 27.19 GB through a 16 GiB CPU tier, so its
"RE-READ-CPU" phase was really a third fs read, which is why its three re-read phases
returned 77.0 / 78.5 / 78.0 s.

Results land in `~/audit/stress/tierbench/REPORT-<salt>.md` and `RESULTS-<salt>.json`.

**It evicts the entire KV cache.** No GPU exclusive window is needed (it drives the HTTP
endpoint), but it needs the box to itself for the duration.

## `patch_kv_offload_instrumentation.py`

Instrumentation-only house patch, applied at container start straight after
`patch_offload_mixed_hit.py`. Adds the three numbers the 2026-09-07 baseline needed and did
not have (see `cache-preemption-patch-plan.md` R3.6):

| metric | why |
|---|---|
| `kv_offload_cpu_cache_evictable_perc` | one of the two terms `prepare_store()` tests for admission |
| `kv_offload_cpu_cache_free_perc` | the other term |
| `kv_offload_fs_inflight_jobs` | the cascade backlog that pins the CPU tier shut |

It also widens both `lookup_async_delay` histograms from a 10 s top bucket to 600 s — the
measured stall was 60.6 s, so every real observation was landing in `+Inf`.

**Do not read `kv_offload_cpu_cache_usage_perc` as residency.** It subtracts evictable blocks,
so it means "fraction pinned by in-flight transfers" and reads 0.0 at idle with hundreds of GB
on the fs tier.

Verify without a GPU window:

```bash
podman run --rm --entrypoint /bin/bash \
  -v <repo>/ggz14-mxfp4:/patches:z \
  -v <repo>/kv-cache:/house:z \
  stilldeadcode/vllm-radiance:0.9.3 -lc \
  'PYTHONPATH=/patches python3 /house/patch_kv_offload_instrumentation.py'
```
