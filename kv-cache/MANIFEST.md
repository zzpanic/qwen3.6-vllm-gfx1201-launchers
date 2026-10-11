# What is in `kv-cache/`

| Path | What it is |
|---|---|
| `README.md` | **Start here.** The two options (RAM only, RAM + disk), the patch set and how it differs from ggz14. |
| `docs/SETUP.md` | Prerequisites, sizing, the reaper, the tools and the metric traps. |
| `patches/` | The house patches, `APPLY-ORDER.txt`, and `patches/README.md`: which build applies each, which are fatal, the env gate on each. **Check against upstream before reusing.** |
| `ops/` | The reaper script (the launcher runs it inside the container), optional systemd units, the fstab lines. |
| `bench/` | Harnesses, shipped for the method, including `turnbench.py`, the cached-vs-cold exactness gate. No result files. |
| `tools/` | `kvwatch.py` and `kvtable.py` (any build), `kvvalidate.py` and `tierreport.py` (need `KVOFF_MINIMAL=0`), and the correctness tools (`lenprobe.py`, `val_resume.py`, ...). |
| `examples/` | One real tier report and the `/metrics` snapshot it came from. |

The launcher is `startup-qwen3.8-27b-mxfp4.sh` at the repository root; [KV-CACHE.md](../KV-CACHE.md) is the overview.
