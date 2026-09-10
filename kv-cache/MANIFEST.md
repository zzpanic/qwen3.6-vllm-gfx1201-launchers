# Release manifest

Assembled by `make-dist.sh` on 2026-09-10T08:33:42Z from the working
tree at `<repo>`.

| Path | Source | What it is |
|---|---|---|
| `README.md` | kv-cache/ | **Start here.** Status, roadmap, architecture. |
| `docs/` | kv-cache/ | Handover, operations runbook, implementation, known issues, future work, references, the correctness suite's specification, and the experimental record. |
| `patches/` | kv-cache/ | The eight house patches, their apply order, and `patches/README.md` — what they need to run, which two are fatal on failure, and the env gate on each. **Reconcile against upstream before using.** |
| `launcher/` | kv-cache/ + vllm/ | The KV-cache entry launcher and the base launcher it wraps. |
| `ops/` | vllm/ | The mandatory reaper, its systemd units, the mount snippets. |
| `bench/` | kv-cache/ | Harnesses, shipped for method. No result files. |
| `tools/` | kv-cache/ | `tierreport.py` — the tier sizing/speed report, its fixture generator and its functional test. Meant to be run for its numbers, on your own traffic. |

Deliberately excluded: `baselines/`, all bench-history, and every result file.
Single-machine numbers, some known polluted — ship the method, not the numbers.

## Reading the documents

The docs were written against the author's flat working tree and the release is
nested, so paths inside them do not always match paths here:

| In the documents | In this release |
|---|---|
| `<repo>` | the author's working tree — a placeholder, not a path you have |
| `<repo>/kv-cache/patch_*.py` | `kv-cache/patches/` |
| `<repo>/kv-cache/*.md` | `kv-cache/docs/` |
| `<repo>/kvcache-reap.*` | `kv-cache/ops/` |
| `<repo>/kv-cache/*bench*.py`, `phase*-read.sh` | `kv-cache/bench/` |
| `<repo>/kv-cache/tierreport.py`, `tier_report_*.py` | `kv-cache/tools/` |
| `<repo>/ggz14-mxfp4` | ggz14's clone, `radiance-vllm-mxfp4/` beside the repo (see the root README's Quick start) |
| `<repo>/benchmarks/`, `bench-history/` | not shipped — distilled into `docs/kv-cache-historical.md` |

**This is a proof of concept.** Read the status section of `README.md` before
using or citing any of it.
