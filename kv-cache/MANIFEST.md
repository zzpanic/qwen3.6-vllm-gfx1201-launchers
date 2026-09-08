# Release manifest

Assembled by `make-dist.sh` on 2026-09-08T23:03:05Z from the working
tree at `<repo>`.

| Path | Source | What it is |
|---|---|---|
| `README.md` | kv-cache/ | **Start here.** Status, roadmap, architecture. |
| `docs/` | kv-cache/ | Handover, operations runbook, implementation, known issues, future work, references. |
| `patches/` | kv-cache/ | The seven house patches + their apply order. **Reconcile against upstream before using.** |
| `launcher/` | kv-cache/ + vllm/ | The KV-cache entry launcher and the base launcher it wraps. |
| `ops/` | vllm/ | The mandatory reaper, its systemd units, the mount snippets. |
| `bench/` | kv-cache/ | Harnesses, shipped for method. No result files. |

Deliberately excluded: `baselines/`, all bench-history, and every result file.
Single-machine numbers, some known polluted — ship the method, not the numbers.

**This is a proof of concept.** Read the status section of `README.md` before
using or citing any of it.
