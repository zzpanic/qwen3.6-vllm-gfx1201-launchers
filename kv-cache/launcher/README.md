# Launchers — and the version gap you need to know about

The entry launcher is **`startup-qwen3.8-27b-kvcache.sh`, at the repository
root**, alongside the other `startup-*.sh` scripts. It is a thin wrapper: it sets
the KV-cache environment, with the reasoning and the measured cost of every knob
inline, and then execs a base launcher that actually serves the model.

`serve-mxfp4-kvcache-base.sh` in this directory is that base launcher.

## Why there are two base launchers in this repository

**This is the first thing to reconcile, and it is deliberate that you can see it.**

The repository root already ships `startup-qwen3.8-27b-mxfp4.sh`. That script is
an **earlier generation**: it has the CPU offload tier only. It has no disk tier,
none of the seven house patches, and none of the eviction-policy, stride,
eagle-group or pending-is-miss knobs the KV-cache work depends on. Point the
wrapper at it and the wrapper's environment is silently ignored — you get a CPU
tier and nothing else, with no error to tell you so.

`serve-mxfp4-kvcache-base.sh` is the later house copy that carries the full
machinery. The wrapper defaults to it for that reason, and only for that reason.

The two scripts are otherwise substantially the same file and **should be merged**
— shipping both is a wart, not a design. Doing that merge is stage 1 (upstream
reconciliation) of the roadmap in the top-level README. It is left visible rather
than papered over because a silent version skew here is exactly the kind of thing
that costs someone a day.

## Porting to another machine

Both scripts resolve every path from an environment variable with a `$HOME`
relative default. On a different box set at least `BASE_LAUNCHER`, `MODELS`,
`REPO` and `HOUSE`. Run either with `-h` for the full knob list.
