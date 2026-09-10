# Launchers — and the version gap you need to know about

The entry launcher is **`startup-qwen3.8-27b-kvcache.sh`, at the repository
root**, alongside the other `startup-*.sh` scripts. It is a thin wrapper: it sets
the KV-cache environment, with the reasoning and the measured cost of every knob
inline, and then execs a base launcher that actually serves the model.

`serve-mxfp4-kvcache-base.sh` in this directory is that base launcher.

## Start from the MXFP4 build that already works

**If you have `startup-qwen3.8-27b-mxfp4.sh` serving, that is the structure to
build on — and this launcher already is it.**

The repository root ships `startup-qwen3.8-27b-mxfp4.sh`: the plain MXFP4 entry,
no disk tier, none of the eight house patches, and every knob in it carrying the
measurement that chose it. Get that serving first. It is the shorter path to a
working engine, and if it does not serve, nothing in `kv-cache/` will either.

`serve-mxfp4-kvcache-base.sh` in this directory is **that same launcher plus a
named delta**. Its tuning defaults are not hand-maintained here — they are copied
across from `startup-qwen3.8-27b-mxfp4.sh` when the release is assembled, so the
two cannot drift. If you have tuned the MXFP4 launcher for your own hardware,
carry the same values over; they are the values everything in this repository was
measured on.

The delta, in full — this is the entire difference between a working MXFP4
instance and the cache work:

| # | Addition | Where |
|---|---|---|
| 1 | `HOUSE` resolution + existence check | beside `REPO` |
| 2 | the `KVOFF_DISK*` / `KVOFF_*` knob block | with the other knobs |
| 3 | `/dev/shm` fit check and the RAM clamp for an explicit tier size | the offload sizing block |
| 4 | the fs-tier `--kv-transfer-config` JSON builder | after the sizing block |
| 5 | `-v $KVOFF_DISK`, `-v $HOUSE`, `PYTHONHASHSEED`, six `RADIANCE_*` gates | the container invocation |
| 6 | eight `PYTHONPATH=/patches python3 /house/patch_*.py` lines | the patch prelude |
| 7 | `${KVOFF_TIER_ARG:+--kv-transfer-config ...}` | the `vllm serve` arguments |

Nothing else differs. If you want to add the cache to a launcher of your own,
those seven items are the whole job.

## Why they are still two files

Shipping both is a wart, not a design, and they should be merged. That merge is
**stage 4 (refactoring)** of the roadmap in the top-level README, and deliberately
not earlier: until the stage-1 metrics harness exists, a merge cannot be shown to
have preserved behaviour. Until then the split has one virtue — the cache work is
a PROOF OF CONCEPT with known correctness errors, and it should not be reachable
by accident from the production path.

## Where both of them come from

Neither script is a fork of anyone's kernels. Both descend from **ggz14's
`serve-mxfp4.sh`** (`codeberg.org/ggz14/radiance-vllm-mxfp4`, v0.11.0, image
`stilldeadcode/vllm-radiance:0.9.3`), which owns the MXFP4 GEMM, the R4D attention
path and the DFlash2 drafter integration. What this repository contributes is the
*arrangement* — which knobs, at which values, on this card — and, in the KV-cache
case, the offload tiers and the eight patches in `../patches/`.

## Porting to another machine

Both scripts resolve every path from an environment variable with a `$HOME`
relative default. On a different box set at least `BASE_LAUNCHER`, `MODELS`,
`REPO` and `HOUSE`. Run either with `-h` for the full knob list.
