#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deep prompt-processing (prefill) sweep against a live llama-swap endpoint.
#
# Why a script and not a command line: BetterBench auto-detects the context
# window from GET /v1/models, and llama-swap's /v1/models lists entry ids only
# -- no max_model_len -- so detection fails and every deep rung is silently
# SKIPPED rather than run. --max-model-len must be passed explicitly, and it
# must match the MAXLEN the entry actually booted with.
#
# The depth ladder is capped deliberately. BetterBench runs a rung only if
#     depth + prefill_max_tokens + prefill_ctx_margin <= max_model_len
# i.e. at MAXLEN=204800 the highest legal depth is 204528. The shipped
# config/deep_prefill.json tops out at 250000, which is above that, so that
# rung would never execute. 200000 is 97.7% of the window and fits.
#
# This does NOT need a GPU exclusive window -- it drives the HTTP endpoint,
# so llama-swap must stay UP. But it is the only thing that should be talking
# to the card while it runs, or the deep rungs contend for the KV pool.
#
# BetterBench version: 0.4.0 (phase flags, --passes, update p50/p99).
# ---------------------------------------------------------------------------
set -euo pipefail

BB=${BB:-$HOME/ai/eval/betterbench-venv/bin/betterbench}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:1234/v1}
MODEL=${MODEL:-qwen3.8-27b-vllm}
MAXLEN=${MAXLEN:-204800}          # must equal the entry's booted max_model_len
PASSES=${PASSES:-5}
OUT=${OUT:-$HOME/ai/bench-history/betterbench-prefill-$(date +%Y%m%d-%H%M%S)}

CFG=$(mktemp /tmp/bb-prefill-XXXX.json)
trap 'rm -f "$CFG"' EXIT
cat > "$CFG" <<JSON
{
  "prefill_depths": [2000, 8000, 16000, 32000, 64000, 128000, 160000, 200000],
  "prefill_runs": $PASSES,
  "prefill_warmup": 1,
  "prefill_max_tokens": 16,
  "max_model_len": $MAXLEN,
  "timeout_s": 1800.0
}
JSON

mkdir -p "$OUT"
cp "$CFG" "$OUT/config.json"

"$BB" run \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --prefill \
  --max-model-len "$MAXLEN" \
  --config "$CFG" \
  --out "$OUT/results.json" \
  --note phase=prefill-deep \
  --note maxlen="$MAXLEN" \
  "$@"


# markdown for the repo (the same numbers, rendered)
"$BB" report "$OUT/results.json" --out "$OUT/results.md" || true

echo "results: $OUT"
