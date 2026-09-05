#!/bin/bash
# BFCL v4 (Berkeley Function-Calling Leaderboard) against a LIVE endpoint.
#
# This is the eval that matters most for an agentic box: it measures tool-calling, which
# is what the model actually spends its day doing here. Like run-gsm8k.sh it drives the
# production serving path over HTTP and takes NO exclusive GPU window.
#
# One-time setup:
#   python3 -m venv ~/ai/eval/bfcl-venv
#   ~/ai/eval/bfcl-venv/bin/pip install bfcl-eval
#   ./run-bfcl.sh --register          # patches BFCL's model registry, idempotent
#
# Usage:
#   ./run-bfcl.sh --register
#   ./run-bfcl.sh                     # subset, ~65 min
#   CATS=all ./run-bfcl.sh            # full 3641 entries, ~3 h
#
# ---------------------------------------------------------------------------------------
# TWO TRAPS. Both cost real time here; neither is obvious.
#
# 1. `underscore_to_dot=True` is MANDATORY for any OpenAI-style handler. BFCL's
#    convert_to_tool() rewrites "." -> "_" in function names UNCONDITIONALLY for
#    ModelStyle.OPENAI_COMPLETIONS, because OpenAI's name regex is ^[a-zA-Z0-9_-]{1,64}$.
#    The flag does not control that rewrite -- it only tells the CHECKER to undo it. With
#    it False every dotted function name mismatches its answer key and you score garbage
#    that looks plausible: simple_java 3.00%, multiple 37.50%, 92/100 java failures typed
#    wrong_func_name. That cost a full re-score.
#
# 2. Do NOT use BFCL's built-in Qwen handler. `QwenFCHandler` hardcodes the Qwen3 chat
#    template and <tool_call>{json}</tool_call> parsing. Qwen3.8 emits the nested
#    <tool_call><function=n><parameter=p> XML form, so that handler scores ~0. It is not
#    needed: the servers here run --enable-auto-tool-choice --tool-call-parser qwen3_xml,
#    so they return properly parsed OpenAI tool_calls objects and BFCL's stock
#    OpenAICompletionsHandler speaks exactly that. It is an api_inference handler, so BFCL
#    never tries to spin up its own server and --skip-server-setup is irrelevant.
# ---------------------------------------------------------------------------------------
set -euo pipefail

VENV="${VENV:-$HOME/ai/eval/bfcl-venv}"
ENTRY="${ENTRY:-qwen3.8-27b-mxfp4}"     # the name your server answers to
KEY="${KEY:-qwen3.8-27b-mxfp4-FC}"      # BFCL model key; keep one per stack (see below)
DISPLAY="${DISPLAY_NAME:-Qwen3.8-27B MXFP4 (FC, local vLLM)}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:1234/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"

# Give each stack its OWN key. Scores are written to score/<key>/, so reusing a key
# silently overwrites the run you wanted to compare against -- and the served entry name
# is NOT a safe key, because the same llama-swap name can serve different weights over
# time. That is exactly what happened here: `qwen3.8-27b-vllm` served int4 in August 2026
# and MXFP4 in September.

# Subset rationale: this covers non-live AST (simple_python/multiple/parallel), live AST
# (live_simple/live_parallel/live_parallel_multiple) and BOTH decision rows (irrelevance,
# live_relevance) in 1347 entries. It omits live_multiple (1052) and live_irrelevance (884),
# which are most of the 3 h in a full run, and simple_java/simple_javascript, which are a
# known model-level weakness (95% python vs 63% java) rather than a quantisation-sensitive
# axis. Use CATS=all when you need a leaderboard-comparable number.
SUBSET="simple_python,multiple,parallel,live_simple,live_parallel,live_parallel_multiple,live_relevance,irrelevance"
CATS="${CATS:-$SUBSET}"
[ "$CATS" = "all" ] && CATS="irrelevance,live_irrelevance,live_multiple,live_parallel,live_parallel_multiple,live_relevance,live_simple,multiple,parallel,parallel_multiple,simple_java,simple_javascript,simple_python"

. "$VENV/bin/activate"

if [ "${1:-}" = "--register" ]; then
  python - "$KEY" "$ENTRY" "$DISPLAY" <<'PY'
import pathlib, sys
import bfcl_eval.constants.model_config as mc
key, entry, disp = sys.argv[1:4]
path = pathlib.Path(mc.__file__); src = path.read_text()
marker = f"# --- local: {key} ---"
if marker in src:
    print(f"{key} already registered"); raise SystemExit
path.write_text(src + f'''

{marker}
MODEL_CONFIG_MAPPING["{key}"] = ModelConfig(
    model_name="{entry}",
    display_name="{disp}",
    url="https://localhost", org="local", license="Apache-2.0",
    model_handler=OpenAICompletionsHandler,
    input_price=None, output_price=None, is_fc_model=True,
    underscore_to_dot=True,   # MANDATORY -- see trap 1 in run-bfcl.sh
)
''')
print(f"registered {key} in {path}")
PY
  echo "re-run this after any 'pip install -U bfcl-eval' -- upgrades overwrite the registry"
  exit 0
fi

# --temperature 0.001 rather than 0: BFCL is scored on structure, not prose, and near-greedy
# keeps the run reproducible. num-threads must not exceed the server's MAXSEQS (2); higher
# just queues.
bfcl generate --model "$KEY" --test-category "$CATS" --num-threads "${THREADS:-2}" --temperature 0.001
bfcl evaluate --model "$KEY" --test-category "$CATS"

echo
echo "Scores are under \$VENV/../bfcl-*/score/$KEY/ (and the aggregate CSVs beside it)."
echo "Compare per-category, not on the aggregate: the aggregate reweights whenever the"
echo "category set changes, so a subset's 'Overall Acc' is NOT comparable to a full run's."
