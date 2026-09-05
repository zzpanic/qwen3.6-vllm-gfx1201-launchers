#!/bin/bash
# GSM8K against a LIVE endpoint -- the quality number for the serving path.
#
# Why against the live server rather than the weights: a weights-only eval tells you about
# the checkpoint. This tells you about the thing you actually serve -- fp8 KV, speculative
# decoding, CUDA graphs, your chat template and your server-default sampler all included.
# Those can cost accuracy, and if they do you want to know.
#
# Takes NO exclusive GPU window. The model must already be resident; this drives the
# endpoint over HTTP like any other client.
#
# One-time setup (851 MB, deliberately no torch -- the [api] extra plus local-completions
# avoids pulling it in):
#   python3 -m venv ~/ai/eval/venv
#   ~/ai/eval/venv/bin/pip install "lm-eval[api]==0.4.12"
#
# Usage:
#   ./run-gsm8k.sh                                  # defaults below
#   MODEL=qwen3.8-27b-int4 TOKENIZER=/path ./run-gsm8k.sh
#
# Runtime: ~76 min for 1319 problems at num_concurrent=2 (~3.3 s/problem).
set -euo pipefail

LM_EVAL="${LM_EVAL:-$HOME/ai/eval/venv/bin/lm-eval}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:1234/v1/completions}"
MODEL="${MODEL:-qwen3.8-27b-mxfp4}"
TOKENIZER="${TOKENIZER:?set TOKENIZER to the local checkpoint dir -- lm-eval needs it to count tokens}"
MAXLEN="${MAXLEN:-131072}"
# MUST NOT exceed the server's MAXSEQS (2 in the shipped configs). Higher just queues, and
# the wall-clock saving is zero.
CONC="${CONC:-2}"
OUT="${OUT:-$HOME/ai/eval/results-gsm8k-$(date +%Y%m%d-%H%M%S)}"

[ -x "$LM_EVAL" ] || { echo "no lm-eval at $LM_EVAL -- see the setup comment above" >&2; exit 1; }

# Temperature 1.0 because that is what these scripts serve, NOT because it is the best
# score available. Greedy scores ~3 points higher on this harness. If you want a number
# comparable to a published greedy figure, set TEMP=0 and say so when you report it.
TEMP="${TEMP:-1.0}"
TOPP="${TOPP:-0.95}"

# Do NOT add min_p here. vLLM returns 400 on min_p under speculative decoding, and the
# server already applies min_p=0.0 as a default, so omitting it matches anyway.
"$LM_EVAL" --model local-completions \
  --model_args "model=$MODEL,base_url=$ENDPOINT,num_concurrent=$CONC,max_retries=3,tokenized_requests=False,max_length=$MAXLEN,tokenizer=$TOKENIZER" \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs "max_gen_toks=8192,do_sample=True,temperature=$TEMP,top_p=$TOPP" \
  --batch_size 1 --log_samples --output_path "$OUT"

echo
echo "results: $OUT"
echo
echo "Reading the number: strict-match and flexible-extract should agree exactly. If they"
echo "do not, the model is not reliably emitting the '#### N' form and some credit is"
echo "coming from loose parsing. The +-Stderr is BINOMIAL error on 1319 problems only"
echo "(~+-0.6 pp); run-to-run sampling variance sits on top of it, so the standard error on"
echo "a DIFFERENCE between two single runs is ~0.9 pp. Gaps inside ~1.7 pp are not"
echo "distinguishable."
