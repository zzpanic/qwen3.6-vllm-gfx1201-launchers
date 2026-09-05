# BFCL v4 — MXFP4 W4A8, 8-category subset (2026-09-05)

Tool-calling accuracy for `startup-qwen3.8-27b-mxfp4.sh` at its shipped defaults, run
against the **live endpoint** with `--temperature 0.001` and `--num-threads 2`. Reproduce
with [`../../run-bfcl.sh`](../../run-bfcl.sh); it took ~50 minutes and needs no exclusive
GPU window.

Handler is the stock `OpenAICompletionsHandler` with `underscore_to_dot=True` and vLLM's
`--tool-call-parser qwen3_xml`. BFCL's own `QwenFCHandler` scores ~0 on this model — it
expects nested XML where Qwen3.8 emits `<tool_call>{json}</tool_call>` — so using it would
have produced a confident, wrong, near-zero result.

## Read the per-category numbers. Ignore "Overall Acc".

`data_overall.csv` in this folder ranks this run at **11.55%**. That number is an artifact
and must not be quoted. BFCL's aggregate divides by every category in the harness, and this
is an 8-category subset: `multi_turn` and the agentic categories were not run, score
`0.00%`/`N/A`, and drag the mean to a sixth of the real figure. **A subset's aggregate is
not comparable to a full run's, or to another subset with a different category set.**
The same trap sits on the int4 side, where a `single_turn`-only run reports "25.29%"
against a true per-category accuracy near 90%.

## MXFP4 against int4, per category

int4 figures are the 2026-08-15 run (`qwen3.8-27b-vllm-FC`, int4 W4A16 + MTP). **Identical
problem counts per category on both sides**, so these are matched pairs.

| category | MXFP4 W4A8 | int4 W4A16 | delta | n |
|---|--:|--:|--:|--:|
| simple_python | 95.75% | 95.00% | +0.75 | 400 |
| multiple | 95.00% | 95.50% | −0.50 | 200 |
| parallel | 91.50% | 90.50% | +1.00 | 200 |
| live_simple | 88.37% | 87.60% | +0.78 | 258 |
| live_parallel | 93.75% | 93.75% | ±0.00 | 16 |
| live_parallel_multiple | 79.17% | 75.00% | +4.17 | 24 |
| live_relevance | 81.25% | 75.00% | +6.25 | 16 |
| irrelevance | 82.92% | 86.25% | −3.33 | 240 |
| **weighted over all 1354** | **90.84%** | **90.84%** | **±0.00** | **1354** |

**The weighted totals are equal to two decimal places.** That is a coincidence in its
exactness, but not in its direction: the per-category deltas scatter either side of zero,
and the three largest ones sit on the three smallest categories — `live_relevance` and
`live_parallel_multiple` have 16 and 24 problems, where a single item is 6.3 and 4.2 points.
One item either way accounts for most of what looks like a gap.

So: **MXFP4 W4A8 and int4 W4A16 are indistinguishable on tool calling here**, which is the
same conclusion GSM8K reached on arithmetic (95.45% vs 94.77%). Both were worth measuring
because a bad quantisation shows up in exactly these two places, and neither shows it.

`irrelevance` (−3.33 on 240 problems, ~8 items) is the only delta large enough on a big
enough category to be worth a second look, and it is the category where a model is scored
for *declining* to call a tool. It is not confirmed as real: this is a single unseeded run
per side, and the arms differ in more than the weight format — the int4 run used MTP ×4 and
a different KV pool. Do not attribute it to MXFP4 without a repeat.

## What is not here

`multi_turn`, the agentic web-search and memory categories, and BFCL's format-sensitivity
sweep. The subset was chosen to cover single-turn AST accuracy plus both relevance
detections at ~50 minutes rather than ~3 hours, since every category already had an int4
score from August to compare against.
