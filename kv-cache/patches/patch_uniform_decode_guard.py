#!/usr/bin/env python3
"""Uniform-decode guard: a prefill chunk of exactly 1 + num_speculative_tokens tokens must not replay
the speculative-decode CUDA graph.

Target: vllm/v1/worker/gpu/model_runner.py (V2 runner), execute_model's batch dispatch.

THE BUG (found 2026-10-05; reproduce it with kv-cache/tools/lenprobe.py). get_uniform_token_count() calls
a batch "uniform" from its SHAPE alone -- every request has the same number of scheduled tokens --
and a uniform count of decode_query_len (1 + 7 with DFlash x7) selects the FULL cudagraph captured
for speculative decode. One request whose final prefill chunk is exactly 8 tokens has that shape
(1 request x 8 tokens), so the decode graph is replayed over a PREFILL step: the reply is a 1-2
token fragment and EOS. In align mode chunks end on block boundaries, so every prompt of length
k * block_size + 8 hits it (1,656 / 3,304 / 4,952 at block 1,648), cold, batch 1, lazy on or off;
neighbouring lengths are fine. About 1 prompt in 1,648. A concurrent request in the same step
breaks the uniform shape, which is why the soak saw it less often with two streams.

THE FIX. Real decode steps carry their drafts in scheduler_output.scheduled_spec_decode_tokens
(the runner itself reads len() of them a few lines later). A batch is only a uniform DECODE batch
if every request has exactly count - 1 drafts scheduled; otherwise it is dispatched as a general
batch (piecewise graph / eager) -- the path any mixed batch already takes. Dummy runs (graph
capture and profiling) build no draft lists and are left exactly as they were, so capture is
unchanged. Inert without speculative decoding (count 1 needs 0 drafts, which every request has).
"""
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
from _patchlib import apply  # noqa: E402

F = Path(sysconfig.get_paths()["purelib"]) / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"

apply(F,
      "        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)\n",
      "        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)\n"
      "        # radiance (patch_uniform_decode_guard.py): uniform by SHAPE is not uniform DECODE. A\n"
      "        # prefill chunk of exactly 1 + num_spec tokens must not replay the spec-decode graph.\n"
      "        if uniform_tok_count is not None and uniform_tok_count > 1 and not dummy_run:\n"
      "            _ugd = scheduler_output.scheduled_spec_decode_tokens or {}\n"
      "            if any(len(_ugd.get(_r, ())) != uniform_tok_count - 1\n"
      "                   for _r in scheduler_output.num_scheduled_tokens):\n"
      "                uniform_tok_count = None\n",
      "radiance (patch_uniform_decode_guard.py)", "V2 runner: uniform-decode guard")
