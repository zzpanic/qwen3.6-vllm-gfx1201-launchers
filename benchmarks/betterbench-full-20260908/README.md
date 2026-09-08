# BetterBench full run — qwen3.8-27b-vllm (via llama-swap)

- Date: 2026-09-08
- Endpoint: http://192.168.1.17:1234/v1 (llama-swap proxy -> container qwen38-27b-vllm, image stilldeadcode/vllm-radiance:0.9.3)
- Model: qwen3.8-27b-vllm (MXFP4 target + FP8 DFlash2 drafter), TP=1, maxseqs=4, max_model_len=204800
- Config: BetterBench 0.4.0 defaults (20 runs/category, 3 warmup, 48 req/level)
  - concurrency capped to [1, 2, 4] (serving instance maxseqs=4)
  - prefill ladder to 200,000 (the shipped 250k rung exceeds the 204800 window and would be skipped)
- Phases: decode + concurrency + prefill (all three, the "full" run)

Outputs: results.json, results.html, results.md
