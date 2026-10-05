# kvwatch.py -- watch the cache work (either build)

```
watch -n 5 python3 kv-cache/tools/kvwatch.py      # from the repository root
```

One screen, refreshed every 5 s: GPU and offload-tier prefix-cache hit rates
(lifetime, and since the last refresh -- read the second), bytes the tier loaded
and stored, and the last few requests with how much of each was cached. Read-only.

It reads only metrics upstream vLLM exports, so it is the one tool here that works
on the **default** build. It reads the `qwen3.8-27b-kvcache` entry through llama-swap
on `:1234`; set `KVWATCH_METRICS=http://127.0.0.1:<port>/metrics` to read vLLM
directly (the per-request table needs llama-swap and is skipped without it). The
first refresh prints no rates -- it has nothing to difference against yet.

The two tools below need the **disk** build (`KVCACHE_DISK_TIER=1`):
the counters they read are added by its instrumentation patches. Do not run
`kvvalidate.py` against the default build -- it still runs, but reads the missing
counters as zero and reports FAILs (`cpu_equals_external`, `disk_vs_engine`) that
are not real.

# tierreport.py -- justify the size and speed of each cache tier

```
./tierreport.py --base http://127.0.0.1:8000 --out ./report
```

Scrapes `/metrics` once and writes `results.md` + `results.json`, BetterBench
style, plus a table on stdout. It answers three questions in plain language:

1. Have I allocated too much (or too little) RAM to the CPU tier?
2. Is my disk too slow -- should I buy NVMe?
3. Is turning this layered cache on adding value at all?

## It needs patch 2

`patches/patch_offload_tier_report.py` (patch 11) adds the 19 `tier`-labelled series the
report reads. Without it the tool still runs, still reports the prompt-token
source split and the recompute baseline, and says exactly what is missing and
why -- but there is no per-tier row, because the engine unlabelled
`load_bytes`/`load_time` cover the CPU tier only. Dividing those describes memcpy
speed and says nothing at all about the disk.

## Why it is all counters and histograms

You run this against your own long-lived server, scraped once. A gauge is an
instantaneous sample: read once off a box that has been up three weeks it
carries almost no information, and most of the existing tier instrumentation is
gauges. Every judgment here comes from a monotonic counter or a histogram
instead. Where a level genuinely matters -- occupancy -- the metric is the
distribution over time, so the report can say "your RAM tier was above 95% full
for 80% of the time", which is a sizing statement, rather than "it is full now",
which is not.

`--snapshot FILE` then later `--since FILE` differences two scrapes, for people
who are benchmarking rather than observing. Counters and histogram components
are differenced; capacity gauges are levels and stand as read.

## Flags worth knowing

| Flag | What it does |
|---|---|
| `--out DIR` | write `results.md` + `results.json` |
| `--json` | the whole report on stdout as JSON |
| `--snapshot F` / `--since F` | window mode instead of lifetime |
| `--calibrate DIR` | time reads of real block files to get the raw device rate, independent of anything vLLM reports |
| `--metrics-file F` | read exposition text from a file (see the fixture below) |

Exit status is a signal: 0 = the cache is earning its place, 1 = at least one
verdict says otherwise, 2 = could not measure.

## Testing it without an engine

```
python3 tierreport.py            # against the live endpoint
./tierreport.py --metrics-file /tmp/fixture.prom
```

The tier-report patch has a fixture and a smoke test that exercise every verdict
branch against invented numbers in the real metric shapes. They are development
tests for the patch itself -- they must run INSIDE the container, where
`prometheus_client` and the vLLM tree exist -- so they are not part of this
release. `kvvalidate.py` is the check to run against a deployment.

## Reading `--calibrate` honestly

It opens real block files with `O_DIRECT` where the kernel allows it, so the
local page cache cannot make an HDD look like RAM. It cannot bypass a cache
BELOW the block device -- a hypervisor host cache, a RAID controller, a ZFS ARC
on the backing store. Run it twice: a much faster second pass means something
underneath is caching, and the first number is the honest one.

If the engine appears FASTER than the calibrated device, the two were not
measuring the same thing -- the calibration reads whole files one at a time
while the engine reads many blocks per job in parallel. The tool says so instead
of picking whichever number supports a purchase.

# Correctness tools (no special build; any OpenAI-compatible endpoint)

All default to llama-swap on `127.0.0.1:1234` and model `qwen3.8-27b-vllm`; pass `--base` /
`--model` otherwise. Greedy, batch 1 unless stated, with a fresh `cache_salt` per run so runs
cannot hit each other's cached prefixes.

- **`lenprobe.py`** -- regression test for the V2 uniform-decode bug
  (`patches/patch_uniform_decode_guard.py`): prompts whose last prefill chunk is exactly
  `1 + num_speculative_tokens` tokens. Exit 0 = guard working. `--block` is the attention block
  size from the boot log, `--spec` the draft count. About 2 minutes.
- **`val_resume.py`** -- the same long prompts answered cold and resumed from the prefix cache,
  compared token by token. `--evict 9` first pushes the cache out of the GPU with ~310k unrelated
  tokens so the resume is restored from the CPU/disk tier -- the test that caught fp16 ssm state
  failing there. Expect bit-exact at fp32.
- **`val_long.py`** -- ~25k-token registry with checkable answers (verbatim copies, deep lookups).
- **`mt_lazy_gate.py`** -- multi-turn, prefix-reuse gate. `--copy N` makes every turn copy N records
  back verbatim, so answers cross cache-block boundaries during decode.
- **`spec_ab.py`** -- greedy output with speculation on vs off (`SPEC_METHOD=none`).
- **`errmargin.py`, `klstat.py`** -- how far apart two runs of the above are: token agreement,
  |delta logprob|, flip margins in bf16 steps, and approximate per-token KL from the top-5
  logprobs. Measure a change against a noise floor -- e.g. `spec_ab.py` on vs off, 99.13% agreement
  and KL 0.0021 on the reference box -- not against zero.
