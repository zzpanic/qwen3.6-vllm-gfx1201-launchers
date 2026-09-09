# Benchmark harnesses

Shipped for the **method**, not for any number they once produced. Results are
deliberately not included: they are single-machine, and some are known polluted.

`equivbench.py` and `mixedbench.py` are the two that check *correctness* rather
than speed, and they are not interchangeable. `equivbench.py` evicts every tier
and asks whether a pure external hit gives the same answer as a recompute.
`mixedbench.py` deliberately leaves the head of the prompt in the GPU cache and
evicts only its tail, so the request hits BOTH tiers at once — the case where
vLLM reports a per-group prefix hit that diverges across KV groups, and the case
that produced a fatal assertion before the fix in `docs/` R3.15. A run that
never reaches the `MIXED` phase has tested nothing, and says so in its verdict.

`tierbench.py` is the one to start with for speed. It measures one long prefix in four
known cache states — cold / GPU / CPU / fs — sizing each eviction from the tier
capacities it reads out of the engine's boot log, and **refusing to report a
phase whose tier state did not come out as intended**. That refusal is the entire
reason it exists: an earlier harness pushed 27 GB through a 16 GiB CPU tier, so
its "CPU re-read" phase was really a third disk read, and all three of its
re-read phases returned the same ~78 s.

## The measurement trap that invalidated a whole benchmark run

vLLM matches the prefix cache by block **content**, not by chained prefix. A
harness that makes a run "cold" by varying a nonce in **block 0 only** does not
get a cold run — the body self-caches from the previous iteration. This is what
invalidated the BetterBench numbers for this stack. **Vary the whole body, or
you are measuring the cache against itself.**

Two more, both learned the expensive way:

* Request wall time is the wrong instrument for tier work. Judge the tier by
  `load_bytes`, never by chunk or token counters — decode inflates them.
* Never compare two promotion strategies without an identical cache fill. An
  80 s vs 14 s result was withdrawn for exactly this.
