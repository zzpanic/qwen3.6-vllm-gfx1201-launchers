# House copy 2026-10-09 of SlyBase sly/radiance_lookup_draft.py (github.com/brianfu9/vllm-sly-radiance @7d2304a).
# ONE house change: default entry rule enter 16 / stay 6 / hot 5 (Sly: 8 / 3 / 3), chosen by the offline replay on
# the author's opencode.db agent history (the author's bench notes (triangle-20261009/lookup-replay, not shipped): 8/3/3 can lose steps on our traffic, 16/6/5 wins
# under both drafter-rate assumptions). The RADIANCE_LOOKUP_* knobs still override. Installed by patch_lookup_draft.py.
"""Prompt-lookup override of the DFlash2 draft (vLLM's V2 model runner), gfx12x.

DFlash2's drafter attends to the last 2048 tokens only, so wherever the text being written is a verbatim copy
of something further back (an edit's old_string, a quoted file, a repeated block) it cannot see the source and the
accepted tokens/step fall from ~7 to ~3.3 at 8k-99k of context. A suffix n-gram lookup over the whole context sees
it. This module lets that lookup replace the DFlash draft for a step -- only when it is very likely to win:

  * one kernel scans a request's token history (vLLM's `req_states.all_token_ids`, the whole context, no window) for
    the longest earlier occurrence of the current suffix (up to 24 tokens, most recent on a tie) -- only occurrences
    whose continuation starts more than the drafter's window back (MIN_DIST, 2048): a source inside the window is one
    the drafter sees, and it is better at it (copying inside the window: 7.3 tokens/step; the lookup lost 5-8 % on
    edits there) -- and one kernel decides per request:
      - enter lookup mode on a match of >= ENTER tokens (default 8),
      - stay in it while the last lookup step accepted >= HOT draft tokens (default 3) and a match of >= STAY tokens
        (default 3) exists,
      - otherwise the DFlash draft is left exactly as the graph wrote it;
  * in lookup mode the draft tokens are the tokens that followed the match, and the cached draft distribution
    (`draft_logits`, the "probabilistic" draft method's q) is rewritten to a point mass on them, so the rejection
    sampler sees a deterministic draft: accept with p(token), on rejection resample from p without it. Lossless
    like any draft; greedy requests compare argmax as before.

The verify width is unchanged (1 + 7 rows per step), the step's GPU work is the same graph plus two tiny kernels
after it (outside the graph; ~2 launches). Everything stays on the device: no host sync, no change to what the
scheduler sees.

Why not "any match": simulated on recorded prod traces (prompt + 200 greedy tokens, DFlash tokens/step measured
per request), lookup that takes every match >= 3 tokens costs 1-17 % on free generation and summarising
(short matches rarely continue), enter-8 / stay-3 costs ~0 there and gains +66..+125 % tokens/step on verbatim copy
from beyond the drafter's window (8k / 32k / 99k of context), +5 % inside it.

Measured (R9700, one stream, greedy, tokens/step lookup off -> on): a far verbatim copy 3.6 / 4.4 / 3.2 -> 7.0 / 7.0 /
7.1 at 8k / 32k / 99k of context, a 30-line edit 3.7 / 4.0 / 3.3 -> 5.8 / 6.5 / 6.8, free generation and summaries
unchanged (-5..0 % on a JSON-list task), 1k of context unchanged; the step time does not move (<= 0.2 ms).

Only the DFlash2 speculator of the V2 runner is touched (`GPUModelRunner.__init__` wraps `speculator.propose` after
it has run). Any exception turns the override off for the rest of the process; the DFlash draft is then what the
graph wrote.

Knobs (read at import):
  RADIANCE_LOOKUP_DRAFT   1 (default) | 0 = the DFlash draft as the graph writes it (A/B control; the hook is not installed)
  RADIANCE_LOOKUP_ENTER   match length that starts lookup mode (8)
  RADIANCE_LOOKUP_STAY    match length that keeps it going after a hot step (3)
  RADIANCE_LOOKUP_HOT     accepted draft tokens of the last lookup step that keep it hot (3)
  RADIANCE_LOOKUP_MIN_DIST  auto (default: the drafter's sliding window from its config, else 2048) | N: only sources
                          whose continuation starts more than N tokens before the current position count; 0 = any
  RADIANCE_LOOKUP_STATS   0 | N: log lookup share / accepted per lookup step every N draft steps (one device read)
  RADIANCE_LOOKUP_SWITCH  path: the override runs only while that file exists (A/B inside one process without a
                          restart: touch / rm it; unset = always on). One os.path.exists per draft step.
"""
import os
import sys

try:
    import triton
    import triton.language as tl
except Exception:                       # no triton (offline check): the kernels are then never launched
    triton = tl = None

ENABLED = os.environ.get("RADIANCE_LOOKUP_DRAFT", "1") != "0"
ENTER = int(os.environ.get("RADIANCE_LOOKUP_ENTER") or 16)  # house default (Sly: 8)
STAY = int(os.environ.get("RADIANCE_LOOKUP_STAY") or 6)  # house default (Sly: 3)
HOT = int(os.environ.get("RADIANCE_LOOKUP_HOT") or 5)  # house default (Sly: 3)
STATS_EVERY = int(os.environ.get("RADIANCE_LOOKUP_STATS") or 0)
SWITCH = os.environ.get("RADIANCE_LOOKUP_SWITCH") or ""
MIN_DIST_ENV = (os.environ.get("RADIANCE_LOOKUP_MIN_DIST") or "auto").strip().lower()
DEFAULT_WINDOW = 2048              # the DFlash2 checkpoint's sliding window, when its config does not say

MAXL = 24                          # match length cap
BLOCK = 1024                       # history positions per scan program
POS_BITS = 20                      # key = match_len << POS_BITS | continuation start; max_model_len must fit
_state = {"failed": False, "ready": False, "logged": False, "calls": 0}


# ---- reference (pure python; the check and the card bench compare the kernels against it) ------------------------

def ref_best_match(ids, length, minl, maxl=MAXL, min_dist=0):
    """(match_len, p): the longest suffix of ids[:length] (>= minl, capped at maxl) that ends at ids[p-1] for some
    1 <= p <= length-1 with length - p >= min_dist; ties -> the largest p. (0, 0) when there is none."""
    best = (0, 0)
    for p in range(1, length - max(min_dist, 1) + 1):
        m = 0
        while m < maxl and p - 1 - m >= 0 and ids[p - 1 - m] == ids[length - 1 - m]:
            m += 1
        if m >= minl and (m, p) > best:
            best = (m, p)
    return best


def ref_decide(m, p, length, k, prev_used, prev_len, num_sampled, enter=ENTER, stay=STAY, hot_min=HOT):
    """(use, continuation_len, hot): what the apply kernel does for one request."""
    hot = bool(prev_used) and prev_len + num_sampled == length and num_sampled - 1 >= hot_min
    need = stay if hot else enter
    use = m > 0 and m >= need
    return use, (min(length - p, k) if use else 0), hot


# ---- kernels -----------------------------------------------------------------------------------------------------

if triton is not None:
    @triton.jit
    def _scan_kernel(ids_ptr, ids_stride, len_ptr, idx_ptr, best_ptr, min_dist,
                     MINL: tl.constexpr, MAXL: tl.constexpr, POS_BITS: tl.constexpr, BLOCK: tl.constexpr):
        """best[row] = max over p of (match_len(p) << POS_BITS | p): p is where the continuation would start, i.e.
        ids[p-1] is the last token of the earlier occurrence; match_len counts tokens equal going backwards from
        p-1 and from the end of the history. Only p with length - p >= min_dist are candidates."""
        row = tl.program_id(0)
        start = tl.program_id(1) * BLOCK
        slot = tl.load(idx_ptr + row)
        if slot < 0:
            return
        length = tl.load(len_ptr + slot)
        if start >= length:
            return
        base = ids_ptr + slot.to(tl.int64) * ids_stride
        p = start + tl.arange(0, BLOCK)
        alive = (p >= 1) & (p < length) & (p + min_dist <= length)
        m = tl.zeros((BLOCK,), dtype=tl.int32)
        for j in tl.static_range(MAXL):
            pos = p - 1 - j
            ok = alive & (pos >= 0) & (length - 1 - j >= 0)
            a = tl.load(base + pos, mask=ok, other=-1)
            b = tl.load(base + (length - 1 - j), mask=length - 1 - j >= 0, other=-2)
            alive = ok & (a == b)
            m += alive.to(tl.int32)
        key = tl.where(m >= MINL, (m << POS_BITS) | p, 0)
        tl.atomic_max(best_ptr + row, tl.max(key, axis=0))

    @triton.jit
    def _apply_kernel(ids_ptr, ids_stride, len_ptr, idx_ptr, best_ptr, num_sampled_ptr, st_len_ptr, st_used_ptr,
                      stats_ptr, tokens_ptr, tokens_stride, logits_ptr, logits_stride_0, logits_stride_1,
                      cached_ptr, top_k,
                      K: tl.constexpr, ENTER: tl.constexpr, STAY: tl.constexpr, HOT: tl.constexpr,
                      POS_BITS: tl.constexpr, HAS_LOGITS: tl.constexpr, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        slot = tl.load(idx_ptr + row)
        if slot < 0:
            return
        length = tl.load(len_ptr + slot)
        key = tl.load(best_ptr + row)
        tl.store(best_ptr + row, 0)
        m = key >> POS_BITS
        p = key & ((1 << POS_BITS) - 1)
        ns = tl.load(num_sampled_ptr + row)
        prev_len = tl.load(st_len_ptr + slot)
        prev_used = tl.load(st_used_ptr + slot)
        cont = (prev_len + ns) == length                      # this request's previous step is the one just verified
        resolved = cont & (prev_used != 0)
        tl.atomic_add(stats_ptr + 2, resolved.to(tl.int32))
        tl.atomic_add(stats_ptr + 3, tl.where(resolved, ns - 1, 0))
        hot = resolved & ((ns - 1) >= HOT)
        need = tl.where(hot, STAY, ENTER)
        use = (m > 0) & (m >= need)
        tl.store(st_len_ptr + slot, length)
        tl.store(st_used_ptr + slot, use.to(tl.int32))
        tl.atomic_add(stats_ptr + 0, 1)
        tl.atomic_add(stats_ptr + 1, use.to(tl.int32))
        if use:
            clen = tl.minimum(length - p, K)
            offs = tl.arange(0, BLOCK_K)
            for step in tl.static_range(K):
                if step < clen:
                    y = tl.load(ids_ptr + slot.to(tl.int64) * ids_stride + p + step)
                    tl.store(tokens_ptr + row * tokens_stride + step, y.to(tl.int64))
                    if HAS_LOGITS:
                        cbase = (slot.to(tl.int64) * K + step) * top_k
                        old = tl.load(cached_ptr + cbase + offs, mask=offs < top_k, other=0)
                        lbase = logits_ptr + slot.to(tl.int64) * logits_stride_0 + step * logits_stride_1
                        tl.store(lbase + old, -float("inf"), mask=(offs < top_k) & (old != y.to(tl.int64)))
                        tl.store(lbase + y.to(tl.int64), 0.0)
                        tl.store(cached_ptr + cbase + offs, y.to(tl.int64), mask=offs < top_k)
else:
    _scan_kernel = _apply_kernel = None


# ---- the override --------------------------------------------------------------------------------------------------

def _log(msg):
    sys.stderr.write(f"[radiance] lookup draft: {msg}\n")
    sys.stderr.flush()


def _min_dist(sp):
    """Sources closer than this are the drafter's to copy: its sliding window (dflash_config.swa_window_size, else the
    checkpoint's sliding_window, else 2048), unless RADIANCE_LOOKUP_MIN_DIST says a number."""
    if MIN_DIST_ENV != "auto":
        return max(0, int(MIN_DIST_ENV))
    try:
        hf = sp.draft_model_config.hf_config
        w = (getattr(hf, "dflash_config", None) or {}).get("swa_window_size") or getattr(hf, "sliding_window", None)
        return int(w) if w else DEFAULT_WINDOW
    except Exception:
        return DEFAULT_WINDOW


def _prepare(sp, rs):
    """Per-process buffers and the shape guards; raises when this speculator / vLLM is not the one it was written for."""
    import torch
    ids = rs.all_token_ids.gpu
    if ids.dtype != torch.int32 or ids.dim() != 2:
        raise RuntimeError(f"all_token_ids is {ids.dtype} {tuple(ids.shape)}")
    if ids.shape[1] >= (1 << POS_BITS):
        raise RuntimeError(f"max_model_len {ids.shape[1]} does not fit {POS_BITS} bits")
    top_k = int(sp.selector_top_k)
    steps = int(sp.num_speculative_steps)
    dl = sp.draft_logits
    if dl is not None:
        vocab = int(sp.vllm_config.model_config.get_vocab_size())
        if dl.dtype != torch.float32 or dl.dim() != 3 or dl.stride(-1) != 1 or dl.shape[-1] < vocab:
            raise RuntimeError(f"draft_logits {dl.dtype} {tuple(dl.shape)} vs vocab {vocab}")
        if tuple(sp._cached_candidate_ids.shape) != (dl.shape[0], steps, top_k):
            raise RuntimeError(f"cached candidates {tuple(sp._cached_candidate_ids.shape)}")
    dev = ids.device
    n = int(rs.total_len.gpu.shape[0])
    _state.update(torch=torch, top_k=top_k, steps=steps, block_k=triton.next_power_of_2(top_k),
                  min_dist=_min_dist(sp),
                  best=torch.zeros(n, dtype=torch.int32, device=dev),
                  st_len=torch.full((n,), -1, dtype=torch.int32, device=dev),
                  st_used=torch.zeros(n, dtype=torch.int32, device=dev),
                  stats=torch.zeros(4, dtype=torch.int32, device=dev), max_len=int(ids.shape[1]), ready=True)


def _override(sp, rs, input_batch, num_sampled):
    if not _state["ready"]:
        _prepare(sp, rs)
    S = _state
    n = int(input_batch.num_reqs)
    ids, lens = rs.all_token_ids.gpu, rs.total_len.gpu
    bound = int(input_batch.seq_lens_cpu_upper_bound[:n].max()) + S["steps"] + 2      # CPU tensor: no device sync
    grid = (n, (min(bound, S["max_len"]) + BLOCK - 1) // BLOCK)
    _scan_kernel[grid](ids, ids.stride(0), lens, input_batch.idx_mapping, S["best"], S["min_dist"],
                       MINL=min(ENTER, STAY), MAXL=MAXL, POS_BITS=POS_BITS, BLOCK=BLOCK, num_warps=4)
    dl = sp.draft_logits
    tokens = sp.draft_tokens
    _apply_kernel[(n,)](ids, ids.stride(0), lens, input_batch.idx_mapping, S["best"], num_sampled, S["st_len"],
                        S["st_used"], S["stats"], tokens, tokens.stride(0),
                        dl if dl is not None else tokens, dl.stride(0) if dl is not None else 0,
                        dl.stride(1) if dl is not None else 0,
                        sp._cached_candidate_ids if dl is not None else tokens, S["top_k"],
                        K=S["steps"], ENTER=ENTER, STAY=STAY, HOT=HOT, POS_BITS=POS_BITS,
                        HAS_LOGITS=dl is not None, BLOCK_K=S["block_k"], num_warps=1)
    S["calls"] += 1
    if STATS_EVERY and S["calls"] % STATS_EVERY == 0:
        steps, taken, resolved, accepted = (int(x) for x in S["stats"].tolist())
        _log(f"{steps} request-steps, lookup took {taken} ({100 * taken / max(steps, 1):.1f} %), "
             f"{accepted / max(resolved, 1):.2f} draft tokens accepted per lookup step ({resolved} resolved)")
    if not S["logged"]:
        S["logged"] = True
        _log(f"active (enter {ENTER} stay {STAY} hot {HOT}, sources > {S['min_dist']} tokens back, "
             f"{'point-mass draft distribution' if dl is not None else 'greedy draft'}, "
             f"steps {S['steps']}, history from {tuple(ids.shape)} int32)")


def _hook(sp, runner):
    orig = sp.propose

    def propose(*args, **kw):
        out = orig(*args, **kw)
        if ENABLED and not _state["failed"] and not kw.get("dummy_run") and (not SWITCH or os.path.exists(SWITCH)):
            try:
                input_batch = kw["input_batch"] if "input_batch" in kw else args[0]
                num_sampled = kw["num_sampled"] if "num_sampled" in kw else args[5]
                _override(sp, runner.req_states, input_batch, num_sampled)
            except Exception as e:      # never take a serve down: the DFlash draft is what the graph wrote
                _state["failed"] = True
                _log(f"off after an error: {e!r}")
        return out

    sp.propose = propose


def install():
    """Wrap GPUModelRunner.__init__ so that a DFlash2 speculator gets the lookup override. True when installed or
    deliberately off (env), False when this vLLM has no V2 runner / DFlash2 speculator."""
    if not ENABLED:
        _log("off (RADIANCE_LOOKUP_DRAFT=0): the DFlash draft as the graph writes it")
        return True
    if min(ENTER, STAY, HOT) < 1 or ENTER < STAY or ENTER > MAXL:
        _log(f"skipped: enter {ENTER} stay {STAY} hot {HOT} out of range")
        return True
    if triton is None:
        return False
    try:
        from vllm.platforms.rocm import on_gfx12x
        if not on_gfx12x():
            _log("skipped: not gfx12x")
            return True
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
    except Exception:
        return False
    if getattr(GPUModelRunner.__init__, "_radiance_lookup", False):
        return True
    orig_init = GPUModelRunner.__init__

    def __init__(self, *args, **kw):
        orig_init(self, *args, **kw)
        try:
            sp = getattr(self, "speculator", None)
            if isinstance(sp, DFlash2Speculator):
                _hook(sp, self)
        except Exception as e:
            _log(f"not hooked: {e!r}")

    __init__._radiance_lookup = True
    GPUModelRunner.__init__ = __init__
    _log(f"installed (V2 runner, DFlash2 speculator; enter {ENTER} stay {STAY} hot {HOT})")
    return True
