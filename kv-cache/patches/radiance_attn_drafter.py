# House copy 2026-10-09 of SlyBase sly/radiance_attn_drafter.py (github.com/brianfu9/vllm-sly-radiance @7d2304a),
# unmodified below this header. Installed by patch_attn_drafter.py (adopted 2026-10-09).
"""Split-KV launch for the DFlash2 drafter's sliding-window attention (vLLM's TRITON_ATTN), gfx12x (RDNA4).

The drafter (5 layers, every layer sliding-window 2048, 32 q / 8 kv heads, head 128) calls vLLM's
`unified_attention` once per engine step per layer with 8 queries per sequence over a window of the last ~2k
keys. On this card that call costs ~200 us in situ (rocprofv3, one stream: 31.6 us at 0.1k context, flat from 32k
on), i.e. ~20-40 GB/s for ~4 MB of K/V, because:

  * `unified_attention` launches BLOCK_M 16 = BLOCK_Q 4 for GQA 4 and closes its 3D (split-KV) path for
    max_seqlen_q > 1, so the grid is (8 // 4 + 1) q-blocks x 8 kv heads = 24 workgroups on 32 CUs, each one
    walking the whole window sequentially;
  * the 3D kernel, if it were opened, partitions [0, seq_len) into segments, not the window, so at 99k context
    the window would sit in one of S segments.

This module rebinds `unified_attention` in `vllm.v1.attention.backends.triton_attn` and, for the drafter's call
(and nothing else), (1) slices the block table and seq_len down to the window in one tiny kernel -- whole leading
blocks are dropped, so every position the kernel uses stays relative to the sequence's end and RoPE is already in
the cached K -- and (2) launches the *same* vLLM kernels in 3D mode over that slice, BLOCK_M 32 (= BLOCK_Q 8 =
exactly the 8 queries x 4 heads of a k=7 verify), TILE 32, 4 warps, 1 stage, S segments by sequence count, then
vLLM's own `reduce_segments`. Numerics are the stock kernel's (rel. error against an fp32 reference 0.0021 vs
0.0023); only the launch changed.

The drafter's call is NON-causal (dflash sets causal=False: a key is allowed when it is < seq_len and within
`window` of the query on either side), so the launch passes USE_CAUSAL through as it came.

Measured with sly/bench_drafter_attn.py (per call, one layer, R9700, window 2048, non-causal; a step makes 5):

    sequences   1          2          4          8
    stock       133 us     205 us     196 us     344 us
    split        38 us      70 us     105 us     188 us

A step is 0.5 ms (1 sequence) to 0.8 ms (8) shorter; end to end on the live config (single stream, 1k..99k of
context) the step is 0.2-0.9 ms shorter (~1-2 %) and the accepted length is unchanged (the numerics are the
stock kernel's).

A wider window (`dflash_config.swa_window_size` 8192) costs only ~0.4 ms per step more with this launch, but the
drafter was trained with 2048: at 8192 the accepted tokens/step fall by 25-43 % for generation and summarising
(copy is mixed, +6 % at 33k, -19 % at 99k), so the checkpoint's window stays.

Untouched, deliberately: every call that is not this shape -- other head sizes / GQA ratios, bf16 KV, a
per-sequence causal tensor, alibi / sinks / softcap / mm-prefix / rswa / chunked, queries per sequence outside
2..16 (prefill chunks), no sliding window. Any exception in the tuned path falls back to vLLM's own call for the
rest of the process.

Knobs (read at import):
  RADIANCE_ATTN_DRAFTER_TUNE      1 (default) | 0 = vLLM's launch (A/B control; baked in at graph capture)
  RADIANCE_ATTN_DRAFTER_SEGMENTS  auto (default) | N = force the split count (a power of two)
"""
import os
import sys

try:
    import triton
    import triton.language as tl
except Exception:                       # no triton (offline check): the tuned path is then never selected
    triton = tl = None

ENABLED = os.environ.get("RADIANCE_ATTN_DRAFTER_TUNE", "1") != "0"
_SEG_ENV = os.environ.get("RADIANCE_ATTN_DRAFTER_SEGMENTS", "auto")

HEAD, NQPKV = 128, 4               # the only shape these numbers were measured at
MAX_Q = 16                         # queries per sequence: a verify block, not a prefill chunk
BLOCK_M, TILE, WARPS, STAGES = 32, 32, 4, 1
# Split count by sequence count: one workgroup per (q-block, kv head, segment); with a single sequence there are
# only 16 (q-block, kv head) pairs, so it wants the most splits, eight sequences already fill the machine.
SEGMENTS_BY_SEQS = {1: 64, 2: 32, 3: 32}
DEFAULT_SEGMENTS = 16

_state = {"orig": None, "wrapper": None, "logged": False, "failed": False, "declined": False,
          "calls": 0}                      # calls: tuned launches


def segments_for(nseq):
    if _SEG_ENV != "auto":
        return int(_SEG_ENV)
    return SEGMENTS_BY_SEQS.get(int(nseq), DEFAULT_SEGMENTS)


def _next_pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


if triton is not None:
    @triton.jit
    def _slice_window(bt_ptr, seq_ptr, bt_out_ptr, seq_out_ptr, bt_stride, nb_max,
                      KEEP: tl.constexpr, BLOCK: tl.constexpr, NB_WIN: tl.constexpr, NB_PAD: tl.constexpr):
        """One program per sequence: first = whole blocks before the window; copy NB_WIN block-table entries
        from there and shorten seq_len by what was dropped."""
        s = tl.program_id(0)
        length = tl.load(seq_ptr + s)
        first = tl.maximum(length - KEEP, 0) // BLOCK
        cols = tl.arange(0, NB_PAD)
        src = tl.minimum(first + cols, nb_max - 1)
        val = tl.load(bt_ptr + s * bt_stride + src, mask=cols < NB_WIN, other=0)
        tl.store(bt_out_ptr + s * NB_WIN + cols, val, mask=cols < NB_WIN)
        tl.store(seq_out_ptr + s, length - first * BLOCK)
else:
    _slice_window = None


# arguments of vLLM's kernels that _run passes by name; install() refuses to hook a vLLM that lacks any of them
_KERNEL_ARGS = (
    "output_ptr", "segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr", "query_ptr", "key_cache_ptr",
    "value_cache_ptr", "sink_ptr", "block_tables_ptr", "seq_lens_ptr", "alibi_slopes_ptr", "qq_bias_ptr",
    "k_scale_cache_ptr", "v_scale_cache_ptr", "scale", "q_scale", "k_scale", "v_scale", "out_scale", "softcap",
    "num_query_heads", "num_queries_per_kv", "block_table_stride", "query_stride_0", "query_stride_1",
    "output_stride_0", "output_stride_1", "qq_bias_stride_0", "BLOCK_SIZE", "TILE_SIZE", "HEAD_SIZE",
    "HEAD_SIZE_PADDED", "USE_ALIBI_SLOPES", "USE_ALIBI_SQRT", "USE_QQ_BIAS", "USE_SOFTCAP", "USE_SINKS",
    "SLIDING_WINDOW", "USE_CAUSAL", "USE_PER_SEQ_CAUSAL", "per_seq_causal_ptr", "USE_MM_PREFIX", "MAX_MM_RANGES",
    "mm_prefix_range_ptr", "rswa_prefix_lens_ptr", "R_SWA_WINDOW", "USE_R_SWA", "stride_k_cache_0",
    "stride_k_cache_1", "stride_k_cache_2", "stride_k_cache_3", "stride_v_cache_0", "stride_v_cache_1",
    "stride_v_cache_2", "stride_v_cache_3", "stride_ks_blk", "stride_ks_slot", "stride_ks_head", "stride_vs_blk",
    "stride_vs_slot", "stride_vs_head", "query_start_len_ptr", "BLOCK_Q", "num_seqs", "BLOCK_M",
    "NUM_SEGMENTS_PER_SEQ", "USE_FP8", "IS_3D", "KV_QUANT_MODE", "Q_IS_FP8", "CHUNK_LOOKBACK", "CHUNK_SIZE",
    "USE_TD", "USE_TD_QO", "MM_PREFIX_CLAMP_SW")
_REDUCE_ARGS = (
    "output_ptr", "segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr", "seq_lens_ptr", "num_seqs",
    "num_query_heads", "out_scale_inv", "output_stride_0", "output_stride_1", "block_table_stride", "TILE_SIZE",
    "HEAD_SIZE", "HEAD_SIZE_PADDED", "query_start_len_ptr", "BLOCK_Q", "NUM_SEGMENTS_PER_SEQ", "USE_FP8")

# features vLLM's wrapper supports and this launch does not: any of them present -> vLLM's own launch
_MUST_BE_NONE = ("alibi_slopes", "sinks", "mm_prefix_range", "rswa_prefix_lens", "rswa_window", "output_scale",
                 "qq_bias", "k_scale_cache", "v_scale_cache", "q_descale")


def _gate(kw):
    """((window, max_q, nseq, causal), None) when this call is the drafter's, else (None, why). Keyword calls only
    (the backend's form)."""
    try:
        q, k, v = kw["q"], kw["k"], kw["v"]
        if kw.get("kv_quant_mode") != _state["fp8_per_tensor"]:
            return None, f"kv_quant_mode={kw.get('kv_quant_mode')}"
        if q.dtype != _state["bf16"] or k.dtype != _state["fp8"] or v.dtype != _state["fp8"]:
            return None, f"dtypes q={q.dtype} k={k.dtype} v={v.dtype}"
        if q.dim() != 3 or q.shape[2] != HEAD or k.shape[3] != HEAD or q.stride(2) != 1:
            return None, f"shapes q={tuple(q.shape)} k={tuple(k.shape)} q_stride={tuple(q.stride(i) for i in range(q.dim()))}"
        kvh = k.shape[2]
        if kvh <= 0 or q.shape[1] % kvh or q.shape[1] // kvh != NQPKV:
            return None, f"heads q={q.shape[1]} kv={kvh}"
        causal = kw.get("causal")
        if not isinstance(causal, bool):                  # a per-sequence tensor is not this launch
            return None, f"causal={causal!r}"
        odd = [n for n in _MUST_BE_NONE if kw.get(n) is not None]
        if odd:
            return None, f"set: {odd}"
        for n, ok in (("softcap", not kw.get("softcap")), ("use_td", not kw.get("use_td")),
                      ("use_alibi_sqrt", not kw.get("use_alibi_sqrt")), ("chunk_lookback", kw.get("chunk_lookback", -1) == -1)):
            if not ok:
                return None, f"{n}={kw.get(n)!r}"
        left, right = kw["window_size"]
        if left < 0 or right != 0:
            return None, f"window_size={kw['window_size']!r}"
        max_q = int(kw["max_seqlen_q"])
        if not 2 <= max_q <= MAX_Q:
            return None, f"max_seqlen_q={max_q}"
        return (int(left) + 1, max_q, len(kw["seqused_k"]), causal), None
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def _params(kw):
    return _gate(kw)[0]


def _run(kw, window, max_q, nseq, causal):
    torch, TU = _state["torch"], _state["TU"]
    q, k, v, out = kw["q"], kw["k"], kw["v"], kw["out"]
    cu, seq, bt = kw["cu_seqlens_q"], kw["seqused_k"], kw["block_table"]
    block = v.shape[1]
    nq = q.shape[1]
    keep = window + max_q - 1                                   # furthest back any query of the batch looks
    nb_win = (keep + block - 2) // block + 1                    # table entries a window (+ its queries) can span
    bt2 = torch.empty((nseq, nb_win), dtype=bt.dtype, device=bt.device)
    seq2 = torch.empty_like(seq)
    _slice_window[(nseq,)](bt, seq, bt2, seq2, bt.stride(0), bt.shape[1], KEEP=keep, BLOCK=block,
                           NB_WIN=nb_win, NB_PAD=_next_pow2(nb_win), num_warps=1)
    segs = segments_for(nseq)
    bq = BLOCK_M // NQPKV
    grid = (q.shape[0] // bq + nseq, k.shape[2], segs)
    so = torch.empty(q.shape[0], nq, segs, HEAD, dtype=torch.float32, device=q.device)
    sm = torch.empty(q.shape[0], nq, segs, dtype=torch.float32, device=q.device)
    se = torch.empty(q.shape[0], nq, segs, dtype=torch.float32, device=q.device)
    TU.kernel_unified_attention[grid](
        output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se, query_ptr=q, key_cache_ptr=k,
        value_cache_ptr=v, sink_ptr=None, block_tables_ptr=bt2, seq_lens_ptr=seq2, alibi_slopes_ptr=None,
        qq_bias_ptr=None, k_scale_cache_ptr=None, v_scale_cache_ptr=None, scale=kw["softmax_scale"],
        q_scale=None, k_scale=kw["k_descale"], v_scale=kw["v_descale"], out_scale=1.0, softcap=0,
        num_query_heads=nq, num_queries_per_kv=NQPKV, block_table_stride=bt2.stride(0),
        query_stride_0=q.stride(0), query_stride_1=q.stride(1), output_stride_0=out.stride(0),
        output_stride_1=out.stride(1), qq_bias_stride_0=0, BLOCK_SIZE=block, TILE_SIZE=TILE, HEAD_SIZE=HEAD,
        HEAD_SIZE_PADDED=HEAD, USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_QQ_BIAS=False,
        USE_SOFTCAP=False, USE_SINKS=False, SLIDING_WINDOW=window, USE_CAUSAL=causal, USE_PER_SEQ_CAUSAL=False,
        per_seq_causal_ptr=None, USE_MM_PREFIX=False, MAX_MM_RANGES=0, mm_prefix_range_ptr=None,
        rswa_prefix_lens_ptr=seq2, R_SWA_WINDOW=0, USE_R_SWA=False, stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3), stride_ks_blk=None, stride_ks_slot=None, stride_ks_head=None,
        stride_vs_blk=None, stride_vs_slot=None, stride_vs_head=None, query_start_len_ptr=cu, BLOCK_Q=bq,
        num_seqs=nseq, BLOCK_M=BLOCK_M, NUM_SEGMENTS_PER_SEQ=segs, USE_FP8=False, IS_3D=True,
        KV_QUANT_MODE=_state["fp8_per_tensor"], Q_IS_FP8=False, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1, USE_TD=False,
        USE_TD_QO=False, MM_PREFIX_CLAMP_SW=False, num_warps=WARPS, num_stages=STAGES)
    TU.reduce_segments[(q.shape[0], nq)](
        output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se, seq_lens_ptr=seq2,
        num_seqs=nseq, num_query_heads=nq, out_scale_inv=1.0, output_stride_0=out.stride(0),
        output_stride_1=out.stride(1), block_table_stride=bt2.stride(0), TILE_SIZE=TILE, HEAD_SIZE=HEAD,
        HEAD_SIZE_PADDED=HEAD, query_start_len_ptr=cu, BLOCK_Q=bq, NUM_SEGMENTS_PER_SEQ=segs, USE_FP8=False)
    if not _state["logged"]:
        _state["logged"] = True
        sys.stderr.write(f"[radiance] drafter attn plan: window={window} q_len<={max_q} split-KV BLOCK_M={BLOCK_M} "
                         f"TILE={TILE} warps={WARPS} splits={segs} (seqs={nseq} tokens={q.shape[0]})\n")
        sys.stderr.flush()


def wrapper(*args, **kw):
    orig = _state["orig"]
    if not ENABLED or args or _state["failed"]:
        return orig(*args, **kw)
    p, why = _gate(kw)
    if p is None:
        if not _state["declined"]:              # once: what the first head-128 call that was not taken looked like
            try:
                if kw["q"].shape[2] == HEAD:
                    _state["declined"] = True
                    sys.stderr.write(f"[radiance] drafter attn tune declined a head-{HEAD} call: {why}\n")
                    sys.stderr.flush()
            except Exception:
                pass
        return orig(*args, **kw)
    try:
        _run(kw, *p)
        _state["calls"] += 1
        return None
    except Exception as e:      # a tune must never take a serve down: vLLM's own launch for the rest of the process
        _state["failed"] = True
        sys.stderr.write(f"[radiance] drafter attn tune fell back to vLLM's launch: {e!r}\n")
        return orig(*args, **kw)


def install():
    """Rebind unified_attention in the Triton backend. True when installed or deliberately off (logged), False
    when this vLLM / platform is not the one the launch was written for."""
    try:
        import torch
        import vllm.v1.attention.backends.triton_attn as TA
        import vllm.v1.attention.ops.triton_unified_attention as TU
        from vllm.platforms import current_platform
        from vllm.v1.kv_cache_interface import KVQuantMode
    except Exception:
        return False
    if getattr(TA.unified_attention, "_radiance_drafter_tune", False):
        return True
    if not ENABLED:
        sys.stderr.write("[radiance] drafter attn tune off (RADIANCE_ATTN_DRAFTER_TUNE=0): vLLM's launch\n")
        return True
    try:
        from vllm.platforms.rocm import on_gfx12x
        if not on_gfx12x():
            sys.stderr.write("[radiance] drafter attn tune skipped: not gfx12x\n")
            return True
    except Exception:
        return False
    if _slice_window is None:
        return False
    for kernel, names in ((TU.kernel_unified_attention, _KERNEL_ARGS), (TU.reduce_segments, _REDUCE_ARGS)):
        missing = set(names) - set(getattr(kernel, "arg_names", ()))
        if missing:
            sys.stderr.write(f"[radiance] drafter attn tune skipped: vLLM's kernel lacks {sorted(missing)}\n")
            return True
    _state.update(orig=TA.unified_attention, torch=torch, TU=TU, bf16=torch.bfloat16,
                  fp8=current_platform.fp8_dtype(), fp8_per_tensor=KVQuantMode.FP8_PER_TENSOR)
    wrapper._radiance_drafter_tune = True
    _state["wrapper"] = wrapper
    TA.unified_attention = wrapper
    sys.stderr.write(f"[radiance] drafter attn tune installed (window-sliced split-KV for the DFlash drafter, "
                     f"segments={_SEG_ENV})\n")
    sys.stderr.flush()
    return True
