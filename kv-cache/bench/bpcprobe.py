#!/usr/bin/env python3
"""blocks_per_chunk probe -- R3.12.5.

Answers one question that could not be settled by reading the source: when the offload
connector is configured with blocks_per_chunk = N, does a Mamba/GDN group's per-chunk
payload stay ONE fixed-size recurrent state (an N-fold cut in write volume, for free, from
configuration alone) or become N copies of it (a bundling change only, no saving)?

Method: point the fs tier at an empty root, push one long prefix, let the cascade settle,
then count the per-group chunk directories and stat one file from each group. The layout is
    <root>/<model>_r<rank>/<hash[0:3]>/b<block>_g<group>/<hash>.bin
so both the count per group and the bytes per group file are directly observable.

Read it as:
    g0 file == g6 file  -> uniform slot size; Mamba is bundled, not deduplicated. Patch needed.
    g0 file <  g6 file  -> Mamba stores one state per chunk. Config-only win; go to N=8.
    g0 count < g6 count -> the store path is already skipping Mamba chunks.

The engine must already be running with the intended blocks_per_chunk; this script does not
restart anything. Usage:  python3 bpcprobe.py --root /kvcache/blocks-bpc2 [--tokens 40000]
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tierbench as T


def tree_stats(root):
    """Per-group chunk-dir count and the set of file sizes seen, walked directly.

    The tree is small by construction (one prefix, an empty root), so unlike the production
    tier this walk is cheap -- fs_block_count's df trick would be wrong here anyway, since
    it assumes a fixed 27,000,832 B block and that assumption is exactly what is under test.
    """
    groups = {}
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if "_g" not in base:
            continue
        try:
            g = int(base.rsplit("_g", 1)[1])
        except ValueError:
            continue
        rec = groups.setdefault(g, {"chunks": 0, "sizes": {}, "bytes": 0})
        rec["chunks"] += 1
        for fn in filenames:
            if not fn.endswith(".bin"):
                continue
            sz = os.path.getsize(os.path.join(dirpath, fn))
            rec["sizes"][sz] = rec["sizes"].get(sz, 0) + 1
            rec["bytes"] += sz
    return groups


def settle(root, quiet_polls=3, poll_s=10.0, timeout_s=600):
    """Wait until the file count under root stops moving."""
    t0, last, quiet = time.time(), -1, 0
    while time.time() - t0 < timeout_s:
        n = sum(g["chunks"] for g in tree_stats(root).values())
        if n == last:
            quiet += 1
            if quiet >= quiet_polls:
                T.log("  settled at %d chunk dirs after %.0fs" % (n, time.time() - t0))
                return n
        else:
            quiet = 0
            T.log("  %d chunk dirs (+%d)" % (n, n - max(last, 0)))
        last = n
        time.sleep(poll_s)
    T.log("  TIMEOUT after %.0fs at %d chunk dirs" % (time.time() - t0, last))
    return last


def has_both(groups):
    """True once at least one Mamba group and one attention group have landed a file."""
    m = any(groups.get(g, {}).get("bytes", 0) > 0 for g in range(6))
    a = any(groups.get(g, {}).get("bytes", 0) > 0 for g in (6, 7))
    return m and a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="fs tier root the engine is writing to")
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--evict-tokens", type=int, default=120000,
                    help="novel tokens pushed per eviction round")
    ap.add_argument("--rounds", type=int, default=4,
                    help="max eviction rounds before giving up on reaching the fs tier")
    ap.add_argument("--salt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit("root %s does not exist" % args.root)

    base = T.detect_endpoint()
    T.log("endpoint %s" % base)
    salt = args.salt or ("bpcprobe-%d" % int(time.time()))

    pre = tree_stats(args.root)
    T.log("root %s holds %d chunk dirs before the probe"
          % (args.root, sum(g["chunks"] for g in pre.values())))

    sizer = T.Sizer(base)
    text, got = sizer.build(salt, "probe", args.tokens)
    T.log("prompt %d tokens (asked %d)" % (got, args.tokens))

    before = T.snapshot(base)
    pushed = 0
    t0 = time.time()
    r = T.chat(base, text, "probe")
    pushed += got
    T.log("probe request done in %.2fs -> %s" % (time.time() - t0, r))

    # Nothing reaches the fs tier until the CPU primary is full and starts cascading, and
    # the cascade only runs while the engine is idle (R3.8). So: push novel context in
    # rounds, settling after each, and stop the moment both a Mamba group and an attention
    # group have actually written a file -- that is all the layout question needs.
    groups = tree_stats(args.root)
    for i in range(args.rounds):
        if has_both(groups):
            break
        T.log("round %d: pushing ~%d novel tokens to force the CPU->fs cascade"
              % (i + 1, args.evict_tokens))
        T.evict(base, sizer, salt, args.evict_tokens, "evict%d" % i)
        pushed += args.evict_tokens
        T.log("round %d: waiting for the cascade to settle" % (i + 1))
        settle(args.root)
        groups = tree_stats(args.root)
        T.log("round %d: %d chunk dirs on the probe tier"
              % (i + 1, sum(g["chunks"] for g in groups.values())))

    d = T.delta(before, T.snapshot(base))
    groups = tree_stats(args.root)

    T.log("")
    T.log("group  chunks  file sizes (bytes x count)")
    for g in sorted(groups):
        rec = groups[g]
        sizes = ", ".join("%d x%d" % (s, c) for s, c in sorted(rec["sizes"].items()))
        T.log("  g%-3d %6d  %s" % (g, rec["chunks"], sizes))

    attn = [groups[g]["sizes"] for g in (6, 7) if g in groups and groups[g]["sizes"]]
    mamba = [groups[g]["sizes"] for g in range(6) if g in groups and groups[g]["sizes"]]
    verdict = "INDETERMINATE (no files from one side; try more --rounds)"
    if attn and mamba:
        a_sz = max(max(s) for s in attn)
        m_sz = max(max(s) for s in mamba)
        m_ct = max(groups[g]["chunks"] for g in range(6) if g in groups)
        a_ct = max(groups[g]["chunks"] for g in (6, 7) if g in groups)
        if m_sz < a_sz or m_ct < a_ct:
            verdict = ("CONFIG-ONLY WIN: mamba %d B x%d vs attention %d B x%d"
                       % (m_sz, m_ct, a_sz, a_ct))
        else:
            verdict = ("BUNDLING ONLY: mamba %d B x%d == attention %d B x%d; "
                       "no volume saved, the store-policy patch is required"
                       % (m_sz, m_ct, a_sz, a_ct))
    T.log("")
    T.log("VERDICT: %s" % verdict)

    # The decisive scalar. At blocks_per_chunk=1 the production tier stores one
    # 27,000,832 B file per group per 1,648-token chunk = 9 * 27000832 / 1648 =
    # 147,459 B/token. Bundling alone leaves that number untouched; a real per-group
    # saving moves it down.
    total = sum(g["bytes"] for g in groups.values())
    ref = 9 * 27000832 / 1648.0
    bpt = float(total) / max(pushed, 1)
    T.log("stored %.2f GB for ~%d tokens pushed = %.0f B/token "
          "(blocks_per_chunk=1 reference: %.0f B/token, %.1f%% of it)"
          % (total / 1e9, pushed, bpt, ref, 100.0 * bpt / ref))

    out = args.out or os.path.join(T.OUT_DIR, "BPCPROBE-%s.json" % salt)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"salt": salt, "root": args.root, "prompt_tokens": got,
                   "tokens_pushed": pushed,
                   "groups": {str(k): v for k, v in groups.items()},
                   "verdict": verdict, "total_bytes": total,
                   "bytes_per_token": bpt, "bpc1_reference_bytes_per_token": ref,
                   "metrics_delta": d}, f, indent=2, default=str)
    T.log("wrote %s" % out)


if __name__ == "__main__":
    main()
