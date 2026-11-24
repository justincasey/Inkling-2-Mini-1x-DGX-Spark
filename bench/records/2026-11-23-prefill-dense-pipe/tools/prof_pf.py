#!/usr/bin/env python3
"""prof_pf.py DB.sqlite CHUNK[,CHUNK...] [TOP]: the kernels of chosen prefill chunks of an `nsys profile -t cuda` run of
`inkling2_loop_prefill_perf::prefill_curve` (chunk i = the i-th run of the embedding kernel, in time order; the warm-up chunk is chunk 0
of the profile, so the timed chunk of position 4096 x j is profile chunk j + 1). Per chunk: the span, the busy time, the idle gaps
(span - union of kernel intervals), and a table per kernel name (launches, total ms, mean us, share of busy time)."""
import sqlite3, sys, collections

db, chunks = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
c = sqlite3.connect(db)
rows = c.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k "
                 "join StringIds s on s.id=k.demangledName order by k.start").fetchall()
marks = [i for i, r in enumerate(rows) if "embed" in r[2].split("(")[0]]
print("kernels", len(rows), "chunks", len(marks))


def short(n):
    n = n.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    return n[:110]


CATS = [("moe: gate/up GEMM", "gateup_kernel"), ("moe: down GEMM", "down_kernel"), ("moe: rot_in", "rot_in_kernel"), ("moe: combine", "combine_f32"),
        ("moe: plan/remap", "plan_"), ("moe: plan/remap", "own_remap"), ("attn: core global", "attn_kernel<(int)8, (int)3, (bool)1"),
        ("attn: core local", "attn_kernel<(int)8, (int)3, (bool)0"), ("attn: prologue", "prologue_kernel"), ("attn: bias tables", "bias_kernel"),
        ("attn: merge", "merge_kernel"), ("attn: ring/commit", "ring_write"), ("attn: ring/commit", "conv_commit"),
        ("dense EXL3 GEMM", "big_kernel"), ("dense EXL3 GEMM", "::gemm_kernel"), ("dense EXL3 input rotation", "rot_in_frag"), ("dense EXL3 input rotation", "::rot_kernel"), ("bf16 linear (taps, sink, controllers)", "lin_wide2"),
        ("controller glue", "blend_kernel"), ("controller glue", "pair_kernel"), ("AttnRes mix", "mix_raw"), ("AttnRes mix", "mix_kernel"),
        ("short conv", "conv_kernel"), ("router", "router"), ("router", "route"), ("norm/embed", "rmsnorm"), ("norm/embed", "embed")]


def category(name):
    for c, sub in CATS:
        if sub in name:
            return c
    return "other: " + name.split("(")[0][:60]


for ch in chunks:
    a = marks[ch]
    b = marks[ch + 1] if ch + 1 < len(marks) else len(rows)
    sel = rows[a:b]
    t0, t1 = sel[0][0], max(r[1] for r in sel)
    busy = sum(r[1] - r[0] for r in sel)
    union, last = 0, t0
    for r in sorted(sel, key=lambda r: r[0]):
        s, e = max(r[0], last), r[1]
        if e > s:
            union += e - s
            last = e
    print(f"\n== chunk {ch}: kernels {len(sel)}, span {(t1 - t0) / 1e6:.2f} ms, union-busy {union / 1e6:.2f} ms, idle gaps {(t1 - t0 - union) / 1e6:.2f} ms, sum of kernel times {busy / 1e6:.2f} ms")
    agg = collections.defaultdict(lambda: [0, 0, 0])
    cat = collections.defaultdict(lambda: [0, 0])
    for r in sel:
        k = short(r[2]).split("(const")[0].split("(Args")[0].split("(Mix")[0].split("(Conv")[0].rstrip()
        a_ = agg[k]
        a_[0] += 1
        a_[1] += r[1] - r[0]
        cc = category(r[2])
        cat[cc][0] += 1
        cat[cc][1] += r[1] - r[0]
    tot = sum(v[1] for v in agg.values())
    print("-- by category: launches | total ms | share")
    for k, (cnt, d) in sorted(cat.items(), key=lambda x: -x[1][1]):
        print(f"{cnt:6d} {d / 1e6:9.3f} {100 * d / tot:5.1f}%  {k}")
    print("-- by kernel: launches | total ms | mean us | share")
    for k, (cnt, d, _) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
        print(f"{cnt:6d} {d / 1e6:9.3f} {d / 1e3 / cnt:9.1f} {100 * d / tot:5.1f}%  {k}")
