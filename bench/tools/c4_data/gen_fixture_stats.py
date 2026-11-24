#!/usr/bin/env python3
"""Regenerates fixture_stats.json from planes.bin (see copy_planes_remote.py) and intake/census.json (stdlib only).
    python3 bench/tools/c4_data/gen_fixture_stats.py planes.bin > bench/tools/c4_data/fixture_stats.json
Thresholds are relative to the row's top router weight (router_w sums to 8 per row and pass); bytes are the census bytes of the experts picked.
Window numbers: 120 windows of w consecutive rows from random starts (seed 1) of the raw-long-window fixture (a counting text: indicative only)."""
import array, collections, json, os, random, re, struct, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
F = {}
data = open(sys.argv[1], "rb").read()
o = 0
while o < len(data):
    hl, dl = struct.unpack_from("<II", data, o); o += 8
    h = json.loads(data[o:o + hl]); o += hl
    F.setdefault(h["file"].replace(".safetensors", ""), {})[h["name"]] = (h, data[o:o + dl]); o += dl


def planes(fn):
    d = F[fn]
    P, N, K = d["router_idx"][0]["shape"]
    idx = array.array("i"); idx.frombytes(d["router_idx"][1])
    w = array.array("f"); w.frombytes(d["router_w"][1])
    return P, N, K, idx, w


mods = json.load(open(os.path.join(ROOT, "intake", "census.json")))["modules"]
eb = collections.defaultdict(int)
for n, v in mods.items():
    r = re.match(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$", n)
    if r:
        eb[(int(r[1]), int(r[2]))] += v["bytes"]
layer_of = lambda p: p if p < 6 else (6 + (p - 6) % 12 if p < 42 else 18 + (p - 42))
ALL = {fn: planes(fn) for fn in sorted(F)}
rank, cnt = [0.0] * 8, 0
for P, N, K, idx, w in ALL.values():
    for p in range(P):
        for n in range(N):
            ws = sorted(w[(p * N + n) * K:(p * N + n) * K + K], reverse=True)
            for i in range(8):
                rank[i] += ws[i] / 8
            cnt += 1


def row_skip(thr):
    tot = kept = picks = drop = rp = 0
    mass = 0.0
    for P, N, K, idx, w in ALL.values():
        for p in range(P):
            L = layer_of(p)
            for n in range(N):
                base = (p * N + n) * K; ws = w[base:base + K]; mx = max(ws); rp += 1
                for k in range(K):
                    b = eb[(L, idx[base + k])]; tot += b; picks += 1
                    if ws[k] < thr * mx:
                        drop += 1; mass += ws[k] / 8
                    else:
                        kept += b
    return dict(rel_threshold=thr, picks_dropped=round(drop / picks, 4), mass_dropped=round(mass / rp, 4), bytes_saved_one_row=round(1 - kept / tot, 4))


def win(thr, wsize, S, nwin=120, seed=1):
    random.seed(seed)
    P, N, K, idx, w = ALL["raw-long-window"]
    tb = kb = 0
    for _ in range(nwin):
        starts = [random.randrange(150, N - wsize) for _ in range(S)]
        for p in range(P):
            L = layer_of(p); allx, keep = set(), set()
            for st in starts:
                for n in range(st, st + wsize):
                    base = (p * N + n) * K; ws = w[base:base + K]; mx = max(ws)
                    for k in range(K):
                        e = idx[base + k]; allx.add(e)
                        if ws[k] >= thr * mx:
                            keep.add(e)
            tb += sum(eb[(L, e)] for e in allx); kb += sum(eb[(L, e)] for e in keep)
    return round(1 - kb / tb, 4)


random.seed(3)
S_ = {fn: [[frozenset(idx[(p * N + n) * K:(p * N + n) * K + K]) for n in range(N)] for p in range(P)] for fn, (P, N, K, idx, w) in ALL.items()}
ov = lambda fa, na, fb, nb: sum(len(S_[fa][p][na] & S_[fb][p][nb]) for p in range(48)) / (48 * 8)
pairs = {}
for fa, fb in (("raw-long-window", "chat-capital-effort0"), ("raw-long-window", "chat-sky-effort0.9"), ("chat-sky-effort0.9", "raw-counting"), ("chat-sky-effort0.9", "chat-capital-effort0")):
    Na, Nb = ALL[fa][1], ALL[fb][1]
    lo = lambda N: 150 if N > 200 else 8
    x = [ov(fa, random.randrange(lo(Na), Na), fb, random.randrange(lo(Nb), Nb)) for _ in range(300)]
    pairs[f"{fa}|{fb}"] = round(sum(x) / len(x), 4)
out = {
    "_provenance": "Derived from router_idx/router_w planes of the 4 spec fixtures (oracle fp64 runs, pack 44c78c9d, real routing of the oracle on prompts: chat-capital-effort0 40 rows, chat-sky-effort0.9 49, raw-counting 50, raw-long-window 1126) read read-only from node-1 /home/spark/inkling-2-mini-1x-dgx-spark/fixtures/spec/ by header-range copy (copy_planes_remote.py); rows are prompt + a few generated rows, NOT served replies; raw-long-window is a counting text (very repetitive: its window unions are lower than real replies). Expert bytes from intake/census.json. Regenerate: gen_fixture_stats.py.",
    "rows_total": sum(v[1] for v in ALL.values()),
    "rank_mass_share": [round(x / cnt, 4) for x in rank],
    "row_skip_pooled": [row_skip(t) for t in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5)],
    "window_bytes_saved_long_window_indicative": {str(t): {"1x1": win(t, 1, 1), "1x8": win(t, 8, 1), "4x8": win(t, 8, 4)} for t in (0.15, 0.2, 0.3, 0.5)},
    "cross_text_pair_overlap_fraction (random 0.0625)": pairs,
    "cross_text_overlap_multiplier_vs_random_mean_of_pairs": round(sum(pairs.values()) / len(pairs) / 0.0625, 2),
    "cross_text_note": "pairs are prompt rows (chat-template and counting text), which share template routes: an upper bound on the overlap of two served streams; the independent closed form is the optimistic end",
}
print(json.dumps(out, indent=1))
