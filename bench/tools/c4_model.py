#!/usr/bin/env python3
"""Byte-and-latency model of one decode window (S streams x R rows) of the Mini on one Spark, and the c=4 plan's tables (stdlib only,
deterministic, offline: no node, no engine).

    python3 bench/tools/c4_model.py              the tables of docs/C4-DECODE-PLAN.md
    python3 bench/tools/c4_model.py --validate   calibration + residuals against the measured points; exit 1 if a served point is
                                                 off by more than 10 % (--strict: the lab-synthetic points too, which fail by design)

The model. Time of a window = O + bytes / (BW x eff) + 0.034 ms x rows, bytes = dense visits + unembed + the KV of each stream
+ the expert bytes of the UNION of the experts the rows pick + the partial-sum traffic of the picks. O (ms, the part of a window no byte
stream hides: glue chains, launches, syncs) and eff (the fraction of the 233.8 GB/s read bandwidth the window sustains) are the ONLY
free parameters; both are fitted on three served serial steps (below) and nothing else is fitted. Everything else is read from a file:

  intake/census.json                            bytes per tensor: dense bytes per visit, the unembed, bytes per expert by band
  bench/records/2026-11-23-spec-speed/raw/routelog.txt   real routes of four 640-token replies: distinct experts a visit in w
                                                consecutive tokens (w = 1..8), the byte-weighted ratio, the 5-bit pick share per class
  bench/records/2026-11-23-spec-speed/raw/fixed7/specbench.json   rounds by accepted-draft count at depth 7 per kind (accept_hist)
  bench/records/2026-11-23-spec-final/raw/A-specbench.json        the c=4 drafted run: per-stream speeds and counters (the tail)
  bench/records/2026-11-23-spec-speed/raw/timing.txt              the rank's lab grid (synthetic ids) and the proposal times
  bench/records/2026-11-23-review-fixes-b/raw/live/bench.out      bench.py decode_c1/c4/c8 with the shipping drafter
  bench/records/2026-11-23-prefill-exact/raw/new/bench-ref.json   the 512-token greedy replies of bench.py's four prompts (copy drafter)
  bench/records/2026-11-23-spec-exact/raw/exactness-serial.json   128-token replies with their prompts (copy drafter, longer prompts)
  bench/tools/c4_data/bench_prompt_ids.json     prompt ids of bench.py's four prompts (HF tokenizer of the base model)
  bench/tools/c4_data/fixture_stats.json        router_w / router_idx statistics of the real-pack fixtures (B-lane skipping)

Every constant below carries its source; DERIVED = a formula on sourced numbers, [INFERENCE] = not measured.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REC = os.path.join(ROOT, "bench", "records")
DATA = os.path.join(ROOT, "bench", "tools", "c4_data")

BW_GBS = 233.8  # sustained read, docs/ROOFLINE.md (233-237 hard ceiling)
SLOTS = 4  # bootstrap profile slots: c=8 is two waves of four, not an 8-stream batch (bench.py phase_conc, graph-baseline raw walls)
VISITS = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))  # docs/ROOFLINE.md: 48 passes
ROW_MS = 0.034  # DERIVED: per-row kernels: split top-k 258 us at 8 rows / 8 (fuse-baseline kernels-rows8) + logit epilogue ~2 us
Z_BYTES_PER_PICK = (27648 + 9216) * 2  # [INFERENCE] moe_experts.cu Z partials per pick: gate/up 27,648 B + down 9,216 B, written once and read once
RING_TOKENS, RING_B = 1031, 4096  # docs/ROOFLINE.md: 1,031-position bf16 ring, 4,096 B/token/pass (5 pass-types: 40 sliding visits)
GLOBAL_B = 2064  # 16,512 B/token int8 over the 8 global visits (docs/ROOFLINE.md)
CTX = 450  # DERIVED: bench prompts 79-251 tokens + half of 512 generated
PROPOSAL_MS = {1: 7.418, 2: 7.907, 4: 8.936}  # timing.txt, ctx 1024 (4K: 8.312 / 8.885 / 9.922; the engine prices 8.3 / 8.9 / 9.9)
PREFILL_S = {2: 0.327, 4: 0.716}  # sum of the solo ttft of the streams' prompts, graph-decode raw/bench-decode.log (0.137, 0.189, 0.181, 0.208)
KINDS = ("chat", "code", "json", "reasoning")  # specbench's c=4 kinds = routelog labels
BENCH_TO_KIND = {"prose": "chat", "code": "code", "structured": "reasoning", "json": "json"}  # [INFERENCE] by the drafted c=1 rates (see plan)

# the engine's round_cost grid (rank time of a window, synthetic ids, 4K): (streams, rows per stream) -> ms, timing.txt / engine.rs WINDOW_MS
_GRID_ROWS = {
    1: [19.050, 21.508, 23.392, 25.026, 25.774, 27.434, 27.641, 29.544],
    2: [23.096, 25.921, 28.438, 30.208, 32.527, 34.282, 35.263, 37.681],
    4: [29.038, 32.999, 37.301, 41.613, 45.448, 47.878, 50.462, 53.014],
}


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def jload(path: str):
    with open(path) as f:
        return json.load(f)


# ───────────────────────── bytes (census) ─────────────────────────
_CENSUS: dict | None = None


def census() -> dict:
    """Bytes per class from intake/census.json: dense bytes per layer, unembed, bytes per expert (layer, expert) and its down K."""
    global _CENSUS
    if _CENSUS is not None:
        return _CENSUS
    mods = jload(os.path.join(ROOT, "intake", "census.json"))["modules"]
    expert: dict[tuple[int, int], int] = collections.defaultdict(int)
    down_k: dict[tuple[int, int], int] = {}
    gate_k: dict[tuple[int, int], int] = {}
    dense: dict[int, int] = collections.defaultdict(int)
    for name, m in mods.items():
        r = re.match(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$", name)
        if r:
            key = (int(r[1]), int(r[2]))
            expert[key] += m["bytes"]
            k = m["tensors"]["trellis"]["shape"][-1] // 16
            if r[3] == "down":
                down_k[key] = k
            elif r[3] == "gate":
                gate_k[key] = k
            continue
        r = re.match(r"model\.llm\.layers\.(\d+)\.", name)
        if r:
            dense[int(r[1])] += m["bytes"]
    b4, b5 = {}, {}
    for layer in range(24):
        x4 = [expert[(layer, e)] for e in range(128) if down_k[(layer, e)] == 4]
        x5 = [expert[(layer, e)] for e in range(128) if down_k[(layer, e)] == 5]
        b4[layer] = sum(x4) / len(x4)
        b5[layer] = sum(x5) / len(x5) if x5 else b4[layer]
    bands = collections.Counter((gate_k[k], gate_k[k], down_k[k]) for k in expert)
    band_bytes = {(gate_k[k], gate_k[k], down_k[k]): expert[k] for k in expert}
    _CENSUS = {
        "dense_visits": sum(dense[v] for v in VISITS),
        "dense_layer": dict(dense),
        "unembed": mods["model.llm.unembed"]["bytes"],
        "other": sum(m["bytes"] for n, m in mods.items() if n.startswith(("model.llm.attn_res", "model.llm.middle"))),
        "expert": dict(expert),
        "b4": b4,
        "b5": b5,
        "bands": dict(bands),
        "band_bytes": band_bytes,
        "experts_total": sum(expert.values()),
        "all_experts_step": sum(sum(expert[(v, e)] for e in range(128)) for v in VISITS),
    }
    return _CENSUS


# ───────────────────────── routes ─────────────────────────
_ROUTE: dict | None = None


def route_stats() -> dict:
    """routelog.txt: distinct[kind][w], bytes_ratio[kind][w], pick5[kind][class]."""
    global _ROUTE
    if _ROUTE is not None:
        return _ROUTE
    dist: dict = collections.defaultdict(dict)
    ratio: dict = collections.defaultdict(dict)
    pick5: dict = collections.defaultdict(dict)
    for ln in read(os.path.join(REC, "2026-11-23-spec-speed", "raw", "routelog.txt")).splitlines():
        m = re.match(r"ROUTE_WINDOW label=(\w+) w=(\d+) distinct_per_visit=([\d.]+)", ln)
        if m:
            dist[m[1]][int(m[2])] = float(m[3])
        m = re.match(r"ROUTE_BYTES label=(\w+) w=(\d+) expert_read_ratio=([\d.]+)", ln)
        if m:
            ratio[m[1]][int(m[2])] = float(m[3])
        m = re.match(r"ROUTE_BIT5 label=(\w+) layer_class=(\w+) pick_share_5bit=([\d.]+)", ln)
        if m:
            pick5[m[1]][m[2]] = float(m[3])
    kinds = sorted(k for k in dist if k != "mean")
    dist["mean"] = {w: sum(dist[k][w] for k in kinds) / len(kinds) for w in dist[kinds[0]]}
    ratio["mean"] = {w: sum(ratio[k][w] for k in kinds) / len(kinds) for w in ratio[kinds[0]]}
    pick5["mean"] = {c: sum(pick5[k][c] for k in kinds) / len(kinds) for c in pick5[kinds[0]]}
    for table in (dist, ratio, pick5):
        table["tool"] = table["code"]  # [INFERENCE] no route log of a tool-call reply: its routes are taken as code's
    _ROUTE = {"distinct": dict(dist), "ratio": dict(ratio), "pick5": dict(pick5)}
    return _ROUTE


def layer_class(layer: int) -> str:
    return "prelude" if layer < 6 else ("core" if layer < 18 else "coda")


def one_row_expert_bytes(kind: str) -> float:
    """Expert bytes one row reads over 48 visits with the kind's real 5-bit pick share (uniform routing gives 1.615 GB)."""
    c, r = census(), route_stats()
    return sum(8 * ((1 - r["pick5"][kind][layer_class(v)]) * c["b4"][v] + r["pick5"][kind][layer_class(v)] * c["b5"][v]) for v in VISITS)


def independent_union(ds, universe: float = 128.0) -> float:
    """Expected distinct experts of streams whose windows touch ds[i] distinct experts each, picked independently: 128 (1 - prod(1 - d/128))."""
    p = 1.0
    for d in ds:
        p *= 1.0 - d / universe
    return universe * (1.0 - p)


def stream_distinct(kind: str, rows: int) -> float:
    return route_stats()["distinct"][kind][min(max(rows, 1), 8)]


def window_bytes(streams, ctx: int = CTX) -> dict:
    """Bytes of one window; streams = [(kind, rows), ...] (distinct streams, rows consecutive tokens each)."""
    c, r = census(), route_stats()
    ds = [stream_distinct(k, n) for k, n in streams]
    union = independent_union(ds)
    e1 = sum(one_row_expert_bytes(k) for k, _ in streams) / len(streams)
    bias = sum(r["ratio"][k][min(n, 8)] / (stream_distinct(k, n) / 8.0) for k, n in streams) / len(streams)
    rows = sum(n for _, n in streams)
    kv = len(streams) * (40 * min(ctx, RING_TOKENS) * RING_B + 8 * ctx * GLOBAL_B)
    return {
        "dense": c["dense_visits"] + c["other"], "unembed": c["unembed"], "kv": kv, "union": union,
        "experts": union / 8.0 * e1 * bias, "z": rows * 8 * Z_BYTES_PER_PICK * 48, "rows": rows,
        "total": c["dense_visits"] + c["other"] + c["unembed"] + kv + union / 8.0 * e1 * bias + rows * 8 * Z_BYTES_PER_PICK * 48,
    }


class Model:
    """T(window) = O + bytes / (BW eff) + ROW_MS x rows. `lev` scales what a lever would change (what-if tables only): dense, unembed,
    experts (byte scales), proposal (ms scale), o (ms override), eff (override)."""

    def __init__(self, o_fit: float, eff_fit: float, **lev):
        self.o, self.eff, self.lev = o_fit, eff_fit, lev

    def window_ms(self, streams, ctx: int = CTX) -> float:
        b = window_bytes(streams, ctx)
        lv = self.lev
        total = (b["dense"] * lv.get("dense", 1.0) + b["unembed"] * lv.get("unembed", 1.0) + b["kv"] + b["experts"] * lv.get("experts", 1.0)
                 + b["z"] * lv.get("experts", 1.0))
        return lv.get("o", self.o) + total / (BW_GBS * 1e9 * lv.get("eff", self.eff)) * 1e3 + ROW_MS * b["rows"]

    def proposal_ms(self, n_drafting: int) -> float:
        return proposal_ms(n_drafting) * self.lev.get("proposal", 1.0)

    def host_ms(self, streams: int) -> float:
        """The part of a drafted round no byte stream explains (commit, drafter commit, launches, sampling): the held-out residual of the
        drafted rounds, linear in streams between the c=1 mean and the c=4 value (set by with_host; zero on the bare fit)."""
        h1, h4 = self.lev.get("_h1", 0.0), self.lev.get("_h4", 0.0)
        return (h1 + (streams - 1) * (h4 - h1) / 3.0) * self.lev.get("host", 1.0)

    def with_lev(self, **lev) -> "Model":
        return Model(self.o, self.eff, **{**self.lev, **lev})


def proposal_ms(n: int) -> float:
    if n <= 0:
        return 0.0
    if n in PROPOSAL_MS:
        return PROPOSAL_MS[n]
    if n == 3:
        return (PROPOSAL_MS[2] + PROPOSAL_MS[4]) / 2
    return PROPOSAL_MS[4] + (n - 4) * (PROPOSAL_MS[4] - PROPOSAL_MS[2]) / 2


def fit(points) -> Model:
    """Least squares on relative residuals of (streams, ms) points: T - ROW_MS rows = O + k x with x = bytes / BW, k = 1 / eff."""
    sa = sb = sc = sd = se = 0.0
    for streams, t in points:
        b = window_bytes(streams)
        x = b["total"] / (BW_GBS * 1e9) * 1e3
        y = t - ROW_MS * b["rows"]
        w = 1.0 / t ** 2
        sa += w
        sb += w * x
        sc += w * x * x
        sd += w * y
        se += w * x * y
    det = sa * sc - sb * sb
    return Model((sd * sc - sb * se) / det, det / (sa * se - sb * sd))


# ───────────────────────── measured points ─────────────────────────
def served_serial_points():
    """The calibration set: steady serial steps with real routes (decode only: the prefill of the span is taken out)."""
    c1 = 1000.0 / 53.394  # bench.py decode_c1_tok_s mean of 4 kinds, graph-decode record (excludes ttft by construction)
    c2 = (2 * 512 / 75.358 * 1000 - PREFILL_S[2] * 1000) / 511  # sweep-decode.log c2 75.358 tok/s over a span of prefill + 511 windows
    c4 = (2048 / 100.385 * 1000 - PREFILL_S[4] * 1000) / 511  # bench-decode.log c4 100.385 tok/s
    return [([("chat", 1)], c1), ([("chat", 1), ("code", 1)], c2), ([(k, 1) for k in KINDS], c4)], (c1, c2, c4)


def accept_hist() -> dict:
    d = jload(os.path.join(REC, "2026-11-23-spec-speed", "raw", "fixed7", "specbench.json"))
    return {k: v["drafted"]["accept_hist"] for k, v in d["kinds"].items()}


def tokens_per_round(hist, depth: int) -> float:
    """1 + E[min(A, depth)], A = accepted drafts at depth 7 (the histogram's rounds by accepted count): greedy truncation is exact."""
    n = sum(hist)
    return 1.0 + sum(c * min(a, depth) for a, c in enumerate(hist)) / n


def expected_tokens(kind: str, depth: int) -> float:
    return tokens_per_round(ACC[kind], depth)


ACC: dict = {}


def init_acc():
    ACC.update(accept_hist())


def served_drafted_c1():
    """Drafted c=1 rounds (spec-speed README table: tok/s, tokens/round) with the mean rows per round from the drafted/rounds counters."""
    d = jload(os.path.join(REC, "2026-11-23-spec-final", "raw", "A-specbench.json"))
    out = {}
    for k, v in d["kinds"].items():
        if k not in KINDS:
            continue
        sp = v["drafted"]["spec"]
        tr = sp["tokens_per_round"]
        rows = 1 + sp["drafted"] / sp["rounds"]
        out[k] = {"round_ms": tr / v["drafted"]["median"] * 1000.0, "rows": rows, "tokens_per_round": tr}
    return out


def window_ms_frac(model: Model, kind: str, rows: float) -> float:
    lo = int(rows)
    hi = min(lo + 1, 8)
    f = rows - lo
    return (1 - f) * model.window_ms([(kind, lo)]) + f * model.window_ms([(kind, hi)])


def c4_drafted_stream_stats():
    """The c=4 drafted run of spec-final: per kind the wall (512 / tok_s), rounds per pass and rows per round; and the phase times."""
    d = jload(os.path.join(REC, "2026-11-23-spec-final", "raw", "A-specbench.json"))
    s = {x["kind"]: x for x in d["concurrency"]["drafted"]["streams"]}
    out = {}
    for k, x in s.items():
        passes = x["spec"]["replies"]
        out[k] = {"wall_s": x["completion_tokens"] / x["tok_s"], "rounds": x["spec"]["rounds"] / passes,
                  "rows": 1 + x["spec"]["drafted"] / x["spec"]["rounds"], "tpr": x["spec"]["tokens_per_round"]}
    span = d["concurrency"]["drafted"]["wall_s"]
    return out, sorted(span)[len(span) // 2]


def phase_round_times(stats, prefill_s: float):
    """Round time at 4, 3, 2, 1 active streams from the finishing order (synchronous rounds: every active stream does one round per round)."""
    order = sorted(stats, key=lambda k: stats[k]["wall_s"])
    t_prev, r_prev, out = prefill_s, 0.0, []
    for i, k in enumerate(order):
        r = stats[k]["rounds"]
        out.append((4 - i, (stats[k]["wall_s"] - t_prev) / (r - r_prev) * 1000.0, k))
        t_prev, r_prev = stats[k]["wall_s"], r
    return out


# ───────────────────────── costs and allotments ─────────────────────────
def engine_grid_ms(s: int, rows_total: int) -> float:
    """The engine's round_cost window price (StepCost::from_grid): linear in rows on a stream-count line, linear in streams between lines."""

    def on_line(line, r):
        pts = [(s_ * (i + 1), t) for i, t in enumerate(_GRID_ROWS[line]) for s_ in [line]]
        if r <= pts[0][0]:
            return pts[0][1] + (r - pts[0][0]) * (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        for (r0, t0), (r1, t1) in zip(pts, pts[1:]):
            if r <= r1:
                return t0 + (r - r0) * (t1 - t0) / (r1 - r0)
        return pts[-1][1] + (r - pts[-1][0]) * (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])

    if s in _GRID_ROWS:
        return on_line(s, rows_total)
    if s == 3:
        return (on_line(2, rows_total) + on_line(4, rows_total)) / 2
    return on_line(4, rows_total) + (s - 4) * (on_line(4, rows_total) - on_line(2, rows_total)) / 2


def engine_round_ms(kinds, depths) -> float:
    """The engine's round price: grid window + the draft layers' time per drafting stream count + 0.2 ms fixed (engine.rs round_cost)."""
    s, rows = len(kinds), sum(1 + d for d in depths)
    nd = sum(1 for d in depths if d > 0)
    return engine_grid_ms(s, rows) + proposal_ms(nd) + (0.2 if nd else 0.0)


def real_round_ms(model: Model, kinds, depths) -> float:
    nd = sum(1 for d in depths if d > 0)
    return model.window_ms([(k, 1 + d) for k, d in zip(kinds, depths)]) + model.proposal_ms(nd) + (model.host_ms(len(kinds)) if nd else 0.0)


def round_tokens(kinds, depths) -> float:
    return sum(expected_tokens(k, d) for k, d in zip(kinds, depths))


def plan_alloc(kinds, cost, max_depth: int = 7):
    """The engine's rule (sched allocation_order + best prefix): hand out rows greedily by marginal expected tokens, keep the prefix that
    maximises expected tokens per ms under `cost(kinds, depths)`."""
    depths = [0] * len(kinds)
    best = (round_tokens(kinds, depths) / cost(kinds, depths), list(depths))
    while True:
        gains = [(expected_tokens(k, d + 1) - expected_tokens(k, d) if d < max_depth else -1.0, i) for i, (k, d) in enumerate(zip(kinds, depths))]
        g, i = max(gains)
        if g <= 0:
            break
        depths[i] += 1
        v = round_tokens(kinds, depths) / cost(kinds, depths)
        if v > best[0]:
            best = (v, list(depths))
    return best[1]


def brute_alloc(kinds, cost, max_depth: int = 7):
    best = (-1.0, None)
    for ds in itertools.product(range(max_depth + 1), repeat=len(kinds)):
        v = round_tokens(kinds, ds) / cost(kinds, ds)
        if v > best[0]:
            best = (v, list(ds))
    return best[1]


def simulate(mix, policy, model: Model, prefill_s: float, n_tokens: int = 512):
    """Fluid simulation of len(mix) concurrent greedy requests of n_tokens: phases of constant active set; `policy(kinds)` -> depths.
    Returns span tok/s (prefill included), steady tok/s (first phase, all streams active), per-kind finish seconds and the phases."""
    remaining = {i: float(n_tokens) for i in range(len(mix))}
    t, phases, finish = prefill_s, [], {}
    while remaining:
        idx = sorted(remaining)
        kinds = [mix[i] for i in idx]
        depths = policy(kinds)
        gains = [expected_tokens(k, d) for k, d in zip(kinds, depths)]
        ms = real_round_ms(model, kinds, depths) if any(depths) else model.window_ms([(k, 1) for k in kinds])
        left = [remaining[i] / g for i, g in zip(idx, gains)]
        n = min(left)
        phases.append({"active": kinds, "depths": depths, "round_ms": ms, "rounds": n, "tok_s": sum(gains) / ms * 1000.0})
        t += n * ms / 1000.0
        for i, g in zip(idx, gains):
            remaining[i] -= n * g
            if remaining[i] <= 1e-9:
                del remaining[i]
                finish[i] = t
    total = n_tokens * len(mix)
    return {"span_tok_s": total / t, "steady_tok_s": phases[0]["tok_s"], "finish": finish, "phases": phases, "span_s": t}


def simulate_rolling(queues, policy, model: Model, prefill_each_s: float, n_tokens: int = 512):
    """Closed loop of len(queues) workers, each running its list of kinds back to back (a finished request is replaced at once; the
    replacement's prefill stalls every stream for prefill_each_s). Fluid phases as in simulate(). Returns tok/s over the whole span and the
    share of the span in which all workers were active."""
    pos = [0] * len(queues)
    rem = [float(n_tokens)] * len(queues)
    t = 0.0
    full = 0.0
    total = n_tokens * sum(len(q) for q in queues)
    t += prefill_each_s * len(queues)
    while any(p < len(q) for p, q in zip(pos, queues)):
        act = [i for i in range(len(queues)) if pos[i] < len(queues[i])]
        kinds = [queues[i][pos[i]] for i in act]
        depths = policy(kinds)
        gains = [expected_tokens(k, d) for k, d in zip(kinds, depths)]
        ms = real_round_ms(model, kinds, depths) if any(depths) else model.window_ms([(k, 1) for k in kinds])
        n = min(rem[i] / g for i, g in zip(act, gains))
        t += n * ms / 1000.0
        if len(act) == len(queues):
            full += n * ms / 1000.0
        for i, g in zip(act, gains):
            rem[i] -= n * g
            if rem[i] <= 1e-9:
                pos[i] += 1
                rem[i] = float(n_tokens)
                if pos[i] < len(queues[i]):
                    t += prefill_each_s
    return {"tok_s": total / t, "span_s": t, "all_active_share": full / t}


def tree_gain(kind: str, depth: int, q: float) -> float:
    """Upper bound of expected extra tokens a round gains from ONE extra candidate row at the first draft position when the primary chain
    is wrong at it: q = P(the alternative is the target's token | the primary was wrong); the alternative's own continuation is not counted
    (nothing is drafted after it). hist P(first draft rejected) = hist[0]/n."""
    h = ACC[kind]
    return q * h[0] / sum(h) if depth > 0 else 0.0


# ───────────────────────── copy drafter (offline) ─────────────────────────
def _extend(tokens, idx, n, state, keep=64):
    for end in range(max(state[0], n), len(tokens) + 1):
        e = idx.setdefault(tuple(tokens[end - n:end]), [])
        e.append(end)
        if len(e) > keep:
            del e[:len(e) - keep]
    state[0] = len(tokens) + 1


def _propose(tokens, idx, n, max_draft, scan=32):
    length = len(tokens)
    if length < n + 1:
        return []
    best = None
    seen = 0
    for end in reversed(idx.get(tuple(tokens[length - n:]), [])):
        if end >= length:
            continue
        seen += 1
        if seen > scan:
            break
        m = n
        while m < end and m < 256 and tokens[end - 1 - m] == tokens[length - 1 - m]:
            m += 1
        if best is None or m > best[0]:
            best = (m, end)
    return [] if best is None else tokens[best[1]:min(length, best[1] + max_draft)]


def copy_sim(prompt, gen, min_match: int = 8, max_draft: int = 7) -> dict:
    """gb_spec::CopyDrafter on a recorded greedy reply: each round proposes from the committed context (hash of the last `min_match`
    tokens, the longest of the 32 most recent earlier matches, up to max_draft tokens), accepts the longest correct prefix, commits it
    plus the bonus token. Returns tokens per round, the share of rounds that had a proposal, accepted per proposal, hist by accepted."""
    tokens, idx, state, pos = list(prompt), {}, [0], 0
    rounds = props = acc_total = 0
    hist: dict[int, int] = collections.defaultdict(int)
    while pos < len(gen):
        _extend(tokens, idx, min_match, state)
        d = _propose(tokens, idx, min_match, max_draft)
        rounds += 1
        a = 0
        if d:
            props += 1
            for i, t in enumerate(d):
                if pos + i < len(gen) and gen[pos + i] == t:
                    a += 1
                else:
                    break
            acc_total += a
        hist[a if d else -1] += 1
        adv = min(a + 1, len(gen) - pos)
        tokens += gen[pos:pos + adv]
        pos += adv
    return {"tokens": len(gen), "rounds": rounds, "tpr": len(gen) / rounds, "proposal_rate": props / rounds,
            "accepted_per_proposal": acc_total / props if props else 0.0, "hist": dict(hist)}


def hybrid_tpr(copy_res: dict, hist7, depth: int = 7) -> float:
    """Upper bound of a copy+block hybrid that verifies the better of the two drafts each round: E[1 + max(A_copy, A_block)] with the
    copy's accepted count on its proposal rounds and the block's accepted count (histogram at depth 7) assumed INDEPENDENT (the real
    correlation is positive, so the true gain is smaller)."""
    n = sum(hist7)
    pb = [c / n for c in hist7]
    ch = {a: c for a, c in copy_res["hist"].items() if a >= 0}
    cn = sum(ch.values())
    rounds = copy_res["rounds"]
    base = 1.0 + sum(p * min(a, depth) for a, p in enumerate(pb))
    if not cn:
        return base
    with_prop = sum((c / cn) * (1.0 + sum(p * max(min(a, depth), ac) for a, p in enumerate(pb))) for ac, c in ch.items())
    share = cn / rounds
    return share * with_prop + (1 - share) * base


# ───────────────────────── c=1: decode_mean4_tok_s (the mean of bench.py's four per-kind rates, one stream) ─────────────────────────
FUSE = os.path.join(REC, "2026-11-23-fuse-decode", "raw")
BENCH_SPEC_KIND = {"prose": "chat", "code": "code", "structured": "reasoning", "json": "json"}  # specbench kind whose depth profile / histogram is used
BENCH_KINDS = ("prose", "code", "structured", "json")
SERIAL_C1_MS = 1000.0 / 53.51  # fuse-decode README: serial c=1 53.51 tok/s


def real_window_table() -> dict:
    """DecodeFuse's windows with real texts (fuse-decode raw/round-cost-table-real-ids-final.txt): {rows: ms} for one stream at 4K, and the
    proposal for one request at 4K."""
    tab, prop = {}, None
    for ln in read(os.path.join(FUSE, "round-cost-table-real-ids-final.txt")).splitlines():
        m_ = re.match(r"LOOP_TIMING window ctx=4096 streams=1 rows=(\d+) drafter=0 ms=([\d.]+)", ln)
        if m_:
            tab[int(m_[1])] = float(m_[2])
        m_ = re.match(r"LOOP_TIMING propose ctx=4096 requests=1 ms=([\d.]+)", ln)
        if m_:
            prop = float(m_[1])
    return {"window": tab, "proposal": prop}


def table_window_ms(rows: float, tab: dict | None = None) -> float:
    tab = tab or real_window_table()["window"]
    lo = max(1, min(int(rows), 8))
    hi = min(lo + 1, 8)
    f = min(max(rows - lo, 0.0), 1.0) if lo < 8 else 0.0
    return tab[lo] * (1 - f) + tab[hi] * f


def bench_counters() -> dict:
    """fuse-decode drafted bench.py decode runs (3 passes after a warm-up): median rate and the spec counters per kind."""
    rates, ctr = collections.defaultdict(list), {}
    for i in (1, 2, 3):
        for ln in read(os.path.join(FUSE, "drafted-final-24a04af", f"bench-decode-{i}.log")).splitlines():
            m_ = re.match(r"decode (\w+): \d+ tokens, ttft [\d.]+ s, ([\d.]+) tok/s, spec \{'drafted': (\d+), 'accepted': (\d+), 'rounds': (\d+), .*'tokens_per_round': ([\d.]+)", ln)
            if m_:
                rates[m_[1]].append(float(m_[2]))
                ctr[m_[1]] = {"drafted": int(m_[3]), "accepted": int(m_[4]), "rounds": int(m_[5]), "tpr": float(m_[6])}
    return {k: {**ctr[k], "rate": sorted(v)[len(v) // 2], "rates": v} for k, v in rates.items()}


def depth_dist(spec_kind: str) -> dict:
    """Rounds by drafted depth of the shipping policy for a specbench kind (fuse-decode specbench.json, per-position `offered`)."""
    d = jload(os.path.join(FUSE, "drafted-final-24a04af", "specbench.json"))["kinds"][spec_kind]["drafted"]
    n = d["position_pass"]["rounds"]
    ge = [n] + [p["offered"] for p in d["per_position"]]
    ge += [0] * (9 - len(ge))
    return {j: (ge[j] - ge[j + 1]) / n for j in range(8)}


def c1_round_ms(kind: str, window: str = "table", ctr: dict | None = None, prop_ms: float | None = None, bare: Model | None = None) -> float:
    """Predicted drafted round of one stream: sum over the policy's depth distribution of window(1 + d) + proposal (when d > 0).
    window = 'table' (DecodeFuse's real-id one-stream table, kind-agnostic) or 'model' (the calibrated byte model with the kind's routes).
    Prose has no per-position record: its depth is the point mass at the mean drafted depth of its counters [INFERENCE]."""
    ctr = ctr or bench_counters()
    prop = real_window_table()["proposal"] if prop_ms is None else prop_ms
    if kind == "prose":
        dist = {ctr[kind]["drafted"] / ctr[kind]["rounds"]: 1.0}
    else:
        dist = {j: p for j, p in depth_dist(BENCH_SPEC_KIND[kind]).items() if p > 0}
    sk = BENCH_SPEC_KIND[kind]
    tot = 0.0
    for dd, p in dist.items():
        r = 1 + dd
        if window == "table":
            w = table_window_ms(r)
        else:
            w = window_ms_frac(bare, sk, r)
        tot += p * (w + (prop if dd > 0 else 0.0))
    return tot


def c1_predict(window: str = "table", bare: Model | None = None) -> dict:
    """{kind: (measured rate, predicted rate, round obs ms, round pred ms)} with tokens per round taken from the counters."""
    ctr = bench_counters()
    out = {}
    for k in BENCH_KINDS:
        pred_round = c1_round_ms(k, window, ctr, bare=bare)
        out[k] = (ctr[k]["rate"], ctr[k]["tpr"] / pred_round * 1000.0, ctr[k]["tpr"] / ctr[k]["rate"] * 1000.0, pred_round)
    return out


def c1_tokens(kind: str, depth: int) -> float:
    return expected_tokens(BENCH_SPEC_KIND[kind], depth)


def c1_best_depth(kind: str, window_ms_fn, prop_ms: float, host_ms: float = 0.0):
    """argmax over depth 0..7 of tokens / (window(1 + d) + proposal [d > 0] + host [d > 0]); returns (depth, tok/s)."""
    best = (-1.0, 0)
    for d in range(8):
        r = c1_tokens(kind, d) / (window_ms_fn(1 + d) + ((prop_ms + host_ms) if d else 0.0)) * 1000.0
        if r > best[0]:
            best = (r, d)
    return best[1], best[0]



def c1_kind_state(bare: Model) -> dict:
    """Per bench kind at c=1: measured rate and round, the depth distribution, the model's decomposition of the round into window +
    proposal and the remaining residual (host), and the hist used for depth what-ifs."""
    ctr, prop = bench_counters(), real_window_table()["proposal"]
    out = {}
    for k in BENCH_KINDS:
        sk = BENCH_SPEC_KIND[k]
        dist = {ctr[k]["drafted"] / ctr[k]["rounds"]: 1.0} if k == "prose" else {j: p for j, p in depth_dist(sk).items() if p > 0}
        win = sum(p * window_ms_frac(bare, sk, 1 + d) for d, p in dist.items())
        share = sum(p for d, p in dist.items() if d > 0)
        obs = ctr[k]["tpr"] / ctr[k]["rate"] * 1000.0
        out[k] = {"rate": ctr[k]["rate"], "tpr": ctr[k]["tpr"], "obs_ms": obs, "dist": dist, "window": win, "share": share, "prop": prop,
                  "resid": obs - win - share * prop}
    return out


def c1_rate_after(st: dict, k: str, d_ms: float = 0.0, tpr_scale: float = 1.0) -> float:
    s = st[k]
    return s["tpr"] * tpr_scale / (s["obs_ms"] - d_ms) * 1000.0


def mean4(rates: dict) -> float:
    return sum(rates[k] for k in BENCH_KINDS) / 4.0


def c1_lever_rates(bare: Model, st: dict, *, prose_serial=False, prop_ms=None, host_frac=1.0, depth=False, copy=0.0, **lev) -> dict:
    """Rates per kind after a set of levers applied to the MEASURED round. Each lever is a delta of ms: the window by the byte model
    (`lev` = Model.with_lev arguments: o, eff, experts, dense), the proposal by the share of rounds that propose, the host by the model's
    residual (host_frac = the part that remains); copy scales tokens per round by `copy` x the oracle hybrid's gain; depth re-optimises
    the depth per kind on the model window; prose_serial replaces prose by the serial rate (a policy that never drafts it)."""
    gains = copy_gains() if copy else {}
    mod = bare.with_lev(**lev)
    out = {}
    for k in BENCH_KINDS:
        s, sk = st[k], BENCH_SPEC_KIND[k]
        if k == "prose" and prose_serial:
            base_serial = prose_serial if prose_serial is not True else 1000.0 / SERIAL_C1_MS  # a float = the rate with the policy's probe rounds
            out[k] = base_serial * bare.window_ms([("chat", 1)]) / mod.window_ms([("chat", 1)])
            continue
        d_ms = sum(p * (window_ms_frac(bare, sk, 1 + d) - window_ms_frac(mod, sk, 1 + d)) for d, p in s["dist"].items())
        prop = s["prop"] if prop_ms is None else prop_ms
        d_ms += s["share"] * (s["prop"] - prop)
        d_ms += max(s["resid"], 0.0) * (1.0 - host_frac)
        scale = 1.0 + copy * gains.get(k, 0.0)
        cur = c1_rate_after(st, k, d_ms, scale)
        if depth and k != "prose":
            host = max(s["resid"], 0.0) * host_frac
            best = max(c1_tokens(k, dd) * scale / (window_ms_frac(mod, sk, 1 + dd) + ((prop + host) if dd else 0.0)) * 1000.0 for dd in range(8))
            cur = max(cur, best)
        out[k] = cur
    return out


def model_rate(bare: Model, st: dict, k: str) -> float:
    """The model's own rate for the kind at its measured depth distribution (ratio base of depth what-ifs)."""
    s = st[k]
    return s["tpr"] / (s["window"] + s["share"] * s["prop"]) * 1000.0


def copy_gains() -> dict:
    """Relative tokens-per-round gain of an oracle-selected copy + block hybrid on bench's kinds (min_match 8, depth 7)."""
    P = jload(os.path.join(DATA, "bench_prompt_ids.json"))
    ref = jload(os.path.join(REC, "2026-11-23-prefill-exact", "raw", "new", "bench-ref.json"))["raw"]
    out = {}
    for k in ("structured", "json"):
        r = copy_sim(P[k], ref["decode/" + k]["generated_ids"], 8)
        h = ACC[BENCH_SPEC_KIND[k]]
        out[k] = hybrid_tpr(r, h) / tokens_per_round(h, 7) - 1.0
    return out


def byte_roofline_c1(kind: str, model_bare: Model) -> tuple:
    """Best depth and tokens/s at the byte floor of a drafted round: (window bytes + drafter weights) / 233.8 GB/s, nothing else."""
    sk = BENCH_SPEC_KIND[kind]
    best = (0.0, 0)
    for d in range(8):
        b = window_bytes([(sk, 1 + d)])["total"] + (339917888 if d else 0)
        v = c1_tokens(kind, d) / (b / (BW_GBS * 1e9) * 1e3) * 1000.0
        if v > best[0]:
            best = (v, d)
    return best[1], best[0]



# ───────────────────────── printing ─────────────────────────
def fmt_table(head, rows, align=None) -> str:
    cols = list(zip(*([head] + rows)))
    w = [max(len(str(c)) for c in col) for col in cols]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) if (align or "l" * len(head))[i] == "l" else str(c).rjust(w[i]) for i, c in enumerate(r)) + " |"
    sep = "|" + "|".join("-" * (x + 2) for x in w) + "|"
    return "\n".join([line(head), sep] + [line(r) for r in rows])


def gb(x: float) -> str:
    return f"{x / 1e9:.3f}"


def section(title: str):
    print(f"\n## {title}\n")


def calibrate():
    pts, steady = served_serial_points()
    return fit(pts), pts, steady


def lab_points():
    """The rank test's lab grid (synthetic ids) quoted by the records; informational (route union unknown)."""
    return [("1 stream x 1 row (19.05)", [("chat", 1)], 19.050), ("2 streams x 1 row (23.10)", [("chat", 1), ("code", 1)], 23.096),
            ("4 streams x 1 row (29.04)", [(k, 1) for k in KINDS], 29.038), ("1 stream x 8 rows (29.54; 29.36 in DECISIONS)", [("code", 8)], 29.544),
            ("4 streams x 8 rows (53.01)", [(k, 8) for k in KINDS], 53.014)]


def implied_distinct(model: Model, t_ms: float, kind: str = "code", rows: int = 8) -> float:
    """Distinct experts a visit that a 1-stream window of `rows` rows would need to take t_ms under the calibrated model (bisection;
    the byte-weighted ratio is rescaled with the count so the expert bytes follow it)."""
    rs = route_stats()
    d0, r0 = rs["distinct"][kind][rows], rs["ratio"][kind][rows]
    bias0 = r0 / (d0 / 8.0)
    lo, hi = 8.0, 128.0
    try:
        for _ in range(60):
            mid = (lo + hi) / 2
            rs["distinct"][kind][rows], rs["ratio"][kind][rows] = mid, mid / 8.0 * bias0
            if model.window_ms([(kind, rows)]) < t_ms:
                lo = mid
            else:
                hi = mid
    finally:
        rs["distinct"][kind][rows], rs["ratio"][kind][rows] = d0, r0
    return (lo + hi) / 2


def validation(model: Model):
    """(label, observed, predicted, gated) rows: fit points, held-out served points, lab points."""
    rows = []
    pts, _ = served_serial_points()
    for (streams, t), name in zip(pts, ("c=1 serial step", "c=2 serial step", "c=4 serial step")):
        rows.append((name + " [fit]", t, model.window_ms(streams), True))
    # c=4 as the records report it (span with prefill) and c=8 (4 slots, two waves)
    t4 = model.window_ms([(k, 1) for k in KINDS])
    rows.append(("c=4 serial span tok/s (100.385)", 100.385, 2048 / (PREFILL_S[4] + 511 * t4 / 1000.0), True))
    rows.append(("c=8 serial tok/s (103.708; 4 slots, 2 waves)", 103.708, 4096 / (PREFILL_S[4] + 1022 * t4 / 1000.0), True))
    for k, v in served_drafted_c1().items():
        rows.append((f"drafted c=1 {k} round ms ({v['rows']:.2f} rows + proposal)", v["round_ms"], window_ms_frac(model, k, v["rows"]) + proposal_ms(1), True))
    stats, _ = c4_drafted_stream_stats()
    phases = phase_round_times(stats, 0.6)
    p4 = phases[0]
    kinds = [k for k in KINDS]
    t_model = model.window_ms([(k, stats[k]["rows"]) if False else (k, round(stats[k]["rows"])) for k in kinds]) + proposal_ms(4)
    rows.append((f"c=4 drafted round at 4 active ms (derived from finish times, prefill 0.6 s [INFERENCE])", p4[1], t_model, True))
    for w in ("table", "model"):
        pr = c1_predict(w, model)
        for k in BENCH_KINDS:
            rows.append((f"c=1 drafted rate {k} tok/s ({'DecodeFuse real-id window table' if w == 'table' else 'byte model'}; fuse-decode)", pr[k][0], pr[k][1], True))
        rows.append((f"c=1 decode_mean4_tok_s ({'real-id table' if w == 'table' else 'byte model'})", sum(v[0] for v in pr.values()) / 4, sum(v[1] for v in pr.values()) / 4, True))
    for label, streams, t in lab_points():
        rows.append(("LAB " + label, t, model.window_ms(streams), False))
    return rows


def with_host(model: Model) -> Model:
    """The bare fit plus the drafted rounds' residual: h1 = mean over the four c=1 drafted rounds of (observed - model), h4 = the c=4 round's."""
    res = {}
    for label, obs, pred, gated in validation(model):
        if label.startswith("drafted c=1"):
            res.setdefault("c1", []).append(obs - pred)
        elif label.startswith("c=4 drafted round"):
            res["c4"] = obs - pred
    return model.with_lev(_h1=sum(res["c1"]) / len(res["c1"]), _h4=res["c4"])


def print_validation(model: Model) -> int:
    rows = validation(model)
    table, bad = [], 0
    for label, obs, pred, gated in rows:
        res = pred / obs - 1.0
        flag = ""
        if gated and abs(res) > 0.10:
            flag, bad = "FAIL", bad + 1
        elif not gated:
            flag = "lab (not gated)"
        table.append([label, f"{obs:.2f}", f"{pred:.2f}", f"{res * 100:+.1f} %", flag])
    print(fmt_table(["point", "measured", "model", "residual", ""], table, "lrrrl"))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--validate", action="store_true", help="calibrate, print the residuals, exit 1 if a served point is off by > 10 %%")
    ap.add_argument("--strict", action="store_true", help="with --validate: gate the lab-synthetic points too")
    a = ap.parse_args(argv)
    init_acc()
    model, pts, steady = calibrate()
    if a.validate:
        print(f"calibrated: O = {model.o:.2f} ms, eff = {model.eff:.3f} of {BW_GBS} GB/s (fit on c=1/2/4 serial steady steps {steady[0]:.2f}/{steady[1]:.2f}/{steady[2]:.2f} ms)\n")
        bad = print_validation(model)
        if a.strict:
            bad += sum(1 for _, o, p, g in validation(model) if not g and abs(p / o - 1) > 0.10)
        print(f"\nvalidate: {'PASS' if bad == 0 else 'FAIL'} ({bad} point(s) off by more than 10 %)")
        return 1 if bad else 0
    report(model, with_host(model), steady)
    return 0


def report(bare: Model, model: Model, steady):
    c = census()
    rs = route_stats()
    ctx = CTX

    section("1. Bytes of one step (census; exact)")
    ex_all = c["all_experts_step"]
    e1_uniform = sum(8 * sum(c["expert"][(v, e)] for e in range(128)) / 128 for v in VISITS)
    print(fmt_table(["item", "GB", "note"], [
        ["dense per step (attention, controllers, router, convs, norms; 48 visits)", gb(c["dense_visits"] + c["other"]), "census, layers 6-17 counted 3x"],
        ["unembed (read once per window, any rows <= 16)", gb(c["unembed"]), "census model.llm.unembed"],
        ["KV per stream per window at ctx %d (ring + int8 global)" % ctx, gb(40 * min(ctx, RING_TOKENS) * RING_B + 8 * ctx * GLOBAL_B), "ROOFLINE constants"],
        ["experts, one row, uniform routing", gb(e1_uniform), "mean expert of each layer x 8 x 48 visits"],
        ["experts, one row, real 5-bit pick share (chat/code/reasoning/json)", " / ".join(gb(one_row_expert_bytes(k)) for k in KINDS), "routelog ROUTE_BIT5 x census bands"],
        ["every expert once per step (the saturation)", gb(ex_all), "128 experts x 48 visits"],
    ], "lrl"))
    print()
    print(fmt_table(["band (gate, up, down bits)", "experts", "bytes each", "share of experts"],
                    [[str(k), v, f"{c['band_bytes'][k]:,}", f"{v / 3072:.3f}"] for k, v in sorted(c["bands"].items())], "lrrr"))
    print(f"\nmean expert bytes: layers 0-5 {sum(c['expert'][(l, e)] for l in range(6) for e in range(128)) / 768 / 1e6:.3f} MB, "
          f"layer 17 {sum(c['expert'][(17, e)] for e in range(128)) / 128 / 1e6:.3f} MB; all experts {c['experts_total'] / 2 ** 30:.3f} GiB")

    section("2. Distinct experts a visit and window bytes (S streams x r rows, mean of the four kinds' real routes)")
    rows = []
    for s in (1, 2, 4):
        for r in (1, 2, 4, 8):
            st = [("mean", r)] * s
            b = window_bytes(st)
            uni = 128 * (1 - (15 / 16) ** (s * r))
            rows.append([s, r, s * r, f"{b['union']:.1f}", f"{uni:.1f}", gb(b["experts"]), gb(b["total"]), f"{model.window_ms(st):.1f}"])
    print(fmt_table(["streams", "rows each", "rows", "distinct / visit (real routes, streams independent)", "if every row were independent", "expert GB", "window GB", "model ms"], rows, "rrrrrrrr"))
    print("\nroutelog distinct experts a visit by w consecutive tokens (kind: w=1..8): " + "; ".join(f"{k} " + " ".join(f"{rs['distinct'][k][w]:.1f}" for w in range(1, 9)) for k in KINDS))

    section("3. Calibration and residuals")
    print(f"free parameters: O = {model.o:.2f} ms, eff = {model.eff:.3f} (x {BW_GBS} GB/s = {model.eff * BW_GBS:.1f} GB/s); fixed per-row term {ROW_MS} ms/row")
    print(f"fit points (steady serial steps with real routes): c=1 {steady[0]:.2f}, c=2 {steady[1]:.2f}, c=4 {steady[2]:.2f} ms\n")
    print_validation(bare)
    print(f"\ndrafted-round residual added to every drafted round below (observed - bare model): h(1 stream) = {model.lev['_h1']:.2f} ms, h(4 streams) = {model.lev['_h4']:.2f} ms")
    print("\nimplied distinct experts a visit for the lab windows (what the synthetic-id rank test touched) vs the real routes:")
    for label, streams, t in lab_points():
        if len(streams) == 1 and streams[0][1] > 1:
            print(f"  {label}: the model needs {implied_distinct(model, t, 'code', 8):.1f} distinct experts a visit; real code reply: {stream_distinct('code', 8):.1f}, "
                  f"independent random rows: {128 * (1 - (15 / 16) ** 8):.1f}")

    section("4. Serial roofline: streams, bytes, model step, bytes-only floor")
    rows = []
    for s in (1, 2, 3, 4, 6, 8, 16):
        st = [(KINDS[i % 4], 1) for i in range(s)]
        b = window_bytes(st)
        t = model.window_ms(st)
        floor = b["total"] / (BW_GBS * 1e9) * 1e3
        rows.append([s, f"{b['union']:.1f}", gb(b["total"]), f"{floor:.1f}", f"{t:.1f}", f"{s / t * 1000:.1f}", f"{s / floor * 1000:.1f}", f"{floor / t:.2f}"])
    print(fmt_table(["streams", "distinct / visit", "GB", "bytes floor ms @233.8", "model step ms", "tok/s", "tok/s at the floor", "floor / step"], rows, "rrrrrrrr"))
    print("\n(streams > 4 is a batch of that many slots; the bootstrap profile has 4, so c=8 in bench.py is two waves of 4.)")

    section("5. Drafted c=4 round, anatomy (the spec-final c=4 run: chat, code, json, reasoning; counters give the rows)")
    stats, span = c4_drafted_stream_stats()
    st = [(k, round(stats[k]["rows"])) for k in KINDS]
    b = window_bytes(st)
    rows = [[k, f"{stats[k]['rows']:.2f}", f"{stats[k]['tpr']:.2f}", f"{stream_distinct(k, round(stats[k]['rows'])):.1f}", f"{stats[k]['rounds']:.0f}", f"{stats[k]['wall_s']:.2f}"] for k in KINDS]
    print(fmt_table(["kind", "rows / round", "tokens / round", "distinct / visit alone", "rounds / 512 tokens", "finishes at s"], rows, "lrrrrr"))
    print(f"\nrows {b['rows']} -> union {b['union']:.1f} distinct / visit, window {gb(b['total'])} GB ({gb(b['experts'])} experts), window {model.window_ms(st):.1f} ms "
          f"+ proposal {proposal_ms(4):.1f} ms = round {model.window_ms(st) + proposal_ms(4):.1f} ms (model)")
    ph = phase_round_times(stats, 0.6)
    print("round time by active streams, derived from the finish times (synchronous rounds, prefill 0.6 s [INFERENCE]): " + ", ".join(f"{n} active {ms:.1f} ms (until {k} ends)" for n, ms, k in ph))
    toks = sum(stats[k]["tpr"] for k in KINDS)
    print(f"tokens per round at 4 active: {toks:.2f} -> steady {toks / ph[0][1] * 1000:.1f} tok/s while all four run; the run's span metric: {2048 / span:.1f} tok/s over {span:.2f} s")

    section("6. The tail: span metric vs steady state")
    pol_engine = lambda ks: plan_alloc(ks, engine_round_ms)
    real_cost = lambda a, b: real_round_ms(model, a, b)
    mixes = {"chat, code, json, reasoning (specbench c=4; bench.py's prose/code/structured/json map onto it)": list(KINDS),
             "4 x code": ["code"] * 4, "4 x json": ["json"] * 4, "4 x reasoning": ["reasoning"] * 4, "4 x chat": ["chat"] * 4}
    rows = []
    for name, mix in mixes.items():
        r = simulate(mix, pol_engine, model, PREFILL_S[4])
        fin = sorted(r["finish"].values())
        rows.append([name.split(" (")[0], " ".join(f"{x:.1f}" for x in fin), f"{r['span_s']:.1f}", f"{r['span_tok_s']:.1f}", f"{r['steady_tok_s']:.1f}",
                     f"{(1 - r['span_tok_s'] / r['steady_tok_s']) * 100:.0f} %", " / ".join(f"{len(p['active'])}:{p['round_ms']:.0f}ms" for p in r["phases"])])
    print(fmt_table(["mix (engine's plan rule on its grid price, real window time)", "finish s", "span s", "span tok/s", "steady tok/s", "tail cost", "phases (active:round)"], rows, "lrrrrrl"))
    print(f"\nmeasured: specbench c=4 drafted {2048 / span:.1f} tok/s over {span:.2f} s (spec-final: 111.373); the same four kinds under bench.py's prompts: c=4 drafted 101.115 and c=8 125.888 "
          f"(review-fixes-b, engine 2e684eb; the two disagree by 10 % at c=4 and the records cannot say why)")

    section("7. Depth allotment at c=4 (real window cost vs the engine's grid price)")
    rows = []
    for name, mix in mixes.items():
        pols = [("serial", lambda ks: [0] * len(ks)), ("fixed 7", lambda ks: [7] * len(ks)), ("fixed 4", lambda ks: [4] * len(ks)), ("fixed 2", lambda ks: [2] * len(ks)),
                ("engine rule on its grid price", pol_engine), ("exhaustive optimum, real cost", lambda ks: brute_alloc(ks, real_cost))]
        base = simulate(mix, pols[0][1], model, PREFILL_S[4])
        for pn, pol in pols:
            r = simulate(mix, pol, model, PREFILL_S[4])
            rows.append([name.split(" (")[0], pn, "/".join(map(str, pol(mix))), f"{r['phases'][0]['round_ms']:.1f}", f"{r['steady_tok_s']:.1f}", f"{r['span_tok_s']:.1f}", f"{r['steady_tok_s'] / base['steady_tok_s']:.3f}"])
    print(fmt_table(["mix", "policy", "depths in mix order", "round ms (4 active)", "steady tok/s", "span tok/s", "steady x serial"], rows, "llrrrrr"))
    print("\nthe engine's own measured depths at c=4, spec-final (rows - 1, mix order chat/code/json/reasoning): " + " / ".join(f"{stats[k]['rows'] - 1:.1f}" for k in KINDS)
          + f"; those depths' tokens per round {round_tokens(KINDS, [round(stats[k]['rows'] - 1) for k in KINDS]):.2f} (hist) vs measured {sum(stats[k]['tpr'] for k in KINDS):.2f}")

    section("8. Marginal row at 4 streams: ms of the next row against the tokens it buys")
    rows = []
    for r in range(1, 8):
        a = model.window_ms([(k, r) for k in KINDS])
        b = model.window_ms([(k, r + 1) for k in KINDS])
        rows.append([f"{r}->{r + 1}", f"{(b - a) / 4:.2f}"] + [f"{expected_tokens(k, r) - expected_tokens(k, r - 1):.2f}" for k in KINDS])
    print(fmt_table(["rows per stream", "ms per added row (every stream grows by one)"] + [f"+tokens {k}" for k in KINDS], rows, "lr" + "r" * 4))
    print("(round price today ~ 88 ms for ~15 tokens = 5.9 ms per token: a row is worth keeping while its added tokens x 5.9 ms exceed its ms)")

    section("9. Lever ceilings at c=4 steady state (specbench mix, exhaustive depth on the real cost unless noted)")
    mix = list(KINDS)
    opt = lambda m: brute_alloc(mix, lambda a, b: real_round_ms(m, a, b))
    engine_depths = pol_engine(mix)

    def steady_of(m: Model, depths=None):
        d = depths or opt(m)
        return round_tokens(mix, d) / real_round_ms(m, mix, d) * 1000.0, d

    base_s, _ = steady_of(model, engine_depths)
    rows = [["today: the engine's allotment on the calibrated model", f"{base_s:.1f}", "1.000", "/".join(map(str, engine_depths))]]
    levs = [("A1 depth allotment on the real cost", model, None),
            ("A2 + proposal 8.9 -> 4.5 ms (half; drafter graph)", model.with_lev(proposal=0.5), None),
            ("A3 + O 4.72 -> 2.7 ms (glue fusion, -2 ms)", model.with_lev(proposal=0.5, o=2.7), None),
            ("A4 + eff 0.997 -> 1.014 (the 237 GB/s DRAM probe)", model.with_lev(proposal=0.5, o=2.7, eff=237.0 / BW_GBS), None),
            ("A5 + RoundOverhead removes the drafted-round residual (host x 0)", model.with_lev(proposal=0.5, o=2.7, eff=237.0 / BW_GBS, host=0.0), None),
            ("bound: proposal free, O = 0, host 0, eff 1.10 (258 GB/s plain stream)", model.with_lev(proposal=0.0, o=0.0, eff=1.10, host=0.0), None)]
    for name, m, dd in levs:
        s_, d = steady_of(m, dd)
        rows.append([name, f"{s_:.1f}", f"{s_ / base_s:.3f}", "/".join(map(str, d))])
    print(fmt_table(["step", "steady tok/s", "x today", "depths"], rows, "lrrl"))

    print("\nsingle levers on today's allotment (what each is worth alone):")
    rows = []
    for name, m in [("proposal 8.9 -> 4.5 ms", model.with_lev(proposal=0.5)), ("proposal free", model.with_lev(proposal=0.0)), ("O 4.72 -> 2.7 ms", model.with_lev(o=2.7)),
                    ("O -> 0", model.with_lev(o=0.0)), ("host residual -> 0", model.with_lev(host=0.0)), ("eff 0.997 -> 1.014", model.with_lev(eff=237.0 / BW_GBS)),
                    ("eff -> 1.10", model.with_lev(eff=1.10))]:
        s_, _ = steady_of(m, engine_depths)
        rows.append([name, f"{s_:.1f}", f"{s_ / base_s:.3f}"])
    print(fmt_table(["lever", "steady tok/s", "x today"], rows, "lrr"))
    print("\nacceptance levers (specbench mix, depth re-optimised):")
    rows = []
    for name, mix2 in [("today", list(KINDS)), ("chat and reasoning replaced by json-like acceptance", ["json", "code", "json", "json"]), ("every stream accepts like code", ["code"] * 4),
                       ("every stream accepts like a tool call (0.82 acceptance)", ["tool"] * 4)]:
        d2 = brute_alloc(mix2, lambda a, b: real_round_ms(model, a, b))
        s_ = round_tokens(mix2, d2) / real_round_ms(model, mix2, d2) * 1000.0
        rows.append([name, "/".join(map(str, d2)), f"{round_tokens(mix2, d2):.2f}", f"{s_:.1f}", f"{s_ / base_s:.3f}"])
    print(fmt_table(["traffic", "depths", "tokens per round", "steady tok/s", "x today"], rows, "llrrr"))
    print("\ntree / multi-candidate verification: one extra candidate row per stream (at the first draft position), q = P(alternative right | primary wrong), upper bound:")
    rows = []
    d0 = opt(model)
    t0 = real_round_ms(model, mix, d0)
    for q in (0.2, 0.35, 1.0):
        gain = sum(tree_gain(k, d, q) for k, d in zip(mix, d0))
        t1 = model.window_ms([(k, 1 + d + (1 if d else 0)) for k, d in zip(mix, d0)]) + model.proposal_ms(sum(1 for d in d0 if d)) + model.host_ms(4)
        rows.append([q, f"{gain:.2f}", f"{t1 - t0:+.1f}", f"{(round_tokens(mix, d0) + gain) / t1 * 1000:.1f}", f"{(round_tokens(mix, d0) + gain) / t1 * t0 / round_tokens(mix, d0):.3f}"])
    print(fmt_table(["q", "+tokens per round (4 streams)", "+ms per round", "steady tok/s", "x A1"], rows, "rrrrr"))

    print("\nB lane (bit-changing), same model:")
    rows = []
    fx = jload(os.path.join(DATA, "fixture_stats.json"))
    for name, m in [("skip experts below 0.15 x the row's top weight (window bytes -3 %)", model.with_lev(experts=1 - fx["window_bytes_saved_long_window_indicative"]["0.15"]["4x8"])),
                    ("skip below 0.2 (-7.4 %, 4.3 % of routing mass)", model.with_lev(experts=1 - fx["window_bytes_saved_long_window_indicative"]["0.2"]["4x8"])),
                    ("skip below 0.3 (-17.7 %, 12.7 % of mass)", model.with_lev(experts=1 - fx["window_bytes_saved_long_window_indicative"]["0.3"]["4x8"])),
                    ("skip below 0.5 (-40.8 %, 34.6 % of mass)", model.with_lev(experts=1 - fx["window_bytes_saved_long_window_indicative"]["0.5"]["4x8"])),
                    ("experts at 3.5 bpw instead of 4.238 mean (bytes x 0.826, a new pack)", model.with_lev(experts=3.5 / 4.238)),
                    ("drop the third pass of the 12 looped layers (visits 36 -> 24)", model.with_lev(**third_pass_levers()))]:
        s_, d = steady_of(m)
        rows.append([name, f"{s_:.1f}", f"{s_ / base_s:.3f}", "/".join(map(str, d))])
    print(fmt_table(["lever (on top of A1 only)", "steady tok/s", "x today", "depths"], rows, "lrrl"))

    print("\nwhere 200 lands: how much faster the window must be at today's acceptance (specbench mix, A1 depths):")
    d = opt(model)
    tok = round_tokens(mix, d)
    need = tok / 200.0 * 1000.0
    print(f"  tokens per round {tok:.2f} at depths {d}; 200 tok/s needs a {need:.1f} ms round; the model's round is {real_round_ms(model, mix, d):.1f} ms "
          f"({real_round_ms(model, mix, d) / need:.2f}x too slow)")

    st = [(k, 1 + x) for k, x in zip(mix, d)]
    wb = window_bytes(st)["total"] + 339917888 * 1  # + the drafter's weights (339,917,888 B, timing.txt) read once per proposal
    print(f"  bytes per generated token at those depths: {wb / tok / 1e9:.3f} GB (window {window_bytes(st)['total'] / 1e9:.1f} GB + drafter weights 0.34 GB, {tok:.2f} tokens) -> "
          f"byte roofline {BW_GBS / (wb / tok / 1e9):.0f} tok/s at 233.8 GB/s, {237.0 / (wb / tok / 1e9):.0f} at 237 GB/s, with O = 0, no host and no drafter latency")
    for name, mix2 in [("4 x code", ["code"] * 4), ("4 x json", ["json"] * 4)]:
        d2 = brute_alloc(mix2, real_cost)
        st2 = [(k, 1 + x) for k, x in zip(mix2, d2)]
        wb2 = window_bytes(st2)["total"] + 339917888
        t2 = round_tokens(mix2, d2)
        print(f"  {name}: depths {d2}, {wb2 / t2 / 1e9:.3f} GB per token -> byte roofline {BW_GBS / (wb2 / t2 / 1e9):.0f} tok/s")

    section("9b. The rolling c=4 metric (4 workers in a closed loop, kinds rotated, 8 requests per worker) against the span metric")
    rot = lambda s: [KINDS[(i + s) % 4] for i in range(4)] * 2
    queues = [rot(s) for s in range(4)]
    prefill_each = sum(PREFILL_S[4] for _ in range(1)) / 4
    for name, pol in [("serial", lambda ks: [0] * len(ks)), ("engine rule", pol_engine), ("optimum on the real cost", lambda ks: brute_alloc(ks, real_cost))]:
        r = simulate_rolling(queues, pol, model, prefill_each)
        one = simulate(list(KINDS), pol, model, PREFILL_S[4])
        print(f"  {name}: rolling {r['tok_s']:.1f} tok/s ({r['all_active_share'] * 100:.0f} % of the span with all four active) | span metric (one request each) {one['span_tok_s']:.1f} | steady {one['steady_tok_s']:.1f}")

    section("10. Copy drafter offline (gb_spec::CopyDrafter on the recorded greedy replies; min_match 8 = the shipped constant)")
    P = jload(os.path.join(DATA, "bench_prompt_ids.json"))
    ref = jload(os.path.join(REC, "2026-11-23-prefill-exact", "raw", "new", "bench-ref.json"))["raw"]
    rows = []
    for k in ("prose", "code", "structured", "json"):
        g = ref["decode/" + k]["generated_ids"]
        for mm in (8, 4):
            r = copy_sim(P[k], g, mm)
            r0 = copy_sim([], g, mm)
            hb = hybrid_tpr(r, ACC[BENCH_TO_KIND[k]])
            rows.append([k, mm, f"{r['tpr']:.3f}", f"{r0['tpr']:.3f}", f"{r['proposal_rate']:.3f}", f"{r['accepted_per_proposal']:.2f}", f"{tokens_per_round(ACC[BENCH_TO_KIND[k]], 7):.2f}", f"{hb:.2f}"])
    print(fmt_table(["bench prompt", "min match", "copy tokens/round (prompt+reply)", "reply only", "rounds with a proposal", "accepted per proposal", f"block drafter (as {'kind'})", "hybrid bound (independence)"], rows, "lrrrrrrr"))
    ex = jload(os.path.join(REC, "2026-11-23-spec-exact", "raw", "exactness-serial.json"))["replies"]
    rows = []
    for e in ex:
        r = copy_sim(e["prompt_ids"], e["generated_ids"], 8)
        rows.append([e["name"], len(e["prompt_ids"]), len(e["generated_ids"]), f"{r['tpr']:.3f}", f"{r['proposal_rate']:.3f}"])
    print("\n" + fmt_table(["spec-exact prompt", "prompt ids", "reply ids", "copy tokens/round", "rounds with a proposal"], rows, "lrrrr"))

    section("11. B lane inputs from the real-pack fixtures (router_w mass)")
    print("mean routing mass by rank of the top-8 (largest first): " + " ".join(f"{x:.3f}" for x in fx["rank_mass_share"]))
    print("dropping the lowest m of 8: " + ", ".join(f"m={m}: {sum(fx['rank_mass_share'][8 - m:]) * 100:.1f} % of the mass, {m / 8 * 100:.1f} % of the picks" for m in (1, 2, 3, 4)))
    print(fmt_table(["row threshold (x top weight)", "picks dropped", "mass dropped", "bytes saved, one row", "window bytes saved 1x8 / 4x8 (counting text: indicative)"],
                    [[s["rel_threshold"], f"{s['picks_dropped']:.3f}", f"{s['mass_dropped']:.3f}", f"{s['bytes_saved_one_row']:.3f}",
                      "%s / %s" % tuple(f"{fx['window_bytes_saved_long_window_indicative'][str(s['rel_threshold'])][k]:.3f}" if str(s['rel_threshold']) in fx['window_bytes_saved_long_window_indicative'] else '-' for k in ('1x8', '4x8'))] for s in fx["row_skip_pooled"]], "rrrrr"))
    mean4_report(bare)


PROPOSAL_BYTES = 339917888 + 6 * 102_940_000 + 347886852  # drafter weights + folded Markov table (201,024 x 256 bf16 = 102.9 MB, DECODE-SPEED) re-read by the 6 further steps + the unembed read once
PROPOSAL_FLOOR_MS = PROPOSAL_BYTES / (BW_GBS * 1e9) * 1e3  # 5.59 ms: the proposal (6.69-6.90 ms) is within 1.3 ms of the bytes it reads; fuse-decode: markov_logits at 231 GB/s
PROPOSAL_FLOOR_CHANGED_MS = (339917888 + 347886852) / (BW_GBS * 1e9) * 1e3  # 2.94 ms: the table read once (a drafter whose Markov logits do not re-read it each step)
# [INFERENCE] prose at 52.5: the lead's 52-53 with the policy's probe rounds; the bound is the plain serial rate (53.51)
EST_C1 = dict(prose_serial=52.5, prop_ms=5.8, host_frac=0.5, o_delta=-0.85, eff=237.0 / BW_GBS, copy=0.5, depth=True)  # prop 5.8: only the items DecodeFuse names (fused argmax 0.17, L2 policy 0.3, global attention ~0.4 [INFERENCE])
BOUND_C1 = dict(prose_serial=True, prop_ms=PROPOSAL_FLOOR_MS, host_frac=0.0, o_delta=-1.7, eff=1.10, copy=1.0, depth=True)
DRAFTER_EST_C1 = {**EST_C1, "prop_ms": 4.5}  # needs a changed drafter arithmetic (table quantised or candidates restricted): the lead's 4.5 ms
DRAFTER_BOUND_C1 = {**BOUND_C1, "prop_ms": PROPOSAL_FLOOR_CHANGED_MS}


def c1_apply(bare: Model, st: dict, **kw) -> dict:
    kw = dict(kw)
    if "o_delta" in kw:
        kw["o"] = bare.o + kw.pop("o_delta")
    return c1_lever_rates(bare, st, **kw)


def mean4_report(bare: Model):
    st = c1_kind_state(bare)
    base = {k: st[k]["rate"] for k in BENCH_KINDS}
    section("12. decode_mean4_tok_s at c=1 (bench decode: prose, code, structured, json; one stream; drafter on; fuse-decode, engine 24a04af)")
    rows = []
    for k in BENCH_KINDS:
        s = st[k]
        rows.append([k, f"{s['rate']:.1f}", f"{s['tpr']:.2f}", f"{1 + sum(d * p for d, p in s['dist'].items()):.2f}", f"{s['obs_ms']:.1f}", f"{s['window']:.1f}", f"{s['share'] * s['prop']:.1f}", f"{s['resid']:+.1f}"])
    print(fmt_table(["kind", "tok/s (median of 3)", "tokens/round", "rows/round", "round ms", "window ms (byte model)", "proposal ms", "residual ms (host)"], rows, "lrrrrrrr"))
    print(f"\nmean4 = {mean4(base):.2f}; the residual (host, commits, drafter commit) at c=1 is {sum(max(st[k]['resid'], 0) for k in BENCH_KINDS) / 3:.1f} ms on code / structured / json (prose: {st['prose']['resid']:+.1f}: the model is high there)")
    print("\nprediction of the four rates (tokens per round from the counters; the round = window + proposal over the shipping policy's depth distribution):\n")
    rows = []
    for w, lab in (("table", "DecodeFuse real-id table (kind-agnostic)"), ("model", "byte model (kind routes)")):
        pr = c1_predict(w, bare)
        rows.append([lab] + [f"{pr[k][1]:.1f} ({(pr[k][1] / pr[k][0] - 1) * 100:+.1f} %)" for k in BENCH_KINDS] + [f"{sum(v[1] for v in pr.values()) / 4:.2f} ({(sum(v[1] for v in pr.values()) / sum(v[0] for v in pr.values()) - 1) * 100:+.1f} %)"])
    print(fmt_table(["window source", *BENCH_KINDS, "mean4"], rows, "lrrrrr"))
    print("\nbyte roofline of a drafted round at c=1 (window bytes + drafter weights at 233.8 GB/s, best depth, kind acceptance from the fixed-depth-7 histograms):\n")
    rows, rf = [], {}
    for k in BENCH_KINDS:
        d, v = byte_roofline_c1(k, bare)
        rf[k] = v
        rows.append([k, d, f"{v:.1f}", f"{base[k] / v * 100:.0f} %"])
    print(fmt_table(["kind", "best depth", "tok/s at the byte floor", "measured / floor"], rows, "lrrr"))
    print(f"\nmean4 at the byte floor: {mean4(rf):.1f} tok/s (prose at its serial floor 71.4: {(mean4(rf) * 4 - rf['prose'] + 71.4) / 4:.1f}); today 83.3 = {mean4(base) / mean4(rf) * 100:.0f} % of it; 100 needs {100 / mean4(rf) * 100:.0f} %")
    print("\nsingle levers (each alone, on the measured rounds): estimate / bound\n")
    singles = [("prose never drafts (serial 52.5 with probe rounds / 53.5)", dict(prose_serial=52.5), dict(prose_serial=True)),
               ("proposal 6.9 -> 5.8 ms (DecodeFuse's named items) / 5.59 ms (its byte floor)", dict(prop_ms=5.8), dict(prop_ms=PROPOSAL_FLOOR_MS)),
               ("proposal with a changed drafter: 4.5 ms / 2.94 ms (table read once)", dict(prop_ms=4.5), dict(prop_ms=PROPOSAL_FLOOR_CHANGED_MS)),
               ("host residual halved / removed", dict(host_frac=0.5), dict(host_frac=0.0)),
               ("O: -0.85 ms / -1.7 ms (half / all of DecodeFuse's 1.7 ms glue chain)", dict(o_delta=-0.85), dict(o_delta=-1.7)),
               ("bandwidth 237 GB/s probe / 258 GB/s stream", dict(eff=237.0 / BW_GBS), dict(eff=1.10)),
               ("depth re-optimised on the model window", dict(depth=True), dict(depth=True)),
               ("copy hybrid, half / oracle gain (structured +6 %, json +8 % tokens a round)", dict(copy=0.5), dict(copy=1.0))]
    rows = []
    for n, e, b in singles:
        re_, rb = mean4(c1_apply(bare, st, **e)), mean4(c1_apply(bare, st, **b))
        rows.append([n, f"{re_:.2f} ({re_ - mean4(base):+.2f})", f"{rb:.2f} ({rb - mean4(base):+.2f})"])
    print(fmt_table(["lever", "mean4 estimate (delta)", "mean4 bound (delta)"], rows, "lrr"))
    print("\ncumulative (levers added in this order):\n")
    acc_e, acc_b, rows = {}, {}, []
    for n, e, b in singles:
        if n.startswith("proposal with a changed"):
            continue
        acc_e.update(e)
        acc_b.update(b)
        re_, rb = c1_apply(bare, st, **acc_e), c1_apply(bare, st, **acc_b)
        rows.append([n.split(" /")[0], f"{mean4(re_):.2f}", f"{mean4(rb):.2f}"])
    print(fmt_table(["+ lever", "mean4 estimate", "mean4 bound"], rows, "lrr"))
    re_, rb = c1_apply(bare, st, **EST_C1), c1_apply(bare, st, **BOUND_C1)
    rd, rdb = c1_apply(bare, st, **DRAFTER_EST_C1), c1_apply(bare, st, **DRAFTER_BOUND_C1)
    print(f"\nthe proposal's byte floor with the same drafter arithmetic is {PROPOSAL_FLOOR_MS:.2f} ms (drafter weights 0.34 GB + the folded table re-read 6x 0.62 GB + the unembed 0.35 GB = {PROPOSAL_BYTES / 1e9:.2f} GB at 233.8 GB/s); "
          f"a drafter that reads the table once would floor at {PROPOSAL_FLOOR_CHANGED_MS:.2f} ms. With the changed drafter: mean4 {mean4(rd):.1f} at 4.5 ms (every other lever at its estimate), {mean4(rdb):.1f} at {PROPOSAL_FLOOR_CHANGED_MS:.2f} ms (every other lever at its bound)")
    print("\nper kind after all A levers: estimate " + ", ".join(f"{k} {re_[k]:.1f}" for k in BENCH_KINDS) + "; bound " + ", ".join(f"{k} {rb[k]:.1f}" for k in BENCH_KINDS))
    def solve(key, lo, hi, base=EST_C1):
        """Value of `key` in [lo, hi] at which mean4 = 100 with every other lever at the estimate (bisection; mean4 is monotone in it)."""
        f = lambda x: mean4(c1_apply(bare, st, **{**base, key: x})) - 100.0
        if f(lo) * f(hi) > 0:
            return None
        for _ in range(50):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    print(f"\nwith every other lever at its estimate, mean4 = 100 needs: a proposal of {solve('prop_ms', 0.0, 6.9):.2f} ms (estimate 5.8, byte floor {PROPOSAL_FLOOR_MS:.2f}), "
          f"or the window's fixed part cut by {-solve('o_delta', -4.7, 0.0):.2f} ms (estimate 0.85), or an effective bandwidth of {solve('eff', 1.0, 1.4) * BW_GBS:.0f} GB/s (estimate 237)")
    print("\nacceptance needed if the A lane stops at the estimate: tokens per round of code / structured / json x f:\n")
    rows = []
    for f in (1.0, 1.02, 1.04, 1.06, 1.1):
        st2 = {k: dict(v) for k, v in st.items()}
        for k in ("code", "structured", "json"):
            st2[k]["tpr"] *= f
        rows.append([f"x{f:.2f}", f"{mean4(c1_apply(bare, st2, **EST_C1)):.1f}", f"{mean4(c1_apply(bare, st2, **{**EST_C1, 'prop_ms': 6.9, 'host_frac': 1.0, 'o_delta': 0.0, 'eff': 1.0, 'copy': 0.0, 'depth': False})):.1f}"])
    print(fmt_table(["tokens per round", "mean4 with the A estimate", "mean4 with only the prose policy"], rows, "lrr"))
    print("\nB lane at c=1 (on top of the A estimate / the A bound; window-byte saving of the 1 x 8 counting-text fixture window, indicative):\n")
    fx = jload(os.path.join(DATA, "fixture_stats.json"))
    rows = []
    for t in ("0.15", "0.2", "0.3", "0.5"):
        sv = 1 - fx["window_bytes_saved_long_window_indicative"][t]["1x8"]
        pk = [x for x in fx["row_skip_pooled"] if str(x["rel_threshold"]) == t][0]
        rows.append([f"skip below {t} x top weight", f"{pk['mass_dropped'] * 100:.1f} %", f"{mean4(c1_apply(bare, st, **EST_C1, experts=sv)):.1f}", f"{mean4(c1_apply(bare, st, **BOUND_C1, experts=sv)):.1f}"])
    for bp in (3.5, 3.0):
        rows.append([f"experts at {bp} bpw (4.238 today)", "-", f"{mean4(c1_apply(bare, st, **EST_C1, experts=bp / 4.238)):.1f}", f"{mean4(c1_apply(bare, st, **BOUND_C1, experts=bp / 4.238)):.1f}"])
    print(fmt_table(["lever", "routing mass dropped", "mean4 on the estimate", "mean4 on the bound"], rows, "lrrr"))



def third_pass_levers() -> dict:
    """Byte scales when the third pass of layers 6-17 is skipped: 12 of 48 visits' dense and expert bytes disappear."""
    c = census()
    d_all = sum(c["dense_layer"][v] for v in VISITS)
    d_third = sum(c["dense_layer"][v] for v in range(6, 18))
    e_all = sum(c["expert"][(v, e)] for v in VISITS for e in range(128))
    e_third = sum(c["expert"][(v, e)] for v in range(6, 18) for e in range(128))
    return {"dense": 1 - d_third / d_all, "experts": 1 - e_third / e_all}


if __name__ == "__main__":
    sys.exit(main())
