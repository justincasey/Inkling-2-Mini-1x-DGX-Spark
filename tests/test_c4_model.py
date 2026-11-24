#!/usr/bin/env python3
"""Unit check of the arithmetic in bench/tools/c4_model.py whose mistakes would change a conclusion of docs/C4-DECODE-PLAN.md (stdlib only,
offline): the tokens-per-round expectation from a histogram of accepted drafts, the expected union of experts over streams, the exact
census bytes, the calibration (it reproduces its own fit points and the held-out served points within 10 %), the optimal depth allotment
(greedy prefix = exhaustive search; a stream that never accepts gets no rows, one that always accepts gets all), the engine's grid price,
the copy-drafter simulation (a repeating text is copied, a text without repeats is not), the finish-order phase times and the
fluid simulation of concurrent requests (the tail makes the span metric lower than the steady state).

    python3 tests/test_c4_model.py

Ends with the line `test_c4_model: N passed, 0 failed` (exit 1 and `F failed` when a test fails).
"""

from __future__ import annotations

import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
import c4_model as m  # noqa: E402

m.init_acc()


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def test_tokens_per_round_from_histogram():
    assert m.tokens_per_round([0] * 7 + [10], 3) == 4.0  # every round accepts all 7: depth 3 buys 3 drafts
    assert m.tokens_per_round([10] + [0] * 7, 7) == 1.0  # nothing is ever accepted
    assert m.tokens_per_round([5, 5, 0, 0, 0, 0, 0, 0], 7) == 1.5
    # the record's own numbers: code at fixed depth 7 accepts 4.7416 drafts a round (spec-speed fixed7: 1 + accepted_per_round; its
    # tokens_per_round 5.7528 counts the truncated last round of each request too)
    assert close(m.expected_tokens("code", 7), 5.7416, 1e-3), m.expected_tokens("code", 7)
    assert close(m.expected_tokens("chat", 7), 1.9692, 0.02)
    # greedy truncation is monotone and never exceeds depth + 1
    for k in m.KINDS:
        v = [m.expected_tokens(k, d) for d in range(8)]
        assert v == sorted(v) and v[0] == 1.0 and v[7] <= 8.0


def test_union_of_independent_streams():
    assert m.independent_union([8.0]) == 8.0
    assert close(m.independent_union([8.0] * 4), 29.12, 0.01)  # ROOFLINE: 128 x (1 - (15/16)^4) = 29.1
    assert close(m.independent_union([8.0] * 8), 51.62, 0.01)
    u = [m.independent_union([29.4] * s) for s in range(1, 9)]
    assert u == sorted(u) and u[-1] < 128.0
    # more rows of one stream never cost fewer distinct experts, and the real routes touch fewer than independent rows
    assert m.stream_distinct("code", 8) < 51.62 and m.stream_distinct("code", 8) > m.stream_distinct("code", 4)


def test_census_bytes_are_exact():
    c = m.census()
    assert c["bands"] == {(4, 4, 4): 2372, (4, 4, 5): 414, (5, 5, 5): 286}
    assert c["band_bytes"][(4, 4, 4)] == 4002060 and c["band_bytes"][(5, 5, 5)] == 4997388
    assert c["experts_total"] == 12716347392  # 11.843 GiB, docs/ROOFLINE.md
    assert c["unembed"] == 347886852
    uniform_one_row = sum(8 * sum(c["expert"][(v, e)] for e in range(128)) / 128 for v in m.VISITS)
    assert close(uniform_one_row / 1e9, 1.615, 0.001)  # ROOFLINE 1.62
    assert close(c["dense_visits"] / 1e9, 1.179, 0.002)  # attention 0.98 + controllers 0.17 + routers 0.03
    assert len(m.VISITS) == 48 and m.VISITS.count(10) == 3 and m.VISITS.count(3) == 1


def test_calibration_reproduces_the_served_points():
    model, pts, steady = m.calibrate()
    for streams, t in pts:
        assert close(model.window_ms(streams) / t, 1.0, 0.02), (streams, t, model.window_ms(streams))
    assert 0.8 < model.eff < 1.2 and 0 < model.o < 10  # a physical fit: it streams near the 233.8 GB/s ceiling and hides most latency
    rows = m.validation(model)
    gated = [(lbl, o, p) for lbl, o, p, g in rows if g]
    assert len(gated) >= 8
    for lbl, o, p in gated:
        assert abs(p / o - 1.0) <= 0.10, (lbl, o, p)
    # the lab (synthetic-id) windows are the ones that do not fit: 4 streams x 1 row is 33 % cheaper in the lab than served
    lab = {lbl: (o, p) for lbl, o, p, g in rows if not g}
    o4, p4 = lab["LAB 4 streams x 1 row (29.04)"]
    assert p4 / o4 > 1.25


def test_engine_grid_price():
    assert close(m.engine_grid_ms(1, 1), 19.050, 1e-6)
    assert close(m.engine_grid_ms(4, 32), 53.014, 1e-6)
    assert close(m.engine_grid_ms(2, 6), 28.438, 1e-6)  # 2 streams x 3 rows
    assert m.engine_grid_ms(4, 5) > m.engine_grid_ms(4, 4)  # interpolated inside a line
    assert m.engine_grid_ms(3, 6) == (m.engine_grid_ms(2, 6) + m.engine_grid_ms(4, 6)) / 2


def test_optimal_allotment_gives_rows_to_acceptance():
    m.ACC["hi"] = [0] * 7 + [100]  # every draft accepted
    m.ACC["lo"] = [100] + [0] * 7  # none accepted
    cost = lambda kinds, depths: 30.0 + 1.0 * sum(1 + d for d in depths)  # 1 ms a row
    assert m.brute_alloc(["hi", "lo"], cost) == [7, 0]
    assert m.plan_alloc(["hi", "lo"], cost) == [7, 0]
    # the greedy prefix equals the exhaustive optimum on the calibrated model for the c=4 mix
    model = m.calibrate()[0]
    real = lambda ks, ds: m.real_round_ms(model, ks, ds)
    assert m.plan_alloc(list(m.KINDS), real) == m.brute_alloc(list(m.KINDS), real)
    # a window that gets expensive per row pulls the allotment down
    pricey = lambda kinds, depths: 30.0 + 40.0 * sum(1 + d for d in depths)
    assert m.brute_alloc(["hi"], pricey) == [7] or m.brute_alloc(["hi"], pricey) == [0]
    assert m.brute_alloc(["hi", "lo"], lambda k, d: 30.0 + 40.0 * sum(1 + x for x in d)) in ([7, 0], [0, 0])


def test_copy_drafter_simulation():
    pattern = list(range(100, 140))
    gen = pattern * 5  # a text that repeats itself every 40 tokens
    r = m.copy_sim(pattern, gen, 8)
    assert r["tpr"] > 4.0 and r["proposal_rate"] > 0.5
    import random

    rnd = random.Random(7)
    noise = [rnd.randrange(1000, 200000) for _ in range(400)]  # no 8-gram repeats
    r0 = m.copy_sim([1, 2, 3], noise, 8)
    assert r0["tpr"] == 1.0 and r0["proposal_rate"] == 0.0
    # a hybrid that takes the better of two drafts is never worse than the block drafter alone
    assert m.hybrid_tpr(r, m.ACC["json"]) >= m.tokens_per_round(m.ACC["json"], 7)
    assert close(m.hybrid_tpr(r0, m.ACC["json"]), m.tokens_per_round(m.ACC["json"], 7), 1e-9)


def test_finish_order_phase_times():
    stats = {"a": {"wall_s": 10.0, "rounds": 100.0}, "b": {"wall_s": 16.0, "rounds": 150.0},
             "c": {"wall_s": 19.0, "rounds": 190.0}, "d": {"wall_s": 21.0, "rounds": 240.0}}
    ph = m.phase_round_times(stats, 1.0)
    assert [p[0] for p in ph] == [4, 3, 2, 1] and [p[2] for p in ph] == ["a", "b", "c", "d"]
    assert close(ph[0][1], 90.0, 1e-9) and close(ph[1][1], 120.0, 1e-9) and close(ph[2][1], 75.0, 1e-9) and close(ph[3][1], 40.0, 1e-9)


def test_fluid_simulation_and_the_tail():
    model = m.calibrate()[0]
    one = m.simulate(["code"], lambda ks: [7] * len(ks), model, 0.0)
    ms = m.real_round_ms(model, ["code"], [7])
    assert close(one["span_tok_s"], m.expected_tokens("code", 7) / ms * 1000.0, 1e-6)
    same = m.simulate(["code"] * 4, lambda ks: [7] * len(ks), model, 0.0)
    assert len(set(round(x, 6) for x in same["finish"].values())) == 1  # equal streams end together: no tail
    mix = m.simulate(list(m.KINDS), lambda ks: m.plan_alloc(ks, m.engine_round_ms), model, 0.0)
    assert mix["span_tok_s"] < 0.85 * mix["steady_tok_s"]  # the slowest stream runs alone at the end
    roll = m.simulate_rolling([[k for k in m.KINDS] * 2 for _ in range(4)], lambda ks: [0] * len(ks), model, 0.0)
    assert roll["all_active_share"] > 0.9 and close(roll["tok_s"], 4 / model.window_ms([(k, 1) for k in m.KINDS]) * 1000.0, 3.0)


def test_c1_per_kind_prediction_and_levers():
    bare = m.calibrate()[0]
    tab = m.real_window_table()
    assert tab["window"][1] == 19.276 and tab["window"][8] == 41.928 and tab["proposal"] == 6.9  # DecodeFuse's real-id table (fuse-decode)
    assert m.table_window_ms(8) == 41.928 and close(m.table_window_ms(4.5), (31.992 + 33.331) / 2, 1e-9)
    for k in m.BENCH_KINDS:
        assert close(sum(m.depth_dist(m.BENCH_SPEC_KIND[k]).values()), 1.0, 1e-9)
    for window in ("table", "model"):  # each kind's drafted rate within 10 %, the mean within 5 %
        pr = m.c1_predict(window, bare)
        for k, (obs, pred, _, _) in pr.items():
            assert abs(pred / obs - 1) < 0.10, (window, k, obs, pred)
        assert abs(sum(v[1] for v in pr.values()) / sum(v[0] for v in pr.values()) - 1) < 0.05
    ctr = m.bench_counters()
    assert close(sum(ctr[k]["rate"] for k in m.BENCH_KINDS) / 4, 83.32, 0.05)  # fuse-decode: 83.35 (median of the three means)
    # the break-even: prose (chat's histogram) never pays with the real window grid, code and json want all seven drafts
    assert m.c1_best_depth("prose", m.table_window_ms, 6.9)[0] == 0
    assert m.c1_best_depth("code", m.table_window_ms, 6.9)[0] == 7
    # levers move the mean the right way and by the right amount
    st = m.c1_kind_state(bare)
    base = m.mean4(m.c1_lever_rates(bare, st))
    assert close(base, 83.32, 0.05)  # no lever reproduces the measured mean
    prose = m.mean4(m.c1_lever_rates(bare, st, prose_serial=True))
    assert close(prose - base, (1000.0 / m.SERIAL_C1_MS - st["prose"]["rate"]) / 4, 1e-6)  # (53.5 - 42.7) / 4
    est, bound = m.c1_apply(bare, st, **m.EST_C1), m.c1_apply(bare, st, **m.BOUND_C1)
    assert base < m.mean4(est) < m.mean4(bound)
    assert m.mean4(m.c1_lever_rates(bare, st, prop_ms=4.5)) > base + 4.0  # a 2.4 ms cheaper proposal on every drafted round
    # a byte lever scales the window and nothing else: halving the experts cannot more than double the rate
    assert m.mean4(m.c1_lever_rates(bare, st, experts=0.5)) < 2 * base



if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fails = 0
    for n, f in tests:
        try:
            f()
        except Exception:
            fails += 1
            print(f"FAIL {n}")
            traceback.print_exc()
    print(f"test_c4_model: {len(tests) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
