"""Pure-function cases of bench/rexl3 (sweeps, profile report, GSM8K, needle_long, fill4); no server, no network."""

from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
sys.path.insert(0, os.path.join(ROOT, "bench", "rexl3"))

import fill4  # noqa: E402
import gsm8k  # noqa: E402
import gsm8k_compare  # noqa: E402
import needle_long  # noqa: E402
import profile_report as pr  # noqa: E402
import sweep_decode  # noqa: E402
import sweep_prefill  # noqa: E402
import sweeplib as sl  # noqa: E402


def _stream(first, last, n, tokens, t0=0.0):
    step = (last - first) / (n - 1)
    return {"events": [(first + i * step, 1) for i in range(n)], "tokens": tokens, "error": None, "t0": t0, "usage": None}


def test_grid_truncates_at_window():
    assert sl.grid(sl.DECODE_GRID, 262_144) == [4096, 16384, 32768, 65536, 131072]
    assert sl.grid(sl.DECODE_GRID, 1_048_576)[-1] == 524288
    assert sl.grid(sl.DECODE_GRID, 1_048_576 + 4096)[-1] == 1_048_576
    assert sl.grid(sl.PREFILL_GRID, 8192) == []


def test_parse_list_top_and_dedupe():
    assert sl.parse_list("4096,top,4096", 262_144, 4096) == [4096, 258_048]
    assert sl.parse_list("", 1000) == []


def test_documented_grid_constants():
    assert sl.DECODE_GRID == tuple(k * 1024 for k in (4, 16, 32, 64, 128, 256, 512, 1024))
    assert sl.PREFILL_GRID[0] == 8192 and sl.PREFILL_GRID[-1] == 1_048_576
    assert sl.DECODE_CONC == (1, 2, 4, 8)


def test_fits_caps_total_tokens():
    assert sl.fits(4, 262_144, sl.MAX_TOTAL_TOKENS)
    assert not sl.fits(2, 1_048_576, sl.MAX_TOTAL_TOKENS)
    assert sl.fits(1, 1_048_576, sl.MAX_TOTAL_TOKENS)


def test_stats_ignores_none():
    s = sl.stats([1.0, None, 3.0, 2.0])
    assert s == {"median": 2.0, "min": 1.0, "max": 3.0, "n": 3}
    assert sl.stats([None])["n"] == 0


def test_stream_rate_uses_usage_tokens():
    s = _stream(1.0, 3.0, 11, 101)
    assert abs(sl.stream_rate(s) - 50.0) < 1e-9          # (101 - 1) / 2 s, not the 11 events
    s["error"] = "x"
    assert sl.stream_rate(s) is None


def test_overlap_metrics_window_and_flags():
    a, b = _stream(1.0, 21.0, 21, 201), _stream(3.0, 23.0, 21, 201)
    m = sl.overlap_metrics([a, b])
    assert m["ok"] and abs(m["overlap_s"] - 18.0) < 1e-9
    assert abs(m["overlap_frac"] - 0.9) < 1e-9
    assert 9.0 < m["aggregate_tok_s"] / 2 < 11.5          # about 10 tok/s a stream
    short = sl.overlap_metrics([_stream(1.0, 5.0, 5, 41), _stream(4.0, 20.0, 17, 161)])
    assert not short["ok"]
    assert not sl.overlap_metrics([a, dict(b, error="boom")])["ok"]


def test_spec_summary():
    rs = [{"spec": {"drafted": 10, "accepted": 6, "rounds": 2}, "completion_tokens": 8},
          {"spec": {"drafted": 10, "accepted": 4, "rounds": 3}, "completion_tokens": 7}]
    s = sl.spec_summary(rs)
    assert s["acceptance"] == 0.5 and s["rounds"] == 5 and s["tokens_per_round"] == 3.0
    assert sl.spec_summary([{"spec": None, "completion_tokens": 3}]) is None


def test_modeapi_serial_injects_draft_false_and_memoises_tokenize():
    class Fake:
        def __init__(self):
            self.calls, self.tok = [], 0

        def complete(self, prompt, max_tokens, **extra):
            self.calls.append(extra)
            return {}, 0.0

        def tokenize(self, text):
            self.tok += 1
            return [1, 2]
    f = Fake()
    api = sl.ModeApi(f)
    api.complete("x", 1, prefix_cache=False)
    s = api.with_mode(True)
    s.complete("x", 1, prefix_cache=False)
    assert f.calls == [{"prefix_cache": False}, {"draft": False, "prefix_cache": False}]
    api.tokenize("a"), s.tokenize("a"), api.tokenize("a")
    assert f.tok == 1


def test_lane_blocks_shift_and_restore():
    orig = sl.bench.block_text
    with sl.lane_blocks(0):
        assert sl.bench.block_text is orig
    with sl.lane_blocks(2):
        assert sl.bench.block_text(1) == orig(2 * sl.LANE_STRIDE + 1)
        assert sl.bench.block_text(1) != orig(1)
    assert sl.bench.block_text is orig


def test_decode_metric_lines_medians():
    res = {"shallow": [{"mode": "serial", "c": 1, "ctx": 4096, "ok": True, "tok_s": v} for v in (60.0, 64.0, 62.0)] +
                      [{"mode": "serial", "c": 1, "ctx": 4096, "ok": False, "tok_s": 1.0}],
           "depth": [{"mode": "default", "c": 2, "ctx": 32768, "ok": True, "tok_s": 90.0, "tokens_per_round": 2.5}]}
    lines = sweep_decode.metric_lines(res)
    assert "METRIC decode_serial_c1_ctx4096_tok_s=62.000" in lines
    assert "METRIC decode_default_c2_ctx32768_tok_s=90.000" in lines
    assert "METRIC decode_default_c2_ctx32768_tpr=2.500" in lines


def test_prefill_metric_lines_and_hit_rule():
    assert sweep_prefill.hit_ok(32600, 32768) and not sweep_prefill.hit_ok(1000, 32768) and not sweep_prefill.hit_ok(None, 5)
    res = {"cells": [{"size": 8192, "c": 1, "ok": True, "tok_s": 2400.0, "ttft_s": 3.4,
                      "hit": {"hit_ok": True, "ttft_s": 0.05}},
                     {"size": 8192, "c": 1, "ok": True, "tok_s": 2500.0, "ttft_s": 3.3, "hit": None}]}
    lines = sweep_prefill.metric_lines(res)
    assert "METRIC prefill_c1_8192_tok_s=2450.0" in lines
    assert "METRIC prefill_cached_8192_ttft_s=0.050" in lines


def test_report_targets_and_verdicts():
    assert pr.verdict(62, 62, 65) == "target" and pr.verdict(66, 62, 65) == "stretch"
    assert pr.verdict(40, 62, 65) == "below" and pr.verdict(40, None, None) == "-"
    assert pr.decode_targets("serial", 1, 262144) == (31.0, 27, 29)
    assert pr.decode_targets("mtp", 1, 4096) == (108, 77, 92)
    assert pr.decode_targets("serial", 4, 4096) == (113, 89, 105)
    assert pr.decode_targets("mtp", 4, 4096) == (None, None, None)
    assert pr.PREFILL_COLD[1048576] == (1533, 380, 610)


def test_report_rows_from_results():
    dec = {"shallow": [{"mode": "default", "c": 1, "ctx": 4096, "ok": True, "tok_s": 80.0, "spec": {"rounds": 3}}],
           "depth": [{"mode": "serial", "c": 1, "ctx": 32768, "ok": True, "tok_s": 50.0}]}
    rows = pr.rows_of(dec)
    by = {(r["ctx"], r["kind"]): r for r in rows}
    assert by[(4096, "mtp")]["target"] == 77 and by[(32768, "serial")]["target"] == 54
    pre = {"cells": [{"size": 32768, "c": 1, "ok": True, "tok_s": 1500.0, "ttft_s": 21.8, "hit": None}]}
    rows = pr.rows_of(pre)
    assert rows[0]["target"] == 2120 and pr.verdict(rows[0]["value"], rows[0]["target"], rows[0]["stretch"]) == "below"
    txt = pr.table(rows + pr.rows_of(dec))
    assert txt.count("\n") >= 4 and "| below |" in txt
    try:
        pr.rows_of({"x": 1})
        raise SystemExit("expected ValueError")
    except ValueError:
        pass


def test_gsm8k_number_and_committed_answer():
    assert gsm8k.number("so $1,234.50 total") == 1234.5
    assert gsm8k.committed_answer("work\n#### 18") == 18.0
    assert gsm8k.committed_answer("no marker 5") is None
    assert gsm8k.committed_answer("#### 3\n#### -7") == -7.0


def test_gsm8k_grade_commit_rules():
    item = {"question": "q", "answer": "reasoning\n#### 1,000"}
    ok = gsm8k.grade(item, "x\n#### 1000", "", "stop", {"completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": 4}}, 3)
    assert ok["ok"] and ok["committed"] and ok["i"] == 3 and ok["reasoning_tokens"] == 4
    cut = gsm8k.grade(item, "x\n#### 1000", "", "length", {"completion_tokens": 9}, 0)
    assert not cut["committed"] and not cut["ok"]
    leak = gsm8k.grade(item, "x\n#### 1000<|end_message|>", "", "stop", {}, 0)
    assert leak["framing_leak"] and not leak["committed"]
    wrong = gsm8k.grade(item, "#### 7", "", "stop", {}, 0)
    assert wrong["committed"] and not wrong["ok"]
    s = gsm8k.summarize([ok, cut, wrong])
    assert s["committed"] == 2 and s["correct"] == 1 and s["length_cut"] == 1 and s["acc"] == 0.5


def test_gsm8k_load_items_and_missing_file_message():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"question": "a", "answer": "x #### 1"}) + "\n\n" + json.dumps({"question": "b", "answer": "y #### 2"}) + "\n")
        assert [i["question"] for i in gsm8k.load_items(p)] == ["a", "b"]
        q = os.path.join(d, "t.json")
        with open(q, "w") as f:
            json.dump([{"question": "c", "answer": "#### 3"}], f)
        assert len(gsm8k.load_items(q)) == 1
        try:
            gsm8k.load_items(os.path.join(d, "none.jsonl"))
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "DATA.md" in str(e)


def test_gsm8k_compare_mcnemar():
    mk = lambda ok: [{"i": i, "ok": o, "committed": True} for i, o in enumerate(ok)]  # noqa: E731
    r = gsm8k_compare.compare(mk([True] * 12 + [False] * 2), mk([False] * 12 + [True] * 2))
    assert r["worse"] and len(r["base_only"]) == 12 and len(r["cand_only"]) == 2 and r["p"] < 0.05
    same = gsm8k_compare.compare(mk([True, False]), mk([True, False]))
    assert same["p"] == 1.0 and not same["worse"]
    assert gsm8k_compare.exact_two_sided(0, 0) == 1.0
    assert abs(gsm8k_compare.exact_two_sided(0, 4) - 0.125) < 1e-12


def test_needle_long_verdicts():
    assert needle_long.verdict({"status": "answered", "found": True})[0]
    assert "refused" in needle_long.verdict({"status": "refused", "http_status": 413, "error": "too big"})[1]
    assert "reasoning" in needle_long.verdict({"status": "answered", "found": False, "found_in_reasoning": True})[1]
    assert "empty" in needle_long.verdict({"status": "empty"})[1]
    s = needle_long.summarize([{"status": "answered", "found": True, "prompt_tokens": 1000, "prefill_s": 4.0},
                               {"status": "answered", "found": False, "answer": "x", "prompt_tokens": 2000, "prefill_s": 4.0}])
    assert s["pass"] == 0.5 and s["tokens"] == 2000 and s["tok_s"] == 250.0
    a = needle_long.lane_args(10, 20, 5, "", 1)
    assert (a.tokens, a.max_model_len, a.max_tokens, a.system, a.seed) == (10, 20, 5, "", 1)


def test_fill4_target_and_failures():
    assert fill4.default_target(262_144, 512) == 262_144 - 512 - 2048
    good = {"status": "answered", "found": True, "answer": "AMBER-1", "prompt_tokens_rendered": 1000,
            "usage": {"prompt_tokens": 1005, "completion_tokens": 5}}
    assert fill4.lane_failures(good, ["COBALT-2"], 262_144) == []
    mixed = dict(good, answer="AMBER-1 COBALT-2")
    assert any("another lane" in w for w in fill4.lane_failures(mixed, ["COBALT-2"], 262_144))
    over = dict(good, usage={"prompt_tokens": 262_000, "completion_tokens": 500}, prompt_tokens_rendered=262_000)
    assert any("> window" in w for w in fill4.lane_failures(over, [], 262_144))
    off = dict(good, usage={"prompt_tokens": 1100, "completion_tokens": 5})
    assert any("sizing" in w for w in fill4.lane_failures(off, [], 262_144))
    assert fill4.lane_failures({"status": "refused", "error": "x"}, [], 10)
