#!/usr/bin/env python3
"""Unit check of the harness's decisions, no node and no engine (stdlib only; a stub server on loopback stands in for the API).

The score is zero unless every required gate has its exact `PASS <id>` line and is capped by the lane count; a reply is held
to its pinned reference byte for byte (a one-id difference, a different finish reason, a different prompt and an unpinned
name all fail), and a concurrent reply is held to the solo reply of its prompt; bench.py end to end against the stub: a run
that matches its pin prints `PASS pin`, an engine that changed one id prints `FAIL pin`, a concurrent reply that differs
from its solo reply fails, a pin run writes and extends the reference, a window too small for a prompt skips it; the memory
gate reads the guard's one-line report.

    python3 tests/test_bench.py

Ends with the line `test_bench: N passed, 0 failed` (exit 1 and `F failed` when a test fails).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import bench  # noqa: E402
import inkling_api as ia  # noqa: E402
import memcheck  # noqa: E402
import sampled  # noqa: E402
import score  # noqa: E402
import stub_api  # noqa: E402

TARGETS = {"decode_c1_tok_s": 62.0, "prefill_32k_tok_s": 2120.0, "lanes_at_max_context": 4.0}
REQ = score.REQUIRED_GATES.split()
ALLPASS = {g: "PASS" for g in REQ}


def check(cond, what):
    if not cond:
        raise AssertionError(what)


# ── the score ──
def test_score_required_gates_are_the_plan_ids() -> None:
    check(REQ == "pin spec cache chunk vision needle capacity memory smoke host goldens".split(), str(REQ))
    check(len(set(REQ)) == 11, "eleven distinct gate ids")


def test_score_targets_met_is_one() -> None:
    raw, got, bad = score.score(TARGETS, ALLPASS, REQ)
    check(math.isclose(raw, 1.0) and math.isclose(got, 1.0) and not bad, f"{raw} {got} {bad}")
    check((score.DECODE_TARGET, score.PREFILL_TARGET, score.LANES_TARGET) == (62.0, 2120.0, 4.0), "the ROOFLINE proposals")


def test_score_half_decode_is_sqrt_half() -> None:
    _, got, _ = score.score({**TARGETS, "decode_c1_tok_s": 31.0}, ALLPASS, REQ)
    check(math.isclose(got, math.sqrt(0.5)), f"got {got}")


def test_score_lane_cap() -> None:
    _, got, _ = score.score({**TARGETS, "lanes_at_max_context": 2.0}, ALLPASS, REQ)
    check(math.isclose(got, 0.5), f"2 of 4 lanes halves the score, got {got}")
    _, got, _ = score.score({**TARGETS, "lanes_at_max_context": 9.0}, ALLPASS, REQ)
    check(math.isclose(got, 1.0), "more lanes than the target do not raise it")


def test_score_failed_gate_zeroes_but_keeps_raw() -> None:
    raw, got, bad = score.score(TARGETS, {**ALLPASS, "spec": "FAIL"}, REQ)
    check(got == 0.0 and math.isclose(raw, 1.0) and bad == ["spec (FAIL)"], f"{got} {raw} {bad}")


def test_score_every_required_gate_zeroes_when_missing() -> None:
    for g in REQ:
        gates = {k: v for k, v in ALLPASS.items() if k != g}
        raw, got, bad = score.score(TARGETS, gates, REQ)
        check(got == 0.0 and bad == [f"{g} (missing)"], f"{g}: {got} {bad}")


def test_score_missing_or_zero_metric_scores_zero() -> None:
    raw, got, _ = score.score({"decode_c1_tok_s": 62.0}, ALLPASS, REQ)
    check(raw == 0.0 and got == 0.0, "a missing metric scores 0")
    raw, _, _ = score.score({**TARGETS, "decode_c1_tok_s": 0.0}, ALLPASS, REQ)
    check(raw == 0.0, "a zero rate scores 0")


def test_score_gate_lines_are_exact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        g = os.path.join(tmp, "gates.txt")
        with open(g, "w") as f:
            f.write("PASS pin\nFAIL spec\nPASS spec\nPASS pin2 extra\nPASS  capacity\npass memory\nPASS-smoke\n")
        parsed = score.read_gates(g)
    check(parsed == {"pin": "PASS", "spec": "PASS"}, f"exact lines only; a later line wins: {parsed}")


def test_score_metric_lines_parse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mf = os.path.join(tmp, "metrics.txt")
        with open(mf, "w") as f:
            f.write("noise\nMETRIC decode_c1_tok_s=12.5\nMETRIC bad=x\nMETRIC prefill_32k_tok_s=3000\n")
        check(score.read_metrics(mf) == {"decode_c1_tok_s": 12.5, "prefill_32k_tok_s": 3000.0}, "metric lines")


def test_score_cli_zero_unless_every_gate_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mf, gf = os.path.join(tmp, "m.txt"), os.path.join(tmp, "g.txt")
        with open(mf, "w") as f:
            f.write("".join(f"METRIC {k}={v}\n" for k, v in TARGETS.items()))
        with open(gf, "w") as f:
            f.write("".join(f"PASS {g}\n" for g in REQ if g != "goldens"))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = score.main(["--metrics", mf, "--gates", gf, "--required", score.REQUIRED_GATES])
        check(rc == 0 and "METRIC target_score=0.0000" in out.getvalue() and "METRIC target_score_raw=1.0000" in out.getvalue()
              and "METRIC gates_passed=10" in out.getvalue() and "NOT PASSED goldens (missing)" in err.getvalue(),
              out.getvalue() + err.getvalue())
        with open(gf, "a") as f:
            f.write("PASS goldens\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            score.main(["--metrics", mf, "--gates", gf, "--required", score.REQUIRED_GATES])
        check("METRIC target_score=1.0000" in out.getvalue(), out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            score.main(["--metrics", mf, "--gates", gf, "--required", score.REQUIRED_GATES, "--screen"])
        check("METRIC target_score=0.0000" in out.getvalue(), "a screen never scores")


# ── pinned replies ──
def rep(ids, finish="length", sha="a" * 64) -> dict:
    return {"prompt_sha256": sha, "generated_ids": ids, "finish_reason": finish}


def test_pin_first_diff_is_byte_exact() -> None:
    want = rep([5, 6, 7, 8])
    check(bench.first_diff("x", rep([5, 6, 7, 8]), want) is None, "equal reply passes")
    d = bench.first_diff("x", rep([5, 6, 9, 8]), want)
    check(d and "differ at 2" in d, str(d))
    d = bench.first_diff("x", rep([5, 6, 7]), want)
    check(d and "differ at 3" in d and "lengths 3/4" in d, str(d))
    d = bench.first_diff("x", rep([5, 6, 7, 8], finish="stop"), want)
    check(d and "finish stop/length" in d, str(d))
    d = bench.first_diff("x", rep([5, 6, 7, 8], sha="b" * 64), want)
    check(d and "prompt differs" in d, str(d))


def test_pin_concurrent_replies_answer_their_solo_entry() -> None:
    ref = {"decode/json": rep([1, 2]), "prefill/32k": rep([3])}
    check(bench.pinned_name("conc8/json#1") == "decode/json" and bench.pinned_name("prefill/32k") == "prefill/32k", "names")
    replies = {"decode/json": {**rep([1, 2]), "x": 1}, "conc4/json#0": rep([1, 2]), "conc8/json#1": rep([1, 9])}
    errs = bench.check_pinned(replies, ref)
    check(len(errs) == 1 and errs[0].startswith("conc8/json#1: generated ids differ at 1"), str(errs))


def test_pin_unpinned_name_fails_and_empty_screen_passes() -> None:
    ref = {"decode/json": rep([1, 2])}
    check(bench.check_pinned({"prefill/128k": rep([4])}, ref) == ["prefill/128k: prefill/128k is not in the pinned reference"], "unpinned")
    check(bench.check_pinned({}, ref) == [], "a screen that ran nothing compares nothing")


def test_pin_extension_adds_new_and_requires_old_reproduced() -> None:
    ref = {"decode/json": {**rep([1, 2]), "prompt_tokens": 9}, "prefill/32k": {**rep([3]), "prompt_tokens": 32768}}
    run = {"decode/json": {**rep([1, 2]), "prompt_tokens": 9, "ttft_s": 1.0}, "conc4/json#0": rep([1, 2]),
           "prefill/224k": {**rep([7, 8], sha="c" * 64), "prompt_tokens": 229376, "wall_s": 3.0}}
    merged, errs = bench.extend_reference(ref, run)
    check(not errs and set(merged) == {"decode/json", "prefill/32k", "prefill/224k"}, f"{errs} {set(merged)}")
    check(merged["prefill/224k"] == {**rep([7, 8], sha="c" * 64), "prompt_tokens": 229376}, "only the pinned fields are kept")
    check(set(ref) == {"decode/json", "prefill/32k"}, "the input reference is not modified")
    merged, errs = bench.extend_reference(ref, {**run, "decode/json": {**rep([1, 9]), "prompt_tokens": 9}})
    check(len(errs) == 1 and errs[0].startswith("decode/json: generated ids differ at 1"), str(errs))
    merged, errs = bench.extend_reference(ref, {"conc8/json#1": rep([5])})
    check(not errs and merged == ref, "concurrent replies are never pinned")


def test_hist_reads_sum_and_count_and_stops_on_a_missing_series() -> None:
    m = {"rexl3_request_ttft_seconds_sum": 2.5, "rexl3_request_ttft_seconds_count": 4.0,
         'rexl3_request_ttft_seconds_bucket{le="1"}': 3.0}
    check(bench.hist(m, bench.TTFT) == (2.5, 4.0), "sum and count, buckets ignored")
    try:
        bench.hist({"rexl3_other": 1.0}, bench.TTFT)
    except SystemExit as e:
        check("no rexl3_request_ttft_seconds_count" in str(e), str(e))
    else:
        raise AssertionError("a missing histogram must stop the run, not read as zero")


def test_window_decides_which_prompts_run() -> None:
    check(bench.fits("32k", 262_144) and bench.fits("128k", 262_144) and bench.fits("224k", 262_144), "bootstrap window")
    check(not bench.fits("1m", 262_144) and bench.fits("1m", 1_048_576), "1M needs the long window")
    check(bench.SIZES["224k"] + bench.LONG_TOKENS < 262_144, "the largest routine prompt leaves room for its answer")


# ── bench.py end to end against the stub ──
def run_bench(url: str, *args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        bench.FAILS.clear()
        rc = bench.main(["--url", url, "--model", "stub-model", *args])
    return rc, out.getvalue(), err.getvalue()


def test_bench_pin_then_ref_passes_and_a_changed_engine_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "reference.json")
        with stub_api.serve() as (url, _):
            rc, out, err = run_bench(url, "--pin", ref, "--phases", "decode,conc", "--engine", "e" * 40)
            check(rc == 0 and "METRIC decode_c1_tok_s=" in out and "PASS pin" not in out, f"pin run: rc {rc}\n{err}")
            pinned = json.load(open(ref))
            check(sorted(pinned["replies"]) == ["decode/code", "decode/json", "decode/prose", "decode/structured"]
                  and pinned["meta"]["engine"] == "e" * 40, f"solo entries only: {sorted(pinned['replies'])}")
            rc, out, err = run_bench(url, "--ref", ref, "--phases", "decode,conc")
            check(rc == 0 and out.rstrip().endswith("PASS pin"), f"same engine passes: rc {rc}\n{out}\n{err}")
        with stub_api.serve(bug="all") as (url, _):
            rc, out, err = run_bench(url, "--ref", ref, "--phases", "decode")
            check(rc == 1 and out.rstrip().endswith("FAIL pin") and "generated ids differ at 3" in err, f"rc {rc}\n{err}")


def test_bench_concurrent_must_equal_solo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with stub_api.serve(bug="conc") as (url, _):
            rc, out, err = run_bench(url, "--raw", "--phases", "decode,conc")
            check(rc == 1 and "concurrent != solo" in err and "FAIL concurrent" in out, f"rc {rc}\n{err}")
            rc, out, err = run_bench(url, "--pin", os.path.join(tmp, "r.json"), "--phases", "decode,conc")
            check(rc == 1 and not os.path.exists(os.path.join(tmp, "r.json")), "a failed pin run writes nothing")


def test_bench_raw_measures_without_a_reference() -> None:
    with stub_api.serve() as (url, _):
        rc, out, err = run_bench(url, "--raw", "--phases", "decode,conc")
    check(rc == 0 and "PASS pin" not in out and "FAIL pin" not in out and "INFO pin: replies not compared" in out, out)
    for k in ("decode_c1_tok_s", "decode_c4_tok_s", "decode_c8_tok_s", "decode_acceptance", "decode_tokens_per_round"):
        check(f"METRIC {k}=" in out, f"{k} missing:\n{out}")


def test_bench_decode_rate_comes_from_usage_and_server_histograms() -> None:
    with stub_api.serve() as (url, _):
        rc, out, err = run_bench(url, "--raw", "--phases", "decode")
    rate = float(next(l for l in out.splitlines() if l.startswith("METRIC decode_c1_tok_s=")).split("=")[1])
    check(rc == 0 and math.isclose(rate, 511 / 5.12, rel_tol=1e-3), f"(512 - 1) tokens / (512 x 0.01 s): {rate}")


def test_bench_cold_requests_opt_out_of_the_prefix_cache() -> None:
    with stub_api.serve() as (url, st):
        run_bench(url, "--raw", "--phases", "decode")
        check(st.requests and all(r["prefix_cache"] is False and r["temperature"] == 0 and r["return_token_ids"]
                                  for r in st.requests), "every bench request is greedy, cold, with ids")


def test_bench_prefill_skips_what_the_window_cannot_hold() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "r.json")
        with stub_api.serve() as (url, _):
            rc, out, err = run_bench(url, "--pin", ref, "--phases", "prefill", "--max-model-len", "40000")
        check(rc == 0 and "prefill 128k: skipped" in err and "prefill 224k: skipped" in err, err)
        check(sorted(json.load(open(ref))["replies"]) == ["prefill/32k"], "only the 32K prompt ran and was pinned")
        with stub_api.serve() as (url, _):
            try:
                run_bench(url, "--raw", "--phases", "prefill-1m", "--max-model-len", "262144")
            except SystemExit as e:
                check("boot profile long or max" in str(e), str(e))
            else:
                raise AssertionError("a 1M run on a 262,144 window must stop")


def test_bench_pin_extends_and_refuses_a_changed_reply() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "r.json")
        with stub_api.serve() as (url, _):
            run_bench(url, "--pin", ref, "--phases", "decode", "--engine", "a" * 40)
            rc, _, err = run_bench(url, "--pin", ref, "--phases", "decode,prefill-32k", "--engine", "b" * 40)
            data = json.load(open(ref))
            check(rc == 0 and "prefill/32k" in data["replies"] and data["meta"]["extended"][0]["added"] == ["prefill/32k"],
                  f"extended: rc {rc}\n{err}")
        before = open(ref).read()
        with stub_api.serve(bug="all") as (url, _):
            rc, _, err = run_bench(url, "--pin", ref, "--phases", "decode")
        check(rc == 1 and open(ref).read() == before, "a pin run that does not reproduce leaves the file unchanged")


# ── rolling: sustained c = 4 ──
KINDS = list(bench.DECODE_TEXTS)


def rec(worker, index, start, first_token, end, tokens=512, finish="length", ids=None) -> dict:
    """A request record as run_rolling and phase_rolling build them (a hand-made timeline)."""
    return {"worker": worker, "index": index, "kind": KINDS[(worker + index) % 4], "start": start,
            "first_token": first_token, "end": end,
            "reply": {"completion_tokens": tokens, "finish_reason": finish, "generated_ids": ids or [0] * tokens,
                      "cached_tokens": 0, "spec": None}}


def test_rolling_rotation_every_worker_sees_each_kind_twice_from_its_own_start() -> None:
    check(bench.ROLLING_WORKERS == 4 and bench.ROLLING_ROUNDS == 2 and bench.DECODE_TOKENS == 512, "the plan's shape: 4 x 8 x 512")
    starts = []
    for w in range(4):
        rot = bench.rotation(w, KINDS)
        check(len(rot) == 8 and all(rot.count(k) == 2 for k in KINDS), f"worker {w} sees each kind twice: {rot}")
        check(rot[0] == KINDS[w] and rot[:4] == [KINDS[(w + j) % 4] for j in range(4)], f"worker {w} starts at kind {w}: {rot}")
        check(rot[4:] == rot[:4], "the second round repeats the first")
        starts.append(rot[0])
    check(sorted(starts) == sorted(KINDS), "the four workers start at four different kinds")


def test_rolling_workers_are_closed_loops_with_four_in_flight() -> None:
    with stub_api.serve() as (url, st):
        api = ia.Api(url, "stub-model")
        prompts = {n: api.tokenize(t) for n, t in bench.DECODE_TEXTS.items()}
        recs = bench.run_rolling(api, prompts)
    check(len(recs) == 32 and len(st.timeline) == 32, f"32 requests: {len(recs)} {len(st.timeline)}")
    check(st.max_inflight == 4, f"four requests in flight at the most, and four at once: {st.max_inflight}")
    for w in range(4):
        mine = sorted((r for r in recs if r["worker"] == w), key=lambda r: r["index"])
        check([r["index"] for r in mine] == list(range(8)) and [r["kind"] for r in mine] == bench.rotation(w, KINDS),
              f"worker {w}: eight requests in rotation order")
        for a, b in zip(mine, mine[1:]):
            gap = b["start"] - a["end"]
            check(0 <= gap < 0.05, f"worker {w}: request {b['index']} starts {gap:.4f} s after request {a['index']} ended "
                                   f"(never two in flight, none held back)")
    firsts = [min(r["start"] for r in recs if r["worker"] == w) for w in range(4)]
    check(max(firsts) - min(firsts) < 0.05, "the four workers start together (one barrier)")
    first_ids = {r["kind"]: r["reply"]["generated_ids"] for r in recs}
    check(all(len(v) == 512 for v in first_ids.values()), "512 ids per reply")


def test_rolling_metrics_come_from_usage_tokens_and_first_start_to_last_end() -> None:
    # 8 requests, one short (400 tokens, finish stop); generated_ids deliberately not the usage count
    recs = [rec(0, 0, 100.0, 100.5, 104.0), rec(0, 1, 104.0, 104.2, 109.0, tokens=400, finish="stop", ids=[1] * 17),
            rec(1, 0, 100.0, 100.6, 105.0), rec(1, 1, 105.0, 105.1, 110.0),
            rec(2, 0, 100.0, 100.7, 106.0), rec(2, 1, 106.0, 106.1, 112.0),
            rec(3, 0, 100.0, 100.8, 107.0), rec(3, 1, 107.0, 107.1, 114.0)]
    m, notes = bench.rolling_metrics(recs)
    tokens = 7 * 512 + 400
    check(m["decode_c4_rolling_tok_s"] == round(tokens / 14.0, 3), f"usage tokens / (114 - 100): {m}")
    check(m["decode_c4_rolling_span_s"] == 14.0 and m["decode_c4_rolling_requests"] == 8, str(m))
    check(m["decode_c4_rolling_short_replies"] == 1, "one short reply counted")
    check("decode_c4_steady_tok_s" not in m and len(notes) == 1 and "under 10 s" in notes[0], f"window 100.8..109 is 8.2 s: {notes}")
    m2, _ = bench.rolling_metrics([rec(0, 0, 0.0, 0.1, 10.0, finish="length", tokens=512)])
    check(m2["decode_c4_rolling_short_replies"] == 0, "a full-length reply is not short")
    check(bench.is_short(rec(0, 0, 0, 0, 1, tokens=511)["reply"], 512) and bench.is_short(rec(0, 0, 0, 0, 1, finish="stop")["reply"], 512),
          "short = fewer tokens than asked, or a finish reason other than length")


def staggered() -> list[dict]:
    """Four workers, two requests each, 100 tokens a request: the first requests end together at 10 s, the second ones at
    20, 22, 24 and 26 s; first tokens of the first requests at 0.5, 0.6, 0.7 and 0.8 s."""
    ends = [20.0, 22.0, 24.0, 26.0]
    out = []
    for w in range(4):
        out.append(rec(w, 0, 0.0, 0.5 + 0.1 * w, 10.0, tokens=100))
        out.append(rec(w, 1, 10.0, 10.0, ends[w], tokens=100))
    return out


def test_rolling_steady_window_runs_from_latest_first_token_to_earliest_last_end() -> None:
    recs = staggered()
    check(bench.steady_window(recs) == (0.8, 20.0), f"{bench.steady_window(recs)}")
    # a later first token of a worker's SECOND request must not move the start
    recs[1]["first_token"] = 10.5
    check(bench.steady_window(recs) == (0.8, 20.0), "only the first requests set the start")
    # and an earlier end of a worker's FIRST request must not move the end
    recs[0]["end"] = 3.0
    recs[0]["first_token"] = 0.5
    check(bench.steady_window(recs) == (0.8, 20.0), "only the last requests set the end")
    check(bench.steady_window([rec(0, 0, 0, 5.0, 6.0), rec(1, 0, 0, 1.0, 4.0)]) is None, "no overlap, no window")


def test_rolling_steady_rate_counts_tokens_inside_the_window() -> None:
    m, notes = bench.rolling_metrics(staggered())
    span = 26.0
    check(m["decode_c4_rolling_tok_s"] == round(800 / span, 3) and not notes, f"{m} {notes}")
    lo, hi = 0.8, 20.0
    inside = sum(100 * (min(10.0, hi) - max(f, lo)) / (10.0 - f) for f in (0.5, 0.6, 0.7, 0.8))     # first requests
    inside += sum(100 * (min(e, hi) - 10.0) / (e - 10.0) for e in (20.0, 22.0, 24.0, 26.0))        # second requests
    check(m["decode_c4_steady_tok_s"] == round(inside / (hi - lo), 3), f"{m['decode_c4_steady_tok_s']} vs {inside / (hi - lo)}")
    check(m["decode_c4_steady_tok_s"] > m["decode_c4_rolling_tok_s"], "the steady rate exceeds the span rate when the ends are staggered")


def test_rolling_steady_metric_is_omitted_under_ten_seconds() -> None:
    short = [rec(w, 0, 0.0, 0.5, 10.0 + w, tokens=100) for w in range(4)]            # window 0.5 .. 10.0: 9.5 s
    m, notes = bench.rolling_metrics(short)
    check("decode_c4_steady_tok_s" not in m and "9.50 s" in notes[0], f"{m} {notes}")
    ok = [rec(w, 0, 0.0, 0.5, 10.6 + w, tokens=100) for w in range(4)]               # window 0.5 .. 10.6: 10.1 s
    check("decode_c4_steady_tok_s" in bench.rolling_metrics(ok)[0], "10.1 s is reported")


def test_rolling_phase_raw_json_and_metrics_against_the_stub() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o.json")
        with stub_api.serve() as (url, _):
            rc, text, err = run_bench(url, "--raw", "--phases", "decode,rolling", "--out", out)
        check(rc == 0 and "METRIC decode_c4_rolling_requests=32" in text and "METRIC decode_c4_rolling_short_replies=0" in text, err)
        check("METRIC decode_c4_rolling_tok_s=" in text and "METRIC decode_c4_steady_tok_s=" not in text
              and "decode_c4_steady_tok_s omitted" in err, "a run under 10 s omits the steady metric and says so")
        r = json.load(open(out))["raw"]["rolling"]
        check(len(r["requests"]) == 32 and r["first_token_source"] == "solo_ttft_estimate", str(r["first_token_source"]))
        want = {"worker", "index", "kind", "start_s", "first_token_s", "end_s", "completion_tokens", "finish_reason", "spec",
                "generated_ids"}
        check(all(set(q) == want and len(q["generated_ids"]) == 512 for q in r["requests"]), "every request's fields and ids")
        check(min(q["start_s"] for q in r["requests"]) == 0.0, "times are seconds from the first start")


def test_rolling_short_reply_is_a_warning_when_raw_and_a_failure_in_a_gate_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "r.json")
        with stub_api.serve() as (url, _):
            rc, _, err = run_bench(url, "--pin", ref, "--phases", "decode,rolling")
            check(rc == 0, err)
        with stub_api.serve(bug="short") as (url, _):
            rc, text, err = run_bench(url, "--raw", "--phases", "rolling")
            check(rc == 0 and "WARNING rolling" in err and "FAIL" not in err and "METRIC decode_c4_rolling_short_replies=32" in text, f"raw: rc {rc}\n{err}")
            rc, text, err = run_bench(url, "--ref", ref, "--phases", "rolling")
            check(rc == 1 and "FAIL rolling w0#0 prose: finish stop, 507 tokens of 512" in err and text.rstrip().endswith("FAIL pin"),
                  f"a --ref run fails: rc {rc}\n{err}")


def test_rolling_reply_must_equal_the_solo_reply_of_its_kind() -> None:
    with stub_api.serve(bug="conc") as (url, _):
        rc, text, err = run_bench(url, "--raw", "--phases", "decode,rolling")
    check(rc == 1 and "rolling != solo" in err and "FAIL concurrent" in text, f"rc {rc}\n{err}")
    with stub_api.serve() as (url, _):
        rc, _, err = run_bench(url, "--raw", "--phases", "decode,rolling")
    check(rc == 0 and "rolling != solo" not in err, "a faithful server passes")


def test_rolling_pin_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "r.json")
        with stub_api.serve() as (url, _):
            rc, text, err = run_bench(url, "--pin", ref, "--phases", "decode,rolling", "--engine", "e" * 40)
            check(rc == 0, err)
            check(sorted(json.load(open(ref))["replies"]) == ["decode/code", "decode/json", "decode/prose", "decode/structured"],
                  "rolling replies are never pinned themselves: each was held to its solo entry")
            rc, text, err = run_bench(url, "--ref", ref, "--phases", "decode,rolling")
            check(rc == 0 and text.rstrip().endswith("PASS pin"), f"{err}")
            rc, text, err = run_bench(url, "--ref", ref, "--phases", "rolling")
            check(rc == 0 and text.rstrip().endswith("PASS pin"), f"rolling alone is held to the pinned solo entries: {err}")
        with stub_api.serve(bug="all") as (url, _):
            rc, text, err = run_bench(url, "--ref", ref, "--phases", "rolling")
        check(rc == 1 and text.rstrip().endswith("FAIL pin") and "rolling/" in err and "differ at 3" in err, f"{err}")
        with stub_api.serve() as (url, _):
            rc, _, err = run_bench(url, "--pin", os.path.join(tmp, "n.json"), "--phases", "rolling")
        check(rc == 1 and "needs the decode phase before rolling" in err and not os.path.exists(os.path.join(tmp, "n.json")),
              "a pin run of rolling alone has no solo reply to hold it to")


def test_rolling_is_not_a_default_phase_and_c8_says_what_it_is() -> None:
    check("rolling" in bench.PHASES and "rolling" not in bench.DEFAULT_PHASES.split(","), "named, not default")
    check("NOT an 8-stream batch" in bench.__doc__ and "SPAN number" in bench.__doc__, "the docstring says what c=8 and c=4 are")
    with stub_api.serve() as (url, _):
        rc, _, err = run_bench(url, "--raw", "--phases", "decode,conc")
    check("conc c=8" in err and "two waves, not an 8-stream batch" in err and "(span number)" in err, err)


# ── the memory gate ──
def memwatch(min_mib, guard="running") -> str:
    return (f"memwatch guard={guard} window=none window_min_mib=n/a loading_min_mib={min_mib + 100} ready_min_mib={min_mib + 400} "
            f"min_mib={min_mib} min_at=2026-11-22T05:49:02Z now_mib={min_mib + 500} updated=2026-11-22T05:59:35Z")


def plan(prefix: str) -> str:
    return f"rexl3 event=inkling2_plan slots=4 kv_tokens=1048576 lanes_at_max_context=4 prefix_gib={prefix} spare_gib=2.1"


def test_memcheck_floor_on_the_guard_minimum() -> None:
    p, m = memcheck.check(memwatch(9875))
    check(p == [] and m["mem_min_gib"] == 9.64 and m["mem_loading_min_gib"] == 9.74, f"{p} {m}")
    p, _ = memcheck.check(memwatch(8000))
    check(len(p) == 1 and "under the 8192 MiB floor" in p[0], str(p))
    p, _ = memcheck.check(memwatch(8000), floor_mib=4096)
    check(p == [], "the floor follows MEMORY_FLOOR_GIB")


def test_memcheck_needs_a_running_guard_and_a_report() -> None:
    p, _ = memcheck.check(memwatch(9875, guard="stopped"))
    check(len(p) == 1 and "not running" in p[0], str(p))
    p, _ = memcheck.check("no guard record\n")
    check(p == ["no `memwatch ...` line in the report"], str(p))
    p, _ = memcheck.check("memwatch guard=running window=none\n")
    check(p == ["no min_mib in the memwatch report"], str(p))


def test_memcheck_normalizes_to_the_recorded_prefix_cache() -> None:
    p, m = memcheck.check(memwatch(9000), plan("4.00"), 6.03)
    check(math.isclose(m["prefix_gib_recorded_gap"], 2.03) and m["mem_min_gib_norm"] == 6.76, f"{m}")
    check(len(p) == 1 and "recorded prefix cache" in p[0], f"a smaller cache measured more headroom: {p}")
    p, m = memcheck.check(memwatch(9000), plan("7.00"), 6.03)
    check(p == [] and m["prefix_gib_recorded_gap"] == 0.0, f"a larger cache is not credited: {p} {m}")
    p, _ = memcheck.check(memwatch(9000), "no plan here\n", 6.03)
    check(len(p) == 1 and "prefix_gib" in p[0], str(p))


def test_api_client_reads_the_node_from_the_environment_only() -> None:
    keys = ("NODES", "PORT", "INKLING2MINI_URL")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        os.environ["NODES"], os.environ["PORT"] = "node.example", "8123"
        check(ia.default_url() == "http://node.example:8123", ia.default_url())
        os.environ["INKLING2MINI_URL"] = "http://other.example:9"
        check(ia.default_url() == "http://other.example:9", "the URL variable wins")
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


# ── sampled.py: the arithmetic of the sampled, reasoning decode benchmark ──
def _row(name="chat-a", effort="high", t=1.0, tokens=101, rate=50.0, spec=None, reasoning=60, finish="stop", ids=None, cat="chat", **kw):
    ids = ids if ids is not None else [1, 2, 3]
    return {"category": cat, "name": name, "effort": effort, "temperature": t, "completion_tokens": tokens, "tok_s": rate,
            "spec": spec, "reasoning_tokens": reasoning, "finish_reason": finish, "ttft_s": 0.5, "ids": ids,
            "ids_sha256": sampled.ids_digest(ids), "start_offset_s": 0.0, "wall_s": 3.0, **kw}


def test_sampled_seed_and_rate_arithmetic() -> None:
    combos = [(i, e, t) for i in range(len(sampled.PROMPTS)) for e in sampled.EFFORTS for t in sampled.TEMPERATURES]
    seeds = [sampled.seed_for(*c) for c in combos]
    check(len(combos) == 64 and len(set(seeds)) == 64, "a seed per (prompt, effort, temperature), none shared")
    check(seeds == [sampled.seed_for(*c) for c in combos], "stable")
    check([p[0] for p in sampled.PROMPTS] == [c for c in sampled.CATEGORIES for _ in range(4)], "four prompts of each category in order")
    check(math.isclose(sampled.decode_rate(101, 1.0, 3.0), 50.0), "100 gaps over 2 s")
    check(sampled.decode_rate(1, 1.0, 3.0) is None and sampled.decode_rate(50, 2.0, 2.0) is None, "one token or one event: no rate")


def test_sampled_summary_with_and_without_spec() -> None:
    sp = lambda d, a, r: {"drafted": d, "accepted": a, "rounds": r}
    drafted = [_row(tokens=100, rate=60.0, spec=sp(200, 100, 50), reasoning=50), _row(tokens=300, rate=100.0, spec=sp(300, 240, 100), reasoning=150, finish="length")]
    s = sampled.summarize(drafted + [_row(error="boom")])
    check(s["requests"] == 3 and s["failed"] == 1 and s["hit_max_tokens"] == 1, f"{s}")
    check(math.isclose(s["tok_s_mean"], 80.0) and math.isclose(s["tok_s_median"], 80.0), "failed rows do not count")
    check(math.isclose(s["tokens_per_round"], 400 / 150) and math.isclose(s["acceptance"], 340 / 500), "counters pooled, not averaged")
    check(math.isclose(s["reasoning_share"], 0.5), f"{s['reasoning_share']}")
    serial = sampled.summarize([_row(rate=40.0), _row(rate=None, tokens=1, reasoning=None)])
    check(serial["tokens_per_round"] is None and serial["acceptance"] is None, "drafter off: no counters, no division")
    check(math.isclose(serial["tok_s_mean"], 40.0) and serial["failed"] == 0, "a one-token request has no rate but still counts")


def test_sampled_wave_aggregate_is_tokens_over_the_span() -> None:
    rows = [_row(tokens=100, rate=40.0, start_offset_s=0.0, wall_s=4.0), _row(tokens=300, rate=60.0, start_offset_s=0.1, wall_s=6.0),
            _row(tokens=50, rate=20.0, start_offset_s=0.2, wall_s=2.0), _row(error="x")]
    a = sampled.wave_aggregate(rows)
    check(a["answered"] == 3 and a["tokens"] == 450 and math.isclose(a["span_s"], 6.1), f"{a}")
    check(math.isclose(a["aggregate_tok_s"], 450 / 6.1) and math.isclose(a["median_tok_s"], 40.0), f"{a}")


class _FakeApi:
    """The stream of one chat completion: a role chunk, a drafted event of three ids, a plain one, the finish chunk, the usage chunk."""
    model = "m"

    def __init__(self, events, usage):
        self.events, self.usage, self.body = events, usage, None

    def stream(self, path, body):
        self.body = body
        yield from self.events
        if self.usage is not None:
            yield 9.0, {"choices": [], "usage": self.usage}


def _chunk(t, ids, reasoning="", content="", finish=None):
    d = {k: v for k, v in (("reasoning_content", reasoning), ("content", content)) if v}
    return t, {"choices": [{"delta": d, "token_ids": ids, "finish_reason": finish}]}


def test_sampled_request_timing_comes_from_token_events() -> None:
    ev = [(0.2, {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}),
          _chunk(1.0, [10, 11, 12], reasoning="think"), _chunk(2.0, [13], content="hi"), _chunk(3.0, [], content=" there"),
          (3.5, {"choices": [{"delta": {}, "finish_reason": "stop"}]})]
    usage = {"prompt_tokens": 7, "completion_tokens": 21, "completion_tokens_details": {"reasoning_tokens": 3}, "spec": {"drafted": 8, "accepted": 4, "rounds": 7}}
    r = sampled.one_request(_FakeApi(ev, usage), sampled.PROMPTS[0], 0, "low", 1.0)
    check(r["ttft_s"] == 1.0 and r["t_last_s"] == 3.0 and r["wall_s"] == 9.0, f"the role chunk is no token; {r['ttft_s']} {r['t_last_s']}")
    check(math.isclose(r["tok_s"], 20 / 2.0), f"(21 - 1) / (3.0 - 1.0): {r['tok_s']}")
    check(r["ids"] == [10, 11, 12, 13] and r["reasoning_tokens"] == 3 and r["finish_reason"] == "stop", f"{r['ids']}")
    check(r["spec"]["tokens_per_round"] == 3.0 and r["reasoning_text"] == "think" and r["content_text"] == "hi there", "counters and text")
    api = _FakeApi(ev, None)
    check(sampled.one_request(api, sampled.PROMPTS[0], 0, "high", 0.0)["error"].endswith("usage"), "no usage chunk is an error row")
    b = api.body
    check(b["seed"] == sampled.seed_for(0, "high", 0.0) and b["max_tokens"] == 2048 and b["prefix_cache"] is False and b["temperature"] == 0.0, f"{b}")


def test_sampled_compare_counts_identical_replies_per_temperature() -> None:
    def run(t1_ids):
        rows = [_row(name=f"p{i}", t=0.0, ids=[i, 1, 2]) for i in range(3)] + [_row(name=f"p{i}", t=1.0, ids=t1_ids[i]) for i in range(3)]
        return {"c1": rows}
    a = run([[0, 5], [1, 5], [2, 5, 6]])
    b = run([[0, 5], [1, 9], [2, 5]])
    c = sampled.compare_runs(a, b)
    check(c["exact"][0.0]["identical"] == 3 and c["exact"][0.0]["pairs"] == 3 and c["exact"][0.0]["first_difference"] is None, f"{c['exact']}")
    e = c["exact"][1.0]
    check(e["identical"] == 1 and e["pairs"] == 3, f"{e}")
    check(e["first_difference"]["request"] == "p1|high|1.0" and e["first_difference"]["first_differing_token"] == 1, f"{e}")
    check(sampled.compare_runs(a, {"c1": b["c1"][:2]})["exact"][1.0]["pairs"] == 0, "only requests present in both runs are paired")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}", file=sys.stderr)
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"test_bench: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
