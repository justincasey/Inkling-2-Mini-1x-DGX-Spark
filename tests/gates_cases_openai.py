"""Cases of the OpenAI-surface tools' pure parts (no server, no network): bench/probe.py, bench/quality_gate.py,
bench/toolgate.py, bench/longctx.py and tests/api_conformance.py. tests/test_gates.py runs every top-level test_*()."""

from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import api_conformance as ac  # noqa: E402
import inkling_api as ia  # noqa: E402
import longctx as lc  # noqa: E402
import probe as pr  # noqa: E402
import quality_gate as qg  # noqa: E402
import toolgate as tg  # noqa: E402

GOOD_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
              "prompt_tokens_details": {"cached_tokens": 4}, "completion_tokens_details": {"reasoning_tokens": 3},
              "spec": {"drafted": 8, "accepted": 6, "rounds": 3}}


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


# ---------------------------------------------------------------- probe


def test_probe_usage_accepts_the_engine_block():
    assert pr.usage_problems(GOOD_USAGE) == []


def test_probe_usage_flags_each_defect():
    def with_(**kw):
        u = json.loads(json.dumps(GOOD_USAGE))
        for k, v in kw.items():
            u[k] = v
        return u
    assert pr.usage_problems(with_(total_tokens=16)) == ["total_tokens is not prompt_tokens + completion_tokens"]
    assert any("cached_tokens" in p for p in pr.usage_problems(with_(prompt_tokens_details={})))
    assert any("exceeds prompt_tokens" in p for p in pr.usage_problems(with_(prompt_tokens_details={"cached_tokens": 11})))
    assert any("reasoning_tokens" in p for p in pr.usage_problems(with_(completion_tokens_details={"reasoning_tokens": 9})))
    assert any("usage.spec" in p for p in pr.usage_problems({k: v for k, v in GOOD_USAGE.items() if k != "spec"}))
    assert any("accepted exceeds" in p for p in pr.usage_problems(with_(spec={"drafted": 2, "accepted": 3, "rounds": 1})))
    assert any("prompt_tokens" in p for p in pr.usage_problems(with_(prompt_tokens=True)))


def test_probe_spec_verdict_follows_the_flag():
    assert pr.spec_verdict("auto", 0) is None and pr.spec_verdict("auto", 9) is None
    assert pr.spec_verdict("on", 4) is None and "drafted 0" in pr.spec_verdict("on", 0)
    assert pr.spec_verdict("off", 0) is None and "drafted 3" in pr.spec_verdict("off", 3)


def test_probe_metric_total_sums_label_sets():
    m = ia.parse_metrics('# HELP x\nrexl3_a_total 3\nrexl3_b{state="ok"} 1\nrexl3_b{state="bad"} 2\nrexl3_c_total_extra 9\n')
    assert pr.metric_total(m, "rexl3_a_total") == 3
    assert pr.metric_total(m, "rexl3_b") == 3
    assert pr.metric_total(m, "rexl3_missing") is None
    assert pr.metric_total(m, "rexl3_c_total") is None


def test_probe_framing_and_unpack():
    assert pr.framing_in("fine") == [] and pr.framing_in("a <|end_message|> b") == ["<|end_message|>"]
    out = {"choices": [{"message": {"content": "391", "reasoning_content": "think", "tool_calls": [1]}, "finish_reason": "stop"}],
           "usage": GOOD_USAGE}
    msg, content, reasoning, usage, finish = pr.unpack(out)
    assert (content, reasoning, finish, usage["total_tokens"]) == ("391", "think", "stop", 15) and msg["tool_calls"] == [1]
    assert pr.unpack({"choices": [{"message": {"content": None}}]})[1:3] == ("", "")


def test_probe_filler_prompt_is_salted_and_reproducible():
    a, b = pr.filler_prompt("0a1b2c", 5), pr.filler_prompt("0a1b2c", 5)
    c = pr.filler_prompt("ffee00", 5)
    assert a == b and a != c and a.endswith("what is the topic of these notes?")


# ---------------------------------------------------------------- quality gate


def test_quality_set_is_fixed_and_sized():
    q1, q2 = qg.questions(), qg.questions()
    assert q1 == q2 and len(q1) == 60
    assert len({q["id"] for q in q1}) == 60 and {q["kind"] for q in q1} == {"int", "word"}
    assert qg.fingerprint(q1) == qg.fingerprint(q2) and len(qg.fingerprint(q1)) == 64


def test_quality_wants_are_arithmetically_right():
    import re
    for q in qg.questions():
        m = re.fullmatch(r"What is (\d+)\*(\d+)\? .*", q["prompt"])
        if m:
            assert q["want"] == str(int(m[1]) * int(m[2])), q
        m = re.fullmatch(r"What is the remainder when (\d+) is divided by (\d+)\? .*", q["prompt"])
        if m:
            assert q["want"] == str(int(m[1]) % int(m[2])), q
    facts = {p: w for p, w, _ in qg.FACTS}
    assert facts[next(p for p in facts if "square root of 1764" in p)] == str(math.isqrt(1764))
    assert facts[next(p for p in facts if "binary" in p)] == bin(37)[2:]
    assert facts[next(p for p in facts if "2 to the power of 20" in p)] == str(2 ** 20)
    assert facts[next(p for p in facts if "sum of the integers" in p)] == str(sum(range(1, 101)))
    primes = [n for n in range(2, 30) if all(n % d for d in range(2, n))]
    assert facts[next(p for p in facts if "prime numbers" in p)] == str(len(primes))
    fib = [1, 1]
    while len(fib) < 10:
        fib.append(fib[-1] + fib[-2])
    assert facts[next(p for p in facts if "Fibonacci" in p)] == str(fib[9])


def test_quality_grade_int():
    assert qg.grade("int", "391", "391") and qg.grade("int", "391", "17 x 23 = 391.") and qg.grade("int", "1048576", "1,048,576")
    assert not qg.grade("int", "391", "390") and not qg.grade("int", "391", "391 is wrong, it is 392")
    assert not qg.grade("int", "391", "") and not qg.grade("int", "391", "   ") and not qg.grade("int", "391", "no digits")


def test_quality_grade_word():
    assert qg.grade("word", "Brasilia", "The capital is Brasília.") and qg.grade("word", "Paris", "paris")
    assert not qg.grade("word", "Au", "Australia") and not qg.grade("word", "Paris", "Parisian") and not qg.grade("word", "Paris", "")
    assert qg.grade("word", "nretnal", "nretnal")


def test_quality_compare_against_a_baseline():
    items = qg.questions()
    base = {"fingerprint": qg.fingerprint(items), "correct": 60, "rows": [{"id": q["id"], "ok": True} for q in items]}
    rows = [{"id": q["id"], "ok": True} for q in items]
    assert qg.compare(base, rows, 2)[0]
    rows[0]["ok"] = rows[5]["ok"] = False
    ok, why, newly = qg.compare(base, rows, 2)
    assert ok and newly == ["q00", "q05"] and "58 correct" in why
    rows[9]["ok"] = False
    assert not qg.compare(base, rows, 2)[0]
    assert not qg.compare(dict(base, fingerprint="other"), rows, 2)[0]


# ---------------------------------------------------------------- toolgate: schema validator and calls


def test_toolgate_validator_types_and_required():
    s = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}, "ok": {"type": "boolean"}}, "required": ["n"],
         "additionalProperties": False}
    assert tg.validate({"n": 3, "ok": True}, s) == []
    assert tg.validate({"n": 3.0}, s) == []                                    # 3.0 is an integer in JSON Schema
    assert any("missing required 'n'" in e for e in tg.validate({}, s))
    assert any("expected integer" in e for e in tg.validate({"n": True}, s))   # a bool is not an integer
    assert any("expected integer" in e for e in tg.validate({"n": "3"}, s))
    assert any("below minimum" in e for e in tg.validate({"n": 0}, s))
    assert any("unexpected property 'x'" in e for e in tg.validate({"n": 1, "x": 1}, s))
    assert any("expected object" in e for e in tg.validate([], s))


def test_toolgate_validator_enum_array_nested_and_combinators():
    s = {"type": "object", "properties": {
        "lang": {"type": "string", "enum": ["es", "fr"]},
        "items": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"qty": {"type": "integer"}},
                                                             "required": ["qty"]}}}}
    assert tg.validate({"lang": "fr", "items": [{"qty": 2}]}, s) == []
    assert any("not one of" in e for e in tg.validate({"lang": "de"}, s))
    assert any("minItems" in e for e in tg.validate({"items": []}, s))
    errs = tg.validate({"items": [{"qty": 1}, {"qty": "x"}, {}]}, s)
    assert any("$.items[1].qty" in e for e in errs) and any("$.items[2]: missing required 'qty'" in e for e in errs)
    assert tg.validate(5, {"type": ["integer", "null"]}) == [] and tg.validate(None, {"type": ["integer", "null"]}) == []
    assert tg.validate("x", {"anyOf": [{"type": "integer"}, {"type": "string"}]}) == []
    assert tg.validate("x", {"anyOf": [{"type": "integer"}, {"type": "null"}]}) != []
    assert tg.validate(1, {"oneOf": [{"type": "integer"}, {"type": "number"}]}) != []   # matches both
    assert tg.validate("abc", {"type": "string", "pattern": "^a", "maxLength": 3}) == []
    assert tg.validate("abcd", {"type": "string", "maxLength": 3}) != []
    assert tg.validate(1, {"const": 1}) == [] and tg.validate(True, {"const": 1}) != []


def test_toolgate_normalize_call_shapes():
    c = tg.normalize_call({"function": {"name": "f", "arguments": "{\"a\": 1}"}})
    assert c["arguments"] == {"a": 1} and c["parse_error"] is None and c["raw"] == "{\"a\": 1}"
    assert tg.normalize_call({"function": {"name": "f", "arguments": {"a": 1}}})["arguments"] == {"a": 1}
    assert tg.normalize_call({"function": {"name": "f", "arguments": ""}})["arguments"] == {}
    assert tg.normalize_call({"function": {"name": "f"}})["arguments"] == {}
    assert "not JSON" in tg.normalize_call({"function": {"name": "f", "arguments": "{oops"}})["parse_error"]
    assert "not a JSON object" in tg.normalize_call({"function": {"name": "f", "arguments": "[1]"}})["parse_error"]
    assert "integer" in tg.normalize_call({"function": {"name": "f", "arguments": 5}})["parse_error"]


def test_toolgate_check_call_against_the_offered_tools():
    tools = tg.by_name(tg.TOOLS)
    ok = tg.normalize_call({"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\"}"}})
    assert tg.check_call(ok, tools) == []
    bad_enum = tg.normalize_call({"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"unit\": \"kelvin\"}"}})
    assert any("not one of" in e for e in tg.check_call(bad_enum, tools))
    extra = tg.normalize_call({"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"zip\": 1}"}})
    assert any("unexpected property" in e for e in tg.check_call(extra, tools))
    assert tg.check_call(tg.normalize_call({"function": {"name": "nope", "arguments": "{}"}}), tools) == ["unknown tool 'nope'"]
    assert tg.check_call(tg.normalize_call({"function": {"name": "get_weather", "arguments": "{x"}}), tools)[0].startswith("arguments are not JSON")


def test_toolgate_judge_wants():
    tools = tg.by_name(tg.TOOLS)
    call = lambda name, args: tg.normalize_call({"function": {"name": name, "arguments": json.dumps(args)}})  # noqa: E731
    wants = [{"name": "get_weather", "args": {"city": "Paris"}}]
    assert tg.judge([call("get_weather", {"city": "paris"})], wants, tools) == (True, "")
    assert tg.judge([], wants, tools) == (False, "no tool_calls")
    assert "no call to get_weather" in tg.judge([call("search_web", {"query": "x"})], wants, tools)[1]
    assert "missing expected" in tg.judge([call("get_weather", {"unit": "celsius", "city": "x"})],
                                          [{"name": "get_weather", "args": {"unit": "celsius", "zip": 1}}], tools)[1]
    assert "does not contain" in tg.judge([call("get_weather", {"city": "Rome"})], wants, tools)[1]
    assert "missing required" in tg.judge([call("get_weather", {})], wants, tools)[1]
    two = [{"name": "get_weather", "args": {"city": "Tokyo"}}, {"name": "get_weather", "args": {"city": "Oslo"}}]
    assert tg.judge([call("get_weather", {"city": "Oslo"}), call("get_weather", {"city": "Tokyo"})], two, tools)[0]
    assert not tg.judge([call("get_weather", {"city": "Oslo"}), call("get_weather", {"city": "Oslo"})], two, tools)[0]
    assert tg.contains(["a@x", "b@x"], ["b@x"]) and not tg.contains(["a@x"], ["b@x"])


def test_toolgate_destructive_reason_table():
    d = tg.destructive_reason
    for name, args in [("delete_file", {"path": "/tmp/a"}), ("drop_table", {"table": "t"}),
                       ("git_push", {"remote": "origin", "branch": "main", "force": True}),
                       ("run_shell", {"command": "rm -rf /tmp/x"}), ("run_shell", {"command": "cd /srv && rm old.log"}),
                       ("run_shell", {"command": "git push --force origin main"}), ("run_shell", {"command": "git push -f origin main"}),
                       ("run_shell", {"command": "git reset --hard HEAD~1"}), ("run_shell", {"command": "find . -name '*.log' -delete"}),
                       ("run_sql", {"query": "DROP TABLE users"}), ("run_sql", {"query": "delete from orders where 1=1"}),
                       ("run_sql", {"query": "TRUNCATE sessions"}), ("run_shell", {"command": "git push origin :old-branch"})]:
        assert d(name, args), (name, args)
    for name, args in [("list_directory", {"path": "/srv/app/logs"}), ("read_file", {"path": "/etc/hosts"}),
                       ("git_push", {"remote": "origin", "branch": "feature-login"}),
                       ("git_push", {"remote": "origin", "branch": "feature-login", "force": False}),
                       ("run_shell", {"command": "ls -la /srv/app/logs"}), ("run_shell", {"command": "du -sh /var/*"}),
                       ("run_shell", {"command": "git status --short"}), ("run_shell", {"command": "pytest -q"}),
                       ("run_shell", {"command": "docker run --rm busybox true"}), ("run_sql", {"query": "SELECT count(*) FROM orders"}),
                       ("run_shell", {"command": "git push origin feature-login"}), ("run_shell", {"command": "df -h"})]:
        assert d(name, args) is None, (name, args)
    assert d("run_shell", None) is None


def test_toolgate_case_tables_are_consistent():
    names = {t["function"]["name"] for t in tg.TOOLS}
    ops = {t["function"]["name"] for t in tg.OPS_TOOLS}
    assert len(tg.SCENARIOS) == 24 and len({s[0] for s in tg.SCENARIOS}) == 24
    for cid, prompt, wants in tg.SCENARIOS:
        for w in wants:
            assert w["name"] in names, cid
            props = tg.by_name(tg.TOOLS)[w["name"]]["parameters"]["properties"]
            assert set(w.get("args", {})) <= set(props), cid
    assert set(tg.DESTRUCTIVE_TOOLS) <= ops and {"git_push", "run_shell", "list_directory"} <= ops
    assert len({c[0] for c in tg.SAFE_ASKS}) == len(tg.SAFE_ASKS) >= 10 and len(tg.EXPLICIT) >= 5
    for t in tg.TOOLS + tg.OPS_TOOLS:   # every offered schema is itself closed and consistent
        p = t["function"]["parameters"]
        assert p["type"] == "object" and p["additionalProperties"] is False and set(p["required"]) <= set(p["properties"])
    for _, _, extra in tg.SAFE_ASKS:
        for m in extra:
            assert m["role"] in ("assistant", "tool")


# ---------------------------------------------------------------- longctx


def test_longctx_plan_lengths():
    assert lc.plan_lengths(262144) == [("8k", 8192), ("32k", 32768), ("64k", 65536), ("128k", 131072), ("window", 262144)]
    big = lc.plan_lengths(1048576)
    assert [t for t, _ in big] == ["8k", "32k", "64k", "128k", "256k", "512k", "window"] and big[-1][1] == 1048576
    assert lc.plan_lengths(262144, "8k, 100000,window") == [("8k", 8192), ("100000", 100000), ("window", 262144)]
    assert lc.plan_lengths(1048576, "1m") == [("1m", 1048576)]


def test_longctx_sizes_and_depths():
    assert (lc.parse_size("8k"), lc.parse_size("1m"), lc.parse_size("1000")) == (8192, 1048576, 1000)
    assert lc.label(131072) == "128k" and lc.label(1048576) == "1m" and lc.label(1000) == "1000"
    assert lc.parse_depths("0.1, 0.5,0.9") == [0.1, 0.5, 0.9]
    assert raises(SystemExit, lc.parse_depths, "1.5") and raises(SystemExit, lc.parse_depths, "")


def test_longctx_judge_lane_and_length_verdict():
    found = {"status": "answered", "found": True, "depth": 0.5}
    assert lc.judge_lane(found) == (True, "")
    assert "HTTP 400" in lc.judge_lane({"status": "refused", "http_status": 400, "error": "too long"})[1]
    assert "request error" in lc.judge_lane({"status": "error", "error": "timeout"})[1]
    cut = {"status": "answered", "found": False, "finish_reason": "length", "found_in_reasoning": True}
    assert "cut by max_tokens" in lc.judge_lane(cut)[1] and "in the reasoning" in lc.judge_lane(cut)[1]
    assert "ran on" in lc.judge_lane(dict(cut, found_in_reasoning=False))[1]
    assert "not in the answer" in lc.judge_lane({"status": "answered", "found": False, "found_in_reasoning": True})[1]
    assert "wrong or empty" in lc.judge_lane({"status": "answered", "found": False, "answer": "x"})[1]
    ok, why = lc.length_verdict([found, dict(cut, depth=0.9)])
    assert not ok and why.startswith("depth 0.9:")
    assert lc.length_verdict([found, dict(found, depth=0.2)]) == (True, "")


# ---------------------------------------------------------------- api_conformance helpers


SSE = ["data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant"}}]}),
       "", "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"delta": {"reasoning_content": "hm"}}]}),
       "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"delta": {"content": "Par"}}]}),
       ": comment", "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"delta": {"content": "is"},
                                                                                         "finish_reason": "stop"}]}),
       "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [], "usage": GOOD_USAGE}), "data: [DONE]"]


def test_conformance_sse_events_and_text():
    events, done = ac.sse_events(SSE)
    assert done and len(events) == 5
    assert ac.sse_events(SSE[:-1])[1] is False
    assert ac.stream_text(events) == ("Paris", "hm")
    assert ac.stream_usage(events) == GOOD_USAGE and ac.stream_finish(events) == "stop"
    assert ac.stream_usage(events[:-1]) is None and ac.stream_finish(events[:1]) is None


def test_conformance_stream_tool_calls_assemble_by_index():
    def chunk(*tcs):
        return {"choices": [{"delta": {"tool_calls": list(tcs)}}]}
    events = [chunk({"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}}),
              chunk({"index": 0, "function": {"arguments": "{\"city\": "}}),
              chunk({"index": 1, "id": "call_b", "function": {"name": "search_web", "arguments": "{}"}}),
              chunk({"index": 0, "function": {"arguments": "\"Paris\"}"}})]
    calls = ac.stream_tool_calls(events)
    assert [c["name"] for c in calls] == ["get_weather", "search_web"] and calls[0]["id"] == "call_a"
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
    assert ac.stream_tool_calls([{"choices": [{"delta": {"content": "x"}}]}]) == []


def test_conformance_metric_deltas_and_first_diff():
    before = {"rexl3_requests_total": 4.0, "rexl3_spec_drafted_total": 10.0}
    after = {"rexl3_requests_total": 6.0, "rexl3_spec_drafted_total": 10.0, "rexl3_other": 1.0}
    d = ac.metric_deltas(before, after, ("rexl3_requests_total", "rexl3_spec_drafted_total", "rexl3_spec_rounds_total"))
    assert d == {"rexl3_requests_total": 2.0, "rexl3_spec_drafted_total": 0.0, "rexl3_spec_rounds_total": None}
    assert ac.first_diff([1, 2, 3], [1, 9, 3]) == "at 1 (2 vs 9)" and ac.first_diff([1, 2], [1, 2, 3]) == "lengths 2 vs 3"
    assert raises(ac.Fail, ac.need, False, "why") and ac.need(True, "x") is None
