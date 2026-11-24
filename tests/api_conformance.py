#!/usr/bin/env python3
"""API conformance of the node's OpenAI-compatible server (stdlib only; one server process, TP1).

Prints one `PASS name: detail` or `FAIL name: why` line per check and, as the LAST line,
`api_conformance: N PASS, F FAIL`; the gate is F = 0 and exit status 0 (exit 1 on any FAIL). Every request is greedy
(temperature 0) except where a check names its own
settings. The server refuses request fields it does not know, so each check sends only fields this server documents.

  health          /health and /health/live answer 200
  models          /v1/models: list shape, the served id (--model must be it), integer max_model_len <= configured_max_model_len;
                  with --window N, max_model_len >= N
  tokenize        /tokenize: integer ids, count == len(ids), the same ids on a repeat and for the `prompt` alias
  chat            a reply with finish_reason, a message and a usage block that adds up (prompt + completion = total, integer
                  cached_tokens and reasoning_tokens, a {drafted, accepted, rounds} spec block)
  render          usage.prompt_tokens equals /v1/chat/completions/render's count for the same messages
  max_tokens      max_tokens and max_completion_tokens cut at the limit with finish_reason length
  stop            a stop string does not appear in the visible reply (reasoning may mention it) and ends it
  streaming       SSE chunks, a final usage chunk (stream_options.include_usage), data: [DONE]
  reasoning       reasoning_effort "high": reasoning in reasoning_content (not content), reasoning_tokens > 0, no framing
                  marker (inkling_api.FRAMING) anywhere; "none" and the system line `Thinking effort level: 0` are accepted
  tools           tool_calls (id, name, JSON-string arguments), finish_reason tool_calls, streamed tool_calls assemble to
                  the same call, tool_choice "none" gives no call, a tool result is answered from
  completions     /v1/completions text, finish_reason, usage
  errors          unknown model, malformed JSON, missing messages, unknown field: HTTP 4xx; a max_tokens past the window: 4xx, or
                  (this server's contract) 200 clamped to the window (prompt + completion <= max_model_len, finish stop or length)
  metrics         /metrics counts the requests made here and reports rexl3_max_model_len equal to /v1/models
  spec            usage.spec against the rexl3_spec_*_total deltas; "draft": false drafts 0 tokens; --spec on needs drafted > 0,
                  --spec off needs drafted == 0 (default auto: either)
  prefix cache    "prefix_cache": false reports 0 cached tokens; with the cache on, a repeated prompt is served from it
  concurrency     two simultaneous requests both answer 200 (the server queues)
  determinism     the same greedy request twice gives the same token ids

    api_conformance.py [--url URL] [--model NAME] [--window 262144] [--spec auto|on|off]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "tools"))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import inkling_api as ia  # noqa: E402
import probe  # noqa: E402

TOOL = probe.TOOL
CHAT = "/v1/chat/completions"


class Fail(Exception):
    pass


def need(cond, why: str) -> None:
    if not cond:
        raise Fail(why)


def sse_events(lines: list[str]) -> tuple[list[dict], bool]:
    """(the JSON events of the `data:` lines, whether `data: [DONE]` was seen)."""
    events, done = [], False
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        events.append(json.loads(data))
    return events, done


def stream_text(events: list[dict]) -> tuple[str, str]:
    """(content, reasoning_content) concatenated over the streamed deltas."""
    c, r = [], []
    for ev in events:
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            c.append(d.get("content") or "")
            r.append(d.get("reasoning_content") or "")
    return "".join(c), "".join(r)


def stream_usage(events: list[dict]) -> dict | None:
    return next((e["usage"] for e in reversed(events) if e.get("usage")), None)


def stream_finish(events: list[dict]) -> str | None:
    return next((ch["finish_reason"] for e in reversed(events) for ch in e.get("choices") or [] if ch.get("finish_reason")), None)


def stream_tool_calls(events: list[dict]) -> list[dict]:
    """The tool calls assembled from streamed `delta.tool_calls` (by index): [{id, name, arguments (the joined string)}]."""
    calls: dict[int, dict] = {}
    for ev in events:
        for ch in ev.get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                c = calls.setdefault(int(tc.get("index", 0)), {"id": None, "name": "", "arguments": ""})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                c["arguments"] += fn.get("arguments") or ""
    return [calls[i] for i in sorted(calls)]


def metric_deltas(before: dict[str, float], after: dict[str, float], names) -> dict[str, float | None]:
    return {n: (None if probe.metric_total(after, n) is None else probe.metric_total(after, n) - (probe.metric_total(before, n) or 0))
            for n in names}


def first_diff(a: list, b: list) -> str:
    """Where two id lists first differ (for a failure message)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"at {i} ({x} vs {y})"
    return f"lengths {len(a)} vs {len(b)}"


class Suite:
    def __init__(self, api: ia.Api, a) -> None:
        self.api, self.a = api, a
        self.passed = self.failed = 0
        self.models: dict = {}
        self.cache_on = False

    def run(self, name: str, fn) -> None:
        try:
            detail = fn()
        except Fail as e:
            self.failed += 1
            print(f"FAIL {name}: {e}", flush=True)
        except (ia.ApiError, OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            self.failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        else:
            self.passed += 1
            print(f"PASS {name}" + (f": {detail}" if detail else ""), flush=True)

    # -- requests
    def chat(self, text: str, max_tokens: int = 256, **extra) -> dict:
        status, out, _ = probe.chat(self.api, text, max_tokens, **extra)
        need(status == 200 and isinstance(out, dict) and out.get("choices"), f"HTTP {status}: {str(out)[:300]}")
        return out


def raw_post(api: ia.Api, path: str, raw: bytes) -> int:
    """The HTTP status of a POST with a raw (possibly malformed) body, through the client's headers."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(api.url + path, data=raw, method="POST")
    req.add_header("Content-Type", "application/json")
    if api.api_key:
        req.add_header("Authorization", f"Bearer {api.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def raw_sse(api: ia.Api, path: str, body: dict) -> list[str]:
    with api._request("POST", path, dict(body, stream=True)) as r:
        return [line.decode("utf-8", "replace") for line in r]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--window", type=int, help="the profile's context window (262144 or 1048576)")
    ap.add_argument("--spec", choices=("auto", "on", "off"), default="auto", help="the drafter's expected state (default: auto)")
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    s = Suite(api, a)
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]

    def health():
        for path in ("/health", "/health/live"):
            status, out, _ = api.call("GET", path, timeout=30)
            need(status == 200, f"{path}: HTTP {status}: {str(out)[:200]}")
        return "200"

    def models():
        status, out, _ = api.call("GET", "/v1/models", timeout=30)
        need(status == 200 and isinstance(out, dict) and out.get("object") == "list" and out.get("data"), f"HTTP {status}: {str(out)[:200]}")
        row = out["data"][0]
        s.models = row
        ids = [r["id"] for r in out["data"]]
        need(api.model in ids, f"{api.model!r} is not served: {ids}")
        ml, cfg = row.get("max_model_len"), row.get("configured_max_model_len")
        need(isinstance(ml, int) and ml > 0, f"max_model_len {ml!r}")
        need(cfg is None or (isinstance(cfg, int) and ml <= cfg), f"max_model_len {ml} exceeds configured {cfg}")
        if a.window is not None:
            need(ml >= a.window, f"max_model_len {ml} is below the profile window {a.window} (kv_pool_tokens {row.get('kv_pool_tokens')})")
        return f"{ids} max_model_len={ml} configured={cfg}"

    def tokenize():
        t1, t2 = api.tokenize("hello world"), api.tokenize("hello world")
        status, out, _ = api.call("POST", "/tokenize", {"model": api.model, "prompt": "hello world"})
        need(t1 and all(isinstance(t, int) for t in t1), f"ids {t1}")
        need(t1 == t2, "two /tokenize calls differ")
        need(status == 200 and out.get("tokens") == t1 and out.get("count") == len(t1), f"prompt alias or count: {out}")
        return f"{len(t1)} tokens"

    def chat_usage():
        out = s.chat(msgs[0]["content"], 512, reasoning_effort="none")
        ch = out["choices"][0]
        need(ch.get("finish_reason") in ("stop", "length"), f"finish_reason {ch.get('finish_reason')!r}")
        need(isinstance(ch.get("message"), dict) and ch["message"].get("role") == "assistant", "no assistant message")
        problems = probe.usage_problems(out.get("usage") or {})
        need(not problems, "; ".join(problems))
        need(out["usage"]["completion_tokens"] > 0, "completion_tokens is 0")
        return json.dumps(out["usage"])

    def render():
        status, out, _ = api.call("POST", CHAT + "/render", {"model": api.model, "messages": msgs, "reasoning_effort": "none"})
        need(status == 200 and isinstance(out, dict) and isinstance(out.get("count"), int), f"HTTP {status}: {str(out)[:200]}")
        reply = s.chat(msgs[0]["content"], 8, reasoning_effort="none")
        need(reply["usage"]["prompt_tokens"] == out["count"], f"usage.prompt_tokens {reply['usage']['prompt_tokens']} != render count {out['count']}")
        return f"{out['count']} prompt tokens"

    def max_tokens():
        text = "Count from 1 to 300, separated by single spaces, and write nothing else."
        for field in ("max_tokens", "max_completion_tokens"):
            status, out, _ = api.call("POST", CHAT, {"model": api.model, "messages": [{"role": "user", "content": text}],
                                                      field: 16, "temperature": 0, "reasoning_effort": "none"})
            need(status == 200, f"{field}: HTTP {status}: {str(out)[:200]}")
            need(out["usage"]["completion_tokens"] <= 16, f"{field}: completion_tokens {out['usage']['completion_tokens']} > 16")
            need(out["choices"][0]["finish_reason"] == "length", f"{field}: finish_reason {out['choices'][0]['finish_reason']!r}")
        return "16 tokens, finish length"

    def stop():
        out = s.chat("Write the five words ALPHA BRAVO CHARLIE DELTA ECHO in that order, separated by spaces, and nothing else.",
                     512, reasoning_effort="none", stop=["CHARLIE"])
        _, content, reasoning, _, finish = probe.unpack(out)
        # the stop string ends and is cut from the visible reply; the model's own reasoning may legitimately mention it
        need("CHARLIE" not in content, f"the stop string is in the reply: {content[:80]!r}")
        need(finish == "stop", f"finish_reason {finish!r}")
        return f"content {content[:40]!r}"

    def streaming():
        body = {"model": api.model, "messages": msgs, "max_tokens": 256, "temperature": 0, "reasoning_effort": "none",
                "stream_options": {"include_usage": True}}
        events, done = sse_events(raw_sse(api, CHAT, body))
        content, reasoning = stream_text(events)
        usage = stream_usage(events)
        need(events and all(e.get("object") == "chat.completion.chunk" for e in events), "events are not chat.completion.chunk")
        need(content or reasoning, "no content or reasoning deltas")
        need(done, "no data: [DONE]")
        need(usage and not probe.usage_problems(usage), f"final usage chunk: {usage}; {probe.usage_problems(usage or {})}")
        need(stream_finish(events) in ("stop", "length"), f"streamed finish_reason {stream_finish(events)!r}")
        return f"{len(events)} chunks, completion_tokens={usage['completion_tokens']}"

    def reasoning():
        out = s.chat("A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?", 4096,
                     reasoning_effort="high")
        _, content, think, usage, finish = probe.unpack(out)
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        need(finish != "length", f"cut by max_tokens after {usage.get('completion_tokens')} tokens ({rt} reasoning): a reasoning loop?")
        need(think.strip(), "empty reasoning_content")
        need(content.strip(), "empty content")
        need(isinstance(rt, int) and 0 < rt <= usage["completion_tokens"], f"reasoning_tokens {rt!r}")
        need(not probe.framing_in(content) and not probe.framing_in(think), "a framing marker in the reply")
        need(think.strip() not in content or len(think.strip()) < 16, "the reasoning text is inside the content")
        for label, system, extra in (("none", None, {"reasoning_effort": "none"}), ("level 0", probe.LEVEL0, {})):
            o = s.chat("Say OK.", 1024, system=system, **extra)
            _, c, r, _, _ = probe.unpack(o)
            need((c.strip() or r.strip()) and not probe.framing_in(c + r), f"effort {label}: empty reply or framing marker")
        return f"reasoning_tokens={rt}, {len(think)} chars of reasoning"

    def tools():
        body = {"messages": [{"role": "user", "content": "What is the weather like in Paris right now? Use the get_weather tool."}],
                "tools": [TOOL], "tool_choice": "auto", "reasoning_effort": "none"}
        out = s.chat(body["messages"][0]["content"], 1024, tools=[TOOL], tool_choice="auto", reasoning_effort="none")
        msg, _, _, _, finish = probe.unpack(out)
        calls = msg.get("tool_calls") or []
        need(calls, f"no tool_calls: {str(msg)[:200]}")
        c = calls[0]
        need(isinstance(c.get("id"), str) and c["id"] and c.get("type") == "function", f"call id/type: {c}")
        fn = c.get("function") or {}
        need(fn.get("name") == "get_weather" and isinstance(fn.get("arguments"), str), f"function: {fn}")
        args = json.loads(fn["arguments"])
        need(isinstance(args.get("city"), str) and "paris" in args["city"].lower(), f"arguments {args}")
        need(finish == "tool_calls", f"finish_reason {finish!r}, want tool_calls")
        events, done = sse_events(raw_sse(api, CHAT, {**body, "model": api.model, "max_tokens": 1024, "temperature": 0,
                                                       "stream_options": {"include_usage": True}}))
        streamed = stream_tool_calls(events)
        need(done and streamed, "no streamed tool_calls or no [DONE]")
        need(streamed[0]["name"] == "get_weather" and streamed[0]["id"], f"streamed call {streamed[0]}")
        need(json.loads(streamed[0]["arguments"]).get("city", "").lower() == args["city"].lower(),
             f"streamed arguments {streamed[0]['arguments']!r} differ from {fn['arguments']!r}")
        need(stream_finish(events) == "tool_calls", f"streamed finish_reason {stream_finish(events)!r}")
        none = s.chat("What is the weather like in Paris right now?", 1024, tools=[TOOL], tool_choice="none", reasoning_effort="none")
        need(not probe.unpack(none)[0].get("tool_calls"), "tool_choice none still gave a tool call")
        # a tool result must be answered from, not called again
        out = api.post(CHAT, {"model": api.model, "temperature": 0, "max_tokens": 1024, "reasoning_effort": "none", "tools": [TOOL],
                              "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."},
                                           {"role": "assistant", "content": "", "tool_calls": [
                                               {"id": "call_1", "type": "function",
                                                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
                                           {"role": "tool", "tool_call_id": "call_1",
                                            "content": "{\"temp_c\": 21, \"conditions\": \"clear\"}"}]})[0]
        _, text, _, _, _ = probe.unpack(out)
        need("21" in text and not probe.unpack(out)[0].get("tool_calls"), f"the answer does not use the tool result: {text[:120]!r}")
        return f"get_weather({args}), streamed identically, tool_choice none, result answered"

    def completions():
        status, out, _ = api.call("POST", "/v1/completions", {"model": api.model, "prompt": "The capital of France is",
                                                                "max_tokens": 16, "temperature": 0})
        need(status == 200 and isinstance(out, dict) and out.get("choices"), f"HTTP {status}: {str(out)[:200]}")
        ch = out["choices"][0]
        need(isinstance(ch.get("text"), str) and ch["text"], f"text {ch.get('text')!r}")
        need(ch.get("finish_reason") in ("stop", "length"), f"finish_reason {ch.get('finish_reason')!r}")
        need(out["usage"]["completion_tokens"] > 0, "no usage")
        return f"text {ch['text'][:40]!r}"

    def errors():
        got = {}
        got["unknown model"] = api.call("POST", CHAT, {"model": "definitely-not-served", "messages": msgs})[0]
        got["bad JSON"] = raw_post(api, CHAT, b"{not json")
        got["no messages"] = api.call("POST", CHAT, {"model": api.model})[0]
        got["unknown field"] = api.call("POST", CHAT, {"model": api.model, "messages": msgs, "no_such_field": 1})[0]
        ml = s.models.get("max_model_len") or 0
        # a max_tokens past the window: this server's contract (every family, `length_budget` in crates/server/src/http.rs and the engine's
        # abuse test) clamps it to the room left in the window and answers 200; a 4xx is the other valid contract. What is never valid is
        # an answer longer than the window: the reply must fit, prompt + completion <= max_model_len, and a clamped one ends on its own
        # (stop) or at length.
        ok_200 = {}
        for mt in (ml + 1, 10_000_000):
            status, out, _ = api.call("POST", CHAT, {"model": api.model, "messages": msgs, "max_tokens": mt})
            key = f"max_tokens {mt}"
            if status == 200 and isinstance(out, dict):
                u = out.get("usage") or {}
                fits = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) <= ml
                fin = out["choices"][0].get("finish_reason")
                need(fits and fin in ("stop", "length"), f"{key}: 200 with usage {u}, finish {fin!r}: not clamped to the window {ml}")
                ok_200[key] = f"200 clamped ({u.get('completion_tokens')} tokens, finish {fin})"
            else:
                got[key] = status
        bad = {k: v for k, v in got.items() if not 400 <= v < 500 or v == 429}
        need(not bad, f"not a 4xx client error: {bad} (all: {got})")
        return str({**got, **ok_200})

    def metrics():
        before = api.metrics()
        s.chat("Say OK.", 16, reasoning_effort="none")
        after = api.metrics()
        d = metric_deltas(before, after, ("rexl3_requests_total",))["rexl3_requests_total"]
        need(d is not None and d >= 1, f"rexl3_requests_total moved by {d}")
        ml = probe.metric_total(after, "rexl3_max_model_len")
        need(ml == s.models.get("max_model_len"), f"rexl3_max_model_len {ml} != /v1/models {s.models.get('max_model_len')}")
        s.cache_on = bool(probe.metric_total(after, "rexl3_prefix_cache_enabled"))
        return f"requests +{int(d)}, max_model_len {int(ml)}, prefix cache {'on' if s.cache_on else 'off'}"

    def spec():
        before = api.metrics()
        out = s.chat("Write a Python function that returns the n-th Fibonacci number, with a docstring, then a short explanation.",
                     256, reasoning_effort="none")
        sc = ia.spec_counters(out["usage"])
        need(sc is not None, "no usage.spec")
        want = {"rexl3_spec_drafted_total": sc["drafted"], "rexl3_spec_accepted_total": sc["accepted"], "rexl3_spec_rounds_total": sc["rounds"]}
        deltas = probe.settle_spec_metrics(api, before, want)
        need(all(deltas[n] == want[n] for n in want), f"/metrics deltas {deltas} differ from usage.spec {sc}")
        why = probe.spec_verdict(a.spec, sc["drafted"])
        need(why is None, why or "")
        serial = s.chat("Write a Python function that returns the n-th Fibonacci number.", 64, reasoning_effort="none", draft=False)
        need(ia.spec_counters(serial["usage"])["drafted"] == 0, f"draft:false drafted {serial['usage']['spec']}")
        return f"drafter {'on' if sc['drafted'] else 'off'}: {sc}"

    def prefix_cache():
        prompt = probe.filler_prompt(os.urandom(6).hex(), 90)
        off = s.chat(prompt, 8, reasoning_effort="none", prefix_cache=False)
        need(ia.cached_tokens(off["usage"]) == 0, f"prefix_cache false but cached_tokens {ia.cached_tokens(off['usage'])}")
        first = s.chat(prompt, 8, reasoning_effort="none")
        again = s.chat(prompt, 8, reasoning_effort="none")
        c1, c2 = ia.cached_tokens(first["usage"]), ia.cached_tokens(again["usage"])
        need(c1 == 0, f"a never-seen prompt reported cached_tokens {c1}")
        if s.cache_on:
            need(c2 and c2 > 0, f"cache on but the repeat of a {again['usage']['prompt_tokens']}-token prompt cached {c2} tokens")
        else:
            need(c2 == 0, f"cache off but cached_tokens {c2}")
        return f"cache {'on' if s.cache_on else 'off'}: repeat cached_tokens={c2} of {again['usage']['prompt_tokens']}"

    def concurrency():
        def one(_):
            return api.call("POST", CHAT, {"model": api.model, "messages": msgs, "max_tokens": 32, "temperature": 0,
                                           "reasoning_effort": "none"})[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(one, range(2)))
        need(codes == [200, 200], f"statuses {codes}")
        return str(codes)

    def determinism():
        ids = []
        for _ in range(2):
            out = s.chat("Compute 37*43 step by step, then state the final answer.", 64, reasoning_effort="none", return_token_ids=True)
            ids.append(out["choices"][0].get("token_ids"))
            time.sleep(0.2)
        need(ids[0] and ids[1], "no token_ids in the replies")
        need(ids[0] == ids[1], f"greedy ids differ: {first_diff(ids[0], ids[1])}")
        return f"{len(ids[0])} identical ids"

    s.run("health", health)
    s.run("models", models)
    s.run("tokenize", tokenize)
    s.run("chat", chat_usage)
    s.run("render", render)
    s.run("max_tokens", max_tokens)
    s.run("stop", stop)
    s.run("streaming", streaming)
    s.run("reasoning", reasoning)
    s.run("tools", tools)
    s.run("completions", completions)
    s.run("errors", errors)
    s.run("metrics", metrics)
    s.run("spec", spec)
    s.run("prefix cache", prefix_cache)
    s.run("concurrency", concurrency)
    s.run("determinism", determinism)
    print(f"api_conformance: {s.passed} PASS, {s.failed} FAIL")
    return 1 if s.failed else 0


if __name__ == "__main__":
    sys.exit(main())
