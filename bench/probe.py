#!/usr/bin/env python3
"""Pre-flight probe of the node's OpenAI surface (stdlib only): run it before trusting any later measurement.

One server process, TP1. Every request is greedy (temperature 0). Prints one `PASS name`, `FAIL name: why` or `WARN name:
why` line per check, then `METRIC name=value` lines for the numbers later runs compare against; exit 1 when a check FAILs.

  health        GET /health answers 200
  models        /v1/models: the served id (--model must be it), max_model_len, configured_max_model_len, kv_pool_tokens,
                prefix_cache_reach_tokens; with --window N the effective max_model_len must be at least N
  tokenize      /tokenize returns integer ids whose count matches, the same ids on a repeat
  chat          reasoning_effort "none": the answer to 17*23 holds 391 in the content; finish_reason is not `length`
  usage         prompt/completion/total tokens add up; prompt_tokens_details.cached_tokens and
                completion_tokens_details.reasoning_tokens are integers; usage.spec is a drafted/accepted/rounds block
  reasoning     reasoning_effort "high": a non-empty reasoning_content, reasoning_tokens > 0, a non-empty content, no framing
                marker in either; a reply cut by `length` is a WARN (the base model can reason in long repetitive loops)
  levels        the declared reasoning levels (/v1/models `reasoning`): the template's seven names with the number each renders
                as (none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh and max 0.99); no level suppresses reasoning
                (`none` is short reasoning, not none), and the `chat` check's own reasoning_tokens at `none` must agree (> 0)
  effort        the system line `Thinking effort level: 0` and reasoning_effort "none" and "high" are accepted (HTTP 200)
  tools         one get_weather(city) tool: choices[0].message.tool_calls names it, the arguments parse with a city
  metrics       /metrics exports the families the bench tools read (rexl3_request_ttft_seconds, rexl3_request_e2e_seconds,
                rexl3_prefix_cache_enabled, rexl3_spec_*_total, rexl3_max_model_len equal to /v1/models)
  spec          a drafted request: usage.spec against the /metrics rexl3_spec_* deltas; --spec on needs drafted > 0, --spec
                off needs drafted == 0, auto (default) takes either; then "draft": false must draft 0 tokens
  prefix-cache  the same ~2K-token prompt twice: with the cache on the repeat reports cached_tokens > 0 (a WARN when it
                does not: the cache only serves prompts that reach its smallest cut); with the cache off, 0
  vision        (only with --vision) a synthetic red square as a data URL image part: the reply names red

  probe.py [--url URL] [--model NAME] [--window 262144] [--spec auto|on|off] [--vision] [--out probe.json]

--out writes the facts (served id, windows, reasoning controls, spec state, per-check results) as one JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import inkling_api as ia  # noqa: E402

TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city's name"}},
                   "required": ["city"]}}}
LEVEL0 = "Thinking effort level: 0"
METRIC_FAMILIES = ("rexl3_request_ttft_seconds_count", "rexl3_request_e2e_seconds_count", "rexl3_prefix_cache_enabled",
                   "rexl3_spec_drafted_total", "rexl3_spec_accepted_total", "rexl3_spec_rounds_total",
                   "rexl3_max_model_len", "rexl3_requests_total")
SPEC_METRICS = ("rexl3_spec_drafted_total", "rexl3_spec_accepted_total", "rexl3_spec_rounds_total")


def framing_in(text: str) -> list[str]:
    return [m for m in ia.FRAMING if m in text]


def metric_total(samples: dict[str, float], name: str) -> float | None:
    """The sum of a metric over its label sets (None when the family is absent)."""
    vals = [v for k, v in samples.items() if k.split("{")[0] == name]
    return sum(vals) if vals else None


def unpack(out: dict) -> tuple[dict, str, str, dict, str | None]:
    """(message, content, reasoning_content, usage, finish_reason) of a chat reply."""
    ch = (out.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return msg, msg.get("content") or "", msg.get("reasoning_content") or "", out.get("usage") or {}, ch.get("finish_reason")


def usage_problems(u: dict) -> list[str]:
    """What is wrong with a usage block (empty list: nothing)."""
    bad = []
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(u.get(k), int) or isinstance(u.get(k), bool):
            bad.append(f"{k} is not an integer")
    if not bad and u["prompt_tokens"] + u["completion_tokens"] != u["total_tokens"]:
        bad.append("total_tokens is not prompt_tokens + completion_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if not isinstance(cached, int) or isinstance(cached, bool):
        bad.append("prompt_tokens_details.cached_tokens is not an integer")
    elif isinstance(u.get("prompt_tokens"), int) and cached > u["prompt_tokens"]:
        bad.append("cached_tokens exceeds prompt_tokens")
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if not isinstance(rt, int) or isinstance(rt, bool):
        bad.append("completion_tokens_details.reasoning_tokens is not an integer")
    elif isinstance(u.get("completion_tokens"), int) and rt > u["completion_tokens"]:
        bad.append("reasoning_tokens exceeds completion_tokens")
    spec = u.get("spec")
    if not isinstance(spec, dict) or not all(isinstance(spec.get(k), int) for k in ("drafted", "accepted", "rounds")):
        bad.append("usage.spec is not a {drafted, accepted, rounds} block of integers")
    elif spec["accepted"] > spec["drafted"]:
        bad.append("spec.accepted exceeds spec.drafted")
    return bad


def spec_verdict(mode: str, drafted: int) -> str | None:
    """Why a drafted-request count contradicts --spec (None: it agrees)."""
    if mode == "on" and drafted <= 0:
        return "--spec on but the request drafted 0 tokens (is the drafter loaded?)"
    if mode == "off" and drafted != 0:
        return f"--spec off but the request drafted {drafted} tokens"
    return None


def filler_prompt(nonce: str, sentences: int) -> str:
    return (f"Notes {nonce}. " + " ".join(ia.filler_sentences(sentences, int(nonce, 16) % 100000))
            + "\n\nIn one short sentence, what is the topic of these notes?")


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.fails = 0

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append({"status": status, "check": name, "detail": detail})
        if status == "FAIL":
            self.fails += 1
        print(f"{status} {name}" + (f": {detail}" if detail else ""), flush=True)

    def ok(self, name: str, detail: str = "") -> None:
        self.add("PASS", name, detail)

    def bad(self, name: str, detail: str) -> None:
        self.add("FAIL", name, detail)

    def warn(self, name: str, detail: str) -> None:
        self.add("WARN", name, detail)


def chat(api: ia.Api, content: str, max_tokens: int, system: str | None = None, **extra) -> tuple[int, object, float]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"model": api.model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, **extra}
    return api.call("POST", "/v1/chat/completions", body)


def settle_spec_metrics(api: ia.Api, before: dict[str, float], want: dict[str, int]) -> dict[str, float | None]:
    """The /metrics spec counters' deltas, polled for a few seconds: they accumulate when a request finishes."""
    deltas: dict[str, float | None] = {}
    for _ in range(10):
        now = api.metrics()
        deltas = {n: (None if metric_total(now, n) is None else metric_total(now, n) - (metric_total(before, n) or 0))
                  for n in SPEC_METRICS}
        if all(deltas[n] == want[n] for n in SPEC_METRICS):
            break
        time.sleep(0.5)
    return deltas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--window", type=int, help="the profile's context window: 262144 (bootstrap) or 1048576 (long, max)")
    ap.add_argument("--spec", choices=("auto", "on", "off"), default="auto",
                    help="the drafter's expected state (default: auto, either)")
    ap.add_argument("--vision", action="store_true", help="also send a synthetic image (a VISION=off boot refuses it)")
    ap.add_argument("--out", help="write the facts and results as JSON here")
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    rep = Report()
    facts: dict = {"url": api.url, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    metrics: dict[str, object] = {}

    # health
    status, out, _ = api.call("GET", "/health", timeout=30)
    if status == 200:
        rep.ok("health", str(out)[:80])
    else:
        rep.bad("health", f"HTTP {status}: {str(out)[:200]}")
        print("the server is not ready; nothing else is probed", file=sys.stderr)
        return 1

    # models
    status, listing, _ = api.call("GET", "/v1/models", timeout=30)
    row = ((listing or {}).get("data") or [{}])[0] if status == 200 and isinstance(listing, dict) else {}
    ids = [r.get("id") for r in ((listing or {}).get("data") or [])] if isinstance(listing, dict) else []
    maxlen = row.get("max_model_len")
    facts["models"] = {k: row.get(k) for k in ("id", "max_model_len", "configured_max_model_len", "kv_pool_tokens",
                                                "prefix_cache_reach_tokens", "pack_validated", "draft_validated", "reasoning")}
    if status != 200 or not ids or not isinstance(maxlen, int) or maxlen <= 0:
        rep.bad("models", f"HTTP {status}, ids {ids}, max_model_len {maxlen!r}")
        return 1
    rep.ok("models", f"{ids} max_model_len={maxlen} configured={row.get('configured_max_model_len')} "
                     f"kv_pool_tokens={row.get('kv_pool_tokens')} prefix_cache_reach_tokens={row.get('prefix_cache_reach_tokens')}")
    if api.model not in ids:
        rep.bad("model id", f"{api.model!r} is not served: {ids}")
    if a.window is not None and maxlen < a.window:
        rep.bad("window", f"max_model_len {maxlen} is below the profile window {a.window} (the KV pool limits it: "
                          f"kv_pool_tokens={row.get('kv_pool_tokens')})")
    elif a.window is not None:
        rep.ok("window", f"max_model_len {maxlen} >= {a.window}")
    metrics["probe_max_model_len"] = maxlen

    # tokenize
    try:
        t1, t2 = api.tokenize("hello world"), api.tokenize("hello world")
        status, raw, _ = api.call("POST", "/tokenize", {"model": api.model, "text": "hello world"})
        count = raw.get("count") if isinstance(raw, dict) else None
        if t1 and t1 == t2 and count == len(t1):
            rep.ok("tokenize", f"'hello world' = {len(t1)} tokens {t1}")
        else:
            rep.bad("tokenize", f"ids {t1} / {t2}, count {count!r}")
    except (ia.ApiError, KeyError, ValueError, TypeError) as e:
        rep.bad("tokenize", str(e)[:300])

    # chat + usage (reasoning_effort none)
    status, out, dt = chat(api, "What is 17*23? Reply with the integer only.", 1024, reasoning_effort="none")
    if status != 200 or not isinstance(out, dict) or not out.get("choices"):
        rep.bad("chat", f"HTTP {status}: {str(out)[:300]}")
    else:
        _, content, reasoning, usage, finish = unpack(out)
        if finish == "length":
            rep.bad("chat", f"cut by max_tokens (finish length) after {usage.get('completion_tokens')} tokens")
        elif "391" in content:
            rep.ok("chat", f"17*23 -> 391 ({dt:.2f}s, reasoning_tokens="
                           f"{(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')})")
        else:
            rep.bad("chat", f"expected 391, got content {content[:120]!r} reasoning {reasoning[:80]!r}")
        metrics["probe_reasoning_tokens_none"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        problems = usage_problems(usage)
        if problems:
            rep.bad("usage", "; ".join(problems) + f" in {json.dumps(usage)}")
        else:
            rep.ok("usage", json.dumps(usage))
        facts["usage_example"] = usage

    # reasoning at effort high
    status, out, dt = chat(api, "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
                           4096, reasoning_effort="high")
    if status != 200 or not isinstance(out, dict) or not out.get("choices"):
        rep.bad("reasoning", f"HTTP {status}: {str(out)[:300]}")
    else:
        _, content, reasoning, usage, finish = unpack(out)
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        metrics["probe_reasoning_tokens_high"] = rt
        why = []
        if not reasoning.strip():
            why.append("empty reasoning_content")
        if not isinstance(rt, int) or rt <= 0:
            why.append(f"reasoning_tokens {rt!r}")
        if finish != "length" and not content.strip():
            why.append("empty content")
        for where, text in (("content", content), ("reasoning", reasoning)):
            if framing_in(text):
                why.append(f"{where} holds {framing_in(text)}")
        if why:
            rep.bad("reasoning", "; ".join(why))
        elif finish == "length":
            rep.warn("reasoning", f"finish length after {usage.get('completion_tokens')} tokens ({rt} reasoning): the model "
                                  "may be looping in its reasoning")
        else:
            rep.ok("reasoning", f"{len(reasoning)} chars, reasoning_tokens={rt}, content {content[:60]!r}")

    # the declared levels against what the engine measured (the `chat` check above ran at none, the `reasoning` check at high)
    declared = (facts["models"].get("reasoning") or {}).get("levels")
    want = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
    if not isinstance(declared, list):
        rep.bad("levels", "/v1/models carries no reasoning levels")
    else:
        got = {l.get("name"): (l.get("effort"), l.get("suppresses_reasoning")) for l in declared}
        why = []
        if list(got) != list(want):
            why.append(f"level names {list(got)}, the template's are {list(want)}")
        for name, number in want.items():
            if name in got and got[name] != (number, False):
                why.append(f"{name}: declared (effort, suppresses_reasoning) {got[name]}, expected ({number}, False)")
        rt_none = metrics.get("probe_reasoning_tokens_none")
        if not isinstance(rt_none, int) or rt_none <= 0:
            why.append(f"declared reasoning at none, measured reasoning_tokens {rt_none!r}")
        if why:
            rep.bad("levels", "; ".join(why))
        else:
            rep.ok("levels", f"{len(got)} levels, none renders 0 and still reasons (reasoning_tokens {rt_none} at none)")

    # effort controls
    for label, kw, system in (("system level 0", {}, LEVEL0), ("effort none", {"reasoning_effort": "none"}, None),
                              ("effort high", {"reasoning_effort": "high"}, None)):
        status, out, _ = chat(api, "Say OK.", 1024, system=system, **kw)
        good = status == 200 and isinstance(out, dict) and out.get("choices")
        if good and any(framing_in(t) for t in unpack(out)[1:3]):
            rep.bad(f"effort {label}", f"framing marker in the reply: {unpack(out)[1:3]}")
        elif good:
            rep.ok(f"effort {label}", f"accepted, completion_tokens={unpack(out)[3].get('completion_tokens')}")
        else:
            rep.bad(f"effort {label}", f"HTTP {status}: {str(out)[:200]}")

    # tools
    status, out, _ = chat(api, "What is the weather like in Paris right now? Use the get_weather tool.", 1024,
                          tools=[TOOL], tool_choice="auto", reasoning_effort="none")
    calls = unpack(out)[0].get("tool_calls") if status == 200 and isinstance(out, dict) and out.get("choices") else None
    if calls:
        fn = calls[0].get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}") if isinstance(fn.get("arguments"), str) else fn.get("arguments")
            good = fn.get("name") == "get_weather" and isinstance(args, dict) and str(args.get("city", "")).strip()
        except ValueError:
            good, args = False, fn.get("arguments")
        if good:
            rep.ok("tools", f"get_weather({args})")
        else:
            rep.bad("tools", f"called {fn.get('name')!r} with {fn.get('arguments')!r}")
    else:
        rep.bad("tools", f"no tool_calls (HTTP {status}): {str(out)[:200]}")

    # metrics families
    try:
        samples = api.metrics()
        missing = [n for n in METRIC_FAMILIES if metric_total(samples, n) is None]
        if missing:
            rep.bad("metrics", f"missing {missing}")
        elif metric_total(samples, "rexl3_max_model_len") != maxlen:
            rep.bad("metrics", f"rexl3_max_model_len {metric_total(samples, 'rexl3_max_model_len')} != /v1/models {maxlen}")
        else:
            rep.ok("metrics", f"{len(samples)} samples; cache enabled={metric_total(samples, 'rexl3_prefix_cache_enabled')}")
        cache_on = bool(metric_total(samples, "rexl3_prefix_cache_enabled"))
    except (ia.ApiError, OSError) as e:
        rep.bad("metrics", str(e)[:300])
        samples, cache_on = {}, False
    facts["prefix_cache_enabled"] = cache_on

    # spec counters against usage.spec
    try:
        before = api.metrics()
        status, out, _ = chat(api, "Write a Python function that returns the n-th Fibonacci number, with a docstring, then "
                                   "a short explanation.", 256, reasoning_effort="none")
        usage = unpack(out)[3] if status == 200 and isinstance(out, dict) and out.get("choices") else {}
        s = ia.spec_counters(usage)
        if s is None:
            rep.bad("spec", f"no usage.spec in the reply (HTTP {status})")
        else:
            facts["spec"] = s
            deltas = settle_spec_metrics(api, before, {"rexl3_spec_drafted_total": s["drafted"],
                                                       "rexl3_spec_accepted_total": s["accepted"],
                                                       "rexl3_spec_rounds_total": s["rounds"]})
            why = spec_verdict(a.spec, s["drafted"])
            if why:
                rep.bad("spec", why)
            elif any(deltas[n] is None for n in SPEC_METRICS):
                rep.bad("spec", f"/metrics lacks {[n for n in SPEC_METRICS if deltas[n] is None]}")
            elif (deltas["rexl3_spec_drafted_total"], deltas["rexl3_spec_accepted_total"],
                  deltas["rexl3_spec_rounds_total"]) != (s["drafted"], s["accepted"], s["rounds"]):
                rep.bad("spec", f"/metrics deltas {deltas} differ from usage.spec {s}")
            else:
                rep.ok("spec", f"drafter {'on' if s['drafted'] else 'off'}: drafted={s['drafted']} accepted={s['accepted']} "
                               f"rounds={s['rounds']} acceptance={s['acceptance']} tokens/round={s['tokens_per_round']}")
            metrics["probe_spec_drafted"] = s["drafted"]
            if s["acceptance"] is not None:
                metrics["probe_spec_acceptance"] = s["acceptance"]
            status, out, _ = chat(api, "Write a Python function that returns the n-th Fibonacci number.", 64,
                                  reasoning_effort="none", draft=False)
            serial = ia.spec_counters(unpack(out)[3]) if status == 200 and isinstance(out, dict) and out.get("choices") else None
            if serial is not None and serial["drafted"] == 0:
                rep.ok("spec serial", "draft:false drafted 0 tokens")
            else:
                rep.bad("spec serial", f"HTTP {status}: draft:false gave {serial}")
    except (ia.ApiError, OSError) as e:
        rep.bad("spec", str(e)[:300])

    # prefix cache
    nonce = os.urandom(6).hex()
    prompt = filler_prompt(nonce, 90)
    got = []
    for _ in range(2):
        status, out, _ = chat(api, prompt, 32, reasoning_effort="none")
        got.append(ia.cached_tokens(unpack(out)[3]) if status == 200 and isinstance(out, dict) and out.get("choices") else None)
    pt = unpack(out)[3].get("prompt_tokens") if isinstance(out, dict) else None
    if None in got:
        rep.bad("prefix-cache", f"a request failed (HTTP {status}): {str(out)[:200]}")
    elif got[0] != 0:
        rep.bad("prefix-cache", f"a never-seen prompt reported cached_tokens={got[0]}")
    elif cache_on and got[1] > 0:
        rep.ok("prefix-cache", f"repeat of a {pt}-token prompt: cached_tokens={got[1]}")
    elif cache_on:
        rep.warn("prefix-cache", f"cache on but the repeat of a {pt}-token prompt cached 0 tokens (below the cache's smallest cut?)")
    elif got[1] == 0:
        rep.ok("prefix-cache", "cache off: cached_tokens 0")
    else:
        rep.bad("prefix-cache", f"cache reported off but cached_tokens={got[1]}")
    metrics["probe_cached_tokens_repeat"] = got[1]

    # vision (optional)
    if a.vision:
        import synth_images as si
        _, question, words = si.IMAGES["red-square"]
        body = {"model": api.model, "max_tokens": 256, "temperature": 0, "reasoning_effort": "none", "messages": [{
            "role": "user", "content": [{"type": "text", "text": question},
                                        {"type": "image_url", "image_url": {"url": si.data_url(si.image_png("red-square"))}}]}]}
        status, out, _ = api.call("POST", "/v1/chat/completions", body)
        if status != 200 or not isinstance(out, dict) or not out.get("choices"):
            rep.bad("vision", f"HTTP {status}: {str(out)[:200]}")
        else:
            text = (unpack(out)[1] + " " + unpack(out)[2]).lower()
            rep.add("PASS" if all(w in text for w in words) else "FAIL", "vision",
                    f"reply {unpack(out)[1][:80]!r}" if all(w in text for w in words) else f"the reply does not name {words}")

    print(f"probe: {'OK' if rep.fails == 0 else str(rep.fails) + ' failure(s)'}")
    for k in sorted(metrics):
        print(f"METRIC {k}={metrics[k]}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"facts": facts, "metrics": metrics, "checks": rep.rows, "fails": rep.fails}, f, indent=1)
            f.write("\n")
    return 0 if rep.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
