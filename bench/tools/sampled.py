#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sampled.py: the decode speed of an interactive session: SAMPLED chat completions that reason, not greedy fixed prompts.

Every other bench tool of this recipe is greedy on fixed prompts; the pack's generation_config.json says `do_sample: true,
temperature: 1.0` and the model always reasons. This tool sends 16 original chat prompts (PROMPTS: four open chat, four
reasoning problems, four code tasks, four structured-extraction / JSON tasks) as streamed chat completions with
reasoning_effort in {low, high} (max_tokens 1024 and 2048) at temperature 0 and 1.0, each request with a fixed seed derived from
(prompt, effort, temperature) so a rerun, and a boot with another SPEC, sends identical requests, and `"prefix_cache": false`
(cold prompt). Streaming gives the time to first token; completion tokens, reasoning tokens and the drafter counters come from
`usage` (stream_options.include_usage), never from chunk counts.

  sampled.py run OUT.json            c = 1: 16 prompts x 2 efforts x 2 temperatures = 64 requests one at a time, then a c = 4
                                     pass: 16 requests (temperature 1.0, effort high) in four waves of four different prompts
                                     (one per category) at once; stores every request (ids, text, timing, counters)
  sampled.py summary OUT.json        the tables of one run: mean and median decode tok/s, tokens per round, acceptance, share of
                                     reasoning tokens, time to first token and the requests cut by max_tokens, per category x
                                     effort x temperature, and the c = 4 waves
  sampled.py compare A.json B.json   two runs of the same requests (A drafted, B serial): the speed ratio per category and how
                                     many replies are identical id for id, temperature 0 and temperature 1.0 separately

Decode tok/s of a request = (completion_tokens - 1) / (t_last - t_first): the first token is the time to first token's, so the
rate is over the gaps between the first and the last token-bearing stream events (a drafted round arrives as one event with
several tokens). The server comes from NODES and PORT (environment or .env) or INKLING2MINI_URL, as for the bench tools.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402

EFFORTS = ("low", "high")
MAX_TOKENS = {"low": 1024, "high": 2048}
TEMPERATURES = (0.0, 1.0)  # 1.0: the pack's generation_config.json
CATEGORIES = ("chat", "reasoning", "code", "structured")
SEED_BASE = 71000
WAVE = 4  # the c = 4 pass: this many different prompts at once

# 16 original prompts, four per category (order fixes each prompt's index, hence its seed).
PROMPTS: list[tuple[str, str, str]] = [
    ("chat", "chat-hobbies", "I keep starting hobbies and dropping them after a few weeks. Give me practical, kind advice on sticking with one, and on how to decide which one deserves to stay."),
    ("chat", "chat-heatpump", "Explain to a curious teenager how a heat pump can deliver more heat than the electricity it uses, without breaking the laws of physics."),
    ("chat", "chat-neighbour", "Draft a warm, short message to a neighbour whose dog barks every afternoon, asking for a fix without starting a feud."),
    ("chat", "chat-newcity", "What would a good first week look like for someone who has just moved to a city where they know nobody? Keep it concrete and low-pressure."),
    ("reasoning", "reason-pencils", "A shop sells pencils at 35 cents each, or a box of 12 for 3.60 dollars. Mara needs exactly 50 pencils and wants to pay as little as possible, buying whole boxes and single pencils only. What does she buy, and what does it cost?"),
    ("reasoning", "reason-race", "Four runners, Ana, Ben, Cy and Dee, finish a race with no ties. Ana finished ahead of Ben. Cy finished immediately after Dee. Ben did not finish last. List every finishing order that fits these facts and explain how you know the list is complete."),
    ("reasoning", "reason-route", "A courier starts at the depot and must visit stops P, Q, R and S once each, then return to the depot. Minutes: depot-P 10, depot-Q 15, depot-R 20, depot-S 12, P-Q 8, P-R 14, P-S 18, Q-R 9, Q-S 16, R-S 7. Find the shortest round trip and show why nothing is shorter."),
    ("reasoning", "reason-bug", "This function should return the moving average of a list over a window, one value per full window. It gives wrong numbers once the list is longer than the window. Find the bug, give a failing input with the expected and actual output, and fix it.\n\ndef moving_average(values, window):\n    out = []\n    total = 0\n    for i, v in enumerate(values):\n        total += v\n        if i >= window:\n            total -= values[i - window - 1]\n        if i >= window - 1:\n            out.append(total / window)\n    return out"),
    ("code", "code-intervals", "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals given as (start, end) tuples in any order. Include a docstring and three assert-based tests."),
    ("code", "code-rust-kv", "Write a Rust function that reads a text file of key=value lines, ignores blank lines and lines starting with #, and returns a HashMap<String, String>, with proper error handling for I/O and for malformed lines."),
    ("code", "code-sql-window", "A table orders(id, customer_id, placed_at, total) holds one row per order. Write a SQL query that returns, for each customer, their most recent order and the running total of all their orders up to and including it. Explain the window functions you use."),
    ("code", "code-ts-debounce", "Write a TypeScript debounce function with correct generic typing that supports a cancel() method and an option to fire on the leading edge. Show a short usage example."),
    ("structured", "json-hire", "Extract the data as JSON with the keys name, role, start_date (ISO 8601), salary_usd (a number) and skills (an array of strings). Output only the JSON.\n\nText: Hi all, please welcome Priya Raman, who joins us on 3 March 2026 as a senior data engineer. Her base is $148,500 a year, and she brings strong Spark, dbt and Airflow experience, plus some Rust."),
    ("structured", "json-table", "Convert this table to a JSON array of objects with numeric fields, then add a final object {\"total_units\": ..., \"total_revenue\": ...} with the sums. Output only the JSON.\n\nitem | units | revenue\nwidget | 120 | 2400.50\ngadget | 75 | 1875.00\nsprocket | 310 | 930.25\nbracket | 42 | 1260.00"),
    ("structured", "json-log", "From these log lines produce a JSON object mapping each host to {\"errors\": n, \"warnings\": n, \"first_seen\": timestamp}. Output only the JSON.\n\n2026-05-01T08:00:01Z web-1 ERROR upstream timed out\n2026-05-01T08:00:04Z web-2 WARN slow response 2.1s\n2026-05-01T08:00:09Z web-1 WARN retry 1 of 3\n2026-05-01T08:01:15Z db-1 ERROR deadlock detected\n2026-05-01T08:01:20Z web-2 ERROR upstream timed out\n2026-05-01T08:02:02Z web-1 ERROR upstream timed out\n2026-05-01T08:02:30Z db-1 WARN replica lag 4s"),
    ("structured", "json-ticket", "Read this support email and fill in a JSON object with the keys customer, product, issue_category (one of billing, shipping, defect, other), urgency (an integer 1 to 5, 5 the most urgent) and requested_action. Output only the JSON.\n\nEmail: Hello, this is Tomas Berg. The kettle I bought two weeks ago (model KT-220) stopped heating this morning and the base smells burnt. I have a dinner for eight tonight and cannot use it. Please send a replacement by tomorrow or refund me."),
]


# ── pure arithmetic (unit-tested, no server) ──
def seed_for(prompt_index: int, effort: str, temperature: float) -> int:
    """The fixed seed of one (prompt, effort, temperature): distinct for every combination, the same on every run."""
    return SEED_BASE + (prompt_index * len(EFFORTS) + EFFORTS.index(effort)) * len(TEMPERATURES) + TEMPERATURES.index(temperature)


def decode_rate(completion_tokens: int, t_first: float | None, t_last: float | None) -> float | None:
    """(completion_tokens - 1) / (t_last - t_first); None when there is no gap to divide by (fewer than two tokens, one event,
    or the stamps missing)."""
    if t_first is None or t_last is None or completion_tokens < 2 or t_last <= t_first:
        return None
    return (completion_tokens - 1) / (t_last - t_first)


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def wave_aggregate(rows: list[dict]) -> dict:
    """One wave of requests started together: aggregate tok/s = all completion tokens over the span from the first start to the
    last end (start_offset_s + wall_s of each row), and the median of the requests' own decode rates."""
    done = [r for r in rows if not r.get("error")]
    if not done:
        return {"requests": len(rows), "answered": 0, "tokens": 0, "span_s": None, "aggregate_tok_s": None, "median_tok_s": None}
    tokens = sum(r["completion_tokens"] for r in done)
    span = max(r["start_offset_s"] + r["wall_s"] for r in done) - min(r["start_offset_s"] for r in done)
    rates = [r["tok_s"] for r in done if r.get("tok_s") is not None]
    return {"requests": len(rows), "answered": len(done), "tokens": tokens, "span_s": span,
            "aggregate_tok_s": tokens / span if span > 0 else None, "median_tok_s": statistics.median(rates) if rates else None}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(rows: list[dict]) -> dict:
    """The summary of requests of one group: decode tok/s (mean, median), the pooled drafter counters, the share of reasoning
    tokens, the time to first token and the requests cut by max_tokens. A request without spec counters (the drafter is off)
    counts in everything but the drafter figures, which are then None; a failed request counts only in `failed`."""
    ok = [r for r in rows if not r.get("error")]
    rates = [r.get("tok_s") for r in ok]
    sp = [r["spec"] for r in ok if r.get("spec")]
    rounds = sum(s["rounds"] for s in sp)
    drafted = sum(s["drafted"] for s in sp)
    spec_tokens = sum(r["completion_tokens"] for r in ok if r.get("spec"))
    comp = sum(r["completion_tokens"] for r in ok)
    reasoning = sum(r.get("reasoning_tokens") or 0 for r in ok)
    return {
        "requests": len(rows), "failed": len(rows) - len(ok), "completion_tokens": comp,
        "tok_s_mean": _mean(rates), "tok_s_median": _median(rates),
        "tokens_per_round": spec_tokens / rounds if rounds else None,
        "acceptance": sum(s["accepted"] for s in sp) / drafted if drafted else None,
        "reasoning_share": reasoning / comp if comp else None,
        "ttft_mean_s": _mean(r.get("ttft_s") for r in ok), "ttft_median_s": _median(r.get("ttft_s") for r in ok),
        "hit_max_tokens": sum(1 for r in ok if r.get("finish_reason") == "length"),
    }


def group_key(r: dict) -> tuple:
    return (r["category"], r["effort"], r["temperature"])


def summarize_run(run: dict) -> dict:
    """The c = 1 requests grouped by category x effort x temperature, plus an `all` category per effort x temperature."""
    rows = [r for r in run["c1"]]
    out = {}
    for cat in (*CATEGORIES, "all"):
        for effort in EFFORTS:
            for t in TEMPERATURES:
                g = [r for r in rows if r["effort"] == effort and r["temperature"] == t and (cat == "all" or r["category"] == cat)]
                if g:
                    out[(cat, effort, t)] = summarize(g)
    return out


def request_key(r: dict) -> str:
    return f"{r['name']}|{r['effort']}|{r['temperature']}"


def compare_runs(a: dict, b: dict) -> dict:
    """A (drafted) against B (serial) over the c = 1 requests both runs answered: the ratio of mean decode tok/s per group, and
    per temperature how many replies are identical (same id list) of all pairs, with the first differing request."""
    bm = {request_key(r): r for r in b["c1"]}
    pairs = [(r, bm[request_key(r)]) for r in a["c1"] if request_key(r) in bm and not r.get("error") and not bm[request_key(r)].get("error")]
    exact = {}
    for t in TEMPERATURES:
        ps = [(x, y) for x, y in pairs if x["temperature"] == t]
        same = [x["ids_sha256"] == y["ids_sha256"] for x, y in ps]
        first = next(((x, y) for (x, y), s in zip(ps, same) if not s), None)
        diff = None
        if first:
            x, y = first
            ia_, ib_ = x["ids"], y["ids"]
            at = next((i for i, (p, q) in enumerate(zip(ia_, ib_)) if p != q), min(len(ia_), len(ib_)))
            diff = {"request": request_key(x), "first_differing_token": at, "len_a": len(ia_), "len_b": len(ib_)}
        exact[t] = {"pairs": len(ps), "identical": sum(same), "first_difference": diff}
    sa, sb = summarize_run({"c1": [p[0] for p in pairs]}), summarize_run({"c1": [p[1] for p in pairs]})
    ratios = {}
    for k, va in sa.items():
        vb = sb.get(k)
        if vb and va["tok_s_mean"] and vb["tok_s_mean"]:
            ratios[k] = {"mean": va["tok_s_mean"] / vb["tok_s_mean"],
                         "median": (va["tok_s_median"] / vb["tok_s_median"]) if va["tok_s_median"] and vb["tok_s_median"] else None}
    return {"exact": exact, "ratios": ratios, "pairs": len(pairs)}


# ── requests ──
def one_request(api: ia.Api, prompt: tuple[str, str, str], index: int, effort: str, temperature: float, wave_t0: float | None = None) -> dict:
    """One streamed chat completion; a failed request is a row with `error` (the run goes on)."""
    category, name, text = prompt
    body = {"model": api.model, "messages": [{"role": "user", "content": text}], "max_tokens": MAX_TOKENS[effort],
            "temperature": temperature, "seed": seed_for(index, effort, temperature), "reasoning_effort": effort,
            "prefix_cache": False, "return_token_ids": True, "stream_options": {"include_usage": True}}
    row = {"category": category, "name": name, "prompt_index": index, "effort": effort, "temperature": temperature,
           "seed": body["seed"], "max_tokens": body["max_tokens"]}
    start = time.monotonic()
    row["start_offset_s"] = (start - wave_t0) if wave_t0 is not None else 0.0
    ids: list[int] = []
    reasoning, content = [], []
    first = last = wall = None
    usage = finish = None
    try:
        for t, ev in api.stream("/v1/chat/completions", body):
            wall = t
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                new = [int(i) for i in ch.get("token_ids") or []]
                if new or d.get("content") or d.get("reasoning_content"):
                    ids += new
                    reasoning.append(d.get("reasoning_content") or "")
                    content.append(d.get("content") or "")
                    first = t if first is None else first
                    last = t
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except (ia.ApiError, OSError, ValueError) as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return row
    if usage is None or finish is None or first is None:
        row["error"] = f"stream ended without {'usage' if usage is None else 'finish reason' if finish is None else 'a token'}"
        return row
    n = int(usage["completion_tokens"])
    text_r, text_c = "".join(reasoning), "".join(content)
    row.update({
        "ttft_s": first, "t_first_s": first, "t_last_s": last, "wall_s": wall, "completion_tokens": n,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "reasoning_tokens": ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")),
        "finish_reason": finish, "tok_s": decode_rate(n, first, last), "spec": ia.spec_counters(usage),
        "ids_count": len(ids), "ids_sha256": ids_digest(ids), "ids": ids,
        "reasoning_text": text_r, "content_text": text_c,
        "text_sha256": hashlib.sha256((text_r + "\x00" + text_c).encode()).hexdigest()})
    return row


def log_row(tag: str, r: dict) -> None:
    if r.get("error"):
        print(f"{tag} {r['name']:18} {r['effort']:4} t={r['temperature']}  ERROR {r['error']}", flush=True)
        return
    sp = r["spec"]
    print(f"{tag} {r['name']:18} {r['effort']:4} t={r['temperature']}  {r['completion_tokens']:5} tok (reasoning {r['reasoning_tokens']}) "
          f"{r['finish_reason']:6} ttft {r['ttft_s']:6.2f}s  {r['tok_s'] if r['tok_s'] is None else round(r['tok_s'], 1)} tok/s"
          f"{'' if not sp else '  tok/round ' + str(sp['tokens_per_round'])}", flush=True)


def run(out: str, url: str | None) -> None:
    api = ia.Api(url or ia.default_url(), None)
    # one /v1/models lookup before the worker threads start (the model name is cached)
    model = api.model
    # warm the kernels and the sampler once (not recorded): a short reply, cold
    api.post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                                      "max_tokens": 32, "temperature": 1.0, "seed": 1, "reasoning_effort": "low", "prefix_cache": False})
    c1 = []
    for i, p in enumerate(PROMPTS):
        for effort in EFFORTS:
            for t in TEMPERATURES:
                r = one_request(api, p, i, effort, t)
                c1.append(r)
                log_row("c1", r)
    c4 = []
    for w in range(len(PROMPTS) // WAVE):
        idx = [w + k * WAVE for k in range(WAVE)]  # one prompt of each category per wave
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(WAVE) as ex:
            futs = [ex.submit(one_request, api, PROMPTS[i], i, "high", 1.0, t0) for i in idx]
            rows = [f.result() for f in futs]
        for r in rows:
            r["wave"] = w
            log_row(f"c4w{w}", r)
        c4 += rows
    rec = {"tool": "bench/tools/sampled.py", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": model,
           "efforts": list(EFFORTS), "temperatures": list(TEMPERATURES), "max_tokens": MAX_TOKENS, "seed_base": SEED_BASE,
           "c1": c1, "c4": c4}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f)
        f.write("\n")
    print(f"wrote {out}: {len(c1)} c=1 requests, {len(c4)} c=4 requests, "
          f"{sum(1 for r in c1 + c4 if r.get('error'))} failed", flush=True)


# ── reports ──
def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def tables(run: dict) -> str:
    lines = ["| category | effort | temp | n | tok/s mean | tok/s median | tokens/round | acceptance | reasoning share | TTFT mean s | hit max_tokens | failed |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (cat, effort, t), s in summarize_run(run).items():
        lines.append(f"| {cat} | {effort} | {t} | {s['requests']} | {fmt(s['tok_s_mean'])} | {fmt(s['tok_s_median'])} | {fmt(s['tokens_per_round'], 2)} | "
                     f"{fmt(s['acceptance'], 3)} | {fmt(s['reasoning_share'], 3)} | {fmt(s['ttft_mean_s'], 2)} | {s['hit_max_tokens']} | {s['failed']} |")
    waves = sorted({r["wave"] for r in run["c4"]})
    if waves:
        lines += ["", "| c=4 wave | requests | answered | tokens | span s | aggregate tok/s | per-request median tok/s |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for w in waves:
            a = wave_aggregate([r for r in run["c4"] if r["wave"] == w])
            lines.append(f"| {w} | {a['requests']} | {a['answered']} | {a['tokens']} | {fmt(a['span_s'], 1)} | {fmt(a['aggregate_tok_s'])} | {fmt(a['median_tok_s'])} |")
        allw = c4_totals(run)
        lines.append(f"| all | {allw['requests']} | {allw['answered']} | {allw['tokens']} | {fmt(allw['span_s'], 1)} | {fmt(allw['aggregate_tok_s'])} | {fmt(allw['median_tok_s'])} |")
    return "\n".join(lines)


def c4_totals(run: dict) -> dict:
    """All c = 4 waves together: tokens over the summed spans of the waves (they ran one after another), and the median of all
    the requests' own rates."""
    waves = sorted({r["wave"] for r in run["c4"]})
    ag = [wave_aggregate([r for r in run["c4"] if r["wave"] == w]) for w in waves]
    ok = [a for a in ag if a["span_s"]]
    tokens, span = sum(a["tokens"] for a in ok), sum(a["span_s"] for a in ok)
    rates = [r["tok_s"] for r in run["c4"] if not r.get("error") and r.get("tok_s") is not None]
    return {"requests": sum(a["requests"] for a in ag), "answered": sum(a["answered"] for a in ag), "tokens": tokens, "span_s": span,
            "aggregate_tok_s": tokens / span if span else None, "median_tok_s": statistics.median(rates) if rates else None}


def compare_report(a: dict, b: dict) -> str:
    c = compare_runs(a, b)
    lines = ["| category | effort | temp | A/B tok/s (means) | A/B tok/s (medians) |", "|---|---|---:|---:|---:|"]
    for (cat, effort, t), r in c["ratios"].items():
        lines.append(f"| {cat} | {effort} | {t} | {r['mean']:.3f} | {fmt(r['median'], 3)} |")
    lines.append("")
    for t, e in c["exact"].items():
        lines.append(f"temperature {t}: {e['identical']} of {e['pairs']} replies identical id for id"
                     + (f"; first difference {e['first_difference']}" if e["first_difference"] else ""))
    return "\n".join(lines)


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="the server's API (default: NODES and PORT, or INKLING2MINI_URL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("out")
    sub.add_parser("summary").add_argument("a")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.out, a.url)
    elif a.cmd == "summary":
        print(tables(load(a.a)))
    else:
        print(compare_report(load(a.a), load(a.b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
