#!/usr/bin/env python3
"""Does the block drafter pay? Drafted vs serial decode speed, acceptance per position and under concurrency (stdlib only).

    specbench.py --out spec.json [--url URL] [--kinds chat,code,json,longctx,tool,reasoning] [--max-tokens 512] [--passes 3]
                 [--concurrency 4] [--no-serial] [--label TEXT] [--longctx-tokens 6000]
    specbench.py --selftest          # the stats helpers on canned numbers, no server

Every request is greedy (temperature 0), non-streaming, `"prefix_cache": false` (each pass is cold and pays the same prefill in
both arms) with `"return_token_ids": true`. The serial reference is `"draft": false`, the drafted arm is `"draft": true` (the
server default). tok/s of a request = usage.completion_tokens / the wall seconds of the request (prefill included); a kind
runs serial and drafted alternately, one warm-up pass of each discarded, then --passes timed passes, the MEDIAN reported. The
drafted ids (and finish reason) must equal the serial ones on every pass: a difference is a FAIL line and exit 1.

Kinds: chat (a prose answer), code and json (bench.py's DECODE_TEXTS prompts as /v1/completions ids), longctx (a document of
about --longctx-tokens tokens of filler with a question at the end), tool (a chat turn with `tools`: ten calls asked for),
reasoning (chat with reasoning_effort high; most tokens are reasoning). Chat kinds use reasoning_effort none except
reasoning. A chat prompt's ids come from /v1/chat/completions/render (its sha256 and token count); a kind that ends early
(a tool turn is short) reports its own completion_tokens.

Per-position acceptance: after the timed passes one more SOLO drafted request runs with nothing else on the server and the
/metrics counters rexl3_spec_position_{offered,verified,reached,accepted}_total and rexl3_spec_seq_rounds_total (labels mode,
streams, position; crates/server/src/spec_metrics.rs) are read before and after. per_position lists, for position 1..N,
offered / verified / reached / accepted, the conditional acceptance accepted/reached and the plain rate accepted/rounds;
accept_hist[k] is the number of rounds that accepted exactly k drafts (derived: accepted[k] - accepted[k+1]). When the server
exports no such counters (or they do not move), per_position is null, the output says so loudly and usage.spec is all there is
(accepted drafts per round, acceptance rate, tokens per round).

--concurrency N runs N streams at once (chat, code, json, reasoning, repeated in that order), drafted and serial, median of
--passes after a discarded pass: aggregate tok/s = total completion tokens / wall of the whole batch. Every stream must equal
its solo serial reply (else its solo drafted reply, else its own first pass). --label is saved in the json (the engine's depth
and policy settings). The json is written after every kind (`"partial": true` until the end); the markdown summary goes to
stdout, progress to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402  (DECODE_TEXTS: the code and json prompts)
import inkling_api as ia  # noqa: E402
import smoke  # noqa: E402  (TOOLS: the tool schema of the tool turn)

KINDS = ("chat", "code", "json", "longctx", "tool", "reasoning")
CONC_KINDS = ("chat", "code", "json", "reasoning")
POSITION_COLUMNS = ("offered", "verified", "reached", "accepted")
FAILS: list[str] = []

CHAT_USER = ("Write a detailed, well-organized explanation, in flowing prose of at least 700 words, of how a refrigerator keeps "
             "food cold: the refrigeration cycle, the job of each component, why the freezer and the fridge compartment sit at "
             "different temperatures, and the usual causes of failure.")
TOOL_USER = ("I am planning a trip through ten cities: Paris, Berlin, Madrid, Rome, Vienna, Prague, Lisbon, Dublin, Oslo and "
             "Helsinki. Call get_weather once for each of them, all in this one turn, and say nothing else.")
REASONING_USER = ("Four colleagues, Ana, Ben, Chloe and Dev, each own exactly one of four pets (a cat, a dog, a parrot and a "
                  "rabbit) and each lives on a different floor, 1 to 4, of one building. Ana lives above the owner of the dog. "
                  "Ben lives on floor 2. The parrot's owner lives on floor 4. Chloe lives directly below Dev. The rabbit's "
                  "owner lives next to the cat's owner. Ana does not own the parrot. Work out who owns which pet and who "
                  "lives on which floor, checking every fact, and explain your reasoning step by step.")
LONG_INTRO = "Below are the notes of a town archive. Read them, then answer the question at the end.\n\n"
LONG_QUESTION = ("\n\nQuestion: Which three kinds of people appear most often in the notes above, and what do they do? Answer in "
                 "several detailed paragraphs and quote examples from the notes.")


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fail(what: str) -> None:
    FAILS.append(what)
    say(f"FAIL {what}")


# ── stats helpers (selftest covers these) ──
def median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median of nothing")
    return statistics.median(xs)


def rate(num: float, den: float, digits: int = 4) -> float | None:
    return round(num / den, digits) if den else None


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def sum_spec(replies: list[dict]) -> dict | None:
    """usage.spec summed over replies (None when none reported one) with the derived rates: acceptance (accepted / drafted),
    accepted drafts per round, tokens per round (completion tokens / rounds, as inkling_api.spec_counters)."""
    specs = [r["spec"] for r in replies if r["spec"] is not None]
    if not specs:
        return None
    d, a, rounds = (sum(s[k] for s in specs) for k in ("drafted", "accepted", "rounds"))
    n = sum(r["completion_tokens"] for r in replies if r["spec"] is not None)
    return {"replies": len(specs), "drafted": d, "accepted": a, "rounds": rounds, "acceptance": rate(a, d),
            "accepted_per_round": rate(a, rounds), "tokens_per_round": rate(n, rounds)}


_SERIES = re.compile(r"^(rexl3_spec_(?:position_(offered|verified|reached|accepted)|seq_rounds)_total)\{(.*)\}$")
_LABEL = re.compile(r'(\w+)="([^"]*)"')


def parse_spec(metrics: dict[str, float]) -> dict[tuple, float]:
    """The speculation position counters of a /metrics read: {(column, "mode/streams", position): value}; column is one of
    POSITION_COLUMNS or "rounds" (rexl3_spec_seq_rounds_total, position 0)."""
    out: dict[tuple, float] = {}
    for key, v in metrics.items():
        m = _SERIES.match(key)
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group(3)))
        col = m.group(2) or "rounds"
        group = f"{labels.get('mode', '?')}/{labels.get('streams', '?')}"
        out[(col, group, int(labels.get("position", 0)))] = v
    return out


def position_delta(before: dict[tuple, float], after: dict[tuple, float]) -> dict:
    """The counters' growth between two parse_spec reads, summed over the label groups: {"rounds": n, "groups": {group:
    rounds}, "offered"/"verified"/"reached"/"accepted": [position 1..P]} with P the highest position any column moved at."""
    moved = {k: after[k] - before.get(k, 0.0) for k in after if after[k] - before.get(k, 0.0) != 0}
    groups = {g: int(v) for (c, g, _), v in moved.items() if c == "rounds"}
    top = max((p for (c, _, p) in moved if c != "rounds"), default=0)
    out = {"rounds": sum(groups.values()), "groups": groups}
    for col in POSITION_COLUMNS:
        out[col] = [int(sum(v for (c, _, p), v in moved.items() if c == col and p == j)) for j in range(1, top + 1)]
    return out


def per_position(delta: dict) -> list[dict]:
    """One row per position: the four counters, the conditional acceptance accepted/reached (P(draft j accepted | drafts
    1..j-1 accepted)) and the plain rate accepted/rounds (P(accepted count >= j))."""
    rows = []
    for j in range(len(delta["offered"])):
        o, v, r, a = (delta[c][j] for c in POSITION_COLUMNS)
        rows.append({"position": j + 1, "offered": o, "verified": v, "reached": r, "accepted": a,
                     "conditional": rate(a, r), "rate": rate(a, delta["rounds"])})
    return rows


def accept_hist(delta: dict) -> list[int]:
    """Rounds by accepted-draft count: hist[k] rounds accepted exactly k drafts. accepted[j] counts the rounds that accepted
    at least j, so hist[k] = accepted[k] - accepted[k+1], hist[0] = rounds - accepted[1]."""
    acc = [delta["rounds"]] + delta["accepted"] + [0]
    return [acc[k] - acc[k + 1] for k in range(len(acc) - 1)]


def diff_note(a: tuple[list[int], str], b: tuple[list[int], str]) -> str | None:
    """None when (ids, finish reason) are equal, else a one-line account of the first difference."""
    if a == b:
        return None
    (x, fx), (y, fy) = a, b
    i = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return f"ids differ at {i} ({x[i:i + 3]} vs {y[i:i + 3]}), lengths {len(x)}/{len(y)}, finish {fx}/{fy}"


# ── prompts ──
def chat_spec(user: str, effort: str, tools: list | None = None) -> dict:
    body = {"messages": [{"role": "user", "content": user}], "reasoning_effort": effort}
    if tools:
        body["tools"] = tools
    return {"path": "/v1/chat/completions", "body": body, "ids": None}


def render_ids(api: ia.Api, spec: dict) -> list[int] | None:
    """The token ids the server renders a chat body to (/v1/chat/completions/render); None when it refuses."""
    body = {k: v for k, v in spec["body"].items() if k in ("messages", "tools", "reasoning_effort")}
    status, out, _ = api.call("POST", "/v1/chat/completions/render", dict(body, model=api.model))
    if status != 200 or not isinstance(out, dict) or "token_ids" not in out:
        return None
    return [int(t) for t in out["token_ids"]]


def longctx_spec(api: ia.Api, target: int) -> dict:
    """A chat prompt of about `target` rendered tokens: filler sentences between LONG_INTRO and LONG_QUESTION, the sentence
    count corrected from the rendered size (within 3 %)."""
    sample = 200
    per = len(api.tokenize(" ".join(ia.filler_sentences(sample, 91)))) / sample
    n = max(1, int(target / per))
    for _ in range(4):
        spec = chat_spec(LONG_INTRO + " ".join(ia.filler_sentences(n, 91)) + LONG_QUESTION, "none")
        ids = render_ids(api, spec)
        got = len(ids) if ids is not None else len(api.tokenize(spec["body"]["messages"][0]["content"])) + 24
        if abs(got - target) <= 0.03 * target:
            break
        n = max(1, int(n * target / got))
    return spec


def build_prompts(api: ia.Api, names, longctx_tokens: int) -> dict[str, dict]:
    out = {}
    for name in names:
        if name == "chat":
            spec = chat_spec(CHAT_USER, "none")
        elif name in ("code", "json"):
            spec = {"path": "/v1/completions", "body": {"prompt": api.tokenize(bench.DECODE_TEXTS[name])}, "ids": None}
            spec["ids"] = spec["body"]["prompt"]
        elif name == "longctx":
            spec = longctx_spec(api, longctx_tokens)
        elif name == "tool":
            spec = chat_spec(TOOL_USER, "none", smoke.TOOLS)
        elif name == "reasoning":
            spec = chat_spec(REASONING_USER, "high")
        else:
            raise SystemExit(f"unknown kind {name!r}; known: {', '.join(KINDS)}")
        if spec["ids"] is None:
            spec["ids"] = render_ids(api, spec)
        if spec["ids"] is not None:
            spec["sha256"], spec["sha256_of"] = sha_ids(spec["ids"]), "ids"
        else:
            say(f"{name}: /v1/chat/completions/render refused the body: the prompt digest is of the request body")
            spec["sha256"] = hashlib.sha256(json.dumps(spec["body"], sort_keys=True).encode()).hexdigest()
            spec["sha256_of"] = "body"
        spec["name"] = name
        out[name] = spec
    return out


# ── requests ──
def request(api: ia.Api, spec: dict, max_tokens: int, draft: bool) -> dict:
    body = {**spec["body"], "model": api.model, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True,
            "draft": draft, "prefix_cache": False}
    resp, wall = api.post(spec["path"], body)
    ids, finish = ia.generated_ids(resp)
    u = resp.get("usage") or {}
    n = int(u["completion_tokens"])
    return {"ids": ids, "finish": finish, "prompt_tokens": int(u["prompt_tokens"]), "completion_tokens": n,
            "wall_s": wall, "tok_s": n / wall, "spec": ia.spec_counters(u)}


def key(r: dict) -> tuple[list[int], str]:
    return r["ids"], r["finish"]


def arm(replies: list[dict]) -> dict:
    """The timed replies of one arm: tok/s and wall per pass, their median."""
    return {"tok_s": [round(r["tok_s"], 3) for r in replies], "wall_s": [round(r["wall_s"], 3) for r in replies],
            "median": round(median([r["tok_s"] for r in replies]), 3)}


def settle(api: ia.Api, before: dict, want_accepted: int) -> dict:
    """The speculation counters after a solo request: the engine refreshes them after the step that ends the request, which
    may be after the reply left, so read until two reads agree and have moved (at most 5 s)."""
    deadline, last = time.monotonic() + 5.0, None
    while True:
        cur = parse_spec(api.metrics())
        d = position_delta(before, cur)
        if cur == last and d["rounds"] > 0 and sum(d["accepted"]) >= want_accepted:
            return cur
        if time.monotonic() > deadline:
            return cur
        last = cur
        time.sleep(0.2)


def position_pass(api: ia.Api, name: str, spec: dict, max_tokens: int) -> tuple[list[dict] | None, list[int] | None, dict]:
    """One solo drafted request with the counters read around it: (per_position, accept_hist, info)."""
    before = parse_spec(api.metrics())
    r = request(api, spec, max_tokens, True)
    after = settle(api, before, (r["spec"] or {}).get("accepted", 0))
    d = position_delta(before, after)
    info = {"rounds": d["rounds"], "groups": d["groups"], "usage_rounds": (r["spec"] or {}).get("rounds"),
            "solo": set(d["groups"]) <= {"greedy/1"}}
    if not after or d["rounds"] == 0 or not d["offered"]:
        why = ("the server exports no rexl3_spec_position_* counters" if not after
               else "the rexl3_spec_position_* counters did not move around a drafted request")
        say(f"!!! {name}: per_position is NULL: {why}")
        info["note"] = why
        return None, None, info
    if not info["solo"]:
        say(f"!!! {name}: counters moved in groups {sorted(d['groups'])}: something else ran on the server, "
            f"the per-position numbers are not solo")
    return per_position(d), accept_hist(d), info


def run_kind(api: ia.Api, spec: dict, a) -> tuple[dict, tuple[list[int], str]]:
    """(the kind's result, its reference reply key: the first serial reply, else the first drafted one)."""
    name = spec["name"]
    serial_all: list[dict] = []
    drafted_all: list[dict] = []
    for _ in range(a.passes + 1):  # pass 0 is the warm-up of both arms
        if not a.no_serial:
            serial_all.append(request(api, spec, a.max_tokens, False))
        drafted_all.append(request(api, spec, a.max_tokens, True))
    ref = key(serial_all[0]) if serial_all else None
    for r in serial_all:
        d = diff_note(key(r), ref)
        if d:
            fail(f"{name}: the serial reply is not stable across passes: {d}")
        if r["spec"] and r["spec"]["drafted"]:
            fail(f"{name}: a serial reply (\"draft\": false) drafted {r['spec']['drafted']} tokens")
    base = ref or key(drafted_all[0])
    for i, r in enumerate(drafted_all):
        d = diff_note(key(r), base)
        if d:
            fail(f"{name}: drafted pass {i} != {'serial' if ref else 'drafted pass 0'}: {d}")
    equal = all(key(r) == base for r in drafted_all) if ref else None
    timed_s, timed_d = serial_all[1:], drafted_all[1:]
    spec_sum = sum_spec(timed_d)
    if spec_sum is None:
        say(f"!!! {name}: the drafted replies carry no usage.spec")
    elif not spec_sum["drafted"]:
        fail(f"{name}: the drafted replies drafted 0 tokens: the server is not drafting")
    pp, hist, info = position_pass(api, name, spec, a.max_tokens)
    drafted = {**arm(timed_d), "spec": spec_sum, "per_position": pp, "accept_hist": hist, "position_pass": info,
               "tokens_per_round": (spec_sum or {}).get("tokens_per_round")}
    serial = arm(timed_s) if timed_s else None
    out = {"prompt_tokens": drafted_all[0]["prompt_tokens"], "prompt_sha256": spec["sha256"],
           "prompt_sha256_of": spec["sha256_of"], "path": spec["path"],
           "completion_tokens": drafted_all[0]["completion_tokens"], "finish_reason": drafted_all[0]["finish"],
           "generated_sha256": sha_ids(drafted_all[0]["ids"]), "serial": serial, "drafted": drafted, "equal": equal,
           "speedup": rate(drafted["median"], serial["median"], 3) if serial else None}
    say(f"{name}: {out['prompt_tokens']} prompt / {out['completion_tokens']} generated tokens; serial "
        f"{serial['median'] if serial else '-'} tok/s, drafted {drafted['median']} tok/s, speedup {out['speedup']}, "
        f"equal {equal}, tokens/round {drafted['tokens_per_round']}")
    return out, base


# ── concurrency ──
def run_batch(api: ia.Api, specs: list[dict], max_tokens: int, draft: bool) -> tuple[list[dict], float]:
    """All requests at once (a barrier): the replies and the wall span from the first start to the last end."""
    barrier = threading.Barrier(len(specs))

    def one(spec):
        barrier.wait()
        t0 = time.monotonic()
        r = request(api, spec, max_tokens, draft)
        return r, t0, time.monotonic()

    with ThreadPoolExecutor(len(specs)) as pool:
        out = list(pool.map(one, specs))
    return [r for r, _, _ in out], max(t for _, _, t in out) - min(t for _, t, _ in out)


def run_concurrency(api: ia.Api, prompts: dict[str, dict], refs: dict[str, tuple[list[int], str]], a) -> dict:
    n = a.concurrency
    names = [CONC_KINDS[i % len(CONC_KINDS)] for i in range(n)]
    specs = [prompts[k] for k in names]
    arms = {}
    for label, draft in (("serial", False), ("drafted", True)):
        if label == "serial" and a.no_serial:
            continue
        passes = []
        for i in range(a.passes + 1):  # pass 0: warm-up, discarded
            replies, span = run_batch(api, specs, a.max_tokens, draft)
            if i:
                passes.append((replies, span))
        arms[label] = passes
        for i, (replies, _) in enumerate(passes):
            for j, (k, r) in enumerate(zip(names, replies)):
                if label == "serial" and r["spec"] and r["spec"]["drafted"]:
                    fail(f"concurrency {k}#{j}: a serial stream drafted {r['spec']['drafted']} tokens")
                d = diff_note(key(r), refs[k])
                if d:
                    fail(f"concurrency {label} pass {i + 1} stream {k}#{j} != its solo reference: {d}")
    out = {"streams": n, "kinds": names, "max_tokens": a.max_tokens}
    for label, passes in arms.items():
        agg = [sum(r["completion_tokens"] for r in replies) / span for replies, span in passes]
        per = []
        for j, k in enumerate(names):
            rs = [replies[j] for replies, _ in passes]
            per.append({"kind": k, "completion_tokens": rs[-1]["completion_tokens"], "tok_s": round(median([r["tok_s"] for r in rs]), 3),
                        "spec": sum_spec(rs) if label == "drafted" else None})
        out[label] = {"agg_tok_s": [round(x, 3) for x in agg], "median": round(median(agg), 3),
                      "wall_s": [round(s, 3) for _, s in passes],
                      "completion_tokens": [sum(r["completion_tokens"] for r in replies) for replies, _ in passes],
                      "streams": per}
    out.setdefault("serial", None)
    out["speedup"] = rate(out["drafted"]["median"], out["serial"]["median"], 3) if out["serial"] else None
    say(f"concurrency {n}: serial {out['serial']['median'] if out['serial'] else '-'} tok/s, drafted "
        f"{out['drafted']['median']} tok/s, speedup {out['speedup']}")
    return out


# ── markdown ──
def fmt(v, nd: int = 2) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_md(res: dict) -> str:
    m = res["meta"]
    lines = [f"# specbench {m['date']}" + (f" - {m['label']}" if m.get("label") else ""), "",
             f"model {m['model']}, {m['max_tokens']} greedy tokens, median of {m['passes']} timed passes (one warm-up discarded), "
             f"wall of the request (prefill included)", ""]
    ks = res["kinds"]
    if ks:
        lines += ["| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in ks.items():
            sp = r["drafted"]["spec"] or {}
            ad = f"{sp.get('accepted')}/{sp.get('drafted')} ({fmt(sp.get('acceptance'))})" if sp else "-"
            lines.append(f"| {k} | {r['prompt_tokens']} | {r['completion_tokens']} | {fmt(r['serial']['median'] if r['serial'] else None)} "
                         f"| {fmt(r['drafted']['median'])} | {fmt(r['speedup'], 3)} | {ad} | {fmt(sp.get('accepted_per_round'))} "
                         f"| {fmt(r['drafted']['tokens_per_round'])} | {fmt(r['equal'])} |")
        lines.append("")
        pp = {k: r["drafted"]["per_position"] for k, r in ks.items()}
        if any(v is None for v in pp.values()):
            lines += ["**per_position: null for " + ", ".join(k for k, v in pp.items() if v is None) +
                      "** - the server exposes no per-position counters (or they did not move); usage.spec above is all there is", ""]
        have = {k: v for k, v in pp.items() if v}
        if have:
            width = max(len(v) for v in have.values())
            head = "| kind | " + " | ".join(f"p{j}" for j in range(1, width + 1)) + " |"
            sep = "|---|" + "---|" * width
            lines += ["Conditional acceptance accepted/reached per draft position:", "", head, sep]
            for k, v in have.items():
                lines.append(f"| {k} | " + " | ".join(fmt(v[j]["conditional"]) if j < len(v) else "-" for j in range(width)) + " |")
            lines += ["", "Rounds by accepted-draft count (solo pass):", "",
                      "| kind | " + " | ".join(f"{j}" for j in range(width + 1)) + " |", "|---|" + "---|" * (width + 1)]
            for k in have:
                h = ks[k]["drafted"]["accept_hist"] or []
                lines.append(f"| {k} | " + " | ".join(str(h[j]) if j < len(h) else "-" for j in range(width + 1)) + " |")
            lines.append("")
    c = res.get("concurrency")
    if c:
        lines += [f"Concurrency {c['streams']} ({', '.join(c['kinds'])}), aggregate tok/s over the wall of the batch:", "",
                  "| arm | agg tok/s | tokens | per stream tok/s |", "|---|---|---|---|"]
        for label in ("serial", "drafted"):
            if c.get(label):
                b = c[label]
                lines.append(f"| {label} | {fmt(b['median'])} | {b['completion_tokens'][-1]} | "
                             + ", ".join(f"{s['kind']} {fmt(s['tok_s'])} ({s['completion_tokens']})" for s in b["streams"]) + " |")
        lines += ["", f"speedup {fmt(c['speedup'], 3)}", ""]
    for f in res["failures"]:
        lines.append(f"FAIL {f}")
    lines.append("PASS drafted == serial" if not res["failures"] else "FAIL drafted == serial or a check above")
    return "\n".join(lines)


def save(path: str, res: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


# ── selftest ──
def selftest() -> int:
    assert median([3.0, 1.0, 2.0]) == 2.0 and median([4.0, 1.0, 2.0, 3.0]) == 2.5 and median([7.0]) == 7.0
    try:
        median([])
        raise AssertionError("median([]) must raise")
    except ValueError:
        pass
    assert rate(1, 0) is None and rate(1, 3) == 0.3333 and rate(10, 4, 3) == 2.5

    def series(rounds: int, acc: list[int], mode="greedy", streams="1") -> dict[str, float]:
        m = {f'rexl3_spec_seq_rounds_total{{mode="{mode}",streams="{streams}"}}': float(rounds)}
        for j, v in enumerate(acc, 1):
            lab = f'mode="{mode}",streams="{streams}",position="{j}"'
            m[f"rexl3_spec_position_accepted_total{{{lab}}}"] = float(v)
            m[f"rexl3_spec_position_offered_total{{{lab}}}"] = float(rounds)
            m[f"rexl3_spec_position_verified_total{{{lab}}}"] = float(rounds)
            m[f"rexl3_spec_position_reached_total{{{lab}}}"] = float(rounds if j == 1 else acc[j - 2] if j - 2 < len(acc) else 0)
        m['rexl3_round_streams_total{streams="1"}'] = 99.0  # an unrelated series: ignored
        return m

    before = parse_spec(series(10, [8, 6, 3, 2]))
    after = parse_spec(series(30, [26, 18, 9, 5]))
    assert len(parse_spec(series(1, [1]))) == 5 and parse_spec({"rexl3_other_total": 1.0}) == {}
    d = position_delta(before, after)
    assert d["rounds"] == 20 and d["groups"] == {"greedy/1": 20}, d
    assert d["accepted"] == [18, 12, 6, 3] and d["offered"] == [20, 20, 20, 20], d
    assert d["reached"] == [20, 18, 12, 6], d
    rows = per_position(d)
    assert [r["position"] for r in rows] == [1, 2, 3, 4]
    assert rows[1]["conditional"] == round(12 / 18, 4) and rows[0]["rate"] == 0.9 and rows[3]["conditional"] == 0.5
    assert accept_hist(d) == [2, 6, 6, 3, 3] and sum(accept_hist(d)) == d["rounds"]
    assert position_delta({}, {}) == {"rounds": 0, "groups": {}, "offered": [], "verified": [], "reached": [], "accepted": []}
    other = parse_spec(series(30, [26], streams="2+"))
    assert set(position_delta(before, {**after, **other})["groups"]) == {"greedy/1", "greedy/2+"}
    s = sum_spec([{"spec": {"drafted": 70, "accepted": 35, "rounds": 10}, "completion_tokens": 45},
                  {"spec": {"drafted": 70, "accepted": 21, "rounds": 10}, "completion_tokens": 31},
                  {"spec": None, "completion_tokens": 9}])
    assert s == {"replies": 2, "drafted": 140, "accepted": 56, "rounds": 20, "acceptance": 0.4, "accepted_per_round": 2.8,
                 "tokens_per_round": 3.8}, s
    assert sum_spec([{"spec": None, "completion_tokens": 1}]) is None
    assert diff_note(([1, 2, 3], "stop"), ([1, 2, 3], "stop")) is None
    assert "differ at 1" in diff_note(([1, 2, 3], "stop"), ([1, 9, 3], "stop"))
    assert "finish" in diff_note(([1], "stop"), ([1], "length"))
    kind = {"prompt_tokens": 30, "completion_tokens": 512, "serial": {"median": 20.0, "tok_s": [20.0], "wall_s": [25.0]},
            "drafted": {"median": 40.0, "spec": s, "tokens_per_round": 3.8, "per_position": rows, "accept_hist": accept_hist(d)},
            "equal": True, "speedup": 2.0}
    res = {"meta": {"date": "d", "label": "L", "model": "m", "max_tokens": 512, "passes": 3},
           "kinds": {"chat": kind, "code": {**kind, "drafted": {**kind["drafted"], "per_position": None, "accept_hist": None}}},
           "concurrency": None, "failures": []}
    md = render_md(res)
    assert "per_position: null for code" in md and "| chat | 30 | 512 | 20.00 | 40.00 | 2.000 |" in md and md.endswith("PASS drafted == serial")
    print("selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", help="the result json (required unless --selftest)")
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma list, a subset of: %(default)s")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--passes", type=int, default=3, help="timed passes per arm after one discarded warm-up (default: %(default)s)")
    ap.add_argument("--concurrency", type=int, default=0, help="also run N streams at once (chat, code, json, reasoning, repeated)")
    ap.add_argument("--no-serial", action="store_true", help="drafted arm only: no serial reference, no equality or speedup")
    ap.add_argument("--label", default="", help="free text saved in the json (the engine's depth/policy settings)")
    ap.add_argument("--longctx-tokens", type=int, default=6000, help="size of the longctx prompt (default: %(default)s)")
    ap.add_argument("--selftest", action="store_true", help="run the stats helpers on canned numbers and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.out:
        ap.error("--out is required")
    kinds = [k for k in a.kinds.split(",") if k]
    bad = [k for k in kinds if k not in KINDS]
    if bad or not kinds:
        ap.error(f"--kinds: unknown {bad}; known: {', '.join(KINDS)}")
    if a.passes < 1 or a.max_tokens < 1 or a.concurrency < 0:
        ap.error("--passes and --max-tokens must be positive, --concurrency not negative")

    api = ia.api_from_args(a)
    need = list(dict.fromkeys(kinds + (list(CONC_KINDS) if a.concurrency else [])))
    prompts = build_prompts(api, need, a.longctx_tokens)
    res = {"meta": {"url": api.url, "model": api.model, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label": a.label, "passes": a.passes, "max_tokens": a.max_tokens, "kinds": kinds,
                    "no_serial": a.no_serial, "concurrency": a.concurrency},
           "kinds": {}, "concurrency": None, "failures": FAILS, "partial": True}
    refs: dict[str, tuple[list[int], str]] = {}
    for k in kinds:
        res["kinds"][k], refs[k] = run_kind(api, prompts[k], a)
        save(a.out, res)
    if a.concurrency:
        # the batch's kinds that --kinds left out get one solo reference request (serial; drafted under --no-serial)
        for k in dict.fromkeys(CONC_KINDS[i % len(CONC_KINDS)] for i in range(a.concurrency)):
            if k not in refs:
                refs[k] = key(request(api, prompts[k], a.max_tokens, a.no_serial))
        res["concurrency"] = run_concurrency(api, prompts, refs, a)
    res["partial"] = False
    save(a.out, res)
    print(render_md(res))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
