#!/usr/bin/env python3
"""Edge cases of drafted == serial against ONE running server booted with a drafter (stdlib only).

    specedge.py --out edge.json [--url URL] [--only eos_in_block,stop_string_in_block,...] [--chunk N] [--label TEXT]
    specedge.py --selftest          # the helpers on canned data, no server

gates.py `spec` covers short/medium/long/xlong/70k solo and concurrent; this is the other half: the places a speculative block
can go wrong. Every property compares the drafted reply (`"draft": true, "prefix_cache": false`) with the serial reference
(`"draft": false, "prefix_cache": false`), greedy, on generated ids + finish_reason + the returned text, /v1/completions with
token-id prompts (/v1/chat/completions where a chat template is needed). Each case records both replies' lengths, finish
reasons and usage.spec (drafted/accepted/rounds) and the first difference; the json holds every case. Exit 1 on any failure.

  eos_in_block             six short chat answers (reasoning_effort none) that end on the EOS: ids with EOS, finish 'stop'
  stop_string_in_block     5 raw prompts x 2 stop strings picked from the greedy text at tokens 20..60: text, finish 'stop'
  max_tokens_in_block      max_tokens 1..10, 15, 17 on 2 prompts: ids equal, finish 'length', exactly max_tokens tokens
  ring_wrap                prompts of 1016 1023 1024 1030 1031 1032 1040 2060 tokens + 160 generated: the decode crosses the
                           drafter's 1,024 window, its 1,032 ring and the target's 1,031 ring
  long_prompt_chunking     12,345 tokens + 192 generated (--chunk N adds 3N+1, 3N-1 and 2N: the prefill chunk grid, 4096 by
                           default, ends on a partial chunk)
  rollback_after_full_reject  four long continuations solo with the /metrics counters (rexl3_spec_position_*, as specbench.py
                           reads them) around every drafted request of the run: the histogram of accepted counts per round. WARN
                           (not a failure) when no round accepted 0 drafts or none accepted 7: the property was not exercised.
                           Equality of every reply is the proof of the rollback
  copy_hybrid              the copy hybrid (`REXL3_INKLING2_LOOP_SPEC_COPY=on`): replies whose draft is a copy of the stream's own text, at
                           max_tokens 1..8 past a repeat, with stop strings picked in the repeating stretch, each equal to its serial
                           reference; the /metrics copy counters (rexl3_spec_copy_*) around every drafted request say whether a copy
                           draft rode (the property FAILS when none did), was rejected at its first token, in the middle (positions
                           2..6) and at its last (position 7; WARN when not seen), and which stop-string requests ended on a round
                           that had copy drafts
  concurrent_mixed         four at once: a short EOS reply, a 1,031-token prompt, a stop-string request, a max_tokens=5 request,
                           each equal to its solo serial reference (three rounds)

Not observable: whether a reply STOPPED INSIDE a draft block (the server crate exports no `stopped_in_draft`); eos_in_block
reports how many drafted replies drafted at all (usage.spec drafted > 0) instead. A run in which no drafted reply drafted a token
fails (equality would be trivial). The stop-string and /metrics reads need the server's detokenised `text` and /metrics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402  (DECODE_TEXTS)
import exactness as ex  # noqa: E402  (TEXTS)
import inkling_api as ia  # noqa: E402
import specbench as sb  # noqa: E402  (the /metrics position-counter helpers)

EOS_PROMPTS = ["Reply with exactly the word OK.", "Reply with exactly the word YES.",
               "What is 2 + 2? Answer with just the number.", "Say 'done' and nothing else.",
               "Reply with exactly: pong", "Name the capital of France in one word."]
MAX_TOKENS_LIST = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 17)
RING_LENGTHS = (1016, 1023, 1024, 1030, 1031, 1032, 1040, 2060)
CHUNK_PROMPT = 12_345
MAX_DRAFTS = 7  # a fully accepted block: the drafter proposes 7 per round
STOP_PROMPTS = ("prose", "code", "structured", "json", "medium")
STOP_TOKENS = 128
WORD = re.compile(r"[A-Za-z][A-Za-z']{2,}")
ANY = re.compile(r"\S{3,8}")


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── pure helpers (selftest covers these) ──
def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def trim_to(blocks: list[list[int]], n: int) -> list[int]:
    """Exactly n ids from the front of the concatenated blocks (ValueError when they hold fewer)."""
    out: list[int] = []
    for b in blocks:
        out += b
        if len(out) >= n:
            return out[:n]
    raise ValueError(f"{len(out)} ids, not {n}")


def pick_stops(full: str, t20: str, t60: str, n: int = 2) -> list[str]:
    """n stop strings out of `full` (the greedy text): words (else 3..8 character runs, else any run) lying between the text of the first
    20 tokens (t20) and of the first 60 (t60) whose FIRST occurrence in `full` is also in that stretch, so the reply stops
    mid-stream and not before. The first and the last such candidate when n = 2."""
    lo, hi = len(t20), max(len(t60), len(t20))
    seen: list[str] = []
    for rx in (WORD, ANY):
        for m in rx.finditer(full[lo:hi]):
            w = m.group(0)
            i = full.find(w)
            if w not in seen and lo <= i < hi:
                seen.append(w)
        if len(seen) >= n:
            break
    if len(seen) < n:  # repetitive text: any 3..24 character run that first occurs inside the stretch
        for i in range(lo, hi):
            for k in range(3, 25):
                w = full[i:i + k]
                if w == w.strip() and full.find(w) == i and w not in seen:
                    seen.append(w)
                    break
            if len(seen) >= n and i > lo + (hi - lo) // 2:
                break
    if len(seen) < n:
        raise ValueError(f"only {len(seen)} usable stop strings between characters {lo} and {hi}")
    return [seen[0], seen[-1]] if n == 2 else seen[:n]


def hist_add(total: list[int], h: list[int]) -> list[int]:
    out = list(total) + [0] * (len(h) - len(total))
    for i, v in enumerate(h):
        out[i] += v
    return out


def exercise_warnings(hist: list[int]) -> list[str]:
    """WARN texts for an accepted-count histogram (hist[k] = rounds that accepted exactly k drafts) that never saw a fully
    rejected block (k = 0) or a fully accepted one (k = MAX_DRAFTS)."""
    out = []
    if not hist or not sum(hist):
        return ["no per-position counter moved: neither a fully rejected nor a fully accepted block can be told to have run"]
    if not hist[0]:
        out.append("no round accepted 0 drafts: a fully rejected block (the rollback) was not exercised")
    if len(hist) <= MAX_DRAFTS or not hist[MAX_DRAFTS]:
        out.append(f"no round accepted {MAX_DRAFTS} drafts: a fully accepted block was not exercised")
    return out


def diff_reply(s: dict, d: dict, finish: str | None = None, tokens: int | None = None) -> str | None:
    """None when drafted `d` equals serial `s` (ids, finish reason, text) and meets the expectations (`finish` in both,
    `tokens` completion tokens in both), else a one-line account of the first difference."""
    note = sb.diff_note((s["ids"], s["finish"]), (d["ids"], d["finish"]))
    if note:
        return "drafted != serial: " + note
    if s["text"] != d["text"]:
        i = next((j for j, (x, y) in enumerate(zip(s["text"], d["text"])) if x != y), min(len(s["text"]), len(d["text"])))
        return f"drafted != serial: text differs at character {i} ({s['text'][i:i + 20]!r} vs {d['text'][i:i + 20]!r})"
    if finish is not None and s["finish"] != finish:
        return f"finish_reason {s['finish']!r} in both modes, expected {finish!r}"
    for r, who in ((s, "serial"), (d, "drafted")):
        if tokens is not None and r["completion_tokens"] != tokens:
            return f"{who} generated {r['completion_tokens']} tokens, expected {tokens}"
    return None


def side(r: dict) -> dict:
    return {"completion_tokens": r["completion_tokens"], "ids_len": len(r["ids"]), "ids_sha256": digest(r["ids"]),
            "finish_reason": r["finish"], "spec": r["spec"], "text_len": len(r["text"])}


def case_record(name: str, prompt_tokens: int | None, s: dict, d: dict, diff: str | None, **more) -> dict:
    return {"name": name, "pass": diff is None, "prompt_tokens": prompt_tokens, "serial": side(s), "drafted": side(d),
            "difference": diff, **more}


def drafted_gt0(cases: list[dict]) -> int:
    return sum(1 for c in cases if (c.get("drafted") or {}).get("spec") and c["drafted"]["spec"]["drafted"] > 0)


# ── requests ──
def call(api: ia.Api, spec: dict, max_tokens: int, draft: bool) -> dict:
    """One greedy request: {ids, finish, text, completion_tokens, prompt_tokens, spec}. A chat reply's text joins its
    reasoning, content and tool calls (name + arguments; the call ids are random and left out)."""
    body = {**spec["body"], "model": api.model, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True,
            "draft": draft, "prefix_cache": False}
    resp, _ = api.post(spec["path"], body)
    ids, finish = ia.generated_ids(resp)
    ch = resp["choices"][0]
    if spec["path"].endswith("/chat/completions"):
        msg = ch.get("message") or {}
        calls = [((c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments"))
                 for c in msg.get("tool_calls") or []]
        text = json.dumps([msg.get("reasoning_content") or "", msg.get("content") or "", calls])
    else:
        text = ch.get("text") or ""
    u = resp.get("usage") or {}
    return {"ids": ids, "finish": finish, "text": text, "completion_tokens": int(u.get("completion_tokens", len(ids))),
            "prompt_tokens": int(u.get("prompt_tokens", 0)), "spec": ia.spec_counters(u)}


def completion_spec(ids: list[int], **more) -> dict:
    return {"path": "/v1/completions", "body": {"prompt": ids, **more}}


def chat_spec(user: str, effort: str = "none") -> dict:
    return {"path": "/v1/chat/completions",
            "body": {"messages": [{"role": "user", "content": user}], "reasoning_effort": effort}}


class Tracker:
    """The accepted-count histogram of the drafted requests that ran SOLO, from the /metrics position counters read around
    each (serial requests also record rounds, so only drafted ones are counted). Switches itself off, with the reason, when the
    server exports no such counters or they do not move."""

    def __init__(self, api: ia.Api):
        self.api, self.hist, self.rounds, self.requests, self.problem = api, [], 0, 0, None

    def before(self):
        if self.problem:
            return None
        try:
            return sb.parse_spec(self.api.metrics())
        except (ia.ApiError, OSError) as e:
            self.problem = f"/metrics unreadable: {e}"
            return None

    def after(self, before, reply: dict) -> None:
        if before is None or self.problem:
            return
        want = (reply["spec"] or {}).get("accepted", 0)
        if not (reply["spec"] or {}).get("rounds"):
            return  # no round ran (max_tokens 1, or no spec block): nothing to wait for in the counters
        try:
            d = sb.position_delta(before, sb.settle(self.api, before, want))
        except (ia.ApiError, OSError) as e:
            self.problem = f"/metrics unreadable: {e}"
            return
        if d["rounds"] == 0:  # the reply ran rounds, the counters did not move: the server does not export them
            self.problem = "the rexl3_spec_position_* counters did not move around a drafted request"
            return
        self.hist = hist_add(self.hist, sb.accept_hist(d))
        self.rounds += d["rounds"]
        self.requests += 1


class Ctx:
    def __init__(self, api: ia.Api, chunk: int | None, conc_rounds: int):
        self.api, self.chunk, self.conc_rounds = api, chunk, conc_rounds
        self.tracker = Tracker(api)
        self._blocks: list[list[int]] = []
        self._ids: dict[str, list[int]] = {}
        self._stops: list[dict] | None = None

    def named(self, name: str) -> list[int]:
        """Token ids of a fixed prompt text (bench.DECODE_TEXTS or exactness.TEXTS)."""
        if name not in self._ids:
            text = bench.DECODE_TEXTS.get(name) or ex.TEXTS[name]
            self._ids[name] = self.api.tokenize(text)
        return self._ids[name]

    def stream(self, n: int) -> list[int]:
        """Exactly n ids of one fixed filler stream (blocks of 200 varied sentences tokenized one by one, as bench.py does)."""
        while True:
            try:
                return trim_to(self._blocks, n)
            except ValueError:
                self._blocks.append(self.api.tokenize(" ".join(ia.filler_sentences(200, 6100 + len(self._blocks)))))

    def pair(self, name: str, spec: dict, max_tokens: int, prompt_tokens: int | None = None, finish: str | None = None,
             tokens: int | None = None, track: bool = True, **more) -> dict:
        """The serial reference then the drafted reply of one request, compared: a case record. A server error is a failed
        case, not a crash."""
        try:
            s = call(self.api, spec, max_tokens, False)
            before = self.tracker.before() if track else None
            d = call(self.api, spec, max_tokens, True)
            if track:
                self.tracker.after(before, d)
        except (ia.ApiError, OSError) as e:
            say(f"  {name}: ERROR {e}")
            return {"name": name, "pass": False, "prompt_tokens": prompt_tokens, "serial": None, "drafted": None,
                    "difference": f"request failed: {e}", **more}
        diff = diff_reply(s, d, finish, tokens)
        rec = case_record(name, prompt_tokens or d["prompt_tokens"], s, d, diff, **more)
        say(f"  {name}: {'equal' if diff is None else 'FAIL ' + diff} (serial {len(s['ids'])} ids {s['finish']}, drafted "
            f"{len(d['ids'])} ids {d['finish']}, spec {d['spec'] and {k: d['spec'][k] for k in ('drafted', 'accepted', 'rounds')}})")
        return rec

    def stop_cases(self) -> list[dict]:
        """5 prompts x 2 stop strings: the greedy 96-token text of each prompt (serial) and the text of its first 20 and 60
        tokens give the stretch the stop strings are picked from (tokens 20..60)."""
        if self._stops is None:
            out = []
            for name in STOP_PROMPTS:
                ids = self.named(name)
                texts = {n: call(self.api, completion_spec(ids), n, False)["text"] for n in (20, 60, 96)}
                try:
                    out += [{"prompt": name, "ids": ids, "stop": s} for s in pick_stops(texts[96], texts[20], texts[60])]
                except ValueError as e:  # this prompt's greedy text offers no usable stop string: a failed case, not a crash
                    out.append({"prompt": name, "ids": ids, "stop": None, "error": str(e)})
            self._stops = out
        return self._stops


def summarize(cases: list[dict], **more) -> dict:
    return {"cases": len(cases), "failed": sum(1 for c in cases if not c["pass"]), "drafted_replies_with_drafts": drafted_gt0(cases),
            **more}


# ── properties ──
def eos_in_block(ctx: Ctx) -> dict:
    cases = [ctx.pair(f"eos/{i}", chat_spec(p), 64, finish="stop") for i, p in enumerate(EOS_PROMPTS)]
    n = drafted_gt0(cases)
    say(f"  {n}/{len(cases)} drafted replies drafted tokens; whether the stop fell inside a draft is not observable "
        "(no stopped_in_draft in the server crate)")
    return {"cases": cases, "summary": summarize(cases, stopped_in_draft="not observable: the server exports no stopped_in_draft")}


def stop_string_in_block(ctx: Ctx) -> dict:
    cases = []
    for sc in ctx.stop_cases():
        if sc["stop"] is None:
            cases.append({"name": f"stop/{sc['prompt']}", "pass": False, "difference": sc["error"], "prompt": sc["prompt"]})
            continue
        spec = completion_spec(sc["ids"], stop=[sc["stop"]])
        c = ctx.pair(f"stop/{sc['prompt']}/{sc['stop']!r}", spec, STOP_TOKENS, finish="stop", prompt=sc["prompt"], stop=sc["stop"])
        cases.append(c)
    return {"cases": cases, "summary": summarize(cases)}


def max_tokens_in_block(ctx: Ctx) -> dict:
    cases = []
    for p in ("prose", "code"):
        spec = completion_spec(ctx.named(p))
        for mt in MAX_TOKENS_LIST:
            cases.append(ctx.pair(f"max_tokens/{p}/{mt}", spec, mt, finish="length", tokens=mt))
    return {"cases": cases, "summary": summarize(cases)}


def stream_cases(ctx: Ctx, prefix: str, lengths, gen: int) -> list[dict]:
    return [ctx.pair(f"{prefix}/{n}", completion_spec(ctx.stream(n)), gen, prompt_tokens=n) for n in lengths]


def ring_wrap(ctx: Ctx) -> dict:
    cases = stream_cases(ctx, "ring", RING_LENGTHS, 160)
    return {"cases": cases, "summary": summarize(cases)}


def long_prompt_chunking(ctx: Ctx) -> dict:
    lengths = [CHUNK_PROMPT]
    if ctx.chunk:
        lengths += [3 * ctx.chunk + 1, 3 * ctx.chunk - 1, 2 * ctx.chunk]
    cases = stream_cases(ctx, "chunking", lengths, 192)
    return {"cases": cases, "summary": summarize(cases, chunk=ctx.chunk, lengths=lengths)}


def rollback_after_full_reject(ctx: Ctx) -> dict:
    cases = [ctx.pair(f"rollback/{p}", completion_spec(ctx.named(p)), 192) for p in ("prose", "code", "structured", "json")]
    t = ctx.tracker
    warnings = exercise_warnings(t.hist) if not t.problem else [t.problem + ": no accepted-count histogram"]
    for w in warnings:
        say(f"WARN rollback_after_full_reject: {w}")
    say(f"  accepted-count histogram over {t.rounds} rounds of {t.requests} solo drafted requests: {t.hist}")
    return {"cases": cases, "warnings": warnings,
            "summary": summarize(cases, accepted_count_histogram=t.hist, rounds=t.rounds, solo_drafted_requests=t.requests,
                                 rounds_accepting_0=t.hist[0] if t.hist else None,
                                 rounds_accepting_7=t.hist[MAX_DRAFTS] if len(t.hist) > MAX_DRAFTS else 0,
                                 counters=t.problem or "ok")}


_COPY = re.compile(r"^rexl3_spec_copy_(?:position_(offered|verified|reached|accepted)|rounds)_total\{(.*)\}$")


def parse_copy(metrics: dict[str, float]) -> dict[str, list[float]]:
    """The copy hybrid's /metrics counters (summed over modes): {"rounds": [n], "verified"/"reached"/"accepted": [position 1..7]}."""
    out: dict[str, list[float]] = {"rounds": [0.0], **{c: [0.0] * MAX_DRAFTS for c in ("offered", "verified", "reached", "accepted")}}
    for key, v in metrics.items():
        m = _COPY.match(key)
        if not m:
            continue
        if m.group(1) is None:
            out["rounds"][0] += v
        else:
            pos = int(dict(re.findall(r'(\w+)="([^"]*)"', m.group(2))).get("position", 0))
            if 1 <= pos <= MAX_DRAFTS:
                out[m.group(1)][pos - 1] += v
    return out


def copy_growth(before: dict[str, list[float]], after: dict[str, list[float]]) -> dict[str, list[int]]:
    return {k: [int(a - b) for a, b in zip(after[k], before[k])] for k in after}


def copy_hybrid(ctx: Ctx) -> dict:
    api = ctx.api
    total = {k: [0] * (1 if k == "rounds" else MAX_DRAFTS) for k in ("rounds", "offered", "verified", "reached", "accepted")}
    stop_with_copy: list[str] = []
    cases: list[dict] = []

    def pair(name: str, spec: dict, max_tokens: int, **more) -> dict:
        try:
            s = call(api, spec, max_tokens, False)
            before = parse_copy(api.metrics())
            d = call(api, spec, max_tokens, True)
            g = copy_growth(before, parse_copy(api.metrics()))
        except (ia.ApiError, OSError) as e:
            say(f"  {name}: ERROR {e}")
            return {"name": name, "pass": False, "serial": None, "drafted": None, "difference": f"request failed: {e}", **more}
        for k in total:
            total[k] = [a + b for a, b in zip(total[k], g[k])]
        diff = diff_reply(s, d, more.get("finish"), more.get("tokens"))
        rec = case_record(name, d["prompt_tokens"], s, d, diff, copy_rounds=g["rounds"][0], **{k: v for k, v in more.items() if k not in ("finish", "tokens")})
        if more.get("finish") == "stop" and g["rounds"][0] > 0 and diff is None:
            stop_with_copy.append(name)
        say(f"  {name}: {'equal' if diff is None else 'FAIL ' + diff} (copy rounds {g['rounds'][0]}, serial {len(s['ids'])} ids {s['finish']}, drafted {len(d['ids'])} ids {d['finish']})")
        return rec

    for p in ("structured", "json", "code", "medium"):
        ids = ctx.named(p)
        cases.append(pair(f"copy/{p}/full", completion_spec(ids), 320))
        for mt in (1, 2, 3, 4, 5, 6, 7, 8):
            cases.append(pair(f"copy/{p}/max{mt}", completion_spec(ids), mt, finish="length", tokens=mt))
        # stop strings from the stretches where the text repeats itself (tokens 60..300 of the serial reply), the first one a draft can cross
        full = call(api, completion_spec(ids), 320, False)["text"]
        t = {n: call(api, completion_spec(ids), n, False)["text"] for n in (60, 140, 220, 300)}
        for lo, hi in ((60, 140), (140, 220), (220, 300)):
            try:
                stops = pick_stops(full, t[lo], t[hi], n=2)
            except ValueError as e:
                say(f"  copy/{p}: no stop string between tokens {lo} and {hi}: {e}")
                continue
            for st in stops:
                cases.append(pair(f"copy/{p}/stop/{lo}-{hi}/{st!r}", completion_spec(ids, stop=[st]), 320, finish="stop", stop=st))
    rej = [r - a for r, a in zip(total["reached"], total["accepted"])]
    warnings: list[str] = []
    if total["rounds"][0] == 0:
        cases.append({"name": "copy/exercised", "pass": False, "difference": "no copy draft rode a window: the hybrid is off or never chose a copy"})
    if rej[0] == 0:
        warnings.append("no copy draft was rejected at its first token")
    if not any(rej[1:MAX_DRAFTS - 1]):
        warnings.append("no copy draft was rejected in the middle (positions 2..6)")
    if rej[MAX_DRAFTS - 1] == 0:
        warnings.append("no 7-token copy draft was rejected at its last token")
    if total["rounds"][0] and not stop_with_copy:
        warnings.append("no stop-string request ran a copy round")
    for w in warnings:
        say(f"WARN copy_hybrid: {w}")
    say(f"  copy rounds {total['rounds'][0]}; reached {total['reached']}; accepted {total['accepted']}; rejected at position 1..7: {rej}")
    return {"cases": cases, "warnings": warnings,
            "summary": summarize(cases, copy_rounds=total["rounds"][0], copy_reached=total["reached"], copy_accepted=total["accepted"],
                                 copy_rejected_at_position=rej, stop_requests_with_copy_rounds=len(stop_with_copy))}


def concurrent_mixed(ctx: Ctx) -> dict:
    sc = next((c for c in ctx.stop_cases() if c["stop"] is not None), None)
    if sc is None:
        raise ValueError("no prompt gave a usable stop string for the stop-string request")
    jobs = [("eos", chat_spec(EOS_PROMPTS[0]), 64, None), ("ring1031", completion_spec(ctx.stream(1031)), 160, None),
            ("stop", completion_spec(sc["ids"], stop=[sc["stop"]]), STOP_TOKENS, None),
            ("max5", completion_spec(ctx.named("prose")), 5, 5)]
    cases = []
    try:
        refs = {n: call(ctx.api, spec, mt, False) for n, spec, mt, _ in jobs}
        for rnd in range(ctx.conc_rounds):
            barrier = threading.Barrier(len(jobs))

            def one(job):
                barrier.wait()
                return call(ctx.api, job[1], job[2], True)

            with ThreadPoolExecutor(len(jobs)) as pool:
                got = list(pool.map(one, jobs))
            for (n, _, _, toks), d in zip(jobs, got):
                diff = diff_reply(refs[n], d, tokens=toks)
                cases.append(case_record(f"concurrent/round{rnd}/{n}", d["prompt_tokens"], refs[n], d, diff))
                say(f"  concurrent round {rnd} {n}: {'equal' if diff is None else 'FAIL ' + diff}")
    except (ia.ApiError, OSError) as e:
        say(f"  concurrent_mixed: ERROR {e}")
        cases.append({"name": "concurrent/error", "pass": False, "difference": f"request failed: {e}"})
    return {"cases": cases, "summary": summarize(cases, rounds=ctx.conc_rounds)}


PROPERTIES = {"eos_in_block": eos_in_block, "stop_string_in_block": stop_string_in_block,
              "max_tokens_in_block": max_tokens_in_block, "ring_wrap": ring_wrap,
              "long_prompt_chunking": long_prompt_chunking, "rollback_after_full_reject": rollback_after_full_reject,
              "copy_hybrid": copy_hybrid, "concurrent_mixed": concurrent_mixed}


def run_property(ctx: Ctx, name: str) -> dict:
    say(f"{name}:")
    t0 = time.monotonic()
    try:
        res = PROPERTIES[name](ctx)
    except (ia.ApiError, OSError, ValueError) as e:  # a setup failure (tokenize, stop-string pick) fails the property
        say(f"  {name}: ERROR {e}")
        res = {"cases": [], "summary": {"error": str(e)}, "error": str(e)}
    res.setdefault("warnings", [])
    res["pass"] = bool(res["cases"]) and all(c["pass"] for c in res["cases"]) and "error" not in res
    res["seconds"] = round(time.monotonic() - t0, 1)
    say(f"{name}: {'PASS' if res['pass'] else 'FAIL'} ({res['summary'].get('failed', '-')} of {res['summary'].get('cases', '-')} cases failed)")
    return res


# ── selftest ──
class FakeApi:
    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def selftest() -> int:
    assert trim_to([[1, 2], [3, 4, 5]], 4) == [1, 2, 3, 4] and trim_to([[1, 2]], 2) == [1, 2]
    try:
        trim_to([[1]], 2)
        raise AssertionError("trim_to must raise")
    except ValueError:
        pass
    ctx = Ctx(FakeApi(), 4096, 3)
    ids = ctx.stream(1031)
    assert len(ids) == 1031 and ctx.stream(1016) == ids[:1016] and len(ctx.stream(2060)) == 2060 and len(ctx.stream(5000)) == 5000 and len(ctx._blocks) >= 2
    full = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind distant hills and quiet fields."
    t20, t60 = full[:20], full[:80]
    stops = pick_stops(full, t20, t60)
    assert len(stops) == 2 and all(full.find(s) >= len(t20) for s in stops) and stops[0] != stops[1], stops
    assert pick_stops("1 2 3 456 789 012 345 678 901 234", "1 2 3 ", "1 2 3 456 789 012 345 ")[0] == "456"
    try:
        pick_stops("aa", "a", "aa")
        raise AssertionError("pick_stops must raise")
    except ValueError:
        pass
    assert hist_add([1, 2], [0, 1, 5]) == [1, 3, 5] and hist_add([], [2]) == [2]
    assert exercise_warnings([0] * 8 + []) and len(exercise_warnings([3, 1, 0, 0, 0, 0, 0, 0])) == 1
    assert exercise_warnings([3, 1, 0, 0, 0, 0, 0, 2]) == [] and len(exercise_warnings([0, 1, 0, 0, 0, 0])) == 2
    base = {"ids": [1, 2, 3], "finish": "length", "text": "abc", "completion_tokens": 3, "spec": None}
    assert diff_reply(base, dict(base)) is None and diff_reply(base, dict(base), "length", 3) is None
    assert "ids differ at 1" in diff_reply(base, {**base, "ids": [1, 9, 3]})
    assert "text differs at character 1" in diff_reply(base, {**base, "text": "aXc"})
    assert "expected 'stop'" in diff_reply(base, dict(base), "stop")
    assert "expected 4" in diff_reply(base, dict(base), None, 4)
    d = {**base, "spec": {"drafted": 14, "accepted": 5, "rounds": 2}}
    rec = case_record("x", 10, base, d, None)
    assert rec["pass"] and rec["drafted"]["ids_len"] == 3 and drafted_gt0([rec]) == 1 and drafted_gt0([{"drafted": None}]) == 0
    assert summarize([rec], k=1) == {"cases": 1, "failed": 0, "drafted_replies_with_drafts": 1, "k": 1}
    print("selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", help="the result json (required unless --selftest)")
    ap.add_argument("--only", help="comma list of properties to run (default all): " + ", ".join(PROPERTIES))
    ap.add_argument("--chunk", type=int, help="the server's prefill chunk: long_prompt_chunking also runs 3N+1, 3N-1 and 2N tokens")
    ap.add_argument("--conc-rounds", type=int, default=3, help="concurrent_mixed batches (default: %(default)s)")
    ap.add_argument("--label", default="", help="free text saved in the json")
    ap.add_argument("--selftest", action="store_true", help="run the helpers on canned data and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.out:
        ap.error("--out is required")
    names = [n for n in (a.only.split(",") if a.only else PROPERTIES) if n]
    bad = [n for n in names if n not in PROPERTIES]
    if bad or not names:
        ap.error(f"--only: unknown {bad}; known: {', '.join(PROPERTIES)}")
    if (a.chunk is not None and a.chunk < 2) or a.conc_rounds < 1:
        ap.error("--chunk must be at least 2 and --conc-rounds at least 1")
    names.sort(key=list(PROPERTIES).index)  # the fixed order: rollback reads what the earlier properties tracked

    api = ia.api_from_args(a)
    ctx = Ctx(api, a.chunk, a.conc_rounds)
    say(f"specedge: running {', '.join(names)} (label {a.label!r})")
    res = {"meta": {"url": api.url, "model": api.model, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label": a.label, "chunk": a.chunk, "properties": names},
           "properties": {}, "failures": [], "warnings": [], "pass": False, "partial": True}
    for n in names:
        res["properties"][n] = run_property(ctx, n)
        res["warnings"] += [f"{n}: {w}" for w in res["properties"][n]["warnings"]]
        sb.save(a.out, res)
    props = res["properties"]
    res["failures"] = [f"{n}: {c['name']}: {c['difference']}" for n, p in props.items() for c in p["cases"] if not c["pass"]]
    res["failures"] += [f"{n}: {p['error']}" for n, p in props.items() if "error" in p]
    if any(not p["cases"] for p in props.values() if "error" not in p):
        res["failures"].append("a property ran no case")
    if not any(drafted_gt0(p["cases"]) for p in props.values()):
        res["failures"].append("no drafted reply drafted a token: the server is not drafting, equality proves nothing")
    res["pass"], res["partial"] = not res["failures"], False
    sb.save(a.out, res)
    for f in res["failures"]:
        say(f"FAIL {f}")
    say(f"specedge: {'PASS' if res['pass'] else 'FAIL'}: {sum(len(p['cases']) for p in props.values())} cases, "
        f"{len(res['failures'])} failures, {len(res['warnings'])} warnings -> {a.out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
