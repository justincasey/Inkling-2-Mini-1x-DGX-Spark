#!/usr/bin/env python3
"""Fixed exact-answer quality gate against the node's chat API (stdlib only, no dataset, greedy: temperature 0).

60 questions kept in this file and generated with a fixed seed: 20 multiplications, 10 remainders, 10 capitals, 8 string
tasks (reverse a word, count a letter) and 12 fixed facts and puzzles, each with one exact answer. The answer is read from
`message.content` only (never from the reasoning); a reply with no visible answer counts wrong. Integer answers are graded
on the last number of the reply, word answers on a whole-word match (case and accents ignored).

Per question the row records the reply, its grade, finish_reason (`length` is flagged: the base model can reason in long
repetitive loops), completion_tokens and reasoning_tokens (usage.completion_tokens_details). A framing marker
(inkling_api.FRAMING) in content or reasoning is a failure by itself, and so is a request error.

  quality_gate.py --out bench/runs/quality.json                  # the first run is the baseline: keep that file
  quality_gate.py --baseline bench/records/quality.json          # later runs: no more than --tolerance (default 2)
                                                                 # fewer correct than the baseline, same question set
  quality_gate.py --effort high --max-tokens 8192 --min 55       # reasoning on; exit 1 below --min correct

Prints `PASS quality: N/60 ...` or `FAIL quality: why`, the misses on stderr, `METRIC quality_*` lines; exit 1 on FAIL.
Effort: reasoning_effort "none" (default) or "high"; --system adds a system message such as `Thinking effort level: 0`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import inkling_api as ia  # noqa: E402

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Brazil", "Brasilia"),
            ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Egypt", "Cairo"), ("Peru", "Lima"), ("Vietnam", "Hanoi")]
REVERSE = ["lantern", "harbor", "compass", "quartz"]
COUNT = [("e", "nevertheless"), ("s", "mississippi"), ("a", "banana"), ("r", "strawberry")]
INT = "Reply with the integer only."
FACTS = [
    ("A farmer has 17 sheep. All but 9 run away. How many sheep are left? " + INT, "9", "int"),
    ("What is the chemical symbol for gold? Reply with the symbol only.", "Au", "word"),
    ("How many sides does a hexagon have? " + INT, "6", "int"),
    ("What is the square root of 1764? " + INT, "42", "int"),
    ("Write the decimal number 37 in binary. Reply with the digits only.", "100101", "int"),
    ("What is 2 to the power of 20? " + INT, "1048576", "int"),
    ("What is the sum of the integers from 1 to 100? " + INT, "5050", "int"),
    ("How many prime numbers are there below 30? " + INT, "10", "int"),
    ("If today is Friday, what day of the week is it 10 days from now? Reply with the day name only.", "Monday", "word"),
    ("How many minutes are there in 3.5 hours? " + INT, "210", "int"),
    ("Which is the largest planet in the solar system? Reply with the planet name only.", "Jupiter", "word"),
    ("What is the 10th number of the Fibonacci sequence 1, 1, 2, 3, 5, 8, ...? " + INT, "55", "int"),
]


def questions() -> list[dict]:
    """The fixed question set: [{id, kind, prompt, want}]; the same list on every run."""
    rng = random.Random(1017)
    items: list[tuple[str, str, str]] = []
    for _ in range(20):
        a, b = rng.randint(12, 999), rng.randint(12, 99)
        items.append(("int", f"What is {a}*{b}? {INT}", str(a * b)))
    for _ in range(10):
        a, b = rng.randint(100, 9999), rng.randint(7, 97)
        items.append(("int", f"What is the remainder when {a} is divided by {b}? {INT}", str(a % b)))
    for country, capital in CAPITALS:
        items.append(("word", f"What is the capital of {country}? Reply with the city name only.", capital))
    for w in REVERSE:
        items.append(("word", f"Write the word '{w}' backwards. Reply with the reversed word only.", w[::-1]))
    for ch, w in COUNT:
        items.append(("int", f"How many times does the letter '{ch}' appear in the word '{w}'? {INT}", str(w.count(ch))))
    items += [(kind, prompt, want) for prompt, want, kind in FACTS]
    return [{"id": f"q{i:02d}", "kind": kind, "prompt": prompt, "want": want} for i, (kind, prompt, want) in enumerate(items)]


def fingerprint(items: list[dict]) -> str:
    """sha256 of the question set, so a baseline of another set is refused."""
    return hashlib.sha256(json.dumps([[q["kind"], q["prompt"], q["want"]] for q in items]).encode()).hexdigest()


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


def grade(kind: str, want: str, content: str) -> bool:
    """Exact grading of the visible reply: the last number for `int`, a whole-word match for `word`."""
    if not content or not content.strip():
        return False
    if kind == "int":
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", content)
        return bool(numbers) and numbers[-1].replace(",", "").rstrip(".") == want
    return re.search(r"(?<![a-z0-9])" + re.escape(fold(want)) + r"(?![a-z0-9])", fold(content)) is not None


def framing_in(text: str) -> list[str]:
    return [m for m in ia.FRAMING if m in text]


def compare(baseline: dict, rows: list[dict], tolerance: int) -> tuple[bool, str, list[str]]:
    """(ok, why, newly wrong ids) of this run against a recorded baseline JSON."""
    if baseline.get("fingerprint") != fingerprint(questions()):
        return False, "the baseline holds another question set (fingerprint differs)", []
    base = {r["id"]: r["ok"] for r in baseline.get("rows", [])}
    now = sum(1 for r in rows if r["ok"])
    newly = [r["id"] for r in rows if not r["ok"] and base.get(r["id"])]
    if now < baseline["correct"] - tolerance:
        return False, f"{now} correct against the baseline's {baseline['correct']} (tolerance {tolerance})", newly
    return True, f"{now} correct against the baseline's {baseline['correct']} (tolerance {tolerance})", newly


def ask(api: ia.Api, q: dict, a) -> dict:
    row = {"id": q["id"], "kind": q["kind"], "prompt": q["prompt"], "want": q["want"], "content": "", "ok": False}
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": q["prompt"]}]
    try:
        out, dt = api.post("/v1/chat/completions", {"model": api.model, "messages": msgs, "temperature": 0,
                                                     "max_tokens": a.max_tokens, "reasoning_effort": a.effort})
    except (ia.ApiError, OSError, ValueError) as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        return row
    ch = (out.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content, reasoning, u = (msg.get("content") or "").strip(), msg.get("reasoning_content") or "", out.get("usage") or {}
    row.update(content=content, ok=grade(q["kind"], q["want"], content), finish_reason=ch.get("finish_reason"),
               completion_tokens=u.get("completion_tokens"),
               reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               seconds=round(dt, 3), framing=framing_in(content) + framing_in(reasoning))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--effort", choices=("none", "high"), default="none", help="reasoning_effort (default: %(default)s)")
    ap.add_argument("--system", default="", help="a system message, e.g. 'Thinking effort level: 0' (default: none)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="per answer (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests (default: %(default)s)")
    ap.add_argument("--min", type=int, default=0, help="fail below this many correct (of the set)")
    ap.add_argument("--baseline", help="a recorded --out file to compare against")
    ap.add_argument("--tolerance", type=int, default=2, help="questions below the baseline still passing (default: %(default)s)")
    ap.add_argument("--out", help="write the JSON record here (the first run's file is the baseline)")
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    items = questions()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        rows = list(pool.map(lambda q: ask(api, q, a), items))
    correct = sum(1 for r in rows if r["ok"])
    errors = [r for r in rows if r.get("error")]
    leaks = [r for r in rows if r.get("framing")]
    cut = [r for r in rows if r.get("finish_reason") == "length"]
    rts = [r["reasoning_tokens"] for r in rows if isinstance(r.get("reasoning_tokens"), int)]
    for r in rows:
        if not r["ok"]:
            print(f"  miss {r['id']} ({r['kind']}): want {r['want']!r}, got {r['content'][:120]!r}"
                  + (f" [{r['error']}]" if r.get("error") else "") + (" [finish length]" if r.get("finish_reason") == "length" else ""),
                  file=sys.stderr)
    problems = []
    if errors:
        problems.append(f"{len(errors)} request errors")
    if leaks:
        problems.append(f"framing markers in {[r['id'] for r in leaks]}")
    if correct < a.min:
        problems.append(f"{correct} correct is below --min {a.min}")
    base_note = "no baseline given (this run can be the baseline)"
    if a.baseline:
        with open(a.baseline) as f:
            ok, why, newly = compare(json.load(f), rows, a.tolerance)
        base_note = why + (f"; newly wrong {newly}" if newly else "")
        if not ok:
            problems.append(why)
    rec = {"url": api.url, "model": api.model, "effort": a.effort, "system": a.system, "max_tokens": a.max_tokens,
           "fingerprint": fingerprint(items), "correct": correct, "total": len(rows), "errors": len(errors),
           "framing_leaks": len(leaks), "length_cut": len(cut), "rows": rows}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
    print(f"METRIC quality_correct={correct}")
    print(f"METRIC quality_total={len(rows)}")
    print(f"METRIC quality_accuracy={correct / len(rows):.4f}")
    print(f"METRIC quality_length_cut={len(cut)}")
    if rts:
        print(f"METRIC quality_reasoning_tokens_mean={sum(rts) / len(rts):.1f}")
    print(f"{'FAIL' if problems else 'PASS'} quality: {correct}/{len(rows)} correct; "
          + ("; ".join(problems) if problems else base_note) + (f"; {len(cut)} replies cut by max_tokens" if cut else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
