#!/usr/bin/env python3
"""GSM8K accuracy on the live server (stdlib only): the quality gate for any change that is not bit-exact.

  gsm8k.py [--url URL] [--n 100] [--offset 0] [--think] [--workers 2] [--data PATH] [--out FILE]

The first --n problems from --offset of the GSM8K test split (openai/gsm8k, MIT licence), greedy, one commit per item. The data
file is NOT in this repository: fetch it as bench/rexl3/DATA.md says (default path bench/rexl3/data/gsm8k_test.jsonl; a JSON
list of {"question", "answer"} objects works too). The node comes from NODES/PORT or INKLING2MINI_URL / --url.

Default: reasoning effort `none` (the template level 0; the model may still reason a little), a step-by-step answer ending
`#### <number>`, 1,024 tokens. --think: effort `high`, 16,384 tokens. An item is COMMITTED when finish_reason is `stop` and a
`#### <number>` line is in the reply; a reply cut by the token limit (finish_reason `length`, the base model's long repetitive
reasoning) is counted uncommitted, not wrong, and is listed. Per item the record keeps reasoning_tokens
(usage.completion_tokens_details) and completion_tokens (usage); a reply whose content or reasoning holds an Inkling framing
marker (inkling_api.FRAMING) is flagged `framing_leak` and counts as uncommitted.

Prints METRIC gsm8k_acc (correct / committed), gsm8k_committed, gsm8k_length_cut and gsm8k_reasoning_tokens_mean. The --out file is
the input of gsm8k_compare.py. Exit 1 when no item was committed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import inkling_api as ia  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "gsm8k_test.jsonl")
PROMPT = "{q}\n\nSolve it step by step. End your reply with a final line `#### <number>` holding only the answer."


def number(s: str) -> float | None:
    """The last number of `s` (thousands separators and $ ignored), or None."""
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def committed_answer(text: str) -> float | None:
    """The number after the last `####` of a reply, None when there is no such line."""
    return number(text.rsplit("####", 1)[1]) if "####" in text else None


def load_items(path: str) -> list[dict]:
    """GSM8K problems from a .jsonl (one object per line) or a JSON list; each needs `question` and `answer`."""
    if not os.path.exists(path):
        raise SystemExit(f"{path}: not found. The GSM8K test split is not shipped with this recipe; fetch it as described in "
                         f"bench/rexl3/DATA.md (or pass --data PATH)")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    items = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    for k, it in enumerate(items):
        if not isinstance(it, dict) or "question" not in it or "####" not in str(it.get("answer", "")):
            raise SystemExit(f"{path}: item {k} lacks `question` / an `answer` with `####`")
    return items


def framing_leak(*texts: str) -> bool:
    return any(m in t for t in texts for m in ia.FRAMING)


def grade(item: dict, content: str, reasoning: str, finish: str | None, usage: dict, index: int) -> dict:
    """The record of one answered item."""
    gold = number(item["answer"].split("####")[-1])
    got = committed_answer(content)
    leak = framing_leak(content, reasoning)
    committed = got is not None and finish == "stop" and not leak
    return {"i": index, "gold": gold, "got": got, "finish": finish, "tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "framing_leak": leak, "committed": committed, "ok": committed and abs(got - gold) < 1e-6}


def summarize(rows: list[dict]) -> dict:
    done = [r for r in rows if r["committed"]]
    rt = [r["reasoning_tokens"] for r in rows if r.get("reasoning_tokens") is not None]
    return {"asked": len(rows), "committed": len(done), "correct": sum(1 for r in done if r["ok"]),
            "acc": sum(1 for r in done if r["ok"]) / max(len(done), 1), "committed_frac": len(done) / max(len(rows), 1),
            "length_cut": sum(1 for r in rows if r["finish"] == "length"),
            "reasoning_tokens_mean": sum(rt) / len(rt) if rt else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="reasoning effort high instead of none")
    ap.add_argument("--max-tokens", type=int, help="default 1024, with --think 16384")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--data", default=DEFAULT_DATA, help="GSM8K test split, .jsonl or a JSON list (default: %(default)s; see DATA.md)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    data = load_items(a.data)[a.offset:a.offset + a.n]
    if not data:
        raise SystemExit("no problems selected (--offset/--n beyond the data)")
    api = ia.api_from_args(a)
    max_tokens = a.max_tokens or (16384 if a.think else 1024)
    t0 = time.time()

    def one(i: int) -> dict:
        resp, _ = api.chat([{"role": "user", "content": PROMPT.format(q=data[i]["question"])}], max_tokens,
                           temperature=a.temperature, reasoning_effort="high" if a.think else "none")
        ch = resp["choices"][0]
        msg = ch["message"]
        return grade(data[i], msg.get("content") or "", msg.get("reasoning_content") or "", ch.get("finish_reason"),
                     resp.get("usage") or {}, a.offset + i)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, range(len(data))))
    s = summarize(rows)
    print(f"{s['correct']}/{s['committed']} correct of {s['asked']} asked ({time.time() - t0:.0f} s, "
          f"{sum(r['tokens'] for r in rows)} tokens, {s['length_cut']} cut by the token limit)", file=sys.stderr)
    for r in rows:
        if not r["ok"]:
            print(f"  wrong/uncommitted #{r['i']}: gold {r['gold']} got {r['got']} finish {r['finish']}"
                  f"{' FRAMING LEAK' if r['framing_leak'] else ''}", file=sys.stderr)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
    print(f"METRIC gsm8k_acc={s['acc']:.4f}")
    print(f"METRIC gsm8k_committed={s['committed_frac']:.4f}")
    print(f"METRIC gsm8k_length_cut={s['length_cut']}")
    if s["reasoning_tokens_mean"] is not None:
        print(f"METRIC gsm8k_reasoning_tokens_mean={s['reasoning_tokens_mean']:.1f}")
    return 0 if s["committed"] else 1


if __name__ == "__main__":
    sys.exit(main())
