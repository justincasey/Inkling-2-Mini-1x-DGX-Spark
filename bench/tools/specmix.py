#!/usr/bin/env python3
"""Does drafting come back when traffic changes? Replay a phased mix (prose, then code) against ONE running server (stdlib only).

    specmix.py --out mix.json [--url URL] [--mix chat,code] [--per-phase 12] [--max-tokens 256] [--reference-tpr 6.0] [--label TEXT]
    specmix.py --selftest          # the summary helpers on canned numbers, no server

The engine's learnt class prior (a server started with the learning budget) is read from the traffic it has served, per sampling mode.
Once prose has pushed the first draft position's acceptance under break-even the class used to be given zero drafts for good: nothing
it could learn from was verified again, so code arriving later was never drafted (same bytes, no speed-up). The class counters now fade
in rounds that draft nothing (`SpecStats::class`), and this tool is the live check: START THE SERVER FRESH (the class counters of a
server that has already served code would not collapse), then run it. Every phase of --mix runs --per-phase greedy requests one at a
time (`"draft": true`, `"prefix_cache": false`, `return_token_ids`), each reported with its tokens per round (usage.completion_tokens /
usage.spec.rounds), drafted and accepted counts and tok/s. Kinds are specbench.py's (chat, code, json, longctx, tool, reasoning).

The verdict compares the LAST THIRD of the final phase with --reference-tpr, the tokens per round of that kind on a fresh server
(specbench.py's drafted tokens/round for it, or this tool run with --mix <kind>): it passes at 0.9 of the reference or better and
fails when the final phase is still not being drafted (tokens per round under 1.2). Without --reference-tpr the verdict is only the
second check. At the end one serial request per kind must equal its first drafted reply (the budget never changes bytes); the serial
requests run last because a round without drafts also fades the class counters.

    # on a freshly started server: the prose phase, then the code phase (default mix chat,code)
    bench/tools/specmix.py --out spec-mix.json --mix chat,code --per-phase 12 --max-tokens 256
    # the reference for the verdict, from a second fresh server: bench/tools/specmix.py --out ref.json --mix code --per-phase 12
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402
import specbench as sb  # noqa: E402  (prompts, request, stats helpers)

FAILS: list[str] = []
RECOVERED_AT = 0.9  # of the reference tokens per round
STILL_COLLAPSED = 1.2  # tokens per round at or under which a phase is plainly not drafted


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tpr(reply: dict) -> float | None:
    """Tokens per round of one reply: completion tokens over usage.spec.rounds (None without it)."""
    s = reply["spec"]
    return sb.rate(reply["completion_tokens"], s["rounds"], 3) if s and s["rounds"] else None


def phase_tpr(rows: list[dict]) -> float | None:
    """Tokens per round of a group of requests: all their tokens over all their rounds."""
    tokens = sum(r["completion_tokens"] for r in rows if r["rounds"])
    rounds = sum(r["rounds"] for r in rows)
    return sb.rate(tokens, rounds, 3)


def verdict(phases: list[dict], reference: float | None) -> list[str]:
    """Failure lines about the final phase's last third (empty: it is drafting, and as well as the reference when given)."""
    last = phases[-1]["requests"]
    third = last[len(last) - max(1, len(last) // 3):]
    got = phase_tpr(third)
    out = []
    if got is None:
        out.append(f"{phases[-1]['kind']}: the last requests carry no usage.spec rounds")
    elif got <= STILL_COLLAPSED:
        out.append(f"{phases[-1]['kind']}: the final phase is not drafted ({got} tokens per round in its last third)")
    elif reference is not None and got < RECOVERED_AT * reference:
        out.append(f"{phases[-1]['kind']}: {got} tokens per round in the last third, under {RECOVERED_AT} of the reference {reference}")
    return out


def run_phase(api: ia.Api, spec: dict, n: int, max_tokens: int) -> dict:
    rows, first = [], None
    for i in range(n):
        r = sb.request(api, spec, max_tokens, True)
        first = first or r
        s = r["spec"] or {}
        rows.append({"completion_tokens": r["completion_tokens"], "rounds": s.get("rounds", 0), "drafted": s.get("drafted", 0),
                     "accepted": s.get("accepted", 0), "tok_s": round(r["tok_s"], 2), "tokens_per_round": tpr(r)})
        say(f"  {spec['name']} {i + 1}/{n}: {rows[-1]['tokens_per_round']} tokens/round, drafted {rows[-1]['drafted']}, "
            f"accepted {rows[-1]['accepted']}, {rows[-1]['tok_s']} tok/s")
    return {"kind": spec["name"], "requests": rows, "tokens_per_round": phase_tpr(rows),
            "first_third_tpr": phase_tpr(rows[:max(1, n // 3)]), "last_third_tpr": phase_tpr(rows[n - max(1, n // 3):]),
            "median_tok_s": round(statistics.median(r["tok_s"] for r in rows), 2)}, first


def render_md(res: dict) -> str:
    m = res["meta"]
    lines = [f"# specmix {m['date']}" + (f" - {m['label']}" if m.get("label") else ""), "",
             f"model {m['model']}, mix {','.join(m['mix'])}, {m['per_phase']} greedy requests of {m['max_tokens']} tokens per phase", "",
             "| phase | kind | tokens/round | first third | last third | median tok/s | drafted | accepted |", "|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(res["phases"], 1):
        d, a = sum(r["drafted"] for r in p["requests"]), sum(r["accepted"] for r in p["requests"])
        lines.append(f"| {i} | {p['kind']} | {p['tokens_per_round']} | {p['first_third_tpr']} | {p['last_third_tpr']} | {p['median_tok_s']} | {d} | {a} |")
    lines.append("")
    if m.get("reference_tpr") is not None:
        lines += [f"reference tokens/round of the final kind on a fresh server: {m['reference_tpr']}", ""]
    lines += [f"FAIL {f}" for f in res["failures"]]
    lines.append("PASS drafting follows the traffic" if not res["failures"] else "FAIL see above")
    return "\n".join(lines)


def save(path: str, res: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def selftest() -> int:
    row = lambda toks, rounds: {"completion_tokens": toks, "rounds": rounds, "drafted": 0, "accepted": 0, "tok_s": 1.0}  # noqa: E731
    assert phase_tpr([row(100, 100), row(100, 50)]) == round(200 / 150, 3)
    assert phase_tpr([row(10, 0)]) is None
    stuck = [{"kind": "code", "requests": [row(256, 256)] * 6}]
    assert verdict(stuck, None) and "not drafted" in verdict(stuck, None)[0]
    good = [{"kind": "code", "requests": [row(256, 256)] * 3 + [row(256, 40)] * 3}]
    assert verdict(good, None) == [] and verdict(good, 6.0) == [], verdict(good, 6.0)
    assert "under 0.9 of the reference" in verdict(good, 9.0)[0]
    assert verdict([{"kind": "code", "requests": [row(5, 0)]}], None)
    assert tpr({"spec": {"rounds": 4}, "completion_tokens": 10}) == 2.5 and tpr({"spec": None, "completion_tokens": 1}) is None
    print("selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", help="the result json (required unless --selftest)")
    ap.add_argument("--mix", default="chat,code", help="the phases in order, kinds of specbench.py (default: %(default)s)")
    ap.add_argument("--per-phase", type=int, default=12, help="greedy requests per phase (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reference-tpr", type=float, help="tokens per round of the final kind on a fresh server (the verdict's reference)")
    ap.add_argument("--no-serial", action="store_true", help="skip the closing serial equality requests")
    ap.add_argument("--label", default="", help="free text saved in the json")
    ap.add_argument("--selftest", action="store_true", help="run the helpers on canned numbers and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.out:
        ap.error("--out is required")
    mix = [k for k in a.mix.split(",") if k]
    bad = [k for k in mix if k not in sb.KINDS]
    if bad or not mix:
        ap.error(f"--mix: unknown {bad}; known: {', '.join(sb.KINDS)}")
    if a.per_phase < 3 or a.max_tokens < 1:
        ap.error("--per-phase must be at least 3 (a verdict reads its last third) and --max-tokens positive")

    api = ia.api_from_args(a)
    prompts = sb.build_prompts(api, list(dict.fromkeys(mix)), 6000)
    res = {"meta": {"url": api.url, "model": api.model, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label": a.label,
                    "mix": mix, "per_phase": a.per_phase, "max_tokens": a.max_tokens, "reference_tpr": a.reference_tpr},
           "phases": [], "failures": FAILS, "partial": True}
    firsts: dict[str, dict] = {}
    for kind in mix:
        say(f"phase {len(res['phases']) + 1}: {kind} x {a.per_phase}")
        phase, first = run_phase(api, prompts[kind], a.per_phase, a.max_tokens)
        res["phases"].append(phase)
        firsts.setdefault(kind, first)
        save(a.out, res)
    FAILS.extend(verdict(res["phases"], a.reference_tpr))
    if not a.no_serial:
        for kind, first in firsts.items():
            serial = sb.request(api, prompts[kind], a.max_tokens, False)
            d = sb.diff_note(sb.key(first), sb.key(serial))
            if d:
                FAILS.append(f"{kind}: the first drafted reply is not the serial one: {d}")
    res["partial"] = False
    save(a.out, res)
    print(render_md(res))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
