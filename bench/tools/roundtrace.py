#!/usr/bin/env python3
"""The host timeline of a drafted decode round (stdlib only). Needs a server booted with REXL3_ROUND_TRACE=1 (EXTRA_ENV).

    roundtrace.py --url URL --ssh spark@node --log /path/serve.log --out trace.json [--kinds prose,code,structured,json]
                  [--passes 3] [--max-tokens 512]

Each pass sends bench.py's DECODE_TEXTS prompt of a kind (greedy, 512 tokens, cold, drafter on) alone, reads the lines the engine
printed meanwhile (`event=round_trace`, one per drafted round: host microseconds from the round's start to each mark, and `gap_us`, the
host time since the previous round's end) over ssh, and averages them over the rounds of the request (the first round of a request, which has
no commit before it, is left out). One warm-up pass is discarded; the table is the median of the other passes, per kind.

Intervals (ms a round): plan (policy, before the drafter), feed (drafter commit: taps to its context), commit (conv/ring commit launches),
propose (stage + launch + wait + read of the proposal), alloc (budget plan after the draft and the request build), prepare (the window's
flush, uploads), launch (the graph launch), wait (the host parked until the window's candidates are ready), read (the download),
accept (sampling, acceptance, pushes), tail (the round's accounting) and gap (outside decode_round: route, snapshot, the server loop).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import inkling_api as ia  # noqa: E402

ORDER = ["plan0", "c_fed", "c_committed", "proposed", "planned", "w_prepared", "w_launched", "w_synced", "verified", "accepted", "end"]
NAMES = ["plan", "feed", "commit", "propose", "alloc", "prepare", "launch", "wait", "read", "accept", "tail"]
# a chained round (the proposal and the window enqueued on one stream, one wait): its own marks
CHAIN_ORDER = ["plan0", "c_fed", "c_committed", "k_proposed", "k_prepared", "k_launched", "k_synced", "w_synced", "verified", "accepted", "end"]
CHAIN_NAMES = ["plan", "feed", "commit", "enqueue_proposal", "prepare", "launch", "wait", "sync_tail", "read", "accept", "tail"]
KINDS = ("prose", "code", "structured", "json")


def lines_of(ssh: str, log: str, n0: int) -> tuple[list[str], int]:
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", ssh, f"tail -n +{n0 + 1} {log}; echo @@; wc -l < {log}"], check=True,
                         capture_output=True, text=True).stdout
    body, _, n = out.rpartition("@@\n")
    return body.splitlines(), int(n.strip())


def parse(lines: list[str]) -> list[dict]:
    rows = []
    for ln in lines:
        if "event=round_trace" not in ln:
            continue
        kv = dict(w.split("=", 1) for w in ln.split() if "=" in w)
        rows.append({k: int(v) for k, v in kv.items() if k not in ("rexl3", "event") and v.isdigit()})
    return rows


def table(rows: list[dict]) -> dict:
    chain = sum(1 for r in rows if "k_proposed" in r) > len(rows) // 2
    order, names = (CHAIN_ORDER, CHAIN_NAMES) if chain else (ORDER, NAMES)
    full = [r for r in rows if all(k in r for k in order)]
    out: dict = {"rounds": len(rows), "rounds_full": len(full), "chained": chain}
    if not full:
        return out
    ms = {}
    for i, (name, key) in enumerate(zip(names, order)):
        ms[name] = statistics.fmean((r[key] - (r[order[i - 1]] if i else 0)) for r in full) / 1000
    ms["gap"] = statistics.fmean(r["gap_us"] for r in full) / 1000
    ms["in_round"] = statistics.fmean(r["end"] for r in full) / 1000
    ms["period"] = ms["in_round"] + ms["gap"]
    out["ms"] = {k: round(v, 3) for k, v in ms.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--ssh", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    api = ia.Api(a.url)
    _, n = lines_of(a.ssh, a.log, 0)
    result: dict = {}
    for kind in a.kinds.split(","):
        ids = api.tokenize(bench.DECODE_TEXTS[kind])
        runs = []
        for p in range(a.passes + 1):
            resp, wall = api.complete(ids, a.max_tokens, prefix_cache=False)
            lines, n = lines_of(a.ssh, a.log, n)
            if p == 0:
                continue
            r = table(parse(lines))
            u = resp.get("usage") or {}
            r["completion_tokens"] = int(u["completion_tokens"])
            r["wall_s"] = round(wall, 4)
            r["tok_s_wall"] = round(r["completion_tokens"] / wall, 2)
            runs.append(r)
            print(kind, p, json.dumps(r), file=sys.stderr, flush=True)
        med = {}
        for k in runs[0].get("ms", {}):
            med[k] = round(statistics.median(r["ms"][k] for r in runs), 3)
        result[kind] = {"median_ms": med, "runs": runs,
                        "tokens_per_round": round(statistics.median(r["completion_tokens"] / r["rounds"] for r in runs), 3)}
    json.dump(result, open(a.out, "w"), indent=1)
    names = list(next(iter(result.values()))["median_ms"].keys())
    print("| kind | tok/round | " + " | ".join(names) + " |")
    print("|---|---:|" + "---:|" * len(names))
    for kind, v in result.items():
        print(f"| {kind} | {v['tokens_per_round']} | " + " | ".join(f"{v['median_ms'].get(k, 0):.2f}" for k in names) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
