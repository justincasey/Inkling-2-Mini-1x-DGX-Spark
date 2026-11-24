#!/usr/bin/env python3
"""The real cost of a verify window by (streams, window rows) and of the proposal by streams, from a traced server (stdlib only).

    roundcost.py --url URL --ssh spark@node --log serve.log --out cost.json [--streams 1,2,3,4] [--reps 2] [--max-tokens 512]

The server must run with REXL3_ROUND_TRACE=1 and REXL3_INKLING2_LOOP_CHAIN=0 (a chained round cannot tell the window's time from the proposal's).
For each stream count it sends that many requests at once (bench.py's four DECODE_TEXTS kinds in turn), reads the trace lines of the run and
reports, per (streams, window rows), the median host milliseconds from the window's launch to its candidates being read (`w_synced - w_launched`,
the window with its tail) and, per stream count, the proposal's (`proposed - c_committed`: stage, launch, wait, read). Rows with fewer than 5
rounds are left out. `window_grid` is the table the family's `round_cost` takes: (streams, rows a stream, ms) is NOT what this prints:
rows here are the window's total rows.
"""
import argparse, json, os, statistics, sys, threading
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench, inkling_api as ia, roundtrace as rt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--ssh", required=True); ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--streams", default="1,2,3,4"); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--serial", action="store_true", help='requests carry "draft": false (every window is one row a stream)')
    a = ap.parse_args()
    api = ia.Api(a.url)
    kinds = list(bench.DECODE_TEXTS)
    ids = {k: api.tokenize(bench.DECODE_TEXTS[k]) for k in kinds}
    _, n = rt.lines_of(a.ssh, a.log, 0)
    win, prop = defaultdict(list), defaultdict(list)
    for s in [int(x) for x in a.streams.split(",")]:
        for rep in range(a.reps):
            order = [kinds[(rep + i) % len(kinds)] for i in range(s)]
            ths = [threading.Thread(target=lambda k=k: api.complete(ids[k], a.max_tokens, prefix_cache=False, **({'draft': False} if a.serial else {}))) for k in order]
            [t.start() for t in ths]; [t.join() for t in ths]
            lines, n = rt.lines_of(a.ssh, a.log, n)
            for r in rt.parse(lines):
                if "w_synced" in r and "w_launched" in r and "c_committed" in r and "proposed" in r:
                    win[(r["streams"], r["rows"])].append((r["w_synced"] - r["w_launched"]) / 1000)
                    prop[r["streams"]].append((r["proposed"] - r["c_committed"]) / 1000)
                elif "w_synced" in r and "w_launched" in r and "c_committed" not in r:
                    win[(r["streams"], r["rows"])].append((r["w_synced"] - r["w_launched"]) / 1000)
                elif "w_synced" in r and "w_launched" in r:
                    win[(r["streams"], r["rows"])].append((r["w_synced"] - r["w_launched"]) / 1000)
    out = {"window_ms": {f"{k[0]}x{k[1]}": [round(statistics.median(v), 3), len(v)] for k, v in sorted(win.items()) if len(v) >= (5 if not a.serial else 3)},
           "proposal_ms": {str(k): [round(statistics.median(v), 3), len(v)] for k, v in sorted(prop.items()) if len(v) >= (5 if not a.serial else 3)}}
    json.dump(out, open(a.out, "w"), indent=1)
    print("window ms by streams x total rows (median, rounds):")
    for k, v in out["window_ms"].items(): print(" ", k, v)
    print("proposal ms by streams:", out["proposal_ms"])

if __name__ == "__main__":
    main()
