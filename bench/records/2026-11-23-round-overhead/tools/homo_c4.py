#!/usr/bin/env python3
"""Four identical streams at once (stdlib only): the aggregate tok/s of 4 x one kind, drafted and serial, and bench.py's mixed c = 4.

    homo_c4.py --url URL --out homo.json [--reps 3] [--max-tokens 512]

For each of bench.py's four DECODE_TEXTS kinds (prose, code, structured, json) and for the mixed set of the four, four requests at once
(`"draft": true` and `"draft": false`, cold, greedy), a discarded pass then --reps timed passes, the median of aggregate tok/s = completion tokens
over the wall of the batch. A drafted reply must equal its serial reply (a difference is a FAIL).
"""
import argparse, json, os, statistics, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench, inkling_api as ia

def batch(api, ids_list, mt, draft):
    outs = [None] * len(ids_list)
    def one(i):
        r, _ = api.complete(ids_list[i], mt, prefix_cache=False, draft=draft)
        outs[i] = (r["choices"][0].get("token_ids") or r["choices"][0].get("generated_ids"), r["usage"]["completion_tokens"])
    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(ids_list))]
    t0 = time.monotonic(); [t.start() for t in ths]; [t.join() for t in ths]
    return outs, time.monotonic() - t0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args(); api = ia.Api(a.url)
    ids = {k: api.tokenize(t) for k, t in bench.DECODE_TEXTS.items()}
    sets = {f"4x{k}": [ids[k]] * 4 for k in ids}
    sets["mixed(bench)"] = [ids[k] for k in ids]
    res, fails = {}, []
    for name, lst in sets.items():
        row = {}
        for draft in (False, True):
            rates, last = [], None
            for rep in range(a.reps + 1):
                outs, span = batch(api, lst, a.max_tokens, draft)
                if rep:
                    rates.append(sum(n for _, n in outs) / span)
                last = outs
            row["drafted" if draft else "serial"] = round(statistics.median(rates), 2)
            row["ids_drafted" if draft else "ids_serial"] = [o[0] for o in last]
        if row["ids_drafted"] != row["ids_serial"]:
            fails.append(name)
        del row["ids_drafted"], row["ids_serial"]
        row["speedup"] = round(row["drafted"] / row["serial"], 3)
        res[name] = row
        print(name, row, flush=True)
    json.dump({"results": res, "fails": fails}, open(a.out, "w"), indent=1)
    print("PASS drafted == serial" if not fails else f"FAIL {fails}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
