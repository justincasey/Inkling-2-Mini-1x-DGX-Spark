#!/usr/bin/env python3
"""Engine greedy vs the oracle's fixtures that live on ANOTHER machine (stdlib only; ssh with BatchMode).

greedy.py needs the fixtures in a local directory. The fixtures of the real pack (40+ MB each) stay where the oracle made them:
this tool streams stlite.py plus a small reader to `python3 -` on that machine, which prints one JSON summary per fixture (prompt ids, the
oracle's own greedy ids, and per generated position the top-1/top-2 token and logit margin from the fixture's `logits`); nothing is
copied. The engine then continues each prompt greedily (token-id prompt, temperature 0, exactly as many tokens as the oracle made) and
the two token lists are compared: equal, or the first divergence with the oracle's top-1 - top-2 margin there (below 0.05, oracle/compare.py's
tie margin, a near-tie is excused). Exit 0 when every fixture is equal or excused.

    greedy_oracle.py --ssh spark@HOST --dir /path/fixtures/spec [--only NAME ...] --out greedy-oracle.json

The report carries the address only as given on the command line in `source`; scrub it before it goes into a committed record.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402

TIE_MARGIN = 0.05  # oracle/compare.py GATES_ENGINE["tie_margin"]

REMOTE = r'''
import sys
FX = sys.argv[1]
names = sys.argv[2:] or sorted(json.load(open(FX + "/manifest.json"))["fixtures"])
outs = []
for name in names:
    path = f"{FX}/{name}.safetensors"
    t, meta = read_header(path)
    plen = int(meta["prompt_len"])
    toks = read_ints(path, "tokens")
    o = {"name": name, "prompt_len": plen, "prompt_ids": toks[:plen], "oracle_generated": toks[plen:], "rows": []}
    if "logits" in t:
        for i in range(len(toks) - plen):
            row = plen - 1 + i
            if row >= t["logits"]["shape"][0]:
                break
            r = read_f32_row(path, "logits", row)
            i1 = max(range(len(r)), key=r.__getitem__)
            v1 = r[i1]
            r[i1] = float("-inf")
            i2 = max(range(len(r)), key=r.__getitem__)
            o["rows"].append({"pos": i, "token": toks[plen + i], "top1": i1, "top2": i2, "margin": v1 - r[i2]})
    outs.append(o)
print(json.dumps(outs))
'''


def fetch(target: str, fx_dir: str, only: list[str]) -> list[dict]:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "stlite.py")) as f:
        src = f.read()
    prog = src + "\n" + REMOTE
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, "python3", "-", fx_dir, *only],
                       input=prog, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ssh {target}: exit {r.returncode}: {r.stderr.strip()[:400]}")
    return json.loads(r.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--ssh", required=True, help="USER@HOST holding the fixtures")
    ap.add_argument("--dir", required=True, help="the fixtures directory there (manifest.json + NAME.safetensors)")
    ap.add_argument("--only", nargs="*", default=[], help="fixture names (default: all in the manifest)")
    ap.add_argument("--out", help="write the JSON report here (also printed)")
    ap.add_argument("--prefill-full", action="store_true",
                    help="before each greedy run, also send the fixture's whole token list but its last id (prompt + the oracle's tokens) as "
                         "one prompt with max_tokens 1: a server started with REXL3_INKLING2_LOOP_DUMP writes the planes of exactly the rows "
                         "the fixture holds, for oracle/engine_gap.py")
    a = ap.parse_args(argv)

    fixtures = fetch(a.ssh, a.dir, a.only)
    api = ia.api_from_args(a)
    rep = {"url": api.url, "model": api.model, "source": f"{a.ssh}:{a.dir}", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "fixtures": {}}
    ok = True
    for fx in fixtures:
        oracle = fx["oracle_generated"]
        if a.prefill_full:
            api.complete(fx["prompt_ids"] + oracle[:-1], 1)
        engine, requests, secs = ia.greedy(api, fx["prompt_ids"], len(oracle))
        diff = [i for i, (x, y) in enumerate(zip(oracle, engine)) if x != y]
        margins = {r["pos"]: r for r in fx["rows"]}
        r = {"prompt_len": fx["prompt_len"], "generated": len(oracle), "equal": not diff, "first_divergence": None,
             "mismatch_positions": diff, "requests": requests, "seconds": round(secs, 3),
             "prefill_full_rows": len(fx["prompt_ids"]) + len(oracle) - 1 if a.prefill_full else None,
             "prompt_ids": fx["prompt_ids"], "engine_generated": engine, "oracle_generated": oracle,
             "oracle_margins": [round(margins[i]["margin"], 4) if i in margins else None for i in range(len(oracle))],
             "oracle_token_is_top1": [margins[i]["token"] == margins[i]["top1"] if i in margins else None for i in range(len(oracle))]}
        if diff:
            i = diff[0]
            m = margins.get(i)
            r.update(first_divergence=i, oracle_token=oracle[i], engine_token=engine[i],
                     oracle_margin=None if m is None else m["margin"], oracle_top2=None if m is None else [m["top1"], m["top2"]],
                     excused_tie=m is not None and m["margin"] < TIE_MARGIN)
        r["pass"] = r["equal"] or bool(r.get("excused_tie"))
        ok &= r["pass"]
        rep["fixtures"][fx["name"]] = r
        print(f"{fx['name']}: {'equal' if r['equal'] else 'DIVERGES at %d' % r['first_divergence']}"
              f"{' (excused tie, margin %.4g)' % r['oracle_margin'] if r.get('excused_tie') else ''}"
              f" ({len(oracle)} tokens, prompt {fx['prompt_len']}, {requests} requests, {secs:.1f}s)", file=sys.stderr)
    rep["pass"] = ok
    txt = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
