#!/usr/bin/env python3
"""The Mini's three missed tool-gate cases as two-turn conversations (plan 6.7; stdlib only, imports bench/toolgate.py).

toolgate.py's `order_nested`, `order_two_skus` and `order_standard` name products ('pen', 'lamp', 'bolt') while create_order
needs a `sku` for each item; the Mini first calls run_sql to look the SKUs up. This tool replays each case: turn 1 as the
gate sends it, turn 2 with a run_sql tool result that holds the SKUs (the lookup the model asked for, answered from a fixed table), and
checks that the next call is create_order with the customer, the quantities, the looked-up SKUs and the expedited flag the
prompt asks for. It also records what turn 1 did (the tool, readonly, and whether the query names the products).

  toolgate_lookup.py [--url URL] [--temperature T] [--runs N] --out lookup.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import inkling_api as ia  # noqa: E402
import toolgate as tg  # noqa: E402

SKUS = {"pen": "PEN-100", "notebook": "NB-200", "lamp": "LMP-300", "cable": "CBL-400", "bolt": "BLT-500"}
# case id -> (prompt, the create_order call the prompt asks for with the SKUs above)
CASES = {
    "order_nested": ("Create an expedited order for customer 7: one 'pen' and three 'notebook'. Use the tool.",
                     {"customer_id": 7, "expedited": True, "items": {"PEN-100": 1, "NB-200": 3}}),
    "order_two_skus": ("Create an order for customer 9 with one 'lamp' and two 'cable'. Use the tool.",
                       {"customer_id": 9, "items": {"LMP-300": 1, "CBL-400": 2}}),
    "order_standard": ("Create a standard order for customer 3 with five 'bolt'. Use the tool.",
                       {"customer_id": 3, "expedited": False, "items": {"BLT-500": 5}}),
}


def rows_for(query: str) -> list[dict]:
    q = query.lower()
    return [{"sku": v, "name": k} for k, v in SKUS.items() if k in q]


def run_case(api, a, cid: str, prompt: str, want: dict) -> dict:
    tools = tg.by_name(tg.TOOLS)
    msgs = [{"role": "user", "content": prompt}]
    first = tg.ask(api, a, msgs, tg.TOOLS)
    out = {"id": cid, "turn1": {k: first.get(k) for k in ("calls", "content", "finish", "error")}}
    calls = first.get("calls") or []
    sql = [c for c in calls if c["name"] == "run_sql"]
    out["turn1_run_sql_readonly"] = bool(sql) and all(c["arguments"].get("readonly") is True for c in sql)
    out["turn1_query_names_products"] = bool(sql) and all(rows_for(c["arguments"].get("query", "")) for c in sql)
    if not sql:
        out["ok"], out["detail"] = False, f"turn 1 made no run_sql call: {[c['name'] for c in calls]}"
        return out
    call = sql[0]
    result = {"rows": rows_for(call["arguments"].get("query", ""))}
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                                                   "function": {"name": "run_sql", "arguments": call["raw"]}}]},
             {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(result)}]
    second = tg.ask(api, a, msgs, tg.TOOLS)
    out["turn2"] = {k: second.get(k) for k in ("calls", "content", "finish", "error")}
    orders = [c for c in second.get("calls") or [] if c["name"] == "create_order"]
    if not orders:
        out["ok"], out["detail"] = False, f"turn 2 made no create_order call: {[c['name'] for c in second.get('calls') or []]}"
        return out
    args = orders[0]["arguments"]
    items = {i.get("sku"): i.get("qty") for i in args.get("items", [])}
    bad = tg.check_call(orders[0], tools)
    for k in ("customer_id", "expedited"):
        if k in want and args.get(k) != want[k]:
            bad.append(f"{k}={args.get(k)!r}, want {want[k]!r}")
    if items != want["items"]:
        bad.append(f"items {items}, want {want['items']}")
    out["ok"], out["detail"] = not bad, "; ".join(bad)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=1, help="repeat each case (sampling: several runs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    cfg = SimpleNamespace(temperature=a.temperature, max_tokens=2048, effort="none")
    rows = []
    for cid, (prompt, want) in CASES.items():
        for run in range(a.runs):
            r = run_case(api, cfg, cid, prompt, want)
            r["run"] = run
            rows.append(r)
            print(f"{'PASS' if r['ok'] else 'FAIL'} {cid} run {run}: lookup={r['turn1_run_sql_readonly']} names_products={r['turn1_query_names_products']}"
                  + ("" if r["ok"] else f": {r['detail']}"))
    n_ok = sum(r["ok"] for r in rows)
    print(f"METRIC lookup_followup_pass={n_ok}\nMETRIC lookup_followup_total={len(rows)}")
    with open(a.out, "w") as f:
        json.dump({"temperature": a.temperature, "skus": SKUS, "rows": rows}, f, indent=1)
        f.write("\n")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
