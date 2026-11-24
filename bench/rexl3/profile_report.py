#!/usr/bin/env python3
"""Turn sweep JSONs into a table against the targets of docs/ROOFLINE.md (stdlib only).

  profile_report.py decode.json prefill.json [--md OUT.md] [--strict]

Any number of files, in any order: a file with `shallow`/`depth` keys is a sweep_decode.py result, one with `cells` a
sweep_prefill.py result. Each row is one cell (median over its runs) with the ceiling, the target and the stretch of
docs/ROOFLINE.md ("Targets" and the context table), the measured value, its percentage of the target and a verdict:
`stretch` (>= stretch), `target` (>= target), `below` (under it), `-` (the roofline names no target for the cell).

The numbers below are copied from docs/ROOFLINE.md, section "Targets (proposals, single Spark, GPU locked at 2,200 MHz)" and
the ceilings of its decode and prefill tables; the file says they are proposals until the owner confirms them. Serial rows
compare with the serial targets, drafted rows (a `default` mode whose requests reported usage.spec) with the MTP targets;
a `default` mode without spec is the serial path. Concurrent-stream targets are the roofline's no-drafting aggregate at 4K.
Prints the table, `METRIC profile_<kind>_<what>_pct=` lines (percent of target) and exits 0, or 1 with --strict when any
cell is below its target.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

# (ceiling, target, stretch), tok/s. docs/ROOFLINE.md "Targets" table; ceilings from its context / concurrency / prefill tables.
DECODE_C1_SERIAL = {4096: (70.1, 62, 65), 32768: (61.5, 54, 57), 131072: (43.3, 38, 40), 262144: (31.0, 27, 29),
                    1048576: (11.5, 10, 11)}
DECODE_C1_MTP = {4096: (108, 77, 92), 32768: (None, 72, 85), 131072: (None, 60, 70), 262144: (None, 49, 56),
                 1048576: (None, 23, 25)}
DECODE_AGGREGATE = {4: (113, 89, 105), 8: (137, 108, 127)}      # no drafting, 4K, aggregate tok/s
PREFILL_COLD = {8192: (9525, 2380, 3810), 32768: (8481, 2120, 3390), 131072: (5894, 1470, 2360), 262144: (4191, 1050, 1680),
                1048576: (1533, 380, 610)}


def verdict(value, target, stretch) -> str:
    if value is None or target is None:
        return "-"
    return "stretch" if stretch is not None and value >= stretch else "target" if value >= target else "below"


def decode_targets(kind: str, c: int, ctx: int):
    """(ceiling, target, stretch) of a decode cell, or (None, None, None). kind: serial | mtp."""
    if c == 1:
        return (DECODE_C1_MTP if kind == "mtp" else DECODE_C1_SERIAL).get(ctx, (None, None, None))
    if kind == "serial" and ctx == 4096:
        return DECODE_AGGREGATE.get(c, (None, None, None))
    return (None, None, None)


def decode_rows(res: dict) -> list[dict]:
    cells: dict[tuple, list[dict]] = {}
    for phase in ("shallow", "depth"):
        for r in res.get(phase, []):
            if r.get("ok") and r.get("tok_s") is not None:
                cells.setdefault((phase, r["mode"], r["c"], r["ctx"]), []).append(r)
    rows = []
    for (phase, mode, c, ctx), rs in sorted(cells.items(), key=lambda kv: (kv[0][3], kv[0][1], kv[0][2], kv[0][0])):
        drafted = mode == "default" and any(r.get("spec") for r in rs)
        kind = "mtp" if drafted else "serial"
        ceil, tgt, stretch = decode_targets(kind, c, ctx)
        v = statistics.median(r["tok_s"] for r in rs)
        tpr = [r["tokens_per_round"] for r in rs if r.get("tokens_per_round") is not None]
        rows.append({"what": f"decode {phase} {mode}{' (spec)' if drafted else ''}", "kind": kind, "c": c, "ctx": ctx, "value": v,
                     "ceiling": ceil, "target": tgt, "stretch": stretch, "runs": len(rs),
                     "extra": f"tokens/round {statistics.median(tpr):.2f}" if tpr else ""})
    return rows


def prefill_rows(res: dict) -> list[dict]:
    cells: dict[tuple, list[dict]] = {}
    hits: dict[int, list[dict]] = {}
    for r in res.get("cells", []):
        if r.get("ok") and r.get("tok_s") is not None:
            cells.setdefault((r["c"], r["size"]), []).append(r)
        if r.get("hit") and r["hit"].get("hit_ok"):
            hits.setdefault(r["size"], []).append(r["hit"])
    rows = []
    for (c, size), rs in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ceil, tgt, stretch = PREFILL_COLD.get(size, (None, None, None)) if c == 1 else (None, None, None)
        tt = [r["ttft_s"] for r in rs if r.get("ttft_s") is not None]
        extra = f"ttft {statistics.median(tt):.2f} s" if tt else ""
        if c == 1 and size in hits:
            extra += f"{', ' if extra else ''}cached ttft {statistics.median(h['ttft_s'] for h in hits[size]):.3f} s"
        rows.append({"what": "prefill cold", "kind": "prefill", "c": c, "ctx": size, "value": statistics.median(r["tok_s"] for r in rs),
                     "ceiling": ceil, "target": tgt, "stretch": stretch, "runs": len(rs), "extra": extra})
    return rows


def rows_of(res: dict) -> list[dict]:
    if "cells" in res:
        return prefill_rows(res)
    if "shallow" in res or "depth" in res:
        return decode_rows(res)
    raise ValueError("neither a decode nor a prefill sweep result")


def fmt(v, nd=1) -> str:
    return "-" if v is None else f"{v:,.{nd}f}" if isinstance(v, float) and v != int(v) else f"{v:,.0f}"


def table(rows: list[dict]) -> str:
    out = ["| cell | c | context | measured tok/s | ceiling | target | stretch | % of target | verdict | runs | note |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in rows:
        pct = None if r["target"] is None else 100 * r["value"] / r["target"]
        out.append(f"| {r['what']} | {r['c']} | {r['ctx']:,} | {fmt(r['value'])} | {fmt(r['ceiling'])} | {fmt(r['target'])} | "
                   f"{fmt(r['stretch'])} | {'-' if pct is None else '%.0f%%' % pct} | {verdict(r['value'], r['target'], r['stretch'])} | "
                   f"{r['runs']} | {r['extra']} |")
    return "\n".join(out)


def metric_lines(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        if r["target"] is not None:
            tag = f"{r['kind']}_c{r['c']}_{r['ctx']}"
            out.append(f"METRIC profile_{tag}_pct={100 * r['value'] / r['target']:.1f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", help="also write the table here")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a cell with a target is below it")
    a = ap.parse_args(argv)
    rows = []
    for p in a.files:
        with open(p) as f:
            rows += rows_of(json.load(f))
    txt = table(rows)
    print(txt)
    if a.md:
        with open(a.md, "w") as f:
            f.write(txt + "\n")
    for line in metric_lines(rows):
        print(line)
    below = [r for r in rows if verdict(r["value"], r["target"], r["stretch"]) == "below"]
    print(f"METRIC profile_cells={len(rows)}\nMETRIC profile_below_target={len(below)}")
    return 1 if a.strict and below else 0


if __name__ == "__main__":
    sys.exit(main())
