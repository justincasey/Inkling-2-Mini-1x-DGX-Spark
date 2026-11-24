"""Ranks the rows of a `teacher_force.py --per-row` report: for every op and pass the value of each row against the median of the OTHER
rows (the ratio), the worst (pass, row) per op, and the per-row summary of chosen rows over all passes.

    python row_table.py REPORT.json [--rows 0,1,2,3,5,23] [--top 8] [--variant kv8+attn_bf16_p+dense_fp16]
"""

import argparse
import json
import statistics

# the ratio's denominator never goes below this scale of the op (an op that is exact on most rows has a median of 0, and a one-ulp flip
# of bf16 on a row is not a ratio of infinity): bf16 half-ulp is 2e-3 relative, an fp32 router logit error is 1e-7 of the sum of |terms|
DEN_FLOOR = {"router logit err": 1e-8, "residual elems": 1.0}
DEN_DEFAULT = 1e-4


def series(rep: dict, variant: str) -> dict[str, list[float]]:
    """The per-row vectors of one pass, by op name (only what the report holds)."""
    out = {}
    a = rep["attn"][variant]
    out["attn_core rel"] = a["core"]["by_row"]["rel"]
    out["attn_out rel"] = a["a"]["by_row"]["rel"]
    out["mix rel"] = rep["mix"]["by_row"]["rel"]
    for k in ("msg", "attn_in"):
        if k in rep:
            out[f"{k} rel"] = rep[k]["by_row"]["rel"]
    out["router logit err"] = rep["router"]["by_row"]["logit_err_rel_sumabs"]
    if "moe" in rep:
        out["moe rel"] = rep["moe"]["moe_fp16"]["by_row"]["rel"]
    if "residual_mismatch_by_row" in rep:
        out["residual elems"] = [float(x) for x in rep["residual_mismatch_by_row"]]
    return out


def ratios(v: list[float], den: float = DEN_DEFAULT) -> list[float]:
    return [x / max(statistics.median(v[:j] + v[j + 1:]), den) for j, x in enumerate(v)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report")
    ap.add_argument("--rows", default="0,1,2,3,5,23")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--variant", default="kv8+attn_bf16_p+dense_fp16")
    ap.add_argument("--floor", type=float, default=0.0, help="values below this are not ranked (exact or noise-level)")
    a = ap.parse_args(argv)
    rep = json.load(open(a.report))
    rows = [int(x) for x in a.rows.split(",")]
    per_op: dict[str, list[tuple]] = {}
    for p in rep["passes"]:
        for op, v in series(p, a.variant).items():
            for j, (x, r) in enumerate(zip(v, ratios(v, DEN_FLOOR.get(op, DEN_DEFAULT)))):
                per_op.setdefault(op, []).append((r, p["pass"], j, x, statistics.median(v[:j] + v[j + 1:])))
    print(f"rows {rep['rows']}, passes {len(rep['passes'])}, variant {a.variant}")
    for op, lst in per_op.items():
        ranked = sorted((t for t in lst if t[3] >= a.floor), reverse=True)[:a.top]
        print(f"\n{op}: worst (pass, row) by ratio to the median of the other rows")
        print("  ratio   pass  row   value      median-of-others")
        for r, p, j, x, m in ranked:
            print(f"  {r:6.2f}  {p:4d}  {j:3d}  {x:.3e}  {m:.3e}")
    print("\nhow often each row is the worst row of a pass (per op; rows with at least 3 of the passes)")
    for op, lst in per_op.items():
        worst = {}
        for p in sorted({t[1] for t in lst}):
            t = max((t for t in lst if t[1] == p), key=lambda t: t[0])
            if t[3] > 0:
                worst[t[2]] = worst.get(t[2], 0) + 1
        top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
        print(f"  {op:20s} " + ", ".join(f"row {j}: {c}" for j, c in top))
    print("\nchosen rows: per op, the max ratio over passes (at pass), and the median ratio over passes")
    print("op".ljust(20) + "".join(f"row {r:<3d}".rjust(22) for r in rows))
    for op, lst in per_op.items():
        cells = []
        for r in rows:
            mine = [t for t in lst if t[2] == r and t[3] >= a.floor]
            allr = [t[0] for t in lst if t[2] == r]
            if not mine:
                cells.append("-".rjust(22))
                continue
            top = max(mine)
            cells.append(f"{top[0]:5.2f}@{top[1]:<2d} med {statistics.median(allr):4.2f}".rjust(22))
        print(op.ljust(20) + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
