#!/usr/bin/env python3
"""Join the reports of `oracle/engine_gap.py` and `greedy_oracle.py` into the oracle-gap record (docs/ORACLE-GAP.md; stdlib only).

Per prompt three comparisons, each written by engine_gap.py:

  dump-vs-engine-NAME.json            the engine's prefill dump against the engine-emulating oracle fixture
  dump-vs-spec-NAME.json              ... against the exact-numerics (spec) fixture
  fixture-engine-vs-spec-NAME.json    the two fixtures against each other: the FLOOR, what changing the oracle's numerics by the
                                      engine's four lossy choices (and nothing else) does to the same planes

and greedy-spec.json / greedy-engine.json from `greedy_oracle.py` (the engine's own greedy tokens against the two fixtures').

    oracle_gap_summary.py --dir DIR [--names a b ...] --out summary.json [--tables tables.md]

`LIMITS` are the proposed limits (docs/ORACLE-GAP.md, "Limits proposed"): each is the worst value measured on the four prompts,
engine against either oracle, times a headroom of 1.5 to 2 (the arithmetic is in the document); the three that existed (hidden
cosine median 0.95, h_final cosine median 0.99, argmax agreement 0.95) are the Small's, kept. `checks` lists every comparison
with its limit and its value. The floor-relative checks (flips, the per-visit curve) do not depend on a limit that was fitted to the
engine: they say the engine is no farther from the oracle than the oracle's own two numerics are from each other.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

NAMES = ["chat-capital-effort0", "chat-sky-effort0.9", "raw-counting", "raw-long-window"]

LIMITS = {
    "hidden_rel_l2_plane_max": 0.40,  # per visit, whole plane: worst measured 0.2598 (x1.5)
    "hidden_rel_l2_row_median_max": 0.15,  # per visit, median over rows: 0.0972 (x1.5)
    "hidden_cos_row_median_min": 0.95,  # per visit, median over rows: 0.996; the Small's (compare.py GATES_SPEC)
    "core_out_rel_l2_plane_max": 0.18,  # 0.116 (x1.5)
    "h_final_rel_l2_plane_max": 0.17,  # 0.111 (x1.5)
    "h_final_cos_row_median_min": 0.99,  # 0.996; the Small's
    "logits_rel_l2_plane_max": 0.16,  # whole plane: 0.1025 (x1.5)
    "logits_rel_l2_row_median_max": 0.12,  # 0.0801 (x1.5)
    "logits_cos_row_min": 0.80,  # 0.871 (the Small: none; a margin of 0.07 on the worst row of the long prompt)
    "kl_mean_max": 1.5e-2,  # 7.34e-3 (x2)
    "kl_max_max": 0.2,  # 0.0923 (x2)
    "argmax_equal_min": 0.95,  # 0.959; the Small's
    "tie_margin": 0.3,  # an argmax difference needs the oracle's top-1 logit within this of the engine's pick: 0.22 (x1.35)
    "top2_equal_min": 0.85,  # 0.898 (x0.95)
    "flip_fraction_max": 0.41,  # share of (visit, row) pairs whose top-8 set differs: 0.274 (x1.5)
    "flip_vs_floor_pooled_max": 1.25,  # engine flips / floor flips, all four prompts pooled: 0.96 measured
    "flip_vs_floor_prompt_max": 1.5,  # same, per prompt: 1.35 measured (chat-capital)
    "curve_vs_floor_max": 1.5,  # per visit 1..48, engine plane rel L2 / floor's: 1.09 measured on the long prompt
    "curve_min_rows": 1000,  # the curve is checked only on a prompt this long: on 40 to 50 rows one row decides a plane (2.19 and 1.58 measured there)
    "greedy_tie_margin": 0.3,
}

KINDS = {"engine": "dump-vs-engine-", "spec": "dump-vs-spec-", "floor": "fixture-engine-vs-spec-"}


def read_json(path: str) -> dict:
    """`path`, or `path.gz` when only that exists (the committed raw reports are gzipped)."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def load(d: str, name: str) -> dict[str, dict]:
    return {k: read_json(os.path.join(d, f"{pre}{name}.json")) for k, pre in KINDS.items()}


def hid(rep: dict, g: str = "all") -> list[dict]:
    return rep["hidden"][g]


def prompt_numbers(rep: dict) -> dict:
    """The numbers of one comparison that the tables and limits use."""
    h, L, r = hid(rep), rep["logits"]["all"], rep["router"]
    return {
        "rows": rep["rows"], "prompt_len": rep["prompt_len"],
        "hidden_rel_l2_plane_max": max(x["rel_l2_plane"] for x in h),
        "hidden_worst_visit": max(range(len(h)), key=lambda i: h[i]["rel_l2_plane"]),
        "hidden_rel_l2_row_median_max": max(x["rel_l2_row_median"] for x in h),
        "hidden_cos_row_median_min": min(x["cos_row_median"] for x in h),
        "hidden_cos_row_min": min(x["cos_row_min"] for x in h),
        "core_out_rel_l2_plane_max": max(x["rel_l2_plane"] for x in rep["core_out"]["all"]),
        "h_final_rel_l2_plane_max": rep["h_final"]["all"][0]["rel_l2_plane"],
        "h_final_cos_row_median_min": rep["h_final"]["all"][0]["cos_row_median"],
        "h_final_cos_row_min": rep["h_final"]["all"][0]["cos_row_min"],
        "logits_rel_l2_plane_max": L["rel_l2_plane"], "logits_rel_l2_row_median_max": L["rel_l2_row_median"],
        "logits_rel_l2_row_max": L["rel_l2_row_max"], "logits_cos_row_min": L["cos_row_min"],
        "kl_mean_max": L["kl_mean"], "kl_max_max": L["kl_max"],
        "argmax_equal": L["argmax_equal"], "argmax_equal_min": L["argmax_equal"] / L["rows"],
        "top2_equal": L["top2_equal"], "top2_equal_min": L["top2_equal"] / L["rows"],
        "argmax_differs": rep["logits"]["argmax_differs"],
        "flips": r["flips"], "pairs": r["pairs"], "flip_fraction_max": r["flips"] / r["pairs"],
        "router_margin_min": r["oracle_margin_min"]["value"],
        "flips_max_oracle_margin": max((f["oracle_margin"] for f in r["flip_list"]), default=None),
        "weights_max_abs_diff_identical_sets": r.get("weights_max_abs_diff_identical_sets"),
    }


def flip_overlap(engine: dict, floor: dict) -> dict:
    """Of the engine's flips against one oracle, the share that also flip in the floor comparison, against the floor's own
    flip rate: a flip that recurs under an unrelated perturbation sits at a near-tie."""
    f = {(x["visit"], x["row"]) for x in floor["router"]["flip_list"]}
    e = [(x["visit"], x["row"]) for x in engine["router"]["flip_list"]]
    both = sum(1 for k in e if k in f)
    return {"engine_flips": len(e), "also_in_floor": both, "share": both / len(e) if e else None,
            "floor_rate": floor["router"]["flips"] / floor["router"]["pairs"]}


def check(rows: list, name: str, value, op: str, limit, **extra) -> bool:
    ok = value <= limit if op == "<=" else value >= limit
    rows.append({"check": name, "value": value, "op": op, "limit": limit, "ok": bool(ok), **extra})
    return ok


def greedy_numbers(sp: dict, en: dict, name: str) -> dict:
    out = {}
    for label, g in (("spec", sp), ("engine", en)):
        f = g["fixtures"][name]
        out[label] = {k: f.get(k) for k in ("prompt_len", "generated", "equal", "first_divergence", "oracle_margin", "excused_tie",
                                            "oracle_margins", "oracle_token_is_top1", "mismatch_positions", "engine_generated",
                                            "oracle_generated")}
    return out


def summarize(d: str, names: list[str]) -> dict:
    lim = LIMITS
    sp, en = read_json(os.path.join(d, "greedy-spec.json")), read_json(os.path.join(d, "greedy-engine.json"))
    per: dict = {}
    tests: list = []
    pooled = {"engine": [0, 0], "spec": [0, 0], "floor": [0, 0]}  # flips, pairs
    am_rows = {"engine": [0, 0], "spec": [0, 0], "floor": [0, 0]}  # argmax equal, rows
    greedy_equal = greedy_total = 0
    worst_long_curve = 0.0
    bins: dict = {k: None for k in ("engine", "spec", "floor")}  # the oracle's top-1 minus top-2 margin of every row, and where the argmax differs
    worst = {"hidden_rel_l2": 0.0, "logits_rel_l2_row": 0.0, "logits_rel_l2_plane": 0.0, "kl_mean": 0.0}
    for name in names:
        rep = load(d, name)
        num = {k: prompt_numbers(v) for k, v in rep.items()}
        p = per[name] = {"numbers": num, "greedy": greedy_numbers(sp, en, name), "checks": []}
        cks: list = p["checks"]
        # the visit curve: the engine's plane error per visit against the floor's (visit 0 is the embedding: equal up to 5 rows' rounding)
        curve = {}
        for k in ("engine", "spec"):
            ratios = [a["rel_l2_plane"] / b["rel_l2_plane"] for a, b in zip(hid(rep[k])[1:], hid(rep["floor"])[1:]) if b["rel_l2_plane"] > 0]
            curve[k] = {"max": max(ratios), "worst_visit": 1 + ratios.index(max(ratios)), "min": min(ratios)}
        p["curve_vs_floor"] = curve
        p["flip_overlap"] = {k: flip_overlap(rep[k], rep["floor"]) for k in ("engine", "spec")}
        for k in ("engine", "spec", "floor"):
            mb = rep[k]["logits"]["margin_bins"]
            bins[k] = [dict(b) for b in mb] if bins[k] is None else [{**a, "rows": a["rows"] + b["rows"], "argmax_differs": a["argmax_differs"] + b["argmax_differs"]} for a, b in zip(bins[k], mb)]
            pooled[k][0] += num[k]["flips"]
            pooled[k][1] += num[k]["pairs"]
            am_rows[k][0] += num[k]["argmax_equal"]
            am_rows[k][1] += num[k]["rows"]
        for k in ("engine", "spec"):
            n = num[k]
            worst["hidden_rel_l2"] = max(worst["hidden_rel_l2"], n["hidden_rel_l2_plane_max"])
            worst["logits_rel_l2_row"] = max(worst["logits_rel_l2_row"], n["logits_rel_l2_row_max"])
            worst["logits_rel_l2_plane"] = max(worst["logits_rel_l2_plane"], n["logits_rel_l2_plane_max"])
            worst["kl_mean"] = max(worst["kl_mean"], n["kl_mean_max"])
            if n["rows"] >= lim["curve_min_rows"]:
                worst_long_curve = max(worst_long_curve, curve[k]["max"])
        # per-test groups: the planes, the logits, the router, the greedy tokens (each against both oracles)
        groups = {"hidden": ["hidden_rel_l2_plane_max", "hidden_rel_l2_row_median_max", "hidden_cos_row_median_min",
                             "core_out_rel_l2_plane_max", "h_final_rel_l2_plane_max", "h_final_cos_row_median_min"],
                  "logits": ["logits_rel_l2_plane_max", "logits_rel_l2_row_median_max", "logits_cos_row_min", "kl_mean_max", "kl_max_max",
                             "argmax_equal_min", "top2_equal_min"],
                  "router": ["flip_fraction_max"]}
        for g, keys in groups.items():
            before = len(cks)
            for k in ("engine", "spec"):
                for key in keys:
                    op = ">=" if key.endswith("_min") else "<="
                    check(cks, f"{g}:{key}", num[k][key], op, lim[key], vs=k)
                if g == "logits":
                    for dd in num[k]["argmax_differs"]:
                        check(cks, "logits:argmax_difference_gap", dd["oracle_gap_to_engine_pick"], "<=", lim["tie_margin"], vs=k,
                              row=dd["row"], oracle_margin=dd["oracle_margin"])
                if g == "router":
                    check(cks, "router:flips_vs_floor_prompt", num[k]["flips"] / max(num["floor"]["flips"], 1), "<=",
                          lim["flip_vs_floor_prompt_max"], vs=k)
                    if num[k]["rows"] >= lim["curve_min_rows"]:
                        check(cks, "router:visit_curve_vs_floor", curve[k]["max"], "<=", lim["curve_vs_floor_max"], vs=k,
                              worst_visit=curve[k]["worst_visit"])
            sub = cks[before:]
            tests.append({"bin": "oracle/engine_gap.py", "filter": f"{name}:{g}", "passed": sum(c["ok"] for c in sub),
                          "failed": sum(not c["ok"] for c in sub)})
        # greedy: the engine's own tokens against the oracle's, per fixture set
        before = len(cks)
        for k in ("spec", "engine"):
            gk = p["greedy"][k]
            n_tok = gk["generated"]
            diff = gk["mismatch_positions"]
            greedy_total += n_tok if k == "spec" else 0
            greedy_equal += (n_tok - len(diff)) if k == "spec" else 0
            if not diff:
                check(cks, "greedy:tokens_equal", n_tok, ">=", n_tok, vs=k)
            else:
                check(cks, "greedy:first_divergence_margin", gk["oracle_margin"], "<=", lim["greedy_tie_margin"], vs=k,
                      position=gk["first_divergence"])
        sub = cks[before:]
        tests.append({"bin": "bench/tools/greedy_oracle.py", "filter": f"{name}:greedy", "passed": sum(c["ok"] for c in sub),
                      "failed": sum(not c["ok"] for c in sub)})
    # pooled
    rate = {k: v[0] / v[1] for k, v in pooled.items()}
    pooled_checks: list = []
    for k in ("engine", "spec"):
        check(pooled_checks, "router:flips_vs_floor_pooled", pooled[k][0] / max(pooled["floor"][0], 1), "<=", lim["flip_vs_floor_pooled_max"], vs=k)
    tests.append({"bin": "oracle/engine_gap.py", "filter": "pooled:router_flips_vs_floor", "passed": sum(c["ok"] for c in pooled_checks),
                  "failed": sum(not c["ok"] for c in pooled_checks)})
    ok_all = all(c["ok"] for p in per.values() for c in p["checks"]) and all(c["ok"] for c in pooled_checks)
    long_name = max(names, key=lambda n: per[n]["numbers"]["engine"]["prompt_len"])
    metrics = {
        "prompts_compared": len(names),
        "long_prompt_tokens": per[long_name]["numbers"]["engine"]["prompt_len"],
        "rows_compared": am_rows["engine"][1],
        "hidden_rel_l2": round(worst["hidden_rel_l2"], 6),
        "logits_rel_l2": round(worst["logits_rel_l2_row"], 6),
        "logits_rel_l2_plane": round(worst["logits_rel_l2_plane"], 6),
        "kl_mean_worst": worst["kl_mean"],
        "argmax_equal_fraction": min(am_rows["engine"][0] / am_rows["engine"][1], am_rows["spec"][0] / am_rows["spec"][1]),
        "oracle_numerics_argmax_equal_fraction": am_rows["floor"][0] / am_rows["floor"][1],
        "router_flips": pooled["spec"][0],
        "router_flips_vs_engine_fixture": pooled["engine"][0],
        "router_flips_floor": pooled["floor"][0],
        "router_pairs": pooled["spec"][1],
        "router_flip_fraction": rate["spec"],
        "router_flip_fraction_floor": rate["floor"],
        "visit_curve_vs_floor_max": round(worst_long_curve, 4),
        "greedy_tokens_equal": greedy_equal,
        "greedy_tokens_total": greedy_total,
    }
    return {"names": names, "limits": LIMITS, "per_prompt": per,
            "pooled": {"flips": pooled, "argmax_equal": am_rows, "margin_bins": bins, "checks": pooled_checks},
            "metrics": metrics, "tests": tests, "pass": ok_all}


# ------------------------------------------------------------------------------------------------------- tables


def _f(x, p=3):
    return f"{x:.{p}g}" if isinstance(x, float) else str(x)


def tables(s: dict) -> str:
    out = []
    names = s["names"]
    out.append("| prompt | rows | comparison | hidden rel L2 (worst visit) | h_final rel L2 | logits rel L2 plane / row median / row max | logits cos min | KL mean / max | argmax equal | top-2 equal | flips of pairs |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    labels = {"engine": "engine vs engine fixture", "spec": "engine vs spec fixture", "floor": "floor: engine fixture vs spec fixture"}
    for n in names:
        for k in ("engine", "spec", "floor"):
            x = s["per_prompt"][n]["numbers"][k]
            out.append(f"| {n if k == 'engine' else ''} | {x['rows'] if k == 'engine' else ''} | {labels[k]} | {_f(x['hidden_rel_l2_plane_max'])} ({x['hidden_worst_visit']}) | "
                       f"{_f(x['h_final_rel_l2_plane_max'])} | {_f(x['logits_rel_l2_plane_max'])} / {_f(x['logits_rel_l2_row_median_max'])} / {_f(x['logits_rel_l2_row_max'])} | "
                       f"{x['logits_cos_row_min']:.5f} | {_f(x['kl_mean_max'])} / {_f(x['kl_max_max'])} | {x['argmax_equal']}/{x['rows']} | "
                       f"{x['top2_equal']}/{x['rows']} | {x['flips']} / {x['pairs']} ({x['flip_fraction_max']:.1%}) |")
    out.append("")
    out.append("| prompt | engine / floor, per visit 1..48 (max at visit; min) | flips that recur in the floor (engine vs spec): share vs the floor's rate | smallest router margin | weights, max abs diff (identical sets) |")
    out.append("|---|---|---|---|---|")
    for n in names:
        p = s["per_prompt"][n]
        c, fo = p["curve_vs_floor"]["spec"], p["flip_overlap"]["spec"]
        x = p["numbers"]["spec"]
        out.append(f"| {n} | {c['max']:.2f} (visit {c['worst_visit']}); {c['min']:.2f} | {fo['also_in_floor']}/{fo['engine_flips']} = {'n/a' if fo['share'] is None else format(fo['share'], '.1%')} vs {fo['floor_rate']:.1%} | "
                   f"{_f(x['router_margin_min'])} | {_f(x['weights_max_abs_diff_identical_sets'])} |")
    out.append("")
    out.append("| prompt | generated | engine tokens vs spec fixture's | vs engine fixture's | oracle top-1 - top-2 margin at the generated positions: min (position); positions under 0.3 |")
    out.append("|---|---|---|---|---|")
    for n in names:
        g = s["per_prompt"][n]["greedy"]
        ms = g["spec"]["oracle_margins"]
        i = min(range(len(ms)), key=lambda j: ms[j])
        eq = lambda x: "equal" if x["equal"] else f"first difference at {x['first_divergence']}"  # noqa: E731
        out.append(f"| {n} | {g['spec']['generated']} | {eq(g['spec'])} | {eq(g['engine'])} | {ms[i]:.3g} ({i}); {sum(1 for m in ms if m < 0.3)} of {len(ms)} |")
    out.append("")
    out.append("| oracle margin of the row | " + " | ".join(f"[{b['lo']:g}, {b['hi']:g})" for b in s["pooled"]["margin_bins"]["engine"]) + " |")
    out.append("|---|" + "---|" * len(s["pooled"]["margin_bins"]["engine"]))
    out.append("| rows (spec fixture's margins, all prompts) | " + " | ".join(str(b["rows"]) for b in s["pooled"]["margin_bins"]["spec"]) + " |")
    for k, lab in (("engine", "engine vs engine fixture"), ("spec", "engine vs spec fixture"), ("floor", "floor: engine fixture vs spec fixture")):
        out.append(f"| argmax differs, {lab} | " + " | ".join(str(b["argmax_differs"]) for b in s["pooled"]["margin_bins"][k]) + " |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--names", nargs="*", default=NAMES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables")
    a = ap.parse_args(argv)
    s = summarize(a.dir, a.names)
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1)
        f.write("\n")
    if a.tables:
        with open(a.tables, "w") as f:
            f.write(tables(s))
    bad = [(n, c) for n, p in s["per_prompt"].items() for c in p["checks"] if not c["ok"]] + [("pooled", c) for c in s["pooled"]["checks"] if not c["ok"]]
    for n, c in bad:
        print(f"FAIL {n} {c['check']} vs {c.get('vs')}: {c['value']} {c['op']} {c['limit']}", file=sys.stderr)
    print(json.dumps(s["metrics"], indent=1))
    print("pass" if s["pass"] else "fail", f"({sum(t['passed'] for t in s['tests'])} checks ok, {sum(t['failed'] for t in s['tests'])} failed)")
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
