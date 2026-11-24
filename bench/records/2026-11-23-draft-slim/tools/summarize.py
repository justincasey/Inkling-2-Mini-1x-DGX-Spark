#!/usr/bin/env python3
"""summarize.py RUNS_DIR [OUT_JSON]: the DraftSlim record's tables from the sessions of session.sh (base, slim, slimoff).

Per session: bench decode (three runs after the discarded warm-up: the medians of decode_c1_tok_s = decode_mean4_tok_s, the four kinds, c=4
and c=8 spans, and each kind's tokens a round from its own `decode <kind>:` line), bench rolling (three runs: decode_c4_rolling_tok_s and
decode_c4_steady_tok_s), specbench (per kind: drafted tok/s, tokens a round, equal; the c=4 aggregate). Prints markdown, writes the numbers."""
import json
import re
import statistics as S
import sys
from pathlib import Path

KINDS = ("prose", "code", "structured", "json")


def decode_runs(d: Path):
    runs = []
    for i in (1, 2, 3):
        p = d / f"bench-decode-{i}.log"
        if not p.exists():
            continue
        r = {"kinds_tpr": {}}
        for line in p.read_text().splitlines():
            m = re.match(r"METRIC (\w+)=([\d.]+)", line)
            if m:
                r[m.group(1)] = float(m.group(2))
            m = re.match(r"decode (\w+): .*tokens_per_round': ([\d.]+)", line)
            if m:
                r["kinds_tpr"][m.group(1)] = float(m.group(2))
            m = re.match(r"conc c=4: .*= ([\d.]+) tok/s", line)
            if m:
                r["c4_span"] = float(m.group(1))
        runs.append(r)
    return runs


def rolling_runs(d: Path):
    out = []
    for i in (1, 2, 3):
        p = d / f"bench-rolling-{i}.log"
        if not p.exists():
            continue
        r = {}
        for line in p.read_text().splitlines():
            m = re.match(r"METRIC (decode_c4_\w+)=([\d.]+)", line)
            if m:
                r[m.group(1)] = float(m.group(2))
        out.append(r)
    return out


def med(xs):
    return round(S.median(xs), 3) if xs else None


def session(d: Path):
    dec, rol = decode_runs(d), rolling_runs(d)
    s = {
        "decode_mean4_tok_s": med([r["decode_c1_tok_s"] for r in dec]),
        "decode_mean4_runs": [r["decode_c1_tok_s"] for r in dec],
        "decode_tokens_per_round": med([r["decode_tokens_per_round"] for r in dec]),
        "decode_c4_tok_s": med([r["decode_c4_tok_s"] for r in dec]),
        "decode_c8_tok_s": med([r["decode_c8_tok_s"] for r in dec]),
        "decode_c4_rolling_tok_s": med([r["decode_c4_rolling_tok_s"] for r in rol]),
        "decode_c4_rolling_runs": [r["decode_c4_rolling_tok_s"] for r in rol],
        "decode_c4_steady_tok_s": med([r["decode_c4_steady_tok_s"] for r in rol]),
    }
    for k in KINDS:
        s[f"decode_{k}_tok_s"] = med([r[f"decode_{k}_tok_s"] for r in dec])
        s[f"decode_{k}_tokens_per_round"] = med([r["kinds_tpr"][k] for r in dec if k in r["kinds_tpr"]])
    p = d / "specbench.json"
    if p.exists():
        sb = json.loads(p.read_text())
        for k, v in sb["kinds"].items():
            s[f"specbench_{k}_tok_s"] = v["drafted"]["median"]
            s[f"specbench_{k}_serial_tok_s"] = v["serial"]["median"]
            s[f"specbench_{k}_tokens_per_round"] = (v["drafted"].get("spec") or {}).get("tokens_per_round")
        c = sb.get("concurrency") or {}
        s["specbench_c4_aggregate_tok_s"] = (c.get("drafted") or {}).get("median")
        s["specbench_c4_serial_tok_s"] = (c.get("serial") or {}).get("median")
    return s


def main():
    runs = Path(sys.argv[1])
    res = {t: session(runs / t) for t in ("base", "slim", "slimoff") if (runs / t).exists()}
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + "\n")
    names = list(res)
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    keys = ["decode_mean4_tok_s", "decode_prose_tok_s", "decode_code_tok_s", "decode_structured_tok_s", "decode_json_tok_s", "decode_c4_tok_s",
            "decode_c8_tok_s", "decode_c4_rolling_tok_s", "decode_c4_steady_tok_s", "decode_tokens_per_round"] + \
           [f"decode_{k}_tokens_per_round" for k in KINDS]
    for k in keys:
        print(f"| {k} | " + " | ".join(str(res[t].get(k)) for t in names) + " |")
    for k in sorted({x for t in names for x in res[t] if x.startswith("specbench_") and not isinstance(res[t][x], (dict, list))}):
        print(f"| {k} | " + " | ".join(str(res[t].get(k)) for t in names) + " |")


if __name__ == "__main__":
    main()
