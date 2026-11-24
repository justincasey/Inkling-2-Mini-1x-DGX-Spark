#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""records_index.py: tables taken from bench/records/*/meta.json; nothing is typed by hand.

    python3 scripts/records_index.py                              # the newest record of every gate, on stdout
    python3 scripts/records_index.py --update FILE.md             # ... between <!-- records-index:start --> and <!-- records-index:end -->
    python3 scripts/records_index.py --headline docs/headline.tsv                    # the headline table, on stdout
    python3 scripts/records_index.py --headline docs/headline.tsv --update README.md # ... between <!-- headline:start --> and <!-- headline:end -->
    python3 scripts/records_index.py --cap-audit                  # the GPU clock evidence of every record (exit 1: a sample above the cap)

A gate's record is the one scripts/check_record.py would judge (newest by date, then by the commit that ADDED its meta.json). The index
shows up to four metrics a record, the throughput ones (`*_tok_s`) first, a value longer than 24 characters (a hash) left out; a record
with no gate (the node probe) is left out. The headline spec is a TSV, one row a line: `label<TAB>gate<TAB>metric<TAB>unit` (`#`
starts a comment): the value is that metric of the gate's newest record, with the record's name and engine commit beside it. `gate` may
be `@RECORD-DIRECTORY` to name one record: for a figure that a newer record of the same gate does not replace (another engine's serial
figure, another profile). A gate with no record, a missing record directory, a metric the record lacks, or a record whose result is not
`pass` stops the run: a published number must come from a passing record.

`--cap-audit` looks at every record whose `node` is node-1..node-4: `locked` when a file in the record directory holds a line
`GPU clocks set to` (the success line of the clock lock, as in raw/cap-proof.txt) or `GPU SM clock capped at` (printed
by scripts/gpu_cap.sh only after the node answered with that success line, exit 0 only then: proof by construction); else `sampled` when meta.json has a string `gpu_clock_sm_max` of the
form "<current> MHz, <hardware max> MHz" (only the FIRST number is a sample; the second is clocks.max.sm); else `none`. It prints
one summary line and the `none` records, and exits 1 naming the record when a sampled current clock exceeds GPU_CLOCK_CAP_MHZ of
config/pins.sh. Free-text fields and clock settings are not parsed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_record import commit_time  # noqa: E402  (one ordering rule for the checker and the index)

MARKERS = {"index": ("<!-- records-index:start -->", "<!-- records-index:end -->"),
           "headline": ("<!-- headline:start -->", "<!-- headline:end -->")}
MAX_METRICS = 4
MAX_VALUE = 24
LOCK_LINES = ("GPU clocks set to", "GPU SM clock capped at")


def newest_per_gate(root: Path) -> dict:
    best = {}
    for meta in sorted((root / "bench" / "records").glob("*/meta.json")):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        gate = d.get("gate") if isinstance(d, dict) else None
        if not isinstance(gate, str) or not gate:
            continue
        key = (str(d.get("date", "")), commit_time(root, meta), meta.parent.name)
        if gate not in best or key > best[gate][0]:
            best[gate] = (key, meta.parent.name, d)
    return best


def metrics_cell(d: dict) -> str:
    m = d.get("metrics") if isinstance(d.get("metrics"), dict) else {}
    m = {k: v for k, v in m.items() if len(str(v)) <= MAX_VALUE}
    shown = [k for k in sorted(m) if k.endswith("_tok_s")] + [k for k in sorted(m) if not k.endswith("_tok_s")]
    return "<br>".join(f"{k} = {m[k]}" for k in shown[:MAX_METRICS]) or "-"


def tests_cell(d: dict) -> str:
    t = d.get("tests") if isinstance(d.get("tests"), list) else []
    if not t:
        return "-"
    passed = sum(int(x.get("passed", 0)) for x in t if isinstance(x, dict))
    failed = sum(int(x.get("failed", 0)) for x in t if isinstance(x, dict))
    return f"{passed} passed, {failed} failed"


def engine_cell(d: dict) -> str:
    e = d.get("engine") if isinstance(d.get("engine"), dict) else {}
    commit = str(e.get("commit") or "")[:12] or "-"
    return f"`{commit}`{' (dirty)' if e.get('dirty') else ''}"


def index_table(root: Path) -> str:
    rows = ["| gate | record | node | engine | result | tests | metrics |", "|---|---|---|---|---|---|---|"]
    for gate, (_, name, d) in sorted(newest_per_gate(root).items()):
        rows.append(f"| `{gate}` | `{name}` | {d.get('node', '-')} | {engine_cell(d)} | {d.get('result', '-')} "
                    f"| {tests_cell(d)} | {metrics_cell(d)} |")
    return "\n".join(rows)


def headline_table(root: Path, spec: Path) -> str:
    best = newest_per_gate(root)
    rows = ["| measure | value | record | engine |", "|---|---:|---|---|"]
    for n, line in enumerate(spec.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        label, gate, metric, unit = (line.split("\t") + ["", "", ""])[:4]
        if not (label and gate and metric):
            raise SystemExit(f"records_index: {spec}:{n}: a row is label<TAB>gate<TAB>metric<TAB>unit")
        if gate.startswith("@"):
            name = gate[1:]
            meta = root / "bench" / "records" / name / "meta.json"
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise SystemExit(f"records_index: {spec}:{n}: record {name} does not exist or is not readable")
        elif gate in best:
            _, name, d = best[gate]
        else:
            raise SystemExit(f"records_index: {spec}:{n}: gate {gate} has no record")
        if d.get("result") != "pass":
            raise SystemExit(f"records_index: {spec}:{n}: the record {name} is '{d.get('result')}', not 'pass'")
        m = d.get("metrics") if isinstance(d.get("metrics"), dict) else {}
        if metric not in m:
            raise SystemExit(f"records_index: {spec}:{n}: record {name} (gate {gate}) has no metric {metric}")
        v = m[metric]
        shown = format(v, ",") if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        rows.append(f"| {label} | {shown}{' ' + unit if unit else ''} | `{name}` | {engine_cell(d)} |")
    return "\n".join(rows)


def replace_between(path: Path, markers: tuple, body: str) -> int:
    start, end = markers
    text = path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        print(f"records_index: {path} needs exactly one {start} before one {end}", file=sys.stderr)
        return 1
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    path.write_text(f"{head}{start}\n{body}\n{end}{tail}", encoding="utf-8")
    print(f"records_index: {len(body.splitlines()) - 2} rows written to {path}")
    return 0


def cap_from_pins(root: Path) -> int:
    m = re.search(r"^GPU_CLOCK_CAP_MHZ=(\d+)\s*$", (root / "config" / "pins.sh").read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit("records_index: no GPU_CLOCK_CAP_MHZ in config/pins.sh")
    return int(m.group(1))


def cap_audit(root: Path) -> int:
    cap = cap_from_pins(root)
    counts = {"locked": 0, "sampled": 0, "none": 0}
    none, over, top = [], [], 0
    for meta in sorted((root / "bench" / "records").glob("*/meta.json")):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or d.get("node") not in ("node-1", "node-2", "node-3", "node-4"):
            continue
        name = meta.parent.name
        if any(p in t for f in meta.parent.rglob("*") if f.is_file()
               for t in [f.read_text(encoding="utf-8", errors="replace")] for p in LOCK_LINES):
            counts["locked"] += 1
            continue
        g = d.get("gpu_clock_sm_max")
        m = re.fullmatch(r"\s*(\d+) MHz,\s*\d+ MHz\s*", g) if isinstance(g, str) else None
        if not m:
            counts["none"] += 1
            none.append(name)
            continue
        counts["sampled"] += 1
        cur = int(m.group(1))
        top = max(top, cur)
        if cur > cap:
            over.append((name, cur))
    print(f"cap-audit: {counts['locked']} locked, {counts['sampled']} sampled (highest current clock {top} MHz), "
          f"{counts['none']} none; cap {cap} MHz")
    for name in none:
        print(f"none: {name}")
    for name, cur in over:
        print(f"OVER CAP: {name} sampled {cur} MHz, above {cap} MHz")
    return 1 if over else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--update", metavar="FILE", help="replace the text between the marker comments of this markdown file")
    ap.add_argument("--headline", metavar="SPEC.tsv", help="the headline table of this spec instead of the index of every gate")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="the recipe tree (default: this one)")
    ap.add_argument("--cap-audit", action="store_true", help="audit the GPU clock evidence of every record")
    args = ap.parse_args()
    root = Path(args.root)
    if args.cap_audit:
        return cap_audit(root)
    body = headline_table(root, Path(args.headline)) if args.headline else index_table(root)
    if not args.update:
        print(body)
        return 0
    return replace_between(Path(args.update), MARKERS["headline" if args.headline else "index"], body)


if __name__ == "__main__":
    sys.exit(main())
