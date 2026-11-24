#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the newest record of a ladder gate: bench/records/<date>-<name>/meta.json with `"gate": "<GATE>"`.

A record is evidence only when it says so itself and the tree agrees. The newest record of the gate (by `date`, then by the
git commit time of its meta.json, then by directory name: two records of one day are ordered by when they were committed, not
by their names) must have:
  - `result` equal to "pass" (a newer failing record outranks an older passing one);
  - README.md beside meta.json, and `node` of the form node-1..node-4 when present;
  - no lab address (192.168.x.x, 10.100.x.x) in any text file of the record directory;
  - with --need-engine: `engine.commit` a 40-hex sha of a committed tree (`engine.dirty` false) that is an ancestor of the
    HEAD of the engine worktree (--engine), so a record counts once its lane is merged;
  - `tests`: a list of {bin, filter, passed, failed} with no failure and at least --min-tests passes in all;
  - `metrics`: every --metric NAME present and numeric, every --min-metric NAME=FLOOR at least FLOOR;
  - with --same-engine GATE (repeatable): the newest record of GATE passes and names the same engine commit, so a speed
    number only counts beside an exactness proof of the very engine that made it.
Prints one `record <dir>: ...` line and exits 0, or one `record: FAIL <reason>` line and exits 1. No network.
`--show-metric NAME` instead prints that metric of the newest record of the gate (pass or not) and exits 0, or exits 1.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HEX40 = re.compile(r"^[0-9a-f]{40}$")
LAB_ADDRESS = re.compile(r"192\.168\.\d+\.\d+|10\.100\.\d+\.\d+")
NODE = re.compile(r"^node-[1-4]$")


def commit_time(root, meta):
    """When the commit that ADDED the meta.json was made (seconds), or its mtime when it is not in git.

    Later edits to an old record (a scrub, a note) must not make it the newest of its gate, so the add commit counts: the last
    line `git log --diff-filter=A` prints is the oldest add."""
    try:
        r = subprocess.run(["git", "-C", str(root), "log", "--diff-filter=A", "--format=%ct", "--", str(meta)], capture_output=True, text=True)
        lines = r.stdout.split() if r.returncode == 0 else []
        if lines and lines[-1].isdigit():
            return int(lines[-1])
    except OSError:
        pass
    try:
        return int(meta.stat().st_mtime)
    except OSError:
        return 0


def newest(root, gate):
    found = []
    for meta in (root / "bench" / "records").glob("*/meta.json"):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("gate") == gate:
            found.append((str(d.get("date", "")), commit_time(root, meta), meta.parent.name, d, meta.parent))
    found.sort(key=lambda r: (r[0], r[1], r[2]))
    if not found:
        return None
    date, _, name, d, path = found[-1]
    return date, name, d, path


def is_ancestor(engine, sha):
    try:
        r = subprocess.run(["git", "-C", str(engine), "merge-base", "--is-ancestor", sha, "HEAD"], capture_output=True)
    except OSError:
        return False
    return r.returncode == 0


def check(args):
    root = Path(args.root)
    rec = newest(root, args.gate)
    if rec is None:
        return f"no record with gate '{args.gate}' under bench/records/", None
    _, name, d, path = rec
    if d.get("result") != "pass":
        return f"{name}: result is '{d.get('result')}', not 'pass'", None
    if not (path / "README.md").is_file():
        return f"{name}: README.md is missing", None
    node = d.get("node")
    if node is not None and not NODE.match(str(node)):
        return f"{name}: node '{node}' is not node-1..node-4", None
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.stat().st_size <= 4_000_000:
            try:
                if LAB_ADDRESS.search(f.read_text(encoding="utf-8", errors="ignore")):
                    return f"{name}: a lab address is in {f.relative_to(path)}", None
            except OSError:
                pass
    if args.need_engine:
        eng = d.get("engine") or {}
        sha = str(eng.get("commit", ""))
        if not HEX40.match(sha):
            return f"{name}: engine.commit is not a 40-hex sha", None
        if eng.get("dirty") is not False:
            return f"{name}: engine.dirty is not false (the run was not on a committed tree)", None
        if not is_ancestor(args.engine, sha):
            return f"{name}: engine commit {sha[:7]} is not in the engine head (merge the lane first)", None
    tests = d.get("tests") or []
    passed = 0
    for t in tests:
        try:
            p, f = int(t["passed"]), int(t["failed"])
        except (KeyError, TypeError, ValueError):
            return f"{name}: a tests entry lacks integer passed and failed", None
        if f != 0:
            return f"{name}: {t.get('bin')} {t.get('filter')} has {f} failing", None
        passed += p
    if passed < args.min_tests:
        return f"{name}: {passed} passing tests recorded (need {args.min_tests})", None
    metrics = d.get("metrics") or {}
    for m in args.metric:
        if not isinstance(metrics.get(m), (int, float)) or isinstance(metrics.get(m), bool):
            return f"{name}: metric '{m}' is missing or not a number", None
    for spec in args.min_metric:
        key, _, floor = spec.partition("=")
        v = metrics.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < float(floor):
            return f"{name}: metric '{key}' is {v!r}, need at least {floor}", None
    mine = str((d.get("engine") or {}).get("commit", ""))
    for other in args.same_engine:
        o = newest(root, other)
        if o is None:
            return f"{name}: no '{other}' record to pair with (a speed record counts only beside an exactness record of the same engine)", None
        _, oname, od, _ = o
        if od.get("result") != "pass":
            return f"{name}: the newest '{other}' record {oname} is '{od.get('result')}', not 'pass'", None
        theirs = str((od.get("engine") or {}).get("commit", ""))
        if not HEX40.match(mine) or mine != theirs:
            return f"{name}: engine {mine[:7] or 'n/a'} differs from {oname}'s {theirs[:7] or 'n/a'} ('{other}'): not proven byte-exact on the same engine commit", None
    eng = (d.get("engine") or {}).get("commit", "")
    return None, f"record {name}: pass, engine {str(eng)[:7] or 'n/a'}, {passed} tests, {len(args.metric)} metrics checked"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gate")
    ap.add_argument("--root", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--need-engine", action="store_true")
    ap.add_argument("--min-tests", type=int, default=0)
    ap.add_argument("--metric", action="append", default=[])
    ap.add_argument("--min-metric", action="append", default=[], help="NAME=FLOOR: a numeric metric of at least FLOOR")
    ap.add_argument("--same-engine", action="append", default=[], help="GATE: its newest record passes and has this record's engine commit")
    ap.add_argument("--show-metric", default="", help="print this metric of the newest record of the gate and exit")
    args = ap.parse_args()
    if args.show_metric:
        rec = newest(Path(args.root), args.gate)
        v = ((rec[2].get("metrics") or {}).get(args.show_metric)) if rec else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            print(v)
            return 0
        return 1
    err, ok = check(args)
    if err:
        print(f"record: FAIL {err}")
        return 1
    print(ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
