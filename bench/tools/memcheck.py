#!/usr/bin/env python3
"""The `memory` gate from `./inkling2mini.sh memwatch report` (one node, one guard; stdlib only).

  memcheck.py --report memwatch.txt [--floor-mib 8192] [--boot-log boot.log --recorded-prefix-gib 6.03]

The report is one line, `memwatch guard=running window=... window_min_mib=... loading_min_mib=... ready_min_mib=...
min_mib=... min_at=... now_mib=...`. `min_mib` is the lowest MemAvailable since the guard armed (the boot's loading phase
included) and must stay at or above the floor (default 8192 MiB, the engine's own floor; pass MEMORY_FLOOR_GIB * 1024 when
the boot set another); the guard must still be running (a stopped guard watched nothing since).

Normalization (optional, off unless both options are given): the capacity plan gives the prefix cache whatever memory is
free at boot, so two boots of the same engine can differ in headroom by the size of that cache. With --boot-log (a file
holding the capacity line, `prefix_gib=...`) and --recorded-prefix-gib (the cache size of the recorded configuration), a boot
whose cache is smaller than the recorded one measured more headroom than the recorded configuration has: that difference
is subtracted from the minimum before the comparison, so the gate gives the same verdict on any boot.

Prints `METRIC mem_min_gib=` (raw), `METRIC mem_loading_min_gib=`, `METRIC mem_ready_min_gib=` when the report has them and,
with normalization, `METRIC mem_min_gib_norm=` and `METRIC prefix_gib_recorded_gap=`; each problem on stderr. Exit 1 on any
problem.
"""

from __future__ import annotations

import argparse
import re
import sys

FLOOR_MIB = 8192


def parse_report(text: str) -> dict[str, str]:
    """The key=value fields of the (last) `memwatch ...` line of a report; {} when there is none."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"memwatch (.*)$", line.strip())
        if m:
            out = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
    return out


def prefix_gib(boot_log: str) -> float | None:
    """`prefix_gib=` of the capacity line (an `event=..._plan` line), or None."""
    m = re.search(r"event=\w*plan\b.* prefix_gib=([0-9.]+)", boot_log)
    return float(m.group(1)) if m else None


def check(report: str, boot_log: str | None = None, recorded_prefix_gib: float | None = None,
          floor_mib: int = FLOOR_MIB) -> tuple[list[str], dict[str, float]]:
    """(problems, metrics)."""
    problems: list[str] = []
    metrics: dict[str, float] = {}
    f = parse_report(report)
    if not f:
        return ["no `memwatch ...` line in the report"], metrics
    if f.get("guard") != "running":
        problems.append(f"the memory guard is {f.get('guard')!r}, not running")
    try:
        low = int(f["min_mib"])
    except (KeyError, ValueError):
        return problems + ["no min_mib in the memwatch report"], metrics
    metrics["mem_min_gib"] = round(low / 1024, 2)
    for key, name in (("loading_min_mib", "mem_loading_min_gib"), ("ready_min_mib", "mem_ready_min_gib")):
        if f.get(key, "").isdigit():
            metrics[name] = round(int(f[key]) / 1024, 2)
    gap_mib = 0.0
    if boot_log is not None and recorded_prefix_gib is not None:
        prefix = prefix_gib(boot_log)
        if prefix is None:
            problems.append("no capacity line with prefix_gib in the boot log")
        else:
            gap_mib = max(0.0, recorded_prefix_gib - prefix) * 1024
        metrics["prefix_gib_recorded_gap"] = round(gap_mib / 1024, 2)
        metrics["mem_min_gib_norm"] = round((low - gap_mib) / 1024, 2)
    if low - gap_mib < floor_mib:
        what = "" if not gap_mib else f", {low - gap_mib:.0f} MiB at the recorded prefix cache,"
        problems.append(f"MemAvailable minimum {low} MiB{what} is under the {floor_mib} MiB floor")
    return problems, metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="./inkling2mini.sh memwatch report output")
    ap.add_argument("--floor-mib", type=int, default=FLOOR_MIB, help="the floor (default: %(default)s)")
    ap.add_argument("--boot-log", help="a file holding the boot's capacity line (for the normalization)")
    ap.add_argument("--recorded-prefix-gib", type=float, help="the recorded configuration's prefix cache, GiB")
    a = ap.parse_args(argv)
    if (a.boot_log is None) != (a.recorded_prefix_gib is None):
        ap.error("--boot-log and --recorded-prefix-gib go together")
    with open(a.report) as f:
        report = f.read()
    boot_log = None
    if a.boot_log:
        with open(a.boot_log) as f:
            boot_log = f.read()
    problems, metrics = check(report, boot_log, a.recorded_prefix_gib, a.floor_mib)
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
