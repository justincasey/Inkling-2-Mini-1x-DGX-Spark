#!/usr/bin/env python3
"""The memory arithmetic of one `./inkling2mini.sh bench profile` run (stdlib only).

  profile_table.py RUN_DIR [RUN_DIR ...] [--json]

Reads capacity.txt (the engine's capacity line), ready.txt (GPU MiB of the server process and MemAvailable at ready),
samples.txt (the node-side sampler) and memwatch.txt (the guard's minimum) and prints for each run one line of the table
docs/FIT.md "Measured" keeps: kv_tokens, lanes_at_max_context, MemAvailable at ready, guard minimum, GPU MiB at ready and at
the sampled peak, and the overhead: what the process holds beyond what the plan prices (weights + KV pages + slot state +
prefix cache + vision tower), the figure the plan's 5 GiB `reserve_gib` has to cover. With --json the numbers of every run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MIB = 1 << 20


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def one(d: Path) -> dict:
    cap = fields((d / "capacity.txt").read_text().splitlines()[0])
    ready = fields((d / "ready.txt").read_text().splitlines()[0])
    mem = fields(next(l for l in (d / "memwatch.txt").read_text().splitlines() if l.startswith("memwatch")))
    gpu, avail = [], []
    for line in (d / "samples.txt").read_text().splitlines():
        f = fields(line)
        if "gpu_mib" in f and f["gpu_mib"].isdigit():
            gpu.append(int(f["gpu_mib"]))
            avail.append(int(f["avail_mib"]))
    pages, bpt = int(cap["kv_pages"]), int(cap["kv_bytes_per_token"])
    kv_mib = pages * 256 * bpt / MIB
    state_mib = int(cap["slots"]) * float(cap["state_mib_per_slot"])
    priced = {"weights": float(cap["weights_gib"]) * 1024, "kv": kv_mib, "slot_state": state_mib,
              "prefix_cache": float(cap.get("prefix_gib", 0)) * 1024, "vision_tower": float(cap.get("tower_gib", 0)) * 1024}
    gpu_ready = int(ready["gpu_mib"])
    out = {"run": d.name, "slots": int(cap["slots"]), "max_context": int(cap["max_context"]), "kv_tokens": int(cap["kv_tokens"]),
           "kv_pages": pages, "lanes_at_max_context": int(cap["lanes_at_max_context"]), "kv_gib": float(cap["kv_gib"]),
           "prefix_gib": float(cap.get("prefix_gib", 0)), "tower_gib": float(cap.get("tower_gib", 0)),
           "spare_gib": float(cap["spare_gib"]), "free_after_gib": float(cap["free_after_gib"]),
           "gpu_ready_mib": gpu_ready, "gpu_peak_mib": max(gpu) if gpu else None,
           "avail_ready_mib": int(ready["avail_mib"]), "avail_sample_min_mib": min(avail) if avail else None,
           "guard_min_mib": int(mem["min_mib"]), "guard_window_min_mib": int(mem["window_min_mib"]) if mem["window_min_mib"].isdigit() else None,
           "priced_mib": {k: round(v) for k, v in priced.items()},
           "overhead_mib": round(gpu_ready - sum(priced.values())), "overhead_gib": round((gpu_ready - sum(priced.values())) / 1024, 2),
           "plan_reserve_gib": float(cap["reserve_gib"]), "plan_floor_gib": float(cap["floor_gib"])}
    return out


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    runs = [Path(a) for a in argv if not a.startswith("--")]
    if not runs:
        print(__doc__)
        return 2
    rows = [one(r) for r in runs]
    if as_json:
        print(json.dumps(rows, indent=1))
        return 0
    print("| run | slots x window | kv_tokens | lanes_at_max | prefix GiB | MemAvailable at ready GiB | guard min GiB | GPU ready MiB | GPU peak MiB | overhead GiB |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['run']} | {r['slots']} x {r['max_context']} | {r['kv_tokens']} | {r['lanes_at_max_context']} | {r['prefix_gib']:.2f} | "
              f"{r['avail_ready_mib'] / 1024:.1f} | {r['guard_min_mib'] / 1024:.2f} | {r['gpu_ready_mib']} | {r['gpu_peak_mib']} | {r['overhead_gib']:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
