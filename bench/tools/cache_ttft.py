#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time to first token with and without the prefix cache, by prompt size, against the node's API (stdlib only, one server).

  cache_ttft.py --out cache-ttft.json [--url URL] [--model NAME] [--sizes 4096,12288,32768] [--reps 3]

Per size n and repetition r (every prompt its own filler stream, so nothing of it is cached yet; every request greedy, one token):

  cold      a prompt of n ids with "prefix_cache": false: the cold prefill (bench.solo)
  first     another prompt of n ids with the cache on, first sighting: the same prefill plus the snapshots it takes
  shared    the first prompt's first n - 64 ids and 64 new ones: the shared prefix is cached (the end cut of `first`), 64 tokens are prefilled
  warm      the first prompt again: the same hit

The time to first token is the server's own histogram `rexl3_request_ttft_seconds` (the prefill; read around each request, the server
otherwise idle). A reply that should be a hit and is not (cached_tokens < n - 512), or a cold one that is served from the cache, fails
the run. Prints `METRIC cache_ttft_<kind>_<n>_s=<median over reps>` and the speed-up, and the cache's counters before and after.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rexl3"))
import sweeplib as sl  # noqa: E402

bench, ia = sl.bench, sl.ia
TAIL = 64
HIT_SLACK = 512
COUNTERS = ("rexl3_prefix_cache_lookups_total", "rexl3_prefix_cache_hits_total", "rexl3_prefix_cache_tokens_saved_total",
            "rexl3_prefix_cache_inserts_total", "rexl3_prefix_cache_evictions_total", "rexl3_prefix_cache_entries",
            "rexl3_prefix_cache_pages", "rexl3_prefix_cache_snapshot_slots", "rexl3_prefix_cache_page_blocks")


def counters(api) -> dict:
    m = api.metrics()
    return {k.replace("rexl3_prefix_cache_", ""): m.get(k) for k in COUNTERS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="4096,12288,32768")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    sizes = [int(s) for s in a.sizes.split(",")]
    errs: list[str] = []
    cells: list[dict] = []
    before = counters(api)
    if not before.get("snapshot_slots"):
        raise SystemExit("the server runs no prefix cache (rexl3_prefix_cache_snapshot_slots is 0 or absent): boot with PREFIX_CACHE=on")
    # a warm-up, discarded: one cold request of the smallest size (kernels, allocator)
    bench.solo(api, sl.lane_prompt(api, min(sizes), 900), 1)
    for si, n in enumerate(sizes):
        for r in range(a.reps):
            # every (size, rep) its own filler streams: prompts of one stream share their leading blocks, and a shared block is a hit
            lane = 1000 + 100 * si + 10 * r
            cold_p = sl.lane_prompt(api, n, lane + 1)
            first_p = sl.lane_prompt(api, n, lane + 2)
            tail = sl.lane_prompt(api, 4 * TAIL, lane + 3)[:TAIL]
            shared_p = first_p[: len(first_p) - TAIL] + tail
            cold = bench.solo(api, cold_p, 1)
            first = sl.solo_cached(api, first_p, 1)
            shared = sl.solo_cached(api, shared_p, 1)
            warm = sl.solo_cached(api, first_p, 1)
            for name, x in (("cold", cold), ("first", first)):
                if x["cached_tokens"]:
                    errs.append(f"n={n} rep {r}: the {name} request was served {x['cached_tokens']} tokens from the cache")
            for name, x in (("shared", shared), ("warm", warm)):
                if x["cached_tokens"] is None or x["cached_tokens"] < n - HIT_SLACK:
                    errs.append(f"n={n} rep {r}: the {name} request was served {x['cached_tokens']} tokens from the cache (wanted >= {n - HIT_SLACK})")
            cells.append({"size": n, "rep": r, **{k: sl.slim(v) for k, v in (("cold", cold), ("first", first), ("shared", shared), ("warm", warm))}})
            print(f"n={n:6d} rep {r}: cold {cold['ttft_s']:.3f}s  first(cache on) {first['ttft_s']:.3f}s  shared {shared['ttft_s']:.3f}s "
                  f"(cached {shared['cached_tokens']})  warm {warm['ttft_s']:.3f}s (cached {warm['cached_tokens']})", flush=True)
    after = counters(api)
    metrics: dict[str, float] = {}
    for n in sizes:
        rows = [c for c in cells if c["size"] == n]
        med = {k: statistics.median(c[k]["ttft_s"] for c in rows) for k in ("cold", "first", "shared", "warm")}
        for k, v in med.items():
            metrics[f"cache_ttft_{k}_{n}_s"] = round(v, 4)
        metrics[f"cache_ttft_speedup_shared_{n}"] = round(med["cold"] / med["shared"], 2)
        metrics[f"cache_ttft_first_over_cold_{n}"] = round(med["first"] / med["cold"], 3)
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")
    with open(a.out, "w") as f:
        json.dump({"meta": {"sizes": sizes, "reps": a.reps, "tail": TAIL, "counters_before": before, "counters_after": after,
                            "model": api.model}, "cells": cells, "metrics": metrics, "errors": errs}, f, indent=1)
    for e in errs:
        print(f"FAIL {e}")
    print(f"cache_ttft: {'pass' if not errs else 'FAIL'} -> {a.out}; counters {before} -> {after}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
