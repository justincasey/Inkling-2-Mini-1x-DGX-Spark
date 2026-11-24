#!/usr/bin/env python3
"""Prefill sweep of the performance profile (stdlib only): cold prefill speed, TTFT and cached TTFT by prompt size, one Spark.

  sweep_prefill.py --out prefill.json [--url URL] [--window 262144] [--sizes 8192,...,top] [--conc 1,2,4] [--reps 3]
                   [--no-cached] [--skip-done]

The node comes from NODES/PORT (environment or .env) or INKLING2MINI_URL / --url. The window is the server's
--max-model-len: 262,144 (bootstrap) or 1,048,576 (long profile); sizes whose prompt plus 4,096 tokens does not fit are
left out. Default grid (sweeplib.PREFILL_GRID): 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, ascending, one warm-up first.

COLD. Every request is greedy, `prefix_cache: false`, max_tokens 1 (a cell that reports cached tokens fails). c = 1: bench.solo,
TTFT from the server's histogram `rexl3_request_ttft_seconds` (the prefill), tok/s = usage.prompt_tokens / TTFT. c > 1: c
distinct prompts (each stream its own block seed) at once through bench.together; aggregate prefill tok/s = the summed
usage.prompt_tokens / the wall span of the cell (a request is one token long, so the span is the slowest prefill). Cells with
c x size over --max-total (default 1,048,576) are skipped.

CACHED. After each c = 1 cold cell the same prompt is sent twice with the prefix cache on: the first stores it, the second is the
CACHED TTFT (histogram again), with cached_tokens; a hit needs cached_tokens >= prompt_tokens - 256, otherwise the cell
records `hit_ok: false` (the reach of the cache may be smaller than the prompt: the server's /v1/models says
prefix_cache_reach_tokens) and the sweep fails. --no-cached skips this.

Prompts: bench.long_ids up to 131,072 tokens (exact), bench.long_text above (the real size is usage.prompt_tokens, recorded).

Writes one JSON (rewritten after every cell) and prints `METRIC prefill_c<c>_<n>_tok_s=`, `prefill_c<c>_<n>_ttft_s=` (median over
runs), `prefill_cached_<n>_ttft_s=`, and `METRIC prefill_sweep_*` counts. <n> is the size in tokens. Exit 1 on a failed cell.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweeplib as sl  # noqa: E402

bench, ia = sl.bench, sl.ia
HIT_SLACK = 256


def hit_ok(cached, prompt_tokens, slack: int = HIT_SLACK) -> bool:
    """A prefix-cache hit: the server supplied all but `slack` prompt tokens."""
    return cached is not None and prompt_tokens is not None and cached >= prompt_tokens - slack


def metric_lines(res: dict) -> list[str]:
    cells: dict[tuple, list[dict]] = {}
    hits: dict[int, list[float]] = {}
    for r in res.get("cells", []):
        if r.get("ok") and r.get("tok_s") is not None:
            cells.setdefault((r["c"], r["size"]), []).append(r)
        h = r.get("hit")
        if h and h.get("hit_ok") and h.get("ttft_s") is not None:
            hits.setdefault(r["size"], []).append(h["ttft_s"])
    out = []
    for (c, size), rs in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out.append(f"METRIC prefill_c{c}_{size}_tok_s={statistics.median(r['tok_s'] for r in rs):.1f}")
        tt = [r["ttft_s"] for r in rs if r.get("ttft_s") is not None]
        if tt:
            out.append(f"METRIC prefill_c{c}_{size}_ttft_s={statistics.median(tt):.3f}")
    for size, v in sorted(hits.items()):
        out.append(f"METRIC prefill_cached_{size}_ttft_s={statistics.median(v):.3f}")
    return out


def cold_cell(api, size: int, c: int, rep: int, prompts: list) -> dict:
    t = time.time()
    cell = {"size": size, "c": c, "rep": rep, "hit": None}
    if c == 1:
        r = bench.solo(api, prompts[0], 1)
        n = r["prompt_tokens"]
        cell.update(prompt_tokens=n, ttft_s=r["ttft_s"], tok_s=round(n / r["ttft_s"], 1) if r["ttft_s"] > 0 else None,
                    cached_tokens=r["cached_tokens"], wall_s=r["wall_s"])
    else:
        out, span = bench.together(api, [(f"s{i}", p, 1) for i, p in enumerate(prompts)])
        n = sum(r["prompt_tokens"] for _, r in out)
        cell.update(prompt_tokens=[r["prompt_tokens"] for _, r in out], wall_s=span, tok_s=round(n / span, 1),
                    ttft_s=None, cached_tokens=[r["cached_tokens"] for _, r in out])
    cell["ok"] = cell["tok_s"] is not None and not any(cell["cached_tokens"] if isinstance(cell["cached_tokens"], list)
                                                       else [cell["cached_tokens"]])
    cell["epoch_start"], cell["epoch_end"] = t, time.time()
    return cell


def cached_ttft(api, prompt) -> dict:
    api.complete(prompt, 1, prefix_cache=True)                       # stores the prompt
    h = sl.solo_cached(api, prompt, 1)
    return {"ttft_s": h["ttft_s"], "cached_tokens": h["cached_tokens"], "prompt_tokens": h["prompt_tokens"],
            "hit_ok": hit_ok(h["cached_tokens"], h["prompt_tokens"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", required=True, help="the sweep's JSON")
    ap.add_argument("--window", type=int, default=sl.DEFAULT_WINDOW, help="the server's --max-model-len (262144 or 1048576)")
    ap.add_argument("--reserve", type=int, default=sl.RESERVE)
    ap.add_argument("--sizes", default=",".join(str(v) for v in sl.PREFILL_GRID), help="prompt tokens, `top` = window - reserve")
    ap.add_argument("--conc", default=",".join(str(v) for v in sl.PREFILL_CONC))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warm", type=int, default=4096, help="tokens of the one discarded warm-up request (0: none)")
    ap.add_argument("--max-total", type=int, default=sl.MAX_TOTAL_TOKENS, help="max streams x prompt tokens of one cell")
    ap.add_argument("--no-cached", action="store_true", help="no cached-TTFT pair after a c = 1 cold cell")
    ap.add_argument("--skip-done", action="store_true", help="resume: keep the cells already in --out (the server's prefix cache keeps what it stored)")
    a = ap.parse_args(argv)
    sizes = sl.grid(sl.parse_list(a.sizes, a.window, a.reserve), a.window, a.reserve)
    concs = [int(c) for c in a.conc.split(",")]
    base = ia.api_from_args(a)
    api = sl.ModeApi(base)
    res = (sl.load_json(a.out) if a.skip_done else None) or {"url": base.url, "model": base.model, "window": a.window, "sizes": sizes,
                                                              "conc": concs, "reps": a.reps, "cells": [], "warmup": None}
    done = {(c["size"], c["c"], c["rep"]) for c in res["cells"]}
    t_all = time.time()
    if a.warm and not res["warmup"]:
        r = bench.solo(api, sl.lane_prompt(api, a.warm, 99), 1)
        res["warmup"] = {"prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"]}
        sl.say(f"warm-up {a.warm}: ttft {r['ttft_s']} s (discarded)")
    for size in sizes:
        for c in concs:
            if not sl.fits(c, size, a.max_total):
                sl.say(f"skip c{c} size {size}: {c} x {size} tokens > --max-total {a.max_total}")
                continue
            for rep in range(1, a.reps + 1):
                if (size, c, rep) in done:
                    continue
                prompts = [sl.lane_prompt(api, size, i) for i in range(c)]
                cell = cold_cell(api, size, c, rep, prompts)
                if c == 1 and cell["ok"] and not a.no_cached:
                    cell["hit"] = cached_ttft(api, prompts[0])
                    cell["ok"] = cell["hit"]["hit_ok"]
                res["cells"].append(cell)
                sl.write_json(a.out, res)
                h = cell["hit"]
                sl.say(f"size {size:>8} c{c} run {rep}: {cell['tok_s']} tok/s, wall {cell['wall_s']:.2f} s"
                       f"{'' if cell['ttft_s'] is None else ', ttft %.3f s' % cell['ttft_s']}"
                       f"{'' if not h else ', cached ttft %.3f s (%s of %s cached)' % (h['ttft_s'], h['cached_tokens'], h['prompt_tokens'])}"
                       f"{'' if cell['ok'] else ' [CHECK FAILED]'}")
    bad = sum(1 for c in res["cells"] if not c["ok"]) + len(bench.FAILS)
    res["summary"] = {"cells": len(res["cells"]), "failed": bad, "bench_failures": bench.FAILS, "wall_s": time.time() - t_all}
    sl.write_json(a.out, res)
    for line in metric_lines(res):
        print(line)
    print(f"METRIC prefill_sweep_cells={len(res['cells'])}\nMETRIC prefill_sweep_failed={bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
