#!/usr/bin/env python3
"""Decode sweep of the performance profile (stdlib only): decode speed by number of streams and by context depth, one Spark.

  sweep_decode.py --out decode.json [--url URL] [--phase shallow|depth|both] [--window 262144] [--ctxs 4096,...,top]
                  [--conc 1,2,4,8] [--modes default,serial] [--reps 3] [--depth-reps 1] [--skip-done]

The node comes from NODES/PORT (environment or .env) or INKLING2MINI_URL / --url. The window is the server's
--max-model-len: 262,144 (bootstrap) or 1,048,576 (long profile); cells whose prompt plus 4,096 tokens does not fit are
left out. Default context grid (sweeplib.DECODE_GRID): 4K, 16K, 32K, 64K, 128K, 256K, 512K, 1M. Streams: 1, 2, 4, 8.

MODES. `default` is the server as booted (the drafter drafts when `--spec` is on; without it this is plain decode);
`serial` sends `"draft": false` on every request (no speculation). Without a drafter both rows are the same path.

SHALLOW (context 4K). The four decode prompts of bench/tools/bench.py (prose, code, structured, json; 512 tokens, greedy,
prefix cache off). c = 1: each prompt alone through bench.solo, the rate from the server's histograms
`rexl3_request_ttft_seconds` / `rexl3_request_e2e_seconds` ((tokens - 1) / (e2e - ttft)), the cell value the mean of the
four. c > 1: c streams at once through bench.together (prompts cycled), aggregate tok/s = usage.completion_tokens summed
over the wall span. Tokens are always usage.completion_tokens.

DEPTH. A cold prompt of --ctxs tokens per stream (bench.long_ids up to 131,072 tokens, bench.long_text above; each stream
its own block seed, so the streams differ), 256 generated tokens, prefix cache off. c = 1: bench.solo (TTFT = the prefill,
decode speed after it). c > 1: the streams are timed over their OVERLAP WINDOW (the last first token to the first last
token): steady aggregate tok/s, per-stream tok/s; a cell whose window is under 3 s or 0.8 of a stream's span is flagged.
Cells with streams x tokens over --max-total (default 1,048,576) are skipped: they would exceed the KV pool of the bootstrap
profile; raise it on the long profile.

Writes one JSON (rewritten after every run) and prints `METRIC decode_<mode>_c<c>_ctx<n>_tok_s=` (median over runs; at
c = 1 the speed of one stream, at c > 1 the aggregate), `..._tpr` (tokens per round) where the drafter reports, and
`METRIC decode_sweep_*` run counts. Exit 1 on a request error, a cache hit on a cold request, or a flagged depth cell.
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


def metric_lines(res: dict) -> list[str]:
    """METRIC lines of a decode sweep result: the median over runs of every (phase, mode, c, ctx) cell."""
    cells: dict[tuple, list[dict]] = {}
    for phase in ("shallow", "depth"):
        for r in res.get(phase, []):
            if r.get("ok"):
                cells.setdefault((r["mode"], r["c"], r["ctx"]), []).append(r)
    out = []
    for (mode, c, ctx), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        vals = [r["tok_s"] for r in rs if r.get("tok_s") is not None]
        if vals:
            out.append(f"METRIC decode_{mode}_c{c}_ctx{ctx}_tok_s={statistics.median(vals):.3f}")
        tpr = [r["tokens_per_round"] for r in rs if r.get("tokens_per_round") is not None]
        if tpr:
            out.append(f"METRIC decode_{mode}_c{c}_ctx{ctx}_tpr={statistics.median(tpr):.3f}")
    return out


def shallow_run(api, prompts: dict, mode: str, c: int, rep: int) -> dict:
    names = list(prompts)
    t = time.time()
    run = {"phase": "shallow", "mode": mode, "c": c, "ctx": 4096, "rep": rep, "ok": True}
    if c == 1:
        rs = [bench.solo(api, ids, bench.DECODE_TOKENS) for ids in prompts.values()]
        rates = [r["decode_tok_s"] for r in rs]
        run.update(per_prompt={n: sl.slim(r) for n, r in zip(names, rs)}, tok_s=statistics.fmean(rates) if all(rates) else None,
                   ttft_s=statistics.fmean(r["ttft_s"] for r in rs), tokens=sum(r["completion_tokens"] for r in rs))
        replies = rs
    else:
        jobs = [(f"{names[i % len(names)]}#{i}", prompts[names[i % len(names)]], bench.DECODE_TOKENS) for i in range(c)]
        out, span = bench.together(api, jobs)
        replies = [r for _, r in out]
        tokens = sum(r["completion_tokens"] for r in replies)
        run.update(tok_s=tokens / span, tokens=tokens, wall_s=span, per_stream_tok_s=tokens / span / c,
                   streams={k: sl.slim(r) for k, r in out})
    spec = sl.spec_summary(replies)
    run["spec"] = spec
    run["tokens_per_round"] = spec["tokens_per_round"] if spec else None
    run["ok"] = run.get("tok_s") is not None
    run["epoch_start"], run["epoch_end"] = t, time.time()
    return run


def depth_run(api, prompts: list, mode: str, c: int, ctx: int, rep: int) -> dict:
    t = time.time()
    run = {"phase": "depth", "mode": mode, "c": c, "ctx": ctx, "rep": rep, "ok": True}
    if c == 1:
        r = bench.solo(api, prompts[0], sl.DEPTH_TOKENS)
        run.update(tok_s=r["decode_tok_s"], ttft_s=r["ttft_s"], prompt_tokens=r["prompt_tokens"], tokens=r["completion_tokens"],
                   spec=r["spec"], tokens_per_round=(r["spec"] or {}).get("tokens_per_round"),
                   prefill_tok_s=round(r["prompt_tokens"] / r["ttft_s"], 1) if r["ttft_s"] else None,
                   finish_reason=r["finish_reason"], ok=r["decode_tok_s"] is not None)
    else:
        bodies = [{"model": api.model, "prompt": p, "max_tokens": sl.DEPTH_TOKENS, "temperature": 0, "prefix_cache": False,
                   **({"draft": False} if api.serial else {})} for p in prompts]
        streams = sl.fire_streams(api, bodies)
        om = sl.overlap_metrics(streams)
        errs = [s["error"] for s in streams if s["error"]]
        cached = [sl.ia.cached_tokens(s["usage"]) or 0 for s in streams]
        spec = sl.spec_summary([{"spec": ia.spec_counters(s["usage"]), "completion_tokens": s["tokens"]} for s in streams])
        run.update(om, tok_s=om.get("aggregate_tok_s"), errors=errs, cached_tokens=cached, spec=spec,
                   tokens_per_round=spec["tokens_per_round"] if spec else None,
                   prompt_tokens=[(s["usage"] or {}).get("prompt_tokens") for s in streams],
                   tokens=sum(s["tokens"] for s in streams))
        run["ok"] = bool(om.get("ok")) and not errs and not any(cached)
        if not om.get("ok"):
            run["flag"] = om.get("reason") or "short overlap window"
    run["epoch_start"], run["epoch_end"] = t, time.time()
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", required=True, help="the sweep's JSON")
    ap.add_argument("--phase", default="both", choices=("shallow", "depth", "both"))
    ap.add_argument("--window", type=int, default=sl.DEFAULT_WINDOW, help="the server's --max-model-len (262144 or 1048576)")
    ap.add_argument("--reserve", type=int, default=sl.RESERVE)
    ap.add_argument("--ctxs", default=",".join(str(v) for v in sl.DECODE_GRID),
                    help="depth contexts in prompt tokens, `top` = window - reserve (default: the documented grid)")
    ap.add_argument("--conc", default=",".join(str(v) for v in sl.DECODE_CONC))
    ap.add_argument("--modes", default="default,serial", help="shallow modes (default,serial)")
    ap.add_argument("--depth-modes", default="default,serial")
    ap.add_argument("--reps", type=int, default=3, help="runs of a shallow cell")
    ap.add_argument("--depth-reps", type=int, default=1, help="runs of a depth cell (each is a cold prefill)")
    ap.add_argument("--max-total", type=int, default=sl.MAX_TOTAL_TOKENS, help="max streams x prompt tokens of one depth cell")
    ap.add_argument("--skip-done", action="store_true", help="resume: keep the runs already in --out")
    a = ap.parse_args(argv)
    for m in a.modes.split(",") + a.depth_modes.split(","):
        if m not in ("default", "serial"):
            raise SystemExit(f"unknown mode {m!r} (default, serial)")
    concs = [int(c) for c in a.conc.split(",")]
    ctxs = sl.grid(sl.parse_list(a.ctxs, a.window, a.reserve), a.window, a.reserve)
    base = ia.api_from_args(a)
    api = sl.ModeApi(base)
    res = (sl.load_json(a.out) if a.skip_done else None) or {"url": base.url, "model": base.model, "window": a.window, "ctxs": ctxs,
                                                              "conc": concs, "shallow": [], "depth": []}
    done = {(r["phase"], r["mode"], r["c"], r["ctx"], r["rep"]) for k in ("shallow", "depth") for r in res[k]}

    def record(run: dict) -> None:
        res[run["phase"]].append(run)
        sl.write_json(a.out, res)
        v = run.get("tok_s")
        sl.say(f"{run['phase']} {run['mode']:>7} c{run['c']} ctx {run['ctx']} run {run['rep']}: "
               f"{'%.2f tok/s' % v if v else 'FAILED'}"
               f"{'' if run.get('tokens_per_round') is None else ', tokens/round %.3f' % run['tokens_per_round']}"
               f"{'' if run['ok'] else ' [FLAGGED %s]' % (run.get('flag') or run.get('errors') or run.get('cached_tokens') or '')}")

    if a.phase in ("shallow", "both"):
        prompts = {n: api.tokenize(t) for n, t in bench.DECODE_TEXTS.items()}
        api.complete(prompts["prose"], 64, prefix_cache=False)           # warm-up, discarded
        for mode in a.modes.split(","):
            mapi = api.with_mode(mode == "serial")
            for c in concs:
                for rep in range(1, a.reps + 1):
                    if ("shallow", mode, c, 4096, rep) not in done:
                        record(shallow_run(mapi, prompts, mode, c, rep))
    if a.phase in ("depth", "both"):
        for ctx in ctxs:
            for mode in a.depth_modes.split(","):
                mapi = api.with_mode(mode == "serial")
                for c in concs:
                    if not sl.fits(c, ctx, a.max_total):
                        sl.say(f"skip c{c} ctx {ctx}: {c} x {ctx} tokens > --max-total {a.max_total}")
                        continue
                    for rep in range(1, a.depth_reps + 1):
                        if ("depth", mode, c, ctx, rep) in done:
                            continue
                        prompts_c = [sl.lane_prompt(api, ctx, i) for i in range(c)]
                        record(depth_run(mapi, prompts_c, mode, c, ctx, rep))
    bad = sum(1 for k in ("shallow", "depth") for r in res[k] if not r["ok"]) + len(bench.FAILS)
    res["summary"] = {"shallow_runs": len(res["shallow"]), "depth_runs": len(res["depth"]), "flagged": bad, "bench_failures": bench.FAILS}
    sl.write_json(a.out, res)
    for line in metric_lines(res):
        print(line)
    print(f"METRIC decode_sweep_shallow_runs={len(res['shallow'])}\nMETRIC decode_sweep_depth_runs={len(res['depth'])}\n"
          f"METRIC decode_sweep_flagged={bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
