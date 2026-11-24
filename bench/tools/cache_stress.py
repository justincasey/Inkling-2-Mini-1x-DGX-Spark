#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The prefix cache under eviction and concurrency against the node's API (stdlib only, one server): every cached reply must equal
its uncached reference, greedy, id for id.

  cache_stress.py --out cache-stress.json [--url URL] [--model NAME] [--prompts 24] [--tokens 1500] [--max-tokens 32]

evict       `--prompts` distinct prompts of `--tokens` ids (more than the cache has snapshot slots). Each one's uncached reply first
            (the reference), then every prompt once with the cache on in order (each leaves a snapshot; the least recently used are
            evicted), then again in REVERSE order: the most recent are still stored (hits: restored from a snapshot), the oldest were
            evicted (misses: prefilled cold, again snapshotted). Every reply equals its reference; at least one eviction, one hit and
            one miss happen (`rexl3_prefix_cache_evictions_total`, cached_tokens).
concurrent  one prompt is stored (a long shared base and a short tail), then three requests sharing the base (each with a tail of its own:
            restored from the stored snapshot) and one unrelated prompt run at the same moment on four slots, 128 tokens each; every
            reply equals its solo uncached reference, and the three that share the base are served from the cache.

Prints `METRIC` lines and the counters; exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rexl3"))
import sweeplib as sl  # noqa: E402

bench, ia = sl.bench, sl.ia
COUNTERS = ("lookups_total", "hits_total", "tokens_saved_total", "inserts_total", "evictions_total", "restore_failures_total",
            "entries", "pages", "snapshot_slots", "page_blocks")


def counters(api) -> dict:
    m = api.metrics()
    return {k: m.get("rexl3_prefix_cache_" + k) for k in COUNTERS}


def delta(a: dict, b: dict) -> dict:
    return {k: (b[k] or 0) - (a[k] or 0) for k in COUNTERS if not k.startswith(("entries", "pages", "snapshot", "page_"))}


def ask(api, prompt, max_tokens: int, cache: bool) -> dict:
    resp, wall = api.complete(prompt, max_tokens, prefix_cache=cache)
    ids, finish = ia.generated_ids(resp)
    return {"ids": ids, "finish": finish, "cached": ia.cached_tokens(resp.get("usage")), "prompt_tokens": resp["usage"]["prompt_tokens"],
            "wall_s": round(wall, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args(argv)
    api = ia.api_from_args(a)
    errs: list[str] = []
    rec: dict = {"meta": {"prompts": a.prompts, "tokens": a.tokens, "max_tokens": a.max_tokens, "model": api.model}}
    c0 = counters(api)
    if not c0.get("snapshot_slots"):
        raise SystemExit("the server runs no prefix cache: boot with PREFIX_CACHE=on")
    rec["meta"]["snapshot_slots"] = int(c0["snapshot_slots"])
    if a.prompts <= c0["snapshot_slots"]:
        errs.append(f"--prompts {a.prompts} does not exceed the {int(c0['snapshot_slots'])} snapshot slots: nothing would be evicted")

    # ── evict ──
    prompts = [sl.lane_prompt(api, a.tokens, 200 + i) for i in range(a.prompts)]
    refs = [ask(api, p, a.max_tokens, False) for p in prompts]
    first = []
    for i, p in enumerate(prompts):
        r = ask(api, p, a.max_tokens, True)
        first.append(r)
        if r["ids"] != refs[i]["ids"]:
            errs.append(f"evict: prompt {i} first pass (cache on, cold) differs from its uncached reply")
    c1 = counters(api)
    second = {}
    for i in reversed(range(a.prompts)):
        r = ask(api, prompts[i], a.max_tokens, True)
        second[i] = r
        if r["ids"] != refs[i]["ids"]:
            errs.append(f"evict: prompt {i} second pass (cache on, {r['cached']} cached tokens) differs from its uncached reply")
    c2 = counters(api)
    hits = sorted(i for i, r in second.items() if r["cached"])
    misses = sorted(i for i, r in second.items() if not r["cached"])
    ev = delta(c0, c2)["evictions_total"]
    if not hits or not misses or ev < 1:
        errs.append(f"evict: {len(hits)} hits, {len(misses)} misses, {ev:.0f} evictions (each must be at least 1)")
    if (c2["restore_failures_total"] or 0) - (c0["restore_failures_total"] or 0):
        errs.append("evict: a restore failed")
    rec["evict"] = {"hits": hits, "misses": misses, "counters_before": c0, "counters_after_first_pass": c1, "counters_after": c2,
                    "first_pass_cached": [r["cached"] for r in first], "second_pass_cached": {str(i): r["cached"] for i, r in second.items()},
                    "reference_sha256": [bench.digest(r["ids"]) for r in refs]}
    print(f"evict: {a.prompts} prompts x {a.tokens} ids; second pass (reverse): {len(hits)} hits {hits}, {len(misses)} misses; "
          f"evictions {ev:.0f}; every reply equal to its uncached one: {not any(e.startswith('evict:') for e in errs)}")

    # ── concurrent ──
    base_n, nreq = 3000, 128
    base = sl.lane_prompt(api, base_n, 900)
    tails = [sl.lane_prompt(api, 4 * 32, 910 + i)[:24 + 8 * i] for i in range(4)]
    other = sl.lane_prompt(api, 2000, 950)
    ask(api, base + tails[0], 4, True)            # stores the base's snapshot (its end cut is inside the base)
    jobs = [base + tails[1], base + tails[2], base + tails[3], other]
    solo = [ask(api, j, nreq, False) for j in jobs]
    c3 = counters(api)
    out: list = [None] * len(jobs)
    barrier = threading.Barrier(len(jobs))

    def one(i):
        barrier.wait()
        out[i] = ask(api, jobs[i], nreq, True)
    th = [threading.Thread(target=one, args=(i,)) for i in range(len(jobs))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    c4 = counters(api)
    for i, (s, r) in enumerate(zip(solo, out)):
        if r["ids"] != s["ids"]:
            errs.append(f"concurrent: stream {i} (cached {r['cached']}) differs from its solo uncached reply")
    if not all(out[i]["cached"] for i in range(3)):
        errs.append(f"concurrent: the streams sharing the base were served {[o['cached'] for o in out[:3]]} cached tokens (all must be > 0)")
    rec["concurrent"] = {"cached_tokens": [o["cached"] for o in out], "reference_sha256": [bench.digest(s["ids"]) for s in solo],
                         "counters_before": c3, "counters_after": c4}
    print(f"concurrent: 4 streams, cached tokens {[o['cached'] for o in out]}; every reply equal to its solo uncached one: "
          f"{not any(e.startswith('concurrent:') for e in errs)}")
    metrics = {"cache_stress_evictions": ev, "cache_stress_hits": len(hits), "cache_stress_misses": len(misses),
               "cache_stress_concurrent_cached_min": min(o["cached"] or 0 for o in out[:3])}
    rec["metrics"], rec["errors"] = metrics, errs
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")
    for e in errs:
        print(f"FAIL {e}")
    print(f"cache_stress: {'pass' if not errs else 'FAIL'} -> {a.out}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
