#!/usr/bin/env python3
"""Cold prefill over a size grid with a unique prompt every pass (stdlib only; one server, TP1).

Sizes 8K, 32K, 128K and 224K by default (`--sizes 8192,32768`). For each size: `--passes` passes (default 4), the first discarded as the
warm-up, each pass a different filler stream (block seed offset 1000 x pass), so no request can hit a cache even with the prefix cache on;
the requests also carry `"prefix_cache": false`. The prompt of pass p at size n is deterministic, so two builds are comparable reply by
reply. A request is greedy with `--tokens` (default 16) new tokens; the time is the server's `rexl3_request_ttft_seconds` (bench.py's
`solo`), the rate is `usage.prompt_tokens` over it, the token count is `usage.completion_tokens`. `--decode-after N` makes every kept pass
generate N tokens, so the decode rate after a cold prompt (`decode_<n>_tok_s`, median of the kept passes) is measured too.

Prints `METRIC prefill_<n>_tok_s=<median>` (n in K), the passes, and writes `--out` JSON with every pass and the generated ids.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import inkling_api as ia  # noqa: E402


def ids_for(api: ia.Api, n: int, offset: int) -> list[int]:
    tail = api.tokenize(bench.SUFFIX)
    ids, i = [], 0
    while len(ids) < n - len(tail):
        ids += api.tokenize(" ".join(ia.filler_sentences(bench.BLOCK_SENTENCES, 7000 + offset + i)))
        i += 1
    return ids[: n - len(tail)] + tail


def label(n: int) -> str:
    return f"{n // 1024}k"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ia.add_api_args(ap)
    ap.add_argument("--sizes", default="8192,32768,131072,229376")
    ap.add_argument("--passes", type=int, default=4, help="passes per size; the first is discarded")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--decode-after", type=int, default=0, help="new tokens of every kept pass (0: --tokens)")
    ap.add_argument("--cache", action="store_true",
                    help="send the requests as a user does (the server's default prefix-cache behaviour: it stores the prompt's prefix) instead of "
                         "`prefix_cache: false`; every prompt is unique (a different stream for every size and pass: prompts of different sizes of one pass would otherwise share a prefix), so none is served from the cache")
    ap.add_argument("--out")
    a = ap.parse_args()
    api = ia.api_from_args(a)
    if a.cache:
        orig = ia.Api.complete

        def complete(self, prompt, max_tokens, **extra):
            extra.pop("prefix_cache", None)
            return orig(self, prompt, max_tokens, **extra)

        ia.Api.complete = complete
    out: dict = {"sizes": {}}
    metrics: dict[str, float] = {}
    for si, n in enumerate(int(x) for x in a.sizes.split(",")):
        rows = []
        for p in range(a.passes):
            ids = ids_for(api, n, 1000 * p + (100_000 * (si + 1) if a.cache else 0))
            mt = a.decode_after if (a.decode_after and p >= 1) else a.tokens
            r = bench.solo(api, ids, mt)
            rate = r["prompt_tokens"] / r["ttft_s"]
            rows.append({"pass": p, "prompt_tokens": r["prompt_tokens"], "prompt_sha256": r["prompt_sha256"], "ttft_s": r["ttft_s"],
                         "tok_s": round(rate, 1), "completion_tokens": r["completion_tokens"], "decode_tok_s": r["decode_tok_s"],
                         "generated_ids": r["generated_ids"]})
            print(f"prefill {label(n)} pass {p}{' (warm-up, discarded)' if p == 0 else ''}: {r['prompt_tokens']} tokens in "
                  f"{r['ttft_s']} s = {rate:.1f} tok/s; decode {r['decode_tok_s']} tok/s ({r['completion_tokens']} tokens)", flush=True)
        kept = rows[1:] if len(rows) > 1 else rows
        med = statistics.median(r["tok_s"] for r in kept)
        medt = statistics.median(r["ttft_s"] for r in kept)
        metrics[f"prefill_{label(n)}_tok_s"] = med
        metrics[f"prefill_{label(n)}_ttft_s"] = medt
        dec = [r["decode_tok_s"] for r in kept if r["decode_tok_s"] and r["completion_tokens"] >= 64]
        if dec:
            metrics[f"decode_after_{label(n)}_tok_s"] = statistics.median(dec)
            print(f"METRIC decode_after_{label(n)}_tok_s={statistics.median(dec)}")
        out["sizes"][str(n)] = {"passes": rows, "median_tok_s": med, "median_ttft_s": medt}
        print(f"METRIC prefill_{label(n)}_tok_s={med}")
        print(f"METRIC prefill_{label(n)}_ttft_s={medt}", flush=True)
    out["metrics"] = metrics
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    if bench.FAILS:
        print("FAIL " + "; ".join(bench.FAILS))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
