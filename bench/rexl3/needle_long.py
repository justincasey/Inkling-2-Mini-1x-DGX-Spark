#!/usr/bin/env python3
"""One needle near the end of the window (stdlib only): the node's chat API over varied filler up to ~1M tokens, one
pass-code sentence at --depth, reasoning effort level 0, greedy. Built on bench/tools/needle.py (its filler, its prompt,
its sizing by /v1/chat/completions/render and its streamed ask); this tool only picks the long sizes and the verdict.

  needle_long.py [--url URL] [--tokens 1000000] [--depths 0.5] [--max-model-len 1048576] [--max-tokens 512] [--out FILE]

The window is the server's --max-model-len of the long profile (1,048,576; default: MAX_MODEL_LEN of the environment or .env,
else 1,048,576); the bootstrap profile's 262,144 admits at most --tokens ~ 258,000. A body of 1M tokens is about 4 MB of JSON: the
server's body and rendered-prompt limits may need raising (EXTRA_FLAGS); a refusal is reported with the server's message and fails
the run. With several --depths the requests run one after another (each a cold prefill of the whole prompt).

Per depth: `PASS needle_long_d<depth>` or `FAIL ...` with the reason (refused / error / empty / not found / found only in the
reasoning / finish_reason length). Prints METRIC needle_long_pass (fraction of depths that found the code), needle_long_tokens
(the largest prompt, usage.prompt_tokens), needle_long_tok_s (prompt tokens / seconds to the first streamed token, the slowest
depth). Exit 1 on a miss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import inkling_api as ia  # noqa: E402
import needle  # noqa: E402

DEFAULT_WINDOW = 1_048_576


def lane_args(tokens: int, max_model_len: int, max_tokens: int, system: str, seed: int) -> argparse.Namespace:
    """The namespace needle.size_lane / needle.ask read."""
    return argparse.Namespace(tokens=tokens, max_model_len=max_model_len, max_tokens=max_tokens, system=system, seed=seed)


def verdict(lane: dict) -> tuple[bool, str]:
    """(passed, reason) of an answered lane record (needle.ask's fields)."""
    st = lane.get("status")
    if st == "refused":
        return False, f"refused (HTTP {lane.get('http_status')}): {str(lane.get('error'))[:200]}"
    if st == "error":
        return False, f"error: {str(lane.get('error'))[:200]}"
    if lane.get("found"):
        return True, "found"
    if st == "empty":
        return False, f"empty answer (finish_reason {lane.get('finish_reason')})"
    if lane.get("found_in_reasoning"):
        return False, f"code only in the reasoning (finish_reason {lane.get('finish_reason')})"
    return False, f"not found (finish_reason {lane.get('finish_reason')}): {str(lane.get('answer'))[:60]!r}"


def summarize(lanes: list[dict]) -> dict:
    ok = [verdict(l)[0] for l in lanes]
    toks = [l.get("prompt_tokens") or 0 for l in lanes]
    rates = [t / l["prefill_s"] for t, l in zip(toks, lanes) if t and l.get("prefill_s")]
    return {"pass": sum(ok) / max(len(ok), 1), "tokens": max(toks, default=0), "tok_s": min(rates) if rates else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--tokens", type=int, default=1_000_000, help="prompt tokens (default: %(default)s)")
    ap.add_argument("--depths", default="0.5", help="comma list of needle depths 0..1")
    ap.add_argument("--max-model-len", type=int, default=int(ia.recipe_setting("MAX_MODEL_LEN", str(DEFAULT_WINDOW))))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--system", default="Thinking effort level: 0", help="system message ('' for none)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="write the JSON report here")
    a = ap.parse_args(argv)
    depths = [float(d) for d in a.depths.split(",") if d.strip()]
    api = ia.api_from_args(a)
    counter = needle.Counter(api, None)
    per_sent = counter.count(" ".join(ia.filler_sentences(2000, a.seed + 99))) / 2000
    na = lane_args(a.tokens, a.max_model_len, a.max_tokens, a.system, a.seed)
    lanes = []
    for k, d in enumerate(depths):
        lane = {"lane": k, "depth": round(d, 4), "target_tokens": a.tokens}
        lane.update(needle.size_lane(api, counter, na, k, d, per_sent))
        t0 = time.monotonic()
        needle.ask(api, na, lane, None)
        lane["wall_s"] = round(time.monotonic() - t0, 3)
        ok, why = verdict(lane)
        print(f"{'PASS' if ok else 'FAIL'} needle_long_d{d:.2f}: {lane.get('prompt_tokens')} tokens, prefill_s {lane.get('prefill_s')}, "
              f"total_s {lane.get('total_s')}, {why}", flush=True)
        lanes.append(lane)
    s = summarize(lanes)
    print(f"METRIC needle_long_pass={s['pass']:.4f}")
    print(f"METRIC needle_long_tokens={s['tokens']}")
    if s["tok_s"] is not None:
        print(f"METRIC needle_long_tok_s={s['tok_s']:.1f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"url": api.url, "model": api.model, "max_model_len": a.max_model_len, "lanes": lanes, **s}, f, indent=1)
            f.write("\n")
    return 0 if s["pass"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
