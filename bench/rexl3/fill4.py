#!/usr/bin/env python3
"""The 4-lane contract test (stdlib only): four requests at once, each filling its lane's window.

  fill4.py [--url URL] [--lanes 4] [--max-model-len 262144] [--target-tokens N] [--max-tokens 512]

Each lane gets its own non-repeating filler (bench/tools/needle.py's varied sentences, a distinct seed per lane), its own
pass-code at 50% depth and the question at the end. All lanes are sent together on one Spark: the server holds four full
windows in its KV pool at once (4 x 262,144 = 1,048,576 tokens on the bootstrap profile; the long profile's 1M lanes need
--max-model-len 1048576 and a pool of 4 x 1M, docs/ROOFLINE.md "Capacity"). Reasoning effort level 0, greedy.

--target-tokens defaults to window - max-tokens - 2048. Each lane is sized by the server's /v1/chat/completions/render count
(needle.size_lane), so the prompt is within the window by the server's own count; the request's usage.prompt_tokens must agree
with it within 16 tokens, else the lane fails. PASS (per lane and overall): no error, the reply names its own lane's code and no
other lane's, and prompt_tokens + completion_tokens <= the window. Prints one JSON line per lane, then METRIC fill4_lanes,
fill4_failed, fill4_prompt_tokens_total, fill4_wall_s, fill4_prefill_tok_s. Exit 1 on a failed lane.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import inkling_api as ia  # noqa: E402
import needle  # noqa: E402
import needle_long  # noqa: E402

MARGIN = 2048
COUNT_SLACK = 16


def default_target(window: int, max_tokens: int) -> int:
    return window - max_tokens - MARGIN


def lane_failures(lane: dict, others: list[str], window: int) -> list[str]:
    """Why a finished lane record fails (empty list: it passes)."""
    out = []
    if lane.get("status") in ("refused", "error") or lane.get("error"):
        out.append(f"{lane.get('status')}: {str(lane.get('error'))[:200]}")
    if not lane.get("found"):
        out.append("its own code is not in the reply")
    ans = lane.get("answer") or ""
    if any(c in ans for c in others):
        out.append("the reply holds another lane's code")
    u = lane.get("usage") or {}
    total = int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
    if total > window:
        out.append(f"prompt + completion tokens {total} > window {window}")
    n, rendered = u.get("prompt_tokens"), lane.get("prompt_tokens_rendered")
    if n is not None and rendered is not None and abs(n - rendered) > COUNT_SLACK:
        out.append(f"the render counted {rendered} prompt tokens but the request used {n}: the sizing is off")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=int(ia.recipe_setting("MAX_MODEL_LEN", "262144")),
                    help="the server's --max-model-len (default: MAX_MODEL_LEN of the environment or .env, else 262144)")
    ap.add_argument("--target-tokens", type=int, help="prompt tokens per lane (default: window - max-tokens - 2048)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--system", default="Thinking effort level: 0", help="system message ('' for none)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.lanes < 1:
        raise SystemExit("--lanes must be >= 1")
    target = a.target_tokens or default_target(a.max_model_len, a.max_tokens)
    api = ia.api_from_args(a)
    counter = needle.Counter(api, None)
    per_sent = counter.count(" ".join(ia.filler_sentences(2000, a.seed + 99))) / 2000
    na = needle_long.lane_args(target, a.max_model_len, a.max_tokens, a.system, a.seed)
    lanes = []
    for i in range(a.lanes):
        lane = {"lane": i, "depth": 0.5, "target_tokens": target}
        lane.update(needle.size_lane(api, counter, na, i, 0.5, per_sent))
        lanes.append(lane)
        print(f"# lane {i}: {lane['sentences']} sentences, {lane.get('prompt_tokens_rendered') or '~%d' % lane['prompt_tokens_est']} "
              f"prompt tokens (target {target}, window {a.max_model_len})", file=sys.stderr, flush=True)
    codes = [l["code"] for l in lanes]
    start = threading.Barrier(a.lanes) if a.lanes > 1 else None
    t0 = time.monotonic()
    threads = [threading.Thread(target=needle.ask, args=(api, na, lane, start)) for lane in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    bad, total = 0, 0
    for lane in lanes:
        why = lane_failures(lane, [c for c in codes if c != lane["code"]], a.max_model_len)
        total += int((lane.get("usage") or {}).get("prompt_tokens") or 0)
        bad += bool(why)
        print(json.dumps({k: lane.get(k) for k in ("lane", "code", "status", "found", "prompt_tokens", "prefill_s", "total_s",
                                                    "finish_reason", "usage")} | {"failures": why}))
        print(f"{'FAIL' if why else 'PASS'} fill4_lane{lane['lane']}" + (": " + "; ".join(why) if why else ""))
    print(f"METRIC fill4_lanes={a.lanes}")
    print(f"METRIC fill4_failed={bad}")
    print(f"METRIC fill4_prompt_tokens_total={total}")
    print(f"METRIC fill4_wall_s={wall:.1f}")
    print(f"METRIC fill4_prefill_tok_s={total / wall:.1f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
