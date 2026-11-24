#!/usr/bin/env python3
"""Long-context recall of the node's chat API, PASS/FAIL per context length up to the profile's window (stdlib only; the
`tokenizers` package is used if --tokenizer is given and installed).

The filler-and-needle approach of tools/needle.py (it supplies the sizing, the prompt builder and the streamed request):
varied deterministic filler sentences with one pass-code sentence at a given depth, sized to the target prompt tokens
(/tokenize calibration, the exact size from /v1/chat/completions/render), asked greedily (temperature 0) for the code. Each
length and depth has its own seed, so no prompt shares a prefix beyond the intro line and nothing is served from the
prefix cache (cached_tokens is recorded to prove it).

Lengths: the powers of two from 8K that are below the window, then the window itself (sized to the window minus the answer
budget). --window 262144 is the bootstrap profile, 1048576 the long and max profiles. A length above what /v1/models reports
as max_model_len (the KV pool can hold less than the configured window) FAILs without being sent. A length PASSes when the
needle is in the reply's `content` at every depth; a code found only in the reasoning, or a reply cut by `length`, is a FAIL
(the base model can reason in long repetitive loops) and says so. Reasoning stays low: the system line is
`Thinking effort level: 0`.

  longctx.py --out bench/runs/longctx.json                          # window 262144, depth 0.5
  longctx.py --window 1048576 --depths 0.25,0.75 --out bench/runs/longctx-1m.json
  longctx.py --lengths 8k,64k,window --stop-on-fail

One JSON result (--out and stdout's last block): per length the lanes (prompt_tokens, prefill_s, total_s, found, finish,
reasoning chars, usage incl. cached_tokens and spec), `pass`, and the overall `pass`. Lines: `PASS longctx LENGTH: ...`,
`FAIL longctx LENGTH: why`, `METRIC longctx_*`; exit 1 when any length fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import inkling_api as ia  # noqa: E402
import needle  # noqa: E402

STEPS = [8192, 32768, 65536, 131072, 262144, 524288, 1048576]


def parse_size(text: str) -> int:
    t = text.strip().lower()
    if t.endswith("k"):
        return int(float(t[:-1]) * 1024)
    if t.endswith("m"):
        return int(float(t[:-1]) * 1024 * 1024)
    return int(t)


def label(n: int) -> str:
    if n % 1048576 == 0:
        return f"{n // 1048576}m"
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def plan_lengths(window: int, spec: str = "auto") -> list[tuple[str, int]]:
    """[(label, nominal tokens)]: `auto` is every power-of-two step of STEPS below the window plus the window itself
    (labelled `window`); an explicit list takes sizes (8k, 100000, 1m) and the word `window`."""
    if spec == "auto":
        return [(label(n), n) for n in STEPS if n < window] + [("window", window)]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(("window", window) if part == "window" else (label(parse_size(part)), parse_size(part)))
    return out


def parse_depths(text: str) -> list[float]:
    depths = [float(x) for x in text.split(",") if x.strip()]
    if not depths or any(not 0.0 <= d <= 1.0 for d in depths):
        raise SystemExit("--depths takes numbers between 0 and 1")
    return depths


def judge_lane(lane: dict) -> tuple[bool, str]:
    """Whether one lane found its needle in the content, and why not."""
    st = lane.get("status")
    if st == "refused":
        return False, f"refused: HTTP {lane.get('http_status')} {str(lane.get('error'))[:200]}"
    if st == "error":
        return False, f"request error: {lane.get('error')}"
    if st == "empty" and lane.get("error"):
        return False, f"stream error: {str(lane['error'])[:200]}"
    if lane.get("found"):
        return True, ""
    if lane.get("finish_reason") == "length":
        return False, ("cut by max_tokens (finish length) before a code: "
                       + ("the code is in the reasoning" if lane.get("found_in_reasoning") else "the reasoning ran on"))
    if lane.get("found_in_reasoning"):
        return False, "the code is in the reasoning but not in the answer"
    return False, f"wrong or empty answer {str(lane.get('answer'))[:80]!r}"


def length_verdict(lanes: list[dict]) -> tuple[bool, str]:
    reasons = []
    for lane in lanes:
        ok, why = judge_lane(lane)
        if not ok:
            reasons.append(f"depth {lane['depth']}: {why}")
    return not reasons, "; ".join(reasons)


def run_length(api: ia.Api, counter, per_sent: float, tag: str, nominal: int, index: int, a, max_len: int) -> dict:
    """One length: a lane per depth through needle.size_lane / needle.ask; the record, with `pass` and `why`."""
    res = {"length": tag, "nominal_tokens": nominal, "lanes": []}
    if nominal > max_len:
        res.update(**{"pass": False, "why": f"the server's max_model_len is {max_len}, below this length (not sent)"})
        return res
    ns = SimpleNamespace(tokens=nominal, max_tokens=a.max_tokens, max_model_len=min(a.window, max_len), system=a.system,
                         seed=a.seed + index)
    for j, depth in enumerate(a.depth_list):
        lane = {"lane": index * 100 + j, "depth": depth, "target_tokens": nominal}
        try:
            lane.update(needle.size_lane(api, counter, ns, lane["lane"], depth, per_sent))
        except SystemExit as e:
            lane.update(status="error", error=str(e))
        else:
            needle.ask(api, ns, lane, None)
        lane.pop("user", None)
        res["lanes"].append(lane)
        print(f"  {tag} depth {depth}: {lane.get('status')} found={lane.get('found')} prompt_tokens={lane.get('prompt_tokens')} "
              f"prefill_s={lane.get('prefill_s')} total_s={lane.get('total_s')}", file=sys.stderr, flush=True)
        if not judge_lane(lane)[0] and a.stop_on_fail:
            break
    ok, why = length_verdict(res["lanes"])
    res.update(**{"pass": ok, "why": why})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ia.add_api_args(ap)
    ap.add_argument("--window", type=int, default=262144,
                    help="the profile's context window: 262144 bootstrap, 1048576 long/max (default: %(default)s)")
    ap.add_argument("--lengths", default="auto", help="auto, or a list such as 8k,64k,window (default: %(default)s)")
    ap.add_argument("--depths", default="0.5", help="needle depths 0..1, comma separated (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=512, help="the answer budget (default: %(default)s)")
    ap.add_argument("--system", default="Thinking effort level: 0", help="system message ('' for none)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokenizer", help="the pack's tokenizer.json (needs the tokenizers package); default /tokenize")
    ap.add_argument("--stop-on-fail", action="store_true", help="stop at the first failing length")
    ap.add_argument("--out", help="write the JSON result here (also printed)")
    a = ap.parse_args(argv)
    a.depth_list = parse_depths(a.depths)
    api = ia.api_from_args(a)
    status, listing, _ = api.call("GET", "/v1/models", timeout=30)
    row = ((listing or {}).get("data") or [{}])[0] if status == 200 and isinstance(listing, dict) else {}
    max_len = row.get("max_model_len")
    if not isinstance(max_len, int) or max_len <= 0:
        raise SystemExit(f"/v1/models gave no max_model_len (HTTP {status})")
    plan = plan_lengths(a.window, a.lengths)
    counter = needle.Counter(api, a.tokenizer)
    per_sent = counter.count(" ".join(ia.filler_sentences(2000, a.seed + 99))) / 2000
    print(f"model {api.model}, window {a.window}, server max_model_len {max_len}, lengths {[t for t, _ in plan]}, "
          f"depths {a.depth_list}", file=sys.stderr)
    t0 = time.monotonic()
    results = []
    for i, (tag, nominal) in enumerate(plan):
        r = run_length(api, counter, per_sent, tag, nominal, i, a, max_len)
        results.append(r)
        lanes = r["lanes"]
        if r["pass"]:
            toks = [l.get("prompt_tokens") for l in lanes]
            pre = [l.get("prefill_s") for l in lanes]
            print(f"PASS longctx {tag}: found at {len(lanes)}/{len(lanes)} depths, prompt_tokens {toks}, prefill_s {pre}")
        else:
            print(f"FAIL longctx {tag}: {r['why']}")
        for l in lanes:
            pt, ps = l.get("prompt_tokens"), l.get("prefill_s")
            if pt and ps:
                print(f"METRIC longctx_{tag}_d{l['depth']}_prompt_tokens={pt}")
                print(f"METRIC longctx_{tag}_d{l['depth']}_prefill_s={ps}")
                print(f"METRIC longctx_{tag}_d{l['depth']}_prefill_tok_s={round(pt / ps, 1)}")
        if not r["pass"] and a.stop_on_fail:
            break
    ok = bool(results) and all(r["pass"] for r in results)
    rep = {"url": api.url, "model": api.model, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "window": a.window, "server_max_model_len": max_len, "depths": a.depth_list, "max_tokens": a.max_tokens,
           "tokens_per_sentence": round(per_sent, 4), "wall_s": round(time.monotonic() - t0, 3), "lengths": results,
           "pass": ok}
    txt = json.dumps(rep, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    print(f"{'PASS' if ok else 'FAIL'} longctx: {sum(1 for r in results if r['pass'])}/{len(results)} lengths up to the {a.window} window")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
