#!/usr/bin/env python3
"""Image requests, drafted vs serial on one server: ids equal, tok/s median of 3 after a discarded pass (stdlib; uses the recipe's tools)."""
import argparse, json, statistics, sys
sys.path.insert(0, __import__("os").path.expanduser("~/projects/inkling2-mini-k-draftwire/bench/tools"))
import inkling_api as ia, synth_images as si

ap = argparse.ArgumentParser()
ia.add_api_args(ap)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()
api = ia.api_from_args(a)
Q = "Describe this picture in detail: list every colour and every shape you can see, and where each one is."
out = {"max_tokens": a.max_tokens, "images": {}}
for name in list(si.IMAGES):
    if name not in si.IMAGES:
        continue
    parts = [{"type": "image_url", "image_url": {"url": si.data_url(si.image_png(name))}}, {"type": "text", "text": Q}]

    def run(draft):
        resp, wall = api.chat([{"role": "user", "content": parts}], a.max_tokens, reasoning_effort="none", prefix_cache=False, draft=draft)
        ch = resp["choices"][0]
        u = resp.get("usage") or {}
        return [int(t) for t in ch["token_ids"]], ch.get("finish_reason"), int(u["completion_tokens"]), wall, ia.spec_counters(u)

    rec = {"serial": [], "drafted": [], "equal": True}
    for p in range(4):
        s = run(False)
        d = run(True)
        if (s[0], s[1]) != (d[0], d[1]):
            rec["equal"] = False
        if p:
            rec["serial"].append(s[2] / s[3])
            rec["drafted"].append(d[2] / d[3])
            rec["spec"] = d[4]
            rec["tokens"] = d[2]
    rec["serial_median"] = round(statistics.median(rec["serial"]), 2)
    rec["drafted_median"] = round(statistics.median(rec["drafted"]), 2)
    rec["speedup"] = round(rec["drafted_median"] / rec["serial_median"], 3)
    out["images"][name] = rec
    print(name, rec["serial_median"], rec["drafted_median"], rec["speedup"], "equal", rec["equal"], rec.get("spec"), file=sys.stderr, flush=True)
json.dump(out, open(a.out, "w"), indent=1)
sys.exit(0 if all(r["equal"] for r in out["images"].values()) else 1)
