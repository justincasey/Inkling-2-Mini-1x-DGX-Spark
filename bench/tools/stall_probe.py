#!/usr/bin/env python3
"""The stall a long prefill puts on a decoding stream (stdlib only; one server).

A short prompt starts a streamed completion of `--decode-tokens` tokens (greedy). Once `--after` events have arrived, a cold prompt of `--prompt-tokens` ids
(its own filler stream, `prefix_cache: false`) is sent as a second request. The gaps between the first stream's events are recorded: before the long prompt
arrives (the steady gap), and from the moment it was sent until it has been prefilled. `max_gap_s` is the longest the stream stood still: the server runs a chunk
of the long prompt between two windows of the stream, so it is about one chunk's duration. Prints `METRIC` lines and writes `--out` JSON.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_api as ia  # noqa: E402
import prefill_sweep as ps  # noqa: E402


def stream(url: str, body: dict, events: list, t_end: list) -> None:
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            events.append(time.monotonic())
    t_end.append(time.monotonic())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ia.add_api_args(ap)
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--decode-tokens", type=int, default=600)
    ap.add_argument("--after", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()
    api = ia.api_from_args(a)
    url = a.url or ia.default_url()
    model = a.model or api.model
    short = api.tokenize("The following is the first chapter of a long literary novel about a lighthouse keeper named Marta.")
    long_ids = ps.ids_for(api, a.prompt_tokens, 7_000_000)
    ev: list = []
    t_end: list = []
    th = threading.Thread(target=stream, args=(url, {"model": model, "prompt": short, "max_tokens": a.decode_tokens, "temperature": 0, "stream": True,
                                                      "prefix_cache": False}, ev, t_end))
    th.start()
    while len(ev) < a.after:
        time.sleep(0.01)
    time.sleep(0.5)
    t_send = time.monotonic()
    t0 = time.monotonic()
    resp, wall = api.complete(long_ids, 4, prefix_cache=False)
    t_done = time.monotonic()
    th.join()
    gaps = [(ev[i], ev[i] - ev[i - 1]) for i in range(1, len(ev))]
    steady = [g for t, g in gaps if t < t_send]
    during = [g for t, g in gaps if t_send <= t <= t_done + 1.0]
    out = {"prompt_tokens": a.prompt_tokens, "long_wall_s": round(wall, 3), "events": len(ev), "steady_gap_median_s": round(statistics.median(steady), 5),
           "steady_gap_max_s": round(max(steady), 5), "max_gap_s": round(max(during), 4) if during else None,
           "gaps_over_0p1s": sorted(round(g, 3) for g in during if g > 0.1)}
    print(f"METRIC stall_steady_gap_median_s={out['steady_gap_median_s']}")
    print(f"METRIC stall_max_gap_s={out['max_gap_s']}")
    print(f"METRIC stall_long_prefill_wall_s={out['long_wall_s']}")
    print("gaps over 0.1 s while the long prompt ran:", out["gaps_over_0p1s"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
