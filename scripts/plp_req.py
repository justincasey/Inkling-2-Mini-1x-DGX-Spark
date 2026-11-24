#!/usr/bin/env python3
"""plp_req.py N [--url URL]: one cold N-id prefill, 1 token (the engine's dump writes every chunk's logits)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "tools"))
import bench  # noqa: E402
import inkling_api as ia  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("n", type=int, help="prompt tokens")
ap.add_argument("--url", help="the server's API (default: NODES and PORT, or INKLING2MINI_URL)")
a = ap.parse_args()
api = ia.Api(a.url or ia.default_url())
ids = bench.long_ids(api, a.n)
r, wall = api.complete(ids, 1, prefix_cache=False)
print(f"{r['usage']['prompt_tokens']} prompt tokens in {wall:.1f} s", flush=True)
