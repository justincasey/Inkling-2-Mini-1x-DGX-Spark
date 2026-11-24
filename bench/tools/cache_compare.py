#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replies of a cached boot against a reference boot, byte for byte (stdlib only): the cache, the vision tower and the decode graphs on
(boot A) against the prefix cache off and the graphs off (boot B).

  cache_compare.py --images A.json B.json       two `cache_images.py` runs (B with --reference-only): every request of A, cached or not,
                                                 equals B's uncached reply of the same prompt; A's own uncached replies equal B's too
  cache_compare.py --gate A.json B.json         two `gates.py cache` runs with the same --nonce and --big-tokens: the generated ids of every
                                                 uncached, cold, warm and shared-prefix reply of A equal B's uncached reply of that prompt

Prints one line per comparison and `identical` or `DIFFERENT (n)`; exit 1 when anything differs.
"""

import argparse
import json
import sys


def images(a: dict, b: dict) -> list[str]:
    bad, n = [], 0
    for label, ref in b["ids"]["refs"].items():
        for what, side in (("uncached", a["ids"]["refs"]), ("cached", a["ids"]["got"])):
            n += 1
            ok = side.get(label) == ref
            print(f"{label} {what}: {'identical' if ok else 'DIFFERS'} ({len(ref)} ids)")
            if not ok:
                bad.append(f"{label} {what}")
    print(f"{n} replies compared")
    return bad


def gate(a: dict, b: dict) -> list[str]:
    bad, n = [], 0
    ref = {p["name"]: p["uncached"]["generated_ids"] for p in b["prompts"]}
    for s in b["shared"]:
        for i, r in enumerate(s["uncached"]):
            ref[f"{s['name']}/{i}"] = r["generated_ids"]
    for p in a["prompts"]:
        for what in ("uncached", "cold", "warm"):
            n += 1
            ok = p[what]["generated_ids"] == ref[p["name"]]
            print(f"{p['name']} {what}: {'identical' if ok else 'DIFFERS'} ({len(ref[p['name']])} ids, cached tokens {p[what]['cached_tokens']})")
            if not ok:
                bad.append(f"{p['name']} {what}")
    for s in a["shared"]:
        for what in ("uncached", "cached"):
            for i, r in enumerate(s[what]):
                n += 1
                key = f"{s['name']}/{i}"
                ok = r["generated_ids"] == ref[key]
                print(f"{key} {what}: {'identical' if ok else 'DIFFERS'} ({len(ref[key])} ids, cached tokens {r['cached_tokens']})")
                if not ok:
                    bad.append(f"{key} {what}")
    print(f"{n} replies compared")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", action="store_true")
    g.add_argument("--gate", action="store_true")
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args(argv)
    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    bad = images(a, b) if args.images else gate(a, b)
    print("identical" if not bad else f"DIFFERENT ({len(bad)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
