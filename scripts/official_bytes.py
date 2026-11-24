#!/usr/bin/env python3
"""Decode bytes per token of the OFFICIAL Inkling-2-Mini checkpoint, for comparison with this pack.

Reads only the safetensors header of `thinkingmachines/Inkling-2-Mini` (HTTP range, no weights) and applies the same
accounting as scripts/roofline.py: attention, router and loop controllers once per layer pass, 8 of 128 experts per
pass (uniform routing), the middle layers (core_start..core_end) read `num_passes` times, the unembed once.

  python3 scripts/official_bytes.py [--rev b85c3cfa615fc413cc9af3379f491b963f3b586a] [--bw-gbs 237]
"""
import argparse
import collections
import json
import math
import re
import struct
import urllib.request

LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")
CORE, PASSES, LAYERS, EXPERTS, TOPK = (6, 18), 3, 24, 128, 8
DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "I32": 4, "I16": 2}


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def header(url):
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    h = json.loads(rng(url, 8, 8 + n - 1))
    h.pop("__metadata__", None)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--rev", default="b85c3cfa615fc413cc9af3379f491b963f3b586a")
    ap.add_argument("--file", default="model-00001-of-00001.safetensors")
    ap.add_argument("--bw-gbs", type=float, default=237.0)
    args = ap.parse_args()
    h = header(f"https://huggingface.co/{args.repo}/resolve/{args.rev}/{args.file}")
    layer = collections.defaultdict(lambda: collections.defaultdict(int))
    glob = collections.defaultdict(int)
    dtypes = collections.Counter()
    for name, t in h.items():
        size = t["data_offsets"][1] - t["data_offsets"][0]
        dtypes[t["dtype"]] += size
        m = LAYER.match(name)
        if m:
            l, rest = int(m.group(1)), m.group(2)
            if ".experts." in rest:
                c = "experts"
            elif rest.startswith("mlp.gate"):
                c = "router"
            elif rest.startswith("controllers."):
                c = "controllers"
            elif rest.startswith(("attn.", "attn_sconv", "mlp_sconv")):
                c = "attention"
            else:
                c = "other"
            layer[l][c] += size
        elif name.startswith("model.llm.unembed"):
            glob["unembed"] += size
        elif name.startswith("model.llm.embed"):
            glob["embed"] += size
        elif name.startswith("model.visual"):
            glob["visual"] += size
        else:
            glob["other"] += size
    mult = [PASSES if CORE[0] <= l < CORE[1] else 1 for l in range(LAYERS)]
    row = {c: sum(mult[l] * layer[l][c] * (TOPK / EXPERTS if c == "experts" else 1) for l in range(LAYERS))
           for c in ("experts", "attention", "router", "controllers", "other")}
    row["unembed"] = glob["unembed"]
    total = sum(row.values())
    print(f"{args.repo}@{args.rev[:8]} {args.file}: {sum(dtypes.values()) / 2**30:.2f} GiB, dtypes (GiB) "
          + ", ".join(f"{k} {v / 2**30:.2f}" for k, v in sorted(dtypes.items())))
    print("decode bytes per row (GB): " + ", ".join(f"{k} {v / 1e9:.2f}" for k, v in row.items()) + f" = {total / 1e9:.2f}")
    print(f"serial ceiling at {args.bw_gbs:g} GB/s: {args.bw_gbs * 1e9 / total:.1f} tok/s")


if __name__ == "__main__":
    main()
