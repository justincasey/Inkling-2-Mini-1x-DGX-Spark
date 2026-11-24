"""The last op alone: the oracle's logits from the ENGINE's own final-norm output (`h_final` of a dump) against the engine's `logits`
plane, row by row: the EXL3 unembed with the fp16 rotated input (`Emulation.dense_fp16`), the soft cap and the padding mask on the engine's
input only. Also the exact-numerics unembed (no fp16 rotation), to see how much of the gap the rotation explains.

    python unembed_probe.py --pack DIR --dump D.safetensors [--dump ...] --out REPORT.json [--rows 5,23]

Per row: relative L2 and max abs of each oracle against the engine's logits, the first-max argmax of the three, the top-1 minus top-2
margin of each, and the engine's logit gap at the fp16 oracle's top-2 tokens against the oracle's.
"""

import argparse
import json

import numpy as np
import torch

import st
from inkling2_ref import F64, np_f16
from pack import PackIndex, PackWeights

VALID = 200058
CAP = 30.0


def softcap(z: np.ndarray) -> np.ndarray:
    return (CAP * np.tanh(z.astype(np.float32).astype(np.float64) / CAP)).astype(np.float32).astype(np.float64)


def top2(x: np.ndarray):
    i = np.argsort(-x, axis=1, kind="stable")[:, :2]
    return i, np.take_along_axis(x, i, 1)


def probe(lin, h: np.ndarray, e: np.ndarray, valid: int = VALID) -> dict:
    out = {}
    z_fp16 = lin(h, in_round=np_f16)[:, :valid]
    z_exact = lin(h)[:, :valid]
    o16, ox = softcap(z_fp16), softcap(z_exact)
    ei, ev = top2(e)
    rows = []
    for name, o in (("fp16", o16), ("exact", ox)):
        d = o - e
        rel = np.linalg.norm(d, axis=1) / np.linalg.norm(e, axis=1)
        oi, ov = top2(o)
        out[name] = {"rel": rel, "max_abs": np.abs(d).max(1), "argmax_equal": oi[:, 0] == ei[:, 0], "margin": ov[:, 0] - ov[:, 1],
                     "gap_err": (e[np.arange(len(e)), oi[:, 0]] - e[np.arange(len(e)), oi[:, 1]]) - (ov[:, 0] - ov[:, 1])}
    d = o16 - ox
    out["rotation"] = {"rel": np.linalg.norm(d, axis=1) / np.linalg.norm(ox, axis=1), "max_abs": np.abs(d).max(1)}
    for r in range(len(e)):
        rows.append({"row": r, "engine_margin": float(ev[r, 0] - ev[r, 1]), "engine_top1": int(ei[r, 0]),
                     **{f"{k}_{m}": (bool(v[m][r]) if v[m].dtype == bool else float(v[m][r])) for k, v in out.items() for m in v}})
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--dump", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    w = PackWeights(PackIndex(a.pack))
    lin = w.exl3_linear("model.llm.unembed.weight")
    res = {}
    for path in a.dump:
        hd = st.read_header(path)
        h = st.read_tensor(hd["h_final"]).to(F64).numpy()
        e = st.read_tensor(hd["logits"]).to(F64).numpy()
        res[path] = probe(lin, h, e)
        print(path, h.shape, flush=True)
    json.dump(res, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
