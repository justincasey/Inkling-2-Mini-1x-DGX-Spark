"""Where the engine's argmax flips come from, per row: an engine dump (`h_final`, `logits`) against the fixtures of the same prompt made by
the oracle with the engine's numerics (`--engine-fixture`) and, as the oracle's own floor, with the exact numerics (`--spec-fixture`).

    python row_gap.py --name NAME --dump D --engine-fixture F --spec-fixture F [--json OUT]

Per row (the dump and the fixtures hold the same rows): the relative L2 of the engine's logits and of its `h_final` against the engine
fixture, the floor (spec against engine fixture), the engine fixture's top-1 minus top-2 margin, the change of that gap in the engine's
logits (`gap_err`, signed: positive = the engine's gap is wider), and whether the argmax differs. Summary: the rms of `gap_err` over the
rows whose margin is above 1 (no selection on the flips), and the expected number of flips for Gaussian gap errors of that rms at the
rows' margins, against the observed.
"""

import argparse
import json
import math

import numpy as np

import st

VALID = 200058


def load(path: str, name: str) -> np.ndarray:
    return st.read_tensor(st.read_header(path)[name]).to(__import__("torch").float64).numpy()


def top2(x: np.ndarray):
    i = np.argsort(-x, axis=1, kind="stable")[:, :2]
    return i, np.take_along_axis(x, i, 1)


def rel(a, b):
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def analyse(dump: str, efix: str, sfix: str) -> dict:
    e, h = load(dump, "logits"), load(dump, "h_final")
    o, oh = load(efix, "logits"), load(efix, "h_final")
    s = load(sfix, "logits")
    n = e.shape[0]
    assert o.shape == e.shape == s.shape, (e.shape, o.shape, s.shape)
    oi, ov = top2(o)
    ei = np.argmax(e, 1)
    r = np.arange(n)
    gap = ov[:, 0] - ov[:, 1]
    gap_err = e[r, oi[:, 0]] - e[r, oi[:, 1]] - gap
    sgap_err = s[r, oi[:, 0]] - s[r, oi[:, 1]] - gap
    return {"rows": n, "logits_rel": rel(e, o), "floor_rel": rel(s, o), "h_rel": rel(h, oh),
            "oracle_margin": gap, "gap_err": gap_err, "floor_gap_err": sgap_err,
            "argmax_differs": ei != oi[:, 0], "floor_argmax_differs": np.argmax(s, 1) != oi[:, 0]}


def expected_flips(a: dict) -> tuple[float, float]:
    ok = a["oracle_margin"] > 1.0
    sigma = float(np.sqrt(np.mean(a["gap_err"][ok] ** 2)))
    # P(the gap error closes the margin) = P(N(0, sigma) < -margin) for the engine's side
    p = [0.5 * math.erfc(m / (sigma * math.sqrt(2))) for m in a["oracle_margin"]]
    return sigma, float(sum(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--engine-fixture", required=True)
    ap.add_argument("--spec-fixture", required=True)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    r = analyse(a.dump, a.engine_fixture, a.spec_fixture)
    sigma, exp = expected_flips(r)
    print(f"{a.name}: rows {r['rows']}  gap_err rms (margin > 1) {sigma:.4f}  expected flips {exp:.2f}  observed {int(r['argmax_differs'].sum())}"
          f"  floor flips {int(r['floor_argmax_differs'].sum())}")
    if a.json:
        out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
        out.update(sigma=sigma, expected_flips=exp)
        json.dump(out, open(a.json, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
