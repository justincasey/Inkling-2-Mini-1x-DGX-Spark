"""Whole-forward deviation of fp8 runs (`fp8_emu.py forward`) against the oracle fixtures (gate `fp8-numerics`).

    python fp8_summary.py --spec DIR --engine DIR --runs DIR --prompts NAME... --out report.json

Per prompt (same token rows on every side): the hidden plane per visit 0..48 (P entering each pass, then after the last): the relative L2
of the whole plane, ||X - R|| / ||R||, for
  floor   engine fixture (the A lane's emulation) against the exact fixture: what the A lane already deviates by,
  added   a run against the engine fixture: what the variant adds on top of the A lane,
  total   a run against the exact fixture;
and the logits rows: argmax and top-2 agreement with the exact fixture, mean and 99th-percentile KL(exact || X) in nats, relative L2 of the plane.
`A` is the engine emulation run through the same code as the fp8 variants (it must equal the engine fixture bit for bit).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import st

F64 = torch.float64


def load(path: str) -> dict[str, torch.Tensor]:
    t = st.load_all(path)
    return {k: t[k] for k in ("hidden", "logits", "tokens")}


def rel(x: torch.Tensor, r: torch.Tensor) -> float:
    x, r = x.to(F64), r.to(F64)
    return float(((x - r).norm() / r.norm()))


def hidden_curve(x: torch.Tensor, r: torch.Tensor) -> list[float]:
    return [rel(x[v], r[v]) for v in range(x.shape[0])]


def kl_rows(ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    lp, lq = torch.log_softmax(ref.to(F64), -1), torch.log_softmax(x.to(F64), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def logits_stats(x: torch.Tensor, ref: torch.Tensor) -> dict:
    top = lambda t: torch.topk(t, 2, dim=-1).indices  # noqa: E731
    kl = kl_rows(ref, x)
    return {
        "rows": int(ref.shape[0]), "argmax_equal": int((x.argmax(-1) == ref.argmax(-1)).sum()),
        "top2_equal": int((top(x) == top(ref)).all(-1).sum()), "rel_l2": rel(x, ref),
        "kl_mean": float(kl.mean()), "kl_p99": float(torch.quantile(kl, 0.99)), "kl_max": float(kl.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["A", "f8_row", "f8_blk128", "f8_row_tile", "f8_wt", "f8_act_row"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report: dict = {}
    for p in a.prompts:
        spec, eng = load(os.path.join(a.spec, p + ".safetensors")), load(os.path.join(a.engine, p + ".safetensors"))
        rep = {"floor_hidden": hidden_curve(eng["hidden"], spec["hidden"]), "floor_logits": logits_stats(eng["logits"], spec["logits"]),
               "variants": {}}
        for v in a.variants:
            path = os.path.join(a.runs, f"{v}-{p}.safetensors")
            if not os.path.exists(path):
                continue
            run = load(path)
            assert torch.equal(run["tokens"], spec["tokens"])
            rep["variants"][v] = {
                "equals_engine_fixture_bitwise": bool(torch.equal(run["hidden"], eng["hidden"]) and torch.equal(run["logits"], eng["logits"])),
                "added_hidden": hidden_curve(run["hidden"], eng["hidden"]), "total_hidden": hidden_curve(run["hidden"], spec["hidden"]),
                "logits_vs_exact": logits_stats(run["logits"], spec["logits"]),
                "logits_vs_engine": logits_stats(run["logits"], eng["logits"]),
            }
        report[p] = rep
        print(p, "floor logits", rep["floor_logits"], flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
