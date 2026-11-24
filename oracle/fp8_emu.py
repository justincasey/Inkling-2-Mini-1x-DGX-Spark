"""fp8 (e4m3) prefill GEMM numerics of the Mini's B lane, emulated on real data, CPU only (gate `fp8-numerics`).

The proposed B lane quantizes the two operands of every prefill GEMM (the MoE experts' gate/up/down and the dense EXL3
projections) to e4m3 and accumulates in fp32. On the EXL3 layer `y = ((((x * suh) H_K) W_q) H_N) * svh` the rotated input
`xh = (x * suh) H_K` is the A operand and the decoded codebook values `W_q` (fp16, exact) the B operand; the product
`xh W_q` is the GEMM. Variants (act, weight):

  exact         none, none         the exact float64 policy (reference (i))
  A             fp16, none         the A lane: fp16-rounded rotated input, float64 stands for fp32 accumulation (ii)
  f8_row        row, direct        e4m3 activations, one fp32 scale a row; e4m3 W_q, no scale
  f8_blk128     blk128, direct     e4m3 activations, one fp32 scale per (row, 128 columns of K)
  f8_row_tile   row, tile          as f8_row; W_q with one fp32 scale per 128 x 128 tile (amax / 448)
  f8_act_row    row, none          activations only (attribution)
  f8_wt         none, direct       weights only (attribution)

Subcommands (run from this directory, `nice -n 19`):

  layers   a real forward of a fixture prompt in the A lane (Emulation.engine()); on every dense EXL3 call and on the MoE of
           the chosen visits all variants are evaluated on the REAL inputs and compared with the exact one: relative L2,
           per-row distribution, max |e| over the row rms, per kind (attention q r gate k v o; MoE gate, up, down given the
           exact activation, whole expert, combined layer output).
  forward  the whole oracle forward over a fixture's tokens with one variant on every layer GEMM (the unembed, attention
           core, glue stay the A lane: `Emulation.engine()`), written as a fixture for `engine_gap.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

import exl3
import pack
import st
from exl3 import fwht_128
from inkling2_ref import Emulation, Inkling2LoopConfig, Inkling2LoopRef, Rounding, np_f16, sigmoid_np, softcap_np
from pack import PROJS, PackExperts, PackIndex, PackWeights, expert_prefix, load_exl3

F8MAX = 448.0
VARIANTS = {
    "exact": ("none", "none"),
    "A": ("fp16", "none"),
    "f8_row": ("row", "direct"),
    "f8_blk128": ("blk128", "direct"),
    "f8_row_tile": ("row", "tile"),
    "f8_act_row": ("row", "none"),
    "f8_wt": ("none", "direct"),
}
F64 = np.float64


def e4m3(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even to e4m3 with saturation (cvt.rn.satfinite.e4m3x2); float64 in and out."""
    t = torch.from_numpy(np.clip(np.asarray(x, dtype=np.float32), -F8MAX, F8MAX))
    return t.to(torch.float8_e4m3fn).to(torch.float32).numpy().astype(F64)


def _scale(amax: np.ndarray) -> np.ndarray:
    s = (amax / F8MAX).astype(np.float32).astype(F64)  # an fp32 scale
    return np.where(s == 0.0, 1.0, s)


def act_quant(xh: np.ndarray, mode: str) -> np.ndarray:
    """The rotated input as the A operand of the mma, dequantized back to float64 (the product with the scales folded)."""
    if mode == "none":
        return xh
    if mode == "fp16":
        return np_f16(xh)
    if mode == "row":
        s = _scale(np.abs(xh).max(axis=1, keepdims=True))
        return e4m3(xh / s) * s
    if mode.startswith("blk"):
        b = int(mode[3:])
        r, k = xh.shape
        xb = xh.reshape(r, k // b, b)
        s = _scale(np.abs(xb).max(axis=2, keepdims=True))
        return (e4m3(xb / s) * s).reshape(r, k)
    raise ValueError(mode)


def wt_quant(wq: np.ndarray, mode: str) -> np.ndarray:
    """W_q [K, N] (fp16 values) as the B operand."""
    if mode == "none":
        return wq
    if mode == "direct":
        return e4m3(wq)
    if mode == "tile":
        k, n = wq.shape
        t = wq.reshape(k // 128, 128, n // 128, 128)
        s = _scale(np.abs(t).max(axis=(1, 3), keepdims=True))
        return (e4m3(t / s) * s).reshape(k, n)
    raise ValueError(mode)


def lin_product(lin: exl3.Exl3Linear, x: np.ndarray, act: str, wt: str, wq: np.ndarray | None = None) -> np.ndarray:
    """The layer's output [rows, N] (float64) for one variant; `wq` = the quantized weight when the caller cached it."""
    wq = wt_quant(lin.wq, wt) if wq is None else wq
    return exl3.forward(x, wq, lin.suh, lin.svh, in_round=lambda xh: act_quant(xh, act))


# ----------------------------------------------------------------------------------------------------------------------
# statistics


def row_stats(err: np.ndarray, ref: np.ndarray) -> dict:
    """Per row: squared error, squared reference, max |err|; columns."""
    return {"e2": (err * err).sum(-1), "r2": (ref * ref).sum(-1), "mx": np.abs(err).max(-1), "cols": ref.shape[-1]}


class Acc:
    def __init__(self):
        self.d: dict[tuple[str, str], list[dict]] = {}

    def add(self, kind: str, variant: str, st_: dict) -> None:
        self.d.setdefault((kind, variant), []).append(st_)

    def report(self) -> dict:
        out: dict = {}
        for (kind, variant), lst in sorted(self.d.items()):
            e2 = np.concatenate([s["e2"] for s in lst])
            r2 = np.concatenate([s["r2"] for s in lst])
            mx = np.concatenate([s["mx"] for s in lst])
            cols = lst[0]["cols"]
            ok = r2 > 0
            rel = np.sqrt(e2[ok] / r2[ok])
            mxr = mx[ok] / np.sqrt(r2[ok] / cols)
            out.setdefault(kind, {})[variant] = {
                "rows": int(ok.sum()), "rel_l2": float(np.sqrt(e2.sum() / r2.sum())),
                "row_rel_p50": float(np.percentile(rel, 50)), "row_rel_p99": float(np.percentile(rel, 99)),
                "row_rel_max": float(rel.max()),
                "maxabs_rms_mean": float(mxr.mean()), "maxabs_rms_p99": float(np.percentile(mxr, 99)),
                "maxabs_rms_max": float(mxr.max()),
            }
        return out


# ----------------------------------------------------------------------------------------------------------------------
# MoE worker


def probe_expert(layer: int, e: int, x: np.ndarray, gate_cap: float, up_cap: float):
    """All variants of one expert on its real rows. Returns (stats per (kind, variant), y per variant float32)."""
    ix = pack._worker_ix
    gate, up, down = [load_exl3(ix, expert_prefix(layer, e, p)) for p in PROJS]
    rd = Rounding()
    res = {}
    wq: dict = {}

    def wqf(lin, wm):
        if (id(lin), wm) not in wq:
            wq[(id(lin), wm)] = wt_quant(lin.wq, wm)
        return wq[(id(lin), wm)]

    for name, (am, wm) in VARIANTS.items():
        f = lambda lin, v, am=am, wm=wm: lin_product(lin, v, am, wm, wqf(lin, wm))  # noqa: E731
        g = rd.np_f32(f(gate, x))
        u = rd.np_f32(f(up, x))
        a = softcap_np(g, gate_cap) * sigmoid_np(g) * softcap_np(u, up_cap)
        res[name] = (g, u, a, f(down, rd.np_bf16(a)))
    a0 = rd.np_bf16(res["exact"][2])
    stats: dict = {}
    ys = {}
    for name, (am, wm) in VARIANTS.items():
        g, u, a, y = res[name]
        ys[name] = y.astype(np.float32)
        if name == "exact":
            continue
        down_iso = lin_product(down, a0, am, wm, wqf(down, wm))
        stats[("moe_gate", name)] = row_stats(g - res["exact"][0], res["exact"][0])
        stats[("moe_up", name)] = row_stats(u - res["exact"][1], res["exact"][1])
        stats[("moe_down_iso", name)] = row_stats(down_iso - res["exact"][3], res["exact"][3])
        stats[("moe_expert_e2e", name)] = row_stats(y - res["exact"][3], res["exact"][3])
    return stats, ys, res["exact"][3].astype(np.float32)


def fp8_ffn_worker(layer: int, e: int, x: np.ndarray, gate_cap: float, up_cap: float, act: str, wt: str) -> np.ndarray:
    ix = pack._worker_ix
    gate, up, down = [load_exl3(ix, expert_prefix(layer, e, p)) for p in PROJS]
    rd = Rounding()
    wqs = {id(lin): wt_quant(lin.wq, wt) for lin in (gate, up, down)}
    f = lambda lin, v: lin_product(lin, v, act, wt, wqs[id(lin)])  # noqa: E731
    g, u = rd.np_f32(f(gate, x)), rd.np_f32(f(up, x))
    a = softcap_np(g, gate_cap) * sigmoid_np(g) * softcap_np(u, up_cap)
    return f(down, rd.np_bf16(a))


# ----------------------------------------------------------------------------------------------------------------------
# models


def _dense_kind(name: str) -> str | None:
    for suffix, kind in (("wq_du", "attn_q"), ("wr_du", "attn_r"), ("gate_attn", "attn_gate"), ("wk_dv", "attn_k"),
                         ("wv_dv", "attn_v"), ("wo_ud", "attn_o")):
        if f".attn.{suffix}.weight" in name:
            return kind
    return None


class ProbeRef(Inkling2LoopRef):
    """A-lane forward that also evaluates every variant on the real dense inputs and on the MoE of `probe_visits`."""

    def __init__(self, *a, probe_visits=(), pool=None, acc=None, **kw):
        super().__init__(*a, **kw)
        self.probe_visits, self.pool, self.acc = set(probe_visits), pool, acc
        self.cur = None
        self._wq: dict = {}

    def layer(self, v, *a, **kw):
        self.cur = v
        t0 = time.time()
        out = super().layer(v, *a, **kw)
        print(f"visit {v.index} layer {v.layer}: {time.time() - t0:.1f}s", flush=True)
        return out

    def lin(self, x, name):
        g = self.exl3_dense(name)
        kind = _dense_kind(name)
        if g is not None and kind is not None:
            xn = x.numpy()
            ref = lin_product(g, xn, "none", "none", g.wq)
            for vn, (am, wm) in VARIANTS.items():
                if vn == "exact":
                    continue
                key = (name, wm)
                if key not in self._wq:
                    self._wq[key] = wt_quant(g.wq, wm)
                self.acc.add(kind, vn, row_stats(lin_product(g, xn, am, wm, self._wq[key]) - ref, ref))
        return super().lin(x, name)

    def moe(self, v, x, rec):
        if v.index not in self.probe_visits:
            return super().moe(v, x, rec)
        c = self.cfg
        logits, idx, w, margin = self.router(v.layer, x)
        if rec is not None:
            rec["idx"].append(idx.to(torch.int32))
            rec["w"].append(w.to(self.out_dtype))
            rec["margin"].append(margin.to(self.out_dtype))
        jobs, rows = self.expert_jobs(x, idx)
        futs = [self.pool.submit(probe_expert, v.layer, e, xr, c.gate_softcap, c.up_softcap) for e, xr in jobs]
        accs = {n: torch.zeros(*x.shape, dtype=torch.float64) for n in VARIANTS}
        accs["ref"] = torch.zeros(*x.shape, dtype=torch.float64)
        for (e, _), (tok, slot), fu in zip(jobs, rows, futs):
            stats, ys, yref = fu.result()
            for (kind, vn), s in stats.items():
                self.acc.add(kind, vn, s)
            wt = w[tok, slot][:, None]
            for vn, y in ys.items():
                accs[vn].index_add_(0, tok, torch.from_numpy(y.astype(np.float64)) * wt)
            accs["ref"].index_add_(0, tok, torch.from_numpy(yref.astype(np.float64)) * wt)
        ref = accs["ref"].numpy()
        for vn in VARIANTS:
            if vn != "exact":
                self.acc.add("moe_layer", vn, row_stats(accs[vn].numpy() - ref, ref))
        return self.r.bf16(accs["A"])


class Fp8Experts:
    def __init__(self, ix: PackIndex, workers: int, act: str, wt: str):
        self.act, self.wt = act, wt
        self.pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=pack._worker_init,
                                        initargs=(ix.root,))

    def ffn_many(self, layer, jobs, gate_cap, up_cap, rounding, fp16_in=False):
        futs = [self.pool.submit(fp8_ffn_worker, layer, e, x, gate_cap, up_cap, self.act, self.wt) for e, x in jobs]
        return [f.result() for f in futs]

    def close(self):
        self.pool.shutdown()


class Fp8Ref(Inkling2LoopRef):
    """Every layer GEMM (dense EXL3 attention projections, MoE experts) in one fp8 variant; the rest is the A lane."""

    def __init__(self, *a, act: str, wt: str, **kw):
        super().__init__(*a, **kw)
        self.act, self.wt = act, wt
        self._wq: dict = {}
        self.cur = None

    def layer(self, v, *a, **kw):
        t0 = time.time()
        out = super().layer(v, *a, **kw)
        print(f"visit {v.index} layer {v.layer}: {time.time() - t0:.1f}s", flush=True)
        return out

    def lin(self, x, name):
        g = self.exl3_dense(name)
        if g is None or _dense_kind(name) is None:
            return super().lin(x, name)
        if name not in self._wq:
            self._wq[name] = wt_quant(g.wq, self.wt)
        return self.r.bf16(torch.from_numpy(lin_product(g, x.numpy(), self.act, self.wt, self._wq[name])))


# ----------------------------------------------------------------------------------------------------------------------


def read_fixture_tokens(path: str) -> tuple[list[int], int]:
    hdr = st.read_header(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n)).get("__metadata__", {})
    return st.read_tensor(hdr["tokens"]).tolist(), int(meta["prompt_len"])


def build(a, experts):
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    ix = PackIndex(a.pack)
    return cfg, ix


def cmd_layers(a) -> int:
    torch.set_num_threads(a.threads)
    cfg, ix = build(a, None)
    tokens, plen = read_fixture_tokens(a.fixture)
    ids = tokens[:plen]
    pool = ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=pack._worker_init, initargs=(ix.root,))
    experts = PackExperts(ix, workers=a.workers)
    acc = Acc()
    visits = [int(s) for s in a.visits.split(",") if s]
    model = ProbeRef(cfg, PackWeights(ix, int(a.weight_cache_gib * (1 << 30))), experts, emulate=Emulation.engine(),
                     probe_visits=visits, pool=pool, acc=acc)
    t0 = time.time()
    out = model.forward(ids, record=False)
    rep = acc.report()
    res = {"fixture": os.path.basename(a.fixture), "prompt_len": plen, "rows": len(ids), "probe_moe_visits": visits,
           "moe_layers": [model.visits[i].layer for i in visits], "seconds": round(time.time() - t0, 1),
           "variants": {k: list(v) for k, v in VARIANTS.items()}, "kinds": rep}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["kinds"], indent=1))
    experts.close()
    pool.shutdown()
    return 0


def cmd_forward(a) -> int:
    torch.set_num_threads(a.threads)
    cfg, ix = build(a, None)
    tokens, plen = read_fixture_tokens(a.fixture)
    act, wt = VARIANTS[a.variant]
    experts = Fp8Experts(ix, a.workers, act, wt)
    model = Fp8Ref(cfg, PackWeights(ix, int(a.weight_cache_gib * (1 << 30))), experts, emulate=Emulation.engine(),
                   act=act, wt=wt)
    t0 = time.time()
    rows = tokens[:-1]
    out = model.forward(rows, record=True)
    t = {"tokens": torch.tensor(tokens, dtype=torch.long), "logits": out.logits, "h_final": out.h_final,
         "hidden": out.hidden, "core_out": out.core_out[:cfg.loop_passes - 1].contiguous(),
         "router_idx": out.router_idx, "router_w": out.router_w, "router_margin": out.router_margin}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    st.write(a.out, t, {"name": os.path.basename(a.fixture), "prompt_len": str(plen), "emulate": f"engine+{a.variant}",
                        "numerics_version": "fp8-lab"})
    print(f"{a.variant}: {len(rows)} rows in {time.time() - t0:.0f}s -> {a.out}", flush=True)
    experts.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("layers", "forward"):
        p = sub.add_parser(name)
        p.add_argument("--pack", required=True)
        p.add_argument("--fixture", required=True, help="an oracle fixture: its tokens (and prompt_len) are the input")
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--weight-cache-gib", type=float, default=8.0)
        if name == "layers":
            p.add_argument("--visits", default="1,8,28,40,46", help="pass indexes whose MoE is probed")
        else:
            p.add_argument("--variant", default="f8_row", choices=[v for v in VARIANTS if v not in ("exact",)])
    a = ap.parse_args(argv)
    return cmd_layers(a) if a.cmd == "layers" else cmd_forward(a)


if __name__ == "__main__":
    sys.exit(main())
