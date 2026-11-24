"""Teacher-forced, op-by-op comparison of an engine dump against the oracle (docs/ORACLE-GAP.md), per LAYER PASS.

Inkling-2-Mini runs 48 layer passes a token goes through (`Inkling2LoopConfig.visits()`: 24 layers, the core [6, 18)
three times with shared weights, but a separate K/V and conv state per pass). Every oracle op here is fed the ENGINE's
inputs of that op, so each comparison measures one op's numerics with nothing accumulated from earlier passes.

Dump contract (the engine side writes it, `st.write` / a merged safetensors; the oracle writes the same thing with
`oracle_dump`). One prefill of T tokens from position 0 on a fresh cache. V = passes (48), H = hidden, E = routed
experts, k = experts per token, n_core = core layers (12), loop_passes = 3; "pass" = the index of `visits()`:

  hidden         [V+1, T, H]   bf16  P (the partial AttnRes block) ENTERING pass j; [V] = P after the last pass. At the
                                     first layer of every core pass after the first, P is the prelude's state again
                                     (the pass restarts), so hidden[j] there equals hidden[first core pass].
  core_out       [loop_passes-1, T, H]  bf16  P leaving the last core layer of core pass 0, 1, ...: the partial block
                                     that the next core pass's messages read (the last core pass needs none: its output
                                     is hidden of the layer after the core).
  attn_core      [V, T, Hq*D]  bf16  attention core output (after the gates, before the output projection)
  attn           [V, T, H]     bf16  attention sublayer output (wo, then attn_sconv)
  mlp_in         [V, T, H]     bf16  the MoE input (AttnRes mix [+ message blend] + mlp_norm)
  router_logits  [V, T, E]     f32
  router_idx     [V, T, k]     int   selected experts, any order      router_w [V, T, k] f32 routing weights
  moe            [V, T, H]     bf16/f32  the MoE output BEFORE mlp_sconv
  msg            [(loop_passes-1)*n_core, 4, T, H]  f32  the four messages (streams attn, k, v, mlp) of core layer
                                     `layer` in core pass q >= 1, row (q-1)*n_core + (layer - loop_start)
  attn_in        [(loop_passes-1)*n_core, 3, T, H]  bf16  the attention q, k, v inputs of that layer in that pass
                                     after the controller blends and the input norms (a local layer has one input norm:
                                     the q/k/v inputs are three separate blends of the same mix)

Ops, per pass (`blocks_for` rebuilds the completed AttnRes blocks from `hidden`: the passes of the CURRENT chain that
opened a block):

  msg        core passes >= 1: bank = the previous core pass's later blocks + its `core_out` -> the AttnRes read with the
             layer's four message queries (vs `msg`)
  attn_in    core passes >= 1: the engine's blocks and P -> AttnRes mix, the controllers blend it with the engine's `msg`,
             input norms (vs `attn_in`)
  attention  the attention inputs (the engine's `attn_in` where dumped, else the oracle's mix + norm of the engine's
             blocks and P) -> attention core (vs `attn_core`) -> wo, attn_sconv (vs `attn`); once per attention
             emulation variant
  mix        the engine's `attn` added to its P -> AttnRes mix [+ blend with the engine's `msg` of the mlp stream] +
             mlp_norm (vs `mlp_in`)
  router     the engine's `mlp_in` -> fp32 logits (vs `router_logits`) and the selection (vs `router_idx`/`router_w`)
  moe        the engine's `mlp_in`, picks and weights -> experts (vs `moe`); with and without the fp16 GEMM-input
             emulation, from one decode of each expert
  residual   the engine's `moe` -> mlp_sconv + residual; vs `hidden[j + 1]` where the next pass continues the same
             chain, vs `core_out` at the end of a core pass that is followed by another (the restart is checked too:
             `restart_mismatch`, elements of the first core layer's `hidden` that differ from the first core pass's)

and noise floors, oracle against oracle: the fully emulated attention sublayer, the message and the mix with fp32
accumulation / fp32 AttnRes read, and the fp16-input experts once with float64 and once with fp32 accumulation of their
products (torch / numpy float32 matmul: another fp32 summation order than the engine's). An engine op whose difference
from the emulating oracle is at that floor differs from it by summation order only.

    python oracle/teacher_force.py --pack DIR --engine ENGINE.safetensors --out REPORT.json
                                   [--passes 0-47] [--workers N] [--threads N] [--no-moe]

Row statistics: `rel` = |o - e| / |e| per row (median and max over rows), `cos_min`, `diff_frac` = fraction of
elements that differ, `abs_max_rms` = the largest element difference over its row's RMS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Callable

import numpy as np
import torch

import exl3
import st
from inkling2_ref import (F64, np_f16, STREAMS, Emulation, Inkling2LoopConfig, Inkling2LoopRef, Rounding, Visit, expert_ffn,
                          rmsnorm_w, route, static_conv)
from pack import PROJS, PackIndex, PackWeights, exit_with_parent, expert_prefix, load_exl3

ATTN_VARIANTS = {
    "none": Emulation(),
    "attn_bf16_p": Emulation(attn_bf16_p=True),
    "kv8": Emulation(kv8=True),
    "kv8+attn_bf16_p": Emulation(kv8=True, attn_bf16_p=True),
    "dense_fp16": Emulation(dense_fp16=True),
    # the engine's attention numerics: the last variant is the noise floor's and the gate's default (`ENGINE_VARIANT`)
    "kv8+attn_bf16_p+dense_fp16": Emulation(kv8=True, attn_bf16_p=True, dense_fp16=True),
}
ENGINE_VARIANT = "kv8+attn_bf16_p+dense_fp16"
# (fp16 GEMM inputs, fp32 accumulation of the expert products)
MOE_VARIANTS = {"none": (False, False), "moe_fp16": (True, False), "moe_fp16_acc32": (True, True)}

# the dump's planes: always required / required when the model has more than one core pass
PLANES = ("hidden", "attn_core", "attn", "mlp_in", "router_logits", "router_idx", "router_w", "moe")
LOOP_PLANES = ("core_out", "msg", "attn_in")

# (layer -> [per expert job -> [per MOE_VARIANTS output [rows, H]]]) for the experts of one layer
ExpertFn = Callable[[int, list], list]

_ix: PackIndex | None = None


# ----------------------------------------------------------------------------------------------------------------------
# the pass schedule: which earlier passes feed a pass


def block_passes(cfg: Inkling2LoopConfig, v: Visit) -> list[int]:
    """The passes whose entering P is a completed AttnRes block when pass `v` starts: the passes before it, in the chain
    that leads to it, whose layer opens a block (`Visit.block_start`). The chain of a core pass q is the prelude plus
    core pass q's earlier layers (every core pass restarts from the prelude's state); the chain of a pass after the core
    is the prelude, the LAST core pass and the layers after the core."""
    chain = v.loop_pass if v.is_core else cfg.loop_passes - 1
    return [u.index for u in cfg.visits()[:v.index]
            if u.block_start and (u.loop_pass is None or u.loop_pass == chain)]


def blocks_for(cfg: Inkling2LoopConfig, v: Visit, hidden: torch.Tensor) -> list[torch.Tensor]:
    """The completed blocks entering pass `v`, float64 [T, H] each, from `hidden` [V+1, T, H] (a block is the P that
    entered the pass that opened it)."""
    return [hidden[j].to(F64) for j in block_passes(cfg, v)]


def bank_for(cfg: Inkling2LoopConfig, v: Visit, hidden: torch.Tensor,
             core_out: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The message bank of a core pass q >= 1 layer: (the blocks core pass q - 1 opened after the first one, its final
    partial block = `core_out[q - 1]`)."""
    if v.loop_pass is None or v.loop_pass < 1:
        raise ValueError(f"pass {v.index} has no message bank (core passes after the first have)")
    prev = v.loop_pass - 1
    opened = [u.index for u in cfg.visits() if u.loop_pass == prev and u.block_start][1:]
    return [hidden[j].to(F64) for j in opened], core_out[prev].to(F64)


def loop_row(cfg: Inkling2LoopConfig, v: Visit) -> int:
    """The row of `msg` / `attn_in` of a core pass q >= 1 layer."""
    return (v.loop_pass - 1) * cfg.n_core + v.layer - cfg.loop_start


def next_state(cfg: Inkling2LoopConfig, v: Visit, hidden: torch.Tensor, core_out: torch.Tensor | None) -> torch.Tensor:
    """What pass `v`'s output must equal in the dump: `hidden[v + 1]`, except at the end of a core pass that another
    follows (hidden there is the restart)."""
    if v.is_core and v.layer == cfg.loop_end - 1 and v.loop_pass < cfg.loop_passes - 1:
        return core_out[v.loop_pass]
    return hidden[v.index + 1]


# ----------------------------------------------------------------------------------------------------------------------
# the oracle's own dump


def oracle_dump(model: Inkling2LoopRef, ids) -> dict[str, torch.Tensor]:
    """A prefill of `ids` on a fresh cache as a dump of the contract above: what a perfect engine writes (used by the
    tests and to diff two oracle configurations)."""
    cfg = model.cfg
    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
    model.taps = {}
    try:
        out = model.forward(ids, record=True)
        taps = model.taps
    finally:
        model.taps = None
    plane = lambda name, dt: torch.stack([taps[v.index][name] for v in model.visits]).to(dt)  # noqa: E731
    loop = [v for v in model.visits if v.loop_pass]
    dump = {
        "tokens": ids, "hidden": out.hidden, "core_out": out.core_out[:cfg.loop_passes - 1],
        "attn_core": plane("core", torch.bfloat16), "attn": plane("attn", torch.bfloat16),
        "mlp_in": plane("mlp_in", torch.bfloat16), "router_logits": plane("router_logits", torch.float32),
        "router_idx": out.router_idx, "router_w": out.router_w, "moe": plane("moe", torch.bfloat16),
    }
    if loop:
        dump["msg"] = torch.stack([taps[v.index]["msg"] for v in loop]).to(torch.float32)
        dump["attn_in"] = torch.stack([taps[v.index]["attn_in"] for v in loop]).to(torch.bfloat16)
    return dump


def check_dump(cfg: Inkling2LoopConfig, eng: dict[str, torch.Tensor]) -> None:
    """Raise ValueError naming what is missing or has the wrong shape for this model's schedule."""
    need = PLANES + (LOOP_PLANES if cfg.loop_passes > 1 else ())
    missing = [k for k in need if k not in eng]
    if missing:
        raise ValueError(f"the engine dump has no {missing}")
    v, t = cfg.n_visits, eng["hidden"].shape[1]
    want = {"hidden": (v + 1, t, cfg.hidden_size), "attn_core": (v, t, cfg.num_heads * cfg.head_dim),
            "attn": (v, t, cfg.hidden_size), "mlp_in": (v, t, cfg.hidden_size),
            "router_logits": (v, t, cfg.n_experts), "router_idx": (v, t, cfg.top_k), "router_w": (v, t, cfg.top_k),
            "moe": (v, t, cfg.hidden_size)}
    if cfg.loop_passes > 1:
        rows = (cfg.loop_passes - 1) * cfg.n_core
        want |= {"core_out": (cfg.loop_passes - 1, t, cfg.hidden_size), "msg": (rows, 4, t, cfg.hidden_size),
                 "attn_in": (rows, 3, t, cfg.hidden_size)}
    bad = [f"{k} {tuple(eng[k].shape)} (want {s})" for k, s in want.items() if tuple(eng[k].shape) != s]
    if bad:
        raise ValueError(f"the engine dump does not match the {v}-pass schedule: " + "; ".join(bad))


# ----------------------------------------------------------------------------------------------------------------------
# the noise-floor oracle and the experts


class Fp32Accum(Inkling2LoopRef):
    """The oracle with its dense products accumulated in fp32 (torch's float32 matmul) and its AttnRes reads computed in
    fp32: a second fp32 implementation of the same arithmetic, for the noise floor. Hooks: `matmul`, `mix_raw`."""

    def matmul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float32) @ w.to(torch.float32).T).to(F64)

    def dense_fp16_product(self, lin, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(lin(x.numpy(), in_round=np_f16, acc32=True)))

    def mix_raw(self, blocks, partial, q):
        f = torch.float32
        x = torch.stack(blocks + [partial]).to(f)
        rho = torch.rsqrt((x * x).mean(-1) + 1e-6)
        p = torch.softmax(torch.einsum("nth,qh->qnt", x, q.to(f)) * rho[None], dim=1)
        return torch.einsum("qnt,nth->qth", p * rho[None], x).to(F64)


def _init(root: str, worker: bool = True) -> None:
    global _ix
    if worker:
        exit_with_parent()
    _ix = PackIndex(root)


def _acc32(lin: exl3.Exl3Linear):
    """The EXL3 layer with its W_q product accumulated in fp32 (inputs and W_q values are exact in fp32)."""
    def f(x: np.ndarray, in_round=None) -> np.ndarray:
        xh = exl3.fwht_128(np.asarray(x, dtype=np.float64) * lin.suh[None, :], axis=1)
        if in_round is not None:
            xh = in_round(xh)
        y = (xh.astype(np.float32) @ lin.wq.astype(np.float32)).astype(np.float64)
        return exl3.fwht_128(y, axis=1) * lin.svh[None, :]
    return f


def ffn_variants(load: Callable[[str], exl3.Exl3Linear], layer: int, e: int, x: np.ndarray, gate_cap: float,
                 up_cap: float, variants: tuple[tuple[bool, bool], ...]) -> list[np.ndarray]:
    """One decode of expert e (`load(prefix)` -> its EXL3 layers), its FFN once per (fp16 inputs, fp32 accumulation)."""
    lins = [load(expert_prefix(layer, e, p)) for p in PROJS]
    out = []
    for fp16, acc32 in variants:
        ls = [_acc32(lin) for lin in lins] if acc32 else lins
        out.append(expert_ffn(*ls, x, gate_cap, up_cap, Rounding(), fp16))
    return out


def _pool_ffn(layer: int, e: int, x: np.ndarray, gate_cap: float, up_cap: float, variants) -> list[np.ndarray]:
    return ffn_variants(lambda prefix: load_exl3(_ix, prefix), layer, e, x, gate_cap, up_cap, variants)


def local_experts(load: Callable[[str], exl3.Exl3Linear], cfg: Inkling2LoopConfig,
                  pool: ProcessPoolExecutor | None = None) -> ExpertFn:
    """The experts of a layer for `teacher_force`: in this process over `load`, or in the pool over the pack."""
    variants = tuple(MOE_VARIANTS.values())

    def run(layer: int, jobs: list) -> list:
        if pool is not None:
            args = [(layer, e, xe, cfg.gate_softcap, cfg.up_softcap, variants) for e, xe in jobs]
            return list(pool.map(_pool_ffn, *zip(*args)))
        return [ffn_variants(load, layer, e, xe, cfg.gate_softcap, cfg.up_softcap, variants) for e, xe in jobs]
    return run


# ----------------------------------------------------------------------------------------------------------------------
# statistics


PER_ROW = False  # `--per-row`: every row_stats entry also carries its per-row values under "by_row"


def row_stats(o: torch.Tensor, e: torch.Tensor) -> dict:
    o, e = o.to(F64).reshape(o.shape[0], -1), e.to(F64).reshape(e.shape[0], -1)
    d = o - e
    en = e.norm(dim=-1)
    rel = d.norm(dim=-1) / en.clamp_min(1e-300)
    cos = (o * e).sum(-1) / (o.norm(dim=-1) * en).clamp_min(1e-300)
    rms = (en / e.shape[1] ** 0.5).clamp_min(1e-300)
    out = {"rel_med": float(rel.median()), "rel_max": float(rel.max()), "rel_max_row": int(rel.argmax()),
           "cos_min": float(cos.min()), "diff_frac": float((d != 0).double().mean()),
           "abs_max_rms": float((d.abs().amax(-1) / rms).max())}
    if PER_ROW:
        out["by_row"] = {"rel": rel.tolist(), "cos": cos.tolist(), "diff_frac": (d != 0).double().mean(-1).tolist(),
                         "abs_max_rms": (d.abs().amax(-1) / rms).tolist()}
    return out


def parse_passes(spec: str, n: int) -> list[int]:
    """'0-47', '3', '0-5,40-47' -> pass indices (ranges inclusive), each in 0..n-1."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if any(not 0 <= i < n for i in out):
        raise ValueError(f"passes out of range 0..{n - 1}: {spec}")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# the ops


def attention_inputs(m: Inkling2LoopRef, v: Visit, p: torch.Tensor, blocks: list[torch.Tensor],
                     msgs: list[torch.Tensor] | None) -> torch.Tensor:
    """The attention q, k, v inputs [3, T, H] of pass `v` from P, the completed blocks and (core passes >= 1) the four
    messages: the AttnRes read, the controllers' blends, the input norms (the structure of `Inkling2LoopRef.layer`)."""
    c, i = m.cfg, v.layer
    ln = f"model.llm.layers.{i}."
    norms = [m.w64(ln + f"{n}_input_norm.weight") for n in "qkv"] if v.is_global else [m.w64(ln + "attn_norm.weight")]
    if i == 0:
        xs = [rmsnorm_w(p, w, c.eps, m.r) for w in norms]
    elif msgs is None:
        xs = m.mix(blocks, p, m.query(f"model.llm.attn_res.attn_queries.{i - 1}"), norms)
    else:
        raw = m.mix_raw(blocks, p, m.query(f"model.llm.attn_res.attn_queries.{i - 1}"))
        dests = [raw[0], raw[1], raw[2]] if v.is_global else [raw[0]] * 3
        xs = [rmsnorm_w(m.blend(i, s, dests[j], msgs[j]), norms[j if v.is_global else 0], c.eps, m.r)
              for j, s in enumerate(STREAMS[:3])]
    return torch.stack(xs if len(xs) == 3 else [xs[0]] * 3)  # a local layer's single input, or its three blends


def mlp_input(m: Inkling2LoopRef, v: Visit, blocks: list[torch.Tensor], part: torch.Tensor,
              msg: torch.Tensor | None) -> torch.Tensor:
    """The MoE input of pass `v`: the AttnRes read over `blocks` and the partial block `part`, the mlp-stream blend with
    `msg` (core passes >= 1), mlp_norm."""
    i = v.layer
    raw = m.mix_raw(blocks, part, m.query(f"model.llm.attn_res.mlp_queries.{i}"))[0]
    if msg is not None:
        raw = m.blend(i, "mlp", raw, msg)
    return rmsnorm_w(raw, m.w64(f"model.llm.layers.{i}.mlp_norm.weight"), m.cfg.eps, m.r)


def teacher_force(cfg: Inkling2LoopConfig, weights, eng: dict[str, torch.Tensor], passes: list[int],
                  experts: ExpertFn | None = None, log: Callable[[str], None] = print) -> dict:
    """The op report of the passes `passes` of an engine dump (contract in the module docstring) against the oracle over
    `weights` (a `WeightSource`). `experts` runs the experts of a layer (`local_experts`); None skips the MoE op."""
    check_dump(cfg, eng)
    hid = eng["hidden"].to(F64)
    core_out = eng["core_out"].to(F64) if "core_out" in eng else None
    t = hid.shape[1]
    pos = torch.arange(t)
    model = Inkling2LoopRef(cfg, weights, experts=None)
    model32 = Fp32Accum(cfg, weights, experts=None, emulate=ATTN_VARIANTS[ENGINE_VARIANT])
    r = model.r
    first_core = next((u.index for u in model.visits if u.is_core), None)
    report: dict = {"rows": t, "passes": []}
    for k in passes:
        t0 = time.time()
        v = model.visits[k]
        i = v.layer
        ln = f"model.llm.layers.{i}."
        looped = v.is_core and v.loop_pass >= 1
        blocks, p = blocks_for(cfg, v, hid), hid[k]
        rep: dict = {"pass": k, "layer": i, "loop_pass": v.loop_pass, "global": v.is_global, "attn": {}}
        e_msg = eng["msg"][loop_row(cfg, v)].to(F64) if looped else None  # [4, T, H]
        if looped:
            # the messages: the previous core pass's bank read with this layer's four queries
            bank = bank_for(cfg, v, hid, core_out)
            queries = model.message_queries()[i - cfg.loop_start]
            msg = torch.stack([r.f32(x) for x in model.mix_raw(*bank, queries)])
            msg32 = torch.stack([r.f32(x) for x in model32.mix_raw(*bank, queries)])
            rep["msg"] = row_stats(msg.transpose(0, 1), e_msg.transpose(0, 1))
            rep["msg_noise"] = row_stats(msg32.transpose(0, 1), msg.transpose(0, 1))
            # the attention inputs: the mix of the engine's blocks and P, blended with the engine's messages
            xs = attention_inputs(model, v, p, blocks, list(e_msg))
            xs32 = attention_inputs(model32, v, p, blocks, list(e_msg))
            e_xs = eng["attn_in"][loop_row(cfg, v)].to(F64)
            rep["attn_in"] = row_stats(xs.transpose(0, 1), e_xs.transpose(0, 1))
            rep["attn_in_noise"] = row_stats(xs32.transpose(0, 1), xs.transpose(0, 1))
            xs = e_xs  # attention is fed the engine's own inputs
        else:
            xs = attention_inputs(model, v, p, blocks, None)  # not in the dump: the oracle's mix + norm
        cw = model.w64(ln + "attn_sconv.weight")[:, 0, :]

        def attn_out(m: Inkling2LoopRef):
            m.taps = {}
            lc = m.new_cache().layers[k]
            a_raw = m.attention(v, xs[0], xs[1], xs[2], pos, lc)
            av, _ = static_conv(a_raw, cw, lc.conv["attn"], r)
            core = m.taps[k]["core"]
            m.taps = None
            return core, av

        for name, em in ATTN_VARIANTS.items():
            model.em = em
            core, av = attn_out(model)
            rep["attn"][name] = {"core": row_stats(core, eng["attn_core"][k]), "a": row_stats(av, eng["attn"][k])}
        core32, av32 = attn_out(model32)  # noise floor: the same emulated sublayer in a second fp32 implementation
        rep["attn_noise"] = {"core": row_stats(core32, core), "a": row_stats(av32, av)}
        model.em = Emulation()
        # the MoE input from the engine's attention output (and messages)
        e_a = eng["attn"][k].to(F64)
        mblocks = blocks + [p] if v.block_start else blocks
        part = e_a if v.block_start else r.f32(p + e_a)
        mlp_in = mlp_input(model, v, mblocks, part, e_msg[3] if looped else None)
        rep["mix"] = row_stats(mlp_in, eng["mlp_in"][k])
        rep["mix_noise"] = row_stats(mlp_input(model32, v, mblocks, part, e_msg[3] if looped else None), mlp_in)
        # router on the engine's MoE input
        x = eng["mlp_in"][k].to(F64)
        wg = model.w64(ln + "mlp.gate.weight")
        logits, idx, w, margin = model.router(i, x)
        e_log = eng["router_logits"][k].to(F64)
        sumabs = x.abs() @ wg.abs().T  # [T, E]: sum_k |x_k w_k|, the scale of an fp32 dot product's error
        dlog = (logits - e_log).abs()
        bias = model.w.tensor(ln + "mlp.gate.bias").to(F64)
        e_idx = eng["router_idx"][k].long()
        # the engine's picks against a float64 selection on the engine's own logits: the selection logic alone
        idx_from_e, w_from_e, _ = route(e_log, bias, cfg.top_k, cfg.route_scale, r)
        same = (idx.sort(-1).values == e_idx.sort(-1).values).all(-1)
        consistent = (idx_from_e.sort(-1).values == e_idx.sort(-1).values).all(-1)
        # score error bound of a row: twice the largest |sigmoid(oracle logit) - sigmoid(engine logit)|
        dsel = (torch.sigmoid(logits) - torch.sigmoid(e_log)).abs().amax(-1)
        flips = [{"row": int(j), "oracle_margin": float(margin[j]), "score_err_bound": float(2 * dsel[j]),
                  "within_bound": bool(margin[j] <= 2 * dsel[j])}
                 for j in torch.nonzero(~same).flatten().tolist()]
        rep["router"] = {
            "logit_err_max": float(dlog.max()),
            "logit_err_max_rel_sumabs": float((dlog / sumabs.clamp_min(1e-300)).max()),
            "picks_identical": float(same.double().mean()),
            "engine_select_consistent": bool(consistent.all()),
            "weights_err_max": float((w_from_e - eng["router_w"][k].to(F64)).abs().max()),
            "margin_min": float(margin.min()),
            "flips": flips,
        }
        if PER_ROW:
            rep["router"]["by_row"] = {"logit_err_rel_sumabs": (dlog / sumabs.clamp_min(1e-300)).amax(-1).tolist(),
                                       "picks_identical": same.tolist(), "oracle_margin": margin.tolist(),
                                       "score_err_bound": (2 * dsel).tolist()}
        # experts on the engine's input, picks and weights; mlp_sconv reads the MoE output rounded to bf16
        mo = r.bf16(eng["moe"][k].to(F64))
        if experts is not None:
            e_w = eng["router_w"][k].to(F64)
            jobs, rows = Inkling2LoopRef.expert_jobs(x, e_idx)
            outs = experts(i, jobs)
            ms = {name: model.combine(jobs, rows, [o[j] for o in outs], e_w, x.shape)
                  for j, name in enumerate(MOE_VARIANTS)}
            rep["moe"] = {name: row_stats(m, mo) for name, m in ms.items() if name != "moe_fp16_acc32"}
            rep["moe_noise"] = row_stats(ms["moe_fp16_acc32"], ms["moe_fp16"])
            rep["moe_experts"] = len(jobs)
        # mlp_sconv + residual from the engine's MoE output
        mc, _ = static_conv(mo, model.w64(ln + "mlp_sconv.weight")[:, 0, :],
                            torch.zeros(cfg.conv_kernel - 1, cfg.hidden_size, dtype=F64), r)
        p_out = r.bf16(r.f32(part + mc))
        bad_res = p_out != next_state(cfg, v, hid, core_out)
        rep["residual_mismatch"] = int(bad_res.sum())
        if PER_ROW:
            rep["residual_mismatch_by_row"] = bad_res.sum(-1).tolist()
        rep["restart_mismatch"] = (int((p != hid[first_core]).sum())
                                   if v.is_core and v.layer == cfg.loop_start and v.loop_pass >= 1 else 0)
        rep["seconds"] = round(time.time() - t0, 1)
        report["passes"].append(rep)
        at, an = rep["attn"], rep["attn_noise"]
        log(f"pass {k:2d} L{i:2d}{'G' if v.is_global else ' '}{'' if v.loop_pass is None else f' c{v.loop_pass}'} "
            f"core rel {at['none']['core']['rel_med']:.2e}/{at['kv8+attn_bf16_p']['core']['rel_med']:.2e}/"
            f"{an['core']['rel_med']:.2e}  a rel {at['none']['a']['rel_med']:.2e}/"
            f"{at['kv8+attn_bf16_p']['a']['rel_med']:.2e}/{an['a']['rel_med']:.2e}  "
            f"mix diff {rep['mix']['diff_frac']:.1e}"
            + (f"  msg rel {rep['msg']['rel_max']:.1e} attn_in rel {rep['attn_in']['rel_max']:.1e}" if looped else "")
            + f"  router same {rep['router']['picks_identical']:.3f} flips {len(flips)} "
            f"(out of bound {sum(not f['within_bound'] for f in flips)})"
            + (f"  moe rel {rep['moe']['none']['rel_med']:.2e}/{rep['moe']['moe_fp16']['rel_med']:.2e}/"
               f"{rep['moe_noise']['rel_med']:.2e}" if "moe" in rep else "")
            + f"  resid {rep['residual_mismatch']}  {rep['seconds']}s")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--engine", required=True, help="the engine dump with the sub-op planes (module docstring)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="0-47", help="layer passes to check, inclusive ranges: 0-5,40-47")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--per-row", action="store_true",
                    help="every metric of the report also per row (`by_row` lists; `residual_mismatch_by_row`): a row that "
                         "stands out against its neighbours is visible (`row_table.py` ranks them)")
    ap.add_argument("--no-moe", action="store_true", help="skip the expert comparison (the costly part)")
    ap.add_argument("--weight-cache-gib", type=float, default=0.0,
                    help="GiB of decoded float64 EXL3 dense weights kept in memory (see gen_fixtures.py)")
    a = ap.parse_args(argv)
    global PER_ROW
    PER_ROW = a.per_row
    if a.threads:
        torch.set_num_threads(a.threads)
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    eng = st.load_all(a.engine)
    try:
        check_dump(cfg, eng)
    except ValueError as err:
        raise SystemExit(f"{a.engine}: {err}")
    ix = PackIndex(a.pack)
    pool, experts = None, None
    if not a.no_moe:
        if a.workers > 0:
            pool = ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=_init,
                                       initargs=(ix.root,))
        else:
            _init(ix.root, worker=False)
        experts = local_experts(lambda prefix: load_exl3(ix, prefix), cfg, pool)
    try:
        report = teacher_force(cfg, PackWeights(ix, int(a.weight_cache_gib * (1 << 30))), eng,
                               parse_passes(a.passes, cfg.n_visits), experts)
    finally:
        if pool is not None:
            pool.shutdown()
    report = {"engine": os.path.abspath(a.engine), **report}
    report["gate"] = gate(report["passes"])
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["gate"], indent=1))
    return 0 if report["gate"]["pass"] else 1


# The correctness gate on identical inputs, against the EMULATING oracle (docs/ORACLE-GAP.md, "Gates"). These are the
# Small oracle's limits (about 3x above the largest value measured on the Small's four default fixtures, below what one
# missing emulation cost there: int8 KV in the attention core, fp16 experts in the MoE output), carried over unchanged:
# Small oracle limits, to be re-measured on the Mini fixtures (docs/ORACLE-GAP.md after the first comparison). `msg` and
# `attn_in` have no Small counterpart: they are the same AttnRes read + norm family as `mix`, so they reuse its limit
# until measured.
GATES = {
    "core_rel_med_max": 2e-3,  # attention core, kv8 + bf16 P emulated: per-pass row median
    "a_rel_med_max": 6e-3,  # attention sublayer output after attn_sconv
    "mix_rel_max": 2e-3,  # MoE input, every row; also the limit of the loop-only msg and attn_in, every row
    "moe_rel_med_max": 1.5e-3,  # MoE output, fp16 inputs emulated, the engine's picks
    "router_logit_err_rel_sumabs_max": 1e-6,  # |logit error| / sum_k |x_k w_k|, every row and expert
}


def gate(passes: list[dict], variant: str = ENGINE_VARIANT, moe_variant: str = "moe_fp16") -> dict:
    """Pass/fail of the op-level correctness gate over a report's passes (router flips must be within their bound;
    the residual and restart mismatches are reported)."""
    g, bad = GATES, []
    for x in passes:
        k = x["pass"]
        em = x["attn"][variant]
        if em["core"]["rel_med"] > g["core_rel_med_max"]:
            bad.append(f"pass {k}: attention core rel {em['core']['rel_med']:.2e}")
        if em["a"]["rel_med"] > g["a_rel_med_max"]:
            bad.append(f"pass {k}: attention output rel {em['a']['rel_med']:.2e}")
        for op in ("msg", "attn_in", "mix"):
            if op in x and x[op]["rel_max"] > g["mix_rel_max"]:
                bad.append(f"pass {k}: {op} rel {x[op]['rel_max']:.2e}")
        if "moe" in x and x["moe"][moe_variant]["rel_med"] > g["moe_rel_med_max"]:
            bad.append(f"pass {k}: moe rel {x['moe'][moe_variant]['rel_med']:.2e}")
        rt = x["router"]
        if rt["logit_err_max_rel_sumabs"] > g["router_logit_err_rel_sumabs_max"] or not rt["engine_select_consistent"]:
            bad.append(f"pass {k}: router logits or selection")
        bad += [f"pass {k}: router flip at margin {f['oracle_margin']:.3g} > bound {f['score_err_bound']:.3g}"
                for f in rt["flips"] if not f["within_bound"]]
    return {"gates": g, "variant": variant, "failures": bad,
            "residual_mismatch": sum(x["residual_mismatch"] for x in passes),
            "restart_mismatch": sum(x["restart_mismatch"] for x in passes), "pass": not bad}


if __name__ == "__main__":
    sys.exit(main())
