"""The engine's whole-forward planes against an oracle fixture, free of teacher forcing (docs/ORACLE-GAP.md).

`teacher_force.py` feeds every oracle op the engine's own inputs and so measures one op at a time. This tool measures what
the engine writes against what the oracle writes when each runs its own forward on the same token ids: the numbers that
accumulate over 48 layer passes. Both sides are fixtures in the layout of `gen_fixtures.py` (an engine dump of the prefill
hooks `REXL3_INKLING2_LOOP_DUMP` / `_OPS` is the same layout, in one file per prefill chunk):

  tokens I64 [N]; logits F32 [N-1, V]; hidden BF16 [P+1, N-1, H]; core_out BF16 [loop_passes-1, N-1, H]; h_final BF16
  [N-1, H]; router_idx I32 [P, N-1, k]; router_w F32 [P, N-1, k]; router_margin F32 [P, N-1] (oracle fixtures only);
  an engine dump with `router_logits` F32 [P, N-1, E] (ops dump) also gives the engine's own selection margin at a flip.

Row t of every plane is the forward at the token at position t (logits row t predicts token t + 1). The engine dump holds
the rows it prefilled: `tokens[:N-1]` of the fixture, so each row has a fixture row. Compared per group of rows
(`groups_for`): all, the prompt's rows, the generated rows, and for a prompt that crosses the 1,024-token sliding window the rows
before it, after it, and the two edges.

  hidden    per visit 0..P: the relative L2 error of the whole plane, the row median and maximum of ||e - o|| / ||o||, the
            row minimum and median cosine, the largest |e - o| over the plane's RMS (`maxabs_rms`: its row maximum).
  core_out, h_final   the same statistics, one entry per plane row.
  logits    per row: cosine, relative L2, max |e - o| over the row's RMS, KL(oracle || engine), argmax equal, top-2 order
            equal (both the top token and the runner-up the same), the oracle's top-1 minus top-2 margin; every row where
            the argmax differs is listed with its margin and with the oracle's gap between its top-1 and the engine's pick.
  router    per visit and row: the engine's top-k set equals the oracle's or it does not (a flip); for a flip the
            oracle's selection margin (k-th minus (k+1)-th of sigmoid(logit) + bias), and the engine's own where the engine
            dump has its router logits and `--pack` gives the bias. The weights of identical sets are compared too.

    python oracle/engine_gap.py --oracle FIXTURE.safetensors --engine DUMP_DIR_OR_FILE [--pack PACK] [--json REPORT]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

import st

F64 = torch.float64
WINDOW = 1024  # the sliding window of the Mini's local layers


# ----------------------------------------------------------------------------------------------------------------------
# loading


def read_meta(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)).get("__metadata__", {})


def load_fixture(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    return st.load_all(path), read_meta(path)


# planes concatenated over chunks, and the axis the rows are on
ROW_AXIS = {"hidden": 1, "core_out": 1, "h_final": 0, "logits": 0, "router_idx": 1, "router_w": 1, "router_logits": 1}


def chunk_runs(dump_dir: str) -> list[dict]:
    """The chunk files of a dump directory in order: {stem, first, rows, tokens} from each `prefillNNNNN.json`."""
    out = []
    for name in sorted(os.listdir(dump_dir)):
        if name.startswith("prefill") and name.endswith(".json"):
            stem = os.path.join(dump_dir, name[:-5])
            if not os.path.exists(stem + ".safetensors"):
                continue
            with open(stem + ".json") as f:
                meta = json.load(f)
            out.append({"stem": stem, "first": int(meta["first"]), "rows": int(meta["rows"]), "tokens": [int(t) for t in meta["tokens"]],
                        "segments": int(meta.get("segments", 1))})
    return out


def find_run(chunks: list[dict], tokens: list[int]) -> list[dict]:
    """The consecutive chunks that prefilled exactly `tokens` from position 0 (one slot, one segment each). A server's
    dump directory holds the chunks of every prompt it served, in the order served; the last matching run wins (a prompt
    sent twice)."""
    found = []
    for i, c in enumerate(chunks):
        if c["first"] != 0 or c["segments"] != 1:
            continue
        run, at = [], 0
        for d in chunks[i:]:
            if d["segments"] != 1 or d["first"] != at or d["tokens"] != tokens[at:at + d["rows"]]:
                break
            run.append(d)
            at += d["rows"]
            if at == len(tokens):
                found.append(run)
                break
    if not found:
        raise ValueError(f"no run of dump chunks prefilled these {len(tokens)} tokens from position 0")
    return found[-1]


def assemble(dump_dir: str, tokens: list[int]) -> dict[str, torch.Tensor]:
    """The planes of the dump chunks that prefilled `tokens`, concatenated over rows; plus `tokens`."""
    run = find_run(chunk_runs(dump_dir), tokens)
    parts: dict[str, list[torch.Tensor]] = {}
    for c in run:
        for name, e in st.read_header(c["stem"] + ".safetensors").items():
            if name in ROW_AXIS:  # the sub-op planes of an ops dump (gigabytes) are not read
                parts.setdefault(name, []).append(st.read_tensor(e))
    out = {name: p[0] if len(p) == 1 else torch.cat(p, dim=ROW_AXIS[name]) for name, p in parts.items()}
    out["tokens"] = torch.tensor(tokens, dtype=torch.long)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# statistics


def groups_for(rows: int, prompt_len: int) -> dict[str, tuple[int, int]]:
    """Row ranges [start, stop) to report: all, the prompt's rows (predicting prompt tokens 1.. and the first generated one
    at the last), the generated rows (the rows whose argmax is a greedy token: from prompt_len - 1), and, for a prompt past the
    window, before it, after it, and the edges (the first rows with the first key out of the window; the tail)."""
    g = {"all": (0, rows), "prompt": (0, min(prompt_len - 1, rows)), "generated": (min(prompt_len - 1, rows), rows)}
    if rows > WINDOW + 16:
        g.update({"before_window": (0, WINDOW), "after_window": (WINDOW, rows), "edge_1024": (WINDOW - 8, WINDOW + 8),
                  "tail": (max(rows - 16, 0), rows)})
    return {k: v for k, v in g.items() if v[1] > v[0]}


def row_stats(o: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per row of [R, D] planes (float64): relative L2 error, cosine, largest |e - o| over the oracle row's RMS, and the
    row's squared error and squared norm (to add up a plane's relative L2)."""
    o, e = o.to(F64), e.to(F64)
    d = e - o
    on, en, dn = o.norm(dim=-1), e.norm(dim=-1), d.norm(dim=-1)
    rms = on / math.sqrt(o.shape[-1])
    cos = torch.where(on * en > 0, (o * e).sum(-1) / (on * en).clamp_min(1e-300), (on == en).to(F64))
    return {"rel": dn / on.clamp_min(1e-300), "cos": cos, "maxabs_rms": d.abs().amax(-1) / rms.clamp_min(1e-300),
            "d2": dn * dn, "o2": on * on}


def _median(x: torch.Tensor) -> float:
    return float(x.sort().values[[(x.numel() - 1) // 2, x.numel() // 2]].mean())


def summarize(s: dict[str, torch.Tensor], lo: int, hi: int) -> dict:
    """Aggregates of `row_stats` over rows [lo, hi)."""
    rel, cos, mabs = s["rel"][lo:hi], s["cos"][lo:hi], s["maxabs_rms"][lo:hi]
    d2, o2 = float(s["d2"][lo:hi].sum()), float(s["o2"][lo:hi].sum())
    return {"rel_l2_plane": math.sqrt(d2 / o2) if o2 > 0 else (0.0 if d2 == 0 else math.inf),
            "rel_l2_row_median": _median(rel), "rel_l2_row_max": float(rel.max()),
            "cos_row_min": float(cos.min()), "cos_row_median": _median(cos), "maxabs_rms_row_max": float(mabs.max()),
            "rows_differing": int((s["d2"][lo:hi] > 0).sum())}


def plane_report(o: torch.Tensor, e: torch.Tensor, groups: dict[str, tuple[int, int]]) -> dict:
    """`o`, `e`: [S, R, D] (S planes, e.g. the visits) -> {group: [summary per plane]}."""
    stats = [row_stats(o[i], e[i]) for i in range(o.shape[0])]
    return {g: [summarize(s, lo, hi) for s in stats] for g, (lo, hi) in groups.items()}


def logits_rows(o: torch.Tensor, e: torch.Tensor, block: int = 64) -> dict[str, torch.Tensor]:
    """Per row statistics of two logit planes [R, V]: the `row_stats` triple, KL(o || e), argmax and top-2 equality, the
    oracle's top-1 minus top-2 margin, the engine's, and the oracle's gap between its own top-1 logit and the engine's
    pick."""
    acc: dict[str, list[torch.Tensor]] = {}
    for a in range(0, o.shape[0], block):
        ob, eb = o[a:a + block].to(F64), e[a:a + block].to(F64)
        s = row_stats(ob, eb)
        lp, lq = torch.log_softmax(ob, -1), torch.log_softmax(eb, -1)
        s["kl"] = (lp.exp() * (lp - lq)).sum(-1)
        ot, et = ob.topk(2, dim=-1), eb.topk(2, dim=-1)
        s["argmax_equal"] = (ot.indices[:, 0] == et.indices[:, 0]).to(F64)
        s["top2_equal"] = ((ot.indices[:, 0] == et.indices[:, 0]) & (ot.indices[:, 1] == et.indices[:, 1])).to(F64)
        s["oracle_margin"] = ot.values[:, 0] - ot.values[:, 1]
        s["engine_margin"] = et.values[:, 0] - et.values[:, 1]
        s["oracle_gap_to_engine_pick"] = ot.values[:, 0] - ob.gather(1, et.indices[:, :1])[:, 0]
        s["oracle_top1"], s["engine_top1"] = ot.indices[:, 0].to(F64), et.indices[:, 0].to(F64)
        for k, v in s.items():
            acc.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in acc.items()}


def logits_report(o: torch.Tensor, e: torch.Tensor, groups: dict[str, tuple[int, int]], block: int = 64) -> dict:
    s = logits_rows(o, e, block)
    out = {}
    for g, (lo, hi) in groups.items():
        r = summarize(s, lo, hi)
        r.update(rows=hi - lo, kl_mean=float(s["kl"][lo:hi].mean()), kl_max=float(s["kl"][lo:hi].max()),
                 argmax_equal=int(s["argmax_equal"][lo:hi].sum()), top2_equal=int(s["top2_equal"][lo:hi].sum()),
                 oracle_margin_min=float(s["oracle_margin"][lo:hi].min()))
        out[g] = r
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, math.inf]
    m = s["oracle_margin"]
    out["margin_bins"] = [{"lo": lo, "hi": hi, "rows": int(((m >= lo) & (m < hi)).sum()),
                           "argmax_differs": int(((m >= lo) & (m < hi) & (s["argmax_equal"] == 0)).sum())}
                          for lo, hi in zip(edges[:-1], edges[1:])]
    order = torch.argsort(s["rel"], descending=True)[:5].tolist()
    out["worst_rows_by_rel_l2"] = [{"row": i, "rel_l2": float(s["rel"][i]), "cos": float(s["cos"][i]), "maxabs_rms": float(s["maxabs_rms"][i]),
                                    "oracle_margin": float(s["oracle_margin"][i]), "argmax_equal": bool(s["argmax_equal"][i])} for i in order]
    out["argmax_differs"] = [
        {"row": int(i), "oracle_top1": int(s["oracle_top1"][i]), "engine_top1": int(s["engine_top1"][i]),
         "oracle_margin": float(s["oracle_margin"][i]), "engine_margin": float(s["engine_margin"][i]),
         "oracle_gap_to_engine_pick": float(s["oracle_gap_to_engine_pick"][i])}
        for i in torch.nonzero(s["argmax_equal"] == 0).flatten().tolist()]
    out["top2_order_differs"] = [
        {"row": int(i), "oracle_margin": float(s["oracle_margin"][i]), "engine_margin": float(s["engine_margin"][i])}
        for i in torch.nonzero((s["top2_equal"] == 0) & (s["argmax_equal"] == 1)).flatten().tolist()]
    return out


def engine_margin(logits: torch.Tensor, bias: torch.Tensor, k: int) -> torch.Tensor:
    """The k-th minus (k+1)-th selection score, sigmoid(logit) + bias in float64, per row of [R, E]."""
    sel = torch.sigmoid(logits.to(F64)) + bias.to(F64)
    top = sel.topk(k + 1, dim=-1).values
    return top[:, k - 1] - top[:, k]


def router_report(o_idx: torch.Tensor, e_idx: torch.Tensor, o_margin: torch.Tensor | None, groups: dict[str, tuple[int, int]],
                  o_w: torch.Tensor | None = None, e_w: torch.Tensor | None = None,
                  e_logits: torch.Tensor | None = None, bias_for=None, k: int | None = None,
                  input_equal: torch.Tensor | None = None) -> dict:
    """Router picks of two sides, [P, R, k] indices: per visit the number of rows whose sets differ (a flip). With `o_margin`
    ([P, R]) every flip carries the oracle's selection margin; with `e_logits` ([P, R, E]) and `bias_for(visit)` the engine's
    too. `input_equal` ([R] bool): the rows whose visit-0 input (the embedding) is bitwise equal on the two sides."""
    oi, ei = o_idx.long().sort(-1).values, e_idx.long().sort(-1).values
    same = (oi == ei).all(-1)  # [P, R]
    flips = []
    for v, r in torch.nonzero(~same).tolist():
        f = {"visit": v, "row": r}
        if o_margin is not None:
            f["oracle_margin"] = float(o_margin[v, r])
        if e_logits is not None and bias_for is not None:
            f["engine_margin"] = float(engine_margin(e_logits[v, r:r + 1], bias_for(v), k or oi.shape[-1])[0])
        flips.append(f)
    rep: dict = {"pairs": int(same.numel()), "flips": len(flips), "identical_fraction": float(same.double().mean()),
                 "flips_per_visit": [int((~same[v]).sum()) for v in range(same.shape[0])],
                 "flips_by_group": {g: int((~same[:, lo:hi]).sum()) for g, (lo, hi) in groups.items()}, "flip_list": flips}
    if o_margin is not None:
        m = o_margin.to(F64)
        v, r = divmod(int(m.argmin()), m.shape[1])
        rep["oracle_margin_min"] = {"value": float(m.min()), "visit": v, "row": r}
        edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, math.inf]
        rep["margin_bins"] = [{"lo": lo, "hi": hi, "pairs": int(((m >= lo) & (m < hi)).sum()),
                               "flips": int(((m >= lo) & (m < hi) & ~same).sum())} for lo, hi in zip(edges[:-1], edges[1:])]
    if input_equal is not None:
        # visit 0 reads the embedding only: a flip on a row whose embedding is bitwise equal cannot come from earlier passes
        rep["visit0_flips_on_equal_input"] = int((~same[0] & input_equal).sum())
        rep["visit0_flips"] = int((~same[0]).sum())
        rep["visit0_rows_with_unequal_input"] = int((~input_equal).sum())
    if o_w is not None and e_w is not None:
        # weights of identical sets, paired by expert id
        ob = torch.zeros(o_w.shape[:-1] + (int(max(o_idx.max(), e_idx.max())) + 1,), dtype=F64)
        eb = torch.zeros_like(ob)
        ob.scatter_(-1, o_idx.long(), o_w.to(F64))
        eb.scatter_(-1, e_idx.long(), e_w.to(F64))
        dw = (ob - eb).abs().amax(-1)
        rep["weights_max_abs_diff_identical_sets"] = float(dw[same].max()) if same.any() else None
    return rep


def compare_planes(oracle: dict[str, torch.Tensor], engine: dict[str, torch.Tensor], prompt_len: int,
                   bias_for=None, k: int | None = None, block: int = 64) -> dict:
    """Everything the two sides share. The oracle's rows are cut to the engine's (a dump that prefilled fewer)."""
    rows = int(engine["logits"].shape[0])
    if not torch.equal(oracle["tokens"][:rows], engine["tokens"][:rows]):
        raise ValueError("the two sides were run on different token ids")
    if oracle["logits"].shape[0] < rows:
        raise ValueError(f"the oracle has {oracle['logits'].shape[0]} rows, the engine {rows}")
    groups = groups_for(rows, prompt_len)
    rep: dict = {"rows": rows, "prompt_len": prompt_len, "groups": {g: list(r) for g, r in groups.items()}}
    rep["logits"] = logits_report(oracle["logits"][:rows], engine["logits"], groups, block)
    if "hidden" in engine:
        rep["hidden"] = plane_report(oracle["hidden"][:, :rows], engine["hidden"], groups)
    if "core_out" in engine and "core_out" in oracle:
        rep["core_out"] = plane_report(oracle["core_out"][:, :rows], engine["core_out"], groups)
    if "h_final" in engine:
        rep["h_final"] = plane_report(oracle["h_final"][None, :rows], engine["h_final"][None], groups)
    if "router_idx" in engine and "router_idx" in oracle:
        rep["router"] = router_report(
            oracle["router_idx"][:, :rows], engine["router_idx"], oracle["router_margin"][:, :rows] if "router_margin" in oracle else None,
            groups, oracle.get("router_w", None) if "router_w" in oracle else None, engine.get("router_w"),
            engine.get("router_logits"), bias_for, k,
            input_equal=(oracle["hidden"][0, :rows] == engine["hidden"][0]).all(-1) if "hidden" in engine else None)
    return rep


# ----------------------------------------------------------------------------------------------------------------------
# the CLI


def gate_bias(pack: str):
    """`(bias_for, k, cfg)`: the router's `mlp.gate.bias` of the layer a visit runs, float64, from the pack."""
    from inkling2_ref import Inkling2LoopConfig
    from pack import PackIndex, PackWeights

    with open(os.path.join(pack, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    w = PackWeights(PackIndex(pack))
    visits = cfg.visits()
    cache: dict[int, torch.Tensor] = {}

    def bias_for(v: int) -> torch.Tensor:
        layer = visits[v].layer
        if layer not in cache:
            cache[layer] = w.tensor(f"model.llm.layers.{layer}.mlp.gate.bias").to(F64)
        return cache[layer]

    return bias_for, cfg.top_k, cfg


def load_side(path: str, tokens: list[int] | None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """A fixture or dump file, or a dump directory (needs `tokens`: which chunks)."""
    if os.path.isdir(path):
        if tokens is None:
            raise ValueError("a dump directory needs the token ids of the prompt to find its chunks")
        return assemble(path, tokens), {}
    return load_fixture(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", required=True, help="the fixture the engine is measured against")
    ap.add_argument("--engine", required=True, help="a dump directory (chunks found by the oracle's token ids), a dump file or a second fixture")
    ap.add_argument("--rows", type=int, help="compare the first ROWS rows only (default: every row the engine side holds)")
    ap.add_argument("--pack", help="the pack: reads the router bias to give the engine's own selection margin at a flip")
    ap.add_argument("--stock-gates", action="store_true",
                    help="also run compare.py's drift gates (the set of the oracle fixture's `emulate`) on the same planes: the engine dump's "
                         "tokens are the fixture's, so its rows are the fixture's rows less the last token's")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    o, meta = load_fixture(a.oracle)
    tokens = [int(t) for t in o["tokens"]]
    e, _ = load_side(a.engine, tokens[:-1])
    if a.rows:
        e = {k: (v[:, :a.rows] if ROW_AXIS.get(k) == 1 else v[:a.rows] if ROW_AXIS.get(k) == 0 else v) for k, v in e.items()}
    bias_for, k = (None, None)
    if a.pack:
        bias_for, k, _ = gate_bias(a.pack)
    rep = compare_planes(o, e, int(meta["prompt_len"]), bias_for, k)
    if a.stock_gates:
        import compare

        which = meta.get("emulate", "none")
        e_full = dict(e, tokens=o["tokens"])
        rep["stock_gates"] = {"gate_set": which, "report": compare.compare(o, e_full, int(meta["prompt_len"]), compare.gates_for(which))}
    rep["oracle"] = {"path": os.path.basename(a.oracle), "emulate": meta.get("emulate"), "numerics_version": meta.get("numerics_version")}
    rep["engine"] = {"path": os.path.basename(a.engine.rstrip("/")), "is_dump": os.path.isdir(a.engine)}
    txt = json.dumps(rep, indent=1)
    if a.json:
        with open(a.json, "w") as f:
            f.write(txt + "\n")
    lg, al = rep["logits"], rep["logits"]["all"]
    print(f"rows {rep['rows']}  logits: argmax equal {al['argmax_equal']}/{al['rows']}  top-2 equal {al['top2_equal']}/{al['rows']}  "
          f"cos min {al['cos_row_min']:.7f}  rel L2 max {al['rel_l2_row_max']:.3e}  KL mean {al['kl_mean']:.3e}")
    if "hidden" in rep:
        h = rep["hidden"]["all"]
        w = max(range(len(h)), key=lambda i: h[i]["rel_l2_plane"])
        print(f"hidden: worst visit {w} rel L2 {h[w]['rel_l2_plane']:.3e} (row max {h[w]['rel_l2_row_max']:.3e}, cos min {h[w]['cos_row_min']:.7f})")
    if "router" in rep:
        r = rep["router"]
        print(f"router: {r['flips']} flips of {r['pairs']} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
