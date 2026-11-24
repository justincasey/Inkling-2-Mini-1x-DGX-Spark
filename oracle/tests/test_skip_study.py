"""`skip_study.py`'s statistics on hand-made planes: the pass-to-layer map, the pick / mass / byte arithmetic of the rule, the union rule of a
window, KL, flips and the route-flip count."""

import json

import pytest
import torch

import skip_study as ss
from inkling2_ref import F64


def test_tags_and_layer_map():
    assert ss.parse_tag("t0.2-renorm") == (0.2, True, False) and ss.parse_tag("t0.15-plain") == (0.15, False, False)
    assert ss.parse_tag("d0.3-renorm") == (0.3, True, True)
    with pytest.raises(ValueError):
        ss.parse_tag("0.2")
    # prelude 0-5, the core 6-17 three times (passes 6-41), the coda 18-23 (passes 42-47)
    assert [ss.layer_of(p) for p in (0, 5, 6, 17, 18, 29, 30, 41, 42, 47)] == [0, 5, 6, 17, 6, 17, 6, 17, 18, 23]


def census(tmp_path, per_expert):
    mods = {}
    for (layer, e), b in per_expert.items():
        for proj in ("gate", "up", "down"):
            mods[f"model.llm.layers.{layer}.mlp.experts.{e}.{proj}_proj"] = {"bytes": b}
    mods["model.llm.layers.0.mlp.gate"] = {"bytes": 999}  # the router is not an expert
    p = tmp_path / "census.json"
    p.write_text(json.dumps({"modules": mods}))
    return ss.expert_bytes(str(p))


def planes(idx, w):
    return {"router_idx": torch.tensor(idx, dtype=torch.int32), "router_w": torch.tensor(w, dtype=torch.float32)}


def test_expert_bytes_sums_gate_up_down_and_ignores_the_router(tmp_path):
    eb = census(tmp_path, {(0, 0): 10, (0, 1): 20})
    assert eb == {(0, 0): 30, (0, 1): 60}


def test_drops_counts_picks_mass_and_bytes(tmp_path):
    eb = census(tmp_path, {(l, e): 1 + e for l in range(24) for e in range(4)})  # bytes of an expert a layer: 3 (1 + e)
    # 48 passes, 2 rows, k = 4; row 0 weights (sum 8): top 4.0, t = 0.25 drops weights < 1.0; row 1 keeps everything
    row0, row1 = [4.0, 2.0, 1.5, 0.5], [2.0, 2.0, 2.0, 2.0]
    pl = planes([[[0, 1, 2, 3], [0, 1, 2, 3]]] * 48, [[row0, row1]] * 48)
    d = ss.drops(pl, 0.25, eb)
    assert d["picks"] == 48 * 8 and d["picks_dropped"] == 48 and d["mass"] == 48 * 16 and d["mass_dropped"] == 48 * 0.5
    # the dropped pick is expert 3 (3 * (1 + 3) = 12 bytes of 3 * (1 + 2 + 3 + 4) = 30 a row)
    assert d["bytes"] == 48 * 60 and d["bytes_dropped"] == 48 * 12


def test_window_union_rule_reads_an_expert_once_while_any_row_keeps_it(tmp_path):
    eb = census(tmp_path, {(l, e): 1 for l in range(24) for e in range(4)})  # 3 bytes an expert
    # two rows pick experts (0,1) and (0,2) with weights (4, 0.5): at t = 0.25 each row drops its second pick, expert 0 is kept by both,
    # experts 1 and 2 are read by nobody; the union of the window is {0, 1, 2} -> kept {0}
    pl = planes([[[0, 1], [0, 2]]] * 48, [[[4.0, 0.5], [4.0, 0.5]]] * 48)
    r = ss.window_saved({"x": pl}, 0.25, eb, rows=2, streams=1, n_windows=5)
    assert r["bytes"] == 5 * 48 * 9 and r["bytes_kept"] == 5 * 48 * 3 and r["saved"] == pytest.approx(2 / 3)
    # a second row that keeps expert 1 (weights 1, 4) keeps it for the window: union {0, 1}, kept {0, 1}
    pl = planes([[[0, 1], [0, 1]]] * 48, [[[4.0, 0.5], [0.5, 4.0]]] * 48)
    r = ss.window_saved({"x": pl}, 0.25, eb, rows=2, streams=1, n_windows=3)
    assert r["saved"] == 0.0


def test_kl_flips_and_route_flips():
    a = {"logits": torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 1.9], [1.0, 1.0, 5.0]]), "router_idx": torch.tensor([[[0, 1], [2, 3], [4, 5]]] * 3)}
    x = {"logits": a["logits"].clone(), "router_idx": a["router_idx"].clone()}
    assert ss.kl_rows(a["logits"], x["logits"]).abs().max() == 0
    x["logits"][1] = torch.tensor([0.0, 1.9, 2.0])  # argmax flips at row 1 (margin 0.1)
    x["router_idx"][2, 0] = torch.tensor([1, 0])  # same set: not a flip
    x["router_idx"][1, 2] = torch.tensor([4, 9])  # another set
    rep = ss.logits_report(a, x, plen=2)
    assert rep["argmax_equal"] == 2 and len(rep["flips"]) == 1
    f = rep["flips"][0]
    assert f["row"] == 1 and f["generated"] is True and f["a_margin"] == pytest.approx(0.1) and f["a_gap_to_pick"] == pytest.approx(0.1)
    assert rep["kl"]["rows"] == 3 and rep["kl_prompt"]["rows"] == 1 and rep["kl_generated"]["rows"] == 2 and rep["kl"]["max"] > 0
    rf = ss.route_flips(x, a)
    assert rf["per_visit"] == [0, 1, 0] and rf["total"] == 1 and rf["after_visit0"] == 1 and rf["pairs"] == 9
    assert ss.rel(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 0.0])) == pytest.approx(4.0 / 3.0)
    assert ss.hidden_curve(torch.ones(3, 2, 2), torch.ones(3, 2, 2)) == [0.0, 0.0, 0.0]
    assert F64 is torch.float64


# ---------------------------------------------------------------------------------------------------------- the forward driver


def test_forward_driver_all_rows_and_decode_only(tmp_path):
    """On a synthetic EXL3 pack: a run through the skip path that drops nothing equals the engine fixture bit for bit (control); decode-only
    skipping leaves the prefill rows (< prompt_len) the A lane's and changes only the decode rows; all-rows skipping changes the prompt rows too."""
    import argparse

    import numpy as np

    import st
    from gen_fixtures import fixture_tensors
    from inkling2_ref import Emulation, Inkling2LoopRef, generate
    from pack import PackExperts, PackIndex, PackWeights
    from synthetic import random_exl3_pack, tiny_config, write_pack

    root = str(tmp_path / "pack")
    cfg = tiny_config(n_experts=8, top_k=4)
    plain, groups, _ = random_exl3_pack(cfg, seed=2)
    write_pack(root, cfg, plain, groups)
    ix = PackIndex(root)
    experts = PackExperts(ix, workers=0)
    model = Inkling2LoopRef(cfg, PackWeights(ix), experts, unembed_chunk=100, emulate=Emulation.engine())
    prompt = np.random.default_rng(4).integers(0, 250, 7).tolist()
    res = generate(model, prompt, 5)
    fx = str(tmp_path / "fx.safetensors")
    st.write(fx, fixture_tensors(cfg, res), {"prompt_len": "7", "emulate": "engine"})

    def run(emulate, decode_only):
        out = str(tmp_path / f"run-{abs(hash((emulate, decode_only)))}.safetensors")
        a = argparse.Namespace(pack=root, fixture=fx, emulate=emulate, out=out, workers=0, threads=1, weight_cache_gib=0.0,
                               decode_only=decode_only, rows=0)
        assert ss.cmd_forward(a) == 0
        return ss.load(out)

    ref_planes = ss.load(fx)
    for decode_only in (False, True):  # nothing dropped: the A lane, bit for bit
        got = run("engine,expert_skip=1e-9,skip_renorm", decode_only)
        for k in ("hidden", "router_idx", "router_w", "h_final"):
            assert torch.equal(got[k], ref_planes[k]), (k, decode_only)
        assert torch.equal(got["logits"], ref_planes["logits"])
    # the tiny model's top-4 weights are within a factor of two: t = 0.9 drops picks in most rows
    dec, allr = run("engine,expert_skip=0.9,skip_renorm", True), run("engine,expert_skip=0.9,skip_renorm", False)
    assert torch.equal(dec["hidden"][:, :7], ref_planes["hidden"][:, :7]) and torch.equal(dec["logits"][:7], ref_planes["logits"][:7])
    assert not torch.equal(dec["logits"][7:], ref_planes["logits"][7:])
    assert not torch.equal(allr["hidden"][1:, :7], ref_planes["hidden"][1:, :7])
    # the summary's slices: decode-only rows start at prompt_len
    rep = ss.logits_report({"logits": ref_planes["logits"]}, {"logits": dec["logits"]}, plen=7, start=7)
    assert rep["rows"] == 4 and rep["kl"]["rows"] == 4 and rep["kl_prompt"]["rows"] == 0 and min(f["row"] for f in rep["flips"] or [{"row": 7}]) >= 7
