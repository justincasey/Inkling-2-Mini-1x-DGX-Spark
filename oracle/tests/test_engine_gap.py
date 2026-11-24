"""engine_gap.py on small synthetic planes: exact statistics of a known perturbation, argmax and top-2 bookkeeping, router
flips with their margins, the groups of a prompt past the window, and the assembly of a server's dump directory (chunks of
several prompts, a prompt cut into chunks)."""

import json
import math
import os

import pytest
import torch

import engine_gap as eg
import st

P, R, H, V, K, E = 5, 12, 8, 30, 3, 6


def fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(R)]) for _ in range(P)]).to(torch.int32)
    return {
        "tokens": torch.arange(R + 1) + 5,
        "logits": torch.randn(R, V, generator=g),
        "hidden": torch.randn(P + 1, R, H, generator=g).to(torch.bfloat16),
        "core_out": torch.randn(2, R, H, generator=g).to(torch.bfloat16),
        "h_final": torch.randn(R, H, generator=g).to(torch.bfloat16),
        "router_idx": idx,
        "router_w": torch.rand(P, R, K, generator=g),
        "router_margin": torch.rand(P, R, generator=g) + 0.5,
    }


def test_identical_planes_have_no_gap():
    o = fixture()
    rep = eg.compare_planes(o, dict(o), prompt_len=8)
    al = rep["logits"]["all"]
    assert al["argmax_equal"] == R and al["top2_equal"] == R and al["kl_max"] == 0
    assert al["rel_l2_plane"] == 0 and al["cos_row_min"] == pytest.approx(1.0) and al["maxabs_rms_row_max"] == 0
    assert rep["logits"]["argmax_differs"] == []
    assert all(s["rel_l2_plane"] == 0 for s in rep["hidden"]["all"]) and len(rep["hidden"]["all"]) == P + 1
    assert rep["router"]["flips"] == 0 and rep["router"]["identical_fraction"] == 1.0
    assert rep["router"]["weights_max_abs_diff_identical_sets"] == 0
    assert rep["groups"] == {"all": [0, R], "prompt": [0, 7], "generated": [7, R]}


def test_scaled_plane_statistics_are_exact():
    # e = 1.01 o: relative L2 0.01 everywhere, cosine 1, largest difference 0.01 |o|max over the row's RMS
    o = torch.randn(2, 6, 16, dtype=torch.float64)
    rep = eg.plane_report(o, o * 1.01, {"all": (0, 6), "tail": (4, 6)})
    for g in ("all", "tail"):
        for s in rep[g]:
            assert s["rel_l2_plane"] == pytest.approx(0.01) and s["rel_l2_row_max"] == pytest.approx(0.01)
            assert s["cos_row_min"] == pytest.approx(1.0)
    row_rms = o[0, 0].norm() / 4
    assert eg.row_stats(o[0], o[0] * 1.01)["maxabs_rms"][0] == pytest.approx(float(0.01 * o[0, 0].abs().max() / row_rms))


def test_plane_relative_l2_adds_squares_not_row_ratios():
    o = torch.zeros(1, 2, 4, dtype=torch.float64)
    o[0, 0] = 1.0  # norm 2
    o[0, 1] = 10.0  # norm 20
    e = o.clone()
    e[0, 0] += 1.0  # row error 2 (rel 1.0)
    rep = eg.plane_report(o, e, {"all": (0, 2)})["all"][0]
    assert rep["rel_l2_row_max"] == pytest.approx(1.0) and rep["rel_l2_row_median"] == pytest.approx(0.5)
    assert rep["rel_l2_plane"] == pytest.approx(math.sqrt(4 / (4 + 400)))


def test_argmax_and_top2_bookkeeping():
    o = torch.zeros(3, 5, dtype=torch.float64)
    o[0, :3] = torch.tensor([5.0, 4.0, 0.0])  # top-2 (0, 1), margin 1
    o[1, :3] = torch.tensor([5.0, 4.9, 0.0])  # top-2 (0, 1), margin 0.1
    o[2, :3] = torch.tensor([5.0, 1.0, 4.0])  # top-2 (0, 2)
    e = o.clone()
    e[1, :2] = torch.tensor([4.9, 5.0])  # argmax flips at the near tie
    e[2, 1:3] = torch.tensor([4.0, 1.0])  # runner-up swaps, argmax holds
    s = eg.logits_rows(o, e, block=2)
    assert s["argmax_equal"].tolist() == [1, 0, 1] and s["top2_equal"].tolist() == [1, 0, 0]
    rep = eg.logits_report(o, e, {"all": (0, 3)}, block=2)
    (d,) = rep["argmax_differs"]
    assert d["row"] == 1 and d["oracle_top1"] == 0 and d["engine_top1"] == 1
    assert d["oracle_margin"] == pytest.approx(0.1) and d["oracle_gap_to_engine_pick"] == pytest.approx(0.1)
    assert [x["row"] for x in rep["top2_order_differs"]] == [2]
    bins = {b["lo"]: (b["rows"], b["argmax_differs"]) for b in rep["margin_bins"]}
    assert bins[0.05] == (1, 1) and bins[1.0] == (2, 0)  # row 1 (margin just under 0.1) differs; rows 0 and 2 have margin 1
    assert sum(b["rows"] for b in rep["margin_bins"]) == 3 and sum(b["argmax_differs"] for b in rep["margin_bins"]) == 1
    w = rep["worst_rows_by_rel_l2"]
    assert len(w) == 3 and w[0]["rel_l2"] >= w[1]["rel_l2"] >= w[2]["rel_l2"] and {x["row"] for x in w} == {0, 1, 2}
    assert rep["all"]["argmax_equal"] == 2 and rep["all"]["top2_equal"] == 1


def test_kl_is_zero_for_equal_and_matches_the_definition():
    o = torch.randn(4, 9, dtype=torch.float64)
    e = o + 0.3 * torch.randn(4, 9, dtype=torch.float64)
    s = eg.logits_rows(o, e)
    p, q = torch.softmax(o, -1), torch.softmax(e, -1)
    assert torch.allclose(s["kl"], (p * (p.log() - q.log())).sum(-1))
    assert eg.logits_rows(o, o)["kl"].abs().max() == 0


def test_router_flip_carries_both_margins():
    o = fixture()
    e = dict(o)
    idx = o["router_idx"].clone()
    row_set = set(idx[2, 4].tolist())
    out = next(x for x in range(E) if x not in row_set)
    idx[2, 4, 0] = out  # one expert replaced in visit 2, row 4
    e["router_idx"] = idx
    # the engine's logits: expert `out` is the 3rd, the replaced one the 4th best (bias zero), margin = sigmoid gap
    logits = torch.full((P, R, E), -3.0)
    logits[2, 4, out], logits[2, 4, int(o["router_idx"][2, 4, 0])] = 1.0, 0.5
    e["router_logits"] = logits
    bias = torch.zeros(E, dtype=torch.float64)
    rep = eg.router_report(o["router_idx"], idx, o["router_margin"], {"all": (0, R), "late": (8, R)}, o["router_w"], o["router_w"],
                           logits, lambda v: bias, K)
    assert rep["flips"] == 1 and rep["pairs"] == P * R and rep["flips_per_visit"] == [0, 0, 1, 0, 0]
    (f,) = rep["flip_list"]
    assert (f["visit"], f["row"]) == (2, 4) and f["oracle_margin"] == pytest.approx(float(o["router_margin"][2, 4]))
    assert rep["flips_by_group"] == {"all": 1, "late": 0}
    # the engine's margin: scores are 1.0 (out), 0.5, and -3.0 for four others: the 3rd best (-3) vs the 4th (-3): a tie of zero
    assert f["engine_margin"] == pytest.approx(0.0, abs=1e-12)
    assert rep["oracle_margin_min"]["value"] == pytest.approx(float(o["router_margin"].min()))
    assert sum(b["pairs"] for b in rep["margin_bins"]) == P * R and sum(b["flips"] for b in rep["margin_bins"]) == 1


def test_margin_bins_and_visit0_input_check():
    o = fixture(1)
    idx = o["router_idx"].clone()
    margin = torch.full((P, R), 0.5)
    margin[1, 2], margin[3, 5], margin[0, 7] = 5e-6, 5e-4, 5e-4  # two pairs in [1e-4, 1e-3), one in [1e-6.., 1e-5)
    for v, r in ((1, 2), (3, 5), (0, 7)):
        idx[v, r, 0] = next(x for x in range(E) if x not in set(o["router_idx"][v, r].tolist()))
    equal = torch.ones(R, dtype=torch.bool)
    equal[7] = False  # the embedding of row 7 differs: its visit-0 flip is explained
    rep = eg.router_report(o["router_idx"], idx, margin, {"all": (0, R)}, input_equal=equal)
    bins = {(b["lo"], b["hi"]): (b["pairs"], b["flips"]) for b in rep["margin_bins"]}
    assert bins[(1e-6 * 0 + 0.0, 1e-5)] == (1, 1) and bins[(1e-4, 1e-3)] == (2, 2) and bins[(1e-1, math.inf)] == (P * R - 3, 0)
    assert rep["visit0_flips"] == 1 and rep["visit0_flips_on_equal_input"] == 0 and rep["visit0_rows_with_unequal_input"] == 1
    equal[7] = True
    assert eg.router_report(o["router_idx"], idx, margin, {"all": (0, R)}, input_equal=equal)["visit0_flips_on_equal_input"] == 1


def test_rows_differing_counts_rows_with_any_error():
    o = torch.randn(1, 5, 4, dtype=torch.float64)
    e = o.clone()
    e[0, 1, 2] += 1e-9
    e[0, 3, 0] -= 1.0
    assert eg.plane_report(o, e, {"all": (0, 5), "head": (0, 2)}) ["all"][0]["rows_differing"] == 2
    assert eg.plane_report(o, e, {"head": (0, 2)})["head"][0]["rows_differing"] == 1


def test_engine_margin_is_kth_minus_next_score():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64)
    bias = torch.tensor([0.0, 0.0, 0.0, -2.0], dtype=torch.float64)
    sel = torch.sigmoid(logits[0]) + bias  # scores of experts 0..3
    top = sel.sort(descending=True).values
    assert eg.engine_margin(logits, bias, 2)[0] == pytest.approx(float(top[1] - top[2]))


def test_router_weights_compare_by_expert_id_not_slot():
    idx = torch.tensor([[[0, 2, 4]]], dtype=torch.int32)
    w = torch.tensor([[[0.1, 0.2, 0.3]]])
    perm = torch.tensor([[[4, 0, 2]]], dtype=torch.int32)
    wp = torch.tensor([[[0.3, 0.1, 0.2 + 1e-3]]])
    rep = eg.router_report(idx, perm, None, {"all": (0, 1)}, w, wp)
    assert rep["flips"] == 0 and rep["weights_max_abs_diff_identical_sets"] == pytest.approx(1e-3, rel=1e-4)


def test_groups_past_the_window():
    g = eg.groups_for(1126, 1119)
    assert g["all"] == (0, 1126) and g["prompt"] == (0, 1118) and g["generated"] == (1118, 1126)
    assert g["before_window"] == (0, 1024) and g["after_window"] == (1024, 1126) and g["edge_1024"] == (1016, 1032)
    assert g["tail"] == (1110, 1126)
    assert "before_window" not in eg.groups_for(49, 26)


# ------------------------------------------------------------------------------------------------ dump directories


def write_chunk(d, n, tokens, first, v=3, h=4, vocab=7, k=2, marker=0.0):
    rows = len(tokens)
    planes = {
        "hidden": torch.full((v + 1, rows, h), marker + first, dtype=torch.float32).to(torch.bfloat16),
        "core_out": torch.zeros(2, rows, h, dtype=torch.bfloat16),
        "h_final": torch.arange(first, first + rows, dtype=torch.float32)[:, None].expand(rows, h).to(torch.bfloat16),
        "logits": torch.arange(first, first + rows, dtype=torch.float32)[:, None].expand(rows, vocab).contiguous(),
        "router_idx": torch.zeros(v, rows, k, dtype=torch.int32),
        "attn_core": torch.zeros(v, rows, 5, dtype=torch.bfloat16),  # an ops plane: not carried over
    }
    stem = os.path.join(d, f"prefill{n:05d}")
    st.write(stem + ".safetensors", planes)
    with open(stem + ".json", "w") as f:
        json.dump({"slot": 0, "first": first, "rows": rows, "segments": 1, "tokens": tokens, "ops": True}, f)


def test_assemble_finds_the_chunks_of_one_prompt_among_others(tmp_path):
    d = str(tmp_path)
    a = [10, 11, 12, 13, 14]
    b = [20, 21, 22, 23, 24, 25, 26]
    write_chunk(d, 0, a, 0)  # prompt a, one chunk
    write_chunk(d, 1, b[:3], 0)  # prompt b, three chunks
    write_chunk(d, 2, b[3:5], 3)
    write_chunk(d, 3, b[5:], 5)
    write_chunk(d, 4, a[:3], 0)  # a prefix of a, sent again later
    ea = eg.assemble(d, a)
    assert ea["logits"].shape == (5, 7) and ea["hidden"].shape == (4, 5, 4) and ea["tokens"].tolist() == a
    assert "attn_core" not in ea
    eb = eg.assemble(d, b)
    assert eb["logits"][:, 0].tolist() == [0, 1, 2, 3, 4, 5, 6]  # row order kept over the chunk files
    assert eb["h_final"][:, 0].tolist() == [0, 1, 2, 3, 4, 5, 6] and eb["hidden"].shape == (4, 7, 4)
    assert eb["router_idx"].shape == (3, 7, 2)
    assert eg.assemble(d, a[:3])["logits"].shape[0] == 3
    with pytest.raises(ValueError, match="no run of dump chunks"):
        eg.assemble(d, a + [99])
    with pytest.raises(ValueError, match="no run of dump chunks"):
        eg.assemble(d, [10, 11, 13])


def test_assemble_takes_the_last_of_two_identical_runs(tmp_path):
    d = str(tmp_path)
    t = [1, 2, 3]
    write_chunk(d, 0, t, 0, marker=1.0)
    write_chunk(d, 1, t, 0, marker=2.0)
    assert float(eg.assemble(d, t)["hidden"][0, 0, 0]) == 2.0


def test_main_end_to_end_on_fixture_files(tmp_path, capsys):
    o = fixture(3)
    meta = {"prompt_len": "9", "emulate": "none", "numerics_version": "2"}
    st.write(str(tmp_path / "o.safetensors"), o, meta)
    e = {k: v.clone() for k, v in o.items()}
    e["hidden"] = (o["hidden"].float() * 1.01).to(torch.bfloat16)
    st.write(str(tmp_path / "e.safetensors"), e)
    out = str(tmp_path / "r.json")
    assert eg.main(["--oracle", str(tmp_path / "o.safetensors"), "--engine", str(tmp_path / "e.safetensors"), "--json", out]) == 0
    rep = json.load(open(out))
    assert rep["rows"] == R and rep["oracle"]["emulate"] == "none" and "stock_gates" not in rep and rep["logits"]["all"]["argmax_equal"] == R
    assert rep["hidden"]["all"][0]["rel_l2_plane"] == pytest.approx(0.01, rel=0.05)
    assert "argmax equal 12/12" in capsys.readouterr().out


def test_stock_gates_run_on_the_same_planes(tmp_path):
    o = fixture(4)
    st.write(str(tmp_path / "o.safetensors"), o, {"prompt_len": "9", "emulate": "none", "numerics_version": "2"})
    out = str(tmp_path / "r.json")
    # an engine side equal to the oracle passes the spec gate set; its tokens are the oracle's (the dump has all but the last)
    e = {k: v for k, v in o.items() if k != "router_margin"}
    e["tokens"] = o["tokens"][:-1]
    st.write(str(tmp_path / "e.safetensors"), e)
    eg.main(["--oracle", str(tmp_path / "o.safetensors"), "--engine", str(tmp_path / "e.safetensors"), "--json", out, "--stock-gates"])
    g = json.load(open(out))["stock_gates"]
    assert g["gate_set"] == "none" and g["report"]["pass"] is True and g["report"]["checks"]["logits"]["positions"] == R
