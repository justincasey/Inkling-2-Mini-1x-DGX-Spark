"""bench/tools/oracle_gap_summary.py on synthetic reports of engine_gap.py: an engine that is as close to the oracle as the oracle's
two numerics are to each other passes every check; an argmax difference at a wide margin, a router that flips far more than the
floor, a hidden plane past the limit, and a greedy divergence at a wide margin each fail their own check and no other."""

import copy
import json
import os
import sys

import pytest
import torch

import engine_gap as eg
from test_engine_gap import R, fixture

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bench", "tools"))
import oracle_gap_summary as sm  # noqa: E402

NAME = "a"


def perturbed(o, eps, seed):
    g = torch.Generator().manual_seed(seed)
    e = {k: v for k, v in o.items() if k != "router_margin"}
    e["hidden"] = (o["hidden"].float() * (1 + eps * torch.randn(o["hidden"].shape, generator=g))).to(torch.bfloat16)
    e["logits"] = o["logits"] + 1e-2 * eps * torch.randn(o["logits"].shape, generator=g)
    e["tokens"] = o["tokens"][:-1]
    return e


def write_reports(tmp, mutate=None):
    o = fixture(7)
    # the floor: the engine fixture (a perturbed oracle) against the spec fixture; the engine: two other perturbations
    fe = perturbed(o, 0.01, 1)
    fe["router_margin"] = o["router_margin"]
    fe["tokens"] = o["tokens"]
    dump = perturbed(o, 0.01, 2)
    reps = {"engine": eg.compare_planes(fe, dump, 8), "spec": eg.compare_planes(o, dump, 8), "floor": eg.compare_planes(o, fe, 8)}
    if mutate:
        mutate(reps)
    for k, pre in sm.KINDS.items():
        json.dump(reps[k], open(os.path.join(tmp, f"{pre}{NAME}.json"), "w"))
    g = {"fixtures": {NAME: {"prompt_len": 8, "generated": 4, "equal": True, "first_divergence": None, "mismatch_positions": [],
                             "oracle_margins": [1, 2, 3, 4], "oracle_token_is_top1": [True] * 4}}}
    for k in ("spec", "engine"):
        json.dump(copy.deepcopy(g), open(os.path.join(tmp, f"greedy-{k}.json"), "w"))
    return g


def summary(tmp):
    return sm.summarize(str(tmp), [NAME])


def failed(s):
    return sorted({c["check"] for c in s["per_prompt"][NAME]["checks"] if not c["ok"]} | {c["check"] for c in s["pooled"]["checks"] if not c["ok"]})


def test_close_engine_passes_every_check(tmp_path):
    write_reports(str(tmp_path))
    s = summary(tmp_path)
    assert s["pass"] and failed(s) == [], failed(s)
    m = s["metrics"]
    assert m["prompts_compared"] == 1 and m["rows_compared"] == R and m["greedy_tokens_equal"] == m["greedy_tokens_total"] == 4
    assert m["router_pairs"] == 5 * R and m["argmax_equal_fraction"] <= 1.0
    # one test entry per group: hidden, logits, router, greedy, and the pooled router check
    assert [t["filter"] for t in s["tests"]] == [f"{NAME}:hidden", f"{NAME}:logits", f"{NAME}:router", f"{NAME}:greedy", "pooled:router_flips_vs_floor"]
    assert all(t["failed"] == 0 and t["passed"] > 0 for t in s["tests"])
    assert "| a |" in sm.tables(s) or "chat" not in sm.tables(s)


def test_an_argmax_difference_at_a_wide_margin_fails_only_its_check(tmp_path):
    def mutate(reps):
        reps["spec"]["logits"]["argmax_differs"] = [{"row": 3, "oracle_top1": 1, "engine_top1": 2, "oracle_margin": 0.9, "engine_margin": 0.1,
                                                     "oracle_gap_to_engine_pick": 0.9}]
    write_reports(str(tmp_path), mutate)
    s = summary(tmp_path)
    assert not s["pass"] and failed(s) == ["logits:argmax_difference_gap"]
    ok = copy.deepcopy(s)  # the same difference at a near tie is excused
    mutate2 = lambda reps: mutate(reps) or reps["spec"]["logits"]["argmax_differs"][0].update(oracle_gap_to_engine_pick=0.2)  # noqa: E731
    write_reports(str(tmp_path), mutate2)
    assert failed(summary(tmp_path)) == [] and ok["pass"] is False


def test_router_flips_far_above_the_floor_fail(tmp_path):
    def mutate(reps):
        r = reps["spec"]["router"]
        reps["floor"]["router"]["flips"] = 10
        r["flips"] = 10 * 3 + 20
    write_reports(str(tmp_path), mutate)
    assert "router:flips_vs_floor_pooled" in failed(summary(tmp_path))


def test_a_hidden_plane_past_the_limit_and_a_greedy_divergence_fail(tmp_path):
    def mutate(reps):
        reps["engine"]["hidden"]["all"][4]["rel_l2_plane"] = 0.9
    g = write_reports(str(tmp_path), mutate)
    assert failed(summary(tmp_path)) == ["hidden:hidden_rel_l2_plane_max"]
    g["fixtures"][NAME].update(equal=False, first_divergence=2, mismatch_positions=[2], oracle_margin=0.8)
    json.dump(g, open(os.path.join(str(tmp_path), "greedy-engine.json"), "w"))
    assert failed(summary(tmp_path)) == ["greedy:first_divergence_margin", "hidden:hidden_rel_l2_plane_max"]


def test_main_writes_summary_and_tables(tmp_path, capsys):
    write_reports(str(tmp_path))
    out, tab = str(tmp_path / "s.json"), str(tmp_path / "t.md")
    assert sm.main(["--dir", str(tmp_path), "--names", NAME, "--out", out, "--tables", tab]) == 0
    assert json.load(open(out))["pass"] is True and open(tab).read().startswith("| prompt |")
    assert "pass (" in capsys.readouterr().out
