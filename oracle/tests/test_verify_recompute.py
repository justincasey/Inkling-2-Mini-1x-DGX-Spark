"""`gen_fixtures.py --verify-recompute` / `--verify-exact`: the cached == recompute check compares the planes the right way.

Real-pack finding behind it (bench/records/2026-11-23-oracle-cacheddiff): the cached decode of a prompt and the full
recompute of the same tokens had bit-identical hidden states, bank partials and routes, and logits that differed in ONE of
9.8 million values by one fp32 ulp: the unembed's float64 product of a 1-row and a 49-row operand sums in a different order
(~5e-15 apart) and that value sat on an fp32 rounding boundary."""

import dataclasses
import json

import numpy as np
import pytest
import torch

import gen_fixtures
import inkling2_ref
from gen_fixtures import LOGIT_ULP_BOUND, compare_recompute
from inkling2_ref import Inkling2LoopRef, generate
from synthetic import random_exl3_pack, tiny_config, write_pack


@pytest.fixture(scope="module")
def pair(tiny):
    """A cached `generate` result and the full recompute of the same tokens on the tiny model."""
    _, _, make = tiny
    m = make()
    res = generate(m, [5, 17, 3, 99, 250, 7, 8], 5)
    full = m.forward(res["tokens"][:-1], record=True)
    return res, full


def nudge(x: torch.Tensor, index, ulps: int) -> torch.Tensor:
    """x (float32) with one element moved by `ulps` fp32 steps."""
    y = x.clone()
    v = y[index].item()
    for _ in range(abs(ulps)):
        v = float(np.nextafter(np.float32(v), np.float32(np.inf if ulps > 0 else -np.inf)))
    y[index] = v
    return y


def with_logits(res, logits):
    return dict(res, logits=logits)


def test_identical_runs_pass_exact_and_tolerant(pair):
    res, full = pair
    for exact in (True, False):
        c = compare_recompute(res, full, exact=exact)
        assert c["ok"] and c["n_differing_logits"] == 0 and c["logit_max_abs"] == 0.0 and c["logit_max_ulps"] == 0.0
    assert c["rows"] == full.logits.shape[0] and c["hidden_equal"] and c["routes_equal"] and c["core_out_equal"]


def test_one_ulp_logit_flip_passes_tolerant_fails_exact(pair):
    """The real-pack case: one logit one fp32 ulp apart, everything else identical."""
    res, full = pair
    row, col = 3, 17
    flipped = with_logits(res, nudge(res["logits"], (row, col), 1))
    c = compare_recompute(flipped, full)
    assert c["ok"] and c["n_differing_logits"] == 1 and c["rows_with_differing_logits"] == [row]
    assert c["logit_max_ulps"] == 1.0 and 0 < c["logit_max_abs"] < 1e-5 and c["logit_max_rel_rms"] < 1e-6
    assert not compare_recompute(flipped, full, exact=True)["ok"]


def test_bound_is_two_ulps(pair):
    res, full = pair
    assert compare_recompute(with_logits(res, nudge(res["logits"], (0, 1), 2)), full)["ok"]
    assert LOGIT_ULP_BOUND == 2
    assert not compare_recompute(with_logits(res, nudge(res["logits"], (0, 1), 3)), full)["ok"]


def test_argmax_flip_fails_even_within_the_bound(pair):
    """A tie at the top that one ulp breaks the other way: within the ulp bound, but the greedy token differs."""
    res, full = pair
    row = 2
    i0, i1 = sorted(torch.topk(full.logits[row], 2).indices.tolist())
    v = float(full.logits[row].max())
    tied = full.logits.clone()
    tied[row, i0] = tied[row, i1] = v  # exact tie: the lower index wins
    tied_full = dataclasses.replace(full, logits=tied)
    broken = nudge(tied, (row, i0), -1)  # one ulp down: the higher index now wins
    ok_row = with_logits(res, tied)
    assert compare_recompute(ok_row, tied_full)["ok"]
    c = compare_recompute(with_logits(res, broken), tied_full)
    assert c["logit_max_ulps"] == 1.0 and not c["argmax_equal"] and not c["ok"]


@pytest.mark.parametrize("plane", ["hidden", "core_out", "h_final"])
def test_bf16_planes_must_match_bit_for_bit(pair, plane):
    res, full = pair
    t = res[plane].clone()
    flat = t.view(torch.int16).flatten()
    flat[7] = flat[7] ^ 1  # one bf16 ulp in one element
    c = compare_recompute(dict(res, **{plane: t}), full)
    assert not c["ok"] and not c[f"{plane}_equal"]


def test_a_changed_route_fails(pair):
    res, full = pair
    idx = res["router_idx"].clone()
    idx[0, 0, 0] = (idx[0, 0, 0] + 1) % 4
    c = compare_recompute(dict(res, router_idx=idx), full)
    assert not c["ok"] and not c["routes_equal"]


def test_router_weight_flip_is_judged_in_ulps(pair):
    res, full = pair
    w = nudge(res["router_w"], (1, 2, 0), 1)
    c = compare_recompute(dict(res, router_w=w), full)
    assert c["ok"] and c["router_w_max_ulps"] == 1.0
    assert not compare_recompute(dict(res, router_w=w), full, exact=True)["ok"]


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pack"))
    cfg = tiny_config()
    plain, groups, _ = random_exl3_pack(cfg, seed=3)
    write_pack(root, cfg, plain, groups)
    return root


def run_main(pack_dir, tmp_path, *flags):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"name": n, "ids": ids, "new_tokens": 3}) + "\n"
                               for n, ids in (("p1", [5, 17, 3, 99, 250, 7]), ("p2", [9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 11]))))
    out = tmp_path / "fx"
    rc = gen_fixtures.main(["--pack", pack_dir, "--prompts", str(prompts), "--out", str(out), *flags])
    return rc, json.loads((out / "manifest.json").read_text()), out


def test_main_verify_recompute_records_the_check(pack_dir, tmp_path):
    rc, man, out = run_main(pack_dir, tmp_path, "--verify-recompute")
    assert rc == 0
    for name in ("p1", "p2"):
        e = man["fixtures"][name]
        assert e["verified_recompute"] is True and e["recompute_check"]["mode"] == f"<= {LOGIT_ULP_BOUND} fp32 ulp"
        assert e["recompute_check"]["hidden_equal"] and e["recompute_check"]["logit_max_ulps"] == 0.0
        assert (out / f"{name}.safetensors").exists()


def test_main_exact_mode_and_failure_keeps_fixtures_and_goes_on(pack_dir, tmp_path, monkeypatch):
    """A one-ulp logit difference in the recompute of the first prompt: `--verify-recompute` accepts it; `--verify-exact`
    marks that prompt failed in the manifest, still writes both fixtures, still runs the second prompt, and exits 1."""
    real = Inkling2LoopRef.forward
    state = {"calls": 0}

    def forward(self, ids, cache=None, record=False):
        out = real(self, ids, cache, record)
        if cache is None and record:  # a full recompute
            state["calls"] += 1
            if state["calls"] == 1:
                out.logits = nudge(out.logits, (1, 4), 1)
        return out

    monkeypatch.setattr(Inkling2LoopRef, "forward", forward)
    (tmp_path / "a").mkdir()
    rc, man, out = run_main(pack_dir, tmp_path / "a", "--verify-recompute")
    assert rc == 0 and man["fixtures"]["p1"]["recompute_check"]["n_differing_logits"] == 1
    state["calls"] = 0
    (tmp_path / "b").mkdir()
    rc, man, out = run_main(pack_dir, tmp_path / "b", "--verify-exact")
    assert rc == 1
    assert man["fixtures"]["p1"]["verified_recompute"] is False and man["fixtures"]["p2"]["verified_recompute"] is True
    assert (out / "p1.safetensors").exists() and (out / "p2.safetensors").exists()
    assert man["fixtures"]["p1"]["recompute_check"]["mode"] == "exact"


def test_numerics_version_is_unchanged_by_the_check():
    """The check changes no fixture content, so the version label of the fixtures stays the model's."""
    assert gen_fixtures.NUMERICS_VERSION == inkling2_ref.NUMERICS_VERSION
