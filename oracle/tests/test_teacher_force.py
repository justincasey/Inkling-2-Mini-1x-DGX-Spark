"""teacher_force.py, compare.py and gen_fixtures.py on the tiny looped model (8 layers, AttnRes blocks of 2, the core [2, 6)
x 3 = 16 layer passes; global layers 1, 3, 5, 7; synthetic EXL3 pack): a dump the emulating oracle itself wrote passes
every op gate exactly; an altered plane fails in its own op and nowhere else; the AttnRes blocks and the message bank of
a pass are rebuilt from the schedule exactly as the oracle forms them; the fixture writer writes the documented tensors."""

import json
import os

import numpy as np
import pytest
import torch

import compare
import gen_fixtures
import st
import teacher_force as tf
from inkling2_ref import DictWeights, Emulation, Inkling2LoopRef, generate
from pack import PackIndex, PackWeights
from synthetic import ExpertGroups, random_exl3_pack, tiny_config, write_pack

SEED = 2
T = 13
V = 16  # layer passes of the tiny config: 8 layers + 2 * 4 core layers


def ids(n, seed):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 250, n))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """cfg, the pack folder, the in-memory weights and expert groups, the emulating oracle and its own dump."""
    cfg = tiny_config()
    plain, groups, reference = random_exl3_pack(cfg, seed=SEED)
    root = str(tmp_path_factory.mktemp("pack"))
    write_pack(root, cfg, plain, groups)
    # the pack folder's weights: the engine emulation includes `dense_fp16`, which needs the EXL3 groups (bit-equal twin of `reference`)
    weights = PackWeights(PackIndex(root))
    model = Inkling2LoopRef(cfg, weights, ExpertGroups(groups), unembed_chunk=100, emulate=Emulation.engine())
    return cfg, root, weights, groups, model, tf.oracle_dump(model, ids(T, 3))


def run(synth, dump, passes=range(V), with_experts=True):
    cfg, _, weights, groups, _, _ = synth
    experts = tf.local_experts(ExpertGroups(groups).lin, cfg) if with_experts else None
    return tf.teacher_force(cfg, weights, dump, list(passes), experts, log=lambda s: None)


# ---------------------------------------------------------------------------------------------------- the schedule


class Recorder(Inkling2LoopRef):
    """The oracle noting, per pass, the completed blocks and the message bank its `layer` is given."""

    seen: dict

    def layer(self, v, p, blocks, pos, lc, rec=None, bank=None):
        self.seen[v.index] = (list(blocks), None if bank is None else (list(bank[0]), bank[1]))
        return super().layer(v, p, blocks, pos, lc, rec, bank)


def test_blocks_for_is_the_schedule(synth):
    cfg = synth[0]
    visits = cfg.visits()
    assert [v.layer for v in visits] == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]
    assert [v.index for v in visits if v.block_start] == [0, 2, 4, 6, 8, 10, 12, 14]
    # the chain of a core pass is the prelude plus that pass's own earlier layers (it restarts from the prelude)
    want = {0: [], 1: [0], 2: [0], 3: [0, 2], 4: [0, 2], 5: [0, 2, 4],
            6: [0], 7: [0, 6], 8: [0, 6], 9: [0, 6, 8], 10: [0], 11: [0, 10], 12: [0, 10], 13: [0, 10, 12],
            # the layers after the core continue the LAST core pass's chain
            14: [0, 10, 12], 15: [0, 10, 12, 14]}
    for k, passes in want.items():
        assert tf.block_passes(cfg, visits[k]) == passes, k


def test_blocks_and_bank_equal_what_the_oracle_forms(synth):
    cfg, _, weights, groups, _, _ = synth
    m = Recorder(cfg, weights, ExpertGroups(groups), unembed_chunk=100)
    m.seen = {}
    out = m.forward(ids(9, 4), record=True)
    for v in cfg.visits():
        blocks, bank = m.seen[v.index]
        mine = tf.blocks_for(cfg, v, out.hidden)
        assert len(blocks) == len(mine)
        assert all(torch.equal(a, b) for a, b in zip(blocks, mine)), v.index
        if v.loop_pass:
            ref_blocks, ref_partial = bank
            my_blocks, my_partial = tf.bank_for(cfg, v, out.hidden, out.core_out)
            assert len(ref_blocks) == len(my_blocks) == 1  # the 2nd block of the previous core pass
            assert torch.equal(ref_blocks[0], my_blocks[0]) and torch.equal(ref_partial, my_partial)
        else:
            assert bank is None
    with pytest.raises(ValueError, match="no message bank"):
        tf.bank_for(cfg, cfg.visits()[3], out.hidden, out.core_out)


def test_pass_selector():
    assert tf.parse_passes("0-2,40,45-47", 48) == [0, 1, 2, 40, 45, 46, 47]
    assert tf.parse_passes("3", 16) == [3]
    with pytest.raises(ValueError, match="out of range"):
        tf.parse_passes("0-16", 16)


# ------------------------------------------------------------------------------------------------ the dump contract


def test_oracle_dump_has_the_documented_shapes(synth):
    cfg, _, _, _, _, dump = synth
    h, hq, e, k = cfg.hidden_size, cfg.num_heads * cfg.head_dim, cfg.n_experts, cfg.top_k
    loop_rows = (cfg.loop_passes - 1) * cfg.n_core
    assert cfg.n_visits == V and loop_rows == 8
    want = {"hidden": (V + 1, T, h), "core_out": (2, T, h), "attn_core": (V, T, hq), "attn": (V, T, h),
            "mlp_in": (V, T, h), "router_logits": (V, T, e), "router_idx": (V, T, k), "router_w": (V, T, k),
            "moe": (V, T, h), "msg": (loop_rows, 4, T, h), "attn_in": (loop_rows, 3, T, h)}
    assert {n: tuple(dump[n].shape) for n in want} == want
    tf.check_dump(cfg, dump)
    # a pass restarts from the prelude's state: hidden entering each core pass's first layer is the same
    assert torch.equal(dump["hidden"][6], dump["hidden"][2]) and torch.equal(dump["hidden"][10], dump["hidden"][2])
    # a local layer's three attention inputs are three different blends of one mix on core passes after the first
    local_pass = next(v for v in cfg.visits() if v.loop_pass and not v.is_global)
    ins = dump["attn_in"][tf.loop_row(cfg, local_pass)]
    assert not torch.equal(ins[0], ins[1]) and not torch.equal(ins[1], ins[2])


def test_check_dump_names_what_is_wrong(synth):
    cfg, _, _, _, _, dump = synth
    with pytest.raises(ValueError, match="no \\['msg'\\]"):
        tf.check_dump(cfg, {k: v for k, v in dump.items() if k != "msg"})
    with pytest.raises(ValueError, match="attn "):  # a plane with fewer token rows than hidden
        tf.check_dump(cfg, {**dump, "attn": dump["attn"][:, :5]})
    with pytest.raises(ValueError, match=f"{V}-pass schedule.*hidden"):
        tf.check_dump(cfg, {**dump, "hidden": dump["hidden"][:-1]})


# ------------------------------------------------------------------------------------------------------ teacher forcing


def test_the_stock_variants_without_dense_fp16_do_not_model_the_engine(synth):
    """The gap this emulation closed: a dump that carries the dense kernels' fp16 rotated input is matched exactly only by the
    variants that emulate it; the variant list keeps the old ones selectable and they differ."""
    _, _, _, _, _, dump = synth
    rep = run(synth, dump, passes=range(4), with_experts=False)
    assert tf.ENGINE_VARIANT in tf.ATTN_VARIANTS and "kv8+attn_bf16_p" in tf.ATTN_VARIANTS
    assert tf.gate(rep["passes"])["variant"] == tf.ENGINE_VARIANT
    assert sum(x["attn"]["kv8+attn_bf16_p"]["core"]["rel_max"] > 0 for x in rep["passes"]) > 0
    assert all(x["attn"][tf.ENGINE_VARIANT]["core"]["rel_max"] == 0 for x in rep["passes"])


def test_oracle_dump_passes_every_gate_exactly(synth):
    cfg, _, _, _, _, dump = synth
    rep = run(synth, dump)
    gate = tf.gate(rep["passes"])
    assert gate["pass"] and not gate["failures"]
    assert gate["residual_mismatch"] == 0 and gate["restart_mismatch"] == 0
    for x in rep["passes"]:
        v = cfg.visits()[x["pass"]]
        em = x["attn"][tf.ENGINE_VARIANT]
        assert em["core"]["rel_max"] == 0 and em["a"]["rel_max"] == 0
        assert x["mix"]["rel_max"] == 0 and x["moe"]["moe_fp16"]["rel_max"] == 0
        assert x["router"]["picks_identical"] == 1.0 and x["router"]["logit_err_max"] == 0
        assert x["router"]["engine_select_consistent"] and x["router"]["weights_err_max"] == 0
        assert x["residual_mismatch"] == 0 and x["restart_mismatch"] == 0
        assert x["attn_noise"]["a"]["rel_max"] < 2e-2 and x["moe_noise"]["rel_max"] < 2e-2
        assert x["mix_noise"]["rel_max"] < 2e-2
        # the loop-only ops exist on core passes from the second on, and only there
        assert ("msg" in x) == ("attn_in" in x) == bool(v.loop_pass)
        if v.loop_pass:
            assert x["msg"]["rel_max"] == 0 and x["attn_in"]["rel_max"] == 0
            assert x["msg_noise"]["rel_max"] < 2e-2 and x["attn_in_noise"]["rel_max"] < 2e-2
    # without the int8 KV the global passes' attention core differs from the engine-emulating dump
    glob = [x for x in rep["passes"] if x["global"]]
    assert len(glob) == 8 and all(x["attn"]["attn_bf16_p"]["core"]["rel_max"] > 0 for x in glob)
    assert all(x["attn"]["none"]["core"]["rel_max"] > 0 for x in glob)


def test_without_experts_the_moe_op_is_skipped(synth):
    rep = run(synth, synth[5], passes=[3, 4], with_experts=False)
    assert all("moe" not in x and "moe_noise" not in x for x in rep["passes"])
    assert tf.gate(rep["passes"])["pass"]


def scaled(dump, name, index, factor):
    out = dict(dump)
    plane = dump[name].clone()
    plane[index] = (plane[index].double() * factor).to(plane.dtype)
    out[name] = plane
    return out


@pytest.mark.parametrize("name,index,factor,pass_,named", [
    ("moe", 9, 1.01, 9, "moe"),
    ("attn_core", 7, 1.05, 7, "attention core"),
    ("attn", 4, 1.05, 4, "attention output"),
    ("mlp_in", 3, 1.05, 3, "mix"),
    ("router_logits", 5, 1.01, 5, "router logits or selection"),
])
def test_altered_plane_is_named(synth, name, index, factor, pass_, named):
    rep = run(synth, scaled(synth[5], name, index, factor), passes=[pass_ - 1, pass_, pass_ + 1])
    gate = tf.gate(rep["passes"])
    assert not gate["pass"]
    assert any(f.startswith(f"pass {pass_}: {named}") for f in gate["failures"]), gate["failures"]
    assert all(f.startswith(f"pass {pass_}:") for f in gate["failures"]), gate["failures"]  # no other pass is named


def test_altered_message_and_attention_input_are_named(synth):
    cfg = synth[0]
    visits = cfg.visits()
    for k, name, factor in ((8, "msg", 1.01), (11, "attn_in", 1.05)):
        row = tf.loop_row(cfg, visits[k])
        rep = run(synth, scaled(synth[5], name, row, factor), passes=[k - 1, k, k + 1])
        failures = tf.gate(rep["passes"])["failures"]
        assert any(f.startswith(f"pass {k}: {name}") for f in failures), failures
        assert all(f.startswith(f"pass {k}:") for f in failures), failures
        by = {x["pass"]: x for x in rep["passes"]}
        assert by[k][name]["rel_max"] > 2e-3 and by[k + 1]["msg"]["rel_max"] == 0


def test_core_out_is_the_bank_and_the_end_of_a_core_pass(synth):
    """core_out[0] feeds the messages of core pass 1 (passes 6..9) and is what pass 5 (the end of core pass 0) must
    produce; nothing else reads it."""
    dump = dict(synth[5])
    core_out = dump["core_out"].clone()
    # a different state, not a rescaled one: the AttnRes read normalizes every source
    core_out[0] = torch.randn(core_out[0].shape, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    dump["core_out"] = core_out
    rep = run(synth, dump, passes=range(4, 11), with_experts=False)
    by = {x["pass"]: x for x in rep["passes"]}
    assert by[5]["residual_mismatch"] > 0 and by[4]["residual_mismatch"] == 0
    assert all(by[k]["msg"]["rel_max"] > 2e-3 for k in (6, 7, 8, 9))
    assert by[10]["msg"]["rel_max"] == 0 and by[9]["residual_mismatch"] == 0  # pass 9 ends on core_out[1]
    failures = tf.gate(rep["passes"])["failures"]
    assert failures and all(any(f.startswith(f"pass {k}: msg") for k in (6, 7, 8, 9)) for f in failures)


def test_restart_state_is_checked(synth):
    dump = scaled(synth[5], "hidden", 6, 1.01)  # the first layer of core pass 1 must see the prelude's state again
    rep = run(synth, dump, passes=[5, 6], with_experts=False)
    by = {x["pass"]: x for x in rep["passes"]}
    assert by[6]["restart_mismatch"] > 0 and by[5]["restart_mismatch"] == 0
    assert tf.gate(rep["passes"])["restart_mismatch"] == by[6]["restart_mismatch"]


def test_main_over_a_pack_with_and_without_workers(synth, tmp_path):
    cfg, root, _, _, _, dump = synth
    eng = str(tmp_path / "eng.safetensors")
    st.write(eng, dump, {"prompt_len": str(T)})
    for workers, passes in ((0, "0-1,5-7"), (2, "9,14")):
        rep_path = str(tmp_path / f"rep{workers}.json")
        code = tf.main(["--pack", root, "--engine", eng, "--out", rep_path, "--passes", passes,
                        "--workers", str(workers)])
        with open(rep_path) as f:
            report = json.load(f)
        assert code == 0 and report["gate"]["pass"] and report["gate"]["residual_mismatch"] == 0
        assert [x["pass"] for x in report["passes"]] == tf.parse_passes(passes, V)
        assert all(x["moe"]["moe_fp16"]["rel_max"] == 0 for x in report["passes"])
    bad = scaled(dump, "moe", 6, 1.01)
    st.write(eng, bad, {"prompt_len": str(T)})
    rep_path = str(tmp_path / "bad.json")
    assert tf.main(["--pack", root, "--engine", eng, "--out", rep_path, "--passes", "5-7"]) == 1
    with open(rep_path) as f:
        failures = json.load(f)["gate"]["failures"]
    assert any(f.startswith("pass 6: moe") for f in failures)
    # a dump without the loop-only planes is refused, naming them
    st.write(eng, {k: v for k, v in dump.items() if k not in ("msg", "attn_in")}, {})
    with pytest.raises(SystemExit, match="msg"):
        tf.main(["--pack", root, "--engine", eng, "--out", rep_path, "--passes", "0", "--no-moe"])


# ------------------------------------------------------------------------------------------------------------ compare


@pytest.fixture(scope="module")
def fixture(synth):
    """A fixture's tensors from the oracle, as gen_fixtures writes them, and its generated continuation."""
    cfg, _, _, _, model, _ = synth
    res = generate(model, ids(7, 5).tolist(), 5)
    return gen_fixtures.fixture_tensors(cfg, res), res["prompt_len"]


def noisy(t, seed, eps, eps_hidden=1e-2):
    """Logits with absolute noise eps; hidden states with relative noise eps_hidden (above bf16's own step, so it
    survives the rounding)."""
    g = torch.Generator().manual_seed(seed)
    out = dict(t)
    out["logits"] = t["logits"] + eps * torch.randn(t["logits"].shape, generator=g)
    for name in ("hidden", "h_final"):
        x = t[name].double()
        out[name] = (x * (1 + eps_hidden * torch.randn(x.shape, generator=g, dtype=torch.float64))).to(torch.bfloat16)
    return out


def test_compare_passes_clean_and_noisy_and_names_a_broken_pass(fixture):
    oracle, n_prompt = fixture
    clean = compare.compare(oracle, oracle, n_prompt, compare.gates_for("engine"))
    assert clean["pass"] and clean["checks"]["logits"]["kl_max"] == 0
    assert clean["checks"]["router"]["identical_sets"] == 1.0
    assert len(clean["checks"]["hidden"]["cos_min_per_pass"]) == V + 1
    assert len(clean["checks"]["core_out"]["cos_min_per_core_pass"]) == 2
    slight = compare.compare(oracle, noisy(oracle, 1, 1e-4), n_prompt, compare.gates_for("engine"))
    assert slight["pass"] and 0 < slight["checks"]["logits"]["kl_max"] < 1e-3
    assert 0.999 <= slight["checks"]["hidden"]["cos_min_per_pass"][5] < 1.0
    bad = dict(oracle)
    hidden = oracle["hidden"].clone()
    hidden[9] = torch.randn(hidden[9].shape).to(torch.bfloat16)  # the state entering pass 9 is garbage
    bad["hidden"] = hidden
    rep = compare.compare(oracle, bad, n_prompt, compare.gates_for("engine"))
    assert not rep["pass"] and not rep["checks"]["hidden"]["pass"] and rep["checks"]["hidden"]["worst_pass"] == 9
    assert rep["checks"]["hidden"]["cos_min_per_pass"][8] == 1.0
    assert rep["checks"]["logits"]["pass"]  # the logits themselves were left alone


def test_compare_rejects_another_schedule_and_other_prompts(fixture):
    oracle, n_prompt = fixture
    with pytest.raises(ValueError, match="rows of passes"):
        compare.compare(oracle, {**oracle, "hidden": oracle["hidden"][:-1]}, n_prompt)
    other = {**oracle, "tokens": oracle["tokens"].clone()}
    other["tokens"][0] += 1
    with pytest.raises(ValueError, match="different prompts"):
        compare.compare(oracle, other, n_prompt)


def test_compare_cli_follows_the_fixture_gate_set(fixture, tmp_path):
    oracle, n_prompt = fixture
    meta = {"prompt_len": str(n_prompt), "emulate": "engine"}
    o, e, rp = str(tmp_path / "o.st"), str(tmp_path / "e.st"), str(tmp_path / "r.json")
    st.write(o, oracle, meta)
    st.write(e, noisy(oracle, 2, 1e-4), {})
    assert compare.main([o, e, "--json", rp]) == 0
    with open(rp) as f:
        assert json.load(f)["gate_set"] == "engine"
    hidden = oracle["hidden"].clone()
    hidden[12] = torch.randn(hidden[12].shape).to(torch.bfloat16)
    st.write(e, {**oracle, "hidden": hidden}, {})
    assert compare.main([o, e, "--json", rp]) == 1
    with open(rp) as f:
        assert json.load(f)["checks"]["hidden"]["worst_pass"] == 12
    st.write(o, oracle, {"prompt_len": str(n_prompt), "emulate": "kv8"})
    with pytest.raises(ValueError, match="no gate set"):
        compare.main([o, e])
    assert compare.main([o, e, "--gates", "engine", "--json", rp]) == 1


# --------------------------------------------------------------------------------------------------------- fixtures


def test_gen_fixtures_writes_the_documented_tensors(synth, tmp_path):
    cfg, root, weights, groups, _, _ = synth
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"name": "p1", "ids": [5, 17, 3, 99, 200, 7], "new_tokens": 4}) + "\n"
                       + json.dumps({"name": "p2", "ids": [9, 8, 7]}) + "\n")
    out = tmp_path / "fx"
    assert gen_fixtures.main(["--pack", root, "--prompts", str(prompts), "--out", str(out), "--only", "p1",
                              "--verify-recompute"]) == 0
    tensors, meta = compare.load(str(out / "p1.safetensors"))
    n = 6 + 4
    h = cfg.hidden_size
    want = {"tokens": ((n,), torch.int64), "logits": ((n - 1, cfg.unpadded_vocab_size), torch.float32),
            "h_final": ((n - 1, h), torch.bfloat16), "hidden": ((V + 1, n - 1, h), torch.bfloat16),
            "core_out": ((cfg.loop_passes - 1, n - 1, h), torch.bfloat16),
            "router_idx": ((V, n - 1, cfg.top_k), torch.int32), "router_w": ((V, n - 1, cfg.top_k), torch.float32),
            "router_margin": ((V, n - 1), torch.float32)}
    assert set(tensors) == set(want)
    for name, (shape, dtype) in want.items():
        assert tuple(tensors[name].shape) == shape and tensors[name].dtype == dtype, name
    assert tensors["tokens"][:6].tolist() == [5, 17, 3, 99, 200, 7]
    assert torch.equal(tensors["hidden"][6], tensors["hidden"][2])  # the core pass restarts from the prelude
    assert meta["prompt_len"] == "6" and meta["name"] == "p1" and meta["emulate"] == "none"
    assert meta["pack_config_sha256"] == gen_fixtures.sha256_file(os.path.join(root, "config.json"))
    assert meta["oracle_source_sha256"] == gen_fixtures.oracle_source_sha()
    assert meta["numerics_version"] == gen_fixtures.NUMERICS_VERSION and "tp" not in meta
    with open(out / "manifest.json") as f:
        manifest = json.load(f)
    assert list(manifest["fixtures"]) == ["p1"] and manifest["fixtures"]["p1"]["verified_recompute"]
    assert manifest["fixtures"]["p1"]["sha256"] == gen_fixtures.sha256_file(str(out / "p1.safetensors"))
    assert manifest["fixtures"]["p1"]["generated"] == tensors["tokens"][6:].tolist()
    # the pack folder (positional reads, EXL3 decode) gives the in-memory twin's fixture bit for bit
    twin = Inkling2LoopRef(cfg, weights, ExpertGroups(groups), unembed_chunk=100)
    want_res = generate(twin, [5, 17, 3, 99, 200, 7], 4)
    for name, value in gen_fixtures.fixture_tensors(cfg, want_res).items():
        assert torch.equal(tensors[name], value), name


def test_gen_fixtures_records_the_emulation_and_it_matters(synth, tmp_path):
    cfg, root, *_ = synth
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"name": "p1", "ids": [5, 17, 3, 99, 200, 7], "new_tokens": 2}) + "\n")
    got = {}
    for tag in ("none", "engine"):
        out = tmp_path / tag
        assert gen_fixtures.main(["--pack", root, "--prompts", str(prompts), "--out", str(out), "--emulate", tag]) == 0
        got[tag] = compare.load(str(out / "p1.safetensors"))
    assert got["none"][1]["emulate"] == "none" and got["engine"][1]["emulate"] == "engine"
    assert not torch.equal(got["none"][0]["hidden"], got["engine"][0]["hidden"])
    rep = compare.compare(got["engine"][0], got["none"][0], 6, compare.gates_for("none"))
    assert rep["checks"]["hidden"]["cos_min_per_pass"][0] == 1.0  # the embedding is untouched by the emulation


def test_per_row_report_shows_a_single_bad_row(synth, monkeypatch):
    """`--per-row`: the by-row values agree with the aggregates, and a one-row perturbation of the engine's attention output is the
    worst row of `row_table` by a wide ratio (the per-pass medians hide it)."""
    import row_table

    monkeypatch.setattr(tf, "PER_ROW", True)
    cfg, _, _, _, _, dump = synth
    bad = {k: v.clone() for k, v in dump.items()}
    bad["attn"][3, 6] = (bad["attn"][3, 6].float() * 1.5).to(bad["attn"].dtype)
    rep = run(synth, bad, passes=range(5), with_experts=False)
    p = rep["passes"][3]
    a = p["attn"][tf.ENGINE_VARIANT]["a"]
    assert len(a["by_row"]["rel"]) == T and max(a["by_row"]["rel"]) == a["rel_max"] and a["rel_max_row"] == 6
    assert len(p["router"]["by_row"]["picks_identical"]) == T and len(p["residual_mismatch_by_row"]) == T
    ops = row_table.series(p, tf.ENGINE_VARIANT)
    r = row_table.ratios(ops["attn_out rel"])
    assert r.index(max(r)) == 6 and max(r) > 50
