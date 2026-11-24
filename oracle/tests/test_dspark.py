"""DSpark drafter reference (`dspark_ref.py`) on a tiny synthetic Mini-style drafter (3 layers, layer 1 global, 4 taps over
the target's layer passes): chunking invariance, rollback, the local window and the global layer, the non-causal block,
f_rnn, the op structure against an independent transcription of the vLLM module flow, the fold, the argmax rule, the RoPE
table, the EXL3 effective-weight path (bit for bit), the config checks and the target-to-drafter feature path."""

import copy
import json
import os

import numpy as np
import pytest
import torch

import st
from dspark_ref import DSparkConfig, DSparkRef, rope_table
from inkling2_ref import DenseExperts, DictWeights, Inkling2LoopRef, Rounding, greedy
from synthetic import (config_json, random_dspark_exl3, random_dspark_weights, random_features, random_weights,
                       tiny_config, tiny_dspark_config)
from vllm_dspark_transcription import VDSpark

N = 24  # committed rows available to the tests
ANCHOR = 17
SCALES = (0.5, 1.5, 3.0, 1.0)  # per-tap feature scales
RECIPE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mtp_json(cfg: DSparkConfig) -> dict:
    """The pack's `mtp_config` for a drafter config."""
    return {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_layers, "num_attention_heads": cfg.num_heads,
            "num_key_value_heads": cfg.num_kv_heads, "head_dim": cfg.head_dim,
            "intermediate_size": cfg.intermediate_size, "rms_norm_eps": cfg.eps, "rope_theta": cfg.rope_theta,
            "local_layer_ids": list(cfg.local_layer_ids), "sliding_window_size": cfg.sliding_window,
            "block_size": cfg.block_size, "markov_rank": cfg.markov_rank, "markov_head_type": "rnn_hidden",
            "q_bias": False, "o_bias": False, "target_layer_ids": list(cfg.target_layer_ids)}


def tiny_pack_config(cfg: DSparkConfig) -> dict:
    conf = config_json(tiny_config())
    conf["mtp_config"] = mtp_json(cfg)
    return conf


@pytest.fixture(scope="module")
def drafter():
    cfg = tiny_dspark_config()
    w = random_dspark_weights(cfg, seed=3)
    feats = random_features(cfg, N + 1, seed=4, scales=SCALES)
    src = DictWeights(w)

    def make(rounding=None, source=None):
        return DSparkRef(cfg, source or src, rounding=rounding, unembed_chunk=40)

    return cfg, w, feats, make


def run(ref, feats, p, anchor=ANCHOR, chunks=None, force_prev=None):
    st_ = ref.new_state()
    cuts = [0] + list(chunks or []) + [p]
    for a, b in zip(cuts, cuts[1:]):
        ref.commit(st_, feats[a:b], a)
    return st_, ref.propose(st_, anchor, force_prev=force_prev, trace=True)


def same(a, b):
    assert a.drafts == b.drafts
    for k in a.trace:
        assert torch.equal(a.trace[k], b.trace[k]), k


# ----------------------------------------------------------------------------------------------------------------------


def test_rounded_differs_from_exact_only_by_rounding(drafter):
    cfg, w, feats, make = drafter
    _, a = run(make(), feats, 15)
    _, x = run(make(Rounding.exact()), feats, 15)
    assert (a.trace["hd"].to(torch.float64) - x.trace["hd"]).abs().max() < 0.1
    assert not torch.equal(a.trace["hd"].to(torch.float64), x.trace["hd"])
    assert a.trace["hd"].dtype == torch.bfloat16 and a.trace["logits"].dtype == torch.float32


@pytest.mark.parametrize("chunks", [[1], [5, 6, 13], list(range(1, 20)), [19]])
def test_incremental_commits_equal_one_commit(drafter, chunks):
    cfg, w, feats, make = drafter
    ref = make()
    s1, one = run(ref, feats, 20)
    s2, inc = run(ref, feats, 20, chunks=chunks)
    same(one, inc)
    assert torch.equal(s1.ctx_kv(), s2.ctx_kv())
    assert s1.ctx_kv().shape == (cfg.num_layers, 20, 2, cfg.num_kv_heads, cfg.head_dim)


def test_rollback_equals_fresh_replay(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    other = random_features(cfg, N, seed=9)
    st_ = ref.new_state()
    ref.commit(st_, other[:18], 0)  # a branch that diverges from position 9 on
    ref.commit(st_, feats[9:14], 9)  # rollback to 9, new rows 9..13: rows 14..17 become stale
    ref.commit(st_, feats[14:16], 14)
    got = ref.propose(st_, ANCHOR, trace=True)
    mixed = torch.cat([other[:9], feats[9:16]])
    fresh_state, fresh = run(ref, mixed, 16)
    same(got, fresh)
    assert torch.equal(st_.ctx_kv(), fresh_state.ctx_kv())


def test_local_window_edge_keys_and_global_layer_reads_position_zero(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    p = 20
    wdw = cfg.sliding_window
    st_, base = run(ref, feats, p)

    def perturbed(layer, pos):
        s, _ = run(ref, feats, p)
        s.k[layer][pos] += 0.5
        s.v[layer][pos] -= 0.5
        return ref.propose(s, ANCHOR, trace=True).trace["hd"]

    assert cfg.local_layer_ids == (0, 2)
    for layer in cfg.local_layer_ids:
        assert torch.equal(perturbed(layer, p - wdw - 1), base.trace["hd"]), layer  # just outside the window
        assert not torch.equal(perturbed(layer, p - wdw), base.trace["hd"]), layer  # oldest visible key
    glob = [i for i in range(cfg.num_layers) if not cfg.is_local(i)]
    assert glob == [1]
    assert not torch.equal(perturbed(1, 0), base.trace["hd"])
    # through the features: row 0 is outside every local window, so only the global layer can see it
    f2 = feats.clone()
    f2[0] = random_features(cfg, 1, seed=11)[0]
    _, moved = run(ref, f2, p)
    assert not torch.equal(moved.trace["hd"], base.trace["hd"])


def test_block_is_non_causal_and_anchor_ctx_row_is_never_read(drafter):
    cfg, w, feats, make = drafter
    p = 15
    _, base = run(make(), feats, p)
    w2 = dict(w)
    w2["mask_embed"] = random_dspark_weights(cfg, seed=12)["mask_embed"]
    _, moved = run(make(source=DictWeights(w2)), feats, p)
    assert not torch.equal(moved.trace["hd"][0], base.trace["hd"][0])  # the anchor row attends to the mask rows
    # a committed-then-rolled-back row at p (the anchor's own position) stays stale and unread
    ref = make()
    st_ = ref.new_state()
    ref.commit(st_, feats[:p + 1], 0)
    ref.commit(st_, feats[p - 1:p], p - 1)
    same(ref.propose(st_, ANCHOR, trace=True), base)


def test_frnn_is_last_committed_row(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    st_, got = run(ref, feats, 12)
    want = ref.mix(feats[11:12].to(torch.float64), cfg.num_layers)[0]
    assert torch.equal(got.trace["f_rnn"].to(torch.float64), want)
    ref.commit(st_, feats[20:21], 11)  # overwrite position 11
    want2 = ref.mix(feats[20:21].to(torch.float64), cfg.num_layers)[0]
    assert torch.equal(ref.propose(st_, ANCHOR, trace=True).trace["f_rnn"].to(torch.float64), want2)
    assert not torch.equal(want, want2)


@pytest.mark.parametrize("kind", ["bf16", "exl3_effective_float64"])
def test_exact_matches_vllm_transcription(drafter, kind):
    """Commit, propose, accept k, commit (overwriting the block's slots in V's cache), propose again, plus a rollback;
    V's flow keeps the stale block K/V and an extra ctx row at the anchor position in its cache. The second weight set
    is float64 throughout (effective EXL3 weights), the first is bf16."""
    cfg, w, feats, make = drafter
    if kind == "bf16":
        weights = w
    else:
        weights = random_dspark_exl3(cfg, seed=5)[2]
    ref = make(Rounding.exact(), DictWeights(weights))
    v = VDSpark(cfg, weights, max_pos=N + 1)
    f64 = feats.to(torch.float64)
    st_ = ref.new_state()

    def commit(a, b):
        ref.commit(st_, f64[a:b], a)
        v.precompute_and_store_context_kv(f64[a:b].reshape(b - a, -1), torch.arange(a, b))
        v.prepare_context_anchor(f64[b - 1])

    def check(p, anchor):
        got = ref.propose(st_, anchor, trace=True)
        drafts, hd, logits = v.draft(anchor, p)
        assert got.drafts == drafts
        assert torch.allclose(got.trace["hd"], hd, rtol=0, atol=1e-9)
        assert torch.allclose(got.trace["logits"], logits, rtol=0, atol=1e-9)
        return drafts

    commit(0, 9)
    d = check(9, ANCHOR)
    commit(9, 12)  # accepted anchor + d1 + d2 -> next anchor at 12
    # V holds a ctx row at the next anchor's position (written by an earlier step); the block overwrites it
    v.precompute_and_store_context_kv(f64[12:13].reshape(1, -1), torch.arange(12, 13))
    check(12, d[2])
    commit(12, 20)
    check(20, 5)
    commit(14, 16)  # rollback
    check(16, 7)


def test_fold_exact(drafter):
    cfg, w, feats, make = drafter
    ref = make(Rounding.exact())
    _, got = run(ref, feats, 10)
    u = w["model.llm.unembed.weight"].to(torch.float64)[: cfg.valid_vocab]
    wout = w["markov_head.hidden_out.weight"].to(torch.float64)
    bout = w["markov_head.hidden_out.bias"].to(torch.float64)
    for latent in got.trace["latent"]:
        want = u @ (wout @ latent + bout)
        assert torch.allclose(ref.markov_bias(latent), want, rtol=0, atol=1e-10)
    rounded = make()
    f = rounded.fold()
    assert torch.equal(f.weight, (u @ wout).to(torch.float32).to(torch.bfloat16).to(torch.float64))
    assert f.weight.shape == (cfg.valid_vocab, cfg.markov_rank)


def test_fold_and_base_logits_stream_float64_unembed_in_row_chunks(drafter):
    """A float64 (effective EXL3) unembed streamed in 7-row chunks gives the same fold and logits as one chunk."""
    cfg, _, feats, _ = drafter
    weights = DictWeights(random_dspark_exl3(cfg, seed=6)[2])
    assert weights.tensor("model.llm.unembed.weight").dtype == torch.bfloat16  # the unembed stays dense here
    f64 = {k: v.to(torch.float64) if k == "model.llm.unembed.weight" else v for k, v in weights.t.items()}
    a = DSparkRef(cfg, DictWeights(f64), unembed_chunk=7)
    b = DSparkRef(cfg, DictWeights(f64), unembed_chunk=cfg.vocab_size)
    assert torch.equal(a.fold().weight, b.fold().weight) and torch.equal(a.fold().bias, b.fold().bias)
    hd = torch.randn(7, cfg.hidden_size, dtype=torch.float64)
    assert torch.equal(a.base_logits(hd), b.base_logits(hd))
    assert a.base_logits(hd).shape == (7, cfg.valid_vocab)


def test_argmax_first_max_and_nan(drafter):
    x = torch.tensor([[1.0, 3.0, 3.0, 2.0], [float("nan"), 0.5, 0.5, -1.0], [float("nan")] * 3 + [-5.0]])
    assert greedy(x).tolist() == [1, 1, 3]
    cfg, w, feats, make = drafter
    ref = make()
    _, got = run(ref, feats, 10)
    lg = got.trace["logits"].to(torch.float64)
    for i, d in enumerate(got.drafts):
        m = lg[i].max()
        assert lg[i, d] == m and (lg[i, :d] < m).all()


def test_teacher_forcing(drafter):
    cfg, w, feats, make = drafter
    ref = make()
    _, free = run(ref, feats, 10)
    _, forced_own = run(ref, feats, 10, force_prev=free.drafts)
    same(free, forced_own)
    other = [(d + 1) % cfg.valid_vocab for d in free.drafts]
    _, forced = run(ref, feats, 10, force_prev=other)
    assert torch.equal(forced.trace["logits"][0], free.trace["logits"][0])  # step 1 reads only the anchor
    assert not torch.equal(forced.trace["logits"][1], free.trace["logits"][1])


def test_rope_table_fp32_formula():
    d, theta = 128, 1e6
    pos = np.array([0, 1, 2, 7, 1023, 1024, 1031, 65535, 1048575])
    i = np.arange(d // 2)
    inv = (np.float32(1.0) / np.float32(theta ** (2.0 * i / d))).astype(np.float32)
    ang = pos.astype(np.float32)[:, None] * inv[None, :]  # one fp32 multiply
    assert ang.dtype == np.float32
    want_c = np.cos(ang.astype(np.float64)).astype(np.float32)
    want_s = np.sin(ang.astype(np.float64)).astype(np.float32)
    c, s = rope_table(torch.from_numpy(pos), d, theta, Rounding())
    assert np.array_equal(c.numpy().astype(np.float32), want_c)
    assert np.array_equal(s.numpy().astype(np.float32), want_s)
    assert np.array_equal(c.numpy(), want_c.astype(np.float64))  # fp32 values, held in float64
    # V's get_rope builds the table with fp32 powf / cosf / sinf, which are not correctly rounded (torch's CPU powf is
    # one ulp off at i = 37 for theta 1e6): the spec's correctly rounded table agrees to a few fp32 ulps of the angle
    tf = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    tc = (torch.from_numpy(pos).float()[:, None] * tf[None]).cos()
    assert np.abs(tc.numpy().astype(np.float64) - c.numpy()).max() < 1e-4


# ---------------------------------------------------------------------------------------------------------------------
# EXL3 effective weights through the pack path


def write_drafter_pack(root, cfg, seed):
    """`mtp.safetensors` with EXL3 4-bit `mul1` linears plus a target file with embed / embed_norm / unembed. Returns the
    in-memory reference weights (effective float64 linears)."""
    plain, groups, reference = random_dspark_exl3(cfg, seed=seed)
    os.makedirs(root, exist_ok=True)
    target = {k: v for k, v in plain.items() if k.startswith("model.llm.")}
    mtp = {k: v for k, v in plain.items() if k not in target}
    for prefix, g in groups.items():
        for part, arr in g.items():
            mtp[f"{prefix}.{part}"] = arr
    st.write(os.path.join(root, "target.safetensors"), target)
    st.write(os.path.join(root, "mtp.safetensors"), mtp)
    return reference, groups


def test_exl3_pack_path_equals_dict_of_effective_weights_bit_for_bit(tmp_path, drafter):
    from gen_dspark_fixtures import DSparkPackWeights

    cfg, _, feats, _ = drafter
    reference, groups = write_drafter_pack(str(tmp_path / "pack"), cfg, seed=7)
    hdr = st.read_header(str(tmp_path / "pack" / "mtp.safetensors"))
    q = "layers.0.self_attn.q_proj"
    assert f"{q}.trellis" in hdr and f"{q}.mul1" in hdr and f"{q}.weight" not in hdr  # really EXL3 on disk
    assert reference[f"{q}.weight"].dtype == torch.float64
    src = DSparkPackWeights(str(tmp_path / "pack"), target_files=["target.safetensors"])
    assert src.tensor(f"{q}.weight").dtype == torch.float64
    a = DSparkRef(cfg, src, unembed_chunk=40)
    b = DSparkRef(cfg, DictWeights(reference), unembed_chunk=40)
    s1, ra = run(a, feats, 20, chunks=[7, 12])
    s2, rb = run(b, feats, 20)
    same(ra, rb)
    assert torch.equal(s1.ctx_kv(), s2.ctx_kv())
    assert torch.equal(a.fold().weight, b.fold().weight)
    # the effective weights are not the bf16 ones: the run is a different (EXL3) model
    other = DSparkRef(cfg, DictWeights(random_dspark_weights(cfg, seed=3)), unembed_chunk=40)
    _, ro = run(other, feats, 20)
    assert not torch.equal(ra.trace["hd"], ro.trace["hd"])


# ---------------------------------------------------------------------------------------------------------------------
# config


def test_config_from_hf_accepts_tiny_and_real_mini():
    cfg = tiny_dspark_config()
    assert DSparkConfig.from_hf(tiny_pack_config(cfg)) == cfg
    path = os.path.join(RECIPE_ROOT, "intake", "files", "config.json")
    with open(path) as f:
        real = json.load(f)
    got = DSparkConfig.from_hf(real)
    assert got == DSparkConfig()
    assert got.target_layer_ids == (5, 23, 41, 48) and got.n_taps == 4 and got.local_layer_ids == (0, 2)
    t = real["text_config"]["loop_config"]
    passes = real["text_config"]["num_hidden_layers"] + (t["num_passes"] - 1) * (t["core_end"] - t["core_start"])
    assert passes == 48 == got.target_layer_ids[-1]


def test_config_from_hf_refuses_bad_configs():
    cfg = tiny_dspark_config()
    base = tiny_pack_config(cfg)
    DSparkConfig.from_hf(base)

    def bad(edit):
        conf = copy.deepcopy(base)
        edit(conf)
        with pytest.raises(ValueError):
            DSparkConfig.from_hf(conf)

    bad(lambda c: c["mtp_config"].update(target_layer_ids=[3, 9, 13, 24]))  # last tap != number of passes (16)
    bad(lambda c: c["mtp_config"].update(target_layer_ids=[3, 13, 9, 16]))  # unsorted
    bad(lambda c: c["mtp_config"].update(target_layer_ids=[3, 9, 9, 16]))  # duplicate
    bad(lambda c: c["mtp_config"].update(markov_head_type="mlp"))
    bad(lambda c: c["mtp_config"].update(q_bias=True))
    bad(lambda c: c["mtp_config"].update(o_bias=True))
    bad(lambda c: c["mtp_config"].update(hidden_size=cfg.hidden_size * 2))
    bad(lambda c: c["text_config"].pop("loop_config"))
    bad(lambda c: c.pop("mtp_config"))
    # the pass count follows the loop: one more pass makes 20 passes, so a last tap of 16 is refused
    bad(lambda c: c["text_config"]["loop_config"].update(num_passes=4))


# ---------------------------------------------------------------------------------------------------------------------
# the taps of the looped target feed the drafter


def test_target_taps_over_passes_feed_a_drafter_commit():
    tcfg = tiny_config()
    dcfg = tiny_dspark_config()
    assert dcfg.target_layer_ids == tcfg.drafter_target_layers and dcfg.target_layer_ids[-1] == tcfg.n_visits == 16
    tw = DictWeights(random_weights(tcfg, seed=1))
    target = Inkling2LoopRef(tcfg, tw, DenseExperts(tw), unembed_chunk=128, query_chunk=5)
    ids = torch.from_numpy(np.random.default_rng(2).integers(0, tcfg.unpadded_vocab_size, size=14))
    out = target.forward(ids, record=True)
    feats = target.drafter_features(out)
    assert feats.shape == (14, dcfg.n_taps, dcfg.hidden_size)
    for j, tap in enumerate(dcfg.target_layer_ids[:-1]):
        assert torch.equal(feats[:, j], out.hidden[tap])  # tap ids index the passes
    assert torch.equal(feats[:, -1], out.h_final)
    ref = DSparkRef(dcfg, DictWeights(random_dspark_weights(dcfg, seed=3)), unembed_chunk=40)
    st_ = ref.new_state()
    ref.commit(st_, feats[:10], 0)
    ref.commit(st_, feats[10:], 10)
    assert st_.length == 14 and torch.equal(st_.last_row.to(torch.bfloat16), feats[-1])
    prop = ref.propose(st_, int(ids[-1]), trace=True)
    assert len(prop.drafts) == 7 and all(0 <= d < dcfg.valid_vocab for d in prop.drafts)
    # one commit of the same rows is identical
    st2 = ref.new_state()
    ref.commit(st2, feats, 0)
    same(prop, ref.propose(st2, int(ids[-1]), trace=True))


# ---------------------------------------------------------------------------------------------------------------------


def test_fixture_generator_contract(tmp_path, drafter):
    """gen_dspark_fixtures.py over a tiny pack (mtp.safetensors with EXL3 linears + target tensors + config.json) and
    tiny target fixtures writes every contract file with the contract's shapes, and its drafts are the reference's."""
    import gen_dspark_fixtures as gen

    cfg, _, _, _ = drafter
    pack, fx, out = tmp_path / "pack", tmp_path / "fx", tmp_path / "out"
    fx.mkdir()
    reference, _ = write_drafter_pack(str(pack), cfg, seed=8)
    (pack / "config.json").write_text(json.dumps(tiny_pack_config(cfg)))
    npass = cfg.target_layer_ids[-1]
    g = torch.Generator().manual_seed(5)
    for name, t in (("raw-counting", 50), ("chat-sky-effort0.9", 49), ("chat-capital-effort0", 40),
                    ("raw-long-window", 1126)):
        st.write(str(fx / f"{name}.safetensors"), {
            "tokens": torch.randint(0, cfg.valid_vocab, (t + 1,), generator=g),
            "hidden": torch.randn(npass + 1, t, cfg.hidden_size, generator=g).to(torch.bfloat16),
            "h_final": torch.randn(t, cfg.hidden_size, generator=g).to(torch.bfloat16)})
    assert set(gen.CASES) == {"real_raw-counting", "real_chat-sky-effort0.9", "real_chat-capital-effort0",
                              "real_raw-long-window", "long"}
    assert len(gen.LONG_SCALES) == cfg.n_taps
    assert gen.main(["--pack", str(pack), "--fixtures", str(fx), "--out", str(out), "--threads", "1"]) == 0
    h, rk, nkv, d = cfg.hidden_size, cfg.markov_rank, cfg.num_kv_heads, cfg.head_dim
    for name, case in gen.CASES.items():
        meta = json.loads((out / f"{name}_meta.json").read_text())
        assert set(meta) == {"anchors", "n_positions", "feature_source", "mtp_sha256", "oracle_source_sha256",
                             "seconds"}
        n, p = meta["n_positions"], len(case["anchors"])
        want = {"feats": (np.uint16, (n, cfg.n_taps, h)), "tokens": (np.int64, (n + 1,)),
                "anchors": (np.int64, (p,)), "drafts": (np.int64, (p, 7)), "hd": (np.uint16, (p, 8, h)),
                "frnn": (np.uint16, (p, h)), "state": (np.uint16, (p, 7, rk)), "latent": (np.uint16, (p, 7, rk)),
                "logits": (np.float32, (p, 7, cfg.valid_vocab)),
                "ctxkv": (np.uint16, (cfg.num_layers, n, 2, nkv, d))}
        arr = {k: np.load(out / f"{name}_{k}.npy") for k in want}
        for k, (dt, shape) in want.items():
            assert arr[k].dtype == dt and arr[k].shape == shape, (name, k)
        # replay the last anchor directly from the stored features through the in-memory effective weights
        ref = DSparkRef(cfg, DictWeights(reference), unembed_chunk=40)
        feats = torch.from_numpy(arr["feats"].view(np.int16)).view(torch.bfloat16)
        st_ = ref.new_state()
        last = case["anchors"][-1]
        ref.commit(st_, feats[:last], 0)
        prop = ref.propose(st_, int(arr["tokens"][last]), trace=True)
        assert prop.drafts == arr["drafts"][-1].tolist()
        assert np.array_equal(prop.trace["logits"].numpy(), arr["logits"][-1])
