"""Whole-model tests on the tiny looped synthetic config (8 layers = 4 AttnRes blocks, global layers 1, 3, 5, 7, core
[2, 6) x 3 passes = 16 layer passes, window 8; a second shape has a core of three blocks)."""

import json
import os

import numpy as np
import pytest
import torch

import hf_transcription as hf
import inkling2_ref as ref
from inkling2_ref import Inkling2LoopConfig, Rounding
from synthetic import config_json, tiny_config


def ids(n, seed=0, vocab=250):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, vocab, size=n))


def rel_max(a, b):
    """Worst token's relative L2 error of a against b."""
    a, b = a.double().reshape(a.shape[0], -1), b.double().reshape(b.shape[0], -1)
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())


BF16_STEP = 2.0 ** -8  # one bf16 rounding moves a value by at most half of this, relative


@pytest.mark.parametrize("fixture,n,seed", [("tiny", 21, 3), ("tiny", 9, 4), ("tiny_wide_core", 21, 5)])
def test_unrounded_oracle_equals_hf_transcription(request, fixture, n, seed):
    """Logits and the partial block entering every layer pass (and after the last) equal the independent HF transcription
    to 1e-9 with the rounding points off: the pass schedule, the restart of every core pass from the prelude's AttnRes
    state, the message bank, the controllers and the per-pass state are the HF model's."""
    cfg, w, make = request.getfixturevalue(fixture)
    x = ids(n, seed=seed)  # longer than the window (8) and the global rel extent (8), past the log-scaling floor (6)
    out = make(Rounding.exact()).forward(x, record=True)
    want_logits, want_hidden = hf.forward(cfg, w, x)
    assert out.hidden.shape == (cfg.n_visits + 1, n, cfg.hidden_size)
    torch.testing.assert_close(out.hidden, want_hidden, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(out.logits, want_logits, rtol=1e-9, atol=1e-9)


def test_messages_matter(tiny):
    """The loop is not decoration: with every controller zeroed the later core passes see no message and the logits
    move (the transcription test above then checks the real path; this guards against a vacuous comparison)."""
    cfg, w, make = tiny
    x = ids(12, seed=6)
    base = make(Rounding.exact()).forward(x).logits
    zero = dict(w)
    for k in list(zero):
        if ".controllers." in k and k.endswith(("gain_weight", "forget_weight")):
            zero[k] = torch.zeros_like(zero[k])
    m = make(Rounding.exact())
    m.w = m.experts.w = ref.DictWeights(zero)
    assert rel_max(m.forward(x).logits, base) > 1e-3


def recorded_inputs(m, x):
    """Run `m` and return, per layer pass, the (P, completed blocks, bank) the pass was called with, and the pass output."""
    calls, orig = [], m.layer

    def spy(v, p, blocks, pos, lc, rec=None, bank=None):
        before = list(blocks)  # the pass appends P to `blocks` when it opens a block
        out = orig(v, p, blocks, pos, lc, rec, bank)
        calls.append((v, p, before, None if bank is None else (list(bank[0]), bank[1]), out))
        return out

    m.layer = spy
    m.forward(x)
    m.layer = orig
    return calls


def fixed_routing(make, w, rounding=None):
    """The tiny model with router biases that pin every token to the same experts (the synthetic router's margins are
    small enough that a bf16 ulp flips routes, and a flip is a discrete jump, not rounding drift)."""
    m = make(rounding)
    fixed = dict(w)
    for i in range(m.cfg.num_layers):
        fixed[f"model.llm.layers.{i}.mlp.gate.bias"] = torch.tensor(
            [3.0 * ((e * 5 + i) % m.cfg.n_experts) for e in range(m.cfg.n_experts)], dtype=torch.float32)
    m.w = m.experts.w = ref.DictWeights(fixed)
    return m


def run_pass(m, call, n):
    v, p, blocks, bank, _ = call
    b = m.r.bf16
    bank = None if bank is None else ([b(t) for t in bank[0]], b(bank[1]))
    return m.layer(v, b(p), [b(t) for t in blocks], torch.arange(n), m.new_cache().layers[v.index], None, bank)


def test_each_pass_rounded_tracks_unrounded(tiny):
    """A whole rounded layer pass, from the unrounded model's state, stays within a few percent of the unrounded pass
    (the per-op errors, through the layer's gain): every pass, the first and the message-carrying ones, local and global."""
    _, w, make = tiny
    x = ids(21, seed=4)
    exact = fixed_routing(make, w, Rounding.exact())
    calls = recorded_inputs(exact, x)
    rounded = fixed_routing(make, w)
    assert len(calls) == 16 and any(c[3] is not None for c in calls)
    for k, call in enumerate(calls):
        assert rel_max(run_pass(rounded, call, 21), call[4]) < 12 * BF16_STEP, f"pass {k}"


@pytest.mark.parametrize("k", [0, 1, 3, 9, 11])
def test_each_op_costs_bf16_level_error(tiny, k):
    """From identical float64 inputs, each rounded op lands within a few bf16 steps of the unrounded op: one step for a
    single-store op (mix + norm, conv, blend), a few for the chains (attention: ~8 stores; MoE: router, g/u, act).
    Passes 0 (layer 0, no mix), 1 (global), 3 (global, core pass 0), 9 and 11 (core pass 1, local and global)."""
    cfg, w, make = tiny
    n = 21
    ex, rd = fixed_routing(make, w, Rounding.exact()), fixed_routing(make, w)
    v, p, blocks, bank, _ = recorded_inputs(ex, ids(n, seed=4))[k]
    ln = f"model.llm.layers.{v.layer}."
    pos = torch.arange(n)
    norms = [ex.w64(ln + f"{c}_input_norm.weight") for c in "qkv"] if v.is_global else [ex.w64(ln + "attn_norm.weight")]
    if v.layer == 0:
        mix = lambda m: [ref.rmsnorm_w(p, norms[0], cfg.eps, m.r)]  # noqa: E731
    else:
        q = ex.query(f"model.llm.attn_res.attn_queries.{v.layer - 1}")
        mix = lambda m: m.mix(blocks, p, q, norms)  # noqa: E731
    xs = mix(ex)
    for got, want in zip(mix(rd), xs):
        assert rel_max(got, want) < BF16_STEP
    if bank is not None:
        mq = ex.message_queries()[v.layer - cfg.loop_start]
        msgs = ex.mix_raw(bank[0], bank[1], mq)
        assert rel_max(rd.mix_raw(bank[0], bank[1], mq)[0], msgs[0]) < 1e-6  # no storage point inside: fp32 constants only
        raw = ex.mix_raw(blocks, p, ex.query(f"model.llm.attn_res.attn_queries.{v.layer - 1}"))
        dests = [raw[0], raw[1], raw[2]] if v.is_global else [raw[0]] * 3
        for s, name in enumerate(ref.STREAMS):
            dest = dests[min(s, 2)]  # the MLP stream blends into an MLP-input mix: any [T, H] mix serves the op test
            assert rel_max(rd.blend(v.layer, name, dest, msgs[s]), ex.blend(v.layer, name, dest, msgs[s])) \
                < 4 * BF16_STEP
    xq, xk, xv = (xs * 3)[:3] if len(xs) == 1 else xs
    a = ex.attention(v, xq, xk, xv, pos, ex.new_cache().layers[v.index])
    assert rel_max(rd.attention(v, xq, xk, xv, pos, rd.new_cache().layers[v.index]), a) < 4 * BF16_STEP
    cw, z = ex.w64(ln + "attn_sconv.weight")[:, 0, :], torch.zeros(3, cfg.hidden_size, dtype=torch.float64)
    assert rel_max(ref.static_conv(a, cw, z, rd.r)[0], ref.static_conv(a, cw, z, ex.r)[0]) < BF16_STEP
    assert rel_max(rd.moe(v, xs[0], None), ex.moe(v, xs[0], None)) < 2 * BF16_STEP


def test_output_dtypes(tiny):
    _, _, make = tiny
    out = make().forward(ids(3), record=True)
    assert out.logits.dtype == torch.float32 and out.hidden.dtype == torch.bfloat16
    assert out.h_final.dtype == torch.bfloat16 and out.router_w.dtype == torch.float32
    assert out.core_out.dtype == torch.bfloat16


def expected_attnres_calls(cfg):
    """(blocks read, queries) of every AttnRes read in execution order, derived from the layer rule only: a layer opens a
    block when `layer % layers_per_block == 0`; the attention read sees the blocks before the layer (3 queries on a global
    layer), the MLP read the blocks including the one this layer opened; a core pass after the first starts with the
    message read over the previous pass's blocks after the one its first layer opened (4 queries)."""
    out, nb = [], 0

    def layer(i, bank=None):
        nonlocal nb
        if bank is not None:
            out.append((bank, 4))
        if i > 0:
            out.append((nb, 3 if cfg.is_global(i) else 1))
        nb += i % cfg.layers_per_block == 0
        out.append((nb, 1))

    for i in range(cfg.loop_start):
        layer(i)
    pre, last = nb, None
    for p in range(cfg.loop_passes):
        nb = pre
        for i in range(cfg.loop_start, cfg.loop_end):
            layer(i, None if p == 0 else last - pre - 1)
        last = nb
    for i in range(cfg.loop_end, cfg.num_layers):
        layer(i)
    out.append((nb, 1))
    return out


@pytest.mark.parametrize("fixture", ["tiny", "tiny_wide_core"])
def test_attnres_schedule(request, fixture):
    cfg, _, make = request.getfixturevalue(fixture)
    m = make()
    calls, names = [], []
    orig_raw, orig_query = m.mix_raw, m.query

    def mix_raw(blocks, partial, q):
        calls.append((len(blocks), q.shape[0]))
        return orig_raw(blocks, partial, q)

    def query(name):
        names.append(name)
        return orig_query(name)

    m.mix_raw, m.query = mix_raw, query
    m.forward(ids(5))
    assert calls == expected_attnres_calls(cfg)
    want_names = []
    for v in m.visits:
        if v.layer > 0:
            want_names.append(f"model.llm.attn_res.attn_queries.{v.layer - 1}")
        want_names.append(f"model.llm.attn_res.mlp_queries.{v.layer}")
    want_names.append("model.llm.attn_res.output_query")
    assert names == want_names


def test_block_bank_has_the_later_blocks_of_the_previous_pass(tiny_wide_core):
    """In the 10-layer shape a core pass opens blocks at layers 2, 4 and 6: the next pass's bank holds the blocks of
    layers 4 and 6 (not the one of layer 2, which every pass shares with the prelude) plus the previous output."""
    cfg, _, make = tiny_wide_core
    m = make()
    banks = []
    orig = m.layer

    def spy(v, p, blocks, pos, lc, rec=None, bank=None):
        if bank is not None and v.layer == cfg.loop_start:
            banks.append((len(bank[0]), bank[0][0].clone(), bank[1].clone()))
        return orig(v, p, blocks, pos, lc, rec, bank)

    inputs = recorded_inputs(make(), ids(7, seed=1))
    m.layer = spy
    m.forward(ids(7, seed=1))
    assert [b[0] for b in banks] == [2]
    first_pass = {c[0].layer: c for c in inputs if c[0].loop_pass == 0}
    assert torch.equal(banks[0][1], first_pass[4][1])  # P entering layer 4 of the first pass
    assert torch.equal(banks[0][2], first_pass[7][4])  # P leaving layer 7 of the first pass


@pytest.mark.parametrize("splits", [[5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 3, 11], [1] * 21, [20, 1]])
def test_prefill_then_decode_equals_recompute(tiny, splits):
    _, _, make = tiny
    m = make()
    x = ids(21, seed=5)
    full = m.forward(x, record=True)
    cache = m.new_cache()
    parts, at = [], 0
    for n in splits:
        parts.append(m.forward(x[at:at + n], cache, record=True))
        at += n
    assert cache.pos == 21
    assert torch.equal(torch.cat([p.logits for p in parts]), full.logits)
    assert torch.equal(torch.cat([p.hidden for p in parts], 1), full.hidden)
    assert torch.equal(torch.cat([p.core_out for p in parts], 1), full.core_out)
    assert torch.equal(torch.cat([p.router_idx for p in parts], 1), full.router_idx)
    # local passes keep a ring of window - 1 positions, global passes everything; every pass has its own state
    assert len(cache.layers) == m.cfg.n_visits
    for v, lc in zip(m.visits, cache.layers):
        assert lc.k.shape[0] == (21 if v.is_global else m.cfg.sliding_window - 1)


def test_core_passes_have_their_own_state(tiny):
    """The three passes of a core layer read and write three separate K/V and conv states: they hold different values
    (each pass sees different inputs) and are distinct objects."""
    cfg, _, make = tiny
    m = make()
    cache = m.new_cache()
    m.forward(ids(12, seed=2), cache)
    for layer in range(cfg.loop_start, cfg.loop_end):
        lcs = [cache.layers[v.index] for v in m.visits if v.layer == layer]
        assert len(lcs) == cfg.loop_passes and len({id(c) for c in lcs}) == cfg.loop_passes
        for a in range(cfg.loop_passes):
            for b in range(a + 1, cfg.loop_passes):
                assert not torch.equal(lcs[a].k, lcs[b].k)
                assert not torch.equal(lcs[a].conv["attn"], lcs[b].conv["attn"])


def test_cached_equals_uncached_continuation(tiny):
    _, _, make = tiny
    m = make()
    x = ids(16, seed=6)
    cache = m.new_cache()
    m.forward(x[:9], cache)
    snap = cache.clone()
    a = m.forward(x[9:], cache)
    b = m.forward(x[9:], snap)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.logits, m.forward(x).logits[9:])


def test_drafter_features_tap_the_passes(tiny):
    cfg, _, make = tiny
    m = make()
    out = m.forward(ids(4), record=True)
    f = m.drafter_features(out)
    assert cfg.drafter_target_layers == (3, 9, 13, 16)
    assert f.shape == (4, 4, cfg.hidden_size)
    for j, k in enumerate((3, 9, 13)):
        assert torch.equal(f[:, j], out.hidden[k])
    assert torch.equal(f[:, -1], out.h_final)


def test_core_out_is_the_bank_partial(tiny):
    """`core_out[p]` is P leaving the last core layer of pass p (not recorded in `hidden`, which holds the restart)."""
    cfg, _, make = tiny
    m = make()
    x = ids(6, seed=8)
    out = m.forward(x, record=True)
    calls = recorded_inputs(make(), x)
    ends = [c[4] for c in calls if c[0].layer == cfg.loop_end - 1 and c[0].loop_pass is not None]
    assert out.core_out.shape[0] == cfg.loop_passes - 1
    for p in range(cfg.loop_passes - 1):
        assert torch.equal(out.core_out[p], ends[p])
    first_coda = cfg.loop_start + cfg.loop_passes * cfg.n_core
    assert torch.equal(out.hidden[first_coda], ends[-1])  # the last pass flows straight into the coda


def test_generate_is_greedy_over_cache(tiny):
    _, _, make = tiny
    m = make()
    res = ref.generate(m, ids(6, seed=9).tolist(), 5)
    toks = res["tokens"]
    assert toks.shape[0] == 11 and res["logits"].shape[0] == 10
    assert res["core_out"].shape[:2] == (2, 10)
    full = m.forward(toks)
    assert torch.equal(full.logits[:10], res["logits"])
    assert torch.equal(ref.greedy(full.logits[5:10]), toks[6:])


def test_image_placeholder_refused(tiny):
    _, _, make = tiny
    with pytest.raises(ValueError, match="vision"):
        make().embed(torch.tensor([ref.IMAGE_PAD_ID]))


def load_pack_config():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "intake", "files", "config.json")
    with open(path) as f:
        return json.load(f)


def test_from_hf_reads_pack_config():
    cfg = Inkling2LoopConfig.from_hf(load_pack_config())
    assert cfg == Inkling2LoopConfig()
    assert (cfg.loop_start, cfg.loop_end, cfg.loop_passes, cfg.gate_rank) == (6, 18, 3, 64)
    assert [v.index for v in cfg.visits() if v.is_global] == [5, 11, 17, 23, 29, 35, 41, 47]
    assert cfg.n_visits == 48 == len(cfg.visits()) and cfg.drafter_target_layers == (5, 23, 41, 48)


def test_schedule_layout():
    """The 48-pass schedule: layers 0-5, the core 6-17 three times, layers 18-23; global passes numbered in order."""
    cfg = Inkling2LoopConfig()
    vs = cfg.visits()
    assert [v.layer for v in vs] == list(range(6)) + 3 * list(range(6, 18)) + list(range(18, 24))
    assert [v.loop_pass for v in vs] == [None] * 6 + [0] * 12 + [1] * 12 + [2] * 12 + [None] * 6
    assert [v.global_ordinal for v in vs if v.is_global] == list(range(8))
    assert [v.index for v in vs if v.block_start] == [0, 6, 12, 18, 24, 30, 36, 42]
    for v in vs:  # HF cache layer of (layer, pass >= 1): num_layers + (pass - 1) * n_core + (layer - loop_start)
        if v.loop_pass:
            assert v.index == cfg.loop_start + v.loop_pass * cfg.n_core + v.layer - cfg.loop_start


def test_from_hf_refuses_what_it_does_not_implement():
    good = load_pack_config()

    def bad(**edits):
        c = json.loads(json.dumps(good))
        for k, v in edits.items():
            tgt, key = (c["text_config"], k[2:]) if k.startswith("t_") else (c, k)
            if v is None:
                tgt.pop(key, None)
            else:
                tgt[key] = v
        return c

    with pytest.raises(ValueError, match="loop_config"):
        Inkling2LoopConfig.from_hf(bad(t_loop_config=None))
    with pytest.raises(ValueError, match="is_varloop"):
        Inkling2LoopConfig.from_hf(bad(t_loop_config={**good["text_config"]["loop_config"], "is_varloop": True}))
    with pytest.raises(ValueError, match="multiple"):
        Inkling2LoopConfig.from_hf(bad(t_loop_config={**good["text_config"]["loop_config"], "core_start": 7}))
    with pytest.raises(ValueError, match="layer passes"):
        Inkling2LoopConfig.from_hf(bad(mtp_config={**good["mtp_config"], "target_layer_ids": [5, 23, 41, 24]}))
    with pytest.raises(ValueError, match="oracle implements"):
        Inkling2LoopConfig.from_hf(bad(t_n_shared_experts=1))
    with pytest.raises(ValueError, match="equal local/global"):
        Inkling2LoopConfig.from_hf(bad(t_swa_num_key_value_heads=4))


def test_config_json_round_trips_the_tiny_config():
    cfg = tiny_config()
    assert Inkling2LoopConfig.from_hf(config_json(cfg)) == cfg
