"""The engine-numerics emulation (inkling2_ref.Emulation, docs/ORACLE-GAP.md) on the tiny looped model: the int8 KV store
of the global passes, the attention core's tiled online softmax with bf16 P, and the fp16 EXL3 GEMM inputs of the
experts."""

import json
import os

import numpy as np
import pytest
import torch

import exl3
import inkling2_ref as ref
from inkling2_ref import F64, DenseExperts, DictWeights, Emulation, Inkling2LoopConfig, Inkling2LoopRef, Rounding
from pack import PackExperts, PackIndex, PackWeights, load_exl3
from synthetic import random_exl3_pack, random_weights, tiny_config, write_pack

# tiny-model chunk/tile sizes: several chunks and tiles inside the window (8) and past it
TINY = dict(tile=2, chunk_global=4, chunk_local=4)
VOCAB = 250  # the tiny config's unpadded vocabulary


def bf16_rows(shape, seed, spread=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    if spread:
        x = x * 10.0 ** rng.uniform(-spread, spread, shape[:-1] + (1,))
    return ref.bf16(torch.from_numpy(x))


def ids(n, seed):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, VOCAB, n))


@pytest.fixture(scope="module")
def dense():
    """(config, factory): the tiny looped model over dense bf16 weights and dense experts."""
    cfg = tiny_config()
    src = DictWeights(random_weights(cfg, seed=1))

    def make(rounding=None, **flags):
        em = Emulation(**flags, **TINY) if flags else None
        return Inkling2LoopRef(cfg, src, DenseExperts(src), rounding=rounding, unembed_chunk=128, query_chunk=5,
                               emulate=em)

    return cfg, make


# ---------------------------------------------------------------------------------------------------------------- kv8


def test_kv8_matches_its_definition():
    """e = the smallest exponent with amax / 2^e <= 127, clamped to [-24, 8]; q = rint(x / 2^e) saturated to +-127."""
    x = bf16_rows((4000, 128), seed=1, spread=7.0)
    got = ref.kv8_roundtrip(x)
    amax = x.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2(amax / 127)).clamp(-24, 8)  # exact: amax / 127 is a power of two only at 127 * 2^k
    want = torch.round(x / torch.exp2(e)).clamp(-127, 127) * torch.exp2(e)
    assert torch.equal(got, want)
    assert torch.equal(ref.bf16(got), got)  # 7 significant bits: exact in bf16
    unsat = (amax <= 127 * 256.0).squeeze(-1)  # above that e is clamped at 8 and the row saturates
    assert unsat.sum() > 1000 and ((got - x).abs() <= torch.exp2(e) / 2)[unsat].all()


def test_kv8_boundaries():
    t = lambda *v: torch.tensor([list(v)], dtype=F64)  # noqa: E731
    # 1.984375 = 127 * 2^-6 keeps e = -6 and q = 127; one bf16 step above it needs e = -5
    assert torch.equal(ref.kv8_roundtrip(t(1.984375, 0.5)), t(1.984375, 0.5))
    assert torch.equal(ref.kv8_roundtrip(t(1.9921875, 0.5)), t(2.0, 0.5))
    assert torch.equal(ref.kv8_roundtrip(t(1.984375, 2.5 / 64)), t(1.984375, 2.0 / 64))  # 2.5 rounds to even
    assert torch.equal(ref.kv8_roundtrip(t(0.0, 0.0)), t(0.0, 0.0))
    assert torch.equal(ref.kv8_roundtrip(t(2.0 ** 20, 1.0)), t(127 * 256.0, 0.0))  # e clamped at 8, saturated
    assert torch.equal(ref.kv8_roundtrip(t(2.0 ** -40, -(2.0 ** -41))), t(0.0, 0.0))  # e clamped at -24


# ---------------------------------------------------------------------------------------------- tiled attention


def attn_case(t, s, k_start, hq=8, hkv=4, d=16, seed=0):
    rng = np.random.default_rng(seed)
    bf = lambda *sh: ref.bf16(torch.from_numpy(rng.standard_normal(sh)))  # noqa: E731
    q, k, v = bf(t, hq, d) * 2, bf(s, hkv, d) * 2, bf(s, hkv, d)
    bias, gate, sink = bf(hq, t, s), bf(t, hq, d), bf(t, hq)
    k_pos = torch.arange(k_start, k_start + s)
    q_pos = k_pos[-t:]
    return q, k, v, bias, gate, sink, q_pos, k_pos


@pytest.mark.parametrize("window,k_start", [(None, 0), (8, 0), (8, 5)])
def test_tiled_attention_without_rounding_is_attention(window, k_start):
    args = attn_case(t=7, s=15, k_start=k_start)
    want = ref.attention_core(*args, window, Rounding.exact())
    for chunk, tile in ((4, 2), (8, 4), (64, 64), (3, 1)):
        got = ref.attention_core_tiled(*args, window, Rounding.exact(), chunk=chunk, tile=tile)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_tiled_attention_bf16_p_is_bf16_level_and_depends_on_tiles():
    args = attn_case(t=7, s=15, k_start=3, seed=4)
    exact = ref.attention_core(*args, 8, Rounding())
    a = ref.attention_core_tiled(*args, 8, Rounding(), chunk=4, tile=2)
    b = ref.attention_core_tiled(*args, 8, Rounding(), chunk=8, tile=4)
    rel = lambda x, y: float((x - y).norm() / y.norm())  # noqa: E731
    assert 0 < rel(a, exact) < 4e-3 and 0 < rel(b, exact) < 4e-3
    assert not torch.equal(a, b)  # P is rounded against the running maximum of its own tile sequence


# ---------------------------------------------------------------------------------------------------- whole model


def test_tiled_attention_is_the_same_structure_in_the_looped_model(dense):
    """Under exact rounding the tiled attention is the plain one in all 16 passes (shared core weights, a K/V per pass)."""
    _, make = dense
    x = ids(19, seed=11)
    plain = make(rounding=Rounding.exact()).forward(x, record=True)
    tiled = make(rounding=Rounding.exact(), attn_bf16_p=True).forward(x, record=True)
    torch.testing.assert_close(tiled.hidden, plain.hidden, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(tiled.logits, plain.logits, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("splits", [[5], [1, 1, 3, 9], [10]])
def test_emulated_prefill_then_decode_equals_recompute(dense, splits):
    cfg, make = dense
    m = make(kv8=True, attn_bf16_p=True)
    x = ids(19, seed=7)
    full = m.forward(x, record=True)
    cache, at, parts = m.new_cache(), 0, []
    for n in splits + [len(x) - sum(splits)]:
        parts.append(m.forward(x[at:at + n], cache, record=True))
        at += n
    assert torch.equal(torch.cat([p.logits for p in parts]), full.logits)
    assert torch.equal(torch.cat([p.hidden for p in parts], 1), full.hidden)
    assert torch.equal(torch.cat([p.core_out for p in parts], 1), full.core_out)
    assert full.hidden.shape[0] == cfg.n_visits + 1


def test_kv8_touches_global_passes_only(dense):
    cfg, make = dense
    x = ids(13, seed=9)
    base, kv8 = make(), make(kv8=True)
    for m in (base, kv8):
        m.taps = {}
        m.forward(x)
    visits = cfg.visits()
    first_global = min(v.index for v in visits if v.is_global)
    assert first_global > 0
    for v in visits[:first_global]:  # identical inputs, local attention: identical bits
        assert torch.equal(base.taps[v.index]["core"], kv8.taps[v.index]["core"])
    assert not torch.equal(base.taps[first_global]["core"], kv8.taps[first_global]["core"])


def test_emulation_costs_a_rounding_on_identical_inputs(dense):
    """On the first global pass (whose input the emulation has not touched yet) int8 KV and bf16 P each change the
    attention core by a rounding, not a different computation. (Whole-model drift is not bounded here: the synthetic
    router's margins are small enough that one rounding flips a route.)"""
    _, make = dense
    x = ids(17, seed=5)
    taps = {}
    for name, flags in (("base", {}), ("kv8", {"kv8": True}), ("p", {"attn_bf16_p": True})):
        m = make(**flags)
        m.taps = {}
        out = m.forward(x)
        assert torch.isfinite(out.logits).all()
        taps[name] = m.taps[1]["core"]  # pass 1 = layer 1, the first global layer
    for name in ("kv8", "p"):
        rel = float((taps[name] - taps["base"]).norm() / taps["base"].norm())
        assert 0 < rel < 4e-2, name



def test_default_tile_and_chunks_are_the_mini_attention_kernels():
    """The engine's attention kernels (REXL3 `inkling2_attn.cu`, `Emulation.attn_bf16_p`) cut keys into absolute chunks of 512 (global)
    and 256 (sliding) made of 64-key tiles at the Small's shape AND at the Mini's 24 / 8 / 3 heads: the launch geometry changed
    (two prologue CTAs a row, row blocks of 16 / 32 rows) but none of the sizes that enter the arithmetic did, so the defaults
    are the Mini kernels' and need no knob change. A change of the kernels' tile or chunk constants must change these defaults
    (and the fixtures that carry the emulation) together."""
    em = Emulation()
    assert (em.tile, em.chunk_global, em.chunk_local) == (64, 512, 256)
    assert Emulation.engine() == Emulation(kv8=True, attn_bf16_p=True, moe_fp16=True, dense_fp16=True)
    cfg = Inkling2LoopConfig()
    # the log scaling floor the kernels take from the config (8,192 at the Mini) is the oracle's own field
    assert (cfg.num_heads, cfg.num_kv_heads, cfg.log_scaling_n_floor, cfg.log_scaling_alpha) == (24, 8, 8192.0, 0.1)


def test_emulation_names():
    assert Emulation.parse("none") == Emulation() and Emulation().tag == "none"
    assert Emulation.parse("engine") == Emulation.engine() and Emulation.engine().tag == "engine"
    assert Emulation.parse("kv8, moe_fp16") == Emulation(kv8=True, moe_fp16=True)
    assert Emulation(kv8=True, moe_fp16=True).tag == "kv8,moe_fp16"
    with pytest.raises(ValueError):
        Emulation.parse("fp8")


def test_dense_experts_refuse_fp16_inputs(dense):
    _, make = dense
    m = make(moe_fp16=True)
    with pytest.raises(ValueError, match="EXL3"):
        m.forward(torch.tensor([1, 2, 3]))


# ------------------------------------------------------------------------------------------------- fp16 experts


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pack"))
    cfg = tiny_config()
    plain, groups, _ = random_exl3_pack(cfg, seed=2)
    write_pack(root, cfg, plain, groups)
    return root


def test_exl3_fp16_input_is_the_rotated_input_rounded(pack_dir):
    lin = load_exl3(PackIndex(pack_dir), "model.llm.layers.2.mlp.experts.3.gate_proj")
    x = bf16_rows((9, lin.k), seed=3).numpy()
    xh = exl3.fwht_128(x * lin.suh[None, :], axis=1)
    want = exl3.fwht_128(ref.np_f16(xh) @ lin.wq, axis=1) * lin.svh[None, :]
    np.testing.assert_array_equal(lin(x, in_round=ref.np_f16), want)
    plain = lin(x)
    rel = np.linalg.norm(lin(x, in_round=ref.np_f16) - plain) / np.linalg.norm(plain)
    assert 0 < rel < 2e-3


def test_fp16_experts_worker_pool_matches_in_process(pack_dir):
    with open(os.path.join(pack_dir, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    ix = PackIndex(pack_dir)
    x = ids(9, seed=5)
    outs = []
    for workers in (0, 2):
        experts = PackExperts(ix, workers=workers)
        try:
            m = Inkling2LoopRef(cfg, PackWeights(ix), experts, unembed_chunk=100, emulate=Emulation(moe_fp16=True))
            outs.append(m.forward(x, record=True))
        finally:
            experts.close()
    assert torch.equal(outs[0].logits, outs[1].logits) and torch.equal(outs[0].hidden, outs[1].hidden)
    plain = Inkling2LoopRef(cfg, PackWeights(ix), PackExperts(ix), unembed_chunk=100).forward(x, record=True)
    assert not torch.equal(plain.hidden, outs[0].hidden)
