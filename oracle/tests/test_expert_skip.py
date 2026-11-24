"""The B lane's adaptive expert skipping (`Emulation.expert_skip`, `skip_renorm`, `skip_picks`; docs/NUMERICS.md "B lane: expert skipping").

Unit tests of the lossy option: off is bit-identical to today's oracle, the dropped set equals the rule on crafted weights, renorm keeps the row
sum, the top pick is never dropped, and on the tiny looped model the experts are asked for exactly the kept picks."""

import numpy as np
import pytest
import torch

import inkling2_ref as ref
from inkling2_ref import F64, DenseExperts, DictWeights, Emulation, Inkling2LoopRef, Rounding, skip_picks
from synthetic import random_weights, tiny_config

VOCAB = 250
R = Rounding()


def w_rows(rows):
    return torch.tensor(rows, dtype=F64)


# ---------------------------------------------------------------------------------------------------------------- the rule


def test_dropped_set_is_below_t_times_the_top_weight():
    # t = 0.25, 0.5, 1 are exact in binary: the thresholds below are exactly 1.0 / 2.0 / 4.0 (row 0) and 0.5 / 1.0 / 2.0 (row 1)
    w = w_rows([[4.0, 2.0, 1.0, 0.9999, 0.2], [2.0, 2.0, 0.5, 0.4999, 0.0], [8.0, 0.0, 0.0, 0.0, 0.0]])
    _, keep = skip_picks(w, 0.25, False, R)  # a weight equal to the threshold stays: dropped only strictly below
    assert keep.tolist() == [[True, True, True, False, False], [True, True, True, False, False], [True, False, False, False, False]]
    _, keep = skip_picks(w, 0.5, False, R)
    assert keep.tolist() == [[True, True, False, False, False], [True, True, False, False, False], [True, False, False, False, False]]
    _, keep = skip_picks(w, 1.0, False, R)  # t = 1 keeps exactly the picks tied with the top
    assert keep.tolist() == [[True, False, False, False, False], [True, True, False, False, False], [True, False, False, False, False]]


def test_without_renorm_kept_weights_are_unchanged_and_dropped_are_zero():
    w = w_rows([[4.0, 2.0, 1.0, 0.8, 0.2]])
    out, keep = skip_picks(w, 0.3, False, R)  # threshold 1.2: keeps 4, 2
    assert out.tolist() == [[4.0, 2.0, 0.0, 0.0, 0.0]] and keep.tolist() == [[True, True, False, False, False]]


def test_renorm_keeps_the_row_sum_and_the_ratios():
    rng = np.random.default_rng(3)
    s = rng.random((200, 8)) + 0.01
    w = R.f32(torch.from_numpy(s / s.sum(-1, keepdims=True) * 8.0))
    for t in (0.15, 0.2, 0.3, 0.5):
        out, keep = skip_picks(w, t, True, R)
        assert (keep.sum(-1) < 8).any() and (keep.sum(-1) >= 1).all()
        # sum preserved to fp32 rounding of 8 products
        torch.testing.assert_close(out.sum(-1), w.sum(-1), rtol=2e-7, atol=0)
        assert (out[~keep] == 0).all() and (out[keep] > 0).all()
        # kept weights are proportional to the originals (one common factor a row) and larger than before
        fac = out[keep] / w[keep]
        assert (fac >= 1 - 1e-7).all()
        rows = keep.nonzero()[:, 0]
        for r in (0, 1, 2):
            f = fac[rows == r]
            assert float(f.max() - f.min()) < 1e-6 * float(f.max())
        # without renorm the same picks are dropped
        _, keep2 = skip_picks(w, t, False, R)
        assert torch.equal(keep, keep2)


def test_the_top_pick_is_never_dropped_and_off_returns_w_itself():
    rng = np.random.default_rng(5)
    for k in (2, 8):
        s = rng.random((500, k)) ** 4 + 1e-6
        w = R.f32(torch.from_numpy(s / s.sum(-1, keepdims=True) * k))
        for t in (0.15, 0.5, 0.99, 1.0):
            for renorm in (False, True):
                out, keep = skip_picks(w, t, renorm, R)
                top = w.argmax(-1)
                assert keep[torch.arange(w.shape[0]), top].all() and (out[torch.arange(w.shape[0]), top] > 0).all()
                assert (keep == (w >= t * w.max(-1, keepdim=True).values)).all()
    out, keep = skip_picks(w, 0.0, False, R)
    assert out is w and keep.all()


# ---------------------------------------------------------------------------------------------------------------- the spelling


def test_parse_and_tag():
    assert Emulation.parse("engine,expert_skip=0.2,skip_renorm") == Emulation(
        kv8=True, attn_bf16_p=True, moe_fp16=True, dense_fp16=True, expert_skip=0.2, skip_renorm=True)
    assert Emulation.parse("engine,expert_skip=0.3") == Emulation(**{**Emulation.engine().__dict__, "expert_skip": 0.3})
    assert Emulation.parse("expert_skip=0.15") == Emulation(expert_skip=0.15)
    assert Emulation.parse("kv8,expert_skip=0.2").tag == "kv8,expert_skip=0.2"
    assert Emulation.parse("engine,expert_skip=0.2,skip_renorm").tag == "engine,expert_skip=0.2,skip_renorm"
    assert Emulation(expert_skip=0.15).tag == "expert_skip=0.15"
    assert Emulation().tag == "none" and Emulation.engine().tag == "engine" and Emulation.parse("none") == Emulation()
    assert Emulation.engine() == Emulation(kv8=True, attn_bf16_p=True, moe_fp16=True, dense_fp16=True)  # off by default
    assert Emulation.engine().expert_skip == 0.0 and Emulation.engine().skip_renorm is False
    assert "expert_skip" not in Emulation.NAMES
    assert Emulation.parse("expert_skip=0") == Emulation()  # 0 is off
    for bad in ("expert_skip=1.5", "expert_skip=-0.1", "skip_renorm", "expert_skip", "expert_skip=x"):
        with pytest.raises(ValueError):
            Emulation.parse(bad)


# ---------------------------------------------------------------------------------------------------------------- the model


class Spy(DenseExperts):
    """Dense experts that record, per MoE call, {expert: rows asked of it}."""

    def __init__(self, weights):
        super().__init__(weights)
        self.calls: list[dict[int, int]] = []

    def ffn_many(self, layer, jobs, *a, **kw):
        self.calls.append({e: x.shape[0] for e, x in jobs})
        return super().ffn_many(layer, jobs, *a, **kw)


@pytest.fixture(scope="module")
def tiny():
    cfg = tiny_config(n_experts=8, top_k=4)
    src = DictWeights(random_weights(cfg, seed=7))

    def run(em, n=9, seed=2):
        spy = Spy(src)
        m = Inkling2LoopRef(cfg, src, spy, unembed_chunk=128, query_chunk=5, emulate=em)
        ids = torch.from_numpy(np.random.default_rng(seed).integers(0, VOCAB, n))
        return m.forward(ids, record=True), spy

    return cfg, run


def test_off_is_bit_identical(tiny):
    _, run = tiny
    base, spy0 = run(None)
    for em in (Emulation(), Emulation(expert_skip=0.0), Emulation(expert_skip=0.0, skip_renorm=True), Emulation.parse("none")):
        out, spy = run(em)
        for f in ("logits", "h_final", "hidden", "core_out", "router_idx", "router_w", "router_margin"):
            assert torch.equal(getattr(out, f), getattr(base, f)), f
        assert spy.calls == spy0.calls
    # a threshold that drops nothing here goes through the skip path and is bit-identical too (renorm factor is exactly 1)
    for renorm in (False, True):
        out, spy = run(Emulation(expert_skip=1e-12, skip_renorm=renorm))
        assert torch.equal(out.logits, base.logits) and torch.equal(out.hidden, base.hidden) and spy.calls == spy0.calls


@pytest.mark.parametrize("t", [0.65, 0.75, 0.9])
def test_experts_are_asked_for_exactly_the_kept_picks(tiny, t):
    cfg, run = tiny
    base, spy0 = run(None)
    for renorm in (False, True):
        out, spy = run(Emulation(expert_skip=t, skip_renorm=renorm))
        # the first MoE call sees identical inputs (nothing before it differs) and the router planes of the first visit are the router's own
        assert torch.equal(out.router_idx[0], base.router_idx[0]) and torch.equal(out.router_w[0], base.router_w[0])
        _, keep = skip_picks(base.router_w[0].to(F64), t, renorm, R)
        assert 0 < int((~keep).sum()) and keep.sum(-1).min() >= 1
        want: dict[int, int] = {}
        for e, kp in zip(base.router_idx[0].flatten().tolist(), keep.flatten().tolist()):
            if kp:
                want[e] = want.get(e, 0) + 1
        assert spy.calls[0] == want
        assert sum(spy.calls[0].values()) == int(keep.sum()) < sum(spy0.calls[0].values())
        # skipping changes the output (it is lossy) but not its shape or finiteness
        assert not torch.equal(out.logits, base.logits) and torch.isfinite(out.logits).all()


def test_the_first_moe_output_is_the_kept_picks_sum(tiny):
    """The first MoE visit's output: unchanged on rows that dropped nothing, and on the others the sum over the kept picks of
    w x expert(x) (renorm: with the rescaled weights), computed here one row and one expert at a time."""
    cfg, _ = tiny
    src = DictWeights(random_weights(cfg, seed=7))
    ids = torch.from_numpy(np.random.default_rng(2).integers(0, VOCAB, 9))
    t = 0.7

    def first_moe(em):
        m = Inkling2LoopRef(cfg, src, DenseExperts(src), unembed_chunk=128, query_chunk=5, emulate=em)
        seen, orig = [], m.moe

        def moe(v, x, rec):
            seen.append((v, x.clone(), orig(v, x, rec)))
            return seen[-1][2]

        m.moe = moe
        out = m.forward(ids, record=True)
        return m, seen[0], out

    _, (v, x, y_off), base = first_moe(None)
    for renorm in (False, True):
        m, (v2, x2, y), _ = first_moe(Emulation(expert_skip=t, skip_renorm=renorm))
        assert torch.equal(x, x2)  # nothing before the first MoE differs
        w, keep = skip_picks(base.router_w[0].to(F64), t, renorm, R)
        dropped = (~keep).any(-1)
        assert dropped.any() and not dropped.all()
        assert torch.equal(y[~dropped], y_off[~dropped])
        assert ((y - y_off).abs().sum(-1) > 0)[dropped].all()
        for row in range(x.shape[0]):
            acc = torch.zeros(x.shape[1], dtype=F64)
            for slot in range(base.router_idx.shape[-1]):
                if keep[row, slot]:
                    e = int(base.router_idx[0, row, slot])
                    (yy,) = m.experts.ffn_many(v.layer, [(e, x[row:row + 1].numpy())], cfg.gate_softcap, cfg.up_softcap, m.r)
                    acc += torch.from_numpy(yy[0]) * w[row, slot]
            torch.testing.assert_close(y[row], ref.bf16(acc), rtol=2e-2, atol=1e-3)
