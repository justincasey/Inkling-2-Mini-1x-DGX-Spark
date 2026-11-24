"""`Emulation.dense_fp16`: the rotated input of every dense EXL3 linear the oracle evaluates (attention q, r, gate, k, v, `wo`, the
unembed) rounded to fp16 before the W_q product, as the engine's dense kernels do (docs/NUMERICS.md "Engine emulation")."""

import numpy as np
import pytest
import torch

import inkling2_ref as ref
from inkling2_ref import F64, DictWeights, Emulation, Inkling2LoopRef
from pack import PackIndex, PackWeights
from synthetic import ExpertGroups, random_exl3_pack, tiny_config, write_pack

DENSE = ("wq_du", "wr_du", "gate_attn", "wk_dv", "wv_dv", "wo_ud")


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    cfg = tiny_config()
    plain, groups, reference = random_exl3_pack(cfg, seed=3)
    root = str(tmp_path_factory.mktemp("pack"))
    write_pack(root, cfg, plain, groups)
    return cfg, root, (reference, groups)


def model(cfg, source, groups=None, **flags):
    return Inkling2LoopRef(cfg, source, experts=ExpertGroups(groups) if groups else None, unembed_chunk=128, emulate=Emulation(**flags) if flags else None)


def test_names_engine_parse_and_tag():
    assert "dense_fp16" in Emulation.NAMES
    assert Emulation.engine().dense_fp16 and Emulation().dense_fp16 is False
    assert Emulation.parse("dense_fp16") == Emulation(dense_fp16=True)
    assert Emulation.parse("kv8,dense_fp16").tag == "kv8,dense_fp16"
    assert Emulation.parse("engine").tag == "engine" and Emulation.parse("none").tag == "none"
    with pytest.raises(ValueError):
        Emulation.parse("dense_fp32")
    assert ref.NUMERICS_VERSION == "2"


def test_lin_changes_exactly_by_the_fp16_rounding_of_the_rotated_input(pack):
    cfg, root, _ = pack
    src = PackWeights(PackIndex(root))
    name = "model.llm.layers.0.attn.wq_du.weight"
    lin = src.exl3_linear(name)
    x = ref.bf16(torch.from_numpy(np.random.default_rng(0).standard_normal((5, lin.k)) * 3))
    off, on = model(cfg, src), model(cfg, src, dense_fp16=True)
    # off: the exact float64 product of the decoded weight (bit-unchanged path)
    assert torch.equal(off.lin(x, name), ref.bf16(x @ src.tensor(name).T))
    # on: the rotated input through fp16, then the same product
    want = ref.bf16(torch.from_numpy(lin(x.numpy(), in_round=ref.np_f16)))
    assert torch.equal(on.lin(x, name), want)
    assert not torch.equal(on.lin(x, name), off.lin(x, name))  # the rounding is visible ...
    rel = (on.lin(x, name) - off.lin(x, name)).norm() / off.lin(x, name).norm()
    assert 0 < rel < 2e-2  # ... and small (fp16 has 11 bits, the output is bf16)
    # the rotation really is rounded: an input whose rotated form is exactly fp16-representable is unchanged
    xh = np.zeros((1, lin.k))
    xh[0, 0] = 1.0
    assert torch.equal(on.lin(torch.from_numpy(xh), name), off.lin(torch.from_numpy(xh), name))


def test_every_dense_exl3_linear_of_the_forward_is_emulated(pack):
    cfg, root, (_, groups) = pack
    seen = []

    class Spy(PackWeights):
        def exl3_linear(self, name):
            g = super().exl3_linear(name)
            if g is not None:
                seen.append(name)
            return g

    m = model(cfg, Spy(PackIndex(root)), groups, dense_fp16=True)
    m.forward(torch.arange(6) + 3)
    kinds = {n.split(".")[-2] for n in seen}
    assert kinds == set(DENSE) | {"unembed"}, kinds
    assert not any("sink" in n or "weight_proj" in n or "controllers" in n or "gate.weight" in n for n in seen)
    # off: the source is never asked
    seen.clear()
    model(cfg, Spy(PackIndex(root)), groups).forward(torch.arange(6) + 3)
    assert seen == []


def test_unembed_and_forward_follow_the_emulation_and_default_is_unchanged(pack):
    cfg, root, (reference, groups) = pack
    src = PackWeights(PackIndex(root))
    ids = torch.arange(7) + 5
    base = model(cfg, src, groups).forward(ids)
    twin = model(cfg, DictWeights(reference), groups).forward(ids)
    assert torch.equal(base.logits, twin.logits) and torch.equal(base.h_final, twin.h_final)  # default off: bit-unchanged
    on = model(cfg, src, groups, dense_fp16=True).forward(ids)
    assert torch.isfinite(on.logits).all() and not torch.equal(on.logits, base.logits)
    # the head: soft-capped fp32 product of the fp16-rounded rotated final-norm output, first unpadded_vocab columns
    h = on.h_final.to(F64)
    z = torch.from_numpy(src.exl3_linear("model.llm.unembed.weight")(h.numpy(), in_round=ref.np_f16))
    z = ref.f32(z[:, :cfg.unpadded_vocab_size])
    assert torch.equal(on.logits, ref.f32(cfg.final_softcap * torch.tanh(z / cfg.final_softcap)).to(torch.float32))


def test_a_source_without_exl3_groups_is_unaffected(pack):
    cfg, _, (reference, groups) = pack
    ids = torch.arange(6) + 2
    a = model(cfg, DictWeights(reference), groups).forward(ids)
    b = model(cfg, DictWeights(reference), groups, dense_fp16=True).forward(ids)
    assert torch.equal(a.logits, b.logits) and torch.equal(a.h_final, b.h_final)


def test_unembed_probe_is_exact_against_an_engine_with_the_rotation(pack):
    """`unembed_probe` on the emulating oracle's own h_final and logits: the fp16 oracle matches to 0, the exact one does not."""
    import unembed_probe

    cfg, root, (_, groups) = pack
    src = PackWeights(PackIndex(root))
    out = model(cfg, src, groups, dense_fp16=True).forward(torch.arange(9) + 4)
    lin = src.exl3_linear("model.llm.unembed.weight")
    res = unembed_probe.probe(lin, out.h_final.to(F64).numpy(), out.logits.to(F64).numpy(), cfg.unpadded_vocab_size)["rows"]
    assert len(res) == 9 and all(r["fp16_rel"] == 0 and r["fp16_argmax_equal"] and r["fp16_gap_err"] == 0 for r in res)
    assert max(r["exact_rel"] for r in res) > 0
