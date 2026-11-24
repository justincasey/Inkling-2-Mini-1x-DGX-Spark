"""PackWeights / load_exl3 on a small synthetic pack folder in the Mini layout: an EXL3 K=6 dense attention-like weight, a
K=6 unembed (9 strips: two slabs), K=4/K=5 mixed routed experts, a BF16 embedding and a plain norm, with an index."""

import json
import os

import numpy as np
import pytest
import torch

import exl3
import pack
import st
from inkling2_ref import Rounding, expert_ffn
from pack import PackExperts, PackIndex, PackWeights, expert_prefix, load_exl3
from test_exl3 import lin_of, make_group

ATTN = "model.llm.layers.0.attn.wq_du"
UNEMBED = "model.llm.unembed"
HID = 256
INTER = 128


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("minipack")
    groups = {ATTN: make_group(1, HID, 384, 6), UNEMBED: make_group(2, HID, 1152, 6)}
    experts = {
        (0, "gate_proj"): make_group(10, HID, INTER, 4), (0, "up_proj"): make_group(11, HID, INTER, 4),
        (0, "down_proj"): make_group(12, INTER, HID, 5),
        (1, "gate_proj"): make_group(13, HID, INTER, 5), (1, "up_proj"): make_group(14, HID, INTER, 5),
        (1, "down_proj"): make_group(15, INTER, HID, 5),
    }
    embed = torch.from_numpy(np.random.default_rng(3).standard_normal((300, HID)).astype(np.float32)).to(torch.bfloat16)
    norm = torch.from_numpy(np.linspace(0.5, 1.5, HID).astype(np.float32))
    f1 = {"model.llm.embed.weight": embed, "model.llm.layers.0.attn_norm.weight": norm}
    for p, g in groups.items():
        for part, a in g.items():
            f1[f"{p}.{part}"] = a
    f2 = {}
    for (e, proj), g in experts.items():
        for part, a in g.items():
            f2[f"{expert_prefix(0, e, proj)}.{part}"] = a
    st.write(str(root / "model-00001-of-00002.safetensors"), f1)
    st.write(str(root / "model-00002-of-00002.safetensors"), f2)
    wm = {n: "model-00001-of-00002.safetensors" for n in f1} | {n: "model-00002-of-00002.safetensors" for n in f2}
    with open(root / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": wm}, f)
    return str(root), groups, experts, embed


def dense_nn(g):
    w = exl3.decode_weight(g["trellis"], "mul1", g["suh"].astype(np.float64), g["svh"].astype(np.float64))
    return torch.from_numpy(np.ascontiguousarray(w.T))


def test_tensor_is_dense_decode_in_linear_layout(pack_dir):
    root, groups, *_ = pack_dir
    pw = PackWeights(PackIndex(root))
    w = pw.tensor(ATTN + ".weight")
    assert w.dtype == torch.float64 and tuple(w.shape) == (384, HID)
    assert torch.equal(w, dense_nn(groups[ATTN]))
    x = np.random.default_rng(4).standard_normal((3, HID))
    y = (torch.from_numpy(x) @ w.T).numpy()  # nn.Linear: x @ W_nn^T
    ref = lin_of(groups[ATTN])(x)
    np.testing.assert_allclose(y, ref, rtol=1e-11, atol=1e-11 * np.abs(ref).max())


def test_row_range_equals_slices_of_dense(pack_dir):
    root, groups, *_ = pack_dir
    pw = PackWeights(PackIndex(root))
    full = dense_nn(groups[UNEMBED])
    for a, b in [(0, 128), (130, 700), (1000, 1152), (0, 1152), (1151, 1152), (127, 129)]:
        got = pw.row_range(UNEMBED + ".weight", a, b)
        assert got.dtype == torch.float64 and torch.equal(got, full[a:b]), (a, b)
    assert torch.equal(pw.tensor(UNEMBED + ".weight"), full)
    with pytest.raises(IndexError):
        pw.row_range(UNEMBED + ".weight", 1100, 1153)


def test_rows_decode_each_strip_once_and_match(pack_dir, monkeypatch):
    root, groups, *_ = pack_dir
    pw = PackWeights(PackIndex(root))
    full = dense_nn(groups[UNEMBED])
    ids = torch.tensor([5, 1100, 5, 127, 128, 6, 1151])
    calls = []
    real = exl3.decode_weight
    monkeypatch.setattr(exl3, "decode_weight", lambda *a: calls.append(a[0].shape) or real(*a))
    got = pw.rows(UNEMBED + ".weight", ids)
    assert torch.equal(got, full[ids])
    assert len(calls) == 3  # ids fall in strips {0, 8, 1}: each distinct strip is decoded once
    assert all(s[1] == 8 for s in calls)


def test_slab_reads_only_its_trellis_columns(pack_dir, monkeypatch):
    root, groups, *_ = pack_dir
    pw = PackWeights(PackIndex(root))
    sizes = []
    real = st._pread
    monkeypatch.setattr(st, "_pread", lambda path, start, n: sizes.append(n) or real(path, start, n))
    pw.row_range(UNEMBED + ".weight", 128, 256)  # one strip = 8 tile columns of 192 bytes, for every one of 16 kt rows
    trellis_reads = [n for n in sizes if n == 8 * 192]
    assert len(trellis_reads) == HID // 16
    assert sum(sizes) < groups[UNEMBED]["trellis"].nbytes


def test_plain_tensors_unchanged(pack_dir):
    root, _, _, embed = pack_dir
    pw = PackWeights(PackIndex(root))
    t = pw.tensor("model.llm.embed.weight")
    assert t.dtype == torch.bfloat16 and torch.equal(t, embed)
    assert torch.equal(pw.row_range("model.llm.embed.weight", 10, 20), embed[10:20])
    assert torch.equal(pw.rows("model.llm.embed.weight", torch.tensor([7, 3, 7])), embed[[7, 3, 7]])
    assert pw.tensor("model.llm.layers.0.attn_norm.weight").dtype == torch.float32


def test_mixed_k_experts_load_with_their_own_k(pack_dir):
    root, _, experts, _ = pack_dir
    ix = PackIndex(root)
    got = [load_exl3(ix, expert_prefix(0, 0, p)).bits for p in pack.PROJS]
    assert got == [4.0, 4.0, 5.0]
    assert [load_exl3(ix, expert_prefix(0, 1, p)).bits for p in pack.PROJS] == [5.0, 5.0, 5.0]
    assert load_exl3(ix, ATTN).bits == 6.0


def test_pack_experts_ffn_with_mixed_k(pack_dir):
    root, *_ = pack_dir
    ix = PackIndex(root)
    x = np.random.default_rng(6).standard_normal((4, HID))
    outs = PackExperts(ix).ffn_many(0, [(0, x), (1, x)], 4.0, 25.0, Rounding())
    for e, y in zip((0, 1), outs):
        lins = [load_exl3(ix, expert_prefix(0, e, p)) for p in pack.PROJS]
        np.testing.assert_array_equal(y, expert_ffn(*lins, x, 4.0, 25.0, Rounding(), False))
        assert y.shape == (4, HID) and np.isfinite(y).all()


def test_group_markers(tmp_path):
    ix_root = str(tmp_path)
    g_none = make_group(20, 128, 128, 4, cb="3inst")  # no marker tensor: the 3inst codebook
    g_both = make_group(21, 128, 128, 4, cb="mul1")
    g_both["mcg"] = np.array(exl3.MARKER_MCG, dtype=np.uint32).view(np.int32).reshape(())
    g_bad = make_group(22, 128, 128, 4, cb="mul1")
    g_bad["mul1"] = np.array(7, dtype=np.int32).reshape(())
    t = {f"{p}.{part}": a for p, g in (("none", g_none), ("both", g_both), ("bad", g_bad)) for part, a in g.items()}
    st.write(os.path.join(ix_root, "m.safetensors"), t)
    ix = PackIndex(ix_root)
    lin = load_exl3(ix, "none")
    np.testing.assert_array_equal(lin.wq, exl3._f16(exl3.unpack(g_none["trellis"], "3inst")))
    with pytest.raises(ValueError, match="both"):
        load_exl3(ix, "both")
    with pytest.raises(ValueError, match="marker"):
        load_exl3(ix, "bad")


def test_cache_hits_budget_and_off(pack_dir):
    root, groups, *_ = pack_dir
    attn_bytes = 384 * HID * 8
    full = dense_nn(groups[ATTN])
    # off by default: equal values, never the same object, nothing held
    off = PackWeights(PackIndex(root))
    a, b = off.tensor(ATTN + ".weight"), off.tensor(ATTN + ".weight")
    assert a is not b and torch.equal(a, b) and off.stats() == {"hits": 0, "misses": 0, "held_bytes": 0}
    # budget for exactly one attention weight plus one 128-row slab of it
    slab_bytes = 128 * HID * 8
    pw = PackWeights(PackIndex(root), cache_bytes=attn_bytes + slab_bytes)
    t1 = pw.tensor(ATTN + ".weight")
    t2 = pw.tensor(ATTN + ".weight")
    assert t1 is t2 and torch.equal(t1, full)
    s1 = pw.row_range(UNEMBED + ".weight", 0, 128)
    assert pw.row_range(UNEMBED + ".weight", 0, 128) is s1
    assert pw.stats() == {"hits": 2, "misses": 2, "held_bytes": attn_bytes + slab_bytes}
    s2 = pw.row_range(UNEMBED + ".weight", 128, 256)  # evicts the least recently used: the attention weight
    assert pw.stats()["held_bytes"] <= attn_bytes + slab_bytes
    assert pw.stats()["held_bytes"] == 2 * slab_bytes
    assert pw.tensor(ATTN + ".weight") is not t1  # a miss again, decoded afresh
    assert torch.equal(pw.row_range(UNEMBED + ".weight", 128, 256), s2)
    # LRU order: touching s1 keeps it over s2
    tiny = PackWeights(PackIndex(root), cache_bytes=2 * slab_bytes)
    x0, x1 = tiny.row_range(UNEMBED + ".weight", 0, 128), tiny.row_range(UNEMBED + ".weight", 128, 256)
    assert tiny.row_range(UNEMBED + ".weight", 0, 128) is x0  # touch x0
    tiny.row_range(UNEMBED + ".weight", 256, 384)  # evicts x1, not x0
    assert tiny.row_range(UNEMBED + ".weight", 0, 128) is x0
    assert tiny.row_range(UNEMBED + ".weight", 128, 256) is not x1
    assert tiny.stats()["held_bytes"] <= 2 * slab_bytes
    # an entry above the budget is returned uncached
    small = PackWeights(PackIndex(root), cache_bytes=slab_bytes)
    w1, w2 = small.tensor(ATTN + ".weight"), small.tensor(ATTN + ".weight")
    assert w1 is not w2 and torch.equal(w1, w2) and small.stats()["held_bytes"] == 0 and small.stats()["hits"] == 0
    # plain tensors bypass the cache
    pw.tensor("model.llm.embed.weight")
    assert pw.stats()["held_bytes"] <= attn_bytes + slab_bytes


def test_files_restricted_index_for_the_drafter(pack_dir, tmp_path):
    """PackIndex(root, files=[...]) scans only those shards and ignores model.safetensors.index.json."""
    root, groups, *_ = pack_dir
    g = make_group(40, HID, 384, 4)
    mtp = {f"layers.0.self_attn.q_proj.{p}": a for p, a in g.items()}
    mtp["layers.0.input_layernorm.weight"] = torch.linspace(0.5, 1.5, HID)
    w1 = torch.from_numpy(np.random.default_rng(41).standard_normal((300, 16)).astype(np.float32)).to(torch.bfloat16)
    mtp["markov_head.markov_w1.weight"] = w1
    for fn in os.listdir(root):  # same folder as the index: copy it so mtp sits beside the main shards and the index
        os.symlink(os.path.join(root, fn), tmp_path / fn)
    st.write(str(tmp_path / "mtp.safetensors"), mtp)
    ix = PackIndex(str(tmp_path), files=["mtp.safetensors"])
    assert "model.llm.embed.weight" not in ix.weight_map and "layers.0.self_attn.q_proj.trellis" in ix
    pw = PackWeights(ix)
    w = pw.tensor("layers.0.self_attn.q_proj.weight")
    wref = exl3.decode_weight(g["trellis"], "mul1", g["suh"].astype(np.float64), g["svh"].astype(np.float64))
    assert w.dtype == torch.float64 and torch.equal(w, torch.from_numpy(np.ascontiguousarray(wref.T)))
    assert pw.tensor("layers.0.input_layernorm.weight").dtype == torch.float32
    assert torch.equal(pw.row_range("markov_head.markov_w1.weight", 5, 9), w1[5:9])
    assert torch.equal(pw.rows("markov_head.markov_w1.weight", torch.tensor([8, 2])), w1[[8, 2]])
    assert pw.tensor("markov_head.markov_w1.weight").dtype == torch.bfloat16
    with pytest.raises(KeyError):
        ix.entry("model.llm.embed.weight")
    assert load_exl3(ix, "layers.0.self_attn.q_proj").bits == 4.0
    # without `files`, the folder's index.json decides and does not know mtp.safetensors
    assert "layers.0.self_attn.q_proj.trellis" not in PackIndex(str(tmp_path)).weight_map
