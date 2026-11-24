"""The real-pack code path on a synthetic pack folder in the Mini layout (tiny looped config; attention linears and the
unembed EXL3 6-bit, routed experts EXL3 4- and 5-bit mixed inside a layer, every codebook `mul1`; two model shards and an
index): loader, per-tensor K, dense EXL3 decode through the model, expert decode, worker pool, error paths."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

import exl3
import pack
from inkling2_ref import DictWeights, Inkling2LoopConfig, Inkling2LoopRef
from pack import PackExperts, PackIndex, PackWeights, load_exl3
from synthetic import ExpertGroups, expert_bits, random_exl3_pack, tiny_config, write_pack


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pack"))
    cfg = tiny_config()
    plain, groups, reference = random_exl3_pack(cfg, seed=2)
    write_pack(root, cfg, plain, groups)
    return root, cfg, groups, reference


def model_for(root, workers=0, cache_bytes=0):
    with open(os.path.join(root, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    ix = PackIndex(root)
    return Inkling2LoopRef(cfg, PackWeights(ix, cache_bytes), PackExperts(ix, workers=workers), unembed_chunk=100)


def ids(n, seed):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 250, n))


def test_config_round_trip(pack_dir):
    root, cfg, _, _ = pack_dir
    with open(os.path.join(root, "config.json")) as f:
        assert Inkling2LoopConfig.from_hf(json.load(f)) == cfg


def test_k_is_read_per_tensor_from_the_trellis(pack_dir):
    """Attention and unembed groups are 6-bit, expert projections 4 or 5 bit (mixed inside one layer, even inside one
    expert: only down_proj may differ), every group `mul1`."""
    root, cfg, groups, _ = pack_dir
    ix = PackIndex(root)
    attn = load_exl3(ix, "model.llm.layers.1.attn.wq_du")
    assert exl3.bits_from_trellis(groups["model.llm.layers.1.attn.wq_du"]["trellis"]) == 6.0
    assert (attn.k, attn.n) == (cfg.hidden_size, cfg.num_heads * cfg.head_dim)
    seen = set()
    for layer in range(cfg.num_layers):
        for e in range(cfg.n_experts):
            for proj in pack.PROJS:
                p = pack.expert_prefix(layer, e, proj)
                k = exl3.bits_from_trellis(st_trellis(ix, p))
                assert k == expert_bits(layer, e, proj)
                seen.add(k)
    assert seen == {4.0, 5.0}
    mixed = [(layer, e) for layer in range(cfg.num_layers) for e in range(cfg.n_experts)
             if len({expert_bits(layer, e, p) for p in pack.PROJS}) > 1]
    assert mixed  # an expert with a different K in down_proj
    gate = load_exl3(ix, "model.llm.layers.3.mlp.experts.2.gate_proj")
    down = load_exl3(ix, "model.llm.layers.3.mlp.experts.2.down_proj")
    assert (gate.k, gate.n) == (cfg.hidden_size, cfg.moe_intermediate)
    assert (down.k, down.n) == (cfg.moe_intermediate, cfg.hidden_size)
    assert ix.weight_map["model.llm.layers.3.mlp.experts.2.gate_proj.trellis"] == "model-00002-of-00002.safetensors"
    assert ix.weight_map["model.llm.layers.3.mlp_norm.weight"] == "model-00001-of-00002.safetensors"


def st_trellis(ix, prefix):
    import st

    return st.read_numpy(ix.entry(prefix + ".trellis"))


def test_dense_exl3_weight_source_equals_the_decoded_reference(pack_dir):
    """`PackWeights.tensor` of an EXL3-dense name is the float64 effective weight in nn.Linear layout, the unembed slab
    the matching rows, plain tensors come back as stored."""
    root, cfg, _, reference = pack_dir
    w = PackWeights(PackIndex(root))
    for name in ("model.llm.layers.0.attn.wq_du.weight", "model.llm.layers.7.attn.wo_ud.weight",
                 "model.llm.layers.5.attn.wk_dv.weight"):
        assert torch.equal(w.tensor(name), reference[name])
    unembed = reference["model.llm.unembed.weight"]
    assert torch.equal(w.row_range("model.llm.unembed.weight", 128, 256), unembed[128:256])
    assert torch.equal(w.rows("model.llm.unembed.weight", torch.tensor([5, 130, 5])), unembed[[5, 130, 5]])
    assert w.tensor("model.llm.layers.2.mlp_norm.weight").dtype == torch.bfloat16
    assert torch.equal(w.tensor("model.llm.layers.2.mlp_norm.weight"), reference["model.llm.layers.2.mlp_norm.weight"])


def test_pack_model_equals_the_in_memory_twin_bit_for_bit(pack_dir):
    """The whole model through the pack folder (positional reads, EXL3 decode, expert workers' code path) equals the
    same model over DictWeights of the decoded weights and in-memory expert groups: logits, every pass's hidden state
    and the routes."""
    root, cfg, groups, reference = pack_dir
    x = ids(13, seed=21)
    got = model_for(root).forward(x, record=True)
    twin = Inkling2LoopRef(cfg, DictWeights(reference), ExpertGroups(groups), unembed_chunk=100)
    want = twin.forward(x, record=True)
    assert torch.isfinite(got.logits).all()
    assert torch.equal(got.logits, want.logits) and torch.equal(got.hidden, want.hidden)
    assert torch.equal(got.router_idx, want.router_idx) and torch.equal(got.core_out, want.core_out)
    # the routed experts really are mixed 4/5 bit in the layers that ran
    assert len({expert_bits(layer, int(e), "down_proj") for layer in range(cfg.num_layers)
                for e in got.router_idx[:, :, :].unique()}) == 2


def test_pack_prefill_decode_equals_recompute(pack_dir):
    root, _, _, _ = pack_dir
    m = model_for(root)
    x = ids(14, seed=11)
    full = m.forward(x, record=True)
    cache = m.new_cache()
    parts = [m.forward(x[:6], cache, record=True)] + [m.forward(x[j:j + 1], cache, record=True) for j in range(6, 14)]
    assert torch.equal(torch.cat([p.logits for p in parts]), full.logits)
    assert torch.equal(torch.cat([p.hidden for p in parts], 1), full.hidden)


def test_weight_cache_changes_nothing_but_work(pack_dir):
    """An LRU of decoded dense weights returns identical results and serves the repeated core passes from memory."""
    root, _, _, _ = pack_dir
    x = ids(7, seed=22)
    a = model_for(root).forward(x)
    m = model_for(root, cache_bytes=1 << 30)
    b = m.forward(x)
    assert torch.equal(a.logits, b.logits)
    st = m.w.stats()
    assert st["hits"] > 0 and st["held_bytes"] <= 1 << 30


def test_worker_pool_matches_in_process(pack_dir):
    root, _, _, _ = pack_dir
    x = ids(7, seed=12)
    a = model_for(root).forward(x)
    mp = model_for(root, workers=2)
    try:
        b = mp.forward(x)
    finally:
        mp.experts.close()
    assert torch.equal(a.logits, b.logits)


def test_workers_exit_when_their_parent_is_killed(pack_dir, tmp_path):
    """A killed generator must not leave its spawn workers behind (they held memory on the oracle host)."""
    root, _, _, _ = pack_dir
    here = os.path.dirname(os.path.abspath(pack.__file__))
    script = tmp_path / "parent.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {here!r})\n"
        "from pack import PackExperts, PackIndex\n"
        "if __name__ == '__main__':\n"
        f"    ex = PackExperts(PackIndex({root!r}), workers=2)\n"
        "    ex.pool.submit(int, 1).result()\n"
        "    print(*ex.pool._processes, flush=True)\n"
        "    time.sleep(600)\n")
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    pids = [int(v) for v in p.stdout.readline().split()]
    assert pids
    p.kill()
    p.wait()

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    deadline = time.time() + 20
    while any(alive(q) for q in pids) and time.time() < deadline:
        time.sleep(0.2)
    assert not any(alive(q) for q in pids)


def test_missing_shard_is_an_error(pack_dir, tmp_path):
    """A folder without the shard that holds the EXL3 tensors (a partial download) fails with a message naming the shard,
    not with a wrong number."""
    root, _, _, _ = pack_dir
    for fn in os.listdir(root):
        if fn != "model-00002-of-00002.safetensors":
            os.symlink(os.path.join(root, fn), tmp_path / fn)
    m = model_for(str(tmp_path))
    with pytest.raises(KeyError, match="does not hold"):
        m.forward([1, 2, 3, 4, 5, 6])


def test_unknown_tensor_is_an_error(pack_dir):
    root, _, _, _ = pack_dir
    with pytest.raises(KeyError, match="not in the pack index"):
        PackWeights(PackIndex(root)).tensor("model.llm.layers.0.attn.no_such.weight")
