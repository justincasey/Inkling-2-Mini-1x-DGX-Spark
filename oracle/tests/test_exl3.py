"""EXL3 numpy port vs REXL3's CPU reference (gb-exl3 via the exl3dq CLI, built live): codebooks, W_q bits and the dense
decode bit for bit at K = 4, 5, 6 with the mul1 codebook, the layer forward to float64 rounding, committed golden digests
of gb-exl3's own output, an independent bitstream round trip, column-slab decoding, and marker handling.
Everything is seeded and offline."""

import hashlib
import json
import os
import subprocess

import numpy as np
import pytest

import exl3
import st

ORACLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE = os.path.join(ORACLE, "exl3dq")
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_exl3.json")


def make_group(seed: int, k: int, n: int, bits: int, cb: str = "mul1") -> dict[str, np.ndarray]:
    """A seeded random EXL3 group: trellis int16 [K/16, N/16, 16*bits], fp16 scales, the codebook's marker."""
    rng = np.random.default_rng(seed)
    trellis = rng.integers(0, 65536, size=(k // 16, n // 16, 16 * bits), dtype=np.uint16).view(np.int16)
    out = {
        "trellis": trellis,
        "suh": (rng.choice([-1.0, 1.0], size=k) * 0.05 * (1 + 0.3 * rng.random(k))).astype(np.float16),
        "svh": (rng.choice([-1.0, 1.0], size=n) * 0.05 * (1 + 0.3 * rng.random(n))).astype(np.float16),
    }
    marker = {"mul1": exl3.MARKER_MUL1, "mcg": exl3.MARKER_MCG}.get(cb)
    if marker is not None:
        out[cb] = np.array(marker, dtype=np.uint32).view(np.int32).reshape(())
    return out


def write_groups(d, groups, name="groups.safetensors"):
    tensors = {f"{prefix}.{part}": a for prefix, g in groups.items() for part, a in g.items()}
    st.write(os.path.join(d, name), tensors)


def lin_of(g, cb="mul1"):
    return exl3.Exl3Linear(g["trellis"], cb, g["suh"].astype(np.float64), g["svh"].astype(np.float64))


@pytest.fixture(scope="session")
def exl3dq():
    subprocess.run(["cargo", "build", "--release", "--locked", "--quiet"], cwd=CRATE, check=True)
    return os.path.join(CRATE, "target", "release", "exl3dq")


def run(bin_, *args):
    subprocess.run([bin_, *map(str, args)], check=True)


CASES = [  # (prefix, K_in, N_out, bits, seed)
    ("k4_sq", 256, 256, 4, 11),
    ("k4_wide", 128, 1024, 4, 12),
    ("k5_n384", 256, 384, 5, 13),
    ("k5_tall", 640, 128, 5, 14),
    ("k6_n384", 384, 384, 6, 15),
    ("k6_tall", 1024, 128, 6, 16),
]
GOLDEN_CASES = [("g4", 256, 384, 4, 4004), ("g5", 256, 384, 5, 5005), ("g6", 256, 384, 6, 6006)]


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("exl3")
    groups = {p: make_group(seed, k, n, bits) for p, k, n, bits, seed in CASES}
    write_groups(d, groups)
    return d, groups


def test_codebooks_match_rust(exl3dq, tmp_path):
    for cb in exl3.CODEBOOKS:
        out = tmp_path / f"{cb}.npy"
        run(exl3dq, "table", cb, out)
        np.testing.assert_array_equal(np.load(out), exl3.codebook(cb))


def test_groups_accepted_by_gb_exl3(exl3dq, synthetic_dir):
    d, _ = synthetic_dir
    out = subprocess.run([exl3dq, "groups", d], check=True, capture_output=True, text=True).stdout
    got = {j["prefix"]: j for j in map(json.loads, out.splitlines())}
    for p, k, n, bits, _ in CASES:
        assert got[p]["codebook"] == "mul1" and (got[p]["k"], got[p]["n"]) == (k, n) and got[p]["bits"] == bits


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_unpack_and_dense_bit_exact_vs_rust(exl3dq, synthetic_dir, tmp_path, case):
    d, groups = synthetic_dir
    prefix, k, n, bits, _ = case
    g = groups[prefix]
    assert exl3.bits_from_trellis(g["trellis"]) == float(bits)
    run(exl3dq, "wq", d, prefix, tmp_path / "wq.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "wq.npy"), exl3.unpack(g["trellis"], "mul1"))
    run(exl3dq, "dense", d, prefix, tmp_path / "w.npy")
    w_rust = np.load(tmp_path / "w.npy")
    w_py = exl3.decode_weight(g["trellis"], "mul1", g["suh"].astype(np.float64), g["svh"].astype(np.float64))
    assert w_py.shape == (k, n)
    np.testing.assert_array_equal(w_py.view(np.uint64), w_rust.view(np.uint64))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_forward_matches_rust(exl3dq, synthetic_dir, tmp_path, case):
    d, groups = synthetic_dir
    prefix, k, n, bits, _ = case
    x = np.random.default_rng(1).standard_normal((5, k))
    np.save(tmp_path / "x.npy", x)
    run(exl3dq, "forward", d, prefix, tmp_path / "x.npy", tmp_path / "y.npy")
    lin = lin_of(groups[prefix])
    assert lin.bits == float(bits) and (lin.k, lin.n) == (k, n)
    y = lin(x)
    np.testing.assert_allclose(y, np.load(tmp_path / "y.npy"), rtol=1e-12, atol=1e-12 * np.abs(y).max())


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_golden_digests_of_gb_exl3_output():
    """sha256 of gb-exl3's W_q fp16 bytes and dense float64 bytes (recorded once from `exl3dq wq` / `exl3dq dense`)
    against the numpy decode of the same seeded groups. No cargo needed."""
    with open(GOLDEN) as f:
        golden = json.load(f)
    assert sorted(golden["groups"]) == sorted(c[0] for c in GOLDEN_CASES)
    for p, k, n, bits, seed in GOLDEN_CASES:
        g = make_group(seed, k, n, bits)
        want = golden["groups"][p]
        assert (want["k"], want["n"], want["bits"], want["seed"]) == (k, n, bits, seed)
        assert sha(exl3.unpack(g["trellis"], "mul1")) == want["wq_sha256"]
        w = exl3.decode_weight(g["trellis"], "mul1", g["suh"].astype(np.float64), g["svh"].astype(np.float64))
        assert sha(w) == want["dense_sha256"]


def test_golden_digests_regenerate_from_rust(exl3dq, tmp_path):
    """The committed digests are what the live Rust decoder produces now (guards the golden file itself)."""
    with open(GOLDEN) as f:
        golden = json.load(f)["groups"]
    groups = {p: make_group(seed, k, n, bits) for p, k, n, bits, seed in GOLDEN_CASES}
    write_groups(tmp_path, groups)
    for p in groups:
        run(exl3dq, "wq", tmp_path, p, tmp_path / "wq.npy")
        run(exl3dq, "dense", tmp_path, p, tmp_path / "w.npy")
        assert sha(np.load(tmp_path / "wq.npy")) == golden[p]["wq_sha256"]
        assert sha(np.load(tmp_path / "w.npy")) == golden[p]["dense_sha256"]


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_independent_bitstream_round_trip(bits):
    """Pack a random ring of 256*K bits into a tile with a separate python packer (layout of the doc comment of
    crates/exl3/src/encode.rs) and require `states`/`unpack` to recover each position's window, derived bit by bit."""
    rng = np.random.default_rng(700 + bits)
    ring_len = 256 * bits
    ring = rng.integers(0, 2, size=ring_len)
    nw = ring_len // 32
    assert nw == 8 * bits
    words = []
    for i in range(nw):  # ring bit 32 i + j is bit 31 - j of word i
        w = 0
        for j in range(32):
            w |= int(ring[32 * i + j]) << (31 - j)
        words.append(w)
    i16 = []
    for w in words:  # two int16 per word, little-endian, low word first
        i16 += [w & 0xFFFF, (w >> 16) & 0xFFFF]
    tile = np.array(i16, dtype=np.uint16).view(np.int16)
    assert tile.shape == (16 * bits,)
    want = []
    for p in range(256):
        end = (p + 1) * bits  # E(p) for an integer width
        state = 0
        for k in range(16):  # ring bit E - 16 + k is state bit 15 - k (big-endian window), the ring wraps
            state |= int(ring[(end - 16 + k) % ring_len]) << (15 - k)
        want.append(state)
    want = np.array(want, dtype=np.uint64)
    np.testing.assert_array_equal(exl3.states(tile[None], 2 * bits)[0], want)
    wq = exl3.unpack(tile.reshape(1, 1, -1), "mul1")
    r, c = exl3.tile_positions()
    expect = np.empty((16, 16), dtype=np.uint16)
    expect[r, c] = exl3.codebook("mul1")[want.astype(np.int64)]
    np.testing.assert_array_equal(wq, expect)


def test_stream_ends_are_integer_widths():
    for bits in (4, 5, 6):
        np.testing.assert_array_equal(exl3.stream_ends(2 * bits), np.arange(1, 257) * bits)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_decode_cols_equals_dense_slices(synthetic_dir, case):
    _, groups = synthetic_dir
    prefix, k, n, bits, _ = case
    g = groups[prefix]
    suh, svh = g["suh"].astype(np.float64), g["svh"].astype(np.float64)
    w = exl3.decode_weight(g["trellis"], "mul1", suh, svh)
    spans = [(0, 128), (n - 128, n), (0, n)] + ([(128, 384)] if n >= 384 else [])
    for c0, c1 in spans:
        a = exl3.decode_cols(g["trellis"], "mul1", suh, svh, c0, c1)
        np.testing.assert_array_equal(a.view(np.uint64), w[:, c0:c1].view(np.uint64))
        b = exl3.decode_cols(g["trellis"][:, c0 // 16:c1 // 16], "mul1", suh, svh[c0:c1], 0, c1 - c0)  # a sliced group
        np.testing.assert_array_equal(b.view(np.uint64), a.view(np.uint64))


def test_decode_cols_refuses_bad_ranges(synthetic_dir):
    _, groups = synthetic_dir
    g = groups["k4_sq"]
    suh, svh = g["suh"].astype(np.float64), g["svh"].astype(np.float64)
    for c0, c1 in [(0, 100), (64, 128), (128, 128), (0, 384)]:
        with pytest.raises(ValueError):
            exl3.decode_cols(g["trellis"], "mul1", suh, svh, c0, c1)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_decode_cols_equals_rust_strip(exl3dq, synthetic_dir, tmp_path, case):
    d, groups = synthetic_dir
    prefix, k, n, bits, _ = case
    g = groups[prefix]
    suh, svh = g["suh"].astype(np.float64), g["svh"].astype(np.float64)
    for s in sorted({0, n // 128 - 1}):
        run(exl3dq, "strip", d, prefix, s, tmp_path / "s.npy")
        rs = np.load(tmp_path / "s.npy")
        assert rs.shape == (k, 128)
        mine = exl3.decode_cols(g["trellis"], "mul1", suh, svh, s * 128, s * 128 + 128)
        np.testing.assert_array_equal(mine.view(np.uint64), rs.view(np.uint64))


def test_linear_layout_and_row_slabs():
    g = make_group(31, 256, 384, 6)
    lin = lin_of(g)
    w = exl3.decode_weight(g["trellis"], "mul1", lin.suh, lin.svh)
    nn_w = lin.weight()
    assert nn_w.shape == (384, 256) and nn_w.dtype == np.float64
    np.testing.assert_array_equal(nn_w.view(np.uint64), np.ascontiguousarray(w.T).view(np.uint64))
    x = np.random.default_rng(2).standard_normal((3, 256))
    np.testing.assert_allclose(x @ nn_w.T, lin(x), rtol=1e-11, atol=1e-11 * np.abs(lin(x)).max())
    np.testing.assert_array_equal(lin.weight_rows(130, 300), nn_w[130:300])
    np.testing.assert_array_equal(lin.weight_rows(0, 128), nn_w[:128])


def test_unsupported_trellis_width_refused():
    with pytest.raises(ValueError, match="unsupported"):
        exl3.bits_from_trellis(np.zeros((1, 1, 72), dtype=np.int16))
    with pytest.raises(ValueError):
        exl3.bits_from_trellis(np.zeros((1, 1, 100), dtype=np.int16))
    assert exl3.bits_from_trellis(np.zeros((1, 1, 80), dtype=np.int16)) == 5.0


def test_dense_decode_is_forward_on_identity():
    g = make_group(3, 256, 384, 4)
    w = exl3.decode_weight(g["trellis"], "mul1", g["suh"].astype(np.float64), g["svh"].astype(np.float64))
    np.testing.assert_allclose(lin_of(g)(np.eye(256)), w, rtol=1e-13, atol=1e-13 * np.abs(w).max())


def test_tile_positions_are_a_permutation():
    r, c = exl3.tile_positions()
    assert sorted((r * 16 + c).tolist()) == list(range(256))


def test_wrong_marker_refused(exl3dq, tmp_path):
    g = make_group(5, 128, 128, 4)
    g["mul1"] = np.array(0x12345678, dtype=np.int32).reshape(())
    write_groups(tmp_path, {"bad": g})
    r = subprocess.run([exl3dq, "wq", tmp_path, "bad", tmp_path / "o.npy"], capture_output=True, text=True)
    assert r.returncode != 0 and "marker" in r.stderr
    with pytest.raises(ValueError, match="marker"):
        exl3.codebook_from_marker("mul1", 0x12345678)
    with pytest.raises(ValueError, match="marker"):  # the mcg value under the mul1 name
        exl3.codebook_from_marker("mul1", exl3.MARKER_MCG)
    assert exl3.codebook_from_marker("mul1", exl3.MARKER_MUL1) == "mul1"
