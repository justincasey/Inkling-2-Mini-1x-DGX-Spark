"""Golden values for the REXL3 MoE GPU test: the oracle's expert FFN on REAL tensors of layers 12, 22 and 23 of the pinned
Inkling-2-Mini pack (our own mixed-precision quantization method for EXL3; the method is not published).

    oracle/.venv/bin/python bench/records/2026-11-22-kernels-moe/gen_moe_golden.py

The oracle checkout (the recipe repo's `oracle/`) is found through INKLING_ORACLE, else
~/recipes/Inkling-2-Mini-1x-DGX-Spark/oracle, else `oracle/` under the current directory. Outputs go to INKLING_GOLDEN_OUT, else
~/projects/rexl3-inkling2-loop-moe/testdata/inkling2_loop/gpu. Tensors are fetched by unauthenticated HTTP range reads
(oracle/tools/verify_exl3_real.py helpers) into /tmp/k-moe-golden-cache/ and reused when present.

Per layer L in (12, 22, 23): R = 16 rows of mlp_in-like bf16 activations (seeded), rows 0..11 routed by the layer's real
router through the oracle's `route`, rows 12..15 with forced routes that cover every K class; the oracle `expert_ffn` on
the real EXL3 experts (variants `exact`: oracle default rounding; `fp16in`: fp16 GEMM inputs); the rank partial combined
exactly like `Inkling2LoopRef.combine` (float64 accumulation over experts in ascending expert order, one bf16 rounding).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
HOME = os.path.expanduser("~")


def find_oracle() -> str:
    for c in (os.environ.get("INKLING_ORACLE"),
              os.path.join(HOME, "recipes", "Inkling-2-Mini-1x-DGX-Spark", "oracle"), os.path.join(os.getcwd(), "oracle")):
        if c and os.path.exists(os.path.join(c, "inkling2_ref.py")):
            return os.path.abspath(c)
    raise SystemExit("oracle checkout not found (set INKLING_ORACLE)")


ORACLE = find_oracle()
sys.path.insert(0, ORACLE)
sys.path.insert(0, os.path.join(ORACLE, "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import exl3  # noqa: E402
import pack  # noqa: E402
import st  # noqa: E402
import verify_exl3_real as net  # noqa: E402
from inkling2_ref import F64, Rounding, expert_ffn, rmsnorm_w, route  # noqa: E402

torch.set_num_threads(4)
os.environ.setdefault("OMP_NUM_THREADS", "4")

OUT = os.environ.get("INKLING_GOLDEN_OUT") or os.path.join(HOME, "projects", "rexl3-inkling2-loop-moe", "testdata",
                                                          "inkling2_loop", "gpu")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "golden-gen.log")
CACHE = "/tmp/k-moe-golden-cache"
LAYERS = (12, 22, 23)
SEED = 20261122
R, NREAL, TOPK, NE, H = 16, 12, 8, 128, 2304
GATE_CAP, UP_CAP, SCALE = 4.0, 25.0, 8.0
PROJS = ("gate_proj", "up_proj", "down_proj")
YROWS = (0, 1, 12)
WORKERS = 4
CMD = "oracle/.venv/bin/python bench/records/2026-11-22-kernels-moe/gen_moe_golden.py"

_log = open(LOG, "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log.write(s + "\n")
    _log.flush()


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def fetch_shard(path: str, tensors: dict[str, tuple[str, st.Entry]]) -> None:
    """Range-fetch the named tensors into one safetensors shard (skipped when the shard exists)."""
    if os.path.exists(path):
        return
    d = {}
    for name, (fn, e) in tensors.items():
        dt = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "I16": np.int16, "I32": np.int32}[e.dtype]
        d[name] = np.frombuffer(net.get(fn, e.start, e.start + e.nbytes), dtype=dt).reshape(e.shape)
    tmp = path + ".part"
    st.write(tmp, d)
    os.replace(tmp, path)


def main() -> int:
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    say("generator:", CMD)
    say("pack:", net.REPO, "@", net.REV)
    headers = {fn: net.remote_header(fn) for fn in net.FILES[:2]}
    where = {n: (fn, e) for fn, h in headers.items() for n, e in h.items()}
    rx = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.trellis$")
    ktabs: dict[int, np.ndarray] = {L: np.zeros((NE, 3), np.int32) for L in LAYERS}
    seen: dict[int, set] = {L: set() for L in LAYERS}
    for n, (fn, e) in where.items():
        m = rx.match(n)
        if m and int(m[1]) in LAYERS:
            L, ex, p = int(m[1]), int(m[2]), PROJS.index(m[3])
            ktabs[L][ex, p] = e.shape[-1] // 16
            seen[L].add((ex, p))
    summary = {}
    for L in LAYERS:
        assert len(seen[L]) == NE * 3, f"layer {L}: {len(seen[L])} expert trellis entries"
        summary[L] = run_layer(L, ktabs[L], where)
    say("--- files")
    for L in LAYERS:
        p = os.path.join(OUT, f"moe_layer{L}.safetensors")
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        say(f"moe_layer{L}.safetensors", os.path.getsize(p), "bytes sha256", sha)
    say(f"wall seconds: {time.time() - T0:.1f}")
    return 0


def run_layer(L: int, ktab: np.ndarray, where: dict) -> dict:
    t_layer = time.time()
    ldir = os.path.join(CACHE, f"L{L}")
    os.makedirs(ldir, exist_ok=True)
    pre = f"model.llm.layers.{L}.mlp."
    classes = [tuple(int(k) for k in ktab[e]) for e in range(NE)]
    cnt = Counter(classes)
    say(f"=== layer {L}: K classes (gate, up, down) -> experts: " + ", ".join(f"{c}: {n}" for c, n in sorted(cnt.items())))
    cls_list = sorted(cnt, key=lambda c: (cnt[c], c))  # smallest class first
    members = {c: [e for e in range(NE) if classes[e] == c] for c in cls_list}

    # --- plain tensors ---------------------------------------------------------------------------------------
    plain_names = {"gate": pre + "gate.weight", "bias": pre + "gate.bias", "norm": f"model.llm.layers.{L}.mlp_norm.weight"}
    fetch_shard(os.path.join(ldir, "plain.safetensors"), {n: where[n] for n in plain_names.values()})
    plain = st.read_header(os.path.join(ldir, "plain.safetensors"))
    gate_w = torch.from_numpy(bf16_bits_to_f32(st.read_numpy(plain[plain_names["gate"]]))).to(F64)
    gate_b = torch.from_numpy(st.read_numpy(plain[plain_names["bias"]]).astype(np.float64))
    gamma = torch.from_numpy(bf16_bits_to_f32(st.read_numpy(plain[plain_names["norm"]]))).to(F64)
    assert gate_w.shape == (NE, H) and gate_b.shape == (NE,) and gamma.shape == (H,)

    # --- inputs ---------------------------------------------------------------------------------------------
    rng = np.random.default_rng([SEED, L])
    p = torch.from_numpy(3.0 * rng.standard_normal((R, H)))
    x = rmsnorm_w(p, gamma, 1e-6, Rounding())  # bf16 values as float64
    x_bf16 = x.to(torch.float32).to(torch.bfloat16)
    assert torch.equal(x_bf16.to(F64), x)

    # --- routes ---------------------------------------------------------------------------------------------
    rnd = Rounding()
    logits = rnd.f32(x[:NREAL] @ gate_w.T)
    idx_r, w_r, margin = route(logits, gate_b, TOPK, SCALE, rnd)
    pick = np.zeros((R, TOPK), np.int32)
    wts = np.zeros((R, TOPK), np.float32)
    pick[:NREAL] = idx_r.numpy().astype(np.int32)
    wts[:NREAL] = w_r.numpy().astype(np.float32)
    assert (w_r.numpy() == wts[:NREAL].astype(np.float64)).all()
    # (c) brute-force top-8 of sigmoid + bias
    score = 1.0 / (1.0 + np.exp(-logits.numpy()))
    sel = score + gate_b.numpy()
    brute_ok = True
    for r_ in range(NREAL):
        order = sorted(range(NE), key=lambda e: (-sel[r_, e], e))[:TOPK]
        brute_ok &= set(order) == set(pick[r_].tolist()) and order == pick[r_].tolist()
    say(f"check (c) real-router rows 0..{NREAL - 1}: pick == brute-force top-8 of sigmoid+bias (set and order): {brute_ok}; "
        f"min margin {float(margin.min()):.3e}")
    assert brute_ok

    # forced rows: row 12 = the whole smallest class (up to 8 experts) + others; rows 13..15 = 2 of each class + random
    forced = []
    first = list(members[cls_list[0]][:TOPK])
    rest = [e for e in range(NE) if e not in first]
    forced.append(first + [int(e) for e in rng.choice(rest, TOPK - len(first), replace=False)])
    for _ in range(3):
        row: list[int] = []
        for c in cls_list[:-1]:
            row += [int(e) for e in rng.choice(members[c], min(2, len(members[c])), replace=False) if int(e) not in row]
        row = row[:TOPK]
        rest = [e for e in range(NE) if e not in row]
        row += [int(e) for e in rng.choice(rest, TOPK - len(row), replace=False)]
        forced.append(row)
    for j, row in enumerate(forced):
        assert len(set(row)) == TOPK
        w = rng.random(TOPK) + 0.25
        w = np.sort(w * (SCALE / w.sum()))[::-1].astype(np.float32)  # descending, like the router's slot order
        pick[NREAL + j] = np.array(row, np.int32)
        wts[NREAL + j] = w
        say(f"forced row {NREAL + j}: experts {row} classes {[classes[e] for e in row]} wsum {float(w.astype(np.float64).sum()):.6f}")
    cover = {}
    for c in cls_list:
        rows = [r_ for r_ in range(R) if any(classes[e] == c for e in pick[r_])]
        experts = sorted({int(e) for e in pick.flatten() if classes[e] == c})
        cover[c] = (len(rows), len(experts))
        say(f"coverage class {c}: {len(rows)} rows, {len(experts)}/{cnt[c]} experts touched ({len(experts)} distinct)")
        assert experts
    if cnt[cls_list[0]]:
        smallest = set(members[cls_list[0]])
        assert smallest <= {int(e) for e in pick.flatten()}
    need = sorted({int(e) for e in pick.flatten()})
    say(f"distinct experts touched: {len(need)}")

    # --- fetch expert groups ----------------------------------------------------------------------------------
    def fetch_expert(e: int) -> None:
        for proj in PROJS:
            prefix = pack.expert_prefix(L, e, proj)
            d = {}
            for part in ("trellis", "suh", "svh", "mul1", "mcg"):
                n = f"{prefix}.{part}"
                if n in where:
                    d[n] = where[n]
            fetch_shard(os.path.join(ldir, f"e{e}.{proj}.safetensors"), d)

    with ThreadPoolExecutor(WORKERS) as ex_:
        list(ex_.map(fetch_expert, need))
    say(f"fetched/cached {len(need)} experts; {net.fetched['requests']} range requests, {net.fetched['bytes'] / 1e6:.1f} MB in this run")
    ix = pack.PackIndex(ldir)

    # --- (a) decode vs the Rust CPU decoder on 3 sampled experts -----------------------------------------------
    rust = os.path.join(ORACLE, "exl3dq", "target", "release", "exl3dq")
    sample = [need[0], need[len(need) // 2], need[-1]]
    if os.path.exists(rust):
        tmp = os.path.join(CACHE, "tmp")
        os.makedirs(tmp, exist_ok=True)
        eq_all = True
        for e in sample:
            for proj in PROJS:
                prefix = pack.expert_prefix(L, e, proj)
                lin = pack.load_exl3(ix, prefix)
                subprocess.run([rust, "dense", ldir, prefix, os.path.join(tmp, "w.npy")], check=True)
                subprocess.run([rust, "wq", ldir, prefix, os.path.join(tmp, "wq.npy")], check=True)
                w_rs = np.load(os.path.join(tmp, "w.npy"))
                g = pack._Group(ix, prefix)
                w_py = exl3.decode_weight(st.read_numpy(g.entry), g.codebook, g.suh, g.svh)
                dense_eq = bool(np.array_equal(w_py.view(np.uint64), w_rs.view(np.uint64)))
                wq_eq = bool(np.array_equal(np.load(os.path.join(tmp, "wq.npy")), exl3.unpack(st.read_numpy(g.entry), g.codebook)))
                eq_all &= dense_eq and wq_eq
                say(f"check (a) layer {L} expert {e} {proj} K={g.bits:g}: oracle dense == exl3dq dense bitwise {dense_eq}; "
                    f"wq == exl3dq wq {wq_eq}")
        assert eq_all
        say(f"check (a) layer {L}: all 9 groups of experts {sample} bit-identical to the Rust CPU decoder (exl3dq)")
    else:
        say("check (a) skipped: exl3dq binary not built")
        eq_all = None

    # --- oracle FFN per expert (rows of an expert together) ---------------------------------------------------------
    x_np = x.numpy()
    y_ex: dict[tuple[int, int], np.ndarray] = {}
    y_fp: dict[tuple[int, int], np.ndarray] = {}
    for e in need:
        tok, slot = np.nonzero(pick == e)
        lins = [pack.load_exl3(ix, pack.expert_prefix(L, e, pr)) for pr in PROJS]
        ye = expert_ffn(*lins, x_np[tok], GATE_CAP, UP_CAP, Rounding(), False)
        yf = expert_ffn(*lins, x_np[tok], GATE_CAP, UP_CAP, Rounding(), True)
        for i, (t_, s_) in enumerate(zip(tok, slot)):
            y_ex[(int(t_), int(s_))] = ye[i]
            y_fp[(int(t_), int(s_))] = yf[i]
        del lins
    assert len(y_ex) == R * TOPK

    def combine(y: dict, by_expert: bool) -> np.ndarray:
        acc = torch.zeros(R, H, dtype=F64)
        pairs = sorted(((int(pick[r_, s_]), r_, s_) for r_ in range(R) for s_ in range(TOPK)), key=None if by_expert else
                       (lambda t: (t[1], t[2])))
        if not by_expert:
            pairs = [(e, r_, s_) for (e, r_, s_) in pairs]
        for e, r_, s_ in pairs:
            acc[r_] += torch.from_numpy(y[(r_, s_)]) * float(wts[r_, s_])  # fp64 weight * fp64 y
        return rnd.bf16(acc).numpy()

    out_exact = combine(y_ex, True)  # Inkling2LoopRef.combine: ascending expert order, fp64, one bf16 rounding
    out_fp16 = combine(y_fp, True)
    out_slot = combine(y_ex, False)
    say(f"layer {L}: out_exact rounding = Inkling2LoopRef.combine (fp64 accumulation of w*y in ascending-expert order, "
        f"one bf16 rounding via fp32); elements differing from slot-order accumulation: {int((out_slot != out_exact).sum())}")

    # (b)
    a, b = out_exact, out_fp16
    cos = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    rms = np.sqrt((a * a).mean(1))
    mad = np.abs(a - b).max(1)
    say(f"check (b) layer {L}: out_exact vs out_fp16in cosine min {cos.min():.8f} (mean {cos.mean():.8f}); "
        f"max|diff|/row-RMS max {(mad / rms).max():.4e} (mean {(mad / rms).mean():.4e}); max|diff| {mad.max():.4e}; row RMS "
        f"min {rms.min():.4e} max {rms.max():.4e}")
    y_ex_arr = np.stack([np.stack([y_ex[(r_, s_)] for s_ in range(TOPK)]) for r_ in YROWS]).astype(np.float32)

    meta = {"generator": CMD, "pack_repo": net.REPO, "pack_revision": net.REV, "layer": str(L), "seed": str(SEED),
            "rows": str(R), "real_router_rows": f"0..{NREAL - 1}", "forced_rows": f"{NREAL}..{R - 1}",
            "gate_cap": str(GATE_CAP), "up_cap": str(UP_CAP), "route_scale": str(SCALE),
            "out_exact": "bf16-rounded rank partial stored as f32; oracle default rounding, bf16 act, f32 g/u, "
                         "fp64 weighted accumulation in ascending expert order, one bf16 rounding",
            "out_fp16in": "same with fp16 GEMM inputs (expert_ffn fp16_in=True)",
            "ktab": "K of (gate, up, down) per expert", "y_exact": "unweighted per-pair y of rows y_rows, slot order",
            "quantization": "our own mixed-precision quantization method for EXL3; the method is not published"}
    tensors = {
        "x": x_bf16, "pick": pick, "wts": wts, "ktab": ktab,
        "out_exact": out_exact.astype(np.float32), "out_fp16in": out_fp16.astype(np.float32),
        "y_rows": np.array(YROWS, np.int32), "y_exact": y_ex_arr,
    }
    path = os.path.join(OUT, f"moe_layer{L}.safetensors")
    st.write(path, tensors, meta)
    say(f"wrote moe_layer{L}.safetensors ({os.path.getsize(path)} bytes) in {time.time() - t_layer:.1f}s")
    # re-read check with the oracle's reader
    back = st.load_all(path)
    assert set(back) == set(tensors)
    return {"cover": cover, "check_a": eq_all}


if __name__ == "__main__":
    sys.exit(main())
