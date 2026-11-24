"""Fetch real EXL3 tensors from the pinned Inkling-2-Mini pack by HTTP range reads (unauthenticated, public repo; no file is
downloaded whole), and check the numpy decode against REXL3's CPU decoder (exl3dq: gb-exl3) bit for bit.

    oracle/.venv/bin/python oracle/tools/verify_exl3_real.py

Covered: routed-expert projections at K=4 and K=5 (gate/up/down triples, one layer in the loop core 6..17 and one outside),
a 6-bit attention weight of a global layer and one of layer 0, two 6-bit unembed column slabs (the first 1024 columns and a
1152-column slab holding the real-vocab edge at token 200058), drafter (`mtp.safetensors`) 4-bit weights, 6-bit vision
dense weights. For each: `exl3.unpack` W_q fp16 bits == `exl3dq wq`, `exl3.decode_weight` float64 bits == `exl3dq dense`,
`exl3.decode_cols` and `exl3dq strip` agree on strips, the marker is the mul1 value. Fetched groups are written to
oracle/.cache/real-groups/ (git-ignored; one safetensors shard each, a folder gb-exl3's scan accepts) and the result to
oracle/records/exl3-real-verify.json. Exit status is non-zero if anything differs.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import resource
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

import exl3  # noqa: E402
import st  # noqa: E402

REPO = "vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw"
REV = "44c78c9d229b43dd45e41506071b9f5f939bed44"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
FILES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "mtp.safetensors")
OUT = os.path.join(ORACLE, ".cache", "real-groups")
NP = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "BF16": np.uint16}

fetched = {"bytes": 0, "requests": 0}


def get(fn: str, start: int | None = None, stop: int | None = None) -> bytes:
    """Plain HTTPS GET of a byte range (no credentials)."""
    req = urllib.request.Request(BASE + fn)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{stop - 1}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if start is not None and len(data) != stop - start:
                raise OSError(f"{fn}: got {len(data)} bytes for a {stop - start}-byte range")
            fetched["bytes"] += len(data)
            fetched["requests"] += 1
            return data
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError


def remote_header(fn: str) -> dict[str, st.Entry]:
    (n,) = struct.unpack("<Q", get(fn, 0, 8))
    return st.parse_header(get(fn, 8, 8 + n), fn, n)


def arr(fn: str, e: st.Entry, off: int = 0, n: int | None = None) -> np.ndarray:
    """A whole tensor, or n bytes from byte offset off inside it, as a flat numpy array."""
    n = e.nbytes - off if n is None else n
    return np.frombuffer(get(fn, e.start + off, e.start + off + n), dtype=NP[e.dtype])


def run(*a) -> None:
    subprocess.run([str(x) for x in a], check=True)


class Verifier:
    def __init__(self, exe: str, headers: dict[str, dict[str, st.Entry]]):
        self.exe, self.headers = exe, headers
        self.where = {n: (fn, e) for fn, h in headers.items() for n, e in h.items()}
        self.tmp = os.path.join(OUT, "tmp")
        os.makedirs(self.tmp, exist_ok=True)
        self.groups: dict[str, dict] = {}
        self.luts: dict[str, dict] = {}

    def has(self, prefix: str) -> bool:
        return prefix + ".trellis" in self.where

    def fetch_group(self, prefix: str, cols: tuple[int, int] | None = None) -> dict:
        """The parts of one group as numpy; with `cols` = (n0, n1) only the trellis tile columns [n0/16, n1/16) (kt range
        requests) and svh[n0:n1]."""
        fn, te = self.where[prefix + ".trellis"]
        kt, nt, last = te.shape
        src = {"file": fn, "tensor_shape": list(te.shape), "trellis_offset": te.start, "trellis_nbytes": te.nbytes}
        if cols is None:
            tr = arr(fn, te).reshape(te.shape)
            sv = arr(*self.where[prefix + ".svh"])
            src["trellis_ranges"] = [[te.start, te.start + te.nbytes]]
        else:
            n0, n1 = cols
            t0, t1 = n0 // 16, n1 // 16
            tb = last * 2
            rows = list(range(kt))

            def row(kk: int) -> np.ndarray:
                return np.frombuffer(get(fn, te.start + (kk * nt + t0) * tb, te.start + (kk * nt + t1) * tb), dtype=np.int16)

            with ThreadPoolExecutor(8) as pool:
                tr = np.stack(list(pool.map(row, rows))).reshape(kt, t1 - t0, last)
            sf, se = self.where[prefix + ".svh"]
            sv = arr(sf, se, n0 * 2, (n1 - n0) * 2)
            src.update(columns=[n0, n1], requests=kt, request_bytes=(t1 - t0) * tb,
                       first_range=[te.start + t0 * tb, te.start + (t1) * tb],
                       trellis_ranges=f"{kt} ranges of {(t1 - t0) * tb} bytes, row kk at offset {te.start}+(kk*{nt}+{t0})*{tb}")
        suh = arr(*self.where[prefix + ".suh"])
        mk = arr(*self.where[prefix + ".mul1"])
        return {"trellis": tr, "suh": suh, "svh": sv, "mul1": mk.reshape(()), "source": src}

    def verify(self, label: str, cache_name: str, g: dict, cols: tuple[int, int] | None = None) -> dict:
        tr, suh, svh = g["trellis"], g["suh"], g["svh"]
        marker = int(g["mul1"].view(np.uint32))
        cb = exl3.codebook_from_marker("mul1", marker)
        bits = exl3.bits_from_trellis(tr)
        shard = os.path.join(OUT, cache_name.replace("/", "_") + ".safetensors")
        st.write(shard, {f"{cache_name}.trellis": tr, f"{cache_name}.suh": suh, f"{cache_name}.svh": svh,
                         f"{cache_name}.mul1": g["mul1"]})
        suh64, svh64 = suh.astype(np.float64), svh.astype(np.float64)
        k, n = tr.shape[0] * 16, tr.shape[1] * 16
        run(self.exe, "wq", OUT, cache_name, os.path.join(self.tmp, "wq.npy"))
        run(self.exe, "dense", OUT, cache_name, os.path.join(self.tmp, "w.npy"))
        wq = exl3.unpack(tr, cb)
        wq_eq = bool(np.array_equal(np.load(os.path.join(self.tmp, "wq.npy")), wq))
        w_py = exl3.decode_weight(tr, cb, suh64, svh64)
        w_rs = np.load(os.path.join(self.tmp, "w.npy"))
        dense_eq = bool(np.array_equal(w_py.view(np.uint64), w_rs.view(np.uint64)))
        # strips: gb_exl3::decode_strip vs exl3.decode_cols vs the dense slice (every strip of a slab, first/last otherwise)
        nstrips = n // 128
        which = list(range(nstrips)) if nstrips <= 16 else [0, 1, nstrips // 2, nstrips - 1]
        strip_eq = cols_eq = True
        for s in which:
            run(self.exe, "strip", OUT, cache_name, s, os.path.join(self.tmp, "s.npy"))
            rs = np.load(os.path.join(self.tmp, "s.npy"))
            strip_eq &= bool(np.array_equal(rs.view(np.uint64), w_rs[:, s * 128:(s + 1) * 128].view(np.uint64)))
            dc = exl3.decode_cols(tr, cb, suh64, svh64, s * 128, (s + 1) * 128)
            cols_eq &= bool(np.array_equal(dc.view(np.uint64), rs.view(np.uint64)))
        if nstrips >= 3:  # a multi-strip window that starts off zero
            dc = exl3.decode_cols(tr, cb, suh64, svh64, 128, min(n, 128 * 3))
            cols_eq &= bool(np.array_equal(dc.view(np.uint64), w_py[:, 128:min(n, 384)].view(np.uint64)))
        match = wq_eq and dense_eq and strip_eq and cols_eq and marker == exl3.MARKER_MUL1
        wqf = exl3._f16(wq)
        rec = {
            "tensor": label, "k": int(bits), "k_bits": bits, "k_in": k, "n_out": n, "shape": [k, n],
            "bytes": int(wq.nbytes + w_rs.nbytes), "bytes_trellis": int(tr.nbytes), "match": bool(match),
            "wq_bits_equal": wq_eq, "dense_bits_equal": dense_eq, "strips_equal_rust": strip_eq,
            "decode_cols_equal": cols_eq, "strips_checked": which, "marker": hex(marker), "cache_prefix": cache_name,
            "source": g["source"],
            "stats": {"w_rms": float(np.sqrt((w_py ** 2).mean())), "w_absmax": float(np.abs(w_py).max()),
                      "wq_rms": float(np.sqrt((wqf ** 2).mean())),
                      "suh_absmin": float(np.abs(suh64).min()), "suh_absmax": float(np.abs(suh64).max()),
                      "svh_absmin": float(np.abs(svh64).min()), "svh_absmax": float(np.abs(svh64).max())},
        }
        self.groups[label] = rec
        print(label, {x: rec[x] for x in ("k", "shape", "match", "wq_bits_equal", "dense_bits_equal",
                                          "strips_equal_rust", "decode_cols_equal")}, flush=True)
        return rec


def ffn_sanity(lins: list[exl3.Exl3Linear], label: str, rng: np.random.Generator) -> dict:
    g, u, d = lins
    z = rng.standard_normal((16, g.k))
    x = z / np.sqrt((z ** 2).mean(-1, keepdims=True))
    gv, uv = g(x), u(x)
    h = gv / (1 + np.exp(-gv)) * uv  # silu(gate) * up, no caps
    y = d(h)
    rms = lambda v: float(np.sqrt((v ** 2).mean()))  # noqa: E731
    return {"bits": [lin.bits for lin in lins], "gate_rms": rms(gv), "up_rms": rms(uv), "y_rms": rms(y),
            "finite": bool(np.isfinite(y).all())}


def main(argv=None) -> int:
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".safetensors"):
            os.remove(os.path.join(OUT, f))
    crate = os.path.join(ORACLE, "exl3dq")
    subprocess.run(["cargo", "build", "--release", "--locked", "--quiet"], cwd=crate, check=True)
    exe = os.path.join(crate, "target", "release", "exl3dq")
    headers = {fn: remote_header(fn) for fn in FILES}
    v = Verifier(exe, headers)

    # --- routed experts: pick by trellis width (last dim 64 = K=4, 80 = K=5) -------------------------------------
    rx = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.trellis$")
    ex: dict[int, dict[int, dict[str, int]]] = {}
    for n, (fn, e) in v.where.items():
        m = rx.match(n)
        if m:
            ex.setdefault(int(m[1]), {}).setdefault(int(m[2]), {})[m[3]] = e.shape[-1] // 16
    layers = sorted(ex)
    core = [x for x in layers if 6 <= x <= 17]
    outside = [x for x in layers if not 6 <= x <= 17]
    pick_layers = [core[len(core) // 2], outside[-1]]

    def choose(layer: int, k: int) -> tuple[int, dict[str, int]]:
        full = [e for e, ks in sorted(ex[layer].items()) if len(ks) == 3 and all(x == k for x in ks.values())]
        if full:
            return full[0], ex[layer][full[0]]
        part = [(sum(x == k for x in ks.values()), e) for e, ks in sorted(ex[layer].items()) if len(ks) == 3]
        best = max(c for c, _ in part)
        e = min(e for c, e in part if c == best)
        return e, ex[layer][e]

    triples = []
    for k in (4, 5):
        for lay in pick_layers:
            e, ks = choose(lay, k)
            triples.append((k, lay, e, ks))
    ffn_in: dict[str, list] = {}
    for k, lay, e, ks in triples:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            pre = f"model.llm.layers.{lay}.mlp.experts.{e}.{proj}"
            g = v.fetch_group(pre)
            rec = v.verify(pre, pre, g)
            rec["triple_k"] = ks
            rec["picked_for_k"] = k
            ffn_in.setdefault(f"{lay}.{e}", []).append(exl3.Exl3Linear(g["trellis"], "mul1", g["suh"].astype(np.float64),
                                                                       g["svh"].astype(np.float64)))

    # --- 6-bit attention, drafter, vision ---------------------------------------------------------------------
    for pre in ("model.llm.layers.11.attn.wq_du", "model.llm.layers.0.attn.wo_ud"):
        v.verify(pre, pre, v.fetch_group(pre))
    for pre in ("layers.1.self_attn.q_proj", "markov_head.hidden_out", "markov_head.rnn_proj"):
        v.verify("mtp:" + pre, "mtp." + pre, v.fetch_group(pre))
    for pre in ("model.visual.proj_out", "model.visual.backbone.layers.0.self_attn.q_proj"):
        v.verify(pre, pre, v.fetch_group(pre))

    # --- unembed column slabs ----------------------------------------------------------------------------------
    ue = "model.llm.unembed"
    n_pad = v.where[ue + ".trellis"][1].shape[1] * 16
    for n0, n1 in ((0, 1024), (n_pad - 1152, n_pad)):
        g = v.fetch_group(ue, (n0, n1))
        v.verify(f"{ue}[cols {n0}:{n1}]", f"{ue}.slab_{n0}_{n1}", g, (n0, n1))

    ents = list(v.groups.values())
    rng = np.random.default_rng(0)
    experts = {key: ffn_sanity(lins, key, rng) for key, lins in ffn_in.items()}
    per_k: dict[str, int] = {}
    for e in ents:
        per_k[str(e["k"])] = per_k.get(str(e["k"]), 0) + 1
    all_equal = all(e["match"] for e in ents) and all(x["finite"] for x in experts.values())
    rec = {
        "repo": REPO, "revision": REV, "command": " ".join([os.path.relpath(sys.argv[0])] + sys.argv[1:]),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.platform(),
        "entries": [{k: e[k] for k in ("tensor", "k", "bytes", "match")} | {x: e[x] for x in e if x not in ("tensor", "k", "bytes", "match")}
                    for e in ents],
        "groups": {e["tensor"]: {x: e[x] for x in ("k_bits", "k_in", "n_out", "bytes_trellis", "wq_bits_equal", "dense_bits_equal",
                                                    "marker", "source", "stats")} for e in ents},
        "experts_ffn_sanity": experts,
        "summary": {
            "entries": len(ents), "entries_per_k": per_k, "all_equal": all_equal,
            "total_bytes_fetched": fetched["bytes"], "total_requests": fetched["requests"],
            "all_markers_mul1": all(e["marker"] == hex(exl3.MARKER_MUL1) for e in ents),
            "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)),
            "seconds": round(time.time() - t_start, 1),
        },
    }
    os.makedirs(os.path.join(ORACLE, "records"), exist_ok=True)
    with open(os.path.join(ORACLE, "records", "exl3-real-verify.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec["summary"], indent=1))
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
