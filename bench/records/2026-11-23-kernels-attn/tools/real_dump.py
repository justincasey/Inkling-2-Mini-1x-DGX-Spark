#!/usr/bin/env python3
"""Real-tensor attention fixture for Inkling-2-Mini: global layer 11 (visit 11, pass 0) and local layer 0 (visit 0).

    oracle/.venv/bin/python real_dump.py [--stage capture|emit|all] [--out real/attn_real.bin]   # generate
    oracle/.venv/bin/python real_dump.py --check real/attn_real.bin                              # list the entries

capture: runs the recipe oracle (`Inkling2LoopRef`, default exact Emulation, real pack tensors fetched by HTTP range reads of
         the public pack repo at the pinned revision, EXL3 dense weights and experts decoded by oracle/exl3.py) over a fixed
         T = 16 token prompt through visits 0..11 and stops at the attention core of visit 11. For visits 0 and 11 it records
         the attention's inputs exactly as `Inkling2LoopRef.attention` sees them (the visit's input P, x_q, x_k, x_v) and the
         layer's weights as stored. Written to <cache>/captures.pt.
emit:    from those captures, re-runs the body of `Inkling2LoopRef.attention` step by step for position offsets
         P0 in {0, 8180, 300000} (fresh LayerCache, zero conv state, k_start = P0) by calling the oracle's own functions
         (static_conv, dynamic_conv, l2clip, log_tau, relative_bias, kv8_roundtrip, attention_core, attention_core_tiled), and
         writes the container below.

Container (little-endian): magic 'I2LR', u32 version = 1, u32 n_entries; per entry: u16 name_len, name (ascii), u8 dtype
(0 = bf16 as u16 bits, 1 = f32, 2 = i32), u8 ndim, u32 dims[ndim], raw data row-major. Entry names: see README.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ORACLE = os.path.expanduser("~/recipes/Inkling-2-Mini-1x-DGX-Spark/oracle")
DEFAULT_CACHE = os.path.expanduser("~/.cache/inkling2-mini-attn-real")
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "real", "attn_real.bin")

REPO = "vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw"
REV = "44c78c9d229b43dd45e41506071b9f5f939bed44"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
SENTENCE = "The quick brown fox jumps over the lazy dog near the quiet river."
T = 16
P0S = (0, 8180, 300000)
BIAS_P0S = (0, 300000)
LAST_VISIT = 11
VISITS = {"l0": 0, "g11": 11}  # entry prefix -> visit index (visit index == layer index before the core loop)
MAGIC = b"I2LR"
DT_BF16, DT_F32, DT_I32 = 0, 1, 2
DT_NAME = {DT_BF16: "bf16", DT_F32: "f32", DT_I32: "i32"}


# ----------------------------------------------------------------------------------------------------------------------
# HTTP range cache (a sparse local copy of the pack shards filled on demand, block by block)

BLOCK = 1 << 20


class RangeCache:
    """The two pack shards as sparse files in <root>/pack: the safetensors header is real, every byte range that is read
    is first fetched (public GET with a Range header, no credentials) into its 1 MiB blocks. `read` is what the oracle's
    positional reads (`st._pread`) are replaced with; `ensure` fetches many ranges at once with a thread pool."""

    def __init__(self, root: str, workers: int = 12):
        self.dir = os.path.join(root, "pack")
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(root, "blocks.json")
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(workers)
        self.size: dict[str, int] = {}
        self.have: dict[str, set[int]] = {fn: set() for fn in SHARDS}
        self.fetched = {"bytes": 0, "requests": 0}
        self.last_save = time.time()
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                saved = json.load(f)
            self.size = saved["size"]
            for fn, blocks in saved["blocks"].items():
                self.have[fn] = set(blocks)

    def path(self, fn: str) -> str:
        return os.path.join(self.dir, fn)

    def get(self, fn: str, start: int, stop: int) -> tuple[bytes, int]:
        """(bytes [start, stop), total file size) of a shard."""
        req = urllib.request.Request(BASE + fn, headers={"Range": f"bytes={start}-{stop - 1}"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                if len(data) != stop - start:
                    raise OSError(f"{fn}: got {len(data)} bytes for a {stop - start}-byte range")
                with self.lock:
                    self.fetched["bytes"] += len(data)
                    self.fetched["requests"] += 1
                return data, total
            except (OSError, urllib.error.URLError, KeyError):
                if attempt == 7:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError

    def save(self, force: bool = False) -> None:
        with self.lock:
            if not force and time.time() - self.last_save < 20:
                return
            self.last_save = time.time()
            blob = {"size": self.size, "blocks": {fn: sorted(b) for fn, b in self.have.items()}}
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blob, f)
        os.replace(tmp, self.state_path)

    def init_shard(self, fn: str) -> None:
        """Sparse file of the full size holding the real safetensors header (8-byte length + JSON)."""
        p = self.path(fn)
        if os.path.exists(p) and fn in self.size and os.path.getsize(p) == self.size[fn]:
            return
        head, total = self.get(fn, 0, 8)
        (n,) = struct.unpack("<Q", head)
        hdr, _ = self.get(fn, 8, 8 + n)
        self.size[fn] = total
        with open(p, "wb") as f:
            f.write(head + hdr)
            f.truncate(total)
        self.have[fn] = {b for b in range((8 + n + BLOCK - 1) // BLOCK) if (b + 1) * BLOCK <= 8 + n}
        self.save(force=True)

    def _fetch_run(self, fn: str, b0: int, b1: int) -> None:
        """Fetch blocks [b0, b1) of a shard (one request) and write them into the sparse file."""
        start, stop = b0 * BLOCK, min(b1 * BLOCK, self.size[fn])
        data, _ = self.get(fn, start, stop)
        fd = os.open(self.path(fn), os.O_WRONLY)
        try:
            os.pwrite(fd, data, start)
        finally:
            os.close(fd)
        with self.lock:
            self.have[fn].update(range(b0, b1))
        self.save()

    def ensure(self, ranges: list[tuple[str, int, int]], max_run: int = 8) -> None:
        """Make every (shard file name, start, stop) byte range local; missing blocks are merged into runs of at most
        `max_run` blocks and fetched in parallel."""
        need: dict[str, set[int]] = {}
        for fn, a, b in ranges:
            if b <= a:
                continue
            miss = {blk for blk in range(a // BLOCK, (b - 1) // BLOCK + 1) if blk not in self.have[fn]}
            if miss:
                need.setdefault(fn, set()).update(miss)
        futs = []
        for fn, blocks in need.items():
            run: list[int] = []
            for blk in sorted(blocks) + [None]:
                if run and (blk is None or blk != run[-1] + 1 or len(run) == max_run):
                    futs.append(self.pool.submit(self._fetch_run, fn, run[0], run[-1] + 1))
                    run = []
                if blk is not None:
                    run.append(blk)
        for f in futs:
            f.result()

    def read(self, path: str, start: int, n: int) -> bytes:
        fn = os.path.basename(path)
        self.ensure([(fn, start, start + n)])
        fd = os.open(self.path(fn), os.O_RDONLY)
        try:
            buf = bytearray(n)
            view, got = memoryview(buf), 0
            while got < n:
                chunk = os.preadv(fd, [view[got:]], start + got)
                if chunk == 0:
                    raise OSError(f"{path}: short read at {start + got}")
                got += chunk
            return bytes(buf)
        finally:
            os.close(fd)


def setup_pack(root: str, rc: RangeCache) -> str:
    """The stub pack folder (config, index, tokenizer, sparse shards); returns its path."""
    for fn in SHARDS:
        rc.init_shard(fn)
    for fn in ("config.json", "model.safetensors.index.json"):
        p = rc.path(fn)
        if not os.path.exists(p):
            req = urllib.request.Request(BASE + fn)
            with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                f.write(r.read())
    return rc.dir


# ----------------------------------------------------------------------------------------------------------------------
# capture


class StopAfterAttention(Exception):
    pass


def capture(oracle: str, cache: str, threads: int) -> str:
    sys.path.insert(0, oracle)
    import numpy as np
    import torch

    import pack
    import st
    from inkling2_ref import F64, Inkling2LoopConfig, Inkling2LoopRef
    from tokenizers import Tokenizer

    torch.set_num_threads(threads)
    t_all = time.time()
    rc = RangeCache(cache)
    root = setup_pack(cache, rc)
    st._pread = rc.read  # every positional read of the oracle's pack sources goes through the range cache
    ids = Tokenizer.from_file(os.path.join(root, "tokenizer.json")).encode(SENTENCE, add_special_tokens=False).ids
    assert len(ids) == T, f"the sentence is {len(ids)} tokens, want {T}"
    print(f"tokens {ids}", flush=True)

    with open(os.path.join(root, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    ix = pack.PackIndex(root)

    def groups_of(name: str) -> list[str]:
        if name.endswith(".weight") and name[:-7] + ".trellis" in ix.weight_map:
            pre = name[:-7]
            return [pre + s for s in (".trellis", ".suh", ".svh", ".mul1", ".mcg") if pre + s in ix.weight_map]
        return [name]

    def ensure_names(names: list[str]) -> None:
        rc.ensure([(ix.weight_map[n], ix.entry(n).start, ix.entry(n).start + ix.entry(n).nbytes) for n in names])

    class Weights(pack.PackWeights):
        def tensor(self, name: str):
            ensure_names(groups_of(name))
            return super().tensor(name)

    class Experts(pack.PackExperts):
        def ffn_many(self, layer, jobs, *a, **k):
            ensure_names([n for e, _ in jobs for p in pack.PROJS for n in groups_of(pack.expert_prefix(layer, e, p) + ".weight")])
            return super().ffn_many(layer, jobs, *a, **k)

    wanted = set(VISITS.values())
    cap: dict[int, dict] = {i: {} for i in wanted}
    timing = {"layers": []}

    lin_keys = {"wq_du": "q", "wr_du": "r", "gate_attn": "gate", "attn_sink_proj": "sink", "wk_dv": "k", "wv_dv": "v"}

    class Capturing(Inkling2LoopRef):
        inputs: dict | None = None

        def lin(self, x, name):
            y = super().lin(x, name)
            if self.inputs is not None:  # inside the attention of a wanted visit: keep the projection outputs
                self.inputs[lin_keys[name.split(".")[-2]]] = y.clone()
            return y

        def _tap(self, k, name, x):
            if k in wanted and name == "core":
                cap[k]["core_tap"] = x.clone()
                if k == LAST_VISIT:
                    raise StopAfterAttention

        def layer(self, v, p, blocks, pos, lc, rec=None, bank=None):
            if v.index in wanted:
                cap[v.index]["hidden"] = p.clone()
            t0 = time.time()
            if v.index in wanted or v.index > 0:
                print(f"visit {v.index} (layer {v.layer}, {'global' if v.is_global else 'local'}) ...", flush=True)
            try:
                return super().layer(v, p, blocks, pos, lc, rec, bank)
            finally:
                timing["layers"].append((v.index, round(time.time() - t0, 1)))
                print(f"  visit {v.index}: {timing['layers'][-1][1]} s, fetched {rc.fetched['bytes'] / 1e6:.0f} MB in "
                      f"{rc.fetched['requests']} requests", flush=True)

        def attention(self, v, xq, xk, xv, pos, lc):
            if v.index in wanted:
                cap[v.index].update(xq=xq.clone(), xk=xk.clone(), xv=xv.clone())
                pre = f"model.llm.layers.{v.layer}.attn."
                w = {}
                for key, name in (("q_scale", "q_l2_clip.scale"), ("r_scale", "rel_l2_clip.scale"),
                                  ("rel_proj", "rel_logits_proj.proj")):
                    w[key] = self.w.tensor(pre + name).clone()
                if v.is_global:
                    for s in "kv":
                        w[f"{s}_conv"] = self.w.tensor(pre + f"{s}_sconv.static_w").clone()
                        w[f"{s}_proj"] = self.w.tensor(pre + f"{s}_sconv.weight_proj").clone()
                else:
                    for s in "kv":
                        w[f"{s}_conv"] = self.w.tensor(pre + f"{s}_sconv.weight").clone()
                cap[v.index]["w"] = w
                self.inputs = cap[v.index]["inputs"] = {}
            try:
                return super().attention(v, xq, xk, xv, pos, lc)
            finally:
                self.inputs = None

    model = Capturing(cfg, Weights(ix), Experts(ix, workers=0))  # default Rounding, default (exact) Emulation
    assert not model.em.kv8 and not model.em.attn_bf16_p and not model.em.moe_fp16
    t0 = time.time()
    try:
        model.forward(ids)
        raise AssertionError("the forward did not stop at the attention of visit 11")
    except StopAfterAttention:
        pass
    rc.save(force=True)
    secs = time.time() - t0
    torch.save({"tokens": ids, "cfg": cfg, "sentence": SENTENCE, "cap": cap, "seconds_forward": secs, "layer_seconds": timing["layers"],
                "fetched_bytes": rc.fetched["bytes"], "fetched_requests": rc.fetched["requests"]},
               os.path.join(cache, "captures.pt"))
    print(f"capture done: forward {secs:.0f} s, total {time.time() - t_all:.0f} s, fetched this run "
          f"{rc.fetched['bytes'] / 1e6:.0f} MB", flush=True)
    return os.path.join(cache, "captures.pt")


# ----------------------------------------------------------------------------------------------------------------------
# emit


def to_bits(x):
    """float64 / bf16 tensor holding bf16 values -> uint16 numpy bits; refuses values that are not bf16."""
    import torch

    if x.dtype != torch.bfloat16:
        b = x.to(torch.bfloat16)
        if not torch.equal(b.to(x.dtype), x):
            raise AssertionError("value is not exactly bf16")
        x = b
    return x.contiguous().view(torch.int16).numpy().view("<u2")


def attention_stepwise(oracle_mods, cfg, c: dict, glob: bool, p0: int):
    """The body of `Inkling2LoopRef.attention` (inkling2_ref.py) with every intermediate kept, on the captured inputs,
    for positions p0 + arange(T) on a fresh LayerCache. Returns a dict of float64 (bf16-valued) tensors."""
    torch, ref = oracle_mods
    F64, r = ref.F64, ref.Rounding()
    t, hq, hkv, d, dr = T, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.d_rel
    w = {k: v.to(F64) for k, v in c["w"].items()}
    pos = torch.arange(p0, p0 + t)
    lc = ref.LayerCache(torch.zeros(0, hkv, d, dtype=F64), torch.zeros(0, hkv, d, dtype=F64), p0,
                        {"k": torch.zeros(3, hkv * d, dtype=F64), "v": torch.zeros(3, hkv * d, dtype=F64)})
    x = {k: c[k].to(F64) for k in ("q", "r", "gate", "sink", "k", "v")}
    q, rr, gate, sink = x["q"].view(t, hq, d), x["r"].view(t, hq, dr), x["gate"].view(t, hq, d), x["sink"]
    out = {}
    if glob:
        yk, _ = ref.dynamic_conv(x["k"], c["xk"].to(F64), w["k_conv"], w["k_proj"], hkv, lc.conv["k"], r)
        yv, _ = ref.dynamic_conv(x["v"], c["xv"].to(F64), w["v_conv"], w["v_proj"], hkv, lc.conv["v"], r)
    else:
        yk, _ = ref.static_conv(x["k"], w["k_conv"][:, 0, :], lc.conv["k"], r)
        yv, _ = ref.static_conv(x["v"], w["v_conv"][:, 0, :], lc.conv["v"], r)
    k_hat = r.bf16(ref.l2clip(yk.view(t, hkv, d), cfg.qk_clip))
    yv = yv.view(t, hkv, d)
    v_hat = r.bf16(yv * torch.rsqrt((yv * yv).mean(-1, keepdim=True) + cfg.eps))
    q_hat = r.bf16(ref.l2clip(q * w["q_scale"], cfg.qk_clip))
    r_hat = r.bf16(ref.l2clip(rr * w["r_scale"], cfg.rel_clip))
    if glob:
        tau = ref.log_tau(pos, cfg.log_scaling_n_floor, cfg.log_scaling_alpha, r)[:, None, None]
        q_hat = r.bf16(q_hat * tau)
        r_hat = r.bf16(r_hat * tau)
    out.update(q_hat=q_hat, r_hat=r_hat, k_hat=k_hat, v_hat=v_hat)

    # the relative-bias table by distance, from the oracle's relative_bias (query t sees keys at distance 0..extent-1)
    ext = w["rel_proj"].shape[1]
    out["bias"] = torch.cat([ref.relative_bias(r_hat[i:i + 1], w["rel_proj"], pos[i:i + 1], pos[i] - torch.arange(ext), r)
                             for i in range(t)], 1).permute(1, 0, 2).contiguous()  # [T, Hq, extent]

    def core(kh, vh, tiled: bool):
        k_pos = torch.arange(p0, p0 + t)
        bias = ref.relative_bias(r_hat, w["rel_proj"], pos, k_pos, r)
        args = (q_hat, kh, vh, bias, gate, sink, pos, k_pos, None if glob else cfg.sliding_window, r)
        if tiled:
            return ref.attention_core_tiled(*args, chunk=512 if glob else 256, tile=64)
        return ref.attention_core(*args)

    def core_t(kh, vh, tiled):
        return core(kh, vh, tiled).reshape(t, hq * d)  # bf16(core output before wo), like the `core` tap

    out["plain"] = core_t(k_hat, v_hat, False)  # no kv8: what the default oracle forward computes
    if glob:
        k_rt, v_rt = ref.kv8_roundtrip(k_hat), ref.kv8_roundtrip(v_hat)
        out.update(k_rt=k_rt, v_rt=v_rt)
        kk, vv = k_rt, v_rt
    else:
        kk, vv = k_hat, v_hat
    out["out_exact"] = core_t(kk, vv, False)
    out["out_emul"] = core_t(kk, vv, True)
    return out


class Container:
    def __init__(self):
        self.entries: list[tuple[str, int, tuple, bytes]] = []
        self.names: set[str] = set()

    def add(self, name: str, dt: int, shape, raw: bytes) -> None:
        assert name not in self.names, name
        self.names.add(name)
        self.entries.append((name, dt, tuple(int(s) for s in shape), raw))

    def bf16(self, name: str, x) -> None:
        bits = to_bits(x)
        self.add(name, DT_BF16, bits.shape, bits.tobytes())

    def i32(self, name: str, values) -> None:
        import numpy as np

        a = np.asarray(values, dtype="<i4")
        self.add(name, DT_I32, a.shape, a.tobytes())

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(MAGIC + struct.pack("<II", 1, len(self.entries)))
            for name, dt, shape, raw in self.entries:
                nb = name.encode("ascii")
                f.write(struct.pack("<H", len(nb)) + nb + struct.pack("<BB", dt, len(shape)))
                f.write(struct.pack(f"<{len(shape)}I", *shape) + raw)


def emit(oracle: str, cache: str, out_path: str) -> dict:
    sys.path.insert(0, oracle)
    import torch

    import inkling2_ref as ref

    torch.set_num_threads(4)
    saved = torch.load(os.path.join(cache, "captures.pt"), weights_only=False)
    cap, tokens = saved["cap"], saved["tokens"]
    t0 = time.time()
    ct = Container()
    ct.i32("meta.tokens", tokens)
    ct.i32("meta.p0s", P0S)
    sanity = {}
    for pre, vi in VISITS.items():
        glob = pre.startswith("g")
        c = cap[vi]
        ref_core = c["core_tap"]  # the oracle forward's own `core` tap (float64, bf16 values), P0 = 0, kv8 off
        raw = {}
        w = c["w"]
        # the attention inputs exactly as attention() computes them (same lin(): bf16(x . W^T), float64 accumulate) -- they
        # are recomputed here from x_q / x_k / x_v and the captured raw projections are the stored ones
        for key, name in (("q_scale", "q_scale"), ("r_scale", "r_scale"), ("rel_proj", "rel_proj"),
                          ("k_conv", "k_conv"), ("v_conv", "v_conv")):
            ct.bf16(f"{pre}.w.{name}", w[key])
        for key in ("q", "r", "gate", "sink", "k", "v"):
            raw[key] = c["inputs"][key]
            ct.bf16(f"{pre}.x.{key}", raw[key])
        if glob:
            for s in "kv":
                dyn = ref.bf16(c[f"x{s}"].double() @ w[f"{s}_proj"].double().T)
                assert dyn.shape == (T, 32)
                ct.bf16(f"{pre}.x.dyn_{s}", dyn)
        ct.bf16(f"{pre}.x.hidden", c["hidden"])
        c.update(raw)
        for p0 in P0S:
            o = attention_stepwise((torch, ref), saved["cfg"], c, glob, p0)
            if p0 == 0:
                assert torch.equal(o["plain"], ref_core), f"{pre}: stepwise core differs from the oracle forward's tap"
                if glob:  # dynamic_conv's taps are exactly the stored dyn values
                    t_ = torch
                    for s in "kv":
                        dyn = ref.bf16(c[f"x{s}"].double() @ w[f"{s}_proj"].double().T).view(T, 8, 1, 4)
                        taps = (w[f"{s}_conv"].double().T.reshape(8, 128, 4)[None] + dyn).reshape(T, 1024, 4)
                        x_ = c[s]
                        acc = t_.zeros_like(x_)
                        full = t_.cat([t_.zeros(3, 1024, dtype=x_.dtype), x_], 0)
                        for lag in range(4):
                            acc = acc + taps[:, :, lag] * full[3 - lag:3 - lag + T]
                        y = ref.bf16((acc + x_) * ref.Rounding().const(9 ** -0.5))
                        yo, _ = ref.dynamic_conv(x_, c[f"x{s}"].double(), w[f"{s}_conv"].double(),
                                                 w[f"{s}_proj"].double(), 8, t_.zeros(3, 1024, dtype=x_.dtype), ref.Rounding())
                        assert t_.equal(y, yo), f"{pre}: dyn_{s} taps are not what dynamic_conv consumes"
                e = (o["out_emul"] - o["out_exact"]).abs().reshape(T, 24, 128)
                m = o["out_exact"].abs().reshape(T, 24, 128)
                sanity[pre] = (e.amax((0, 2)) / m.amax((0, 2))).tolist()
            tag = f"{pre}.p{p0}"
            for name in ("q_hat", "r_hat", "k_hat", "v_hat") + (("k_rt", "v_rt") if glob else ()):
                ct.bf16(f"{tag}.{name}", o[name])
            if p0 in BIAS_P0S:
                ct.bf16(f"{tag}.bias", o["bias"])
            ct.bf16(f"{tag}.out_exact", o["out_exact"])
            ct.bf16(f"{tag}.out_emul", o["out_emul"])
    ct.write(out_path)
    return {"seconds": time.time() - t0, "sanity": sanity, "tokens": tokens, "sentence": saved["sentence"],
            "forward_seconds": saved["seconds_forward"], "layer_seconds": saved["layer_seconds"],
            "fetched_bytes": saved["fetched_bytes"], "fetched_requests": saved["fetched_requests"]}


# ----------------------------------------------------------------------------------------------------------------------
# check


def expected_names() -> list[str]:
    names = ["meta.tokens", "meta.p0s"]
    for pre in VISITS:
        glob = pre.startswith("g")
        names += [f"{pre}.w.{n}" for n in ("q_scale", "r_scale", "rel_proj", "k_conv", "v_conv")]
        names += [f"{pre}.x.{n}" for n in ("q", "r", "gate", "sink", "k", "v")]
        names += ([f"{pre}.x.dyn_k", f"{pre}.x.dyn_v"] if glob else []) + [f"{pre}.x.hidden"]
        for p0 in P0S:
            names += [f"{pre}.p{p0}.{n}" for n in ("q_hat", "r_hat", "k_hat", "v_hat")]
            names += [f"{pre}.p{p0}.{n}" for n in (("k_rt", "v_rt") if glob else ())]
            names += [f"{pre}.p{p0}.bias"] if p0 in BIAS_P0S else []
            names += [f"{pre}.p{p0}.{n}" for n in ("out_exact", "out_emul")]
    return names


def read_container(path: str) -> list[tuple[str, int, tuple, int]]:
    """(name, dtype, shape, byte offset of the data) of every entry; checks the framing and that the file ends exactly."""
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == MAGIC, "bad magic"
    version, n = struct.unpack_from("<II", buf, 4)
    assert version == 1, f"version {version}"
    off, out = 12, []
    size = {DT_BF16: 2, DT_F32: 4, DT_I32: 4}
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", buf, off)
        name = buf[off + 2:off + 2 + nl].decode("ascii")
        off += 2 + nl
        dt, nd = struct.unpack_from("<BB", buf, off)
        off += 2
        shape = struct.unpack_from(f"<{nd}I", buf, off)
        off += 4 * nd
        nbytes = size[dt]
        for s in shape:
            nbytes *= s
        out.append((name, dt, shape, off))
        off += nbytes
    assert off == len(buf), f"{len(buf) - off} trailing bytes"
    return out


def check(path: str) -> int:
    entries = read_container(path)
    for name, dt, shape, _ in entries:
        print(f"{name:28s} {DT_NAME[dt]:5s} {list(shape)}")
    got = [e[0] for e in entries]
    missing = sorted(set(expected_names()) - set(got))
    extra = sorted(set(got) - set(expected_names()))
    print(f"{len(entries)} entries, {os.path.getsize(path)} bytes, sha256 "
          f"{hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
    if missing or extra or len(got) != len(set(got)):
        print("MISSING", missing, "UNEXPECTED", extra)
        return 1
    return 0


# ----------------------------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--oracle", default=DEFAULT_ORACLE, help="the recipe's oracle/ directory")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="scratch directory (sparse pack copy, captures); keep outside the repo")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--stage", choices=("capture", "emit", "all"), default="all")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--check", metavar="FILE", help="re-read a container and list its entries")
    a = ap.parse_args(argv)
    if a.check:
        return check(a.check)
    os.makedirs(a.cache, exist_ok=True)
    if a.stage in ("capture", "all"):
        capture(a.oracle, a.cache, a.threads)
    if a.stage in ("emit", "all"):
        info = emit(a.oracle, a.cache, a.out)
        print(json.dumps({k: info[k] for k in ("seconds", "tokens", "sentence", "forward_seconds")}))
        for pre, per_head in info["sanity"].items():
            print(f"{pre} P0=0 max|out_emul-out_exact|/max|out_exact| per head: max {max(per_head):.3e}  "
                  + " ".join(f"{x:.1e}" for x in per_head))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
