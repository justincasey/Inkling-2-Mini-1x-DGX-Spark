"""The EXL3 pack as oracle weight/expert sources.

`PackWeights` reads plain tensors (BF16/F32) from a pack folder by positional reads and decodes EXL3-dense weights
(`X.weight` backed by `X.trellis/suh/svh/mul1`, any K the trellis says) into nn.Linear layout, through
`model.safetensors.index.json` when present (else every *.safetensors header), so only what a step asks for is resident.
`PackExperts` decodes routed experts (`model.llm.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj` EXL3 groups:
trellis/suh/svh + codebook marker) with `oracle/exl3.py` and runs their FFNs, optionally in worker processes.
A folder that lacks some experts (a rank's split staging) raises KeyError when a missing expert is selected.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Sequence

import numpy as np
import torch

import exl3
import st
from inkling2_ref import Rounding, expert_ffn

PROJS = ("gate_proj", "up_proj", "down_proj")


class PackIndex:
    def __init__(self, root: str, files: list[str] | None = None):
        """`files` (names relative to root) restricts the index to exactly those shards: their headers are scanned and
        `model.safetensors.index.json` is ignored (e.g. the drafter's `mtp.safetensors`)."""
        self.root = root
        idx = os.path.join(root, "model.safetensors.index.json")
        self._entries: dict[str, st.Entry] = {}
        self._headers: dict[str, dict[str, st.Entry]] = {}
        if files is None and os.path.exists(idx):
            with open(idx) as f:
                self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        else:
            self.weight_map = {}
            for fn in sorted(files) if files is not None else sorted(os.listdir(root)):
                if files is not None or fn.endswith(".safetensors"):
                    for name in self._header(fn):
                        if name in self.weight_map:
                            raise ValueError(f"{name} appears in {self.weight_map[name]} and {fn}")
                        self.weight_map[name] = fn

    def _header(self, fn: str) -> dict[str, st.Entry]:
        if fn not in self._headers:
            self._headers[fn] = st.read_header(os.path.join(self.root, fn))
        return self._headers[fn]

    def entry(self, name: str) -> st.Entry:
        if name not in self._entries:
            fn = self.weight_map.get(name)
            if fn is None:
                raise KeyError(f"{name} is not in the pack index")
            path = os.path.join(self.root, fn)
            if not os.path.exists(path):
                raise KeyError(f"{name} lives in {fn}, which this folder does not hold")
            hdr = self._header(fn)
            if name not in hdr:
                raise KeyError(f"the index places {name} in {fn}, whose header does not list it")
            self._entries[name] = hdr[name]
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        fn = self.weight_map.get(name)
        return fn is not None and os.path.exists(os.path.join(self.root, fn))


_SLAB = 1024  # columns of W decoded at once: K x 1024 float64 is 19 MB at K = 2304


class _Group:
    """Metadata of one EXL3 group (small: scales, codebook, trellis entry); the trellis is read by column slab."""

    def __init__(self, ix: "PackIndex", prefix: str):
        self.prefix = prefix
        self.entry = ix.entry(prefix + ".trellis")
        if len(self.entry.shape) != 3 or self.entry.dtype != "I16":
            raise ValueError(f"{prefix}.trellis: expected int16 [K/16, N/16, 16*bits], got {self.entry.dtype} {self.entry.shape}")
        self.kt, self.nt, self.last = self.entry.shape
        self.bits = exl3.bits_twice_from_trellis_last(self.last) / 2.0
        self.k, self.n = self.kt * 16, self.nt * 16
        self.suh = st.read_numpy(ix.entry(prefix + ".suh")).astype(np.float64)
        self.svh = st.read_numpy(ix.entry(prefix + ".svh")).astype(np.float64)
        if self.suh.shape != (self.k,) or self.svh.shape != (self.n,):
            raise ValueError(f"{prefix}: scales {self.suh.shape}/{self.svh.shape} do not match K={self.k}, N={self.n}")
        self.codebook = _codebook_of(ix, prefix)

    def trellis(self, c0: int, c1: int) -> np.ndarray:
        """Trellis tile columns [c0/16, c1/16) for every kt row: kt positional reads of one contiguous run each."""
        t0, t1 = c0 // 16, c1 // 16
        if t0 == 0 and t1 == self.nt:
            return st.read_numpy(self.entry)
        e, tb = self.entry, self.last * 2
        out = np.empty((self.kt, t1 - t0, self.last), dtype=np.int16)
        for kk in range(self.kt):
            raw = st._pread(e.path, e.start + (kk * self.nt + t0) * tb, (t1 - t0) * tb)
            out[kk] = np.frombuffer(raw, dtype=np.int16).reshape(t1 - t0, self.last)
        return out

    def slab(self, c0: int, c1: int) -> np.ndarray:
        """Effective weight columns [c0, c1) (multiples of 128) in nn.Linear layout, float64 [c1 - c0, K]."""
        w = exl3.decode_weight(self.trellis(c0, c1), self.codebook, self.suh, self.svh[c0:c1])
        return np.ascontiguousarray(w.T)


def _codebook_of(ix: "PackIndex", prefix: str) -> str:
    markers = [m for m in ("mcg", "mul1") if f"{prefix}.{m}" in ix.weight_map]
    if len(markers) > 1:
        raise ValueError(f"{prefix}: both mcg and mul1 markers")
    if not markers:
        return "3inst"  # a group with no marker tensor is the default 3inst codebook
    raw = st.read_numpy(ix.entry(f"{prefix}.{markers[0]}"))
    return exl3.codebook_from_marker(markers[0], int(raw.reshape(-1).view(np.uint32)[0]))


class PackWeights:
    """Weights by name from a pack. Plain tensors come back as stored (BF16 stays bfloat16). A name `X.weight` whose
    `X.trellis` exists is an EXL3-dense weight: it comes back decoded, float64, in nn.Linear layout [N_out, K_in], and
    `row_range`/`rows` decode only the 128-column strips of W (= weight rows) they need.

    `cache_bytes` > 0 keeps decoded EXL3-dense `tensor(name)` results and `row_range(name, a, b)` slabs in an LRU holding
    at most that many float64 bytes (an entry above the budget is returned uncached). A hit returns the SAME tensor
    object, so callers must not mutate it. 0 (default) = no cache."""

    def __init__(self, index: PackIndex, cache_bytes: int = 0):
        self.ix = index
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._held = 0
        self._hits = 0
        self._misses = 0
        self._groups: dict[str, _Group] = {}
        self._lin: OrderedDict[str, exl3.Exl3Linear] = OrderedDict()
        self._lin_held = 0
        self.linear_cache_bytes = 24 << 30

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "held_bytes": self._held}

    def _exl3(self, name: str) -> _Group | None:
        if not name.endswith(".weight"):
            return None
        prefix = name[: -len(".weight")]
        if prefix + ".trellis" not in self.ix.weight_map:  # by the index, so a missing shard is named by `entry`
            return None
        if prefix not in self._groups:
            self._groups[prefix] = _Group(self.ix, prefix)
        return self._groups[prefix]

    def exl3_linear(self, name: str) -> exl3.Exl3Linear | None:
        """The EXL3 group behind `X.weight` as a callable linear (`inkling2_ref.Emulation.dense_fp16`), None for a plain tensor.
        Decoded W_q stays in memory, least recently used first out, within `linear_cache_bytes` (a linear above it is not kept)."""
        if self._exl3(name) is None:
            return None
        lin = self._lin.get(name)
        if lin is not None:
            self._lin.move_to_end(name)
            return lin
        lin = load_exl3(self.ix, name[: -len(".weight")])
        size = lin.wq.nbytes
        if size <= self.linear_cache_bytes:
            while self._lin_held + size > self.linear_cache_bytes:
                _, old = self._lin.popitem(last=False)
                self._lin_held -= old.wq.nbytes
            self._lin[name] = lin
            self._lin_held += size
        return lin

    def _cached(self, key: tuple, make) -> torch.Tensor:
        if self.cache_bytes <= 0:
            return make()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self._hits += 1
            return hit
        self._misses += 1
        t = make()
        size = t.numel() * t.element_size()
        if size <= self.cache_bytes:
            while self._held + size > self.cache_bytes:
                _, old = self._cache.popitem(last=False)
                self._held -= old.numel() * old.element_size()
            self._cache[key] = t
            self._held += size
        return t

    def _decode(self, g: _Group, r0: int, r1: int) -> torch.Tensor:
        if not 0 <= r0 < r1 <= g.n:
            raise IndexError(f"rows [{r0}, {r1}) outside the {g.n} rows of {g.prefix}")
        a, b = r0 // exl3.HAD * exl3.HAD, -(-r1 // exl3.HAD) * exl3.HAD
        out = torch.empty(r1 - r0, g.k, dtype=torch.float64)
        for c0 in range(a, b, _SLAB):
            c1 = min(c0 + _SLAB, b)
            w = g.slab(c0, c1)
            lo, hi = max(r0, c0), min(r1, c1)
            out[lo - r0:hi - r0] = torch.from_numpy(w[lo - c0:hi - c0])
        return out

    def tensor(self, name: str) -> torch.Tensor:
        g = self._exl3(name)
        if g is None:
            return st.read_tensor(self.ix.entry(name))
        return self._cached((name, 0, g.n), lambda: self._decode(g, 0, g.n))

    def rows(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        g = self._exl3(name)
        if g is None:
            e = self.ix.entry(name)
            uniq = sorted(set(int(i) for i in ids.tolist()))
            got = {i: st.read_rows(e, i, i + 1)[0] for i in uniq}
            return torch.stack([got[int(i)] for i in ids.tolist()])
        lst = [int(i) for i in ids.tolist()]
        strips: dict[int, torch.Tensor] = {}
        for i in lst:
            if not 0 <= i < g.n:
                raise IndexError(f"row {i} outside the {g.n} rows of {name}")
            if i // exl3.HAD not in strips:
                s0 = i // exl3.HAD * exl3.HAD
                strips[i // exl3.HAD] = self._decode(g, s0, s0 + exl3.HAD)
        return torch.stack([strips[i // exl3.HAD][i % exl3.HAD] for i in lst])

    def row_range(self, name: str, start: int, stop: int) -> torch.Tensor:
        g = self._exl3(name)
        if g is None:
            return st.read_rows(self.ix.entry(name), start, stop)
        return self._cached((name, start, stop), lambda: self._decode(g, start, stop))


def load_exl3(ix: PackIndex, prefix: str) -> exl3.Exl3Linear:
    """One EXL3 group by prefix: trellis int16, suh/svh fp16 (the pack's `out_scales: always`), codebook marker."""
    trellis = st.read_numpy(ix.entry(prefix + ".trellis"))
    suh = st.read_numpy(ix.entry(prefix + ".suh")).astype(np.float64)
    svh = st.read_numpy(ix.entry(prefix + ".svh")).astype(np.float64)
    return exl3.Exl3Linear(trellis, _codebook_of(ix, prefix), suh, svh)


def expert_prefix(layer: int, e: int, proj: str) -> str:
    return f"model.llm.layers.{layer}.mlp.experts.{e}.{proj}"


_worker_ix: PackIndex | None = None


def exit_with_parent(poll_s: float = 1.0) -> None:
    """For a pool worker's initializer: exit as soon as the process that started it is gone. A spawn worker whose
    parent is killed would otherwise wait on its call queue forever (reparented to init), holding its memory."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(poll_s)
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _worker_init(root: str) -> None:
    global _worker_ix
    exit_with_parent()
    _worker_ix = PackIndex(root)


def _worker_ffn(layer: int, e: int, x: np.ndarray, gate_cap: float, up_cap: float, exact: bool,
                fp16_in: bool = False) -> np.ndarray:
    ix = _worker_ix
    lins = [load_exl3(ix, expert_prefix(layer, e, p)) for p in PROJS]
    return expert_ffn(*lins, x, gate_cap, up_cap, Rounding.exact() if exact else Rounding(), fp16_in)


class PackExperts:
    def __init__(self, index: PackIndex, workers: int = 0):
        self.ix = index
        self.pool = None
        if workers > 0:
            self.pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_worker_init,
                                            initargs=(index.root,))

    def ffn_many(self, layer: int, jobs: Sequence[tuple[int, np.ndarray]], gate_cap: float, up_cap: float,
                 rounding: Rounding, fp16_in: bool = False) -> list[np.ndarray]:
        if self.pool is None:
            out = []
            for e, x in jobs:
                lins = [load_exl3(self.ix, expert_prefix(layer, e, p)) for p in PROJS]
                out.append(expert_ffn(*lins, x, gate_cap, up_cap, rounding, fp16_in))
            return out
        futs = [self.pool.submit(_worker_ffn, layer, e, x, gate_cap, up_cap, rounding.is_exact, fp16_in)
                for e, x in jobs]
        return [f.result() for f in futs]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
