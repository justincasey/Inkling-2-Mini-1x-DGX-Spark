"""Diagnostic wrapper: oracle/teacher_force.py with the engine's `exl3_fp16` emulation (NUMERICS.md "Engine emulation") applied to every
EXL3 dense linear the oracle's attention path calls (`Inkling2LoopRef.lin`: q, r, gate, k, v): the rotated input (x * suh) H is rounded
to fp16 before the W_q product. Run from oracle/ with the oracle's own arguments (it only patches `lin`); `wo_ud` (`matmul`) is left as
the oracle has it, so only the `core` rows of the report measure the emulation."""
import sys

import numpy as np
import torch

import inkling2_ref as ir
import teacher_force as tf
from pack import PackIndex, load_exl3

_cache = {}
_ix = {}
_orig = ir.Inkling2LoopRef.lin


def lin(self, x, name):
    w = self.w
    g = w._exl3(name)
    if g is None:
        return _orig(self, x, name)
    if name not in _cache:
        _cache[name] = load_exl3(w.ix, name[: -len(".weight")])
    y = _cache[name](x.numpy(), in_round=ir.np_f16)
    return self.r.bf16(torch.from_numpy(np.ascontiguousarray(y)))


ir.Inkling2LoopRef.lin = lin
sys.exit(tf.main())
