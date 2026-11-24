import os
import sys

ORACLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ORACLE, os.path.join(ORACLE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))


def _tiny(cfg, seed):
    from inkling2_ref import DenseExperts, DictWeights, Inkling2LoopRef
    from synthetic import random_weights

    w = random_weights(cfg, seed=seed)
    src = DictWeights(w)

    def make(rounding=None):
        return Inkling2LoopRef(cfg, src, DenseExperts(src), rounding=rounding, unembed_chunk=128, query_chunk=5)

    return cfg, w, make


@pytest.fixture(scope="session")
def tiny():
    """The tiny looped model (8 layers, AttnRes blocks of 2, core [2, 6) x 3 passes = 16 passes, window 8) with dense
    random weights: (config, weights by pack name, `make(rounding=None)` -> Inkling2LoopRef)."""
    from synthetic import tiny_config

    return _tiny(tiny_config(), seed=1)


@pytest.fixture(scope="session")
def tiny_wide_core():
    """A second shape: 10 layers, a core [2, 8) of three AttnRes blocks run twice (the message bank holds two blocks)."""
    from synthetic import tiny_config

    cfg = tiny_config(num_layers=10, local_layer_ids=(0, 2, 4, 6, 8), loop_start=2, loop_end=8, loop_passes=2,
                      drafter_target_layers=(3, 9, 13, 16))
    return _tiny(cfg, seed=2)
