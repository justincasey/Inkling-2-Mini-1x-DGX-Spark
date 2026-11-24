import sys, subprocess, json, os
which, root = sys.argv[1], sys.argv[2]
sys.path[:0] = [root + "/oracle", root + "/oracle/tests"]
import numpy as np, torch
from inkling2_ref import *
from synthetic import *
cfg = tiny_config(); plain, groups, ref = random_exl3_pack(cfg, seed=5)
m = Inkling2LoopRef(cfg, DictWeights(ref), ExpertGroups(groups), unembed_chunk=128)
r = generate(m, [3, 9, 27, 81, 243, 5, 7, 11, 13], 6)
torch.save({k: v for k, v in r.items() if torch.is_tensor(v)}, f"/tmp/cd/{which}.pt")
