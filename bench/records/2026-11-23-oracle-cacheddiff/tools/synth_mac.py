"""Cached vs full recompute, bitwise, tiny looped config at growing hidden size (dense experts)."""
import sys; sys.path[:0]=[__import__('os').path.expanduser('~/projects/inkling2-mini-oracle/oracle'),__import__('os').path.expanduser('~/projects/inkling2-mini-oracle/oracle/tests')]
import numpy as np, torch, json
from inkling2_ref import *
from synthetic import tiny_config, random_weights
torch.set_num_threads(4)
H=int(sys.argv[1]); 
cfg=tiny_config(hidden_size=H, head_dim=128, d_rel=16, num_heads=8, num_kv_heads=2, moe_intermediate=H//2, n_experts=8, top_k=4,
                sliding_window=16, rel_extent=16, vocab_size=256, unpadded_vocab_size=250, gate_rank=32)
w=random_weights(cfg, seed=7); src=DictWeights(w)
m=Inkling2LoopRef(cfg, src, DenseExperts(src), unembed_chunk=128, query_chunk=64)
res=[]
for n_prompt in (3,7,12,17,25,33):
    x=torch.from_numpy(np.random.default_rng(n_prompt).integers(0,250,n_prompt)).tolist()
    g=generate(m,x,10)
    full=m.forward(g["tokens"][:-1], record=True)
    d=(full.logits-g["logits"]).abs()
    rms=full.logits.pow(2).mean(-1).sqrt()
    dh=(full.hidden-g["hidden"]).abs().amax((0,2)) if False else None
    res.append((n_prompt, bool(torch.equal(full.logits,g["logits"])), float((d.amax(-1)/rms).max())))
    print(res[-1], flush=True)
