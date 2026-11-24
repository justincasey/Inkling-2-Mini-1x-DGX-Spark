import sys, os, json
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle")]
import numpy as np, torch
from pack import PackIndex, PackWeights
from inkling2_ref import Inkling2LoopConfig, Inkling2LoopRef, f32, F64, rmsnorm_w
torch.set_num_threads(8)
pk = "/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw"
cfg = Inkling2LoopConfig.from_hf(json.load(open(pk + "/config.json")))
raw = torch.load("out/chat-sky-effort0.9.raw.pt", weights_only=False)
H = raw["hidden_full"]  # [49, 49, 2304] bf16
m = Inkling2LoopRef(cfg, PackWeights(PackIndex(pk)), None)
vis = m.visits
starts = [v.index for v in vis if v.block_start]            # passes whose layer opens a block
last = {}                                                    # the chain that reaches the end: layer -> its last pass
for v in vis: last[v.layer] = v.index
blocks_idx = [last[l] for l in (0, 6, 12, 18)]
blocks = [H[k].double() for k in blocks_idx]; p = H[48].double()
hf = m.mix(blocks, p, m.query("model.llm.attn_res.output_query"), [m.w64("model.llm.norm.weight")])[0]  # [49, 2304] bf16 values
print("block passes", blocks_idx)
u = m.w.row_range("model.llm.unembed.weight", 49152, 65536)  # chunk 3 of 16384
tok = 64928 - 49152
single = hf[28:29] @ u.T
batch = hf @ u.T
print("fp64 z single / batch:", repr(float(single[0, tok])), repr(float(batch[28, tok])), "diff", float(single[0, tok] - batch[28, tok]))
cap = cfg.final_softcap
for name, z in (("single(cached)", single[0, tok]), ("batch(full)", batch[28, tok])):
    zf = f32(z); out = f32(cap * torch.tanh(zf / cap))
    print(name, "f32(z) =", repr(float(zf)), " logit =", repr(float(out)))
print("saved cached logit", repr(float(raw["logits_cached"][28, 64928])), "saved full logit", repr(float(raw["logits_full"][28, 64928])))
ex = (hf[28].numpy().astype(np.longdouble) * u[tok].numpy().astype(np.longdouble)).sum()
print("long double z =", repr(float(ex)), "(longdouble eps %.3g)" % np.finfo(np.longdouble).eps)
z32 = float(np.float32(float(ex))); 
print("f32 neighbours of z: spacing %.3g" % np.spacing(np.float32(z32)))
