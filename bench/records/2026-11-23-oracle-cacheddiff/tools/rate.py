import sys, os, json, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle")]
import numpy as np, torch
from pack import PackIndex, PackWeights
from inkling2_ref import Inkling2LoopConfig, Inkling2LoopRef, f32
torch.set_num_threads(8)
pk = "/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw"
cfg = Inkling2LoopConfig.from_hf(json.load(open(pk + "/config.json")))
raw = torch.load("out/chat-sky-effort0.9.raw.pt", weights_only=False)
H = raw["hidden_full"]
m = Inkling2LoopRef(cfg, PackWeights(PackIndex(pk)), None)
blocks = [H[k].double() for k in (0, 30, 36, 42)]
hf = m.mix(blocks, H[48].double(), m.query("model.llm.attn_res.output_query"), [m.w64("model.llm.norm.weight")])[0]
u = m.w.row_range("model.llm.unembed.weight", 0, 32768)
cap = cfg.final_softcap
f = lambda z: f32(cap * torch.tanh(f32(z) / cap))
gen = torch.Generator().manual_seed(1)
tot = diff = diff_z = 0; maxulp = 0.0; t0 = time.time()
for it in range(40):
    s = (torch.randint(0, 2, (2304,), generator=gen) * 2 - 1).double()
    h = hf * s
    zb = h @ u.T
    zs = torch.cat([h[i:i + 1] @ u.T for i in range(h.shape[0])])
    lb, ls = f(zb), f(zs)
    d = lb != ls
    tot += lb.numel(); diff += int(d.sum()); diff_z += int((f32(zb) != f32(zs)).sum())
    if d.any():
        ulp = ((lb - ls).abs() / torch.from_numpy(np.spacing(np.abs(ls.numpy()).astype(np.float32)).astype(np.float64)))[d].max()
        maxulp = max(maxulp, float(ulp))
print("logits compared:", tot, " differing after softcap:", diff, " differing f32(z):", diff_z, " max |diff| in fp32 ulps:", maxulp, " rate per logit: %.3g" % (diff / tot), " seconds", round(time.time() - t0))
