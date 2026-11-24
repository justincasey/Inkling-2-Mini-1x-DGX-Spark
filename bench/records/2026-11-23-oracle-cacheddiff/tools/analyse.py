import sys, os, json, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle")]
import numpy as np, torch
from pack import PackIndex, PackWeights
from inkling2_ref import f32
torch.set_num_threads(8)
raw = torch.load("out/chat-sky-effort0.9.raw.pt", weights_only=False)
lc, lf = raw["logits_cached"], raw["logits_full"]
d = (lc != lf)
print("rows with differing logits:", torch.nonzero(d.any(-1)).flatten().tolist(), "elements differing per row:", d.sum(-1)[d.any(-1)].tolist())
for t in torch.nonzero(d.any(-1)).flatten().tolist():
    v = torch.nonzero(d[t]).flatten().tolist()
    print("row", t, "token ids", v, "cached", [float(lc[t, i]) for i in v], "full", [float(lf[t, i]) for i in v], "row max", float(lf[t].max()), "argmax", int(lf[t].argmax()))
    print("hidden equal in every pass (bf16 bits):", bool(torch.equal(raw["hidden_cached"][:, t], raw["hidden_full"][:, t])), " routes equal:", bool(torch.equal(raw["idx_cached"][:, t], raw["idx_full"][:, t])))
# the suspected source: the unembed product of a 1-row and an n-row float64 matmul
pk = "/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw"
w = PackWeights(PackIndex(pk))
u = w.row_range("model.llm.unembed.weight", 0, 32768)  # float64 [32768, 2304]
g = torch.Generator().manual_seed(0)
h = torch.randn(49, 2304, generator=g, dtype=torch.float64).to(torch.bfloat16).double()  # bf16-valued rows, unit scale
batch = f32(h @ u.T)
single = torch.cat([f32(h[i:i + 1] @ u.T) for i in range(49)])
exact = torch.cat([f32(torch.from_numpy(np.longdouble(1) * 0 + (h[i].numpy().astype(np.longdouble) @ u.numpy().astype(np.longdouble).T).astype(np.float64))) for i in range(3)]) if False else None
dd = (batch != single)
print("random-row unembed test: logits", batch.numel(), "differing fp32 values between 1-row and 49-row products:", int(dd.sum()),
      "max abs fp64 difference of the products:", float((h @ u.T - torch.cat([h[i:i+1] @ u.T for i in range(49)])).abs().max()))
