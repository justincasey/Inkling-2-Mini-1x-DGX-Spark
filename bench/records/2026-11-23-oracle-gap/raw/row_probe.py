"""Rows of the chat prompts where the engine's argmax differs from the oracle's: the per-row logits error of the engine against the
engine- and spec-fixtures and of the two fixtures against each other, and the hidden-state error of that row per visit."""
import sys, json
sys.path.insert(0, "/home/spark/inkling-2-mini-1x-dgx-spark/lab/fixgate/oracle")
import torch, st, engine_gap as eg
B = "/home/spark/inkling-2-mini-1x-dgx-spark"
L = B + "/lab/fixgate"
name, rows = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
fe, _ = eg.load_fixture(f"{B}/fixtures/engine/{name}.safetensors")
fs, _ = eg.load_fixture(f"{B}/fixtures/spec/{name}.safetensors")
tokens = [int(t) for t in fe["tokens"]]
d = eg.assemble(L + "/dump", tokens[:-1])
out = {}
for label, (a, b) in {"engine_vs_engfix": (fe, d), "engine_vs_specfix": (fs, d), "floor_engfix_vs_specfix": (fs, fe)}.items():
    s = eg.logits_rows(a["logits"], b["logits"] if "logits" in b else None)
    out[label] = {"rows": {r: {"rel": round(float(s["rel"][r]), 4), "maxabs_rms": round(float(s["maxabs_rms"][r]), 3), "oracle_margin": round(float(s["oracle_margin"][r]), 3), "engine_margin": round(float(s["engine_margin"][r]), 3)} for r in rows}}
    h = [eg.row_stats(a["hidden"][v, rows], b["hidden"][v, rows])["rel"] for v in range(49)]
    out[label]["hidden_rel_by_visit"] = {r: [round(float(h[v][i]), 4) for v in range(0, 49, 4)] for i, r in enumerate(rows)}
print(name, "token ids at rows", {r: tokens[r] for r in rows}, "next", {r: tokens[r + 1] for r in rows})
print(json.dumps(out, indent=1))
