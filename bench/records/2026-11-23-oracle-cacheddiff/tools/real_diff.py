"""Cached (prefill + greedy decode) vs full recompute of one prompt on the real pack, with per-position statistics and the
per-pass hidden / sub-op comparison. Usage: real_diff.py PACK OUT_DIR PROMPT_NAME [--workers N] [--threads N] [--cache-gib G]"""
import argparse, json, os, sys, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle")]
import numpy as np, torch
from tokenizers import Tokenizer
from inkling2_ref import Inkling2LoopConfig, Inkling2LoopRef, greedy
from pack import PackExperts, PackIndex, PackWeights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("name")
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-gib", type=float, default=10.0)
    ap.add_argument("--new-tokens", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = Inkling2LoopConfig.from_hf(json.load(open(os.path.join(a.pack, "config.json"))))
    prompts = {p["name"]: p for p in map(json.loads, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle", "prompts.jsonl")))}
    p = prompts[a.name]
    ids = p["ids"] if "ids" in p else Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json")).encode(p["text"], add_special_tokens=False).ids
    n_new = a.new_tokens or int(p.get("new_tokens", 16))
    ix = PackIndex(a.pack)
    m = Inkling2LoopRef(cfg, PackWeights(ix, int(a.cache_gib * (1 << 30))), PackExperts(ix, workers=a.workers))
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
    log("prompt", a.name, len(ids), "tokens, new", n_new, "threads", a.threads, "workers", a.workers)

    # cached: prefill, then one token at a time (the generator's own loop), with taps and records per call
    cache = m.new_cache(); calls = []
    tokens = list(ids)
    t0 = time.time()
    m.taps = {}
    outs = [m.forward(ids, cache, record=True)]; calls.append((0, len(ids), m.taps)); log("prefill done", round(time.time() - t0), "s")
    for i in range(n_new):
        nxt = int(greedy(outs[-1].logits[-1:])[0]); tokens.append(nxt)
        if len(tokens) - len(ids) == n_new: break
        m.taps = {}
        t1 = time.time()
        outs.append(m.forward([nxt], cache, record=True)); calls.append((len(tokens) - 1, 1, m.taps))
        log("decode step", i, round(time.time() - t1, 1), "s")
    cat = lambda f, d: torch.cat([getattr(o, f) for o in outs], dim=d)
    lc, hc, ic, mc = cat("logits", 0), cat("hidden", 1), cat("router_idx", 1), cat("router_margin", 1)
    log("cached total", round(time.time() - t0), "s")

    t0 = time.time(); m.taps = {}
    full = m.forward(torch.tensor(tokens[:-1]), record=True); ftaps = m.taps
    log("full recompute", round(time.time() - t0), "s")
    lf, hf, jf, mf = full.logits, full.hidden, full.router_idx, full.router_margin
    n = lf.shape[0]
    torch.save({'tokens': tokens, 'logits_cached': lc, 'logits_full': lf, 'hidden_cached': hc, 'hidden_full': hf, 'idx_cached': ic, 'idx_full': jf, 'margin_cached': mc, 'margin_full': mf, 'calls': calls, 'ftaps': ftaps}, os.path.join(a.out, a.name + '.raw.pt'))
    log('raw saved')
    rep = {"prompt": a.name, "prompt_len": len(ids), "n_rows": n, "tokens": tokens, "positions": []}
    for t in range(n):
        d = (lf[t] - lc[t]).double().abs()
        rms = float(lf[t].double().pow(2).mean().sqrt())
        top = lambda x: torch.topk(x, 2).indices.tolist()
        first_hid = None
        dh = (hf[:, t].double() - hc[:, t].double()).abs().amax(-1)  # [V+1]
        nz = torch.nonzero(dh).flatten().tolist()
        rep["positions"].append({
            "t": t, "role": "prompt" if t < len(ids) - 1 else "generated-input", "logit_max_abs": float(d.max()), "logit_rel_rms": float(d.max()) / rms,
            "argmax_equal": int(lf[t].argmax()) == int(lc[t].argmax()), "top2_equal": top(lf[t]) == top(lc[t]),
            "first_pass_hidden_differs": nz[0] if nz else None, "hidden_max_abs": float(dh.max()),
            "routes_differ_passes": [k for k in range(jf.shape[0]) if sorted(jf[k, t].tolist()) != sorted(ic[k, t].tolist())],
        })
    bad = [q for q in rep["positions"] if q["logit_max_abs"] > 0]
    rep["n_differing_positions"] = len(bad)
    if bad:
        t = bad[0]["t"]; k = bad[0]["first_pass_hidden_differs"]
        rep["first_differing"] = {"t": t, "pass": k, "router_margin_cached_each_pass": [float(x) for x in mc[:, t]], "router_margin_full": [float(x) for x in mf[:, t]]}
        # sub-op localisation at (t, pass k-1 .. k): which tap of which pass differs first
        ci = max(j for j, (s, nn, _) in enumerate(calls) if s <= t); s0, nn, ct = calls[ci]; r = t - s0
        row = lambda name, x, rr: (x[:, rr] if name in ("msg", "attn_in") else x[rr])
        ops = []
        for k2 in sorted(ct):
            for name in ("msg", "attn_in", "core", "attn", "mlp_in", "router_logits", "moe"):
                if name in ct[k2] and name in ftaps.get(k2, {}):
                    x, y = row(name, ct[k2][name], r).double(), row(name, ftaps[k2][name], t).double()
                    if not torch.equal(x, y): ops.append({"pass": k2, "op": name, "max_abs": float((x - y).abs().max()), "n_elems": int((x != y).sum()), "of": x.numel()})
        rep["first_differing"]["ops_in_pass_order"] = ops[:12]
    json.dump(rep, open(os.path.join(a.out, f"{a.name}.report.json"), "w"), indent=1)
    torch.save({"tokens": tokens, "logits_cached": lc, "logits_full": lf}, os.path.join(a.out, f"{a.name}.logits.pt"))
    log("differing positions", len(bad), "of", n)
    for q in rep["positions"]:
        print(q["t"], q["role"], "%.3e %.3e" % (q["logit_max_abs"], q["logit_rel_rms"]), q["argmax_equal"], q["top2_equal"], q["first_pass_hidden_differs"], q["routes_differ_passes"], flush=True)
    if bad: print(json.dumps(rep["first_differing"], indent=1))
    m.experts.close()

if __name__ == '__main__':
    main()
