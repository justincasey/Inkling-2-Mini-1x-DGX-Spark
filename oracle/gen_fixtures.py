"""Generate oracle fixtures over a pack folder holding the whole Inkling-2-Mini model (every routed expert).

    python oracle/gen_fixtures.py --pack DIR --prompts oracle/prompts.jsonl --out DIR [--workers N] [--threads N]
                                  [--only NAME ...] [--verify-recompute | --verify-exact] [--emulate none|engine|NAMES]

Prompts (JSON lines): {"name": ..., "text": ...} (tokenized with the pack's tokenizer.json, special-token strings
recognised, nothing added) or {"name": ..., "ids": [...]}, optional "new_tokens" (default --new-tokens).
Per prompt the oracle prefills, then greedy-decodes from its cache, and writes OUT/NAME.safetensors. With V = the number
of layer passes a token goes through (48 on the real model: 24 layers, the core [6, 18) three times) and N tokens:
  tokens I64 [N] (prompt + generated), logits F32 [N-1, unpadded vocab] (row t predicts token t+1), h_final BF16
  [N-1, H], hidden BF16 [V+1, N-1, H] (P entering each pass, then P after the last), core_out BF16
  [loop_passes-1, N-1, H] (P leaving the last core layer of every core pass but the last: the message bank partial of
  the next core pass; `hidden` cannot hold it because the next pass restarts from the prelude's state),
  router_idx I32 / router_w F32 [V, N-1, k], router_margin F32 [V, N-1];
metadata: name, prompt_len, pack_config_sha256, oracle_source_sha256, numerics_version, emulate.
plus OUT/manifest.json (sha256 of every fixture, decoded continuations, timings). `--emulate` reproduces the engine's
lossy numerics (inkling2_ref.Emulation; docs/ORACLE-GAP.md) and is recorded in the metadata as `emulate`; write each
emulation to its own OUT. The model is single-device (no tensor parallelism): there is no rank option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import st
from inkling2_ref import NUMERICS_VERSION, Emulation, Inkling2LoopConfig, Inkling2LoopRef, generate
from pack import PackExperts, PackIndex, PackWeights

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def oracle_source_sha() -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(HERE)):
        if fn.endswith(".py"):
            with open(os.path.join(HERE, fn), "rb") as f:
                h.update(fn.encode() + b"\0" + f.read())
    return h.hexdigest()


def load_prompts(path: str, only: list[str] | None) -> list[dict]:
    with open(path) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    names = [p["name"] for p in prompts]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate prompt names")
    if only:
        missing = set(only) - set(names)
        if missing:
            raise ValueError(f"no such prompts: {sorted(missing)}")
        prompts = [p for p in prompts if p["name"] in only]
    return prompts


def fixture_tensors(cfg: Inkling2LoopConfig, res: dict) -> dict[str, torch.Tensor]:
    """The tensors of a fixture from `generate(...)`'s result. `core_out` keeps the passes whose partial block a later
    pass reads as its bank (all core passes but the last; the last one's output is `hidden` of the layer after the
    core)."""
    t = {k: res[k] for k in ("tokens", "logits", "h_final", "hidden")}
    t["core_out"] = res["core_out"][:cfg.loop_passes - 1].contiguous()
    t.update({k: res[k] for k in ("router_idx", "router_w", "router_margin")})
    return t


# fp64 sums in a different order (a 1-row product against an n-row product: BLAS gemv vs gemm) differ by ~1e-14 absolute,
# which is a small fraction of an fp32 ulp of a logit: now and then a logit lands on the other side of an fp32 rounding
# boundary and differs by exactly 1 ulp. Everything the model stores as bf16 absorbs it (oracle/README.md).
LOGIT_ULP_BOUND = 2


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in fp32 ulps of the larger magnitude (float32 tensors)."""
    big = torch.maximum(a.abs(), b.abs()).numpy()
    return (a.double() - b.double()).abs() / torch.from_numpy(np.spacing(big).astype(np.float64))


def compare_recompute(cached: dict, full, exact: bool = False) -> dict:
    """Cached prefill + decode (`generate`'s result) against the full recompute (`forward(..., record=True)`) of the same
    tokens. The bf16 planes (`hidden`, `core_out`, `h_final`) and the routed experts must be bit for bit equal. The fp32
    planes (logits, routing weights) must be within LOGIT_ULP_BOUND fp32 ulp (0 with `exact`), and every row's argmax and
    top-2 order equal. Returns the maxima that were measured and `ok`."""
    lc, lf = cached["logits"], full.logits
    d = (lc.double() - lf.double()).abs()
    rms = lf.double().pow(2).mean(-1).sqrt()
    top2 = lambda x: torch.topk(x, 2, dim=-1).indices  # noqa: E731
    out = {
        "rows": int(lf.shape[0]), "n_differing_logits": int((lc != lf).sum()), "logit_max_abs": float(d.max()),
        "logit_max_rel_rms": float((d.amax(-1) / rms).max()), "logit_max_ulps": float(_ulps(lc, lf).max()),
        "rows_with_differing_logits": torch.nonzero((lc != lf).any(-1)).flatten().tolist(),
        "argmax_equal": bool(torch.equal(lc.argmax(-1), lf.argmax(-1))),
        "top2_order_equal": bool(torch.equal(top2(lc), top2(lf))),
        "hidden_equal": bool(torch.equal(cached["hidden"], full.hidden)),
        "core_out_equal": bool(torch.equal(cached["core_out"], full.core_out)),
        "h_final_equal": bool(torch.equal(cached["h_final"], full.h_final)),
        "routes_equal": bool(torch.equal(cached["router_idx"], full.router_idx)),
        "router_w_max_ulps": float(_ulps(cached["router_w"], full.router_w).max()),
        "router_margin_max_abs": float((cached["router_margin"] - full.router_margin).abs().max()),
    }
    bound = 0 if exact else LOGIT_ULP_BOUND
    out["ok"] = bool(out["logit_max_ulps"] <= bound and out["router_w_max_ulps"] <= bound and out["argmax_equal"]
                     and out["top2_order_equal"] and out["hidden_equal"] and out["core_out_equal"]
                     and out["h_final_equal"] and out["routes_equal"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0, help="expert worker processes (0 = in process)")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = torch default)")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--verify-recompute", action="store_true",
                    help="also run every sequence uncached and compare with the cached run (doubles the cost): hidden "
                         "states, bank partials, h_final and routes bit for bit; logits and routing weights within "
                         f"{LOGIT_ULP_BOUND} fp32 ulp; argmax and top-2 order equal (README: cached decode vs recompute)")
    ap.add_argument("--verify-exact", action="store_true",
                    help="like --verify-recompute but the logits and routing weights must be bit for bit equal too (fails "
                         "about once per 10^7 logits: float64 summation order of the unembed)")
    ap.add_argument("--emulate", default="none",
                    help="engine numerics to reproduce: none (default), engine (all), or a comma list of "
                         + ", ".join(Emulation.NAMES))
    ap.add_argument("--weight-cache-gib", type=float, default=0.0,
                    help="keep this many GiB of decoded float64 EXL3 dense weights (attention, unembed slabs) in memory: "
                         "each core layer's weights are read by 3 passes of every forward, and the unembed by every step")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    prompts = load_prompts(a.prompts, a.only)
    tok = None
    if any("text" in p for p in prompts):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    ix = PackIndex(a.pack)
    experts = PackExperts(ix, workers=a.workers)
    em = Emulation.parse(a.emulate)
    model = Inkling2LoopRef(cfg, PackWeights(ix, int(a.weight_cache_gib * (1 << 30))), experts, emulate=em)
    os.makedirs(a.out, exist_ok=True)
    meta_common = {
        "pack_config_sha256": sha256_file(os.path.join(a.pack, "config.json")),
        "oracle_source_sha256": oracle_source_sha(),
        "numerics_version": NUMERICS_VERSION,
        "emulate": em.tag,
    }
    manifest_path = os.path.join(a.out, "manifest.json")
    manifest = {"meta": meta_common, "fixtures": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest["fixtures"] = json.load(f).get("fixtures", {})
    verify = a.verify_recompute or a.verify_exact
    failed: list[str] = []
    try:
        for p in prompts:
            ids = p["ids"] if "ids" in p else tok.encode(p["text"], add_special_tokens=False).ids
            n_new = int(p.get("new_tokens", a.new_tokens))
            t0 = time.time()
            res = generate(model, ids, n_new)
            secs = time.time() - t0
            path = os.path.join(a.out, f"{p['name']}.safetensors")
            st.write(path, fixture_tensors(cfg, res),
                     dict(meta_common, name=p["name"], prompt_len=str(res["prompt_len"])))
            gen = res["tokens"][res["prompt_len"]:].tolist()
            entry = {
                "sha256": sha256_file(path), "prompt_len": res["prompt_len"], "generated": gen,
                "generated_text": tok.decode(gen, skip_special_tokens=False) if tok else None,
                "seconds": round(secs, 1), "verified_recompute": False,
            }
            if verify:
                t1 = time.time()
                check = compare_recompute(res, model.forward(res["tokens"][:-1], record=True), exact=a.verify_exact)
                mode = "exact" if a.verify_exact else f"<= {LOGIT_ULP_BOUND} fp32 ulp"
                entry["recompute_check"] = dict(check, mode=mode, seconds=round(time.time() - t1, 1))
                entry["verified_recompute"] = check["ok"]
                if check["ok"]:
                    print(f"{p['name']}: cached == recompute ({mode}): {check['n_differing_logits']} differing logits, "
                          f"max {check['logit_max_ulps']} fp32 ulp, max {check['logit_max_abs']:.3g} abs", flush=True)
                else:
                    failed.append(p["name"])
                    print(f"{p['name']}: CACHED DECODE DIFFERS FROM THE FULL RECOMPUTE: {json.dumps(check)}", flush=True)
            manifest["fixtures"][p["name"]] = entry
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=1)
            print(f"{p['name']}: {res['prompt_len']} + {len(gen)} tokens in {secs:.0f}s -> {path}", flush=True)
    finally:
        experts.close()
    if failed:
        print(f"verification failed for: {', '.join(failed)} (their fixtures were written; the manifest says so)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
