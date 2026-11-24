"""B1 of docs/C4-DECODE-PLAN.md: the quality cost of adaptive expert skipping (`Emulation.expert_skip`), measured offline.

    python skip_study.py forward --pack PACK --fixture ENGINE_FIXTURE.safetensors --emulate engine,expert_skip=0.2,skip_renorm --out RUN.safetensors
    python skip_study.py control --run RUN.safetensors --fixture ENGINE_FIXTURE.safetensors --out control.json   (a --rows run against the fixture)
    python skip_study.py summary --engine DIR --spec DIR --runs DIR --census intake/census.json --prompts NAME... --tags TAG... --out report.json
    python skip_study.py tables --report report.json --out tables.md

`forward` is the oracle's whole forward (CPU, float64) over a fixture's own token sequence (prompt + the A lane's generated tokens, teacher
forced) with the given emulation, written in the fixture layout (`gen_fixtures.py`): row t of every plane is comparable with row t of
`fixtures/engine/<prompt>.safetensors` (the A lane = `Emulation.engine()`) and `fixtures/spec/` (the exact numerics). `forward` of
`--emulate engine` equals the engine fixture (planes of `fp8_emu.py forward --variant A`, record 2026-11-23-fp8-numerics).

`summary` per prompt and run (tag `tT-renorm` / `tT-plain`, the file `TAG-PROMPT.safetensors`), against the A lane's fixture:
  logits   KL(A || skip) per row of the softmax over the valid vocabulary (mean, p99, max; all rows, prompt rows, generated rows), argmax
           agreement and every flip with the A lane's top-1 minus top-2 margin and its gap to the pick; the floor: the same for the A lane
           against the exact fixture (KL(A || exact)).
  hidden   per visit 0..48 the relative L2 of the whole plane ||X - A|| / ||A|| (the `added` deviation) next to the floor's ||A - exact|| /
           ||exact|| at the same visit, and h_final.
  router   per (visit, row) the top-8 set differs from the A lane's (downstream flips; visit 0 reads the embedding only), next to the floor's.
  drops    picks and routing mass dropped by the rule (`skip_picks`) on the A lane's own routes and on the run's routes; the bytes of the dropped
           picks (intake/census.json) over the bytes of all picks of one row, and in windows of 8 consecutive rows (one stream) and four such
           streams by the union rule (an expert is read once if any row of the window keeps it).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import st  # noqa: E402
from inkling2_ref import F64, Emulation, ForwardOut, Inkling2LoopConfig, Inkling2LoopRef, Rounding, skip_picks  # noqa: E402

PLANES = ("tokens", "logits", "hidden", "h_final", "router_idx", "router_w")


# ----------------------------------------------------------------------------------------------------------------------
# forward


def read_meta(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)).get("__metadata__", {})


def read_fixture_tokens(path: str) -> tuple[list[int], int]:
    return st.read_tensor(st.read_header(path)["tokens"]).tolist(), int(read_meta(path)["prompt_len"])


def cmd_forward(a) -> int:
    from pack import PackExperts, PackIndex, PackWeights

    torch.set_num_threads(a.threads)
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = Inkling2LoopConfig.from_hf(json.load(f))
    ix = PackIndex(a.pack)
    tokens, plen = read_fixture_tokens(a.fixture)
    em = Emulation.parse(a.emulate)
    experts = PackExperts(ix, workers=a.workers)
    try:
        model = Inkling2LoopRef(cfg, PackWeights(ix, int(a.weight_cache_gib * (1 << 30))), experts, emulate=em)
        t0 = time.time()
        if a.rows:  # a control on the first rows: causal, so they equal the fixture's first rows
            tokens = tokens[:a.rows + 1]
        if a.decode_only:
            # the lever's real domain: the prompt is prefilled in the A lane (every expert read anyway), only the decode steps skip. Row plen - 1
            # (the prefill's last row) is the A lane's; the generated tokens are fed one by one to the skipping model on the same cache.
            exact = Inkling2LoopRef(cfg, model.w, experts, emulate=Emulation.engine())
            cache = exact.new_cache()
            outs = [exact.forward(tokens[:plen], cache, record=True)]
            outs += [model.forward([tok], cache, record=True) for tok in tokens[plen:-1]]
            cat = lambda f: torch.cat([getattr(o, f) for o in outs], dim=1 if f.startswith(("hidden", "router", "core")) else 0)  # noqa: E731
            out = ForwardOut(logits=cat("logits"), h_final=cat("h_final"), hidden=cat("hidden"), router_idx=cat("router_idx"),
                             router_w=cat("router_w"), router_margin=cat("router_margin"), core_out=cat("core_out"))
        else:
            out = model.forward(tokens[:-1], record=True)
        secs = time.time() - t0
    finally:
        experts.close()
    t = {"tokens": torch.tensor(tokens, dtype=torch.long), "logits": out.logits, "h_final": out.h_final, "hidden": out.hidden,
         "core_out": out.core_out[:cfg.loop_passes - 1].contiguous(), "router_idx": out.router_idx, "router_w": out.router_w,
         "router_margin": out.router_margin}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    st.write(a.out, t, {"name": os.path.basename(a.fixture), "prompt_len": str(plen), "emulate": em.tag, "seconds": f"{secs:.1f}",
                        "skip_rows": "decode" if a.decode_only else "all",
                        "numerics_version": "skip-study"})
    print(f"{em.tag}: {len(tokens) - 1} rows in {secs:.0f}s -> {a.out}", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# summary


def load(path: str) -> dict[str, torch.Tensor]:
    t = st.load_all(path)
    return {k: t[k] for k in PLANES}


def rel(x: torch.Tensor, r: torch.Tensor) -> float:
    x, r = x.to(F64), r.to(F64)
    return float((x - r).norm() / r.norm())


def kl_rows(ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    lp, lq = torch.log_softmax(ref.to(F64), -1), torch.log_softmax(x.to(F64), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def stats(v: torch.Tensor) -> dict:
    if v.numel() == 0:
        return {"rows": 0}
    return {"rows": int(v.numel()), "mean": float(v.mean()), "p99": float(torch.quantile(v, 0.99)), "max": float(v.max())}


def logits_report(a: dict, x: dict, plen: int, start: int = 0) -> dict:
    """A lane `a` against another plane set `x` (skip run or exact fixture), rows from `start` on: KL(A || x) per row and the argmax flips
    (`row` is the absolute row)."""
    la, lx = a["logits"][start:], x["logits"][start:]
    kl = kl_rows(la, lx)
    gen0 = max(plen - 1 - start, 0)  # row t predicts token t + 1: the generated tokens are predicted from row plen - 1
    top2 = torch.topk(la.to(F64), 2, dim=-1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    pick = lx.argmax(-1)
    flips = []
    for r in (la.argmax(-1) != pick).nonzero().flatten().tolist():
        flips.append({"row": r + start, "generated": r >= gen0, "a_margin": float(margin[r]),
                      "a_gap_to_pick": float(la[r].to(F64).max() - la[r, pick[r]].to(F64))})
    return {"rows": int(la.shape[0]), "kl": stats(kl), "kl_prompt": stats(kl[:gen0]), "kl_generated": stats(kl[gen0:]),
            "argmax_equal": int((la.argmax(-1) == pick).sum()), "flips": flips, "kl_rows": [float(v) for v in kl]}


def hidden_curve(x: torch.Tensor, r: torch.Tensor, start: int = 0) -> list[float]:
    return [rel(x[v, start:], r[v, start:]) for v in range(x.shape[0])]


def route_flips(x: dict, r: dict, start: int = 0) -> dict:
    """Per visit: rows (from `start`) whose top-8 set differs from the reference's."""
    sx, sr = x["router_idx"][:, start:].sort(-1).values, r["router_idx"][:, start:].sort(-1).values
    diff = (sx != sr).any(-1)  # [V, T]
    return {"per_visit": [int(v) for v in diff.sum(1)], "total": int(diff.sum()), "pairs": int(diff.numel()),
            "after_visit0": int(diff[1:].sum()), "pairs_after_visit0": int(diff[1:].numel())}


def layer_of(p: int) -> int:
    """The layer of pass p (the core layers 6-17 run three times: passes 6-41)."""
    return p if p < 6 else (6 + (p - 6) % 12 if p < 42 else 18 + (p - 42))


def expert_bytes(census: str) -> dict[tuple[int, int], int]:
    with open(census) as f:
        mods = json.load(f)["modules"]
    eb: dict[tuple[int, int], int] = {}
    for n, v in mods.items():
        m = re.match(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$", n)
        if m:
            eb[(int(m[1]), int(m[2]))] = eb.get((int(m[1]), int(m[2])), 0) + v["bytes"]
    return eb


def drops(planes: dict, t: float, eb: dict) -> dict:
    """Picks, routing mass and expert bytes the rule drops on these planes' routes (one row: its bytes are not shared)."""
    idx, w = planes["router_idx"], planes["router_w"].to(F64)
    keep = torch.stack([skip_picks(w[p], t, False, Rounding())[1] for p in range(w.shape[0])])
    bytes_all = bytes_dropped = 0
    for p in range(idx.shape[0]):
        lay = layer_of(p)
        b = torch.tensor([[eb[(lay, int(e))] for e in row] for row in idx[p].tolist()], dtype=torch.long)
        bytes_all += int(b.sum())
        bytes_dropped += int(b[~keep[p]].sum())
    return {"picks": int(keep.numel()), "picks_dropped": int((~keep).sum()), "mass": float(w.sum()), "mass_dropped": float(w[~keep].sum()),
            "bytes": bytes_all, "bytes_dropped": bytes_dropped}


def window_saved(planes_by_prompt: dict, t: float, eb: dict, rows: int, streams: int, n_windows: int = 120, seed: int = 1) -> dict:
    """Bytes read by the union rule (an expert is read once if any row of the window picks it) without and with skipping, over `n_windows`
    windows of `streams` streams x `rows` consecutive rows; every stream starts at a random row of a random prompt (seed `seed`)."""
    rnd = random.Random(seed)
    names = sorted(planes_by_prompt)
    total = kept_b = 0
    for _ in range(n_windows):
        spans = []
        for _s in range(streams):
            n = rnd.choice(names)
            nrows = planes_by_prompt[n]["router_idx"].shape[1]
            spans.append((n, rnd.randrange(0, nrows - rows + 1)))
        for p in range(48):
            lay = layer_of(p)
            allx, keep = set(), set()
            for n, s0 in spans:
                idx = planes_by_prompt[n]["router_idx"][p, s0:s0 + rows]
                w = planes_by_prompt[n]["router_w"][p, s0:s0 + rows].to(F64)
                kp = skip_picks(w, t, False, Rounding())[1]
                for e, k in zip(idx.flatten().tolist(), kp.flatten().tolist()):
                    allx.add(e)
                    if k:
                        keep.add(e)
            total += sum(eb[(lay, e)] for e in allx)
            kept_b += sum(eb[(lay, e)] for e in keep)
    return {"windows": n_windows, "bytes": total, "bytes_kept": kept_b, "saved": 1 - kept_b / total}


def cmd_control(a) -> int:
    """A run of the first rows against the same rows of the engine fixture: bit-equal planes mean the harness changes nothing but the option."""
    r, f = load(a.run), load(a.fixture)
    n = r["logits"].shape[0]
    out = {"rows": n, "emulate": read_meta(a.run)["emulate"]}
    for k in ("hidden", "h_final", "router_idx", "router_w"):
        ref = f[k][:, :n] if k in ("hidden", "router_idx", "router_w") else f[k][:n]
        out[k + "_equal"] = bool(torch.equal(r[k], ref))
    ref = f["logits"][:n]
    d = (r["logits"].to(F64) - ref.to(F64)).abs()
    out["logits_equal"] = bool(torch.equal(r["logits"], ref))
    out["logits_differing"] = int((d > 0).sum())
    out["logits_max_abs"] = float(d.max())
    out["kl_max"] = float(kl_rows(ref, r["logits"]).max())
    print(json.dumps(out, indent=1))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    return 0 if all(v for k, v in out.items() if k.endswith("_equal") and k != "logits_equal") else 1


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def render(report: dict) -> str:
    """The record's tables from the report (markdown)."""
    P, S = report["prompts"], report["settings"]
    out: list[str] = []
    short = lambda p: p.split("-effort")[0]  # noqa: E731
    # 1. what the rule drops (the A lane's own routes; independent of renorm)
    out += ["### What the rule drops", "",
            "On the A lane's routes of the prompts below (all 48 passes, every row); bytes from `intake/census.json`; windows: 120 random windows "
            "(seed 1) of 8 consecutive rows (1 x 8) and of four such streams (4 x 8), every stream starting at a random row of a random prompt, "
            "union rule.", "",
            "| t | prompts | picks dropped | routing mass dropped | bytes dropped, one row | window bytes saved 1 x 8 | window bytes saved 4 x 8 |",
            "|--:|---|--:|--:|--:|--:|--:|"]
    seen = set()
    for tag, s in S.items():
        if s["t"] in seen or not s["done"]:
            continue
        seen.add(s["t"])
        a = [s["prompts"][p]["a_routes"] for p in s["done"]]
        out.append(f"| {s['t']:g} | {', '.join(short(p) for p in s['done'])} | {pct(sum(x['picks_dropped'] for x in a) / sum(x['picks'] for x in a))} | "
                   f"{pct(sum(x['mass_dropped'] for x in a) / sum(x['mass'] for x in a))} | "
                   f"{pct(sum(x['bytes_dropped'] for x in a) / sum(x['bytes'] for x in a))} | {pct(s['window_saved_1x8']['saved'])} | "
                   f"{pct(s['window_saved_4x8']['saved'])} |")
    # 2. quality
    out += ["", "### Quality against the A lane", "",
            "KL = KL(A || run) of the softmax over the 200,058 valid logits, per row, in nats (mean / p99 / max over the rows). `floor` = the A lane against "
            "the exact fixture on the same rows (what its own numerics already cost). Hidden = relative L2 of the whole plane per visit, mean over visits 1..48 "
            "(visit 0 is the embedding, equal), and its maximum; flips = (visit, row) pairs after visit 0 whose top-8 set differs from the A lane's.", "",
            "| setting | prompt | rows | KL mean | KL p99 | KL max | top-1 equal | generated-row flips | hidden mean (max) | h_final | router flips |",
            "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for p in P:
        v = P[p]
        n = v["rows"]
        k = v["floor_logits"]["kl"]
        fh = v["floor_hidden"]
        gen_fl = sum(1 for f in v["floor_logits"]["flips"] if f["generated"])
        fr = v["floor_router"]
        out.append(f"| floor | {short(p)} | {n} | {k['mean']:.4f} | {k['p99']:.4f} | {k['max']:.4f} | {v['floor_logits']['argmax_equal']}/{n} | {gen_fl} | "
                   f"{sum(fh[1:]) / 48:.3f} ({max(fh):.3f}) | {v['floor_h_final']:.3f} | {pct(fr['after_visit0'] / fr['pairs_after_visit0'])} |")
    for tag, s in S.items():
        for p in s["done"]:
            v = s["prompts"][p]
            n = v["logits"]["rows"]
            k = v["logits"]["kl"]
            h = v["added_hidden"]
            gen_fl = sum(1 for f in v["logits"]["flips"] if f["generated"])
            r = v["router"]
            out.append(f"| {tag} | {short(p)} | {n} | {k['mean']:.4f} | {k['p99']:.4f} | {k['max']:.4f} | {v['logits']['argmax_equal']}/{n} | {gen_fl} | "
                       f"{sum(h[1:]) / 48:.3f} ({max(h):.3f}) | {v['h_final']:.3f} | {pct(r['after_visit0'] / r['pairs_after_visit0'])} |")
    out += ["", "Pooled over the prompts a setting ran on (rows weighted equally):", "",
            "| setting | prompts | rows | KL mean | KL p99 | KL max | KL mean, generated rows | floor KL mean (same rows) | KL mean / floor | top-1 equal |",
            "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for tag, s in S.items():
        if not s["done"]:
            continue
        fl = torch.cat([torch.tensor(P[p]["floor_logits"]["kl_rows"][s["start"][p]:]) for p in s["done"]])
        k, kg = s["pooled_kl"], s["pooled_kl_generated"]
        out.append(f"| {tag} | {', '.join(short(p) for p in s['done'])} | {k['rows']} | {k['mean']:.4f} | {k['p99']:.4f} | {k['max']:.4f} | {kg['mean']:.4f} | "
                   f"{float(fl.mean()):.4f} | {k['mean'] / float(fl.mean()):.0f} x | {s['pooled_argmax_equal'][0]}/{s['pooled_argmax_equal'][1]} |")
    # 3. argmax flips
    out += ["", "### Argmax flips against the A lane", "",
            "Row t predicts token t + 1; `generated` rows are the ones whose argmax is a token the A lane greedily produced. Margin = the A lane's top-1 minus top-2 "
            "logit at the row; gap = the A lane's top-1 minus its logit at the run's pick.", "",
            "| setting | prompt | flips | rows (generated?: margin / gap) |", "|---|---|--:|---|"]
    for tag, s in S.items():
        for p in s["done"]:
            fl = s["prompts"][p]["logits"]["flips"]
            out.append(f"| {tag} | {short(p)} | {len(fl)} | " + "; ".join(
                f"{f['row']}{'g' if f['generated'] else ''}: {f['a_margin']:.2f} / {f['a_gap_to_pick']:.2f}" for f in fl) + " |")
    for p in P:
        fl = P[p]["floor_logits"]["flips"]
        out.append(f"| floor | {short(p)} | {len(fl)} | " + "; ".join(f"{f['row']}{'g' if f['generated'] else ''}: {f['a_margin']:.2f} / {f['a_gap_to_pick']:.2f}" for f in fl) + " |")
    # 4. hidden per visit
    out += ["", "### Hidden plane per visit (relative L2 against the A lane's plane; floor = A lane against exact)", ""]
    vis = [1, 2, 3, 6, 12, 24, 36, 42, 48]
    for p in P:
        out += [f"**{short(p)}**", "", "| setting | " + " | ".join(f"v{v}" for v in vis) + " |", "|---|" + "--:|" * len(vis)]
        out.append("| floor | " + " | ".join(f"{P[p]['floor_hidden'][v]:.4f}" for v in vis) + " |")
        for tag, s in S.items():
            if p in s["done"]:
                out.append(f"| {tag} | " + " | ".join(f"{s['prompts'][p]['added_hidden'][v]:.4f}" for v in vis) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def cmd_tables(a) -> int:
    with open(a.report) as f:
        text = render(json.load(f))
    with open(a.out, "w") as f:
        f.write(text)
    print(text)
    return 0


def parse_tag(tag: str) -> tuple[float, bool, bool]:
    """tT-renorm / tT-plain: skipping on every row; dT-renorm / dT-plain: only in the decode steps (prompt prefilled in the A lane).
    Returns (t, renorm, decode_only)."""
    m = re.fullmatch(r"([td])([0-9.]+)-(renorm|plain)", tag)
    if not m:
        raise ValueError(f"tag {tag!r} is not tT-renorm / tT-plain / dT-renorm / dT-plain")
    return float(m[2]), m[3] == "renorm", m[1] == "d"


def cmd_summary(a) -> int:
    eb = expert_bytes(a.census)
    report: dict = {"prompts": {}, "settings": {}}
    A, SPEC, plen = {}, {}, {}
    for p in a.prompts:
        A[p] = load(os.path.join(a.engine, p + ".safetensors"))
        SPEC[p] = load(os.path.join(a.spec, p + ".safetensors"))
        plen[p] = int(read_meta(os.path.join(a.engine, p + ".safetensors"))["prompt_len"])
        assert torch.equal(A[p]["tokens"], SPEC[p]["tokens"])
        report["prompts"][p] = {
            "rows": int(A[p]["logits"].shape[0]), "prompt_len": plen[p],
            "floor_hidden": hidden_curve(A[p]["hidden"], SPEC[p]["hidden"]), "floor_h_final": rel(A[p]["h_final"], SPEC[p]["h_final"]),
            "floor_logits": logits_report(A[p], SPEC[p], plen[p]), "floor_router": route_flips(A[p], SPEC[p]),
        }
    for tag in a.tags:
        t, renorm, decode_only = parse_tag(tag)
        s: dict = {"t": t, "renorm": renorm, "decode_only": decode_only, "prompts": {}, "start": {}}
        done = [p for p in a.prompts if os.path.exists(os.path.join(a.runs, f"{tag}-{p}.safetensors"))]
        for p in done:
            run = load(os.path.join(a.runs, f"{tag}-{p}.safetensors"))
            assert torch.equal(run["tokens"], A[p]["tokens"])
            assert torch.equal(run["router_idx"][0], A[p]["router_idx"][0]), "visit 0 routes must be the A lane's (the embedding only)"
            st0 = plen[p] if decode_only else 0  # decode-only runs: the prefill rows (< plen) are the A lane's own, the skip rows start at plen
            s["start"][p] = st0
            cut = lambda d: {k: (v[:, st0:] if k.startswith(("hidden", "router")) else v[st0:]) for k, v in d.items() if k != "tokens"}  # noqa: E731
            if decode_only:
                for k in ("logits", "h_final"):
                    assert torch.equal(run[k][:st0], A[p][k][:st0]), f"{tag} {p}: prefill rows of {k} are not the A lane's"
                for k in ("hidden", "router_idx", "router_w"):
                    assert torch.equal(run[k][:, :st0], A[p][k][:, :st0]), f"{tag} {p}: prefill rows of {k} are not the A lane's"
            s["prompts"][p] = {
                "logits": logits_report(A[p], run, plen[p], st0), "added_hidden": hidden_curve(run["hidden"], A[p]["hidden"], st0),
                "h_final": rel(run["h_final"][st0:], A[p]["h_final"][st0:]), "router": route_flips(run, A[p], st0),
                "run_routes": drops(cut(run), t, eb), "a_routes": drops(cut(A[p]), t, eb),
            }
        s["done"] = done
        if done:
            aw = {p: {k: A[p][k][:, plen[p] if decode_only else 0:] for k in ("router_idx", "router_w")} for p in a.prompts}  # windows of decode rows
            s["window_saved_1x8"] = window_saved(aw, t, eb, 8, 1)
            s["window_saved_4x8"] = window_saved(aw, t, eb, 8, 4)
            s["pooled_kl"] = stats(torch.cat([torch.tensor(s["prompts"][p]["logits"]["kl_rows"]) for p in done]))
            s["pooled_kl_generated"] = stats(torch.cat([torch.tensor(s["prompts"][p]["logits"]["kl_rows"][max(plen[p] - 1 - s["start"][p], 0):])
                                                        for p in done]))
            s["pooled_argmax_equal"] = [sum(s["prompts"][p]["logits"]["argmax_equal"] for p in done),
                                        sum(s["prompts"][p]["logits"]["rows"] for p in done)]
        report["settings"][tag] = s
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: {"done": v["done"], "pooled_kl": v.get("pooled_kl"), "argmax": v.get("pooled_argmax_equal")}
                      for k, v in report["settings"].items()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forward")
    f.add_argument("--pack", required=True)
    f.add_argument("--fixture", required=True, help="an engine fixture: its tokens (and prompt_len) are the input")
    f.add_argument("--emulate", required=True, help="e.g. engine,expert_skip=0.2,skip_renorm")
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=3)
    f.add_argument("--threads", type=int, default=4)
    f.add_argument("--weight-cache-gib", type=float, default=6.0)
    f.add_argument("--decode-only", action="store_true", help="prefill the prompt in the A lane, skip only in the decode steps")
    f.add_argument("--rows", type=int, default=0, help="only the first ROWS rows of the fixture (controls)")
    s = sub.add_parser("summary")
    s.add_argument("--engine", required=True)
    s.add_argument("--spec", required=True)
    s.add_argument("--runs", required=True)
    s.add_argument("--census", required=True)
    s.add_argument("--prompts", nargs="+", required=True)
    s.add_argument("--tags", nargs="+", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("control")
    c.add_argument("--run", required=True)
    c.add_argument("--fixture", required=True)
    c.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--report", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"tables": cmd_tables, "forward": cmd_forward, "summary": cmd_summary, "control": cmd_control}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
