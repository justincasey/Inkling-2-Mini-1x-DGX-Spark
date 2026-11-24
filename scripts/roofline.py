#!/usr/bin/env python3
"""Roofline for Inkling-2-Mini-SAGE-REXL3-4.23bpw on ONE GB10 Spark, from the pack census and config.

Hardware numbers are carried from measured records (docs/ROOFLINE.md, "Hardware"); no engine number is read or
written here. Everything else is arithmetic on `intake/census.json` and the pack's `config.json`:

  decode   a step reads, per layer pass, the attention + router + loop-controller bytes once and the routed experts
           that the step's rows pick (8 of 128 per row; several rows = the union of their picks, independent-picks
           model); the middle layers run `num_passes` times with shared weights and are read every pass.
  prefill  FLOPs per prompt token: 2 x the weights a token multiplies (all passes) + global attention
           (4 * heads * head_dim * L/2 per global pass, averaged over a prompt of L) + sliding attention.

  python3 scripts/roofline.py [--census intake/census.json] [--config intake/files/config.json] [--json]
"""
import argparse
import io
import json
import math
import re
import sys

GIB = 1 << 30

# Measured on one GB10 (provenance: docs/ROOFLINE.md). GB/s sustained read, TFLOP/s dense cuBLASLt 8192^3.
HW = {
    "bw_gbs": 233.8,         # tools/roofline.cu streaming read, 8 GiB (GLM-5.3-EXL3 recipe, a GB10 node, 2026-11-04)
    "bw_rexl3_gbs": 237.0,   # the byte floor REXL3's EXL3 kernels are profiled against (CYBERFROST exec plan)
    "bf16_tflops": 93.4,
    "fp16_tflops": 91.9,
    "fp8_tflops": 195.1,
    "mma_peak_tflops": 107.4,  # register-only mma probe (CYBERFROST exec plan)
}
NODE_MEM_GIB = 121.69        # MemTotal of a Spark (Inkling-2-Small FIT.md)
OS_UNACCOUNTED_GIB = 11.0    # OS + CUDA context (Inkling-2-Small FIT.md)
MEM_FLOOR_GIB = 10.0         # MemAvailable floor kept under load (Inkling-2-Small FIT.md)
ACTIVATION_GIB = 5.0         # engine activation/graph reserve (CYBERFROST FIT.md)
IDLE_MS = 0.3                # GPU idle per round (CYBERFROST exec plan: 0.25-0.35 ms)
# distinct experts per layer / independent-picks model, measured on GLM-5.3-Flash (256 experts, top 8): k=2 13.8/15.8,
# 4 23.7/30.5, 6 32.2/44.4, 8 39.7/57.4 (GLM-5.3-EXL3 recipe docs/ROOFLINE.md); 3 and 5 interpolated. NOT measured on Inkling.
GLM_OVERLAP = {1: 1.0, 2: 0.873, 3: 0.82, 4: 0.777, 5: 0.75, 6: 0.725, 8: 0.692}
# CYBERFROST measured per-position conditional MTP acceptance, positions 1-6 (exec plan); position 7 repeats position 6.
ACCEPT_CF = [0.736, 0.77, 0.82, 0.89, 0.86, 0.79, 0.79]
ACCEPT_FLAT = 0.69          # Inkling-2-Small DSpark: flat-p fit of T=3.04 at depth 7 (run 20 record)

LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")


def tokens_per_round(accept, d):
    """1 + sum over drafted positions of the running acceptance product."""
    t, p = 1.0, 1.0
    for i in range(d):
        p *= accept[i] if isinstance(accept, list) else accept
        t += p
    return t


def numel(shape):
    return math.prod(shape) if shape else 1


def mod_weights(mod):
    """Multiplied weights of a module (EXL3 trellis -> in*out; plain matrices -> elements)."""
    t = mod["tensors"]
    if "trellis" in t:
        a, b, _ = t["trellis"]["shape"]
        return a * 16 * b * 16
    for key in ("weight", "tensor"):
        if key in t and len(t[key]["shape"]) == 2:
            return numel(t[key]["shape"])
    return 0


def load(census_path, config_path):
    census = json.load(open(census_path))
    cfg = json.load(open(config_path))
    return census, cfg["text_config"], cfg.get("mtp_config", {})


def tally(census, tc):
    n = tc["num_hidden_layers"]
    lay = [dict(attn=0, attn_w=0, router=0, router_w=0, ctrl=0, ctrl_w=0, exp=0, exp_w=0, other=0) for _ in range(n)]
    g = dict(embed=0, unembed=0, unembed_w=0, glue=0, visual=0, other=0)
    for name, mod in census["modules"].items():
        size, w = mod["bytes"], mod_weights(mod)
        m = LAYER.match(name)
        if m:
            l, rest = int(m.group(1)), m.group(2)
            d = lay[l]
            if rest.startswith("mlp.experts."):
                d["exp"] += size
                d["exp_w"] += w
            elif rest.startswith("mlp.gate"):
                d["router"] += size
                d["router_w"] += w
            elif rest.startswith("controllers."):
                d["ctrl"] += size
                d["ctrl_w"] += w
            elif rest.startswith(("attn.", "attn_sconv", "mlp_sconv")):
                d["attn"] += size
                d["attn_w"] += w
            else:
                d["other"] += size
        elif name.startswith("model.llm.embed"):
            g["embed"] += size
        elif name.startswith("model.llm.unembed"):
            g["unembed"] += size
            g["unembed_w"] += w
        elif name.startswith("model.llm."):
            g["glue"] += size
        elif name.startswith("model.visual"):
            g["visual"] += size
        else:
            g["other"] += size
    return lay, g


class Model:
    def __init__(self, census, tc, mtp):
        self.tc = tc
        self.lay, self.g = tally(census, tc)
        self.n = tc["num_hidden_layers"]
        self.E, self.k = tc["n_routed_experts"], tc["num_experts_per_tok"]
        loop = tc["loop_config"]
        self.mult = [loop["num_passes"] if loop["core_start"] <= l < loop["core_end"] else 1 for l in range(self.n)]
        self.passes = sum(self.mult)
        local = set(tc["local_layer_ids"])
        self.glob_passes = sum(self.mult[l] for l in range(self.n) if l not in local)
        self.local_passes = self.passes - self.glob_passes
        self.window = tc["sliding_window_size"]
        self.kvh, self.hd = tc["num_key_value_heads"], tc["head_dim"]
        self.heads = tc["num_attention_heads"]
        self.mtp = mtp
        self.total_bytes = census["total_bytes"]

    # ---- decode ----
    def distinct(self, rows):
        return self.E * (1 - (1 - self.k / self.E) ** rows)

    def weight_bytes(self, rows=1, overlap=1.0):
        """Bytes one decode step of `rows` rows reads: expert union per pass, everything else once per pass.
        `overlap` scales the expert union (1.0 = independent picks)."""
        dist = self.distinct(rows) * overlap
        t = {"experts": 0.0, "attention": 0.0, "router": 0.0, "controllers": 0.0, "layer other": 0.0}
        for l, d in enumerate(self.lay):
            m = self.mult[l]
            t["experts"] += m * dist * d["exp"] / self.E
            t["attention"] += m * d["attn"]
            t["router"] += m * d["router"]
            t["controllers"] += m * d["ctrl"]
            t["layer other"] += m * d["other"]
        t["unembed"] = self.g["unembed"]
        t["loop glue + attn-res"] = self.g["glue"]
        t["embed rows"] = rows * self.tc["hidden_size"] * 2
        return t

    def kv_pass_bytes(self, kv):
        """Bytes per token per pass, K and V, all KV heads."""
        per = {"bf16": 2 * self.kvh * self.hd * 2, "int8": 2 * self.kvh * (self.hd + 1), "q4": 1088}
        return per[kv]

    def kv_token_bytes(self, kv, drafter=True):
        b = self.glob_passes * self.kv_pass_bytes(kv)
        if drafter:  # the drafter's one global layer keeps a plane on the same page table
            b += self.kv_pass_bytes(kv)
        return b

    def kv_read_bytes(self, ctx, kv):
        """KV one step reads for one stream at context `ctx`: global passes whole, sliding passes the ring (bf16)."""
        glob = self.glob_passes * ctx * self.kv_pass_bytes(kv)
        ring = self.local_passes * min(ctx, self.window + 8) * self.kv_pass_bytes("bf16")
        return glob + ring

    def step_bytes(self, rows=1, ctx=0, kv="int8", streams=1):
        return sum(self.weight_bytes(rows).values()) + streams * self.kv_read_bytes(ctx, kv)

    # ---- prefill ----
    def linear_flops(self):
        f = 0.0
        for l, d in enumerate(self.lay):
            per_pass = d["attn_w"] + d["router_w"] + d["ctrl_w"] + self.k * d["exp_w"] / self.E
            f += self.mult[l] * 2 * per_pass
        return f

    def attn_flops(self, L):
        hd_all = self.heads * self.hd
        glob = self.glob_passes * 4 * hd_all * (L / 2)
        loc = self.local_passes * 4 * self.tc.get("swa_num_attention_heads", self.heads) * self.tc.get("swa_head_dim", self.hd) * min(L / 2, self.window)
        return glob, loc

    def prefill_flops(self, L):
        glob, loc = self.attn_flops(L)
        return self.linear_flops() + glob + loc


def fmt_gb(b):
    return f"{b / 1e9:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", default="intake/census.json")
    ap.add_argument("--config", default="intake/files/config.json")
    ap.add_argument("--manifest", default="intake/manifest.tsv")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    real_stdout = sys.stdout
    if args.json:
        sys.stdout = io.StringIO()
    census, tc, mtp = load(args.census, args.config)
    mdl = Model(census, tc, mtp)
    bw = HW["bw_rexl3_gbs"] * 1e9
    out = {"hw": HW}

    # draft sidecar bytes come from the manifest (the census covers the main shards only)
    draft = 0
    for line in open(args.manifest):
        sha, size, path = line.rstrip("\n").split("\t")
        if path == "mtp.safetensors":
            draft = int(size)
    resident = mdl.total_bytes + draft

    print("## Pack bytes (census, main shards)\n")
    cls = {"routed experts": sum(d["exp"] for d in mdl.lay), "attention (EXL3 6-bit + small tensors)": sum(d["attn"] for d in mdl.lay),
           "routers": sum(d["router"] for d in mdl.lay), "loop controllers (BF16)": sum(d["ctrl"] for d in mdl.lay),
           "embedding (BF16)": mdl.g["embed"], "unembed (EXL3 6-bit)": mdl.g["unembed"], "vision tower": mdl.g["visual"],
           "loop glue + attn-res": mdl.g["glue"], "layer norms etc.": sum(d["other"] for d in mdl.lay) + mdl.g["other"],
           "drafter sidecar (mtp.safetensors file)": draft}
    print("| class | GiB |\n|---|---:|")
    for k, v in cls.items():
        print(f"| {k} | {v / GIB:.3f} |")
    print(f"| **resident total** | **{resident / GIB:.3f}** |\n")
    out["pack_gib"] = {k: v / GIB for k, v in cls.items()}

    print(f"## Structure: {mdl.n} layers, {mdl.passes} passes per token ({mdl.glob_passes} global, {mdl.local_passes} sliding)\n")

    print("## Decode: bytes one row reads (uniform routing)\n")
    one = mdl.weight_bytes(1)
    print("| item | GB |\n|---|---:|")
    for k, v in one.items():
        print(f"| {k} | {fmt_gb(v)} |")
    tot1 = sum(one.values())
    print(f"| **weights, one row** | **{fmt_gb(tot1)}** |\n")
    out["decode_row_bytes"] = tot1
    exp_pp = [d["exp"] / mdl.E for d in mdl.lay]
    print(f"Mean expert: {min(exp_pp) / 1e6:.2f}-{max(exp_pp) / 1e6:.2f} MB by layer (5-bit tensors sit in the looped layers). "
          f"Serial ceiling at {HW['bw_rexl3_gbs']:.0f} GB/s: **{bw / tot1:.1f} tok/s** ({tot1 / bw * 1e3:.2f} ms); at {HW['bw_gbs']:.1f} GB/s: {HW['bw_gbs'] * 1e9 / tot1:.1f}; "
          f"at 258 GB/s (a plain stream, Inkling-2-Small plan): {258e9 / tot1:.1f}.\n")

    print("### Serial decode vs context (ceiling at 237 GB/s, KV read added)\n")
    print("| context | KV bf16 | KV int8 | KV q4 |\n|---:|---:|---:|---:|")
    ctxs = [4096, 32768, 131072, 262144, 524288, 1048576]
    ser = {}
    for c in ctxs:
        row = []
        for kv in ("bf16", "int8", "q4"):
            b = mdl.step_bytes(1, c, kv)
            row.append(bw / b)
            ser[f"{c}_{kv}"] = bw / b
        print(f"| {c // 1024}K | " + " | ".join(f"{x:.1f}" for x in row) + " |")
    out["serial_vs_ctx"] = ser
    print()

    print("### Verify window / batched decode: rows share one weight read\n")
    print("| rows | experts/layer | GB | ms @237 | rows/s ceiling | x serial bytes |\n|---:|---:|---:|---:|---:|---:|")
    win = {}
    for r in (1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 32):
        b = sum(mdl.weight_bytes(r).values())
        win[r] = b
        print(f"| {r} | {mdl.distinct(r):.1f} | {fmt_gb(b)} | {b / bw * 1e3:.1f} | {r / (b / bw):.0f} | {b / tot1:.2f} |")
    out["window_bytes"] = win
    print()

    # drafter cost per round: weights once + the Markov head's folded unembed table once per drafted step (inference
    # from Inkling-2-Small docs/PORT-SPEC.md 4.4: bias_i = latent_i . fold_W^T over [V,256], 7 sequential steps)
    markov_tbl = tc["vocab_size"] * mtp.get("markov_rank", 256) * 2
    steps = mtp.get("block_size", 8) - 1
    draft_hi = draft + steps * markov_tbl
    print(f"Drafter per round: weights {fmt_gb(draft)} GB once = {draft / bw * 1e3:.2f} ms; if the Markov head re-reads a "
          f"[{tc['vocab_size']},{mtp.get('markov_rank', 256)}] bf16 table per step ({steps} steps, {fmt_gb(markov_tbl)} GB each) "
          f"the round pays {fmt_gb(draft_hi)} GB = {draft_hi / bw * 1e3:.2f} ms [INFERENCE].\n")
    out["draft_ms_low_high"] = [draft / bw * 1e3, draft_hi / bw * 1e3]

    print("### MTP: tokens per second = T / round, round = window + drafter, engine factor f = 1.08 on bytes\n")
    print("| rows | round ms | tok/s at T=2.0 / 2.5 / 3.0 / 3.5 | T for 75 tok/s | T for 100 tok/s |\n|---:|---:|---|---:|---:|")
    spec = {}
    for r in (2, 3, 4, 5, 6, 8):
        wb = win.get(r) or sum(mdl.weight_bytes(r).values())
        rnd = 1.08 * (wb + draft_hi) / bw + IDLE_MS * 1e-3
        ts = [t / rnd for t in (2.0, 2.5, 3.0, 3.5)]
        spec[r] = {"round_ms": rnd * 1e3, "tok_s": ts, "T_for_75": 75 * rnd, "T_for_100": 100 * rnd}
        print(f"| {r} | {rnd * 1e3:.1f} | " + " / ".join(f"{x:.0f}" for x in ts) + f" | {75 * rnd:.1f} | {100 * rnd:.1f} |")
    out["spec"] = spec
    print()

    print("### MTP scenarios (T from an acceptance profile; the profile and the expert overlap are NOT measured on this model)\n")
    print(f"- conservative: independent expert picks, drafter at its high bound, f=1.08, {IDLE_MS} ms idle per round, flat per-position acceptance {ACCEPT_FLAT} (Inkling-2-Small DSpark fit, T=3.04 at depth 7)")
    print("- sibling-calibrated: expert-union factors measured on GLM-5.3-Flash (distinct/independent), drafter at its low bound, f=1.08, same idle, CYBERFROST's measured per-position acceptance")
    print("- ceiling: as sibling-calibrated with f=1.00 and no idle\n")
    print("| rows | T flat / CF | conservative | sibling-calibrated | ceiling |\n|---:|---|---:|---:|---:|")
    scen = {}
    for r in (2, 3, 4, 5, 6, 8):
        d = r - 1
        t_flat, t_cf = tokens_per_round(ACCEPT_FLAT, d), tokens_per_round(ACCEPT_CF, d)
        o = GLM_OVERLAP[r]
        cons = t_flat / (1.08 * (sum(mdl.weight_bytes(r).values()) + draft_hi) / bw + IDLE_MS * 1e-3)
        wb_o = sum(mdl.weight_bytes(r, overlap=o).values())
        sib = t_cf / (1.08 * (wb_o + draft) / bw + IDLE_MS * 1e-3)
        ceil = t_cf / ((wb_o + draft) / bw)
        scen[r] = {"T_flat": t_flat, "T_cf": t_cf, "conservative": cons, "sibling": sib, "ceiling": ceil}
        print(f"| {r} | {t_flat:.2f} / {t_cf:.2f} | {cons:.0f} | {sib:.0f} | {ceil:.0f} |")
    out["spec_scenarios"] = scen
    print(f"\nSerial for comparison: {bw / tot1:.1f} (f=1.00), {bw / (1.08 * tot1):.1f} (f=1.08), {258e9 / tot1:.1f} at 258 GB/s.\n")

    print("### MTP at depth: 4-row window, int8 KV, one stream (KV read once per round, so long context favours drafting)\n")
    print("| context | serial f=1.00 | serial f=1.08 | MTP conservative | MTP sibling-calibrated |\n|---:|---:|---:|---:|---:|")
    mctx = {}
    r = 4
    t_flat, t_cf, o = tokens_per_round(ACCEPT_FLAT, r - 1), tokens_per_round(ACCEPT_CF, r - 1), GLM_OVERLAP[r]
    for c in (4096, 32768, 131072, 262144, 524288, 1048576):
        kvb = mdl.kv_read_bytes(c, "int8")
        s1 = bw / mdl.step_bytes(1, c, "int8")
        s108 = s1 / 1.08
        cons = t_flat / (1.08 * (sum(mdl.weight_bytes(r).values()) + kvb + draft_hi) / bw + IDLE_MS * 1e-3)
        sib = t_cf / (1.08 * (sum(mdl.weight_bytes(r, overlap=o).values()) + kvb + draft) / bw + IDLE_MS * 1e-3)
        mctx[c] = {"serial_f1": s1, "serial_f108": s108, "conservative": cons, "sibling": sib}
        print(f"| {c // 1024}K | {s1:.1f} | {s108:.1f} | {cons:.0f} | {sib:.0f} |")
    out["mtp_vs_ctx"] = mctx
    print()

    print("### Concurrent streams (no drafting): one row per stream in the same step, int8 KV per stream at 4K\n")
    print("| streams | GB/step | aggregate tok/s f=1.00 | aggregate tok/s f=1.08 | per stream f=1.08 |\n|---:|---:|---:|---:|---:|")
    conc = {}
    for c in (1, 2, 4, 8, 16):
        b = mdl.step_bytes(c, 4096, "int8", streams=c)
        conc[c] = {"f1": c * bw / b, "f108": c * bw / (1.08 * b)}
        print(f"| {c} | {fmt_gb(b)} | {c * bw / b:.0f} | {c * bw / (1.08 * b):.0f} | {bw / (1.08 * b):.1f} |")
    out["concurrency_tok_s"] = conc
    print()

    print("## Prefill: FLOPs per prompt token\n")
    lin = mdl.linear_flops()
    print(f"Linears (all {mdl.passes} passes, 8 of {mdl.E} experts): {lin / 1e9:.2f} GFLOP/token.\n")
    print("| prompt | global attn GFLOP | sliding GFLOP | total GFLOP/token | ceiling bf16 93.4 | 40% of it | 25% | fp8 195.1 at 40% |\n|---:|---:|---:|---:|---:|---:|---:|---:|")
    pre = {}
    for L in (4096, 8192, 32768, 131072, 262144, 524288, 1048576):
        glob, loc = mdl.attn_flops(L)
        tot = lin + glob + loc
        c = HW["bf16_tflops"] * 1e12 / tot
        pre[L] = {"gflop": tot / 1e9, "ceiling_bf16": c}
        print(f"| {L // 1024}K | {glob / 1e9:.2f} | {loc / 1e9:.2f} | {tot / 1e9:.2f} | {c:,.0f} | {0.4 * c:,.0f} | {0.25 * c:,.0f} | {0.4 * HW['fp8_tflops'] * 1e12 / tot:,.0f} |")
    out["prefill"] = pre
    chunk = 4096
    chunk_bytes = sum(mdl.weight_bytes(chunk).values())
    print(f"\nA {chunk}-row chunk reads {fmt_gb(chunk_bytes)} GB of weights (every expert, every pass): {chunk_bytes / bw * 1e3:.0f} ms at 237 GB/s = "
          f"{chunk / (chunk_bytes / bw):,.0f} tok/s: bandwidth is not the prefill limit, tensor work and trellis decode are.\n")
    out["prefill_chunk_bw_tok_s"] = chunk / (chunk_bytes / bw)

    print("## KV and capacity\n")
    print("| KV | bytes/token (target) | + drafter plane | 1,048,576-token lane |\n|---|---:|---:|---:|")
    for kv in ("bf16", "int8", "q4"):
        t = mdl.kv_token_bytes(kv, drafter=False)
        td = mdl.kv_token_bytes(kv, drafter=True)
        print(f"| {kv} | {t:,} | {td:,} | {td * 1048576 / GIB:.2f} GiB |")
    ring = mdl.local_passes * (mdl.window + 7) * mdl.kv_pass_bytes("bf16")
    print(f"\nPer-slot sliding rings (bf16, {mdl.local_passes} passes x {mdl.window + 7} positions): {ring / GIB:.3f} GiB (the card says ~0.17 GiB with conv state).\n")
    budget = NODE_MEM_GIB - OS_UNACCOUNTED_GIB - MEM_FLOOR_GIB
    left = budget - resident / GIB - ACTIVATION_GIB
    print(f"Budget: {NODE_MEM_GIB} MemTotal - {OS_UNACCOUNTED_GIB} OS/CUDA - {MEM_FLOOR_GIB} floor = {budget:.1f} GiB; "
          f"- {resident / GIB:.2f} weights - {ACTIVATION_GIB} activations = **{left:.1f} GiB** for KV, slot state and the prefix cache.\n")
    print("| prefix cache | slots | KV GiB | int8 tokens (with drafter) | bf16 tokens |\n|---:|---:|---:|---:|---:|")
    cap = {}
    for pc, slots in ((0, 1), (8, 4), (12, 4), (8, 8)):
        kvg = left - pc - slots * ring / GIB
        ti = kvg * GIB / mdl.kv_token_bytes("int8")
        tb = kvg * GIB / mdl.kv_token_bytes("bf16")
        cap[f"{pc}_{slots}"] = {"kv_gib": kvg, "int8_tokens": ti, "bf16_tokens": tb}
        print(f"| {pc} GiB | {slots} | {kvg:.1f} | {ti / 1e6:.2f}M ({ti / 1048576:.1f} lanes of 1M) | {tb / 1e6:.2f}M ({tb / 1048576:.1f} lanes of 1M) |")
    out["capacity"] = cap

    if args.json:
        sys.stdout = real_stdout
        json.dump(out, sys.stdout, indent=1, default=float)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
