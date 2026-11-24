"""Generate Inkling 2 Mini DSpark drafter fixtures over a pack folder holding `mtp.safetensors` and the target's tensors.

    python oracle/gen_dspark_fixtures.py --pack DIR --fixtures DIR --out DIR [--threads 8] [--only CASE ...]

The drafter's EXL3 linears (`mtp.safetensors`, 4-bit) and the target's embed / embed_norm / unembed (EXL3 6-bit unembed)
come through `pack.PackWeights` as effective weights; the drafter has no tensor parallelism.

Cases (C):
  real_raw-counting, real_chat-sky-effort0.9, real_chat-capital-effort0, real_raw-long-window
      features of the target oracle fixture FIXTURES/<name>.safetensors (gen_fixtures.py):
      F[t] = stack(hidden[5], hidden[23], hidden[41], h_final)[t], tokens = its `tokens`; `hidden` [V+1, T, H] is indexed
      by layer pass (P entering the pass), the ids being `cfg.target_layer_ids[:-1]`;
  long
      N = 1100 seeded bf16 features (normal, per-tap scales LONG_SCALES), seeded random tokens < 200058.
Per case the reference commits the rows up to each anchor p (ascending) and proposes at p with anchor tokens[p]; the
step logits are those of its own drafts (teacher forcing on the reference's drafts equals the free-running chain).
Files OUT/C_*.npy and OUT/C_meta.json follow the drafter reference contract:
  feats u16 [N, 4, H] (bf16 bits), tokens i64 [N+1], anchors i64 [P], drafts i64 [P, 7], hd u16 [P, 8, H],
  frnn u16 [P, H], state / latent u16 [P, 7, R], logits f32 [P, 7, 200058], ctxkv u16 [3, N, 2, 8, 128].
For the real cases it prints how many leading drafts match the target's next tokens tokens[p+1 .. p+7].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import st
from dspark_ref import DSparkConfig, DSparkRef, bf16_bits
from gen_fixtures import oracle_source_sha, sha256_file
from pack import PackIndex, PackWeights
from synthetic import random_features

CASES = {
    "real_raw-counting": dict(fixture="raw-counting", anchors=[1, 13, 25, 35, 38, 41, 44, 50]),
    "real_chat-sky-effort0.9": dict(fixture="chat-sky-effort0.9", anchors=[1, 13, 25, 26, 30, 36, 42, 49]),
    "real_chat-capital-effort0": dict(fixture="chat-capital-effort0", anchors=[1, 8, 16, 24, 25, 30, 36, 40]),
    # 1,126 feature rows: anchors on both sides of the 1,024 window (the drafter's local layers read 1,024 positions back)
    "real_raw-long-window": dict(fixture="raw-long-window", anchors=[1, 600, 1023, 1024, 1025, 1100, 1119, 1126]),
    "long": dict(fixture=None, anchors=[1, 600, 1023, 1024, 1025, 1031, 1032, 1100]),
}
LONG_N = 1100
LONG_SEED = 20261117
LONG_SCALES = (0.5, 1.0, 2.0, 4.0)  # one per tap


class DSparkPackWeights:
    """The drafter's tensors (unprefixed `mtp.safetensors` names) through `PackWeights` over that one file, everything
    else (model.llm.embed / embed_norm / unembed) through `PackWeights` over the whole pack; EXL3 `<prefix>.weight`
    names come back decoded. `target_files` restricts the target index (a test folder without an index)."""

    TARGET = ("model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.unembed.weight")

    def __init__(self, pack: str, mtp_file: str = "mtp.safetensors", target_files: list[str] | None = None,
                 cache_bytes: int = 0):
        self.mtp = PackWeights(PackIndex(pack, files=[mtp_file]), cache_bytes)
        self.target = PackWeights(PackIndex(pack, files=target_files), cache_bytes)

    def _src(self, name: str) -> PackWeights:
        return self.target if name in self.TARGET else self.mtp

    def tensor(self, name: str) -> torch.Tensor:
        return self._src(name).tensor(name)

    def rows(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        return self._src(name).rows(name, ids)

    def row_range(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self._src(name).row_range(name, start, stop)


def real_features(cfg: DSparkConfig, path: str) -> tuple[torch.Tensor, np.ndarray]:
    hdr = st.read_header(path)
    hidden = st.read_tensor(hdr["hidden"])  # [L+1, T, H]
    h_final = st.read_tensor(hdr["h_final"])  # [T, H]
    feats = torch.stack([hidden[j] for j in cfg.target_layer_ids[:-1]] + [h_final], 1)
    tokens = st.read_tensor(hdr["tokens"]).numpy().astype(np.int64)
    if tokens.shape[0] != feats.shape[0] + 1:
        raise ValueError(f"{path}: {tokens.shape[0]} tokens for {feats.shape[0]} feature rows")
    return feats, tokens


def long_features(cfg: DSparkConfig) -> tuple[torch.Tensor, np.ndarray]:
    feats = random_features(cfg, LONG_N, seed=LONG_SEED, scales=LONG_SCALES)
    tokens = np.random.default_rng(LONG_SEED).integers(0, cfg.valid_vocab, LONG_N + 1, dtype=np.int64)
    return feats, tokens


def accepted(drafts: list[int], tokens: np.ndarray, p: int) -> tuple[int, int]:
    """(matching prefix length, number of comparable drafts) against tokens[p+1 .. p+7]."""
    truth = tokens[p + 1:p + 1 + len(drafts)].tolist()
    k = 0
    while k < len(truth) and drafts[k] == truth[k]:
        k += 1
    return k, len(truth)


def run_case(ref: DSparkRef, feats: torch.Tensor, tokens: np.ndarray, anchors: list[int]) -> dict:
    n = feats.shape[0]
    if sorted(anchors) != anchors or anchors[0] < 1 or anchors[-1] > n:
        raise ValueError(f"anchors {anchors} must ascend within [1, {n}]")
    state = ref.new_state()
    cur = 0
    out = {k: [] for k in ("drafts", "hd", "frnn", "state", "latent", "logits")}
    for p in anchors:
        if p > cur:
            ref.commit(state, feats[cur:p], cur)
            cur = p
        t0 = time.time()
        prop = ref.propose(state, int(tokens[p]), trace=True)
        tr = prop.trace
        out["drafts"].append(prop.drafts)
        out["hd"].append(bf16_bits(tr["hd"]))
        out["frnn"].append(bf16_bits(tr["f_rnn"]))
        out["state"].append(bf16_bits(tr["state"]))
        out["latent"].append(bf16_bits(tr["latent"]))
        out["logits"].append(tr["logits"].to(torch.float32).numpy())
        print(f"  p={p} anchor={int(tokens[p])} drafts={prop.drafts} ({time.time() - t0:.1f}s)", flush=True)
    if n > cur:
        ref.commit(state, feats[cur:n], cur)
    res = {k: np.stack(v) for k, v in out.items()}
    res["drafts"] = res["drafts"].astype(np.int64)
    res["ctxkv"] = bf16_bits(state.ctx_kv())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--fixtures", required=True, help="target oracle fixtures (gen_fixtures.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8, help="torch intra-op threads")
    ap.add_argument("--weight-cache-gib", type=float, default=0.0,
                    help="GiB of decoded float64 EXL3 weights kept in memory: every proposal streams the whole unembed "
                         "(3.7 GB of float64), which is decoded once and then served from here")
    ap.add_argument("--only", nargs="*", choices=sorted(CASES))
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = DSparkConfig.from_hf(json.load(f))
    weights = DSparkPackWeights(a.pack, cache_bytes=int(a.weight_cache_gib * (1 << 30)))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    mtp_sha = sha256_file(os.path.join(a.pack, "mtp.safetensors"))
    src_sha = oracle_source_sha()
    base = DSparkRef(cfg, weights)
    base.fold()
    print(f"mtp sha256 + fold: {time.time() - t0:.0f}s", flush=True)
    for name in a.only or list(CASES):
        case = CASES[name]
        t0 = time.time()
        if case["fixture"]:
            path = os.path.join(a.fixtures, f"{case['fixture']}.safetensors")
            feats, tokens = real_features(cfg, path)
            ids = ", ".join(str(j) for j in cfg.target_layer_ids[:-1])
            source = (f"oracle fixture {case['fixture']}.safetensors (sha256 {sha256_file(path)}): "
                      f"stack(hidden[{ids}], h_final)")
        else:
            feats, tokens = long_features(cfg)
            source = f"synthetic: seed {LONG_SEED}, normal bf16 features with per-tap scales {list(LONG_SCALES)}"
        print(f"{name}: N={feats.shape[0]} anchors={case['anchors']}", flush=True)
        res = run_case(base, feats, tokens, case["anchors"])
        secs = time.time() - t0
        arrays = {"feats": bf16_bits(feats), "tokens": tokens, "anchors": np.array(case["anchors"], dtype=np.int64),
                  **res}
        for k, v in arrays.items():
            np.save(os.path.join(a.out, f"{name}_{k}.npy"), v)
        meta = {"anchors": case["anchors"], "n_positions": int(feats.shape[0]),
                "feature_source": source, "mtp_sha256": mtp_sha, "oracle_source_sha256": src_sha,
                "seconds": round(secs, 1)}
        with open(os.path.join(a.out, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if case["fixture"]:
            acc = [accepted(d, tokens, p) for d, p in zip(res["drafts"].tolist(), case["anchors"])]
            print(f"{name}: acceptance (prefix/comparable) " + " ".join(f"p{p}:{k}/{m}" for p, (k, m)
                                                                         in zip(case["anchors"], acc)), flush=True)
        print(f"{name}: {secs:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
