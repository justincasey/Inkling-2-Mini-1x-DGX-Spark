"""Synthetic Inkling 2 Mini models for tests: a tiny looped config with the real model's structure (AttnRes blocks, global
layers every other layer, GQA, windowed local attention, 4 short convs per layer, a core of layers that runs 3 times with
shared weights, per-core-layer message queries and controllers, sigmoid top-k router, softcapped experts), random weights
under the pack's tensor names, random EXL3 groups (mixed K, `mul1`) written as a pack folder; plus a tiny DSpark drafter
(config, `mtp.safetensors`-named weights, seeded bf16 target features).

Sizes are multiples of 128 wherever a tensor is EXL3 in the real pack (K and N of an EXL3 group must be).
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
import torch

import exl3
import st
from dspark_ref import DSparkConfig
from inkling2_ref import Inkling2LoopConfig, expert_ffn

STREAM_NAMES = ("attn", "k", "v", "mlp")


def tiny_config(**over) -> Inkling2LoopConfig:
    """8 layers, AttnRes blocks of 2 layers (boundaries at 0, 2, 4, 6), global layers 1, 3, 5, 7, the core [2, 6) x 3
    passes = 16 layer passes (global passes 1, 3, 5, 7, 9, 11, 13, 15), the drafter taps over the passes. Clip thresholds
    sit below the synthetic q/k/r norms, so every clip is active."""
    base = dict(
        hidden_size=128, num_layers=8, vocab_size=256, unpadded_vocab_size=250, num_heads=4, num_kv_heads=2,
        head_dim=64, d_rel=32, rel_extent=8, sliding_window=8, local_layer_ids=(0, 2, 4, 6), conv_kernel=4,
        layers_per_block=2, n_experts=4, top_k=2, moe_intermediate=128, gate_softcap=4.0, up_softcap=25.0,
        route_scale=3.0, qk_clip=6.0, rel_clip=4.0, log_scaling_n_floor=6.0, log_scaling_alpha=0.1, eps=1e-6,
        final_softcap=30.0, loop_start=2, loop_end=6, loop_passes=3, gate_rank=16, drafter_target_layers=(3, 9, 13, 16),
    )
    base.update(over)
    return Inkling2LoopConfig(**base)


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def tiny_dspark_config(**over) -> DSparkConfig:
    """A tiny 3-layer drafter with the real structure: layer 1 global, layers 0 and 2 local (window 6, shorter than the
    tests' contexts), GQA 2, block 8, 4 taps, padded vocab 256 with 250 valid ids."""
    base = dict(
        hidden_size=128, num_layers=3, num_heads=4, num_kv_heads=2, head_dim=64, intermediate_size=128,
        local_layer_ids=(0, 2), sliding_window=6, block_size=8, markov_rank=128, rope_theta=1e6, eps=1e-6,
        target_layer_ids=(3, 9, 13, 16), vocab_size=256, valid_vocab=250, softcap=30.0,
    )
    base.update(over)
    return DSparkConfig(**base)


def random_features(cfg: DSparkConfig, n: int, seed: int = 0, scales: tuple[float, ...] | None = None) -> torch.Tensor:
    """Seeded bf16 drafter features [n, taps, H]: normal values scaled per tap."""
    g = torch.Generator().manual_seed(seed)
    scales = scales or (1.0,) * cfg.n_taps
    x = torch.randn(n, cfg.n_taps, cfg.hidden_size, generator=g, dtype=torch.float64)
    return _bf(x * torch.tensor(scales, dtype=torch.float64)[None, :, None])


# ----------------------------------------------------------------------------------------------------------------------
# EXL3 groups


def random_exl3_group(rng: np.random.Generator, k: int, n: int, twice: int = 8, cb: str = "mul1",
                      scale_in: float = 0.03, scale_out: float = 0.03) -> dict[str, np.ndarray]:
    """One random EXL3 group (trellis int16 [K/16, N/16, 8*twice], suh/svh fp16 with random signs, codebook marker)."""
    trellis = rng.integers(-32768, 32768, size=(k // 16, n // 16, 8 * twice), dtype=np.int64).astype(np.int16)
    sgn = lambda m: rng.choice([-1.0, 1.0], size=m)  # noqa: E731
    out = {
        "trellis": trellis,
        "suh": (sgn(k) * scale_in * (1 + 0.2 * rng.random(k))).astype(np.float16),
        "svh": (sgn(n) * scale_out * (1 + 0.2 * rng.random(n))).astype(np.float16),
    }
    if cb != "3inst":
        marker = exl3.MARKER_MCG if cb == "mcg" else exl3.MARKER_MUL1
        out[cb] = np.array(marker, dtype=np.uint32).view(np.int32).reshape(())
    return out


def random_exl3_weights(rng: np.random.Generator, specs: dict[str, tuple[int, int, float, float]],
                        cb: str = "mul1") -> tuple[dict[str, dict], dict[str, torch.Tensor]]:
    """EXL3 replacements of dense `<prefix>.weight` tensors. `specs`: prefix -> (out features N, in features K, bits,
    target effective-weight std). Returns (groups by prefix, effective float64 weights `<prefix>.weight` in nn.Linear
    layout [N, K] decoded with `exl3.decode_weight`)."""
    groups, eff = {}, {}
    for prefix, (n, k, bits, std) in specs.items():
        g = random_exl3_group(rng, k, n, twice=int(2 * bits), cb=cb, scale_in=std ** 0.5, scale_out=std ** 0.5)
        groups[prefix] = g
        eff[prefix + ".weight"] = torch.from_numpy(
            exl3.decode_weight(g["trellis"], cb, g["suh"].astype(np.float64), g["svh"].astype(np.float64)).T.copy())
    return groups, eff


# ----------------------------------------------------------------------------------------------------------------------
# target model


def _dense_specs(cfg: Inkling2LoopConfig, bits: float = 6.0) -> dict[str, tuple[int, int, float, float]]:
    """Which tensors are dense (EXL3 in the pack) and their (out, in, bits, std): attention linears and the unembed."""
    h, hq, hkv, d, dr = cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.d_rel
    s = h ** -0.5
    out: dict[str, tuple[int, int, float, float]] = {}
    for i in range(cfg.num_layers):
        a = f"model.llm.layers.{i}.attn."
        out[a + "wq_du"] = (hq * d, h, bits, s * 3)
        out[a + "wr_du"] = (hq * dr, h, bits, s * 3)
        out[a + "gate_attn"] = (hq * d, h, bits, s)
        out[a + "wk_dv"] = (hkv * d, h, bits, s * 3)
        out[a + "wv_dv"] = (hkv * d, h, bits, s)
        out[a + "wo_ud"] = (h, hq * d, bits, (hq * d) ** -0.5)
    out["model.llm.unembed"] = (cfg.vocab_size, h, bits, h ** -0.5 * 4)
    return out


def random_weights(cfg: Inkling2LoopConfig, seed: int = 0, dense_experts: bool = True) -> dict[str, torch.Tensor]:
    """Plain tensors by pack name (BF16 / F32 as in the pack), every tensor dense. Dense experts use nn.Linear layout
    [out, in]. Use `random_exl3_pack` for the pack's real mix of EXL3 and plain tensors."""
    g = torch.Generator().manual_seed(seed)
    h, hq, hkv, d, dr = cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.d_rel
    rn = lambda *s, std=1.0: torch.randn(*s, generator=g, dtype=torch.float64) * std  # noqa: E731
    near1 = lambda *s: 1.0 + rn(*s, std=0.1)  # noqa: E731
    w: dict[str, torch.Tensor] = {
        "model.llm.embed.weight": _bf(rn(cfg.vocab_size, h)),
        "model.llm.embed_norm.weight": _bf(near1(h)),
        "model.llm.unembed.weight": _bf(rn(cfg.vocab_size, h, std=h ** -0.5 * 4)),
        "model.llm.norm.weight": _bf(near1(h)),
        "model.llm.attn_res.output_query": rn(h, std=2.0).to(torch.float32),
        # (attn, mlp) then (k, v) queries of each core layer, as the pack stores them
        "model.llm.middle_cycle_cross_pass_queries": rn(2 * cfg.n_core, h, std=2.0).to(torch.float32),
        "model.llm.middle_cycle_cross_pass_kv_queries": rn(2 * cfg.n_core, h, std=2.0).to(torch.float32),
    }
    for i in range(cfg.num_layers):
        pre = f"model.llm.layers.{i}."
        a = pre + "attn."
        glob = cfg.is_global(i)
        if i > 0:
            shape = (3, h) if glob else (h,)
            w[f"model.llm.attn_res.attn_queries.{i - 1}"] = rn(*shape, std=2.0).to(torch.float32)
        w[f"model.llm.attn_res.mlp_queries.{i}"] = rn(h, std=2.0).to(torch.float32)
        if glob:
            for n in "qkv":
                w[pre + f"{n}_input_norm.weight"] = _bf(near1(h))
            for n in "kv":
                w[a + f"{n}_sconv.static_w"] = _bf(rn(cfg.conv_kernel, hkv * d, std=0.4))
                w[a + f"{n}_sconv.weight_proj"] = _bf(rn(hkv * cfg.conv_kernel, h, std=0.4 * h ** -0.5))
        else:
            w[pre + "attn_norm.weight"] = _bf(near1(h))
            for n in "kv":
                w[a + f"{n}_sconv.weight"] = _bf(rn(hkv * d, 1, cfg.conv_kernel, std=0.4))
        w[pre + "mlp_norm.weight"] = _bf(near1(h))
        w[pre + "attn_sconv.weight"] = _bf(rn(h, 1, cfg.conv_kernel, std=0.4))
        w[pre + "mlp_sconv.weight"] = _bf(rn(h, 1, cfg.conv_kernel, std=0.4))
        s = h ** -0.5
        w[a + "wq_du.weight"] = _bf(rn(hq * d, h, std=s * 3))
        w[a + "wr_du.weight"] = _bf(rn(hq * dr, h, std=s * 3))
        w[a + "gate_attn.weight"] = _bf(rn(hq * d, h, std=s))
        w[a + "attn_sink_proj.weight"] = _bf(rn(hq, h, std=s))
        w[a + "wk_dv.weight"] = _bf(rn(hkv * d, h, std=s * 3))
        w[a + "wv_dv.weight"] = _bf(rn(hkv * d, h, std=s))
        w[a + "wo_ud.weight"] = _bf(rn(h, hq * d, std=(hq * d) ** -0.5))
        w[a + "q_l2_clip.scale"] = _bf(near1(hq, d))
        w[a + "rel_l2_clip.scale"] = _bf(near1(hq, dr))
        extent = cfg.rel_extent if glob else cfg.sliding_window
        w[a + "rel_logits_proj.proj"] = _bf(rn(dr, extent, std=0.5))
        w[pre + "mlp.gate.weight"] = _bf(rn(cfg.n_experts, h, std=s))
        w[pre + "mlp.gate.bias"] = rn(cfg.n_experts, std=0.05).to(torch.float32)
        if cfg.loop_start <= i < cfg.loop_end:
            r = cfg.gate_rank
            for n in STREAM_NAMES:
                c = pre + f"controllers.{n}."
                w[c + "down_weight"] = _bf(rn(r, 2 * h, std=(2 * h) ** -0.5 * 1.5))
                w[c + "gain_weight"] = _bf(rn(h, r, std=r ** -0.5 * 0.8))
                w[c + "forget_weight"] = _bf(rn(h, r, std=r ** -0.5 * 0.8))
        if dense_experts:
            for e in range(cfg.n_experts):
                ep = pre + f"mlp.experts.{e}."
                w[ep + "gate_proj.weight"] = _bf(rn(cfg.moe_intermediate, h, std=s * 2))
                w[ep + "up_proj.weight"] = _bf(rn(cfg.moe_intermediate, h, std=s * 2))
                w[ep + "down_proj.weight"] = _bf(rn(h, cfg.moe_intermediate, std=cfg.moe_intermediate ** -0.5))
    return w


def expert_bits(layer: int, e: int, proj: str) -> float:
    """Deterministic K of a synthetic expert projection, mixed as in the pack (4 or 5 bit; down_proj alone may differ)."""
    return 5.0 if (layer * 7 + e * 3 + (1 if proj == "down_proj" else 0)) % 4 in (0, 1) else 4.0


class ExpertGroups:
    """`ExpertSource` over in-memory EXL3 groups (the same `Exl3Linear` + `expert_ffn` path as `pack.PackExperts`)."""

    def __init__(self, groups: dict[str, dict]):
        self.groups = groups
        self._lin: dict[str, exl3.Exl3Linear] = {}

    def lin(self, prefix: str) -> exl3.Exl3Linear:
        if prefix not in self._lin:
            g = self.groups[prefix]
            cb = "mul1" if "mul1" in g else "mcg" if "mcg" in g else "3inst"
            self._lin[prefix] = exl3.Exl3Linear(g["trellis"], cb, g["suh"].astype(np.float64),
                                                g["svh"].astype(np.float64))
        return self._lin[prefix]

    def ffn_many(self, layer, jobs, gate_cap, up_cap, rounding, fp16_in=False):
        pre = lambda e, p: f"model.llm.layers.{layer}.mlp.experts.{e}.{p}"  # noqa: E731
        return [expert_ffn(self.lin(pre(e, "gate_proj")), self.lin(pre(e, "up_proj")), self.lin(pre(e, "down_proj")),
                           x, gate_cap, up_cap, rounding, fp16_in) for e, x in jobs]


def random_exl3_pack(cfg: Inkling2LoopConfig, seed: int = 0):
    """The pack's mix for a tiny config: the attention linears and the unembed EXL3 6-bit, the routed experts EXL3 4 or 5
    bit (mixed), everything else plain. Returns (plain tensors, EXL3 groups by prefix, reference weights): the reference
    weights are the plain tensors plus the effective float64 weights of every dense EXL3 group (`<prefix>.weight`,
    nn.Linear layout), for `DictWeights` + `ExpertGroups(groups)`, the in-memory twin of `PackWeights` + `PackExperts`."""
    base = random_weights(cfg, seed, dense_experts=False)
    rng = np.random.default_rng(seed + 1)
    specs = _dense_specs(cfg)
    plain = {k: v for k, v in base.items() if k[:-len(".weight")] not in specs or not k.endswith(".weight")}
    groups, eff = random_exl3_weights(rng, specs)
    s, h, inter = cfg.hidden_size ** -0.5, cfg.hidden_size, cfg.moe_intermediate
    for layer in range(cfg.num_layers):
        e_specs = {}
        for e in range(cfg.n_experts):
            ep = f"model.llm.layers.{layer}.mlp.experts.{e}."
            for proj, (n, k, std) in (("gate_proj", (inter, h, s * 2)), ("up_proj", (inter, h, s * 2)),
                                      ("down_proj", (h, inter, inter ** -0.5))):
                e_specs[ep + proj] = (n, k, expert_bits(layer, e, proj), std)
        g, _ = random_exl3_weights(rng, e_specs)
        groups.update(g)
    reference = dict(plain)
    reference.update(eff)
    return plain, groups, reference


def write_pack(root: str, cfg: Inkling2LoopConfig, plain: dict[str, torch.Tensor], groups: dict[str, dict]) -> None:
    """A pack folder in the real layout: plain tensors in `model-00001-of-00002.safetensors`, EXL3 parts
    (`<prefix>.trellis/.suh/.svh/.mul1`) in `model-00002-of-00002.safetensors`, an index, and a config.json."""
    os.makedirs(root, exist_ok=True)
    weight_map: dict[str, str] = {}
    for fn, tensors in (("model-00001-of-00002.safetensors", plain),
                        ("model-00002-of-00002.safetensors",
                         {f"{p}.{part}": arr for p, g in groups.items() for part, arr in g.items()})):
        st.write(os.path.join(root, fn), tensors)
        weight_map.update({n: fn for n in tensors})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config_json(cfg), f, indent=1)


def config_json(cfg: Inkling2LoopConfig) -> dict:
    """The pack's config.json layout (top level + text_config with loop_config + mtp_config.target_layer_ids)."""
    text = {
        "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_layers, "vocab_size": cfg.vocab_size,
        "unpadded_vocab_size": cfg.unpadded_vocab_size, "num_attention_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_kv_heads, "head_dim": cfg.head_dim, "d_rel": cfg.d_rel,
        "rel_extent": cfg.rel_extent, "sliding_window_size": cfg.sliding_window,
        "local_layer_ids": list(cfg.local_layer_ids), "sconv_kernel_size": cfg.conv_kernel,
        "n_layers_per_attn_res_block": cfg.layers_per_block, "n_routed_experts": cfg.n_experts,
        "num_experts_per_tok": cfg.top_k, "intermediate_size": cfg.moe_intermediate,
        "mlp_gate_softcap": cfg.gate_softcap, "mlp_up_softcap": cfg.up_softcap, "route_scale": cfg.route_scale,
        "qk_l2_clip_threshold": cfg.qk_clip, "rel_l2_clip_threshold": cfg.rel_clip,
        "log_scaling_n_floor": cfg.log_scaling_n_floor, "log_scaling_alpha": cfg.log_scaling_alpha,
        "rms_norm_eps": cfg.eps, "final_logit_softcapping": cfg.final_softcap, "n_shared_experts": 0,
        "use_gate_bias": True, "use_value_norm_no_gamma": True, "attn_res_normalize_values": True,
        "loop_config": {"core_start": cfg.loop_start, "core_end": cfg.loop_end, "num_passes": cfg.loop_passes,
                        "gate_rank": cfg.gate_rank, "is_varloop": False},
    }
    return {"model_type": "inkling2_loop", "text_config": text,
            "mtp_config": {"target_layer_ids": list(cfg.drafter_target_layers)},
            "quantization_config": {"quant_method": "exl3"}}


# ----------------------------------------------------------------------------------------------------------------------
# drafter


def dspark_dense_specs(cfg: DSparkConfig, bits: float = 4.0) -> dict[str, tuple[int, int, float, float]]:
    """The drafter's EXL3 linears (every projection, `rnn_proj`, `hidden_out`): prefix -> (out, in, bits, std)."""
    h, nh, nkv, d, inter, rk = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_size,
                                cfg.markov_rank)
    s = h ** -0.5
    out = {
        "markov_head.rnn_proj": (3 * rk, 2 * rk + 2 * h, bits, 2 * (2 * rk + 2 * h) ** -0.5),
        "markov_head.hidden_out": (h, rk, bits, rk ** -0.5),
    }
    for i in range(cfg.num_layers):
        a = f"layers.{i}.self_attn."
        out[a + "q_proj"] = (nh * d, h, bits, s)
        out[a + "k_proj"] = (nkv * d, h, bits, s)
        out[a + "v_proj"] = (nkv * d, h, bits, s)
        out[a + "k_proj_ctx"] = (nkv * d, h, bits, s)
        out[a + "v_proj_ctx"] = (nkv * d, h, bits, s)
        out[a + "o_proj"] = (h, nh * d, bits, (nh * d) ** -0.5)
        out[f"layers.{i}.mlp.gate_proj"] = (inter, h, bits, s * 2)
        out[f"layers.{i}.mlp.up_proj"] = (inter, h, bits, s * 2)
        out[f"layers.{i}.mlp.down_proj"] = (h, inter, bits, inter ** -0.5)
    return out


def random_dspark_weights(cfg: DSparkConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    """BF16 tensors under the unprefixed `mtp.safetensors` names plus the target's embed, embed_norm and unembed."""
    g = torch.Generator().manual_seed(seed)
    h, nh, nkv, d, inter, rk = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_size,
                                cfg.markov_rank)
    rn = lambda *s, std=1.0: _bf(torch.randn(*s, generator=g, dtype=torch.float64) * std)  # noqa: E731
    near1 = lambda *s: _bf(1.0 + torch.randn(*s, generator=g, dtype=torch.float64) * 0.1)  # noqa: E731
    s = h ** -0.5
    w = {
        "model.llm.embed.weight": rn(cfg.vocab_size, h),
        "model.llm.embed_norm.weight": near1(h),
        "model.llm.unembed.weight": rn(cfg.vocab_size, h, std=s * 4),
        "fuse_weight": rn(cfg.num_layers + 1, cfg.n_taps),
        "mask_embed": rn(h),
        "norm.weight": near1(h),
        "markov_head.dflare_norm.weight": near1(h),
        "markov_head.markov_w1.weight": rn(cfg.vocab_size, rk),
        "markov_head.rnn_proj.weight": rn(3 * rk, 2 * rk + 2 * h, std=2 * (2 * rk + 2 * h) ** -0.5),
        "markov_head.rnn_proj.bias": rn(3 * rk, std=0.2),
        "markov_head.hidden_out.weight": rn(h, rk, std=rk ** -0.5),
        "markov_head.hidden_out.bias": rn(h, std=0.2),
    }
    for i in range(cfg.num_layers):
        p = f"layers.{i}."
        a = p + "self_attn."
        for n in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
            w[p + n + ".weight"] = near1(h)
        for n in ("q_norm", "k_norm", "k_norm_ctx"):
            w[a + n + ".weight"] = near1(d)
        w[a + "q_proj.weight"] = rn(nh * d, h, std=s)
        w[a + "k_proj.weight"] = rn(nkv * d, h, std=s)
        w[a + "v_proj.weight"] = rn(nkv * d, h, std=s)
        w[a + "k_proj_ctx.weight"] = rn(nkv * d, h, std=s)
        w[a + "v_proj_ctx.weight"] = rn(nkv * d, h, std=s)
        w[a + "o_proj.weight"] = rn(h, nh * d, std=(nh * d) ** -0.5)
        w[p + "mlp.gate_proj.weight"] = rn(inter, h, std=s * 2)
        w[p + "mlp.up_proj.weight"] = rn(inter, h, std=s * 2)
        w[p + "mlp.down_proj.weight"] = rn(h, inter, std=inter ** -0.5)
    return w


def random_dspark_exl3(cfg: DSparkConfig, seed: int = 0):
    """The drafter as the pack has it: every linear EXL3 4-bit `mul1` in `mtp.safetensors`-style groups, norms, fuse_weight,
    mask_embed, biases and `markov_w1` plain. Returns (plain, groups, reference) like `random_exl3_pack`; `plain` also
    holds the target's embed / embed_norm / unembed (dense, as the drafter reads them through the target's source)."""
    base = random_dspark_weights(cfg, seed)
    specs = dspark_dense_specs(cfg)
    rng = np.random.default_rng(seed + 1)
    groups, eff = random_exl3_weights(rng, specs)
    plain = {k: v for k, v in base.items() if not (k.endswith(".weight") and k[:-len(".weight")] in specs)}
    reference = dict(plain)
    reference.update(eff)
    return plain, groups, reference
