"""An independent float64 transcription of the HF eager looped model (transformers be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef;
src/transformers/models/inkling2_loop/modeling_inkling2_loop.py: `Inkling2LoopTextModel.forward`, the four decoder layer
classes, `Inkling2LoopMessageGate`, `Inkling2LoopMessageAttentionResidual`, `Inkling2LoopBlockAttentionResidual`) over the
pack's tensor names, with every bf16 cast removed. It keeps HF's own op forms and control flow (the pre-loop / first pass /
remaining passes loops over `block_states`, the `num_pre_loop_blocks + 1` slice, F.conv1d with padding, unfold/flip windows
for the dynamic conv, the concatenated-sink softmax, repeat_kv, the eager-attention form of the AttnRes reads, index_add
MoE) so that comparing it with the oracle under `Rounding.exact()` checks the oracle's op structure (indexing, orientation,
schedule, state bookkeeping) independently of its rounding points. The pack-to-HF tensor renames
(`conversion_mapping.py`: `controllers.attn|k|v|mlp` -> q/k/v/mlp message gates, `down_weight`, `gain_weight` ->
`message_gate_proj`, `forget_weight` -> `forget_gate_proj`, `attn_res.attn_queries.{i}` -> layer i+1's `attention_input.query`,
the two `middle_cycle_cross_pass*queries` merged by `Inkling2LoopMergeMessageQueries`) are applied inline: the `controllers.*`
prefixes in `decoder_layer` and `hf_message_queries`.
Whole sequences only (no cache)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

F64 = torch.float64


def _w(w, name):
    return w[name].to(F64)


def rms_norm(x, weight, eps):  # Inkling2LoopRMSNorm
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def unweighted(x, eps=1e-6):  # Inkling2LoopUnweightedRMSNorm
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def l2_clip(x, threshold, scale=None):  # Inkling2LoopL2Clip / Inkling2LoopScaledL2Clip
    if scale is not None:
        x = x * scale
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def short_conv(x, weight):  # Inkling2LoopShortConvolution, x [T, C], weight [C, 1, K]
    k = weight.shape[-1]
    t, c = x.shape
    out = F.conv1d(x.T[None], weight, padding=k - 1, groups=c)[0, :, :t].T
    return (out + x) / math.sqrt(k + 1)


def dynamic_conv(x, dyn_in, static_w, weight_proj, heads):  # Inkling2LoopDynamicShortConvolution
    t, c = x.shape
    k = static_w.shape[0]
    dynamic = (dyn_in @ weight_proj.T).view(1, t, heads, 1, k)
    weights = static_w.T.reshape(heads, -1, k) + dynamic
    context = x[None].transpose(1, 2)
    context = F.pad(context, (k - 1, 0))[..., -(t + k - 1):]
    windows = context.unfold(-1, k, 1).flip(-1).transpose(1, 2)
    windows = windows.reshape(1, t, heads, -1, k)
    y = (windows * weights).sum(-1).reshape(t, c)
    return (y + x) / math.sqrt(2 * k + 1)


def rel_logits(rel_states, proj, qpos, kpos):  # Inkling2LoopRelativePositionBias, rel_states [1, T, H, d]
    extent = proj.shape[1]
    rl = (rel_states @ proj).transpose(1, 2)
    distance = (qpos[:, None] - kpos[None, :])[None, None]
    idx = distance.clamp(0, extent - 1).expand(*rl.shape[:2], -1, -1)
    bias = rl.gather(-1, idx)
    return bias.masked_fill((distance < 0) | (distance >= extent), 0.0)


def attention(cfg, w, i, xq, xk, xv, pos):
    """Inkling2LoopSlidingAttention / Inkling2LoopFullAttention forward on whole sequences; xq, xk, xv [T, H]."""
    a = f"model.llm.layers.{i}.attn."
    t = xq.shape[0]
    hq, hkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    glob = cfg.is_global(i)
    q = xq @ _w(w, a + "wq_du.weight").T
    if glob:
        k = dynamic_conv(xk @ _w(w, a + "wk_dv.weight").T, xk, _w(w, a + "k_sconv.static_w"),
                         _w(w, a + "k_sconv.weight_proj"), hkv)
        v = dynamic_conv(xv @ _w(w, a + "wv_dv.weight").T, xv, _w(w, a + "v_sconv.static_w"),
                         _w(w, a + "v_sconv.weight_proj"), hkv)
    else:
        k = short_conv(xk @ _w(w, a + "wk_dv.weight").T, _w(w, a + "k_sconv.weight"))
        v = short_conv(xv @ _w(w, a + "wv_dv.weight").T, _w(w, a + "v_sconv.weight"))
    rel = xq @ _w(w, a + "wr_du.weight").T
    gate = xq @ _w(w, a + "gate_attn.weight").T
    sinks = xq @ _w(w, a + "attn_sink_proj.weight").T
    q = l2_clip(q.view(1, t, hq, d), cfg.qk_clip, _w(w, a + "q_l2_clip.scale"))
    rel = l2_clip(rel.view(1, t, hq, -1), cfg.rel_clip, _w(w, a + "rel_l2_clip.scale"))
    if glob:
        tau = 1.0 + cfg.log_scaling_alpha * torch.log(((pos + 1).to(F64) / cfg.log_scaling_n_floor).clamp(min=1.0))
        q = q * tau[None, :, None, None]
        rel = rel * tau[None, :, None, None]
    q = q.transpose(1, 2)
    k = l2_clip(k.view(1, t, hkv, d), cfg.qk_clip).transpose(1, 2)
    v = unweighted(v.view(1, t, hkv, d), cfg.eps).transpose(1, 2)
    bias = rel_logits(rel, _w(w, a + "rel_logits_proj.proj"), pos, pos)
    grp = hq // hkv
    k = k[:, :, None].expand(1, hkv, grp, t, d).reshape(1, hq, t, d)  # repeat_kv
    v = v[:, :, None].expand(1, hkv, grp, t, d).reshape(1, hq, t, d)
    scores = torch.matmul(q, k.transpose(2, 3)) * d ** -0.5 + bias
    dist = pos[:, None] - pos[None, :]
    allowed = dist >= 0
    if not glob:
        allowed &= dist < cfg.sliding_window
    scores = scores + torch.where(allowed, 0.0, float("-inf")).to(F64)
    sink = sinks.view(1, t, hq).transpose(1, 2).unsqueeze(-1)
    probs = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    out = torch.matmul(probs, v).transpose(1, 2).reshape(t, hq * d)
    out = out * torch.sigmoid(gate)  # Inkling2LoopGatedOutputProjection
    return out @ _w(w, a + "wo_ud.weight").T


def moe(cfg, w, i, x):
    pre = f"model.llm.layers.{i}.mlp."
    logits = x @ _w(w, pre + "gate.weight").T
    scores = torch.sigmoid(logits)
    idx = torch.topk(scores + _w(w, pre + "gate.bias"), cfg.top_k, dim=-1).indices
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * cfg.route_scale
    out = torch.zeros_like(x)
    for e in range(cfg.n_experts):
        tok, slot = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        ep = pre + f"experts.{e}."
        g = x[tok] @ _w(w, ep + "gate_proj.weight").T
        u = x[tok] @ _w(w, ep + "up_proj.weight").T
        act = torch.tanh(g / cfg.gate_softcap) * cfg.gate_softcap * torch.sigmoid(g) * \
            torch.tanh(u / cfg.up_softcap) * cfg.up_softcap  # Inkling2LoopSoftcappedSwiGLU
        y = act @ _w(w, ep + "down_proj.weight").T
        out.index_add_(0, tok, y * wts[tok, slot, None])
    return out


def eager_read(query, states, hidden):
    """eager_attention_forward as the AttnRes modules call it (no mask, no sinks, scaling hidden**-0.5):
    query [T, Q, H], key = value = states [T, N, H] -> [T, Q, H]."""
    scores = torch.matmul(query, states.transpose(1, 2)) * hidden ** -0.5
    return torch.matmul(F.softmax(scores, dim=-1), states)


def block_attn_res(block_states, partial, query, hidden):  # Inkling2LoopBlockAttentionResidual
    stacked = unweighted(torch.cat([block_states, partial[:, None]], dim=1))  # [T, N, H]
    q = query.to(F64).view(-1, hidden)[None].expand(stacked.shape[0], -1, -1)
    return eager_read(q, stacked, hidden).unbind(-2)


def message_attn_res(block_states, partial, query, loop_layer_idx, hidden):  # Inkling2LoopMessageAttentionResidual
    stacked = unweighted(torch.cat([block_states, partial[:, None]], dim=1))
    q = query[loop_layer_idx].to(F64)[None].expand(stacked.shape[0], -1, -1)  # [T, 4, H]
    return eager_read(q, stacked, hidden)


def message_gate(w, prefix, hidden_states, message, eps):  # Inkling2LoopMessageGate
    gate_hidden = torch.cat([unweighted(hidden_states, eps), unweighted(message, eps)], dim=-1)
    gate_hidden = gate_hidden @ _w(w, prefix + "down_weight").T
    forget_gate = torch.tanh(gate_hidden @ _w(w, prefix + "forget_weight").T)
    message_gate_ = torch.tanh(gate_hidden @ _w(w, prefix + "gain_weight").T)
    return (1 - forget_gate) * hidden_states + message_gate_ * message


def hf_message_queries(w, cfg):
    """`Inkling2LoopMergeMessageQueries.convert`: two (2 * n_core, H) tensors -> (n_core, 4, H) = query, key, value, mlp."""
    queries = w["model.llm.middle_cycle_cross_pass_queries"].to(F64).unflatten(0, (-1, 2))
    kv_queries = w["model.llm.middle_cycle_cross_pass_kv_queries"].to(F64).unflatten(0, (-1, 2))
    return torch.cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1)


def decoder_layer(cfg, w, i, hidden_states, block_states, pos, messages=None, query_names=None):
    """The four HF decoder layer classes in one function: sliding / full attention layer, looped (`messages` given on the
    passes after the first, with the layer's four message gates) or not. `hidden_states` [T, H], `block_states` [T, N, H]."""
    hdim = cfg.hidden_size
    ln = f"model.llm.layers.{i}."
    is_block_boundary = i % cfg.layers_per_block == 0
    glob = cfg.is_global(i)
    block_residual = hidden_states
    if i == 0:  # attention_input is None
        mixes = (hidden_states, hidden_states, hidden_states)
    else:
        mixes = block_attn_res(block_states, hidden_states, w[f"model.llm.attn_res.attn_queries.{i - 1}"], hdim)
    if glob:
        qs, ks, vs = mixes
        if messages is not None:
            qm, km, vm, mm = messages.unbind(-2)
            qs = message_gate(w, ln + "controllers.attn.", qs, qm, cfg.eps)
            ks = message_gate(w, ln + "controllers.k.", ks, km, cfg.eps)
            vs = message_gate(w, ln + "controllers.v.", vs, vm, cfg.eps)
        qs = rms_norm(qs, _w(w, ln + "q_input_norm.weight"), cfg.eps)
        ks = rms_norm(ks, _w(w, ln + "k_input_norm.weight"), cfg.eps)
        vs = rms_norm(vs, _w(w, ln + "v_input_norm.weight"), cfg.eps)
    else:
        x = mixes[0]
        if messages is None:
            qs = ks = vs = rms_norm(x, _w(w, ln + "attn_norm.weight"), cfg.eps)
        else:
            qm, km, vm, mm = messages.unbind(-2)
            nw = _w(w, ln + "attn_norm.weight")
            qs = rms_norm(message_gate(w, ln + "controllers.attn.", x, qm, cfg.eps), nw, cfg.eps)
            ks = rms_norm(message_gate(w, ln + "controllers.k.", x, km, cfg.eps), nw, cfg.eps)
            vs = rms_norm(message_gate(w, ln + "controllers.v.", x, vm, cfg.eps), nw, cfg.eps)
    a = attention(cfg, w, i, qs, ks, vs, pos)
    a = short_conv(a, _w(w, ln + "attn_sconv.weight"))
    if is_block_boundary:
        block_states = torch.cat([block_states, block_residual[:, None]], dim=1)
        residual = a
    else:
        residual = block_residual + a
    m = block_attn_res(block_states, residual, w[f"model.llm.attn_res.mlp_queries.{i}"], hdim)[0]
    if messages is not None:
        m = message_gate(w, ln + "controllers.mlp.", m, messages.unbind(-2)[3], cfg.eps)
    m = moe(cfg, w, i, rms_norm(m, _w(w, ln + "mlp_norm.weight"), cfg.eps))
    m = short_conv(m, _w(w, ln + "mlp_sconv.weight"))
    return residual + m


def forward(cfg, w, ids):
    """Logits [T, unpadded_vocab] float64 and the partial block entering each of the layer passes (then the last output):
    `Inkling2LoopTextModel.forward` + `Inkling2LoopForCausalLM.forward`, line by line."""
    hdim = cfg.hidden_size
    ids = torch.as_tensor(ids)
    t = ids.shape[0]
    pos = torch.arange(t)
    loop_start, loop_end = cfg.loop_start, cfg.loop_end
    num_loop_layers = loop_end - loop_start
    message_query = hf_message_queries(w, cfg)
    hidden_states = rms_norm(_w(w, "model.llm.embed.weight")[ids], _w(w, "model.llm.embed_norm.weight"), cfg.eps)
    block_states = hidden_states.new_zeros(t, 0, hdim)
    hidden = []

    def run(i, messages=None):
        nonlocal hidden_states, block_states
        hidden.append(hidden_states)
        new_hidden_states = decoder_layer(cfg, w, i, hidden_states, block_states, pos, messages)
        if i % cfg.layers_per_block == 0:  # decoder_layer.is_block_boundary
            block_states = torch.cat([block_states, hidden_states[:, None]], dim=1)
        hidden_states = new_hidden_states

    for i in range(loop_start):  # pre loop layers
        run(i)
    pre_loop_hidden_states, pre_loop_block_states = hidden_states, block_states
    num_pre_loop_blocks = pre_loop_block_states.shape[-2]
    for i in range(loop_start, loop_end):  # first pass over loop layers
        run(i)
    for _ in range(1, cfg.loop_passes):  # remaining passes restart from the pre loop states
        previous_block_states = block_states[:, num_pre_loop_blocks + 1:, :]
        previous_hidden_states = hidden_states
        hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states
        for loop_layer_idx, i in enumerate(range(loop_start, loop_end)):
            messages = message_attn_res(previous_block_states, previous_hidden_states, message_query, loop_layer_idx,
                                        hdim)
            run(i, messages)
    for i in range(loop_end, cfg.num_layers):  # post loop layers
        run(i)
    hidden.append(hidden_states)
    x = block_attn_res(block_states, hidden_states, w["model.llm.attn_res.output_query"], hdim)[0]
    x = rms_norm(x, _w(w, "model.llm.norm.weight"), cfg.eps)
    logits = x @ _w(w, "model.llm.unembed.weight").T
    logits = torch.tanh(logits / cfg.final_softcap) * cfg.final_softcap
    assert len(hidden) == cfg.n_visits + 1 and num_loop_layers == cfg.n_core
    return logits[:, :cfg.unpadded_vocab_size], torch.stack(hidden)
