# Inkling 2 Mini -> REXL3: operation-by-operation port spec

This spec defines the forward pass of the target model (the looped 24-layer / 48-pass text model), the 3-layer block
drafter and the vision tower of pack `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw@44c78c9d`. It maps every op to the tensors
of that pack, gives the TP1 shapes and the rounding points of the reference implementations, and inventories what the
REXL3 `inkling2` family already has. It follows `docs/PORT-SPEC.md` of Inkling-2-Small (key `SMALL`), which it names
where an op is identical, and it describes the Mini in full wherever the loop, the dimensions, the mixed-K EXL3 weights or
the drafter differ. The reference we build (the oracle) and the engine implement this document. Where the references
disagree or are silent, the item is an UNKNOWN (section 9) and `docs/NUMERICS.md` settles it or names the experiment.

Reading order for the loop: 3.2 (schedule), 3.3 (AttnRes), 3.4 (loop messages).

## 0 Sources and citation keys

Every source is pinned by a full revision; nothing floats on `main`. Citation form in the text: the key in square
brackets, then a locator in backticks, for example [HFL] `modeling:1308-1449` is `modeling_inkling2_loop.py` lines
1308-1449 at the pinned revision. Locator prefixes: `modeling` = `modeling_inkling2_loop.py`, `modular` =
`modular_inkling2_loop.py`, `config` = `configuration_inkling2_loop.py` (all under
`src/transformers/models/inkling2_loop/`); `n/` = `vllm/models/inkling2/nvidia/`, `n/ops/` = `vllm/models/inkling2/nvidia/ops/`.

| Key | What | Repo / path | Pinned revision |
|---|---|---|---|
| HFL | Eager reference of the Mini: `modeling_inkling2_loop.py` (84,280 B), `modular_inkling2_loop.py` (32,226 B), `configuration_inkling2_loop.py` (10,259 B). Primary source of semantics | `huggingface/transformers` `src/transformers/models/inkling2_loop/` (main, committed 2026-11-22T18:17:47Z) | `be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef` |
| HFS | Eager reference of the Small, same tree: `models/inkling2/` (modeling 83,879 B, configuration, image processing). Used for the vision tower, which the Mini reuses through `AutoModel`, and for diffing | `huggingface/transformers` `src/transformers/models/inkling2/` | `be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef` |
| HFX | Checkpoint-name conversion: `conversion_mapping.py` and `core_model_loading.py` (the `inkling2_loop_text` renames, `Inkling2LoopMergeMessageQueries`, `Inkling2UnpadVocab`, `Inkling2IndexOffsetWeightRename`) | `huggingface/transformers` `src/transformers/` | `be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef` |
| VLM | vLLM model definition, second source of semantics and of rounding points: `vllm/models/inkling2/` (`configs.py`, `nvidia/{model,loop,attention,dspark,moe,sconv,vision}.py`, `nvidia/ops/*`, `speculator*.py`), plus `vllm/renderers/inkling{,_encoding}.py` and `vllm/parser/inkling.py`. Semantics only: no code is taken from it | `vllm-project/vllm` main (committed 2026-11-23T02:40:48Z) | `2b55f82b085a765e4c813691d505563b397b4173` |
| PR249 | Open PR #69249 "[Doc] Document Inkling 2 Mini DSpark on SM120": the tested Mini serving configuration (7 speculative tokens, fp8 KV, `limit-mm-per-prompt image 0`) | `vllm-project/vllm` pull 69249, head | `3af01349730afd2d5c4d6367c2607897b08f8c4c` |
| PR181 | Open PR #69181 "[Perf][Model] Inkling 2: back only readable sconv-state blocks on SM12x": states that a step reads only the 3 conv positions before its first token; no model-semantics change | `vllm-project/vllm` pull 69181, head | `b8596d0dd78d441b1841c5557bffeb2d75da5dda` |
| PR096 | Open PR #69096 "[Bugfix][Model] Avoid reshape_and_cache_flash in Inkling2 DSpark KV write": the drafter's KV write only; layout, slot mapping and scaling unchanged | `vllm-project/vllm` pull 69096, head | `6fab538568eb1c303fe5b4050cde423144f0517f` |
| PACK | The pack under port: `config.json`, `generation_config.json`, `processor_config.json`, `tokenizer_config.json`, `chat_template.jinja`, `README.md` (the card), `intake/census.json` (10,075 modules of the two trunk shards) and the `mtp.safetensors` header (HTTP range read) | `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` (in this repo as `intake/`) | `44c78c9d229b43dd45e41506071b9f5f939bed44` |
| BASE | The base model the pack quantizes. Its `config.json` equals the pack's except `quantization_config` (modelopt NVFP4 vs exl3) | `thinkingmachines/Inkling-2-Mini` | `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` |
| SMALL | The Small port spec, numerics and oracle (our own files, Apache-2.0): `docs/PORT-SPEC.md`, `docs/NUMERICS.md`, `oracle/inkling2_ref.py`, `oracle/dspark_ref.py`, `oracle/exl3.py` | `~/recipes/Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark` (git, local) | `12337a2afc4fcf831a3feec08b75ea679ff7bd83` |
| ENG | The engine tree the family is built in: branch `inkling2-loop` (upstream main `02ef8cb` plus the Small family). Reuse inventory (section 8) | `~/projects/rexl3-inkling2-loop` (git, local) | `82bbda060e0987794af263228c25f461ce44399f` |
| RCP | This recipe at the commit the spec lane branched from: `docs/FIT.md`, `docs/ROOFLINE.md`, the execution plan | `~/recipes/Inkling-2-Mini-1x-DGX-Spark` (git, local) | `61382c75873a099bae9734cde9361cff50898210` |

Provenance rule (plan, Constraints): the engine's kernels are REXL3-original. HFL, VLM and the Small recipe's own files
are read for what the model computes, never copied or paraphrased into kernel code. The HF files are Apache-2.0, the
vLLM files Apache-2.0; the engine repo is AGPL-3.0-only.

## 1 Global facts

All values below come from `intake/files/config.json` (jq paths are against that file), `intake/census.json` or the
`mtp.safetensors` header. The command that reproduces each one is in the third column; `A.n` is an entry of the
appendix, where the command and its printed output are kept (run on 2026-11-22 from the recipe root).

### 1.1 Facts

| Fact | Value | Reproduce |
|---|---|---|
| Architecture / model type | `Inkling2LoopForConditionalGeneration` / `inkling2_loop` | `jq -c '[.architectures,.model_type]' intake/files/config.json` |
| Hidden size H | 2304 | `.text_config.hidden_size` (A.1) |
| Decoder layers of weights | 24 | `.text_config.num_hidden_layers` (A.1) |
| Heads (global and sliding alike) | 24 query, 8 KV, head_dim 128: GQA 3, q width 3072, KV width 1024 | `.text_config.{num_attention_heads,num_key_value_heads,head_dim,swa_*}` (A.1) |
| Relative-logit state | d_rel 16, rel_extent 1024, window 1024 | `.text_config.{d_rel,rel_extent,sliding_window_size}` (A.1) |
| Short-conv kernel | 4 | `.text_config.sconv_kernel_size` (A.1) |
| Clips / log scaling | q,k l2 threshold 22.627416997969522, r l2 threshold 8.0; tau = 1 + 0.1 ln(max((pos+1)/8192, 1)) | `.text_config.{qk_l2_clip_threshold,rel_l2_clip_threshold,log_scaling_n_floor,log_scaling_alpha}` (A.1) |
| AttnRes | block of 6 layers, normalized values, value norm without gamma | `.text_config.{n_layers_per_attn_res_block,attn_res_normalize_values,use_value_norm_no_gamma}` (A.1) |
| Global-attention layers | {5, 11, 17, 23} = layers with (i+1) % 6 == 0; the other 20 are sliding | A.2 |
| Loop | core layers 6..17 (`core_end` 18 is exclusive), 3 passes, gate rank 64, not variable | `.text_config.loop_config` (A.1, A.2) |
| Layer visits (passes) per token | 48 = 24 + (3-1) x 12 | A.2 |
| Global visits | 8: visits 5, 11, 17, 23, 29, 35, 41, 47 (every visit v with v % 6 == 5, ordinal v // 6) | A.2 |
| Sliding visits | 40 | A.2 |
| Experts | 128 routed, top 8, intermediate 1152, no shared expert, sigmoid scores with a bias, route scale 8 | `.text_config.{n_routed_experts,num_experts_per_tok,intermediate_size,n_shared_experts,route_scale,use_gate_bias}` (A.1); router bias tensor `F32[128]` per layer (A.5) |
| Expert activation softcaps | gate 4.0, up 25.0 | `.text_config.{mlp_gate_softcap,mlp_up_softcap}` (A.1) |
| Expert tensors | 24 x 128 x 3 = 9,216: 7,944 at K = 4, 1,272 at K = 5; per expert (gate, up, down) is (4,4,4) x 2,372, (4,4,5) x 414 or (5,5,5) x 286; 1,252 of the 1,272 5-bit tensors sit in layers 6-17 | A.3 |
| Expert trellis shapes | gate/up: `I16 [144, 72, 16K]`; down: `I16 [72, 144, 16K]` (K = 4: last dim 64; K = 5: 80) | A.3 |
| Dense EXL3 tensors | attention `wq_du wk_dv wv_dv wr_du gate_attn wo_ud`, `unembed`, all vision block and projector linears: K = 6 (last trellis dim 96); the vision patch embedding is BF16 | A.4, A.10 |
| Codebook | every EXL3 tensor carries `mul1`: 9,501 of the trunk (9,216 experts, 144 attention, 1 unembed, 140 vision) and all 29 of the drafter; none carries `mcg` | A.4, A.11 |
| Unembed | EXL3 `[144, 12568, 96]`: out 201,088 rows (padded), vocab 201,024, valid logits 200,058 | A.4, A.1 |
| Final soft cap | 30.0; EOS 200006; pad 199999 | `.text_config.final_logit_softcapping`, `.eos_token_id` (A.1, A.12) |
| Router / controller / convs / queries | dense, not EXL3: router BF16 `[128, 2304]` + F32 bias, `attn_sink_proj` BF16 `[24, 2304]`, controllers BF16, conv weights BF16, AttnRes queries F32 | A.5 |
| Loop tensors | 12 loop layers x 4 controllers (`attn`, `k`, `v`, `mlp`) x (`down_weight` `[64, 4608]`, `gain_weight` `[2304, 64]`, `forget_weight` `[2304, 64]`), all BF16; `middle_cycle_cross_pass_queries` and `..._kv_queries` F32 `[24, 2304]` | A.5, A.6 |
| Drafter | 3 layers, H 2304, 24 / 8 / 128 heads, MLP 9216, rope theta 1e6, local layers {0, 2} (window 1024), layer 1 global, block 8 (7 drafted tokens), Markov rank 256 `rnn_hidden`, taps [5, 23, 41, 48], `fuse_weight` BF16 `[4, 4]`; 141 tensors; every drafter linear EXL3 K = 4 | `.mtp_config` (A.11) |
| Vision | 23 blocks, width 1024 (16 heads x 64), MLP 4096, patch 20, merge 2 (one LM token = 40 x 40 px), position grid 32 x 32, projector 4096 -> 4096 -> 2304 (the 2,304 is the text hidden size, not the tower width); the block and projector linears are EXL3 K = 6 with BF16 biases, the patch embedding is BF16 `[1024, 1200]` | `.vision_config` (A.10) |
| Resident weights | 15,032,344,840 B = 14.000 GiB: routed experts 11.843, attention (EXL3 + small tensors) 0.457, embedding 0.863, unembed 0.324, vision 0.226, drafter 0.220, controllers + routers + glue 0.067 GiB | A.7 |
| KV per token | bf16 32,768 B; int8 + exponent bytes 16,512 B; + drafter plane 36,864 / 18,576 B | A.8 |
| Fixed state per slot | target rings 168,919,040 B (0.1573 GiB bf16), drafter rings 8,454,144 B, conv state 1,916,928 B: 0.167 GiB | A.8 |

The card quotes 26.4B total and 3.4B active parameters (PACK `README.md:163-165`); a parameter count from the dimensions
gives 26.038 B for the text model alone and 1.087 B per extra loop pass (A.9). Nothing in this spec uses the card's counts.

### 1.2 What changes against the Small

The Small is a 48-layer, H = 4608 model sharded over four ranks with BF16 dense weights and `mcg` EXL3 experts
(SMALL section 1). The Mini differs as follows; the first five change kernels.

1. **Loop.** Layers 6-17 execute three times with the same weights (48 visits). Each visit owns its KV (ring or global
   pages), ring and four conv streams; visits of the second and third pass (`p = 1, 2`) take four per-stream *messages* from the previous pass
   through a gated low-rank controller (3.4). New ops: message AttnRes read, pair RMSNorm + `down` + `gain`/`forget` +
   blend + norm. The AttnRes stack restarts from the prelude at every pass and the coda continues from the last pass (3.2).
2. **Dimensions and GQA at TP1.** H 2304, 24 q / 8 KV heads (GQA 3 where the Small had 12 / 2 = 6 per rank), 128 experts
   of width 1152 (Small: 384 of 2304, 96 a rank), no collectives, no vocab or head sharding. Every constant baked in for
   4608 / 2304 / 384 / 96 / GQA 6 changes (section 8).
3. **Dense EXL3 almost everywhere.** Attention q/k/v/r/gate/o, the unembed (201,088 padded rows), the vision block and projector
   linears and the drafter's linears are EXL3 (K = 6 and 4) where the Small had BF16. The router, `attn_sink_proj`, controllers,
   convs, the vision patch embedding and the AttnRes queries stay dense. Each EXL3 tensor has its own `suh`/`svh`, so an input
   shared by several tensors is rotated once per tensor.
4. **Mixed K inside a layer and inside an expert.** Routed experts are K = 4 or K = 5 per tensor, with `mul1` codebooks
   (Small: one K, `mcg`). Layers 6-17 hold all three (gate, up, down) classes and 1,252 of the 1,272 5-bit tensors; layers 0-5 and 20 are entirely 4-bit (A.3).
5. **Drafter.** 3 layers (layer 1 global, layers 0 and 2 local), 4 taps over the 48 visits, EXL3 4-bit linears including
   `rnn_proj` `[768, 5120]` and `hidden_out`, so the folded `unembed x hidden_out` projection of the Small must be
   derived from two quantized matrices (UNKNOWN-17). Small: 5 layers, 8 taps, BF16.
6. Log scaling starts at position 8192 instead of 128000 (a constant, but tau > 1 now applies to most of a long context). The
   global KV per token happens to equal the Small's: 32 KiB in bf16, because 8 global visits (Small: 8 global layers) each hold
   8 KV heads x 128 x 2 x 2 B = 4 KiB.
7. The chat template of the pack differs from the Small's in how tool-call arguments are serialised (7).

## 2 Notation, dtypes and the TP1 layout

- `T` = tokens in the batch, `H` = 2304, `V` = 48 visits, `E` = 128, `k` = 8, `I` = 1152, `Hq` = 24, `Hkv` = 8, `G` = 3
  (query heads per KV head), `D` = 128, `d_rel` = 16, `X` = rel_extent = window = 1024, `W` = conv kernel = 4. `eps` = 1e-6
  (`rms_norm_eps`) everywhere, including the weightless norms (HFL hardcodes 1e-6 in `Inkling2LoopMessageAttentionResidual`
  and `Inkling2LoopBlockAttentionResidual`, [HFL] `modeling:145-152, 828-836`; the value equals `rms_norm_eps` here).
- A **layer** is one of the 24 weight sets (`l` = 0..23); a **visit** `v` = 0..47 is one execution of a layer in time
  order; a **pass** `p` = 0..2 is a repetition of the 12 core layers. Maps in 3.2.
- Tensor names are the pack's (`model.llm.layers.N.attn.wq_du`, ...). HF names are in [HFX] `conversion_mapping:301-416,
  2654-2708`.
- `rmsnorm_w(x, w) = bf16( fp32(x) * rsqrt(mean(fp32(x)^2) + eps) * fp32(w) )` ([HFL] `modeling:731-748`). The
  weightless `rmsnorm_plain(x) = x * rsqrt(mean(x^2) + eps)` stays in fp32 ([HFL] `modeling:67-76`).
- `l2clip(x, c) = x * min(c / (||x||_2 + 1e-6), 1)` in fp32 ([HFL] `modeling:176-187`). `softcap(x, c) = c * tanh(x / c)`.
- **Dense BF16 linears** (router, `attn_sink_proj`, controller `down`): `y = bf16( sum_k x_k w_k )` with fp32 accumulate; the
  summation order is not specified by HFL (UNKNOWN-3).
- **Dense EXL3 linears** (everything else dense): the tensor stores `trellis` (`I16 [in/16, out/16, 16K]`), `suh` (F16,
  length `in`), `svh` (F16, length `out`) and `mul1` (I32 scalar, the codebook id). The linear computes
  `y = ((x (*) suh) H_in) W_q H_out (*) svh` with `W_q` the exact decoded codebook matrix, in the form of `gb-exl3`
  `forward` that SMALL `docs/NUMERICS.md` (U11) uses; `oracle/exl3.py` of the Small decodes `mul1`, `mcg` and `3inst`
  and the oracle lane verifies it on real tensors of this pack (UNKNOWN-16). Output rounding is per op (3.4-3.9).
- **TP1**: one device holds everything. The Small's all-reduce rounding points disappear: the `wo` output and the MoE
  output are each rounded once to bf16 (HFL's `o_proj` and `experts` output dtype). There is no vocabulary sharding:
  the argmax runs over `[0, 200058)` directly.
- Activation dtypes follow HFL: the residual stream `P` and the completed blocks `B` are bf16; AttnRes mixes, controller
  blends, router, convs, clips and softmax are fp32 internally ([HFL] `modeling:1176-1200`, the `_keep_in_fp32_modules_strict`
  list: `e_score_correction_bias`, `mlp.gate.weight`, the output convs, `attention_input`, `mlp_input`, `final_input`,
  `message_input`, `message_gate_proj`, `forget_gate_proj`).

## 3 Target text model, op by op

HFL is the authority. Where HFL is silent about a rounding point or VLM rounds differently, the item is an UNKNOWN
(section 9) and `docs/NUMERICS.md` fixes the oracle's choice. Ops identical to the Small's are described again in short form
(the formulas are what kernels need), with the Small's section named.

### 3.0 Layer map (24 layers, 48 visits)

Layer type: global iff `(l + 1) % 6 == 0`, else sliding ([HFL] `config:150-153`, the `layer_types` default; the pack's
`local_layer_ids` lists the same 20 sliding layers). A block boundary is a layer with `l % 6 == 0`
([HFL] `modeling:861, 918, 1005, 1070`).

| Layers | Type | Region | AttnRes queries at the attention input | Boundary | Visits (v) |
|---|---|---|---|---|---|
| 0 | sliding | prelude | none: `rmsnorm_w(P, attn_norm)` | yes | 0 |
| 1-4 | sliding | prelude | 1 (`attn_queries.{l-1}`) | no | 1-4 |
| 5 | global | prelude | 3 (q, k, v rows) | no | 5 |
| 6 | sliding | core | 1 | yes | 6, 18, 30 |
| 7-10 | sliding | core | 1 | no | 7-10, 19-22, 31-34 |
| 11 | global | core | 3 | no | 11, 23, 35 |
| 12 | sliding | core | 1 | yes | 12, 24, 36 |
| 13-16 | sliding | core | 1 | no | 13-16, 25-28, 37-40 |
| 17 | global | core | 3 | no | 17, 29, 41 |
| 18 | sliding | coda | 1 | yes | 42 |
| 19-22 | sliding | coda | 1 | no | 43-46 |
| 23 | global | coda | 3 | no | 47 |

Every layer has attention and a MoE: there is no dense layer and no shared expert. All 24 layers share the same op
structure; they differ in the head of the attention (global: three input streams, dynamic K/V conv, log scaling; sliding:
one stream, static K/V conv) and in the loop controllers (core layers only). Sources: A.2, A.5, A.6; [HFL]
`modeling:856-1150` (the four layer classes `SlidingDecoderLayer`, `SlidingLoopDecoderLayer`, `FullDecoderLayer`,
`FullLoopDecoderLayer`).

### 3.1 Embedding and embed_norm

| | |
|---|---|
| Formula | `e_t = rmsnorm_w(embed[id_t], embed_norm.weight)`; bf16 output ([HFL] `modeling:1240-1247`, `Inkling2LoopNormedEmbedding`). Image placeholder positions (`id = 200054`) are **overwritten** by the vision embeddings after this, with no further norm ([HFL] `modeling:1591-1610, 1694`) |
| Shapes | id -> `[2304]` bf16. TP1: a plain row gather; ids are below 201,024 |
| Tensors | `model.llm.embed.weight` BF16 `[201024, 2304]`; `model.llm.embed_norm.weight` BF16 `[2304]` |

Same as SMALL 3.1 with H = 2304 and no vocabulary shard.

### 3.2 The 48-visit schedule and the AttnRes state machine

The whole stream is a state machine over two objects: the list `B` of **completed blocks** (bf16 `[T, H]` each) and the
**partial** `P` (bf16 `[T, H]`), the running sum since the last block boundary. Initially `B = []`, `P = e`
([HFL] `modeling:1358-1360`). The Small's version is SMALL 3.2; the Mini adds the loop control flow below.

**Id maps** (checked by A.2):

```
visit v (0..47)  ->  layer l, pass p
  v < 6          :  l = v,                    p = 0          # prelude
  6 <= v < 42    :  p = (v - 6) // 12,  l = 6 + (v - 6) % 12 # core, three passes
  v >= 42        :  l = v - 24,               p = 0          # coda (layers 18..23)
layer l, pass p  ->  visit v:  l < 6: l;  6 <= l < 18: l + 12 p;  l >= 18: l + 24
global visit     :  v % 6 == 5, i.e. v in {5, 11, 17, 23, 29, 35, 41, 47};  global ordinal g = v // 6  (0..7)
sliding visit    :  v % 6 != 5;  sliding ordinal s = v - v // 6  (0..39)
HF cache layer   :  l for p = 0;  24 + (p - 1) * 12 + (l - 6) for core layers with p >= 1
block boundary   :  l % 6 == 0, i.e. v in {0, 6, 12, 18, 24, 30, 36, 42}
```

The three visits of a core layer are `v, v + 12, v + 24`. A layer's weights, the queries `attn_queries.{l-1}` /
`mlp_queries.{l}` and the controller `controllers.*` of the layer are shared by its visits; the KV (ring or pages), the
sliding ring and the four conv streams are **per visit** ([HFL] `modeling:1354-1358, 1404-1405`: the cache is extended by
two copies of the 12 core cache layers, and the pass `p >= 1` uses `cache_offset = 24 + (p - 1) * 12 - 6`).
[VLM] `n/loop.py:342-346` writes the same visit order and `n/configs.py` `layer_visit_ids` the same map.

**Per-visit step** `visit(l, p, B, P)`, the layer body of [HFL] `modeling:937-997, 1091-1150` (and 876-910, 1022-1062 for
the passes without messages). `mix(B, partial, query)` is the AttnRes mix of 3.3; `blend` is the controller of 3.4;
`msg` is `None` for `p = 0`.

```
P_in = P                                                    # bf16
# --- attention inputs --------------------------------------------------------------------
if l == 0:             x_q = x_k = x_v = rmsnorm_w(P_in, attn_norm)       # no mix at layer 0
elif sliding:          d = mix(B, P_in, attn_queries[l-1])                 # one fp32 mix [T, H]
                       if msg is None: x_q = x_k = x_v = rmsnorm_w(d, attn_norm)
                       else:           x_s = rmsnorm_w(blend_s(d, msg_s), attn_norm)  for s in (q, k, v)   # same norm weights
else (global):         d_q, d_k, d_v = mix(B, P_in, attn_queries[l-1][0..2])  # three fp32 mixes
                       if msg is not None:  d_s = blend_s(d_s, msg_s)  for s in (q, k, v)
                       x_q, x_k, x_v = rmsnorm_w(d_q, q_input_norm), rmsnorm_w(d_k, k_input_norm), rmsnorm_w(d_v, v_input_norm)
A  = attention(x_q, x_k, x_v)         # 3.5-3.9, bf16 [T, H]; uses and updates this visit's KV and K/V conv state
a  = attn_sconv(A)                    # 3.6, bf16; ATTN conv stream of this visit
# --- mlp input ---------------------------------------------------------------------------
if l % 6 == 0:  B = B + [P_in];  partial = fp32(a)                        # boundary: the old partial becomes a block
else:           partial = fp32(P_in) + fp32(a)                              # fp32, NOT rounded to bf16
d = mix(B, partial, mlp_queries[l])                                         # fp32 [T, H]
if msg is not None:  d = blend_mlp(d, msg_mlp)
y = rmsnorm_w(d, mlp_norm)                                                   # bf16
m = mlp_sconv(MoE(y))                 # 3.10, 3.6: bf16; MLP conv stream of this visit
P = bf16( partial + fp32(m) )         # one rounding
```

A block boundary at the visit appends `P_in` (the layer's input) to `B` *before* the layer's MLP mix, and the layer
output starts a new partial (`P = bf16(a + m)`). Boundary layers therefore have no residual add of `P_in`: it lives on in
`B`. This is the rule of SMALL 3.2 (there with 48 layers of weights and no loop).

**Model-level control flow** ([HFL] `modeling:1308-1449`), the loop:

```
B = [];  P = e
for v in 0..5:  (B, P) = visit(layer v, pass 0, B, P)                 # prelude: layers 0-5;  now B = [e]  (layer 0 appended e)
(B0, P0) = (B, P);  n0 = len(B0)                                     # n0 = 1
for p in 0..2:                                                        # three passes over layers 6..17
    if p >= 1:  bank = (B[n0 + 1:], P)                                # the previous pass's blocks after its own first one, and its final P
                (B, P) = (B0, P0)                                     # EVERY pass restarts from the prelude's state
                msgs = message_mix(bank)                              # 3.4: per loop layer j = l - 6, four fp32 messages
    for l in 6..17: (B, P) = visit(layer l, pass p, B, P, msg = None if p == 0 else msgs[l - 6])
for l in 18..23: (B, P) = visit(layer l, pass 0, B, P)                 # coda continues from the LAST pass's (B, P)
h = bf16( rmsnorm_w( mix(B, P, output_query), model.llm.norm ) )       # 3.12
```

Facts that follow (all checked in A.2 or by reading the pseudo-code against HFL):

- After the prelude `B = [e]`. Layer 6 appends the prelude's `P` (every pass appends the identical value) and layer 12 appends
  that pass's `P_in(12)`, so a pass ends with `B = [e, P0, P_in(12)]` and `bank = ([P_in(12)], P_out(17))`: **one block plus the
  partial**, hence 2 message sources. HFL's `num_pre_loop_blocks + 1` skips the shared `P0` ([HFL] `modeling:1399-1400`).
- Messages of pass `p` come from pass `p - 1` only (the third pass, `p = 2`, reads the second, never the first).
- The coda's first boundary (layer 18) appends `P_in(18)`, so at the end `B = [e, P0, P_in(12) of the third pass, P_in(18)]`.
  The AttnRes source count is at most **5** (4 blocks + the partial: at layer 18's MLP mix and at the final mix); the Small's
  was at most 9. Sources per mix (A.13): the attention-input mix has 2 sources at layers 1-6, 3 at layers 7-12, 4 at layers 13-18 and 5 at layers 19-23 (blocks `[e]`, `[e, P0]`,
  `[e, P0, P_in(12)]`, `[e, P0, P_in(12), P_in(18)]`, plus `P`); the MLP-input mix has the same count, one more at the boundary layers 6, 12 and 18.
- The mix at layer 6 of every pass has the same inputs (`B0`, `P0`) before blending: the pre-blend `d` of the first
  core layer is identical in the three passes and can be computed once ([VLM] `n/loop.py:359-374`, `prelude_dests`). After the
  blend the passes diverge.
- `P` and every `B` entry are bf16 values, so the stack can live in bf16 `[5, T, H]` plus the fp32 `partial` of the MLP mix.
- The final mix at the end of the coda ignores which pass produced `P`: it sees the third pass only.
- **Dumping `P` per visit does not capture the bank.** `P` entering visit 18 or 30 (the first core layer of the second and third pass) is the prelude's `P`, not the previous pass's output.
  A dump must therefore also keep `P` leaving layer 17 of the first and second pass (`p = 0, 1`: the bank partial, the oracle's `core_out`, shape `[2, T, H]`); the output of the last core pass is the
  `P` entering visit 42. The drafter taps are inputs of visits 5, 23 and 41, none of which is affected.

**Per-visit state** (all keyed by visit `v`; the schedule above says which weights run):

| State | Sliding visit (40) | Global visit (8) |
|---|---|---|
| K, V | ring of the last 1,031 positions (6.2), bf16 | global pages: all positions, int8 + exponent per (token, KV head) or bf16 |
| conv streams K, V, ATTN, MLP | 3 raw inputs each (1024, 1024, 2304, 2304 wide) | same widths |

### 3.3 AttnRes mix (block attention residual)

Used for the attention input(s), the MLP input, the final output, and (with 4 queries) the loop messages (3.4). Per token and
per query ([HFL] `modeling:828-853, 145-173`; the Small's op is SMALL 3.2 "Mix op"):

```
sources x_n = [B_0 .. B_{nb-1}, partial]                       # fp32; `partial` is P (bf16 value) or P+a (fp32, unrounded)
xn_n   = x_n * rsqrt(mean(x_n^2) + 1e-6)                         # weightless RMSNorm, fp32 (also the VALUES: normalized sources)
l_n    = (q . xn_n) * H^-0.5                                     # q = the learned fp32 pseudo-query, H = 2304
p_n    = softmax_n(l_n)                                          # fp32
out    = sum_n p_n * xn_n                                        # fp32 [H]
```

The output stays fp32 until the consumer (the next RMSNorm with gamma, or a controller blend) rounds it. `attn_res_normalize_values
= true` is built into HFL (values are the normalized states, [HFL] `modeling:838-853`); [VLM] `n/configs.py` rejects a
checkpoint without it for loop models. Number of sources: 1 + `len(B)`, at most 5 (3.2).

| Use | Query tensor | Shape | Notes |
|---|---|---|---|
| attention input, sliding layer `l >= 1` | `model.llm.attn_res.attn_queries.{l-1}` | F32 `[2304]` | The checkpoint index is one below the layer ([HFX] `core_model_loading:1266-1290`, `offset=1`): 19 tensors for the sliding layers other than layer 0, and the 4 global layers below (A.6) |
| attention input, global layer (5, 11, 17, 23) | `attn_queries.{4, 10, 16, 22}` | F32 `[3, 2304]` | rows (q, k, v) give three mixes over the same sources |
| MLP input, layer `l` | `model.llm.attn_res.mlp_queries.{l}` | F32 `[2304]` | 24 tensors |
| final output | `model.llm.attn_res.output_query` | F32 `[2304]` | followed by `model.llm.norm` |
| loop messages | `model.llm.middle_cycle_cross_pass_queries`, `..._kv_queries` | F32 `[24, 2304]` each | 3.4 |

HFL scales the logits by `H^-0.5` inside attention; VLM pre-scales the query once at load (`q' = q * H^-0.5`). The two differ
at the ulp level (UNKNOWN-10). There is no `1/sqrt` pre-norm: the mix output feeds `rmsnorm_w` (3.2).

### 3.4 Loop messages and controllers (new in the Mini)

Active only for passes `p >= 1` (visits 18-41), on the 12 core layers `l = 6..17`, `j = l - 6`.

**Message mix** ([HFL] `modeling:145-173, 1397-1409`). Once per pass (the bank is constant within the pass) and per loop layer
`j`, four queries `Q_j = (q_q, q_k, q_v, q_mlp)` read the bank `[bank_blocks..., bank_partial]` (2 sources, 3.2) with the
mix of 3.3, giving four fp32 messages `msg_q, msg_k, msg_v, msg_mlp` of `[T, H]`.
`message_input.query` has shape `[12, 4, 2304]` in HF; the pack stores two `[24, 2304]` tensors and [HFX]
`core_model_loading:823-853` merges them as

```
queries    = middle_cycle_cross_pass_queries.view(12, 2, 2304)      # [:, 0] = q stream, [:, 1] = mlp stream
kv_queries = middle_cycle_cross_pass_kv_queries.view(12, 2, 2304)   # [:, 0] = k stream, [:, 1] = v stream
Q_j = (queries[j, 0], kv_queries[j, 0], kv_queries[j, 1], queries[j, 1])    # order q, k, v, mlp
```

[VLM] `n/loop.py:170-179` does the same (`(attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1])`) and pre-scales by `H^-0.5`.

**Controller** (`Inkling2LoopMessageGate`, [HFL] `modeling:79-96`; [VLM] `n/loop.py:38-64`), one per stream `s` of every core layer:

```
n      = bf16( [ rmsnorm_plain(dest), rmsnorm_plain(msg) ] )          # fp32 norms, concatenated [4608], rounded once to bf16
hdn    = bf16( n . down_s^T )                                         # down_s: [64, 4608] bf16 -> [64]
gain   = fp32( hdn . gain_s^T )                                       # gain_s: [2304, 64]
forget = fp32( hdn . forget_s^T )                                     # forget_s: [2304, 64]
out    = (1 - tanh(forget)) * dest + tanh(gain) * msg                 # fp32 [2304]; no rounding
```

`dest` is the fp32, unrounded AttnRes mix of the stream (3.2). Pack names (A.5) map to HF names by [HFX]
`conversion_mapping:2684-2700`: `controllers.attn` -> `q_message_gate`, `controllers.k` -> `k_message_gate`, `controllers.v` ->
`v_message_gate`, `controllers.mlp` -> `mlp_message_gate`; `down_weight` -> `down_proj`, `gain_weight` -> `message_gate_proj`
(the `tanh(.) * msg` branch), `forget_weight` -> `forget_gate_proj` (the `(1 - tanh(.)) * dest` branch). The stream called
`attn` in the pack is the **query** stream.

For a sliding core layer the one mix `d` is blended three times (q, k, v streams with their own messages and controllers);
the single `attn_norm` then normalizes all three results, so a sliding visit with messages has three different attention inputs
`x_q`, `x_k`, `x_v` where the Small, and pass 0, have one. The MLP stream blends the MLP mix before `mlp_norm`.
The blend output feeds the layer's RMSNorm in the same kernel in VLM (`blend_rms_norm`, [VLM] `n/ops/loop.py:65-132`), unfused
and without FMA ("as the reference").

Cost per pass-visit (core layer, `p >= 1`): 4 pair-norms, 4 `down` GEMVs `[64, 4608]` (bf16), 4 `gain`/`forget` GEMV pairs
`[2304, 64]`, plus one message mix per pass and layer. Controller weights: 12 x 4 x (64 x 4608 + 2 x 2304 x 64) x 2 B =
56,623,104 B (0.053 GiB, the "loop controllers" line of `scripts/roofline.py`), read once in each of the second and third pass.

### 3.5 Q / K / V / R / gate / sink projections (TP1)

Inputs: sliding layers use `x_q`, `x_k`, `x_v` of 3.2 (equal for `l = 0` and for `p = 0`; three different tensors for core
layers with messages); global layers use their three streams ([HFL] `modeling:575-636, 668-728`).

| Output | Formula | Per token | Tensor (EXL3 K = 6 unless stated) |
|---|---|---|---|
| q | `x_q Wq^T` | `[24, 128]` | `attn.wq_du` in 2304, out 3072 |
| r | `x_q Wr^T` | `[24, 16]` | `attn.wr_du` in 2304, out 384 |
| gate | `x_q Wg^T` | `[3072]` | `attn.gate_attn` in 2304, out 3072 |
| sink | `x_q Ws^T` | `[24]` | `attn.attn_sink_proj` **BF16** `[24, 2304]` (not EXL3) |
| k_raw | `x_k Wk^T` | `[8, 128]` | `attn.wk_dv` in 2304, out 1024 |
| v_raw | `x_v Wv^T` | `[8, 128]` | `attn.wv_dv` in 2304, out 1024 |

All outputs bf16. HF names (`q_proj.linear`, `relative_position_bias.linear`, `o_proj.gate_proj`, `sink_proj`, `k_proj.linear`,
`v_proj.linear`) are in [HFX] `conversion_mapping:353-416`. Each EXL3 tensor has its own `suh`: five EXL3 matmuls share `x_q`
(or `x_k`, `x_v`) but rotate it differently.

### 3.6 Short convolutions (W = 4) and their state

**Static conv** (sliding `k_sconv`/`v_sconv` on 1024 channels; `attn_sconv`/`mlp_sconv` on every layer, 2304 channels)
([HFL] `modeling:329-393`):

```
y_t[c] = bf16( ( sum_{k=0..3} w[c, k] * x_{t-3+k}[c]  +  x_t[c] ) / sqrt(5) )     # tap 3 = the current token; x_{<0} = 0; fp32 math
```

Inputs are bf16 values (the K/V projections, the `wo` output and the MoE output are all bf16), weights BF16 `[C, 1, 4]`
(HF holds them in fp32; the upcast is exact). `attn_sconv.weight` / `mlp_sconv.weight` `[2304, 1, 4]`; `k_sconv.weight` /
`v_sconv.weight` `[1024, 1, 4]` on the 20 sliding layers.

**Dynamic conv** (global layers' `k_sconv`/`v_sconv`) ([HFL] `modeling:396-455`; SMALL 3.4):

```
dyn_t[h, l] = fp32( x_in_t . weight_proj^T )[h * 4 + l]            # x_in = the k (or v) stream's normalized input [2304]; [32, 2304] weight, 8 heads x 4 lags
y_t[c] = bf16( ( sum_{l=0..3} (static_w[l, c] + dyn_t[head(c), l]) * x_{t-l}[c]  +  x_t[c] ) / sqrt(9) )     # l = backward distance; head(c) = c // 128
```

Tensors `attn.{k,v}_sconv.static_w` BF16 `[4, 1024]`, `attn.{k,v}_sconv.weight_proj` BF16 `[32, 2304]` on layers 5, 11, 17, 23.
The dynamic taps depend on the *current* token's stream input, applied to its own window. HF computes `weight_proj` as a bf16
`nn.Linear` and takes `.float()` (so `dyn` is rounded to bf16 first: UNKNOWN-3 covers the order, the bf16 store is HFL's).

**State per visit**: the last 3 raw conv inputs of each stream: K (`k_raw`, 1024), V (`v_raw`, 1024), ATTN (the `wo` output, 2304),
MLP (the MoE output, 2304). HF allocates 4 (`conv_kernel_size`) and never reads the oldest
([HFL] `modeling:286-303`, [HFL] `modeling:430-440` read `-(seq_len + 3)` positions); the oracle needs 3. A speculative window of 8
tokens needs the 3 positions before its first token ([PR181] states the same: "a later step only reads the 3 positions before its first
token"); a rollback restores to a position that has its 3 predecessors.

### 3.7 K and V post-processing, Q and R clips, log scaling

- **K**: `k̂ = bf16( l2clip( fp32(y_k), 22.627417 ) )` per KV head, no learned scale ([HFL] `modeling:176-187, 515-529`).
- **V**: `v̂ = bf16( y_v * rsqrt(mean(y_v^2) + 1e-6) )` per KV head, head_dim 128 (`use_value_norm_no_gamma`; [HFL] `modeling:67-76, 515-529`).
- **Q (sliding)**: `q̂ = bf16( l2clip( fp32(q) * fp32(scale[h]), 22.627417 ) )`, tensor `attn.q_l2_clip.scale` BF16 `[24, 128]`
  ([HFL] `modeling:190-202, 458-475`).
- **Q (global)**: the same, then `q̂' = bf16( fp32(q̂) * tau_t )` ([HFL] `modeling:223-236`, `Inkling2LoopLogScaledL2Clip`: clip, round to
  bf16, then scale and round again).
- **R (all layers)**: `r̂ = bf16( l2clip( fp32(r) * fp32(r_scale[h]), 8.0 ) )`, `attn.rel_l2_clip.scale` BF16 `[24, 16]`; on global layers
  also `r̂' = bf16( r̂ * tau_t )` ([HFL] `modeling:239-270`).
- **tau** (global layers only): `tau_t = 1 + 0.1 * ln( max( (pos_t + 1) / 8192, 1 ) )` in fp32 ([HFL] `modeling:205-220`). It is exactly 1 for
  `pos_t <= 8191` and grows slowly (A.14): tau = 1.0693 at pos 16,383, 1.2773 at 131,071 and 1.4852 at 1,048,575 (fp32 evaluation of the formula; the
  Small's floor was 128,000). K is not scaled. The positions are absolute and identical in all three passes.

### 3.8 Relative-position logits

```
bias_t[h, d] = bf16( sum_{j<16} r̂'_t[h, j] * rel_proj[j, d] )    for d = pos_t - pos_s in [0, 1024)
             = 0                                                 for d >= 1024  (and for d < 0; the causal mask removes those keys)
```

([HFL] `modeling:239-270`; [VLM] `n/ops/sconv.py` `shift_rel_proj` is the same table). Tensor `attn.rel_logits_proj.proj` BF16 `[16, 1024]`
per layer, shared by the visits of the layer. Sliding layers use `rel_extent = sliding_window_size`, which is 1024 as well
([HFL] `modeling:550-554`). No per-key state: the bias needs only the query's `r̂'` and integer distances. HF computes the table as a
bf16 matmul (bf16 output). Work per query token `[24, 16] x [16, 1024]`.

### 3.9 Attention core: masks, sinks, gate, output projection

Per query head `h` (24) over KV head `h // 3` (8 KV heads, GQA 3), D = 128, scale `128^-0.5` ([HFL] `modeling:111-142, 546-636, 639-728`):

```
s_{t,u} = q̂'_t . k̂_u * 128^-0.5 + bias_t[h, pos_t - pos_u]
visible:   0 <= pos_t - pos_u            (global)
           0 <= pos_t - pos_u < 1024     (sliding: 1024 keys including the current)
softmax over {visible s_{t,u}} U {sink_t[h]}         # one extra logit per (token, head), value 0, no scale, no bias
o_t[h] = sum_u P_{t,u} v̂_u                           # P excludes the sink column
out_t[h, :] = bf16( o_t[h, :] * sigmoid(gate_t[h, :]) )     # gate = gate_attn(x_q), per element
```

With the sink inside the softmax, the probability left for the keys is `sum_u P_{t,u} = e^{lse} / (e^{lse} + e^{sink}) = sigmoid(lse - sink)`,
the form the Small writes as `o * sigma(lse - sink)` (SMALL 3.11). Output projection `wo_ud` (EXL3, in 3072, out 2304):
`A = bf16( out Wo^T )`; no all-reduce at TP1; then `attn_sconv` (3.6).

HFL computes the logits in bf16 (`matmul` output, the scale multiply, the bias add each round to bf16), the softmax in fp32, rounds `P`
to bf16 for `P V`, rounds `o` to bf16 and multiplies the sigmoid gate in fp32 (`GatedOutputProjection`, [HFL] `modeling:531-543`). VLM keeps fp32
logits with FA4's online softmax and bf16 `P`. These differ (UNKNOWN-5).

### 3.10 MoE: router and experts

**Router** ([HFL] `modeling:1152-1173`; one per layer, dense, fp32):

```
logit_e = fp32(y) . fp32(Wgate[e])                     # y = bf16 mlp input [2304]; Wgate BF16 [128, 2304]; bias F32 [128]
sc_e    = sigmoid(logit_e)
sel     = top8_e( sc_e + bias_e )                       # `topk(sorted=False)`: ties resolve by the framework (UNKNOWN-18)
w_j     = 8 * sc_j / ( sum_{j in sel} sc_j + 1e-20 )    # normalized over the 8 selected, times route_scale
```

Tensors `layers.l.mlp.gate.weight` BF16 `[128, 2304]` and `.bias` F32 `[128]` (HF name `e_score_correction_bias`).

**Experts** (`Inkling2LoopExperts`, [HFL] `modeling:760-812`). For each selected `(token, expert e)`:

```
g = EXL3(gate_proj_e, y) ;  u = EXL3(up_proj_e, y)          # [1152] each; K per tensor: 4 or 5 (A.3)
a = softcap(g, 4) * sigmoid(g) * softcap(u, 25)              # fp32; sigmoid sees the RAW gate
a' = bf16(a)
z_j = EXL3(down_proj_e, a')                                  # [2304]; K 4 or 5
out = bf16( sum_j w_j * z_j )                                # accumulated in fp32 (HF: ascending expert id), one rounding
```

The routing weight multiplies the expert output after `down` ([HFL] `modeling:787-805`). VLM folds it into the activation before `down` with a bf16 rounding
(UNKNOWN-2). Bounds: `|softcap(g, 4) * sigmoid(g)| < 4`, `|softcap(u, 25)| < 25`, so `|a| < 100`, `|a w| < 800`: fp16 intermediates are safe.
Experts are three EXL3 tensors each (`mlp.experts.{e}.{gate,up,down}_proj` = `trellis`, `suh`, `svh`, `mul1`), shapes in A.3; there is no fused
`gate_up`; `gate_proj` is the gate (UNKNOWN-15). Layers 6-17 mix the classes (4,4,4), (4,4,5), (5,5,5) (A.3), so a grouped kernel must pick `K` per tensor and
`down` may be wider than `gate`/`up` for the same expert.

### 3.11 Layer residual update

Folded into the visit step of 3.2 (`P = bf16(partial + m)`): the MLP's short conv output `m` is added to the partial, which is
`fp32(a)` on a boundary layer and `fp32(P_in) + fp32(a)` otherwise ([HFL] `modeling:898-911`).

### 3.12 Final mix, norm, unembed, soft cap

```
h      = bf16( rmsnorm_w( mix(B, P, output_query), model.llm.norm ) )     # after the coda; 5 sources at most
z      = EXL3(unembed, h)  -> fp32  [201088]                               # in 2304, out 201088 (padded), K = 6
logits = 30 * tanh(z / 30)    fp32                                         # soft cap
logits = logits[:200058]                                                   # unpadded vocab; rows 200058..201087 are padding
```

([HFL] `modeling:1442-1443, 1534-1535`; padded rows dropped by `Inkling2UnpadVocab`, [HFX] `core_model_loading:856-880`.) Only the first
200,058 outputs are needed; with 16-wide trellis tiles that is 12,504 of 12,568 output tiles (200,064 columns). HF keeps the output of
`lm_head` in bf16 before the soft cap's upcast; VLM keeps fp32 (its config comment: the unembedding "runs in BF16 with an FP32 output", UNKNOWN-6). `unembed` is **not tied** to `embed`.
Greedy sampling takes the first maximum over `[0, 200058)`.

## 4 Block drafter

The drafter is `mtp.safetensors` of the pack (141 tensors, 236,558,740 B of tensor data, 0.220 GiB; A.7, A.11). It is a
DSpark-style block drafter: an anchor plus mask block runs through a small dense transformer over target features, then a recurrent
Markov head. HFL has no drafter; its semantics come from VLM (`n/dspark.py`, `n/ops/dspark.py`, `speculator.py`, `configs.py`), as in
the Small (SMALL section 4, which this section follows op for op with the Mini's dimensions).

Config (`mtp_config`, A.11): 3 layers, H = 2304, 24 query / 8 KV heads, head_dim 128, MLP 9216 (dense SiLU), rope theta 1e6, local
layers {0, 2} with window 1024, **layer 1 global**, `block_size` 8, `markov_rank` 256, `markov_head_type = rnn_hidden`,
`target_layer_ids` [5, 23, 41, 48]. `q_bias` and `o_bias` are false ([VLM] `n/configs.py:267-300` rejects anything else). The drafter
shares the target's embedding table, `embed_norm` and unembed: it has none of its own.

### 4.1 Target features consumed (the taps)

`target_layer_ids` are **visit** indices (3.2) of the 48-visit schedule. For each id below 48 the feature is the **input partial `P`
(bf16 `[2304]`) of that visit, before the visit's AttnRes mix**; the last id, 48 = the number of visits, is the **final-norm output `h`** of
3.12 ([VLM] `n/loop.py:375-376, 399-401`, `n/configs.py:255-262, 280-292`). The four features are:

| Tap | Meaning |
|---|---|
| 5 | `P` entering visit 5 = layer 5, pass 0 (global, last prelude layer) |
| 23 | `P` entering visit 23 = layer 11, pass 1 (the second pass; global) |
| 41 | `P` entering visit 41 = layer 17, pass 2 (the third pass; global) |
| 48 | the final-norm output `h` (what the unembed reads) |

(A.2 prints the map.) All four taps sit at global visits (5, 23, 41 are visits with v % 6 == 5) or at the output. They are captured
for every committed token (and, in the verify forward, for the 8 query rows) and stored as `[T, 4 x 2304]` bf16: 18,432 B per token.
The Small's taps were layer indices below 48; the Mini's index the 48 passes and must be validated against 48, not against the 24 layers.

### 4.2 Feature fusion

`alpha = softmax_fp32(fuse_weight)` over the 4 taps; `fuse_weight` is BF16 `[4, 4]`: rows 0-2 belong to draft layers 0-2, row 3 to the Markov head
([VLM] `n/dspark.py:388-440, 497-535`). `ctx_l = bf16( rmsnorm_w( sum_i alpha[l, i] * f_i , layers.l.dflare_norm ) )`, fp32 math
([VLM] `n/ops/dspark.py:40-67`). Tensors: `layers.l.dflare_norm.weight` BF16 `[2304]`, `markov_head.dflare_norm.weight` BF16 `[2304]`.

### 4.3 Context K/V (what is cached)

Per draft layer `l` and committed token ([VLM] `n/dspark.py:249-266, 517-531`):
`k = rope_neox( rmsnorm_w( ctx_l Wk_ctx^T per head, k_norm_ctx ), pos )`, `v = ctx_l Wv_ctx^T` (no v norm), written to the layer's own paged KV at the token's slot.
Tensors: `layers.l.self_attn.{k_proj_ctx, v_proj_ctx}` EXL3 K = 4 (in 2304, out 1024), `k_norm_ctx.weight` BF16 `[128]`.
Retention: the local layers 0 and 2 keep the last 1,024 positions before the block start plus the 8 block positions (window 1,024 + 8 =
1,032 keys, [VLM] `n/dspark.py:173`); layer 1 keeps the full context at 4,096 B/token (bf16).
The global layer does **not** read the target's K/V: its keys and values are projections of the fused features above, held by the draft layer's own `Attention` cache ([VLM] `n/dspark.py:118-166`).
In the engine they are one more plane (the ninth) of the same page pool as the target's 8 global visits, priced in the target's int8 page format in `docs/FIT.md` and in the execution plan step 4.6 ("on the target's int8 pages").

### 4.4 Block drafting (block 8 -> 7 drafted tokens)

The scheduler needs `num_speculative_tokens = block_size - 1 = 7` ([VLM] `n/dspark.py:467-473`, [PR249] runs the Mini with 7). The block is an anchor plus
7 masks, each predicting the token at its own position (`sample_from_anchor = False`).

1. Block input `[8, 2304]` bf16: row 0 = `rmsnorm_w(embed[anchor], embed_norm)` (the target's normed embedding); rows 1-7 = `mask_embed` BF16 `[2304]`
   ([VLM] `n/dspark.py:424-431`). The anchor (last accepted or bonus token) is at position `p`; the masks sit at `p+1 .. p+7`.
2. Per layer (pre-norm, [VLM] `n/dspark.py:118-318`):
   - layer 0: `residual = x; x = rmsnorm_w(x, input_layernorm)`; later layers `residual = bf16(residual + x_prev)` and `x = rmsnorm_w(residual, input_layernorm)`
     (UNKNOWN-21: VLM's single-GPU path norms the unrounded sum instead).
   - `q, k, v = x Wq^T, x Wk^T, x Wv^T` (EXL3 K = 4: `q_proj` out 3072, `k_proj`/`v_proj` out 1024). Per head `q = rope( rmsnorm_w(q, q_norm) )`,
     `k = rope( rmsnorm_w(k, k_norm) )`: norm output rounded to bf16, RoPE in fp32, rounded to bf16 ([VLM] `n/ops/dspark.py:116-175`). The block K/V are written at slots `p .. p+7`.
   - Attention is **non-causal within the block**: each block query sees the context keys (local layers: positions >= p - 1024; global layer 1: all) plus
     all 8 block keys. Scale `128^-0.5`, GQA 3, no bias, no sink, no gate.
   - `o_proj` (EXL3 K = 4, in 3072, out 2304); `residual = bf16(residual + o)`; `x = rmsnorm_w(residual, post_attention_layernorm)`.
   - MLP: `g, u = x Wg^T, x Wu^T` (EXL3 K = 4, out 9216 each), `a = bf16( silu(g) * u )` with precise exp and division in fp32, `down_proj` (in 9216, out 2304);
     `residual = bf16(residual + mlp)` ([VLM] `n/ops/dspark.py:176-200`).
   - Final: `hd = rmsnorm_w(residual, norm)`.
   - No all-reduces at TP1 (the Small's per-layer all-reduce rounding points vanish).
3. **Markov head** (`rnn_hidden`, rank 256) runs sequentially over the 7 mask outputs `hd_i`, i = 1..7 ([VLM] `n/dspark.py:321-385, 544-570`,
   `n/ops/dspark.py:202-254`). Identical to SMALL 4.4 step 3 with H = 2304:
   ```
   f_rnn   = bf16( rmsnorm_w( sum_i alpha[3, i] f_i(last committed token), markov_head.dflare_norm ) )
   ctx_i   = fp32( rnn_bias + [hd_i, f_rnn] . Wrnn[:, 512:]^T )          # once for all i;  Wrnn = rnn_proj [768, 5120]
   base_i  = bf16( hd_i . unembed^T )                                       # bf16 GEMM, bf16 output over the 201,024 vocabulary rows
   state_0 = 0 (bf16 [256]);  prev_0 = anchor id
   for i in 1..7:
     raw   = fp32( ctx_i + [state, markov_w1(prev)] . Wrnn[:, :512]^T )     # [768]
     g = bf16 sigmoid(bf16 raw[0:256]);  c = bf16 tanh(bf16 raw[256:512]);  o = bf16 raw[512:768]
     state = bf16( bf16(g * state) + bf16( bf16(1 - g) * c ) );   latent = bf16 tanh(o)
     bias  = bf16( fold_b + latent . fold_W^T )                             # fold_W = bf16(unembed . hidden_out.W), fold_b = bf16(unembed . hidden_out.b)
     logit = bf16( 30 * tanh( bf16(base_i + bias) / 30 ) )
     prev  = draft_i = argmax_{v < 200058} logit        # NaN -> -inf, first maximum
   ```
   Tensors: `markov_head.rnn_proj` EXL3 K = 4 (in 5120, out 768) + BF16 bias `[768]`, input columns `[state 256 | markov_w1(prev) 256 | hidden 2304 | f_rnn 2304]`,
   output `[gate | cand | out]`; `markov_head.markov_w1.weight` BF16 `[201024, 256]`; `markov_head.hidden_out` EXL3 K = 4 (in 256, out 2304) + BF16 bias `[2304]`;
   `fuse_weight` `[4, 4]`; `mask_embed` `[2304]`; `norm.weight` `[2304]` (A.11).
   **What is new at the Mini**: `rnn_proj`, `hidden_out` and the unembed are EXL3, so the column split of `rnn_proj` (whole-input Hadamard rotation) and the
   `unembed . hidden_out` fold are not a product of bf16 matrices. The oracle works from the exactly decoded weights (UNKNOWN-17); the engine's choice is open.

### 4.5 Verification and acceptance

The target runs one forward over `[anchor, d1 .. d7]` (8 query rows through all 48 visits, so the 8 rows read the rings and pages of each visit). Greedy acceptance keeps the longest
prefix where the target argmax equals the draft, plus a bonus token. After acceptance the context K/V of the newly committed tokens are computed from their stored 4-tap
features. `f_rnn` is recomputed from the last committed row. **Rollback**: the KV and the three conv positions are indexed by absolute position per visit, so rejected
positions are overwritten; the conv history needs `3 + 8` positions per stream and visit if the engine keeps rollback history ([PR181]: "a later step only reads the 3
positions before its first token").

## 5 Vision tower

The Mini reuses the Small's tower unchanged except for the output width ([HFL] `modeling:1575-1580` builds the tower with `AutoModel.from_config(config.vision_config)`, the
`inkling2_vision` class of [HFS] `modeling:983-1260, 1533-1610`). Architecture (A.10): 23 blocks of width **1024**, 16 heads x 64, MLP 4096 (exact-erf GELU), LayerNorm eps 1e-5 with bias,
layer scale, a class token per image, then a 2 x 2 merge projector `proj_in` 4096 -> 4096 -> GELU -> `proj_out` 4096 -> **2304** (no bias). It runs in bf16. SMALL section 5 is the
step-by-step table; it applies with these changes:

| Step | Mini detail | Tensors (census A.10) |
|---|---|---|
| Preprocess | Decode to RGB, composite alpha on an 8-px checkerboard (204 / 153), **no resize**, pad to a multiple of 40 px with raw value -1, one 40 x 40 cell = one LM token (grid rows x cols = ceil(H/40) x ceil(W/40)); `processor_config.json`: size 40 x 40, mean/std 0.5 ([HFS] `image_processing_inkling2.py:36-110`) | `processor_config.json` |
| Pixel norm | VLM: `bf16((v/255 - clip_mean)/clip_std')` then `2 (. * clip_std + clip_mean) - 1`; HF: `(v - 127.5)/127.5` in fp32 (UNKNOWN-7). The engine's host code already implements VLM's form ([ENG] `crates/vision/src/inkling2.rs:12-14, 52-55`) | |
| Patch embed | `bf16(patch Wp^T)` `[1200 -> 1024]`, no bias. **BF16, not EXL3** (K = 1200 is not a multiple of 128; the engine pads it to 1216) | `backbone.embeddings.patch_embedding.weight` BF16 `[1024, 1200]` |
| Pos embed | bilinear (align_corners = false) from the 32 x 32 grid to the image's native-patch grid; `hidden = bf16(bf16(patch) + bf16(interp))`; the CLS row is `class_embedding + cls_position_embedding` | `position_embedding` BF16 `[1024, 1024]`, `class_embedding`, `cls_position_embedding` BF16 `[1024]` |
| ln_pre | LayerNorm, rounded to bf16 | `backbone.ln_pre.{weight, bias}` |
| Blocks x 23 | `h = bf16(h + bf16(ls_1 * Attn(LN1(h))))`; `h = bf16(h + bf16(ls_2 * MLP(LN2(h))))` ([HFS] `modeling:1181-1208`) | `ln_1`, `ln_2` (weight, bias), `ls_1.gamma`, `ls_2.gamma` BF16 `[1024]` |
| Attention | qkv with biases, **2D axial RoPE** on interleaved pairs: pairs 0-15 rotate by (x+1), pairs 16-31 by (y+1), the CLS row by 0; `inv_freq = 10000^(-4i/64)`; non-causal per image; scale `64^-0.5` ([HFS] `modeling:983-1058`; [VLM] `n/ops/vision.py:19-22, 240-290`) | `self_attn.{q,k,v,out}_proj`: EXL3 K = 6 `[64, 64, 96]` + BF16 bias `[1024]` |
| MLP | `fc2(gelu_erf(fc1(x)))` with biases | `mlp.c_fc` EXL3 K = 6 `[64, 256, 96]` + bias `[4096]`; `mlp.c_proj` `[256, 64, 96]` + bias `[1024]` |
| Merge | Drop the CLS rows; concatenate the 4 consecutive patch rows of one LM token (sub-patch order (0,0), (0,1), (1,0), (1,1)) to `[4096]`; `proj_out(gelu(proj_in(.)))` gives `[2304]` per LM token ([HFS] `modeling:1253-1270, 1602`) | `proj_in` EXL3 K = 6 `[256, 256, 96]`; `proj_out` EXL3 K = 6 `[256, 144, 96]` |

Every EXL3 vision linear has K = 6 (A.4); biases are BF16 and are added to the fp32 EXL3 result before the single bf16 rounding (3.5 applies unchanged).

**Token stream**: the renderer emits `<|content_image|>` (200005); the processor expands it to `200005` followed by `rows x cols` copies of `<|unused_200054|>` (200054); those
positions take the projector outputs **in place of** their normed text embeddings (3.1, [HFL] `modeling:1591-1610, 1694`). `vision_config.temporal_patch_size` (2) has no
effect for stills; the pack's config has no `image_token_id` (HF default 200054, A.12).

## 6 KV and state per token

Reproduces `docs/FIT.md` ("KV per token"); every number is recomputed in A.8 from the dimensions of A.1. Layers hold no state other than K/V and the four conv streams.
Because the loop gives each of the 48 visits its own cache, "layers" below are **visits**.

| Item | Visits | What | Per token | Growth |
|---|---|---|---|---|
| Target K, V, global | 8 (v % 6 == 5) | k̂, v̂: 8 KV heads x 128 x 2 (K and V) | bf16: 2 x 1024 x 2 B = 4,096 B a visit = **32,768 B**; int8 + one exponent byte per (token, KV head) for K and for V: 2 x 1024 + 2 x 8 = 2,064 B a visit = **16,512 B** | proportional to context |
| Target K, V, sliding | 40 | same, last 1,031 positions: 1,024 for the window plus 7 more because the verify rows `p .. p+7` each need 1,024 keys back | bf16 ring: 40 x 1,031 x 4,096 B = 168,919,040 B (0.1573 GiB) a slot | fixed |
| Drafter K, V, global | draft layer 1 | ctx k (RoPE'd), v | bf16 **4,096 B**; int8 2,064 B (total with the target: 36,864 B bf16, 18,576 B int8) | proportional to context |
| Drafter K, V, local | draft layers 0, 2 | last 1,024 positions plus the block of 8 = 1,032 | 2 x 1,032 x 4,096 B = 8,454,144 B a slot | fixed |
| Relative-position state | all | none (the bias is recomputed from the query's `r̂'`) | 0 | |
| Conv state | 48 visits | last 3 raw inputs of K (1024), V (1024), ATTN (2304), MLP (2304) | 3 x 6,656 x 2 B = 39,936 B a visit = **1,916,928 B** a slot (1.828 MiB); x (3+8)/3 with an 8-position rollback history | fixed |
| Target features for the drafter | | 4 x 2,304 bf16 | 18,432 B, only for tokens whose context K/V is not yet written | transient |
| AttnRes stack | | at most 4 completed blocks plus the partial, `[T, 2304]` bf16, plus an fp32 `partial` for the MLP mix | | transient per forward |

- Fixed state per slot: 168,919,040 + 8,454,144 + 1,916,928 = 179,290,112 B = 0.167 GiB (A.8); the card quotes "~0.17 GiB fixed sliding-window/convolution state"
  and 8,704 B / 9,792 B per token for its q4 KV (PACK `README.md:39`); FIT.md counts the target rings only (0.157 GiB x 4 slots = 0.63 GiB).
- A 1,048,576-token lane: int8 16.125 GiB, bf16 32.0 GiB, int8 with the drafter plane 18.141 GiB (A.8), matching FIT's 18.14 GiB.
- A global int8 page of 256 tokens holds, for one visit, `256 x 1024` B of K and of V plus 2,048 B of exponent bytes each: 528,384 B (arithmetic; the engine's Nemotron-shared header
  is hard-wired at 2 KV heads, section 8).
- **Per-visit keying**: pages and rings are owned by `(slot, visit)`; the weights are shared. A prefix-cache hit must restore the KV of all 48 visits and the conv streams of all
  48 visits at the split position (UNKNOWN-12). Messages need no state: they are recomputed from the previous pass inside the same forward.
- HF holds 4 raw inputs per conv stream (`conv_kernel_size`), the model reads 3 (3.6).

## 7 Chat template, reasoning and tool calls

Special tokens (PACK `tokenizer_config.json` `added_tokens_decoder`, A.12): `<|endoftext|>` 199999 (pad), `<|message_user|>` 200000, `<|message_model|>` 200001, `<|message_system|>` 200002,
`<|message_tool|>` 200003, `<|content_text|>` 200004, `<|content_image|>` 200005, `<|content_model_end_sampling|>` 200006 (**= `eos_token_id`**), `<|content_thinking|>` 200008,
`<|end_message|>` 200010, `<|content_tool_error|>` 200022, `<|content_xml|>` 200024, `<|begin_of_text|>` 200028, `<|content_invoke_tool_json|>` 200049, `<|unused_200054|>` 200054 (image),
`<|content_invoke_tool_text|>` 200057. `generation_config.json`: `do_sample` true, temperature 1.0, eos 200006, pad 199999; no top-p, top-k or penalty is set anywhere in the pack.

The template is the pack's `chat_template.jinja` (6,868 B; sha256 in `intake/manifest.tsv`). It matches the Small's except for tool-call argument serialization and the `generation` markers (below).

- **Message framing**: `ROLE_TOKEN [author name] CONTENT_KIND_TOKEN body <|end_message|>`; assistant turns end with `<|content_model_end_sampling|>` (template, lines 41-132; [VLM] `vllm/renderers/inkling_encoding.py:287-313`). The assistant message is wrapped in
  `{% generation %}...{% endgeneration %}` (a Hugging Face extension marking trainable spans; it emits nothing).
- **Generation prompt**: there is none. The template has no `add_generation_prompt` branch; the model writes its own `<|message_model|>` ([VLM] `vllm/renderers/inkling_encoding.py:436-441`).
- **Reasoning**: `<|message_model|><|content_thinking|>...<|end_message|>` followed by `<|message_model|><|content_text|>...<|end_message|>`; blocks may repeat; generation stops at
  `<|content_model_end_sampling|>` ([VLM] `vllm/parser/inkling.py`). A past assistant turn re-renders its `reasoning_content` as a thinking block before its text.
- **Effort control**: `<|message_system|><|content_text|>Thinking effort level: X<|end_message|>` inserted before the first non-system message, or at the end if there is none. Default 0.9,
  range [0, 0.99]; map none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh and max 0.99 (template lines 1-23). The template prints `0` for 0.0 and Jinja's float text otherwise
  (`0.1`, `0.2`, `0.7`, `0.9`, `0.99`); VLM prints `f"{x:g}"` of the 2-decimal value: the same strings for the template's map values. The card warns that effort "none" does not fully disable reasoning, and
  that the looped model can repeat itself in long reasoning (PACK `README.md:274-278`): bound output tokens.
- **Tools**: declared once as `<|message_system|>tool_declare<|content_xml|>` + the JSON of `{description, name, parameters, type}` objects (`tojson` with sorted keys, compact separators) + `<|end_message|>`.
  A call is `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>`; a result is `<|message_tool|>NAME<|content_text|>...<|end_message|>`.
  **Different from the Small**: the Mini template serializes `args` as `{` + for each key in case-sensitive sorted order `"key":` + `tojson(value, compact)` + `}`: only the top level is sorted, nested objects keep their order
  (template lines 113-121); the Small's template sorted every nesting level (`args | tojson(sort_keys=true)`). VLM's renderer also sorts the top level only (`vllm/renderers/inkling_encoding.py:583-590`).
- VLM additionally appends a tool-choice instruction (`required`, a named function, `parallel_tool_calls = false`) to the system or effort text (`vllm/renderers/inkling_encoding.py:486-496, 616-632`); the pack's template has no such branch (UNKNOWN-9).
- The engine renders the pack's template through its own Jinja renderer ([ENG] `crates/server/src/formats/inkling2.rs:100-112`, `crates/server/src/template.rs`); whether that renderer executes the `{% generation %}` block and
  `dictsort(case_sensitive=true)` is open (UNKNOWN-9).

## 8 REXL3 reuse inventory

Surveyed read-only on the engine tree [ENG] (branch `inkling2-loop`, nothing built or run); the cited start lines were re-read for every row used below. Paths:
`cuda/` = `crates/cuda/`, `fam/` = `crates/model/src/families/inkling2/`, `srv/` = `crates/server/src/`. Verdicts: **AS IS** (usable for the Mini unchanged),
**PARAM** (usable after changing the named constants or limits), **SMALL** (Small-specific: must be rewritten), **ABSENT** (nothing found; the search terms were
`num_passes`, `core_start`, `loop_config`, `message_gate` and the op names). "Constants" are the baked-in values that differ from the Mini's.

| Op (section) | Closest existing | file:line | Constants / limits | Verdict | Work for the Mini |
|---|---|---|---|---|---|
| Embedding (3.1) | `gb_embed`; Rust `glue::embed` | `cuda/csrc/glue.cu:744`, `cuda/src/glue.rs:257`; used at `fam/model.rs:1078` | none (dim % 8; out-of-range id gives a zero row) | AS IS | `fam/device.rs:225` uploads the whole table to every rank: fine at TP1 |
| RMSNorm with gamma (3.1-3.3, 3.12) | `gb_rmsnorm`, `gb_add_rmsnorm` | `cuda/csrc/glue.cu:755, 764`, `cuda/src/glue.rs:271` | bf16 in and out, fp32 gamma (convert at load) | AS IS | the loop blend needs an fp32-input variant (below) |
| Weightless RMSNorm, fp32 (3.4, 3.7) | none as a kernel (inline in `inkling2_attn.cu:423` and the mix) | `cuda/csrc/inkling2_attn.cu:423` | | ABSENT | new fp32 pair-norm for the controllers; V norm stays inside the attention prologue |
| AttnRes mix, layer mixes (3.3) | `mix_kernel`, `launch_n`, Rust `mix`, `prescale_query` | `cuda/csrc/inkling2_mix.cu:71`, `cuda/src/inkling2/mix.rs:46, 40` | `kMaxN = 9`, `kMaxQ = 3`, `kMaxThreads = 576` (`inkling2_mix.cu:23-25`); `MAX_SOURCES = 9`, `MAX_QUERIES = 3` (`mix.rs:34, 36`); H % 256 == 0 | AS IS (H = 2304 = 9 x 256; at most 5 sources, 3 queries) | |
| AttnRes message mix (3.4) | same kernel | `cuda/csrc/inkling2_mix.cu:71` | 4 queries > `kMaxQ`; output gamma is always applied; the output must stay fp32 | PARAM | raise `kMaxQ` to 4, add a no-gamma fp32-output mode; bank of 2 sources |
| Controller blend (3.4) | none | | | ABSENT | new: pair-norm, `down` (`linear`), `gain`/`forget` (`linear`, bf16 weights, fp32 out), tanh-blend fused with the next RMSNorm |
| Dense bf16 linear (router, sink, `down`, `gain`/`forget`, `weight_proj`, patch embed) | `lin_narrow_kernel`, `lin_wide_kernel`, `lin_wide2_kernel`, Rust `linear`, `Bf16Linear` | `cuda/csrc/inkling2_linear.cu:99, 169, 238`, `cuda/src/inkling2/linear.rs:97, 113, 176` | k % 32 == 0; bf16 x and w; outputs bf16 or f32 or f32 with soft cap | AS IS | fp32-weight form (HF) is not needed: the pack stores these tensors in bf16 |
| Dense EXL3 linear (3.5, 3.9, 3.12, 4, 5) | `Exl3Weights::{from_loaded, forward, forward_chunked, forward_k2}`, `forward_big`, `gb_exl3_linear` | `cuda/src/exl3_linear.rs:450, 584, 704, 749`, `cuda/src/exl3_big.rs:103`, `cuda/csrc/exl3_linear.cu:691` (rot_in `:136`) | in % 128 == 0 and out % 128 == 0 (true for every tensor of the pack, A.15); bits 1..8; codebooks `3INST/MCG/MUL1`; decode <= 128 rows; rows are batch-invariant; the rotation output is fp16 (`xh`), the epilogue `(c H) * svh` in fp32 | AS IS (exists, not wired) | `fam/bind.rs:491` binds attention as bf16: bind q/k/v/r/gate/o, unembed, vision and drafter as EXL3, one launch per tensor (each has its own `suh`); precedent for a padded 6-bit `mul1` head: `crates/model/src/families/minicpmv4_7/device.rs:153` |
| Static short conv + residual (3.6, 3.2) | `conv_kernel`, `commit_kernel`; Rust `ShortConv`, `commit`, `layer_table` | `cuda/csrc/inkling2_conv.cu:74, 142`, `cuda/src/inkling2/conv.rs:125, 145, 213, 254` | W = 4, scale 5^-1/2, C runtime (C % 8), slot state `[slots, 3, C]` | AS IS for `attn_sconv`/`mlp_sconv` (C = 2304) | the fused residual is `bf16((P+a)+m)`; state tables must be per visit (48), not per layer |
| K/V static and dynamic conv, prologue (3.6-3.8) | `plan_kernel`, `prologue_kernel<GLB>`; Rust `prologue` | `cuda/csrc/inkling2_attn.cu:196, 297`, `cuda/src/inkling2_attn.rs:614` | `NQ = 12`, `NKV = 2`, `GRP = 6`, `D = 128`, `REL = WIN = 1024` (`:53`); `PRO_THREADS = 512` with `static_assert(2 * PRO_THREADS == REL)` (`:278-279`); the dynamic-conv warp map assumes 2 KV heads and a 256-channel split (`:343-355`); tau as literals `128000` and `0.1` (`:307`); Rust `N_Q_HEADS = 12`, `N_KV_HEADS = 2`, `RING_ROWS = 1031` | SMALL (per-element math reusable) | 24 query warps plus 16 K/V warps exceed 1,024 threads: re-split; `n_floor` 8192; ring rows stay 1,031 |
| Attention core: ring, int8 global KV, sinks, gate, bias (3.9) | `attn_kernel<GLB, RB, FOLD>`, `Geo`, `merge_kernel`, `ring_write_kernel`, `conv_commit_kernel` | `cuda/csrc/inkling2_attn.cu:529, 514, 860, 957, 978`, `cuda/src/inkling2_attn.rs:685, 729, 743` | `GRP = 6` and `NW = RB * GRP / 16` (`:515`: GRP = 3 breaks it at RB = 8); V offset literal `2*D` (`:615`); `EXP_TILE = TILE * NKV` (`:508`); exponent loads by `tid < 8` / `tid < 16` (`:602-603`); `conv_commit_kernel` launched with `KVROW = 512` threads (2,048 at the Mini) | PARAM with rework | window, REL and chunk sizes (512 global, 256 local) equal the Small's |
| KV page, ring and state memory (6) | `kv8` header, `NemoKv`, `new_global_kv`, `Ring`, memory model | `cuda/csrc/nemotron_kv8.cuh` (`PAGE_TOKENS = 256`, `NKV = 2`, `D = 128`, page 66,048 B), `cuda/src/nemotron_attn.rs:140, 269`, `cuda/src/inkling2_attn.rs:574, 476`, `fam/memory.rs:42-63, 147, 189, 268, 354`, `fam/model.rs:566-590` | layout hard-wired at 2 KV heads in a header Nemotron shares; `QB_COLS = 3276`, `KB_LD = 264` (`fam/memory.rs:56, 59`); all state arrays indexed by layer and sized `cfg.layers`; conv state `cfg.layers * 3 * (kv + 2H)` | SMALL | copy the header for 8 KV heads (K or V page: 262,144 B + 2,048 B of exponents); size state by visit (48 sets, 8 global) |
| MoE router (3.10) | `select_kernel`, Rust `Router::{new, route, select}` | `cuda/csrc/inkling2_router.cu:26`, `cuda/src/inkling2/router.rs:33, 43, 72` | `kMaxE = 512`, `kMaxK = 16`; `1e-20` literal; `(score / (sum + 1e-20)) * scale`; ties to the lower index | AS IS | logits from the bf16 `linear` with f32 output (128 x 2304) |
| MoE experts, decode (3.10) | `MoeExperts::run_softcap_partial`; `gb_moe_gemv`, `gateup_epilogue_kernel<ACT=1>`, `gb_moe_down_combine_f32w_bf16` | `cuda/src/inkling2/moe.rs:119` (checks `:101`), `cuda/csrc/moe_experts.cu:210, 331, 503, 608`, `cuda/csrc/inkling2_act.cuh:11` | dims from the consts `HIDDEN`, `INTER`, `EXPERTS` (`cuda/src/inkling2/mod.rs:31-37`); `softcap_args_ok` needs `bands.len() == 1`, equal gate and up bits, no split; `gb_moe_gemv` takes any K2 2..16 and codebooks 0..2 (so `mul1` works) | PARAM (one K per layer only) | mixed K inside a layer: see the next rows |
| MoE experts, prefill (3.10) | `run_prefill_softcap_partial`; `gb_moe_pf_gateup_softcap`, `gateup_kernel`, `down_kernel`, `combine_f32_kernel` | `cuda/src/inkling2/moe.rs:256`, `cuda/csrc/moe_prefill.cu:796` (refuses `codebook != 1` at `:800`; only `launch_gateup<K2, 1, ...>` instantiated), `:349, 416, 495` | softcap prefill is `mcg` only; `MAX_ROWS = 4096` | PARAM (template must gain codebook 2) | `mul1` prefill with the softcap epilogue does not exist |
| Mixed K per tensor (3.10) | `MoeExperts::from_groups` + `band_partition`; `MoeSubbanks` | `cuda/src/moe.rs:779, 685, 737`, `cuda/src/moe_subbanks.rs:60, 102, 596, 717` | `from_groups` refuses gate and up of different widths (the Mini keeps gate = up in every expert, A.3); decode `run` refuses `bands > 1` (`cuda/src/moe.rs:1363`); prefill loops the bands (`cuda/src/moe_prefill.rs:615-626`); the sub-bank path uses the clamp epilogue (`cuda/src/moe_subbanks.rs:377-389, 553-572`), not the softcap | ABSENT for the softcap | softcap epilogue inside the sub-bank path; per layer up to 3 expert classes (4,4,4), (4,4,5), (5,5,5) |
| Unembed, soft cap, argmax (3.12) | `vocab::unembed`, `Shard`, `topk_local`, `topk_merge`; `local_kernel`, `fill_kernel`, `merge_kernel` | `cuda/src/inkling2/vocab.rs:69, 49, 95, 141`, `cuda/csrc/inkling2_vocab.cu:29, 98, 108` | `VOCAB_VALID = 200058`, `VOCAB_PADDED = 201024`, `VOCAB_SHARD_TP4 = 50256` hard-wired (`cuda/src/inkling2/mod.rs:46-49`); top-k reads f32 logits with `ld`, `n`, `base`, `valid` | top-k AS IS; EXL3 unembed ABSENT | soft cap and valid-vocab mask exist only in the bf16 `linear`; the EXL3 head has n = 201,088 against `ld` 201,024 |
| TP plan, collectives (2) | `RANKS = 4`, `group()`, `Inkling2Rank::new`, `GroupComm`, `Tp::LEADER_OF_FOUR`, `TpLeader` | `fam/tp.rs:43, 327`, `fam/model.rs:96, 546`, `fam/memory.rs:77`, `fam/mod.rs:54, 110`, `crates/model/src/lockstep.rs:2085`, `fam/config.rs:527`, `fam/bind.rs:422` | `group()` refuses any size but 4; `Rank::new` refuses ranks != 4 ("12 query and 2 KV heads a rank"); `GroupComm::sum` early-returns at 1 rank but `sum_part_rounded` still reads `comm.rsag_applies`; `Engine::build` wraps a `TpLeader` | SMALL (plumbing) | TP1 backend without collectives; `DraftComm` has a `Solo` implementation (`cuda/src/inkling2/dspark.rs:724, 737`) |
| Drafter, feature fusion (4.2) | `mix_norm_kernel` | `cuda/csrc/inkling2_dspark.cu:123`, consts `:46-64`, `cuda/src/inkling2/dspark.rs:111-142, 385` | `H = 4608`, `TAPS = 8`, `MIXES = 6`, `NT_ROW = 512` (`PER = H / NT_ROW = 9`); `LAYERS = 5`, `GLOBAL_LAYER = 2`, `HEADS = 48`, `INTER = 18432`, `LOCAL = [0,1,3,4]` | PARAM | H = 2304 needs `NT_ROW = 256` (changes the summation tree); 3 layers, 4 taps, 4 mixes |
| Drafter, ctx KV, block input and prep (4.3-4.4) | `ctx_kv_kernel`, `block_input_kernel`, `add_norm_kernel`, `block_prep_kernel` | `cuda/csrc/inkling2_dspark.cu:259, 152, 177, 219` | `nkv` runtime, `BLK = 8`, rope theta 1e6 literal, D = 128 | PARAM | EXL3 weights for q/k/v/o and the ctx projections |
| Drafter, block attention (4.4) | `attn_part_kernel`, `attn_merge_kernel` | `cuda/csrc/inkling2_dspark.cu:324, 466` | `GRP = 6` gives `QV = 48`, `PV_Q = 24`, `ATT_SMEM` (`:58, 62, 321`); `ATT_THREADS = 256` | PARAM | GRP = 3 gives `QV = 24`, `PV_Q = 12`: re-verify tiling |
| Drafter, Markov head and fold (4.4) | `markov_concat_kernel`, `markov_step`, `markov_logits`, `fold_kernel` | `cuda/csrc/inkling2_dspark.cu:503, 522, 591, 656`, `cuda/src/inkling2/dspark.rs:446` | `RANK = 256`, `RNN_OUT = 768`, `RNN_LO = 512`; `fold_kernel` needs a bf16 unembed `[shard][H]` | PARAM | with an EXL3 unembed and EXL3 `rnn_proj`/`hidden_out` there is no bf16 matrix to fold or to column-split (UNKNOWN-17) |
| Drafter host side, taps (4.1, 4.5) | `Drafter::{commit, propose, adopt_context, save_slot, load_slot}`, `feed_drafter`, config check | `cuda/src/inkling2/dspark.rs:1061, 1224, 1186, 1001, 1013`, `fam/model.rs:969, 1621`, `fam/config.rs:481-487` | the tap check needs `last == cfg.layers`, every tap `< layers`, `layers.len() + 1 == TAPS (8)` | SMALL | `[5, 23, 41, 48]` live in the 48-visit index space and are refused (41 >= 24) |
| Vision tower (5) | `embed_kernel`, `qkv_post_kernel`, `attn_kernel`, `resid_kernel`, `bias_gelu_kernel`; `Inkling2VisionTower::{load, encode}`; host `normalize` | `cuda/csrc/inkling2_vision.cu:81, 125, 156, 244, 266`, `cuda/src/inkling2/vision.rs:240, 322`, `fam/vision.rs:57`, `crates/vision/src/inkling2.rs:52` | `DIM = 1024`, `HD = 64`, `HEADS = 16`, `GRID_SIDE = 32`, `MLP = 4096`, `LAYERS = 23`, `PATCH_K_PAD = 1216` (`cuda/csrc/inkling2_vision.cu:23-28`); `OUT_HIDDEN = 4608` (`cuda/src/inkling2/vision.rs:104`) | kernels AS IS; wiring PARAM | every dimension equals the Mini's; `OUT_HIDDEN` becomes 2304; the six block linears and the projector go through dense EXL3 (K = 6), the patch embed stays bf16 |
| Config parse and validate (1) | `Inkling2Config::from_json`, `validate`, `Section::refuse_unknown`, `parse_quant` | `fam/config.rs:339, 423, 260, 593`, known keys `:35-118` | `ARCH` and `MODEL_TYPE` pinned to the Small (`:30, 32`) | SMALL | refuses the Mini's config on: architecture and model type, missing `pre_norm_pre_affine` / `final_norm_pre_affine` (`:363-364`), unknown `loop_config`, `text_config.model_type`, `vision_config.model_type`, `text_hidden_size`, `quantization_config` lacking `out_scales`, `head_bits`, `bits`, `codebook` (`:593-603`), and the tap check (`:481`) |
| Family registration, chat format (7) | `FAMILIES`, `Inkling2Family`, `REQUIRED`, format `ENTRY` | `crates/model/src/families/mod.rs:12, 29`, `crates/model/src/family.rs:213`, `fam/mod.rs:60, 86-160`, `srv/formats/inkling2.rs:22`, `srv/main.rs:817-829` | names pinned to `Inkling2ForConditionalGeneration` / `inkling2`; `main.rs` forces int8 KV and adds the Small's model aliases keyed on the family id | PARAM | new family id and architecture names; tests assert the id lists |
| Weight inventory and bind (3.5-3.10) | `PackInventory`, `scan`, `add_drafter`; `Binder::{exl3, attention, layer, drafter, vision}`; `device::load` | `fam/inventory.rs:41, 49`, `fam/bind.rs:464, 491, 558, 610, 698, 753`, `fam/device.rs:213` | `Binder::exl3` demands one `(bits, codebook)` for all groups (`:464`); attention bound as bf16; no hard-coded sizes outside tests | PARAM | per-tensor K, dense EXL3 groups, BF16 embed (already), loop controller and cross-pass-query tensors, 48-visit naming |
| Memory model (6) | `memory_profile`, `slot_items`, `step_items`, `kv_layout` | `fam/memory.rs:354, 163, 189, 268` | `Tp::LEADER_OF_FOUR`; per-layer arrays | PARAM | 8 global KV visits instead of 4, 48 state sets |
| Spec accept, verify (4.5) | `gb_spec::{accept, accept_tree, CopyDrafter, Arbiter}`, `Engine::decode_round`, `round_cost`, `propose_batch_scored` | `crates/spec/src/lib.rs:33`, `crates/spec/src/tree.rs:422`, `crates/engine/src/lib.rs:1958`, `fam/backend.rs:461` | `round_cost` table measured on the four-rank Small; `MAX_DRAFTS = 7`, `SLOT_WINDOW_ROWS = 8` | AS IS (re-measure the cost table) | |
| Loop, per-visit state (3.2) | none | | per-layer state is indexed by layer (`fam/model.rs:566-590`); the closest notion is `AttnResConfig.block` (`fam/config.rs:127`) | ABSENT | 48-visit schedule table, per-visit KV/ring/conv state, per-pass message bank |

## 9 UNKNOWNs

Each item says what is unclear and which source disagrees or is silent. `docs/NUMERICS.md` carries one resolution line for every id.

- UNKNOWN-1: **Which reference is canonical when HFL and VLM differ.** HFL (eager, bf16 modules with fp32 islands) and VLM (fused kernels that "follow the reference
  implementation") agree at every module boundary on storage dtype but differ inside ops (UNKNOWN-2 to 6). The Small spec took VLM as primary; this spec takes HFL as primary
  and VLM for rounding where HFL is silent. Neither is bit-exact with EXL3 weights.
- UNKNOWN-2: **Where the routing weight is applied.** HFL multiplies the `down` output by `w_j` and accumulates in fp32 ([HFL] `modeling:787-805`); VLM folds `w_j` into the activation
  and rounds `a * w` to bf16 before `down` ([VLM] `n/ops/activation.py:63-79`). The engine's combine already applies the weight after `down`.
- UNKNOWN-3: **Summation order of dense linears.** HFL leaves it to torch (bf16 GEMM, fp32 accumulate); VLM fixes 1,024-wide K groups for its bf16 linears (`n/ops/linear.py`);
  the EXL3 path has its own fixed order (`cuda/csrc/exl3_linear.cu` header). Applies to the router, `attn_sink_proj`, controller `down`/`gain`/`forget`, dynamic-conv `weight_proj` and every EXL3 linear.
- UNKNOWN-4: **Conv output scale.** HFL divides by `math.sqrt(5)` (static, [HFL] `modeling:392`) and `sqrt(9)` (dynamic, `modeling:454`); VLM multiplies by `(W+1)^-1/2` and `(2W+1)^-1/2`
  (`n/ops/sconv.py`). They differ by at most one fp32 ulp before the bf16 store.
- UNKNOWN-5: **Attention softmax numerics.** HFL rounds the logits to bf16 three times (`matmul`, scale, bias add), takes an fp32 softmax over logits and sink, rounds `P` to bf16 and `o` to bf16 before the gate
  ([HFL] `modeling:111-142, 531-543`); VLM keeps fp32 logits with FA4's online softmax (bf16 `P`, lazy rescale) and writes `o * sigma(gate) * sigma(lse - sink)` in one step
  (`n/ops/attention.py`). The engine's kernel has its own order. The sink probability mass is the same in all three.
- UNKNOWN-6: **Unembed output precision.** HFL rounds the `lm_head` output to bf16 and then upcasts (`modeling:1534`); VLM keeps fp32 (`head_dtype = float32`, `n/configs.py:220`). The pack's unembed is EXL3
  and the sources are silent about the output dtype of an EXL3 head.
- UNKNOWN-7: **Vision pixel normalization.** HF's processor maps `v` to `(v/255 - 0.5)/0.5` in fp32 ([HFS] `image_processing_inkling2.py:90-91`); VLM and the engine's host code go through the CLIP mean and std with a bf16
  intermediate (`n/ops/vision.py:85-86`, `crates/vision/src/inkling2.rs:52-55`). Same map, different rounding of one intermediate.
- UNKNOWN-8: **Drafter local window and context visibility.** Only VLM defines the drafter. A local layer at block start `p` sees context keys `[p - 1024, p - 1]` plus the 8 block keys (`n/dspark.py:173`), and the anchor's own
  ctx features are never visible; the Mini's local layers are 0 and 2 (the Small's were 0, 1, 3, 4). HFL is silent.
- UNKNOWN-9: **Chat rendering.** The pack's template is a Jinja file with `{% generation %}` blocks, `dictsort(case_sensitive=true)` and per-value `tojson`; the engine renders it with its own Jinja renderer
  ([ENG] `crates/server/src/template.rs`), which lists `generation` among reserved words but was not shown to execute the block. VLM's native renderer differs from the template by adding a tool-choice instruction and by
  formatting the effort with `%g`. HFL has no template.
- UNKNOWN-10: **AttnRes and message query pre-scale.** HFL passes `scaling = H^-1/2` into attention; VLM scales each fp32 query by `H^-1/2` once at load (`n/loop.py:179`, `n/model.py:160`). The engine's `prescale_query` does the
  same (`cuda/src/inkling2/mix.rs:40`). They differ at the ulp level.
- UNKNOWN-11: **Expert input.** The sources define the bf16 `mlp_norm` output as the expert input, but HFL multiplies in bf16 while the EXL3 kernels rotate and convert the input to fp16 (`cuda/csrc/exl3_linear.cu:134-150`);
  the pack was calibrated by a method that is not published.
- UNKNOWN-12: **State at a prefix-cache split.** HFL holds one cache per pass (`modeling:1354-1358`) and four conv inputs per stream; VLM backs three positions ([PR181]). Nothing says whether a snapshot at an arbitrary position
  (not a page boundary) is required.
- UNKNOWN-13: **Index space of `target_layer_ids`.** HFL has no drafter and no hidden-state taps for it; VLM defines the ids as visit indices of the input partial, the last being the visit count and meaning the final-norm output
  (`n/configs.py:255-262, 280-292`, `n/loop.py:375-376`); the pack's `[5, 23, 41, 48]` is consistent with that reading only.
- UNKNOWN-14: **Greedy argmax ties.** HFL leaves `argmax` to the sampler; VLM's `greedy_argmax` (`vllm/v1/worker/gpu/sample/greedy_argmax.py`) takes the first maximum with NaN as -inf. The engine's top-k kernels (`cuda/csrc/inkling2_vocab.cu`) are not shown to do the same across the head's tile boundary.
- UNKNOWN-15: **Which expert tensor is the gate.** HFL's converter de-interleaves the original `w13` rows into `gate_up_proj` and `chunk(2)` makes the first half the gate ([HFX] `conversion_mapping:2658-2672`, [HFL] `modeling:809-811`); the EXL3 pack stores separate
  `gate_proj` and `up_proj` tensors and no source says how they were split from the interleaved rows.
- UNKNOWN-16: **EXL3 dense decode on this pack.** The `mul1` codebook, K = 6 attention tensors, the padded 201,088-row unembed, the 5,120-input `rnn_proj` and biased vision linears are described by EXL3's own format, not by HFL or VLM.
  The Small's oracle decodes `mul1` and `mcg` but was never run on K = 6 or on these shapes.
- UNKNOWN-17: **Quantized Markov head.** VLM splits the BF16 `rnn_proj` by columns and folds `hidden_out` into the BF16 unembed (`n/dspark.py:348-385`). In this pack `rnn_proj`, `hidden_out` and the unembed are EXL3, so a column split or an
  `unembed x hidden_out` product has no stored matrix; no source says whether to dequantize and fold or to apply the stages unfolded.
  **Resolved (2026-11-23):** the engine folds (variant (a)); the unfolded variant (c) accepts the same drafts and costs 7.4 ms a round (`docs/NUMERICS.md`, `bench/records/2026-11-23-markov-variants/`).
- UNKNOWN-18: **Router tie order.** HFL uses `torch.topk(sorted=False)`, whose order and tie rule are undefined; VLM breaks ties toward the lower expert index (`n/ops/moe.py:35`); the engine's router does the same.
- UNKNOWN-19: **Config keys HFL ignores.** The pack's `text_config` carries `use_value_norm_no_gamma`, `attn_res_normalize_values`, `use_gate_bias`, `q_bias`, `o_bias`, `n_shared_experts`, `local_layer_ids`, `loop_config.is_varloop`
  and the vision key `temporal_patch_size`, none of which HFL's `Inkling2LoopTextConfig` reads (it hardcodes the behaviour, or derives `layer_types` from the layer index); VLM reads most of them. All carry the values HFL assumes; `scoring_func` and `image_token_id` are absent (HF defaults: sigmoid, 200054).
- UNKNOWN-20: **int8 KV across three passes.** The Small's int8 page format (power-of-two scale per (token, KV head)) was measured for 8 global layers; the Mini reads 8 global visits whose inputs depend on three
  repetitions of the same weights, and PR249 only tested fp8 KV on a GPU ([PR249]). No source measures the quality cost of int8 KV on the looped model, or on the drafter's ninth plane (which only changes which tokens are proposed).
- UNKNOWN-21: **Drafter residual-add norm input.** VLM's non-fused path (FlashInfer `fused_add_rmsnorm`, the one a single GPU takes) normalizes "from the unrounded sum" (`n/dspark.py:53-68`), its fused all-reduce path norms the bf16 residual
  (`n/dspark.py:71-110`); the engine's existing `add_norm_kernel` (`cuda/csrc/inkling2_dspark.cu:176`) and the Small's oracle norm the bf16 residual. HFL has no drafter.

## Appendix: commands

Every command was run from the recipe root (`~/projects/inkling2-mini-spec`; the files `intake/files/config.json`, `intake/files/processor_config.json`, `intake/files/tokenizer_config.json` and `intake/census.json` are pinned to the pack revision) on 2026-11-22; the output below is what each command printed. A.7, A.11 and A.15 read the `mtp.safetensors` header with an HTTP range request at the pinned revision (nothing is downloaded beyond the header). The commands need `jq` and Python 3 with the standard library only.

### A.1 Text-config dimensions (one line, from `intake/files/config.json`)

```sh
jq -c '.text_config | {hidden_size,num_hidden_layers,num_attention_heads,num_key_value_heads,head_dim,swa_num_attention_heads,swa_num_key_value_heads,swa_head_dim,d_rel,rel_extent,sliding_window_size,sconv_kernel_size,log_scaling_n_floor,log_scaling_alpha,qk_l2_clip_threshold,rel_l2_clip_threshold,n_layers_per_attn_res_block,attn_res_normalize_values,use_value_norm_no_gamma,final_logit_softcapping,vocab_size,unpadded_vocab_size,n_routed_experts,num_experts_per_tok,n_shared_experts,intermediate_size,mlp_gate_softcap,mlp_up_softcap,route_scale,use_gate_bias,rms_norm_eps,q_bias,o_bias,loop_config}' intake/files/config.json
```

```text
{"hidden_size":2304,"num_hidden_layers":24,"num_attention_heads":24,"num_key_value_heads":8,"head_dim":128,"swa_num_attention_heads":24,"swa_num_key_value_heads":8,"swa_head_dim":128,"d_rel":16,"rel_extent":1024,"sliding_window_size":1024,"sconv_kernel_size":4,"log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,"n_layers_per_attn_res_block":6,"attn_res_normalize_values":true,"use_value_norm_no_gamma":true,"final_logit_softcapping":30.0,"vocab_size":201024,"unpadded_vocab_size":200058,"n_routed_experts":128,"num_experts_per_tok":8,"n_shared_experts":0,"intermediate_size":1152,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,"use_gate_bias":true,"rms_norm_eps":0.000001,"q_bias":false,"o_bias":false,"loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}}
```

### A.2 Layer schedule: global layers, loop core, 48 visits, global and sliding visits, tap meaning

```sh
python3 - <<'E'
import json
t = json.load(open("intake/files/config.json"))["text_config"]
n = t["num_hidden_layers"]; lc = t["loop_config"]; s, e, p = lc["core_start"], lc["core_end"], lc["num_passes"]
glob = sorted(set(range(n)) - set(t["local_layer_ids"]))
print("global layers", glob, "= (i+1)%6==0:", glob == [i for i in range(n) if (i + 1) % 6 == 0])
print("loop core layers", list(range(s, e)), "passes", p, "gate_rank", lc["gate_rank"], "is_varloop", lc["is_varloop"])
visits = list(range(s)) + [i for _ in range(p) for i in range(s, e)] + list(range(e, n))
print("visits", len(visits), "= 24 + (3-1)*12 =", n + (p - 1) * (e - s))
def pass_of(v):
    return (v - s) // (e - s) if s <= v < s + p * (e - s) else 0
gv = [v for v, L in enumerate(visits) if L in glob]
print("global visits", gv, "count", len(gv), "| v%6==5 for all:", all(v % 6 == 5 for v in gv), "| ordinal v//6:", [v // 6 for v in gv])
print("sliding visits", len(visits) - len(gv))
print("core visits per layer:", {L: [v for v, x in enumerate(visits) if x == L] for L in (5, 6, 11, 17, 18, 23)})
tl = json.load(open("intake/files/config.json"))["mtp_config"]["target_layer_ids"]
print("taps", tl, "-> visit (layer,pass):", [(visits[v], pass_of(v) if s <= v < s + p * (e - s) else 0) if v < len(visits) else "final-norm output h" for v in tl])
E
```

```text
global layers [5, 11, 17, 23] = (i+1)%6==0: True
loop core layers [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] passes 3 gate_rank 64 is_varloop False
visits 48 = 24 + (3-1)*12 = 48
global visits [5, 11, 17, 23, 29, 35, 41, 47] count 8 | v%6==5 for all: True | ordinal v//6: [0, 1, 2, 3, 4, 5, 6, 7]
sliding visits 40
core visits per layer: {5: [5], 6: [6, 18, 30], 11: [11, 23, 35], 17: [17, 29, 41], 18: [42], 23: [47]}
taps [5, 23, 41, 48] -> visit (layer,pass): [(5, 0), (11, 1), (17, 2), 'final-norm output h']
```

### A.3 Expert tensors and their EXL3 K (trellis last dim / 16), `intake/census.json`

```sh
python3 - <<'E'
import json, re, collections
m = json.load(open("intake/census.json"))["modules"]
K = collections.Counter(); pat = collections.defaultdict(dict); shp = collections.defaultdict(set)
for name, v in m.items():
    r = re.match(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$", name)
    if r:
        tr = v["tensors"]["trellis"]["shape"]
        k = tr[-1] // 16; K[k] += 1; pat[(int(r[1]), int(r[2]))][r[3]] = k; shp[(r[3], k)].add(tuple(tr))
print("routed expert tensors by K:", dict(K), "total", sum(K.values()), "= 24*128*3 =", 24 * 128 * 3)
print("experts", len(pat), "(gate,up,down) K classes:", dict(collections.Counter((p["gate"], p["up"], p["down"]) for p in pat.values())))
print("trellis shapes:", {f"{w}_proj K{k}": sorted(s) for (w, k), s in sorted(shp.items())})
print("5-bit tensors in layers 6-17:", sum(1 for (L, e), p in pat.items() for k in p.values() if k == 5 and 6 <= L <= 17), "of", K[5])
print("5-bit tensors per layer:", {L: sum(1 for (l, e), p in pat.items() if l == L for k in p.values() if k == 5) for L in range(24)})
for L in range(24):
    c = collections.Counter((p["gate"], p["up"], p["down"]) for (l, e), p in pat.items() if l == L)
    print(f"layer {L:2d} expert classes (gate,up,down):", dict(sorted(c.items())))
E
```

```text
routed expert tensors by K: {4: 7944, 5: 1272} total 9216 = 24*128*3 = 9216
experts 3072 (gate,up,down) K classes: {(4, 4, 4): 2372, (5, 5, 5): 286, (4, 4, 5): 414}
trellis shapes: {'down_proj K4': [(72, 144, 64)], 'down_proj K5': [(72, 144, 80)], 'gate_proj K4': [(144, 72, 64)], 'gate_proj K5': [(144, 72, 80)], 'up_proj K4': [(144, 72, 64)], 'up_proj K5': [(144, 72, 80)]}
5-bit tensors in layers 6-17: 1252 of 1272
5-bit tensors per layer: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 10, 7: 30, 8: 68, 9: 85, 10: 96, 11: 128, 12: 29, 13: 71, 14: 122, 15: 153, 16: 224, 17: 236, 18: 8, 19: 5, 20: 0, 21: 1, 22: 5, 23: 1}
layer  0 expert classes (gate,up,down): {(4, 4, 4): 128}
layer  1 expert classes (gate,up,down): {(4, 4, 4): 128}
layer  2 expert classes (gate,up,down): {(4, 4, 4): 128}
layer  3 expert classes (gate,up,down): {(4, 4, 4): 128}
layer  4 expert classes (gate,up,down): {(4, 4, 4): 128}
layer  5 expert classes (gate,up,down): {(4, 4, 4): 128}
layer  6 expert classes (gate,up,down): {(4, 4, 4): 120, (4, 4, 5): 7, (5, 5, 5): 1}
layer  7 expert classes (gate,up,down): {(4, 4, 4): 108, (4, 4, 5): 15, (5, 5, 5): 5}
layer  8 expert classes (gate,up,down): {(4, 4, 4): 84, (4, 4, 5): 32, (5, 5, 5): 12}
layer  9 expert classes (gate,up,down): {(4, 4, 4): 71, (4, 4, 5): 43, (5, 5, 5): 14}
layer 10 expert classes (gate,up,down): {(4, 4, 4): 70, (4, 4, 5): 39, (5, 5, 5): 19}
layer 11 expert classes (gate,up,down): {(4, 4, 4): 50, (4, 4, 5): 53, (5, 5, 5): 25}
layer 12 expert classes (gate,up,down): {(4, 4, 4): 107, (4, 4, 5): 17, (5, 5, 5): 4}
layer 13 expert classes (gate,up,down): {(4, 4, 4): 81, (4, 4, 5): 35, (5, 5, 5): 12}
layer 14 expert classes (gate,up,down): {(4, 4, 4): 60, (4, 4, 5): 41, (5, 5, 5): 27}
layer 15 expert classes (gate,up,down): {(4, 4, 4): 47, (4, 4, 5): 45, (5, 5, 5): 36}
layer 16 expert classes (gate,up,down): {(4, 4, 4): 28, (4, 4, 5): 38, (5, 5, 5): 62}
layer 17 expert classes (gate,up,down): {(4, 4, 4): 28, (4, 4, 5): 32, (5, 5, 5): 68}
layer 18 expert classes (gate,up,down): {(4, 4, 4): 120, (4, 4, 5): 8}
layer 19 expert classes (gate,up,down): {(4, 4, 4): 123, (4, 4, 5): 5}
layer 20 expert classes (gate,up,down): {(4, 4, 4): 128}
layer 21 expert classes (gate,up,down): {(4, 4, 4): 127, (4, 4, 5): 1}
layer 22 expert classes (gate,up,down): {(4, 4, 4): 125, (4, 4, 5): 2, (5, 5, 5): 1}
layer 23 expert classes (gate,up,down): {(4, 4, 4): 127, (4, 4, 5): 1}
```

### A.4 Dense EXL3 tensors (K from the trellis) and the unembed padding

```sh
python3 - <<'E'
import json, re
m = json.load(open("intake/census.json"))["modules"]
def row(n):
    t = m[n]["tensors"]; tr = t["trellis"]["shape"]
    return f"{n}: trellis {tr} K={tr[-1] // 16} in={tr[0] * 16} out={tr[1] * 16} suh={t['suh']['shape']} svh={t['svh']['shape']} mul1={t['mul1']['dtype']}"
for n in ("model.llm.layers.0.attn.wq_du", "model.llm.layers.0.attn.wk_dv", "model.llm.layers.0.attn.wv_dv", "model.llm.layers.0.attn.wr_du", "model.llm.layers.0.attn.gate_attn", "model.llm.layers.0.attn.wo_ud", "model.llm.unembed", "model.visual.proj_in", "model.visual.proj_out", "model.visual.backbone.layers.0.mlp.c_fc", "model.visual.backbone.layers.0.self_attn.q_proj"):
    print(row(n))
ks = {}
for n, v in m.items():
    if "trellis" in v["tensors"] and ".mlp.experts." not in n:
        grp = re.sub(r"\.\d+", ".N", n.split(".", 3)[2] if n.startswith("model.llm.layers") else n)
        ks.setdefault(re.sub(r"model\.(llm|visual)\.(backbone\.)?(layers\.\d+\.)?", r"\1:", n).replace("\\d", ""), set()).add(v["tensors"]["trellis"]["shape"][-1] // 16)
print("K of every non-expert EXL3 tensor kind:", {k: sorted(v) for k, v in sorted(ks.items())})
print("codebook dtype tensors named mul1:", sum(1 for v in m.values() if "mul1" in v["tensors"]), "| tensors named mcg:", sum(1 for v in m.values() if "mcg" in v["tensors"]))
E
```

```text
model.llm.layers.0.attn.wq_du: trellis [144, 192, 96] K=6 in=2304 out=3072 suh=[2304] svh=[3072] mul1=I32
model.llm.layers.0.attn.wk_dv: trellis [144, 64, 96] K=6 in=2304 out=1024 suh=[2304] svh=[1024] mul1=I32
model.llm.layers.0.attn.wv_dv: trellis [144, 64, 96] K=6 in=2304 out=1024 suh=[2304] svh=[1024] mul1=I32
model.llm.layers.0.attn.wr_du: trellis [144, 24, 96] K=6 in=2304 out=384 suh=[2304] svh=[384] mul1=I32
model.llm.layers.0.attn.gate_attn: trellis [144, 192, 96] K=6 in=2304 out=3072 suh=[2304] svh=[3072] mul1=I32
model.llm.layers.0.attn.wo_ud: trellis [192, 144, 96] K=6 in=3072 out=2304 suh=[3072] svh=[2304] mul1=I32
model.llm.unembed: trellis [144, 12568, 96] K=6 in=2304 out=201088 suh=[2304] svh=[201088] mul1=I32
model.visual.proj_in: trellis [256, 256, 96] K=6 in=4096 out=4096 suh=[4096] svh=[4096] mul1=I32
model.visual.proj_out: trellis [256, 144, 96] K=6 in=4096 out=2304 suh=[4096] svh=[2304] mul1=I32
model.visual.backbone.layers.0.mlp.c_fc: trellis [64, 256, 96] K=6 in=1024 out=4096 suh=[1024] svh=[4096] mul1=I32
model.visual.backbone.layers.0.self_attn.q_proj: trellis [64, 64, 96] K=6 in=1024 out=1024 suh=[1024] svh=[1024] mul1=I32
K of every non-expert EXL3 tensor kind: {'llm:attn.gate_attn': [6], 'llm:attn.wk_dv': [6], 'llm:attn.wo_ud': [6], 'llm:attn.wq_du': [6], 'llm:attn.wr_du': [6], 'llm:attn.wv_dv': [6], 'llm:unembed': [6], 'visual:mlp.c_fc': [6], 'visual:mlp.c_proj': [6], 'visual:proj_in': [6], 'visual:proj_out': [6], 'visual:self_attn.k_proj': [6], 'visual:self_attn.out_proj': [6], 'visual:self_attn.q_proj': [6], 'visual:self_attn.v_proj': [6]}
codebook dtype tensors named mul1: 9501 | tensors named mcg: 0
```

### A.5 Non-EXL3 tensors of the trunk (name pattern, dtype, shape, count)

```sh
python3 - <<'E'
import json, re, collections
m = json.load(open("intake/census.json"))["modules"]
g = collections.OrderedDict()
for n, v in m.items():
    if "trellis" in v["tensors"]:
        continue
    p = re.sub(r"\.\d+(?=\.|$)", ".N", n)
    key = (p, tuple((t, x["dtype"], tuple(x["shape"])) for t, x in sorted(v["tensors"].items())))
    g[key] = g.get(key, 0) + 1
for (p, ts), c in g.items():
    if "visual" in p:
        continue
    print(c, p, " | ".join(f"{t}:{d}{list(s)}" for t, d, s in ts))
E
```

```text
19 model.llm.attn_res.attn_queries.N tensor:F32[2304]
4 model.llm.attn_res.attn_queries.N tensor:F32[3, 2304]
24 model.llm.attn_res.mlp_queries.N tensor:F32[2304]
1 model.llm.attn_res.output_query tensor:F32[2304]
1 model.llm.embed weight:BF16[201024, 2304]
1 model.llm.embed_norm weight:BF16[2304]
24 model.llm.layers.N.attn.attn_sink_proj weight:BF16[24, 2304]
20 model.llm.layers.N.attn.k_sconv weight:BF16[1024, 1, 4]
24 model.llm.layers.N.attn.q_l2_clip.scale tensor:BF16[24, 128]
24 model.llm.layers.N.attn.rel_l2_clip.scale tensor:BF16[24, 16]
24 model.llm.layers.N.attn.rel_logits_proj.proj tensor:BF16[16, 1024]
20 model.llm.layers.N.attn.v_sconv weight:BF16[1024, 1, 4]
20 model.llm.layers.N.attn_norm weight:BF16[2304]
24 model.llm.layers.N.attn_sconv weight:BF16[2304, 1, 4]
24 model.llm.layers.N.mlp.gate bias:F32[128] | weight:BF16[128, 2304]
24 model.llm.layers.N.mlp_norm weight:BF16[2304]
24 model.llm.layers.N.mlp_sconv weight:BF16[2304, 1, 4]
12 model.llm.layers.N.controllers.attn.down_weight tensor:BF16[64, 4608]
12 model.llm.layers.N.controllers.attn.forget_weight tensor:BF16[2304, 64]
12 model.llm.layers.N.controllers.attn.gain_weight tensor:BF16[2304, 64]
12 model.llm.layers.N.controllers.k.down_weight tensor:BF16[64, 4608]
12 model.llm.layers.N.controllers.k.forget_weight tensor:BF16[2304, 64]
12 model.llm.layers.N.controllers.k.gain_weight tensor:BF16[2304, 64]
12 model.llm.layers.N.controllers.mlp.down_weight tensor:BF16[64, 4608]
12 model.llm.layers.N.controllers.mlp.forget_weight tensor:BF16[2304, 64]
12 model.llm.layers.N.controllers.mlp.gain_weight tensor:BF16[2304, 64]
12 model.llm.layers.N.controllers.v.down_weight tensor:BF16[64, 4608]
12 model.llm.layers.N.controllers.v.forget_weight tensor:BF16[2304, 64]
12 model.llm.layers.N.controllers.v.gain_weight tensor:BF16[2304, 64]
4 model.llm.layers.N.attn.k_sconv.static_w tensor:BF16[4, 1024]
4 model.llm.layers.N.attn.k_sconv.weight_proj tensor:BF16[32, 2304]
4 model.llm.layers.N.attn.v_sconv.static_w tensor:BF16[4, 1024]
4 model.llm.layers.N.attn.v_sconv.weight_proj tensor:BF16[32, 2304]
4 model.llm.layers.N.k_input_norm weight:BF16[2304]
4 model.llm.layers.N.q_input_norm weight:BF16[2304]
4 model.llm.layers.N.v_input_norm weight:BF16[2304]
1 model.llm.middle_cycle_cross_pass_kv_queries tensor:F32[24, 2304]
1 model.llm.middle_cycle_cross_pass_queries tensor:F32[24, 2304]
1 model.llm.norm weight:BF16[2304]
```

### A.6 Which layers carry which tensors (AttnRes queries, norms, dynamic conv)

```sh
python3 - <<'E'
import json, re
m = json.load(open("intake/census.json"))["modules"]
aq = {int(re.search(r"\.(\d+)$", n)[1]): v["tensors"]["tensor"]["shape"] for n, v in m.items() if ".attn_queries." in n}
print("attn_queries ids", min(aq), "..", max(aq), "count", len(aq), "| shape [3,2304] at", sorted(i for i, s in aq.items() if len(s) == 2), "-> layers", sorted(i + 1 for i, s in aq.items() if len(s) == 2))
mq = [n for n in m if ".mlp_queries." in n]; print("mlp_queries", len(mq))
lay = lambda pat: sorted({int(re.search(r"layers\.(\d+)\.", n)[1]) for n in m if re.search(pat, n)})
print("attn_norm layers (local):", lay(r"\.attn_norm$"))
print("q_input_norm layers (global):", lay(r"\.q_input_norm$"))
print("static_w (dynamic conv) layers:", lay(r"k_sconv\.static_w$"))
print("k_sconv.weight (static conv) layers:", lay(r"k_sconv$"))
print("controllers layers:", lay(r"\.controllers\.attn\.down_weight$"))
print("cross-pass queries:", {n: m[n]["tensors"]["tensor"]["shape"] for n in m if "cross_pass" in n})
E
```

```text
attn_queries ids 0 .. 22 count 23 | shape [3,2304] at [4, 10, 16, 22] -> layers [5, 11, 17, 23]
mlp_queries 24
attn_norm layers (local): [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
q_input_norm layers (global): [5, 11, 17, 23]
static_w (dynamic conv) layers: [5, 11, 17, 23]
k_sconv.weight (static conv) layers: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
controllers layers: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
cross-pass queries: {'model.llm.middle_cycle_cross_pass_kv_queries': [24, 2304], 'model.llm.middle_cycle_cross_pass_queries': [24, 2304]}
```

### A.7 Resident bytes by component (reproduces docs/FIT.md); mtp tensor bytes from the safetensors header

```sh
python3 - <<'E'
import json, re, collections, struct, urllib.request
c = json.load(open("intake/census.json")); m = c["modules"]
cat = collections.Counter()
for n, v in m.items():
    b = v["bytes"]
    if ".mlp.experts." in n: k = "routed experts"
    elif re.search(r"\.layers\.\d+\.(attn\.|attn_sconv|mlp_sconv)", n): k = "attention (EXL3 6-bit + small tensors + sconv)"
    elif n == "model.llm.embed": k = "embedding"
    elif n == "model.llm.unembed": k = "unembed"
    elif n.startswith("model.visual"): k = "vision"
    else: k = "controllers, routers, glue"
    cat[k] += b
rev = c["revision"]
url = f"https://huggingface.co/{c['repo']}/resolve/{rev}/mtp.safetensors"
def rng(a, b):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
n = struct.unpack("<Q", rng(0, 7))[0]; hdr = json.loads(rng(8, 7 + n))
mtp = sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in hdr.items() if k != "__metadata__")
cat["drafter (mtp.safetensors)"] = mtp
G = 2 ** 30
for k, v in cat.items(): print(f"{k:30s} {v:>16,d} B {v / G:8.3f} GiB")
tot = sum(cat.values()); print(f"{'total':30s} {tot:>16,d} B {tot / G:8.3f} GiB  (trunk census total_bytes {c['total_bytes']:,d})")
E
```

```text
controllers, routers, glue           72,016,896 B    0.067 GiB
embedding                           926,318,592 B    0.863 GiB
attention (EXL3 6-bit + small tensors + sconv)      490,353,216 B    0.457 GiB
routed experts                   12,716,347,392 B   11.843 GiB
unembed                             347,886,852 B    0.324 GiB
vision                              242,863,152 B    0.226 GiB
drafter (mtp.safetensors)           236,558,740 B    0.220 GiB
total                            15,032,344,840 B   14.000 GiB  (trunk census total_bytes 14,795,786,100)
```

### A.8 Per-token KV and fixed state, from the dimensions

```sh
python3 - <<'E'
import json
t = json.load(open("intake/files/config.json"))["text_config"]
kvh, hd, H = t["num_key_value_heads"], t["head_dim"], t["hidden_size"]
gvis, lvis, W = 8, 40, t["sliding_window_size"]
bf16 = gvis * 2 * kvh * hd * 2; int8 = gvis * (2 * kvh * hd * 1 + 2 * kvh * 1)
print("bf16 B/token", bf16, "| int8 + 1 exponent byte per (token, kv head, K and V) B/token", int8, "| drafter global layer plane bf16", 2 * kvh * hd * 2, "int8", 2 * kvh * hd + 2 * kvh)
print("with drafter plane: bf16", bf16 + 2 * kvh * hd * 2, "int8", int8 + 2 * kvh * hd + 2 * kvh)
ring = W + 7
print("ring positions", ring, "| target ring bytes per slot (40 sliding visits, bf16)", lvis * ring * 2 * kvh * hd * 2, "=", round(lvis * ring * 2 * kvh * hd * 2 / 2**30, 4), "GiB")
dr = 2 * (W + 8) * 2 * kvh * hd * 2
print("drafter rings (2 local layers, 1032 positions, bf16) B", dr)
conv = 3 * (kvh * hd * 2 + H * 2) * 2
print("conv state per visit B (3 tokens x (K 1024 + V 1024 + attn 2304 + mlp 2304) x bf16)", conv, "| 48 visits", 48 * conv, "=", round(48 * conv / 2**20, 3), "MiB")
print("1,048,576-token lane int8 GiB", round(1048576 * int8 / 2**30, 3), "| bf16 GiB", round(1048576 * bf16 / 2**30, 3), "| +drafter int8", round(1048576 * (int8 + 2 * kvh * hd + 2 * kvh) / 2**30, 3))
print("fixed state per slot (rings + drafter rings + conv incl. drafter none) GiB", round((lvis * ring * 2 * kvh * hd * 2 + dr + 48 * conv) / 2**30, 4))
E
```

```text
bf16 B/token 32768 | int8 + 1 exponent byte per (token, kv head, K and V) B/token 16512 | drafter global layer plane bf16 4096 int8 2064
with drafter plane: bf16 36864 int8 18576
ring positions 1031 | target ring bytes per slot (40 sliding visits, bf16) 168919040 = 0.1573 GiB
drafter rings (2 local layers, 1032 positions, bf16) B 8454144
conv state per visit B (3 tokens x (K 1024 + V 1024 + attn 2304 + mlp 2304) x bf16) 39936 | 48 visits 1916928 = 1.828 MiB
1,048,576-token lane int8 GiB 16.125 | bf16 GiB 32.0 | +drafter int8 18.141
fixed state per slot (rings + drafter rings + conv incl. drafter none) GiB 0.167
```

### A.9 Parameter count from the dimensions

```sh
python3 - <<'E'
import json
c = json.load(open("intake/files/config.json")); t = c["text_config"]; v = c["vision_config"]; d = c["mtp_config"]
H, L, E, I, hd, nq, nkv, dr_, V, Vu = t["hidden_size"], t["num_hidden_layers"], t["n_routed_experts"], t["intermediate_size"], t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"], t["d_rel"], t["vocab_size"], t["unpadded_vocab_size"]
expert = 3 * H * I
attn = H * nq * hd * 2 + 2 * H * nkv * hd + H * nq * dr_ + nq * hd * H + H * nq
router = E * H + E
sconv = 4 * H + 2 * nkv * hd * 4
per_layer = E * expert + attn + router
embed, unembed = V * H, Vu * H
text = L * per_layer + embed + unembed
print(f"experts/layer {E * expert / 1e6:.2f} M, attention/layer {attn / 1e6:.2f} M, router {router}, per layer {per_layer / 1e6:.2f} M")
print(f"text (24 layers + embed {embed / 1e6:.1f} M + unembed {unembed / 1e6:.1f} M) = {text / 1e9:.3f} B")
act = 8 * expert + attn
print(f"active per layer pass {act / 1e6:.2f} M; one extra pass over the 12 core layers adds {12 * act / 1e9:.3f} B")
print(f"active per token without loop repeats {(L * act + unembed) / 1e9:.3f} B; with 3 passes {((L + 24) * act + unembed) / 1e9:.3f} B")
vis = v["num_hidden_layers"] * (4 * v["hidden_size"] ** 2 + 2 * v["hidden_size"] * 4096) + 4 * v["hidden_size"] * v["projection_hidden_dim"] + v["projection_hidden_dim"] * t["hidden_size"]
print(f"vision blocks + projector approx {vis / 1e9:.3f} B")
dl = d["num_hidden_layers"] * (H * nq * hd * 2 + 4 * H * nkv * hd + 3 * H * d["intermediate_size"])
print(f"drafter layers approx {dl / 1e9:.3f} B")
E
```

```text
experts/layer 1019.22 M, attention/layer 26.89 M, router 295040, per layer 1046.40 M
text (24 layers + embed 463.2 M + unembed 460.9 M) = 26.038 B
active per layer pass 90.59 M; one extra pass over the 12 core layers adds 1.087 B
active per token without loop repeats 2.635 B; with 3 passes 4.809 B
vision blocks + projector approx 0.316 B
drafter layers approx 0.262 B
```

### A.10 Vision config and tensors

```sh
python3 - <<'E'
import json, re, collections
c = json.load(open("intake/files/config.json")); print("vision_config", json.dumps(c["vision_config"], sort_keys=True))
print("processor", json.dumps(json.load(open("intake/files/processor_config.json")), sort_keys=True))
m = json.load(open("intake/census.json"))["modules"]
vis = {n: v for n, v in m.items() if n.startswith("model.visual")}
print("vision modules", len(vis), "bytes", sum(v["bytes"] for v in vis.values()))
for n in ("model.visual.backbone.embeddings.patch_embedding", "model.visual.backbone.embeddings.position_embedding", "model.visual.backbone.embeddings.class_embedding", "model.visual.backbone.embeddings.cls_position_embedding", "model.visual.backbone.ln_pre", "model.visual.backbone.layers.0.ln_1", "model.visual.backbone.layers.0.ls_1.gamma", "model.visual.backbone.layers.0.mlp.c_fc", "model.visual.backbone.layers.22.self_attn.out_proj", "model.visual.proj_in", "model.visual.proj_out"):
    print(n, {t: (x["dtype"], x["shape"]) for t, x in m[n]["tensors"].items()})
E
```

```text
vision_config {"adapter_merge_size": 2, "hidden_size": 1024, "layer_norm_eps": 1e-05, "mlp_ratio": 4.0, "model_type": "inkling2_vision", "native_patch_size": 20, "num_attention_heads": 16, "num_hidden_layers": 23, "posemb_grid_size": 32, "projection_hidden_dim": 4096, "projector_bias": false, "temporal_patch_size": 2, "text_hidden_size": 2304}
processor {"image_processor": {"do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Inkling2ImageProcessor", "image_std": [0.5, 0.5, 0.5], "rescale_factor": 0.00392156862745098, "size": {"height": 40, "width": 40}}, "image_token": "<|unused_200054|>", "processor_class": "Inkling2Processor"}
vision modules 237 bytes 242863152
model.visual.backbone.embeddings.patch_embedding {'weight': ('BF16', [1024, 1200])}
model.visual.backbone.embeddings.position_embedding {'tensor': ('BF16', [1024, 1024])}
model.visual.backbone.embeddings.class_embedding {'tensor': ('BF16', [1024])}
model.visual.backbone.embeddings.cls_position_embedding {'tensor': ('BF16', [1024])}
model.visual.backbone.ln_pre {'bias': ('BF16', [1024]), 'weight': ('BF16', [1024])}
model.visual.backbone.layers.0.ln_1 {'bias': ('BF16', [1024]), 'weight': ('BF16', [1024])}
model.visual.backbone.layers.0.ls_1.gamma {'tensor': ('BF16', [1024])}
model.visual.backbone.layers.0.mlp.c_fc {'bias': ('BF16', [4096]), 'mul1': ('I32', []), 'suh': ('F16', [1024]), 'svh': ('F16', [4096]), 'trellis': ('I16', [64, 256, 96])}
model.visual.backbone.layers.22.self_attn.out_proj {'bias': ('BF16', [1024]), 'mul1': ('I32', []), 'suh': ('F16', [1024]), 'svh': ('F16', [1024]), 'trellis': ('I16', [64, 64, 96])}
model.visual.proj_in {'mul1': ('I32', []), 'suh': ('F16', [4096]), 'svh': ('F16', [4096]), 'trellis': ('I16', [256, 256, 96])}
model.visual.proj_out {'mul1': ('I32', []), 'suh': ('F16', [4096]), 'svh': ('F16', [2304]), 'trellis': ('I16', [256, 144, 96])}
```

### A.11 Drafter (`mtp.safetensors`) config and tensor shapes (HTTP range read of the header at the pinned revision)

```sh
python3 - <<'E'
import json, re, struct, urllib.request, collections
c = json.load(open("intake/census.json")); print("mtp_config", json.dumps(json.load(open("intake/files/config.json"))["mtp_config"], sort_keys=True))
url = f"https://huggingface.co/{c['repo']}/resolve/{c['revision']}/mtp.safetensors"
rng = lambda a, b: urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
n = struct.unpack("<Q", rng(0, 7))[0]; h = json.loads(rng(8, 7 + n)); print("header bytes", n, "tensors", len(h))
g = collections.OrderedDict()
for k, v in h.items():
    p = re.sub(r"\.\d+(?=\.|$)", ".N", k); g.setdefault((p, v["dtype"], tuple(v["shape"])), 0); g[(p, v["dtype"], tuple(v["shape"]))] += 1
for (p, d, s), cnt in g.items(): print(cnt, p, d, list(s))
tr = {k: v["shape"] for k, v in h.items() if k.endswith(".trellis")}
print("K by trellis:", sorted({s[-1] // 16 for s in tr.values()}))
E
```

```text
mtp_config {"block_size": 8, "head_dim": 128, "hidden_size": 2304, "intermediate_size": 9216, "local_layer_ids": [0, 2], "markov_head_type": "rnn_hidden", "markov_rank": 256, "max_position_embeddings": 1048576, "num_attention_heads": 24, "num_hidden_layers": 3, "num_key_value_heads": 8, "o_bias": false, "q_bias": false, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "sliding_window_size": 1024, "target_layer_ids": [5, 23, 41, 48], "torch_dtype": "bfloat16"}
header bytes 14008 tensors 141
1 fuse_weight BF16 [4, 4]
3 layers.N.dflare_norm.weight BF16 [2304]
3 layers.N.input_layernorm.weight BF16 [2304]
3 layers.N.mlp.down_proj.mul1 I32 []
3 layers.N.mlp.down_proj.suh F16 [9216]
3 layers.N.mlp.down_proj.svh F16 [2304]
3 layers.N.mlp.down_proj.trellis I16 [576, 144, 64]
3 layers.N.mlp.gate_proj.mul1 I32 []
3 layers.N.mlp.gate_proj.suh F16 [2304]
3 layers.N.mlp.gate_proj.svh F16 [9216]
3 layers.N.mlp.gate_proj.trellis I16 [144, 576, 64]
3 layers.N.mlp.up_proj.mul1 I32 []
3 layers.N.mlp.up_proj.suh F16 [2304]
3 layers.N.mlp.up_proj.svh F16 [9216]
3 layers.N.mlp.up_proj.trellis I16 [144, 576, 64]
3 layers.N.post_attention_layernorm.weight BF16 [2304]
3 layers.N.self_attn.k_norm.weight BF16 [128]
3 layers.N.self_attn.k_norm_ctx.weight BF16 [128]
3 layers.N.self_attn.k_proj.mul1 I32 []
3 layers.N.self_attn.k_proj.suh F16 [2304]
3 layers.N.self_attn.k_proj.svh F16 [1024]
3 layers.N.self_attn.k_proj.trellis I16 [144, 64, 64]
3 layers.N.self_attn.k_proj_ctx.mul1 I32 []
3 layers.N.self_attn.k_proj_ctx.suh F16 [2304]
3 layers.N.self_attn.k_proj_ctx.svh F16 [1024]
3 layers.N.self_attn.k_proj_ctx.trellis I16 [144, 64, 64]
3 layers.N.self_attn.o_proj.mul1 I32 []
3 layers.N.self_attn.o_proj.suh F16 [3072]
3 layers.N.self_attn.o_proj.svh F16 [2304]
3 layers.N.self_attn.o_proj.trellis I16 [192, 144, 64]
3 layers.N.self_attn.q_norm.weight BF16 [128]
3 layers.N.self_attn.q_proj.mul1 I32 []
3 layers.N.self_attn.q_proj.suh F16 [2304]
3 layers.N.self_attn.q_proj.svh F16 [3072]
3 layers.N.self_attn.q_proj.trellis I16 [144, 192, 64]
3 layers.N.self_attn.v_proj.mul1 I32 []
3 layers.N.self_attn.v_proj.suh F16 [2304]
3 layers.N.self_attn.v_proj.svh F16 [1024]
3 layers.N.self_attn.v_proj.trellis I16 [144, 64, 64]
3 layers.N.self_attn.v_proj_ctx.mul1 I32 []
3 layers.N.self_attn.v_proj_ctx.suh F16 [2304]
3 layers.N.self_attn.v_proj_ctx.svh F16 [1024]
3 layers.N.self_attn.v_proj_ctx.trellis I16 [144, 64, 64]
1 markov_head.dflare_norm.weight BF16 [2304]
1 markov_head.hidden_out.bias BF16 [2304]
1 markov_head.hidden_out.mul1 I32 []
1 markov_head.hidden_out.suh F16 [256]
1 markov_head.hidden_out.svh F16 [2304]
1 markov_head.hidden_out.trellis I16 [16, 144, 64]
1 markov_head.markov_w1.weight BF16 [201024, 256]
1 markov_head.rnn_proj.bias BF16 [768]
1 markov_head.rnn_proj.mul1 I32 []
1 markov_head.rnn_proj.suh F16 [5120]
1 markov_head.rnn_proj.svh F16 [768]
1 markov_head.rnn_proj.trellis I16 [320, 48, 64]
1 mask_embed BF16 [2304]
1 norm.weight BF16 [2304]
K by trellis: [4]
```

### A.12 Vocabulary, special ids, generation config

```sh
jq -c '{eos_token_id, vocab: .text_config.vocab_size, unpadded: .text_config.unpadded_vocab_size, image_token_id: (.image_token_id // "absent in pack config; HF default 200054")}' intake/files/config.json; jq -c . intake/files/generation_config.json
python3 -c "
import json; t=json.load(open('intake/files/tokenizer_config.json')); d=t.get('added_tokens_decoder',{})
for i in ('199999','200000','200001','200002','200003','200004','200005','200006','200008','200010','200022','200024','200028','200049','200054','200057'):
    print(i, d.get(i,{}).get('content'))
print({k: t[k] for k in t if k in ('eos_token','bos_token','pad_token','model_max_length','tokenizer_class')})"
```

```text
{"eos_token_id":200006,"vocab":201024,"unpadded":200058,"image_token_id":"absent in pack config; HF default 200054"}
{"do_sample":true,"temperature":1.0,"eos_token_id":200006,"pad_token_id":199999}
199999 <|endoftext|>
200000 <|message_user|>
200001 <|message_model|>
200002 <|message_system|>
200003 <|message_tool|>
200004 <|content_text|>
200005 <|content_image|>
200006 <|content_model_end_sampling|>
200008 <|content_thinking|>
200010 <|end_message|>
200022 <|content_tool_error|>
200024 <|content_xml|>
200028 <|begin_of_text|>
200049 <|content_invoke_tool_json|>
200054 <|unused_200054|>
200057 <|content_invoke_tool_text|>
{'tokenizer_class': 'PreTrainedTokenizerFast', 'model_max_length': 1048576, 'eos_token': '<|content_model_end_sampling|>', 'pad_token': '<|endoftext|>'}
```

### A.13 AttnRes sources per mix over the 48 visits (state machine of 3.2 replayed on counts)

```sh
python3 - <<'E'
n0 = 1
B = []; srcs = []
def visit(l, p):
    global B
    glob = (l + 1) % 6 == 0
    attn = 0 if l == 0 else len(B) + 1
    first = l % 6 == 0
    if first: B = B + ["P_in"]
    mlp = len(B) + 1
    srcs.append((l, p, attn, mlp, 3 if (glob and l > 0) else (1 if l > 0 else 0)))
for l in range(6): visit(l, 0)
B0 = list(B); bank = None
for p in range(3):
    if p >= 1: bank = (B[n0 + 1:], "P"); B = list(B0)
    for l in range(6, 18): visit(l, p)
    if p >= 1: pass
for l in range(18, 24): visit(l, 0)
final = len(B) + 1
print("visits", len(srcs), "| max sources in a layer mix", max(max(a, m) for _, _, a, m, _ in srcs), "| final mix sources", final)
print("blocks after prelude", len(B0), "| blocks at the end", len(B), "| message bank per pass: blocks", len(bank[0]), "+ partial")
print("per visit (layer, pass, attn sources, mlp sources, attn queries):", [s for s in srcs if s[1] == 0 and s[0] in (0, 1, 5, 6, 11, 12, 17, 18, 19, 23)])
E
```

```text
visits 48 | max sources in a layer mix 5 | final mix sources 5
blocks after prelude 1 | blocks at the end 4 | message bank per pass: blocks 1 + partial
per visit (layer, pass, attn sources, mlp sources, attn queries): [(0, 0, 0, 2, 0), (1, 0, 2, 2, 1), (5, 0, 2, 2, 3), (6, 0, 2, 3, 1), (11, 0, 3, 3, 3), (12, 0, 3, 4, 1), (17, 0, 4, 4, 3), (18, 0, 4, 5, 1), (19, 0, 5, 5, 1), (23, 0, 5, 5, 3)]
```

### A.14 Log scaling tau at sample positions (fp32 formula of 3.7, n_floor 8192, alpha 0.1)

```sh
python3 - <<'E'
import math, struct
f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]          # round to fp32 after every op, as the fp32 formula does
for pos in (0, 8190, 8191, 8192, 16383, 131071, 262143, 1048575):
    q = max(f32(f32(pos + 1) / 8192.0), 1.0)
    t = f32(1.0 + f32(f32(0.1) * f32(math.log(q))))
    print(pos, "tau =", repr(t))
E
```

```text
0 tau = 1.0
8190 tau = 1.0
8191 tau = 1.0
8192 tau = 1.0000121593475342
16383 tau = 1.06931471824646
131071 tau = 1.2772588729858398
262143 tau = 1.3465735912322998
1048575 tau = 1.4852030277252197
```

### A.15 Every EXL3 tensor satisfies in % 128 == 0 and out % 128 == 0 (trunk census and mtp header)

```sh
python3 - <<'E'
import json, struct, urllib.request
c = json.load(open("intake/census.json")); bad = []; n = 0
for name, v in c["modules"].items():
    t = v["tensors"]
    if "trellis" in t:
        n += 1; k, o = t["trellis"]["shape"][0] * 16, t["trellis"]["shape"][1] * 16
        if k % 128 or o % 128: bad.append((name, k, o))
url = f"https://huggingface.co/{c['repo']}/resolve/{c['revision']}/mtp.safetensors"
rng = lambda a, b: urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
hl = struct.unpack("<Q", rng(0, 7))[0]; h = json.loads(rng(8, 7 + hl)); m = 0
for name, v in h.items():
    if name.endswith(".trellis"):
        m += 1; k, o = v["shape"][0] * 16, v["shape"][1] * 16
        if k % 128 or o % 128: bad.append((name, k, o))
print("trunk EXL3 tensors", n, "| mtp EXL3 tensors", m, "| not multiples of 128:", bad)
E
```

```text
trunk EXL3 tensors 9501 | mtp EXL3 tensors 29 | not multiples of 128: []
```
