# Drafter survey: is there a better block drafter for Inkling-2-Mini than the one we ship?

Research only, 2026-11-23, public pages and APIs read without a login; no weights downloaded (only safetensors headers by HTTP range request, 5,664 bytes each).
Nothing here was run on a node. Every number is quoted from the linked source or marked *derived* (arithmetic on quoted numbers) or `[INFERENCE]`.

## Verdict

**Only ours, and it is the vendor's.** Thinking Machines Lab (TML) ships exactly one drafter for Inkling-2-Mini: `mtp.safetensors` inside the base repo, BF16, 635,801,416 bytes.
The pack's drafter is a quantization of that file (same 55 tensor names, same shapes). No newer revision, no larger or retrained Mini drafter, and no published acceptance length
for the Mini's drafter exists in any source I could read. Every community "Inkling-2-Mini DSpark" repo carries the same file, converted (GGUF, MLX) or unchanged.
The one published BF16 original is the base repo's file; a BF16 drafter would only pay if it raised tokens a round by about 11-15% (section 4), and nothing in the sources
shows 4-bit costs that much. Recommendation: keep the shipped drafter (section 6).

## 1. The base repos

Base repo `thinkingmachines/Inkling-2-Mini`, `main` = `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` (2026-11-23), license Apache-2.0, not gated
([API with blobs](https://huggingface.co/api/models/thinkingmachines/Inkling-2-Mini?blobs=true), [commits](https://huggingface.co/api/models/thinkingmachines/Inkling-2-Mini/commits/main)).

| file | bytes |
|---|---:|
| `model-00001-of-00001.safetensors` (target; F8_E4M3 4,076,863,488 + U8 20,384,317,440 + BF16 1,927,217,664 + F32 247,848 elements) | 19,526,019,168 |
| `mtp.safetensors` (the drafter; LFS sha256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`) | 635,801,416 |
| `tokenizer.json` / `tiktoken/tokenizer.model` | 26,178,678 / 3,386,494 |
| `README.md` (card) / `config.json` / `hf_quant_config.json` / `model.safetensors.index.json` | 54,898 / 2,722 / 67,924 / 109,300 |
| `chat_template.jinja`, `generation_config.json`, `processor_config.json`, `special_tokens_map.json`, `tokenizer_config.json` | 6,868 / 98 / 461 / 529 / 13,235 |

- **History of the drafter:** four commits (`1fb4134b` upload 2026-11-20, `ed40e7da` dtype in config, `b85c3cfa` README, `7a67ae23` eval table). The LFS sha256 of `mtp.safetensors` is the same at
  `ed40e7da` and at `7a67ae23` (`x-linked-etag` header of both), so no newer drafter revision exists. (The pack card's base revision `6e7c8248` is not served; `Sawfwair/Inkling-2-Mini-Q8-Head` also cites it.)
- **Model card text on speculation:** none. `grep -ci "speculat\|mtp\|drafter"` on the 54,898-byte `README.md` returns 0. The card has no acceptance length, no speed-up, no recommended speculative-token count.
- **Config** ([config.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/config.json)) `mtp_config`: hidden 2304, 3 layers, 24 heads / 8 KV, head_dim 128, intermediate 9216, `local_layer_ids [0, 2]`, window 1024,
  rope_theta 1e6, `block_size 8`, `markov_rank 256`, `markov_head_type "rnn_hidden"`, `target_layer_ids [5, 23, 41, 48]`, `torch_dtype bfloat16`. This is exactly the geometry we serve.
- **Where TML documents the drafter:** the [Hugging Face launch blog](https://huggingface.co/blog/inkling2mini) ("SGLang can use a three-layer DSpark draft ... reads the inputs to decoder visits 5, 23, and 41, plus the final normalized state at index 48";
  a throughput/time-per-token chart on one B300 at concurrency 1/4/16/64, image only, no readable numbers; "a DSpark drafter is also included for additional speed-up at a small memory cost"),
  the [vLLM blog](https://vllm.ai/blog/2026-11-20-inkling2-mini) ("DSpark draft from inside the loop", reads four visits) and the [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)
  (`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`; "DSpark acceptance depends on the prompts and the sampling temperature: random-token prompts accept almost nothing").
  The [TML announcement](https://thinkingmachines.ai/news/inkling-2-mini/) does not mention the drafter.
- **Siblings by TML** ([author listing](https://huggingface.co/api/models?author=thinkingmachines)): `Inkling-2-Small` (2026-11-04), `TML-d24-varloop-research` (2026-11-18), `Inkling`, `Inkling-NVFP4`, `Inkling-Small`, `Inkling-Small-NVFP4`, `Inkling-2-Small-MXFP8`.
  - **Inkling-2-Small** (`885d78bf`, 2026-11-20, [config](https://huggingface.co/thinkingmachines/Inkling-2-Small/resolve/main/config.json)): `mtp.safetensors` 3,423,429,800 bytes, `mtp_config`: hidden 4608, **5 layers**, 48 heads, intermediate 18432,
    `local_layer_ids [0, 1, 3, 4]`, `block_size 8`, `markov_rank 256`, `rnn_hidden`, **8 taps** `[5, 11, 17, 23, 29, 35, 41, 48]`, BF16. So yes: the Mini's drafter is the same architecture family (block 8, Markov rnn head, taps over the 48 visits),
    scaled down (3 layers, 4 taps, width 2304). No newer revision of either.
  - **TML-d24-varloop-research** ([repo](https://huggingface.co/thinkingmachines/TML-d24-varloop-research), Apache-2.0): its own `mtp.safetensors`, 635,801,416 bytes but a different LFS sha256 (`905d5486...c76c7`): a drafter for the variable-loop model, not ours; the vLLM blog says "DSpark requires the default [3 passes], since the draft reads fixed layer visits".
  - **Inkling / Inkling-Small** (the July 2026 family, not the "2" models): different targets (66 and 42 layers); see the table below.

## 2. Is the pack's `mtp.safetensors` a quantization of the base file?

Yes. I read both headers by range request (`bytes=0-7`, then the JSON header) and compared them by script:

- Base: 55 tensors, all BF16, 635,795,744 bytes of data ([header](https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/mtp.safetensors)). Pack at `44c78c9d...` ([file](https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw/resolve/44c78c9d229b43dd45e41506071b9f5f939bed44/mtp.safetensors)): 141 tensors, 236,572,756 bytes on disk (`intake/manifest.tsv`).
- **Quantized by the pack's maker (29 linears, EXL3 trellis, K = 4 by the trellis shapes, each with `suh`, `svh`, `mul1`, `trellis`):** per layer `mlp.{gate,up,down}_proj`, `self_attn.{q,k,v,o}_proj`, `self_attn.{k,v}_proj_ctx` (9 x 3 = 27),
  plus `markov_head.rnn_proj` (base `[768, 5120]`, pack trellis `[320, 48, 64]`) and `markov_head.hidden_out` (base `[2304, 256]`, trellis `[16, 144, 64]`). Example: `layers.0.mlp.down_proj` base `[2304, 9216]` -> trellis `[576, 144, 64]` = 2304/16 x 9216/16 x (16 x 4).
- **Kept BF16, same name and shape as the base (checked one by one: no base tensor is missing from the pack):** `fuse_weight [4,4]`, `mask_embed`, `norm.weight`, all norms (`input_layernorm`, `post_attention_layernorm`, `dflare_norm`, `q_norm`, `k_norm`, `k_norm_ctx`),
  `markov_head.markov_w1.weight [201024, 256]` (102,924,288 B), `markov_head.dflare_norm.weight`, the two biases (`rnn_proj.bias`, `hidden_out.bias`).
- **Bytes:** base layers 3 x 174,587,904 = 523,763,712 B BF16 (*derived from the shapes*) against 124,174,848 B 4-bit in the pack (`docs/KERNEL-AUDIT.md`): the layers shrink 4.2x, the BF16 `markov_w1` stays.
- **A BF16 original is published:** the base repo's file, 635,801,416 B (0.592 GiB). Several mirrors carry it unchanged (e.g. [Blackfrost-AI/Inkling-2-Mini-Derisked](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked): "byte-identical to upstream", same sha256;
  [Sawfwair/Inkling-2-Mini-Q8-Head](https://huggingface.co/Sawfwair/Inkling-2-Mini-Q8-Head); [Youssofal/Inkling-2-Mini-MTPLX-*-FP16](https://huggingface.co/Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed-FP16)). Unsloth also hosts a bit-exact BF16 GGUF of it
  ([dspark/dspark-Inkling-2-Mini-BF16.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/dspark), 647,792,800 B).
- **Same drafter in the `REXL3` org pack** ([REXL3/Inkling-2-Mini-EXL3-4.23bpw-SAGE-REXL3](https://huggingface.co/REXL3/Inkling-2-Mini-EXL3-4.23bpw-SAGE-REXL3), its `mtp.safetensors` is 236,572,756 B, the pack's size; its card says "DSpark has 29 local 4-bpw projections and retains BF16 state" and the Markov fold is a separate 102,830,308 B BF16 file).
- **Memory of a BF16 drafter, derived:** weights 635.8 MB against about 227 MB (124.2 MB layers + 102.9 MB `markov_w1`); +400 MB at the layers alone (3 x 174.6 MB - 124.2 MB). One proposal reads the weights once (8 rows): +400 MB is +1.71 ms at 233.8 GB/s (derived; a floor), against the +3.7 ms the lead estimated for a 4x read and any dequant-free-kernel inefficiency.
  The BF16 kernel branch exists for the Small; the fold of the unembed product is independent of the layer dtype.

## 3. Every other drafter, speculator or MTP head found

Search: [HF `Inkling`](https://huggingface.co/api/models?search=Inkling&limit=300), `Inkling-2-Mini`, `inkling2`, `inkling-2`, `dspark`, `mtp`, `speculator`, `drafter`, `inkling2_loop`; GitHub repos/issues/PRs; vLLM source `vllm/models/inkling2/`.
"Fit" = hidden 2,304, vocab 201,024 (200,058 valid), taps [5, 23, 41, 48], block 8.

| hit | what | size | license | format | fits the Mini? | needs another engine path? |
|---|---|---:|---|---|---|---|
| [thinkingmachines/Inkling-2-Mini `mtp.safetensors`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | the vendor's drafter | 635,801,416 | Apache-2.0 | BF16 safetensors | yes, it is the reference | BF16 branch only |
| pack `vcruz305/...` `mtp.safetensors` (ours) | its 4-bit EXL3 quantization | 236,572,756 | Apache-2.0 | EXL3 K=4 + BF16 | yes (shipped) | no |
| [REXL3/Inkling-2-Mini-EXL3-4.23bpw-SAGE-REXL3](https://huggingface.co/REXL3/Inkling-2-Mini-EXL3-4.23bpw-SAGE-REXL3) | same quantization, RTX 3090 build | 236,572,756 | Apache-2.0 | EXL3 | yes | no |
| [unsloth/Inkling-2-Mini-GGUF `dspark/`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/dspark) | Q8_0 (402,278,560) and BF16 (647,792,800) GGUF of the vendor's file | 0.40 / 0.65 GB | Apache-2.0 | GGUF | yes, same weights ("`Q8_0` ... accepts drafts at the same rate as `BF16` in our tests", no numbers) | convert only |
| [shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter) | MLX 4-bit (group 64), 178,875,695 B each; **two distilled variants** (`..._5p4gb_v2`, `..._8p9gb_v2`) trained against that author's own quantized targets | 178.9 MB x3 | Apache-2.0 | MLX safetensors | stock one yes (same weights); distilled ones only help the author's quantization | MLX only; no CUDA path |
| [Youssofal/Inkling-2-Mini-MTPLX-*(-FP16)](https://huggingface.co/Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed-FP16) (6 repos) | MLX packs; `mtp.safetensors` 635,801,416 B (the vendor file; the FP16 variants store it as float16) | 0.64 GB | Apache-2.0 | MLX | same weights | MLX |
| [Sawfwair/Inkling-2-Mini-Q8-Head](https://huggingface.co/Sawfwair/Inkling-2-Mini-Q8-Head) | "official DSpark draft", `mtp.safetensors` "unchanged" | 635,801,416 | Apache-2.0 | safetensors | same | MLX |
| [Blackfrost-AI/Inkling-2-Mini-Derisked](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked), `JoaoZaokk`, `TokenAI-zer`, `vanch007`, `netprojects`, ... (about 60 repos tagged `dspark`/`mtp`) | derivatives that re-host the vendor's drafter (Blackfrost: "byte-identical", sha256 matches) | 635,801,416 | mixed (`license:other` for Blackfrost) | safetensors/GGUF/MLX | same weights | n/a |
| [JMNI-Labs/Inkling-2-Mini-FLARE](https://huggingface.co/JMNI-Labs/Inkling-2-Mini-FLARE) | NVIDIA hyperflare package for one DGX Spark: "NVFP4 DSpark draft/Markov heads", "MXFP8 ... DSpark body" | n/a (branch `gb10-tp1`, not read) | Apache-2.0 | Flare package | same weights, requantized | different engine |
| [RadixArk/Inkling-DSpark-Preview](https://huggingface.co/RadixArk/Inkling-DSpark-Preview) (2026-07-22) | SpecForge DSpark for the **July `Inkling-NVFP4`** (66 layers): 5 layers, hidden 6144, taps [5,17,35,47,59], block 7; mean acc_len 3.6649 (T=0, 9 sets) | 3,477,573,874 | none stated | BF16 safetensors | **no** (hidden 6144, 66-layer target) | n/a |
| [RadixArk/Inkling-Small-DSpark](https://huggingface.co/RadixArk/Inkling-Small-DSpark) (2026-08-06, + `-Preview`) | DSpark for the July `Inkling-Small-NVFP4` (42 layers): 6 layers, hidden 4096, taps [1,6,12,17,23,28,34,39], block 7; mean acc_len 3.8226 (T=0, 12 sets) | 2,789,674,138 | none stated | BF16 | **no** | n/a |
| [modal-labs/Inkling-NVFP4-DFlash](https://huggingface.co/modal-labs/Inkling-NVFP4-DFlash), [inference-optimization/thinkingmachines-Inkling-speculator.dflash](https://huggingface.co/inference-optimization/thinkingmachines-Inkling-speculator.dflash) | DFlash drafters (block 16) for the July `Inkling-NVFP4`; acc_len 2.9-5.0 | 6,945,962,376 / 11,886,353,152 | Apache-2.0 / none stated | BF16/FP8 | **no** (hidden 6144) | n/a |
| [inference-optimization/Inkling-0.92B-MTP](https://huggingface.co/inference-optimization/Inkling-0.92B-MTP), [nm-testing/Inkling-Dense-54M-MTP](https://huggingface.co/nm-testing/Inkling-Dense-54M-MTP) | test fixtures for the July architecture | n/a | Apache-2.0 | safetensors | no | n/a |
| [Wa1k3r/Inkling-2-Mini-8Q-RCO-GPTQ](https://huggingface.co/Wa1k3r/Inkling-2-Mini-8Q-RCO-GPTQ) ("draft-head") | a GGUF with a bundled `.img.xz` disk image: not a drafter checkpoint I could identify | 18.3 GB | Apache-2.0 | GGUF | not checked, no drafter file | n/a |
| [lookevink/inkling2-loop-retrofits](https://huggingface.co/lookevink/inkling2-loop-retrofits), [brandonmusic/...Flare](https://huggingface.co/brandonmusic/Inkling2-mini-variable-looped-transformer-Flare) | loop adapters for other models, not drafters | n/a | Apache-2.0 / other | n/a | no | n/a |

Code, not weights:
- **vLLM** (`vllm/models/inkling2/` at `2b55f82b085a765e4c813691d505563b397b4173`: `speculator.py`, `speculator_config.py`, `configs.py`): [`speculator.py`](https://raw.githubusercontent.com/vllm-project/vllm/2b55f82b085a765e4c813691d505563b397b4173/vllm/models/inkling2/speculator.py) keeps the four raw taps, runs the recurrent Markov head one step per draft
  (`markov_step`), asserts `not enable_adaptive_verification` and no draft top-k, and takes the draft's base logits "from a bf16 GEMM, unlike the target's fp32 head". Draft length is fixed at 7 by the checkpoint (confirmed by [sxuff's write-up](https://github.com/sxuff/inkling-2-mini-spark)).
  [vLLM speculators](https://github.com/vllm-project/speculators) has `dspark`, `dflash`, `eagle3`, `mtp` model directories but I found no Inkling-2-Mini checkpoint there.
- **llama.cpp** [PR #32218](https://github.com/ggml-org/llama.cpp/pull/32218) "Inkling 2 DSpark support": open, not merged (state read via the API, 2026-11-23); Unsloth's fork runs the same GGUF drafters (`--spec-draft-n-max 3` default, Studio uses 2). No new drafter weights.
- **Other engines on one GB10:** [sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) (vLLM + DSpark, INT8 draft kernels, 66.7 tok/s median single stream vs 60.5 stock; text byte-identical; "tokens per step" 3.10 -> 3.16 with the INT8 drafter);
  [thefullctx/inkling-2-mini-sglang-dgx-spark](https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark); [tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark). All use the vendor's drafter.
- **No EAGLE, Medusa or retrained drafter for the Mini** turned up in the HF searches `eagle`, `medusa`, `speculator`, `drafter`, `mtp`, `dspark` combined with `Inkling`.

**What DSpark is:** [DSpark (DeepSeek and PKU, arXiv 2607.05147)](https://www.alphaxiv.org/abs/2607.05147): a parallel block backbone plus a lightweight sequential Markov (or gated RNN) head, and a confidence head for variable-length verification; "the draft shares and freezes the target's `embed_tokens` and `lm_head`"
([NVIDIA recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding)). Reported: macro-average accepted length +30.9% / +26.7% / +30.0% over Eagle3 and +16.3% / +18.4% / +18.3% over DFlash on Qwen3-4B / 8B / 14B (abstract/introduction), 60-85% faster per-user generation than MTP-1 on DeepSeek-V4-Flash.
TML's own training recipe for the Mini's drafter is **not published** that I could find (the card, announcement and both launch blogs do not describe it). The RadixArk cards show what a SpecForge recipe looks like for the July models (400K target regenerations, loss `0.1 CE + 0.9 L1 + 1.0 BCE`, 512 anchors, block 7), not what TML did.
Whether the Mini's drafter was trained on the NVFP4/MXFP8 target specifically is therefore `[INFERENCE]` (the July RadixArk drafters require "the exact matching target checkpoint"). Our target is a different, lossier quantization of it (publisher: top-1 agreement 98.33%, mean KLD 0.0088).

## 4. What quantizing the drafter costs, and what a BF16 drafter must gain

**Evidence on the 4-bit cost (none is an Inkling EXL3 ablation):**
- No source publishes acceptance for the vendor BF16 drafter against our 4-bit EXL3 one, and the pack's card (`intake/files/README.md`) has no drafter acceptance numbers (grep: only footprint and KV lines mention the draft).
- Unsloth: Q8_0 projections (worst relative RMSE 0.62%) "accepts drafts at the same rate as BF16 in our tests" (no numbers) ([dspark README](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/dspark/README.md)).
- sxuff, vLLM on a GB10: INT8 draft layers/head "accept slightly more, not less": 3.10 -> 3.16 tokens a step, outputs equal in 9 of 9 ([README](https://github.com/sxuff/inkling-2-mini-spark)).
- LookAtMyQuant, a *different* model (Qwen3.5-4B DSpine drafter): 4-bit drafter "kept 4.35 tokens per check against 4.38" ([README](https://github.com/ShinMegamiBoson/lookatmyquant)).
- The one measured gain from touching Mini's drafter is **retraining against the quantized target**, not precision: the author's `mtp_4bit_8p9gb_v2.safetensors` "lifts held-out `tpc_bl8` 3.26 -> 3.53" (+8.3%) on his 8.9 GB pack ([drafter card](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter)); that is a different pack and a MLX runtime, and the training data and code for it are not in the repo.

**Break-even arithmetic (derived from `bench/records/2026-11-23-spec-final/` and `-spec-speed/`):**
- Round at 4K from the rank: verify 29.36 ms + proposal 8.31 ms = 37.7 ms (`-spec-speed`). End to end, round = tokens a round / drafted tok/s: chat 41.5 ms, code 48.6, JSON 52.4, tool 52.8, reasoning 45.0 (`-spec-final`).
- A proposal that costs +3.7 ms is +9.8% on the rank round and +7.0% to +8.9% end to end (code 48.6 -> 52.3, reasoning 45.0 -> 48.7): tokens a round must rise by about 7-9% on the kind in question just to keep today's speed-up. With the byte-only floor (+1.7 ms) it is +3.3% to +4.2%.
- **100 tok/s:** serial 52.2 tok/s needs a mean speed-up of 100 / 52.2 = 1.916. Today's mean of the four kinds is 1.79 only for the set chat, code, JSON, tool call (0.899 + 2.244 + 1.636 + 2.379) / 4 = 1.790 -> 93.4 tok/s (derived; this is the set that reproduces 1.79, `[INFERENCE]` that it is the benchmark's four kinds).
  The gap is +7.0% in tokens a round at equal round cost. A BF16 drafter at +3.7 ms needs **+15.0%** (chat 1.95 -> 2.24, code 5.71 -> 6.57, JSON 4.43 -> 5.10, tool 6.47 -> 7.44 of a hard maximum of 8 a round); at +1.7 ms it needs +10.7%.
  The drafter would have to be close to a different model, not a more precise copy: per-position acceptance is already 0.91-0.95 at p1 on code and tool calls and 0.56 on chat (`-spec-speed`), so the headroom is in chat and reasoning, and BF16 precision is not what limits those.
- Verdict: the evidence supports "4-bit costs about nothing", so a BF16 drafter would gain a few percent of acceptance at most (`[INFERENCE]`, no ablation) against a cost of +3.3% to +8.9% in round time.

## 5. Other speculative methods for this family

- **Copy / n-gram:** already in the engine (`copy` drafter). Suffix decoding ([SuffixDecoding, Snowflake, arXiv 2411.04975](https://arxiv.org/abs/2411.04975): suffix trees over prompts and past outputs, "up to 5.3x" on agentic SWE-Bench / Text-to-SQL, per its abstract) is the same family; no Inkling-specific result.
- **Tree drafts for semi-AR drafters:** [TreeSpark, arXiv 2609.22098](https://arxiv.org/pdf/2609.22098) (title read only); the LookAtMyQuant card says it offers "opt-in 8-node causal trees" for the Mini and "opt-in tight-margin greedy (+31-36%)", which is **bit-changing** (not A lane). Trees need a tree-masked verify the engine does not have.
- **Self-speculation from fewer loop passes, published for looped models in general:** [LoopSpec (arXiv 2609.17184)](https://arxiv.org/abs/2609.17184): training-free, drafts from early recurrent states and pipelines draft and verify, "up to 6.83x" on "diverse Looped Transformers";
  [WaveFront Decoding (2609.23033)](https://arxiv.org/pdf/2609.23033); [LoopAhead (2610.21594)](https://arxiv.org/abs/2610.21594); [LoopFill (2611.19649)](https://arxiv.org/abs/2611.19649) (5.85x on Ouro-2.6B, 5.13x on Huginn-3.5B); [DLoop (2610.07659)](https://arxiv.org/pdf/2610.07659).
  These are validated on Ouro and Huginn, models whose pass p+1 consumes pass p's output with one shared KV per layer; Inkling-2-Mini differs (every pass starts from the prelude output and reads gated messages; 48 KV and conv caches, one per pass, [vLLM blog](https://vllm.ai/blog/2026-11-20-inkling2-mini)).
  What the Mini's own sources support: the **variable-loop research checkpoint** runs 1 pass at 1.51 ms against 3.85 ms for 3 passes per decode step (batch 1, GB200, 4K context, forward only), and its no-CoT time horizon is 16 s at 1 pass against 37 s at 3 ([TML](https://thinkingmachines.ai/news/inkling-2-mini/)); "Both options are rejected on fixed-loop checkpoints such as Inkling 2 Mini".
  So a 1- or 2-pass draft from the shipped Mini weights is not trained for, has no published acceptance, and is only an idea: whether a 1-pass state (visit 18, then the 6 coda layers = 24 visits) predicts the 3-pass token is `[INFERENCE]` and unmeasured. It would also need the draft's passes to write their own per-pass KV (the cache is per pass), which the serial engine path does not do.
  The obvious reading of the model code that supports it: the DSpark taps already read visits 5, 23, 41 and 48, i.e. the prelude, the pass-2 and pass-3 inputs and the final state; nothing in the sources reads visit 17.
- **Variable verification length (the "S" of DSpark's scheduler):** the vendor stack fixes it (vLLM asserts no adaptive verification for Inkling2); our learnt `plan` budget already does the job (`docs/DECISIONS.md`).

## 6. Recommendation

**Keep the shipped drafter.** The pack's 4-bit `mtp.safetensors` is the vendor's only drafter for the Mini, quantized tensor-for-tensor (29 linears to K = 4, norms, masks, `markov_w1` and biases stay BF16); there is no newer revision, no larger Mini drafter, and no other checkpoint that fits (hidden 2,304, taps [5, 23, 41, 48], block 8).
The only alternative the survey found is the vendor's own BF16 file, which we would have to fetch and run through the BF16 branch: `hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --revision 7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6 --local-dir <dir>` (635,801,416 B, sha256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`; I did not download it).
Against it: the engine would need a BF16 load path for the drafter's nine per-layer linears and the two Markov linears (`inkling2_dspark.cu` / `dspark.rs` have the Small's BF16 path and the Mini kernels carry a bf16/fp32 branch, per the assignment; I did not read the engine), the fold of the unembed product re-run, and +400 MB resident (derived);
the gain would have to be +10.7% to +15.0% tokens a round for 100 tok/s (section 4), and every source that compares 4/8-bit with 16-bit drafters (Unsloth Q8, sxuff INT8, LookAtMyQuant on Qwen3.5) found no acceptance loss. Checks: the proposal-side batteries (`crates/cuda/tests/inkling2_dspark.rs`) would need BF16 references; the A lane's byte-exactness stays (a drafter only changes what is proposed, the target verifies every token) but the 20 prefix cuts and the 37 graph windows that carry drafter state must be re-run (`[INFERENCE]`, engine not read).
If the owner still wants an experiment, it is an A/B on node-2 (BF16 load, same plan budget, `specbench.py`, accept ratios per kind) with the break-even above as the pre-registered bar; expected outcome `[INFERENCE]`: no pay.
The lever that has measured evidence (+8.3% tokens a check on the LookAtMyQuant pack) is **distilling the drafter against our own target's outputs** (retraining), which needs a SpecForge-style training run and data the survey did not find published for the Mini; that is a research project, not a download.
For the 100 tok/s goal the cheaper levers are on the round (verify 29.4 ms, proposal 8.3 ms, engine 3-12 ms of unattributed time per `-spec-speed`), not on the drafter's precision.

## Not verified here
- Acceptance length of the vendor's drafter in any source (the card, the blogs and the vLLM page publish none; the SGLang chart on the HF blog is an image).
- Whether TML trained the drafter on the exact NVFP4/MXFP8 target (no source).
- Contents of branch `gb10-tp1` of `JMNI-Labs/Inkling-2-Mini-FLARE`, of `Strata` repos, and of the engine files named in section 6.
- TreeSpark beyond its title; SuffixDecoding and the looped-model papers beyond their abstracts; the DSpark paper beyond its first 300 lines (abstract, introduction, drafter architecture).
