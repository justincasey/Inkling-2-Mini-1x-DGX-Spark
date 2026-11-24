# Kernel audit: REXL3 kernels against Inkling-2-Mini

**Summary.** Read-only audit of the engine tree `~/projects/rexl3-inkling2-loop` (`inkling2-loop` @ `82bbda0` = upstream `main` `02ef8cb` + the Small family) against the Mini (`Inkling2LoopForConditionalGeneration`, TP1, EXL3 mixed K). Nothing was built, run or edited; every number below is a file:line in that tree, a value in the pack census or config, or the output of a script named next to it. Of 38 rows in the master table (section 7) **13 are `as-is`, 20 `parameterise`, 5 `new`**. **Plan trigger: mixed K inside one MoE layer needs NO new CUDA kernel.** K is a runtime argument per launch of the existing grouped decode and prefill kernels; the engine already splits a layer into width bands (`cuda/moe.rs:652-701`). What is missing is host composition in the Small's softcap expert path (decode and prefill, bands instead of one K) and one template instantiation (`mul1` + softcap in the prefill gate/up kernel, which exists for `mcg` only). Biggest real work: attention at 24/8/3 heads (the prologue needs 1,280 threads, CUDA allows 1,024; the int8 page header is hard-wired to 2 KV heads and shared with Nemotron), the drafter at H = 2304, and the new loop ops. CYBERFROST adds no per-projection-K mechanism that `main` lacks (its delta on the files that matter is speed work, whose "same bits" claims I did not verify); nothing from it is required for correctness. Items not verified by reading code are marked UNKNOWN (section 11).

Paths: `cuda/` = `crates/cuda/src/`, `csrc/` = `crates/cuda/csrc/`, `fam/` = `crates/model/src/families/inkling2/`, all in the loop tree unless prefixed `cf:` (read-only worktree `~/projects/rexl3-cyberfrost`, branch `cyberfrost` @ `a644961`). "Spec" = the spec lane's `docs/PORT-SPEC.md` (`~/projects/inkling2-mini-spec`, `a6ff63b`), cited only for model semantics I did not derive from code.

## 1. Facts checked first

Run against `intake/census.json` (python over `modules[*].tensors.trellis.shape[2] / 16`):

| Fact | Result |
|---|---|
| routed experts 4-bit / 5-bit | 7,944 / 1,272 tensors (matches the plan) |
| expert classes by (gate, up, down) | (4,4,4) 2,372; (4,4,5) 414; (5,5,5) 286 experts of 3,072. gate K == up K for all 3,072 (0 mismatches); no expert has a 5-bit gate but 4-bit down |
| bands per layer | 1 band: layers 0-5 and 20 (7 layers); 2 bands: 18, 19, 21, 23; 3 bands: 6-17 and 22 (13 layers). Smallest band of a layer: 1 expert (layers 6, 21, 22, 23) |
| 6-bit tensors | `wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, **`gate_attn`**, **`wr_du`** (24 layers each), `model.llm.unembed`, 23 x 6 vision block linears + `proj_in`, `proj_out`. Not in the plan's list: `gate_attn` and `wr_du` |
| dense, not EXL3 | embed BF16 `[201024, 2304]`; router `mlp.gate` BF16 `[128, 2304]` + F32 bias; `attn_sink_proj` BF16 `[24, 2304]`; controllers BF16; convs BF16; AttnRes queries F32; vision biases **BF16** inside the EXL3 groups |
| codebook | every EXL3 tensor carries a `mul1` marker (Small's tests assert `mcg`: `fam/config.rs:652`) |
| unembed shape | trellis `[144, 12568, 96]`, `svh` 201,088 = 64 columns past `vocab_size` 201,024 |
| drafter K = 4 | UNKNOWN from census (`mtp.safetensors` is not in it). Byte arithmetic is consistent: 3 layers at 4 bit (124,174,848 B) + BF16 `markov_w1` (102,924,288 B) = 227,099,136 B against 236,558,740 B (FIT total minus census total) |
| `config.json` `quantization_config` | `{"quant_method": "exl3"}` only: no `bits`, `head_bits`, `codebook`, `out_scales`, `note` |

## 2. Plan trigger, explicit

> *Does mixed K need NEW MoE kernels?* **No.**

- Decode: `gb_moe_gemv` takes `bits` as an argument and dispatches `2*bits == K2` over K2 = 2..16 and codebooks 0..2 (`csrc/moe_experts.cu:488-491, 539-558`). One K per launch; gate and up share a launch (`mats=2`), down is its own launch with its own K (`MoeExperts.bits_gu`, `.bits_d`: `cuda/moe.rs:707-708`). Gate/up epilogue and the down combine do not depend on K.
- Prefill: `gb_moe_pf_gateup` / `gb_moe_pf_down` likewise take `bits` and instantiate every K2 x codebook (`csrc/moe_prefill.cu:648-651, 761-790, 847-862`).
- Host: `MoeExperts::from_groups` already partitions a mixed layer into bands keyed by (bits_gu, bits_d, codebook), each with its own pointer tables, stacked scales and `local_of` map (`cuda/moe.rs:652-701, 779-989`); `run_prefill` loops the bands (`cuda/moe_prefill.rs:615-740`); a rank-owned variant with decode exists (`cuda/moe_subbanks.rs`, built for RED-SNOW).
- What does not exist for the Mini: (1) decode over bands with the **softcap** epilogue and f32 weights (`cuda/inkling2/moe.rs:108` demands `bands.len() == 1`; `cuda/moe.rs:1363` refuses bands in `run`; the subbank path hard-codes the clamp epilogue, `cuda/moe_subbanks.rs:377-391`); (2) prefill over bands with the softcap epilogue (`cuda/moe_prefill.rs:658` and `:712` call the SiLU and non-bf16 variants); (3) `mul1` for the prefill softcap kernel (`csrc/moe_prefill.cu:800, 818-824` accept codebook 1 only, and `cuda/inkling2/moe.rs:273` refuses anything but `Mcg`).
- **Enough host-side grouping:** per layer, up to 3 launch groups, each the Small's 7-launch decode chain (`cuda/inkling2/moe.rs:142-250`: remap, group, rot_in, gemv gate/up, softcap epilogue, gemv down, down-combine) on the band's own tables, down-combine writing `y_pairs`; then one `gb_moe_pf_combine_f32_bf16` over the global picks (`csrc/moe_prefill.cu:902-910`, "the decode `gb_moe_down_combine_f32w_bf16`'s twin"). All launchers exist (`cuda/inkling2/moe.rs:32-64`).
- **Cost of splitting (derived from code, not measured):** launches a decode token: a 1-band layer 7, 2 bands 15, 3 bands 22 (7 per band + 1 combine, as `MoeSubbanks::decode_launches`, `cuda/moe_subbanks.rs:711-713`). With the census band counts and the 48-pass schedule: 6 x 7 (layers 0-5) + 36 x 22 (layers 6-17, three passes each) + (15 + 15 + 7 + 15 + 22 + 15) (layers 18-23) = **923 launches a token against 336** for a uniform bank. DRAM bytes read are unchanged (a pair reads its own expert once); the grouping, rotation and epilogue work is partitioned, not repeated. Time cost: UNKNOWN until measured on a node (graphs are in the engine, `cuda/graph.rs`). Prefill repeats remap + plan twice a band (`cuda/moe_prefill.rs:628-639, 680-690`), negligible against 4,096-row GEMMs but a candidate to drop later.
- **Later speed option (not needed to boot):** FrostMoE (`csrc/fmoe.cu`) carries `ktab[e][3]` = K2 per expert and projection (`:1763-1771, 648, 777`) and would run all three classes in one launch, but it is `mul1`-only, routes internally (softmax, or a 64-expert-group router), applies SiLU with a clamp (`:731-736`), requires a shared-expert gate in prefill (`:1965`) and, at H = 2304, has no fast `KC` (`H/128 = 18`; its shared-memory switch covers KC 2, 4, 5, 8, 10, 16, 20 only: `:1809-1837`). Different reduction order from `moe_experts.cu`, so adopting it after the pinned reference is a bit change (plan 7.3, owner approval).

## 3. (a) MoE with mixed K

| Piece | Behaviour | Source |
|---|---|---|
| K per launch | runtime `bits` and `codebook` args; template K2 inside | `csrc/moe_experts.cu:539-558`; `csrc/moe_prefill.cu:761-790` |
| K per projection | gate/up one K (fused launch), down another | `cuda/moe.rs:707-708`; Mini needs only (4,4,4), (4,4,5), (5,5,5) |
| gate != up | refused by `from_groups` (`:805`); split-width exists only in subbanks (decode only, `SPLIT_PREFILL`, `cuda/moe_subbanks.rs:44, 497-499`) | not needed: gate == up for every Mini expert |
| plans | `gb_moe_default_plan(k, n, bits, mats)` is shape-only (`csrc/moe_experts.cu:503-512`). Script mirror: gate/up 2304 x 1152 -> (sk 3, wk 4); down 1152 x 2304 -> (1, 4); both divide | `cuda/moe.rs:624-646` |
| block counts | I = 1152 is 9 blocks of 128, H = 2304 is 18: not multiples of 4. Decode `rot_in` / epilogue guard `blk*128 >= K/N` (`csrc/moe_experts.cu:140, 339`), prefill `rot_in` guards at `csrc/moe_prefill.cu:320`, GEMM grids use `N/128` (`:608, 638`) | safe by reading; GPU run still needed |
| decode, Small | single K from the config (`fam/bind.rs:474`), `softcap_args_ok` requires one band, equal gate/up bits, no split (`cuda/inkling2/moe.rs:101-115`) | |
| prefill, Small | single band, `mcg` only (`cuda/inkling2/moe.rs:256-278`; kernel `csrc/moe_prefill.cu:796-830`) | |
| bank build | `from_groups` / `from_exl3`: one device allocation per tensor (24 x 128 x 3 = 9,216 for the Mini), mixed layers one table set per band, legacy fields mirror band 0 (`cuda/moe.rs:907-988`). `from_region_subbank` + `SubbankRegion::stacked` is the region-based alternative (`cuda/moe_subbanks.rs:130-209`) | boot cost of either: UNKNOWN |
| ownership | not needed at TP1 (`own: None`); the band `local_of` tables give the remap the same shape as the rank remap | `cuda/moe.rs:662, 901` |

**CYBERFROST** (`cyberfrost` @ `a644961`; 149 commits past the merge-base `e96c48e` with `main`, 47 on `main` not in it; `git log` counts): per-projection K there is the same thing `main` has. Dense groups take `Bits` from the trellis shape per tensor (`cf:crates/model/src/families/qwen_flash_next/weights.rs:180-196`, codebook fixed `Mul1`); expert tables are `ExpertTables` + `ktab` K2 per (expert, projection) (`cuda/fn_moe.rs:65-112`; `cf:.../qwen_flash_next/device.rs:684`; MiniCPM-V on `main` uses it too, `crates/model/src/families/minicpmv4_7/device.rs:29`). The branch's delta for the files that matter: decode EXL3 ring of bulk copies (`506b3a8`), decode input rotation folded into the linear (`5d08e00`, `gb_exl3_set_fuse`, `gb_exl3_fused_in`), strips of a row group together in the large-row linear (`4870de1`, `gb_exl3_big_set_second_form`), FrostMoE route/tile-decode/L2-hint work, a `moe_prefill_v2.cuh` expert GEMM, prefill chunk 8,192. The doc comments say "same bits" for the fuse and second-form hooks; I did not verify that (UNKNOWN). All of it is Flash-Next-shaped (H = 2560, shared expert, SiLU). **Carry-over: none needed; candidates for Phase 7** are `506b3a8`, `5d08e00` and `4870de1` as byte-preserving speed.

## 4. (b) Dense EXL3 linear

| Need | Covered by | Source |
|---|---|---|
| M = 1 decode row, 8-row verify window | `Exl3Weights::forward`, 1..=128 rows, row-invariant | `csrc/exl3_linear.cu:35, 694`; `cuda/exl3_linear.rs:41, 704-744` |
| prefill rows (4,096) | `forward_big`: 1..65,535 rows, each row bit-equal to the decode path alone or in any window; workspace `ceil(m/16)*16*K*2` | `cuda/exl3_big.rs:1-22`; `csrc/exl3_big.cu:2-3` |
| K = 6 (and 4, 5) | `bits` 1..8 all codebooks | `csrc/exl3_linear.cu:461-463` |
| shapes | K, N multiples of 128: all Mini tensors qualify (2304, 3072, 1024, 384, 4096, 201,088). Script mirror of `plan_for` (`:433-446`): q/gate 2304 x 3072 (4,4); k/v 2304 x 1024 (4,4); `wr_du` 2304 x 384 (4,4); `wo` 3072 x 2304 (4,4); unembed 2304 x 201,088 (1,8); vision 1024 x 1024 (2,4), 1024 x 4096 (2,4), 4096 x 1024 / x 4096 / x 2304 (8,4) | existing GPU tests use GLM / MiniCPM shapes (`cuda/exl3_linear.rs:1646-1677`), not these: UNKNOWN until golden |
| bias | optional, uploaded as fp16 (`cuda/exl3_linear.rs:599`); the vision biases are BF16 in the pack | exact unless a value leaves fp16's range: UNKNOWN |
| output | `y` is contiguous `[m, n]` (`:694-699`), no leading dimension | attention takes separate q / r / gate / sink pointers (`csrc/inkling2_attn.cu:75-83`), so four buffers work; vision `qkv_post` wants one `[rows, 3072]` (below) |

Missing: (1) the bind (attention q/k/v/r/gate/o, unembed, vision, drafter bound as BF16 today: `fam/bind.rs:524-540, 646`) and one launch per tensor, since each matrix has its own `suh` (no fusing of q/k/v/r/gate into one GEMM); (2) the unembed epilogue: soft cap 30 and the valid-vocabulary mask exist only in the BF16 linear (`cuda/inkling2/vocab.rs:69-93`, mode 3 of `gb_inkling2_lin`, `csrc/inkling2_linear.cu:85-90, 349-360`); on EXL3 F32 logits of n = 201,088 a small epilogue is new (output precision is the spec's UNKNOWN-6); (3) vision: `qkv_post_kernel` reads one `acc [rows, 3*1024]` and a bias (`csrc/inkling2_vision.cu:125-147`), so three EXL3 outputs need a three-pointer variant, and the bias moves into the EXL3 epilogue or is passed as null; (4) golden tests at the new shapes. The pinned cuBLASLt prefill (`csrc/exl3_linear.cu:615-635, 913-926`) is not recommended: it reconstructs W_q per call (the unembed's would be 926,613,504 B by script, though the engine only needs unembed rows at window width, `fam/memory.rs:228-229`) and its bits differ from decode.

## 5. (c) Attention at 24 / 8 / 3 heads (`csrc/inkling2_attn.cu`)

Unchanged by the Mini (config equal to the Small's constants): `D = 128`, `DR = 16`, `REL = WIN = 1024`, `PAGE = 256`, `TILE = 64`, `CK_GLOBAL = 512`, `CK_LOCAL = 256`, `QK_CLIP = 22.627416997969522`, `REL_CLIP = 8`, `EPS = 1e-6` (`:53-62`; Mini `config.json`: `head_dim` 128, `d_rel` 16, `rel_extent` 1024, `sliding_window_size` 1024, `qk_l2_clip_threshold` 22.627416997969522, `rel_l2_clip_threshold` 8.0, `rms_norm_eps` 1e-6). `RING_ROWS = 1031` stays (`cuda/inkling2_attn.rs:59`). The per-element arithmetic (clips, relative table by key, sink, gate, online softmax, left fold of chunks) is written per head and per key, so it carries over.

| Site | Small value | Mini | Source |
|---|---|---|---|
| heads | `NQ=12, NKV=2, GRP=6` | 24, 8, 3 | `:53` |
| k/v row | `KVROW = 2*NKV*D = 512` | 2,048 | `:54` |
| prologue CTA | `PRO_THREADS = (NQ+4)*32 = 512`, `static_assert(2*PRO_THREADS == REL)` | (24 + 16) x 32 = **1,280 > 1,024**: cannot launch; assert fails | `:278-279, 293, 323, 1026-1027` |
| prologue maps | warp = query head, warps 12..15 = K0 K1 V0 V1; `rhat[NQ][DR]`; bias table two entries a thread | warp map, thread count and entries a thread all change (split the row over CTAs or loop) | `:300, 323, 342-355, 449-460` |
| K/V channels | 256 per K or V as literals | `NKV*D = 1,024` | `:287, 344, 352` |
| dynamic conv | `static_w [4][256]`, taps `[2 heads x 4]`; Mini `static_w [4,1024]`, `weight_proj [32,2304]` | 8 heads x 4 lags = 32 | `:352-355`; census |
| log scaling | `tau = 1 + 0.1 ln(max((p+1)/128000, 1))` literals | config `log_scaling_n_floor` 8192, alpha 0.1; the kernel ignores the parsed field (`fam/config.rs:222, 389`); CPU reference `LOG_FLOOR = 128_000` (`cuda/inkling2_attn_ref.rs:22-23`). The formula's form for 8192 is the spec's to confirm | `:307` |
| row block | `NW = RB*GRP/16` threads; `rb` 8 or 16 | GRP 3: RB = 8 gives 1.5 warps. Use RB 16 / 32 | `:515, 1037-1038`; `cuda/inkling2_attn.rs:357, 375` |
| tile loop | `qi / GRP`, `qi % GRP`, `w_first = warp*16/GRP` | integer math general; re-verify tiling | `:558-571, 578-580` |
| V offset in K/V tile | literal `2*D` | `NKV*D` | `:612-615` |
| exponent loads | `tid < 8` K, `tid < 16` V (2 heads x 64 keys = 128 B) | 8 heads: 512 B per tile: 32 x 16 B each | `:508, 602-603, 717-718` |
| int8 page | `kv8::PAGE_TOKENS=256, NKV=2, ROW=256, PAGE_BYTES=66,048` | K or V page 262,144 B + 2,048 B exponents = 264,192 B (script). Header is shared with Nemotron (`csrc/nemotron_attn*.cu`, `cuda/nemotron_attn.rs:78`): add a Mini copy rather than edit it | `csrc/nemotron_kv8.cuh:22-27`; `:436-445, 598` |
| ring write / conv commit | 64 threads x 16 B = one 512-element row; `__launch_bounds__(KVROW)` | 256 threads; **2,048 threads invalid** | `:957-966, 978, 1049, 1055, 1064` |
| grid | `(blocks, NKV, chunk)`; merge `(rows, NQ)` | scale; more CTAs | `:1005, 1041` |
| Rust | `N_Q_HEADS 12`, `N_KV_HEADS 2`, `KV_ROW 512`, `Q_ROW 1536`, `ConvState::SLOT_BYTES 3,072`, `BIAS_ROW_BYTES 24,576`, `PARTIAL_BYTES 6,240` | 24, 8, 2,048, 3,072, 12,288, 49,152 (= 192 MiB per 4,096-row chunk), 12,480 | `cuda/inkling2_attn.rs:49-66, 461, 514-516` |
| q-side buffers | q/gate/r/sink in one buffer: `QB_COLS 3,276`, `KB_LD 264`, `QKV_COLS 3,792` | 6,552, 1,056, 8,600; q, gate, r come from three EXL3 launches, sink from a BF16 one | `fam/memory.rs:54-63` |
| relative logits, clips, sinks, gate, window vs global | | as-is | `:278-460, 529-853` |

## 6. (d) Everything else

| Op | Small code | Mini |
|---|---|---|
| short convs `attn_sconv`, `mlp_sconv` | `csrc/inkling2_conv.cu` C runtime, `C % 8` (`:178-179`), 4 taps, state `[slots, 3, C]`; the file comment names 4608 and 256 only in prose (`:3`) | C = 2,304: as-is. State tables per visit (48), not per layer (spec) |
| AttnRes mix | `csrc/inkling2_mix.cu`: `kMaxN = 9`, `kMaxQ = 3`, `kMaxThreads = 576` (`:23-25`), H runtime, `H % 256`, `H/8 <= 576` (`:174`); `cuda/inkling2/mix.rs:34-36, 69` | H = 2,304 -> 288 threads: as-is. Layers with 3 queries (census: `attn_queries.{4,10,16,22}` are `[3, 2304]`) fit `kMaxQ`. Spec bank sizes (at most 5 sources) fit `kMaxN` (not checked by me) |
| loop message mix | none. Spec 3.4: 4 queries (q, k, v, mlp), fp32 output, no gamma | `kMaxQ` 3 -> 4 and an fp32 no-gamma mode: parameterise |
| loop controllers, pair-norm | none (case-insensitive search of `cuda/src`, `csrc` and `fam/` for controller / cross_pass / loop_config / message_gate: no match) | new: fp32 pair-norm of (dest, msg), `down` `[64,4608]`, `gain`/`forget` `[2304,64]` (BF16 weights, existing linear), tanh blend fused with the next norm; spec 3.4. Only the 12 core layers, passes 1-2 |
| BF16 linears (router 128 x 2304, sink 24 x 2304, controllers, `weight_proj` 32 x 2304) | `csrc/inkling2_linear.cu:326-395`, `k % 32` (K = 64, 2304, 4608 all pass) | as-is (K order fixed by K alone: the oracle must use it) |
| router | `csrc/inkling2_router.cu:104`: E <= 512, k <= 16, sigmoid + bias, ties to the lower index | E = 128, k = 8, scale 8.0: as-is |
| softcaps, activation | `GATE_CAP 4`, `UP_CAP 25`, `LOGIT_CAP 30` (`cuda/inkling2/mod.rs:41-44`), `inkling2_act.cuh:11-15` | equal to Mini config (4.0, 25.0, 30.0): as-is, take from config |
| vocabulary top-k | `VOCAB_VALID 200,058`, `VOCAB_PADDED 201,024`, `VOCAB_SHARD_TP4 50,256` (`mod.rs:46-49`); kernels take `n`, `ld`, `base`, `valid` (`csrc/inkling2_vocab.cu:29`) | valid and padded equal the Mini's (`unpadded_vocab_size` 200,058, `vocab_size` 201,024); TP1: one shard, `ld` 201,088: as-is |
| embedding | BF16 gather | as-is |
| drafter | `csrc/inkling2_dspark.cu:46-64`: `H 4608`, `GQA 6`, `TAPS 8`, `MIXES 6`, `NT_ROW 512`, `PER = H/NT_ROW = 9`, `QV = 48`, `PV_Q = 24`; `cuda/inkling2/dspark.rs:111-142`: `LAYERS 5`, `GLOBAL_LAYER 2`, `HEADS 48`, `INTER 18,432`, `LOCAL [0,1,3,4]`, `TAPS 8`, `MIXES = LAYERS + 1` | `H 2304`, `GQA 3` (`QV 24`, `PV_Q 12`), `TAPS 4`, `MIXES 4` (inference from `dspark.rs:424`), `LAYERS 3`, `GLOBAL_LAYER 1`, `LOCAL [0,2]`, `HEADS 24`, `INTER 9,216`. `H/512 = 4.5`: `NT_ROW` 256 or 576, which changes a summation tree. Weights go from BF16 to EXL3 4-bit (UNKNOWN, 1); `fold_kernel` needs a BF16 unembed (`:654-689`), which an EXL3 unembed does not give (spec UNKNOWN-17). Tests assert Small dims (`dspark.rs:1437-1440, 1503, 1513`) |
| vision | `csrc/inkling2_vision.cu:23-28`: `DIM 1024`, `HD 64`, `HEADS 16`, `GRID_SIDE 32`; `cuda/inkling2/vision.rs:87-106`: `WIDTH 1024`, `MLP 4096`, `LAYERS 23`, `PROJ_HIDDEN 4096`, `PATCH_K_PAD 1216` | every one equals the Mini (`vision_config`; patch embedding `[1024, 1200]`). Only `OUT_HIDDEN 4608 -> 2304` (`vision.rs:104`, used `:235, 291, 305, 434`). Linears -> dense EXL3 (section 4) |
| model dims | `HIDDEN 4608`, `INTER 2304`, `EXPERTS 384` (`cuda/inkling2/mod.rs:31-35`); `moe::dims()` (`inkling2/moe.rs:67-69`) | 2,304 / 1,152 / 128 |

## 7. Master table

Verdicts: `as-is` (works for the Mini unchanged), `parameterise` (constants, limits, wiring), `new` (code that does not exist). Effort S < 1 day, M 1-3, L > 3 (agent days, an estimate).

| # | op | existing code (file:line) | verdict | eff | notes |
|---|---|---|---|---|---|
| 1 | expert bank, per-band K | `cuda/moe.rs:652-701, 779-989` | as-is | S | gate == up everywhere; <= 3 bands |
| 2 | MoE decode GEMV, K per launch | `csrc/moe_experts.cu:488-491, 539-558` | as-is | S | K2 2..16 x 3 codebooks |
| 3 | MoE prefill GEMMs, K per launch | `csrc/moe_prefill.cu:761-790, 847-862` | as-is | S | all codebooks |
| 4 | gate/up softcap epilogue, down combine | `csrc/moe_experts.cu:330-382, 608-618`; `csrc/inkling2_act.cuh:11-15`; `csrc/moe_prefill.cu:902-910` | as-is | S | codebook independent |
| 5 | MoE decode over bands + softcap | `cuda/inkling2/moe.rs:101-252`; `cuda/moe.rs:1363`; `cuda/moe_subbanks.rs:257-430` | new | M | host composition only; 22 launches a 3-band layer |
| 6 | MoE prefill over bands + softcap | `cuda/moe_prefill.rs:615-740`; `cuda/inkling2/moe.rs:256-372` | new | M | clone band loop with softcap and bf16 combine |
| 7 | prefill softcap gate/up at `mul1` | `csrc/moe_prefill.cu:796-830` (`:800`, `:818-824`); `cuda/inkling2/moe.rs:273` | parameterise | S | instantiate K2 8, 10 for codebook 2; decode == prefill bits test |
| 8 | MoE shape constants | `cuda/inkling2/mod.rs:31-35`; `inkling2/moe.rs:67-69` | parameterise | S | 2304 / 1152 / 128 |
| 9 | router select | `csrc/inkling2_router.cu:104`; `cuda/inkling2/router.rs:33-122` | as-is | S | |
| 10 | single-K expert bind, config parse | `fam/bind.rs:464-489`; `fam/config.rs:593-618` | parameterise | M | K and codebook per tensor from the pack; config has no `bits` (host lane) |
| 11 | per-expert-K kernel (FrostMoE / CoopMK) | `csrc/fmoe.cu:1763-1857`; `cuda/fmoe.rs:78-108`; `cuda/fn_moe.rs:57-59` | new | L | only if 923 launches a token costs too much; deferred |
| 12 | dense EXL3 decode, M <= 128 | `csrc/exl3_linear.cu:35, 433-463, 694`; `cuda/exl3_linear.rs:41, 704` | as-is | S | covers M = 1 and 8 |
| 13 | dense EXL3 large rows | `cuda/exl3_big.rs:1-22`; `csrc/exl3_big.cu:2-3` | as-is | S | 4,096 rows, decode bits |
| 14 | EXL3 golden at new shapes | `cuda/exl3_linear.rs:1646-1677`; `tests/exl3_big.rs` | parameterise | S | add the 10 Mini shapes (section 4) |
| 15 | bind attention / unembed / vision / drafter as EXL3 | `fam/bind.rs:491-556, 646`; `cuda/exl3_linear.rs:584` | parameterise | M | one launch a tensor (own `suh`) |
| 16 | unembed soft cap + valid mask on EXL3 logits | `cuda/inkling2/vocab.rs:69-93`; `csrc/inkling2_linear.cu` mode 3 | new | S | n 201,088, 64 pad columns; precision per spec UNKNOWN-6 |
| 17 | vision `qkv_post` from 3 EXL3 outputs | `csrc/inkling2_vision.cu:125-147`; `cuda/inkling2/vision.rs:350-443` | parameterise | S | three pointers; bias in EXL3 epilogue |
| 18 | vision LN / attention / residual / GELU | `csrc/inkling2_vision.cu:23-28, 81, 156, 244` | as-is | S | dims equal Mini |
| 19 | vision wiring, `OUT_HIDDEN` | `cuda/inkling2/vision.rs:87-106, 235, 291, 434` | parameterise | S | 4608 -> 2304 |
| 20 | attention constants | `csrc/inkling2_attn.cu:53-55` | parameterise | M | NQ/NKV/GRP, KVROW |
| 21 | attention prologue | `csrc/inkling2_attn.cu:278-279, 293-460, 1026-1027` | parameterise | L | 1,280 threads: re-split |
| 22 | K/V conv channels, dynamic taps | `csrc/inkling2_attn.cu:287, 344, 352-355` | parameterise | M | 256 -> 1,024; 2 -> 8 heads |
| 23 | log scaling floor | `csrc/inkling2_attn.cu:307`; `cuda/inkling2_attn_ref.rs:22-23` | parameterise | S | 128000 -> config 8192 |
| 24 | attention core `Geo`, row block | `csrc/inkling2_attn.cu:513-524, 558-571, 1037-1038`; `cuda/inkling2_attn.rs:357, 375` | parameterise | M | RB 16 / 32; exponents, V offset |
| 25 | int8 KV page layout | `csrc/nemotron_kv8.cuh:22-27`; `csrc/inkling2_attn.cu:436-445, 598-603` | parameterise | M | copy header at NKV 8 |
| 26 | ring write, conv commit | `csrc/inkling2_attn.cu:957-1064` | parameterise | S | thread counts |
| 27 | attention Rust constants, scratch | `cuda/inkling2_attn.rs:49-76, 461, 514-516`; `fam/memory.rs:54-63` | parameterise | S | section 10 |
| 28 | attention per-element math | `csrc/inkling2_attn.cu:278-460, 529-853` | as-is | S | relative table, clips, sink, gate, window / global, softmax |
| 29 | short convs | `csrc/inkling2_conv.cu:24, 178-179` | as-is | S | C = 2,304 |
| 30 | AttnRes mix | `csrc/inkling2_mix.cu:23-27, 174`; `cuda/inkling2/mix.rs:34-36, 69` | as-is | S | H = 2,304 |
| 31 | loop message mix (4 queries, fp32, no gamma) | `csrc/inkling2_mix.cu:21-36, 156-165` | parameterise | S | `kMaxQ` 4 |
| 32 | loop controllers + fp32 pair-norm + blend | none (searched, section 6); nearest `csrc/inkling2_mix.cu:71`, `csrc/inkling2_linear.cu:99` | new | M | spec 3.4 |
| 33 | BF16 linears | `csrc/inkling2_linear.cu:326-395` | as-is | S | |
| 34 | vocab top-k, softcaps, act, embedding | `csrc/inkling2_vocab.cu:29`; `cuda/inkling2/mod.rs:41-49`; `csrc/inkling2_act.cuh:11-15`; `csrc/glue.cu:744` | as-is | S | constants equal Mini |
| 35 | drafter kernels | `csrc/inkling2_dspark.cu:46-64, 321, 654-689` | parameterise | L | H, GQA 3, taps 4, mixes 4, NT_ROW |
| 36 | drafter Rust | `cuda/inkling2/dspark.rs:111-142, 424` | parameterise | M | layers 3, local [0,2], global 1 |
| 37 | TP plumbing | `fam/tp.rs:43, 326-333`; `fam/model.rs:546`; `fam/mod.rs:54, 110`; `fam/memory.rs:77` | parameterise | M | EngineHost lane |
| 38 | collapsed host dims and tests | `fam/memory.rs:56-63, 394`; `fam/config.rs:652-655` | parameterise | S | section 10 |

## 8. Recommended order

1. **MoE mixed K first** (experts are 52% of the bytes a token reads, `docs/ROOFLINE.md`; and it is the plan trigger). Order inside: row 7 (`mul1` prefill softcap) with a decode == prefill bit test on synthetic mixed layers, then rows 5 and 6 with a per-pair equality test against the single-band Small functions on uniform layers, then the Mini shape golden (2304 x 1152, 128 experts, top 8) against the oracle op dumps.
2. **Attention in parallel from day one** (largest risk, rows 20-27): copy the kv8 header, fix the launch geometry (rows 21, 26), then the `inkling2_attn_oracle` test at 24/8/3 heads. Needs no MoE or dense work to start.
3. **Dense EXL3 wiring** (rows 12-17): bind per tensor, goldens at the 10 shapes, unembed epilogue; the engine cannot run attention end to end without it.
4. **Small ops** (rows 29-34): message mix and controllers are the only new ones; the rest are re-checks.
5. After assembly (plan 4.4): drafter (4.6, rows 35-36) and vision (4.7, rows 17-19).

Node time: the first window should run row 12-13 goldens and row 7 first (they decide whether a kernel is missing); then rows 5-6.

## 9. Lane split for plan step 4.3

One engine worktree per lane. New `GB_EXPORT` names keep each file's existing prefix (`gb_moe_*`, `gb_i2_*`, `gb_inkling2_*`) so symbols do not collide. `build.rs` compiles every `csrc/*.cu` (`crates/cuda/build.rs:165`), so no lane edits it.

| Lane | Owns (edits) | Reads only |
|---|---|---|
| **A: MoE mixed K** | `cuda/inkling2/moe.rs`; `csrc/moe_prefill.cu` (the softcap entry `:796-830` only); new `crates/cuda/tests/inkling2_moe_mixed.rs`; `crates/cuda/tests/inkling2_moe.rs` | `cuda/moe.rs`, `cuda/moe_prefill.rs`, `cuda/moe_subbanks.rs`, `csrc/moe_experts.cu`, `csrc/moe_owned.cu`, `cuda/moe_owned.rs` (new band code goes in `inkling2/moe.rs`, leaving upstream files untouched) |
| **B: attention** | `csrc/inkling2_attn.cu`; `cuda/inkling2_attn.rs`; `cuda/inkling2_attn_ref.rs`; new `csrc/inkling2_kv8_mini.cuh`; `crates/cuda/tests/inkling2_attn.rs`, `inkling2_attn_oracle.rs` | `csrc/nemotron_kv8.cuh` (not edited: Nemotron shares it) |
| **C: dense EXL3 + small ops** | `cuda/inkling2/{linear,mix,conv,router,vocab,vision,reference}.rs`; `csrc/inkling2_{linear,mix,conv,router,vocab,vision}.cu`; new `cuda/inkling2/loop_ctl.rs` + `csrc/inkling2_loop.cu`; **`cuda/inkling2/mod.rs` (the shared constants file: only this lane edits it)**; `crates/cuda/tests/inkling2_ops.rs`, `inkling2_support/`; `cuda/exl3_linear.rs` / `exl3_big.rs` only if a golden fails (message the lead first) | everything else |

Outside the three lanes: `csrc/inkling2_dspark.cu`, `cuda/inkling2/dspark.rs` (step 4.6), `fam/*` (EngineHost). `crates/cuda/src/lib.rs` is not touched (new modules live under `inkling2/`, registered in `inkling2/mod.rs`, lane C). Lanes A and B take dimensions as parameters and never import `inkling2::{HIDDEN, INTER, EXPERTS}`.

Gates each lane must show before its rows are done (all need the GPU; none was run for this audit):

| Lane | Byte-level checks that already exist as patterns |
|---|---|
| A | decode == prefill per row at `mul1` for K = 4 and 5 (`cuda/inkling2/moe.rs:18` states the Small's equality; `tests/moe_prefill.rs` is the synthetic-layer pattern); band split == single band per pair on a uniform layer (`tests/moe_subbanks.rs`, per-pair pattern); a poisoned `y_pairs` check that every pair row is written once (TP1 has no filler); the 2304 x 1152 shape against the oracle op dump |
| B | the existing oracle test at 24 / 8 / 3 heads (`tests/inkling2_attn_oracle.rs`), chunking and row-block invariance (`rb` 16 against 32), local ring vs global int8 page at the 1,024 boundary, conv commit rollback |
| C | row invariance of every new EXL3 shape, decode rows 1..8 against `forward_big` at 4,096 (`cuda/exl3_big.rs:13-20` states the contract), unembed softcap + mask against the BF16 linear's mode 3 on the same logits, message mix at `nq = 4` against the reference in `cuda/inkling2/reference.rs` |

## 10. Host-side constants list (EngineHost)

Every row is a compile-time constant or assert tied to the Small. Values for the Mini are from `config.json` / census unless noted.

| Constant | Small | Mini | File:line |
|---|---|---|---|
| `HIDDEN` / `INTER` / `EXPERTS` | 4,608 / 2,304 / 384 | 2,304 / 1,152 / 128 | `cuda/inkling2/mod.rs:31-35` |
| `TOPK`, `ROUTE_SCALE`, `GATE_CAP`, `UP_CAP`, `LOGIT_CAP`, `EPS` | 8, 8.0, 4, 25, 30, 1e-6 | same | `cuda/inkling2/mod.rs:37-51` |
| `VOCAB_VALID`, `VOCAB_PADDED` | 200,058, 201,024 | same (EXL3 head has 201,088 columns) | `cuda/inkling2/mod.rs:46-48` |
| `VOCAB_SHARD_TP4` | 50,256 | n/a (TP1: one shard) | `cuda/inkling2/mod.rs:49`; `vocab.rs:56-57` |
| `N_Q_HEADS` / `N_KV_HEADS` | 12 / 2 | 24 / 8 | `cuda/inkling2_attn.rs:49, 51` |
| `KV_ROW`, `Q_ROW` | 512, 1,536 | 2,048, 3,072 | `cuda/inkling2_attn.rs:64, 66` |
| `RING_ROWS`, `WINDOW`, `PAGE_TOKENS`, `CHUNK_*`, `MAX_ROWS` | 1,031, 1,024, 256, 512 / 256, 8,192 | same | `cuda/inkling2_attn.rs:57-76` |
| `ConvState::SLOT_BYTES` | 3,072 | 12,288 | `cuda/inkling2_attn.rs:461` |
| `BIAS_ROW_BYTES`, `PARTIAL_BYTES` | 24,576, 6,240 | 49,152, 12,480 | `cuda/inkling2_attn.rs:514-516` |
| kv8 page (K or V) | 66,048 B | 264,192 B | `csrc/nemotron_kv8.cuh:22-27` |
| int8 KV per global pass | | `kv8_page_bytes_per_layer` is already generic in `kv_heads / ranks`: 2 x 256 x 8 x 129 = 528,384 B per layer page; 8 passes / 256 tokens = 16,512 B a token (matches `docs/FIT.md`) | `fam/memory.rs:152-155` |
| `kv8_page_bytes_tp4` | `2 * kv8::PAGE_BYTES` | follows the new header | `fam/memory.rs:394-396` |
| `QB_COLS`, `QB_LD`, `KB_LD`, `QKV_COLS` | 3,276, 3,280, 264, 3,792 | 6,552, 6,552, 1,056, 8,600 (q 3,072 + gate 3,072 + r 384 + sink 24; k 1,024 + 32 dynamic taps). Layout changes anyway: q, gate, r are three EXL3 outputs | `fam/memory.rs:54-63` |
| `ACTIVATION_RESERVE` | 6 GiB | FIT assumes ~5 GiB | `fam/memory.rs:45`; `docs/FIT.md` |
| `SLOT_WINDOW_ROWS`, `PARTIAL_ROWS`, `MAX_CANDIDATES`, `PREFILL_ROWS_MAX` | 8, 8, 64, 4,096 | same (`block_size` 8) | `fam/memory.rs:48-53`; `fam/engine.rs:33` |
| `Tp::LEADER_OF_FOUR`, `RANKS`, `MAX_TP_RANKS` | 4 | 1 | `fam/memory.rs:77`; `fam/tp.rs:43, 326-333`; `fam/mod.rs:54, 110, 130-135` |
| rank guard | "12 query and 2 KV heads a rank" | remove | `fam/model.rs:546-551` |
| `ARCH`, `MODEL_TYPE` | `Inkling2ForConditionalGeneration`, `inkling2` | `Inkling2LoopForConditionalGeneration`, `inkling2_loop` | `fam/config.rs:30, 32` |
| `parse_quant` | needs `bits`, `head_bits == 16`, `codebook`, `out_scales`, `note` | Mini has none: read K and codebook per tensor | `fam/config.rs:593-618`; `crates/exl3/src/checkpoint.rs:202` |
| drafter tap check | `last == layers`, taps below `layers` | taps `[5, 23, 41, 48]` index the 48 visits | `fam/config.rs:481-486` |
| single K bind | `(k, n, cfg.quant.bits_twice)` | per tensor | `fam/bind.rs:474-478` |
| bf16 attention bind, drafter bind | | EXL3 groups | `fam/bind.rs:524-540, 646` |
| per-layer state arrays, conv tables | sized `cfg.layers` | per visit: 48 sets, 8 global KV passes, 40 rings | `fam/model.rs:566-590` (spec 8) |
| `loop_config` | none | `core_start 6, core_end 18, num_passes 3, gate_rank 64`; passes 6..17 run x3 | `config.json` |
| drafter | `LAYERS 5, GLOBAL_LAYER 2, HEADS 48, KV_HEADS 8, INTER 18,432, TAPS 8, MIXES 6, LOCAL [0,1,3,4]` | 3, 1, 24, 8, 9,216, 4, 4 (inference), [0, 2] | `cuda/inkling2/dspark.rs:111-142` |
| `OUT_HIDDEN` | 4,608 | 2,304 | `cuda/inkling2/vision.rs:104` |
| tests asserting Small dims | | update | `fam/config.rs:652-655`; `cuda/inkling2/dspark.rs:1437-1440, 1503, 1513`; `fam/memory.rs:413-423` |

## 11. UNKNOWNs

1. Drafter K (4-bit) and its tensor list: `mtp.safetensors` is not in `census.json`; only byte arithmetic supports 4-bit (section 1).
2. Time cost of 923 launches a decode token against 336 (graphs, PDL): not measured.
3. Whether the new shapes (2304 x 384, 3072 x 2304, 2304 x 201,088, vision 4096 x 4096) run row-invariantly: tests cover other shapes; needs the GPU golden.
4. Boot cost of 9,216 per-tensor allocations (`from_groups`) against region-based subbanks.
5. Exactness of casting the BF16 vision biases to fp16 (`cuda/exl3_linear.rs:599`).
6. That CYBERFROST's fuse / second-form / ring changes preserve bits (their comments say so; not tested here).
7. Whether FrostMoE routed with `n_group = 2` could reproduce the Mini's router, and its fast `KC` at H = 2304.
8. The exact log-scaling formula for `n_floor` 8192, the loop's per-visit AttnRes source counts (spec says at most 5; `kMaxN` is 9) and the drafter `MIXES` count: spec items, not code.
9. Output precision of the EXL3 unembed (spec UNKNOWN-6) and whether an EXL3 drafter can still fold the unembed (spec UNKNOWN-17).
10. Whether prefill's second remap + plan per band (`cuda/moe_prefill.rs:680-690`) is needed or only defensive.

## 12. What the host half settled (lane engine-host, engine branch `inkling2-loop-host`)

Written after the audit, from the host code and its tests (`REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked families::inkling2_loop::`: 33 passed).
Paths below are `crates/model/src/families/inkling2_loop/` on that branch.

| Audit row / UNKNOWN | Status | Where |
|---|---|---|
| row 10 (single-K bind, config parse) | done on the host side: config parses the real Mini config (no `bits`, `head_bits`, `codebook`, `out_scales`, `note`); K and codebook are read per group from the trellis and marker; mixed K inside a layer and inside an expert binds and is reported as a census | `config.rs`, `bind.rs` (`KCensus`), commit `4489015` |
| row 15 (bind as EXL3: attention, unembed, vision, drafter) | bind done (6 attention linears and `gate_attn`/`wr_du`, unembed padded to 201,088, vision linears with BF16 bias, projector, drafter incl. `rnn_proj` and `hidden_out` with bias); the launch side is lane C | `bind.rs`, `inventory.rs` (drafter groups recognised by `.trellis`) |
| rows 37, 38 (TP plumbing, collapsed dims) | host side done: no ranks, no slices, one-rank memory plan; the device constants are listed by `dims::device_constant_gaps` (22 rows, file:line, Small and Mini value, tested) | `dims.rs`, `memory.rs`, `schedule.rs` |
| UNKNOWN 1 (drafter K = 4) | the 29 drafter linears are 4-bit by the header shapes the spec lane recorded (PORT-SPEC A.11, `K by trellis: [4]`); `mtp.safetensors` is still not in the census, so the bind fixture takes them from A.11 | `testdata/inkling2_loop/README.md` |
| FIT.md reproduction | the closed form of the config and the K census gives 15,032,344,840 B (= FIT.md's total) and every component to three decimals; the TP1 plan gives 16,512 B a token (18,576 with the drafter), rings of 1,031 positions (168,919,040 B) and FIT.md's four profiles | `weights.rs`, `memory.rs` |

Findings the audit did not have:

1. **The Mini's `chat_template.jinja` does not open in the server's template engine**: `chat template failed: syntax error: unknown statement generation (in chat:129)` (`{% generation %}` wraps assistant messages; the template also builds `rendered` with `{% set %}...{% endset %}` and serialises tool-call arguments with `dictsort`). The format entry accepts the Mini's architecture names (commit `48e1f06`), but a chat request cannot be rendered until the template engine learns `generation` (a Mini-specific format or an engine change). This is outside kernels and outside step 4.2; it blocks any chat smoke gate. Checked once with a throwaway test against the pack's template at the pinned revision; not committed.
2. **`dspark.cu:60` `PER = H / NT_ROW`** is 9 at 4,608 and 4.5 at 2,304 (already in section 6); `dims::tests` asserts `2304 % 512 == 256`, so a stale reading of this row fails a test.
3. **The loop cross-pass query tensors** hold `2 x core layers` rows (`[24, 2304]` for 12 core layers): a weight formula that writes `2 x layers` is right only by coincidence at the shipped config; the bind and the weight model use `2 x core` and a test compares them byte for byte on the fixture.
4. **Planning is not allowed to price what is not sized**: `memory::UNPRICED` names five buffers the kernels do not size yet (attention partials and row plans at 24 / 8 / 3, conv row descriptors, loop message buffers, EXL3 input-rotation workspace, drafter per-proposal scratch); lanes B and C move each into `step_items` with its kernel, and the 5 GiB reserve covers the rest until then. The listed buffers total less than the reserve (tested).
5. Registration: `Registry::global()` lists `inkling2_loop` after `inkling2`; its `build` refuses by name until the seven components of `REQUIRED` are in `BUILT_COMPONENTS`. Lane C owns `inkling2/mod.rs` (audit section 9); the Mini's component names are the `inkling2_loop_*` ones in `inkling2_loop/mod.rs`, so no lane edits another's list.
