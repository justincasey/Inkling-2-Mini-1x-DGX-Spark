# kernels-dense (plan step 4.3, lane C): dense EXL3, message mix, loop controllers

Gate `kernels-dense`. Node-4 (one DGX Spark, GB10 sm_121), engine worktree branch `inkling2-loop-dense` at
`867486ea4c351e48b3f75c4f33e82f0a450e353d` (clean: `dirty` false in `meta.json`), the pinned Mini pack staged on the node by the
node-ops lane. Raw outputs: `ops-mini.txt` (the Mini's tests, every measured value), `ops-all.txt` (the whole `inkling2_ops`
binary: the Small's own tests plus the Mini's), `dspark-all.txt` (the Small's drafter tests, which use the changed linear and vocab
modules), `node-status.txt`.

Engine commits (local, three): `FEAT: Added the dense EXL3 projection and the EXL3 unembed epilogue for the Mini`,
`FEAT: Added the raw AttnRes mix at four queries and the loop message controllers`,
`TEST: Added Mini dense EXL3, message mix and controller tests with oracle goldens`.

## What is proven (each line: the test, the printed result in `ops-mini.txt`)

| Claim | Test | Result |
|---|---|---|
| Dense EXL3 on the 12 real tensors (attention layer 11 `wq_du` 2304x3072, `wk_dv`, `wv_dv` 2304x1024, `wo_ud` 3072x2304, `gate_attn` 2304x3072, `wr_du` 2304x384; unembed 2304x201,088; vision `q_proj` 1024x1024, `c_fc` 1024x4096, `c_proj` 4096x1024, `proj_in` 4096x4096, `proj_out` 4096x2304) equals the oracle's `x . W_eff` over 5-6 column strips each (8 rows) | `mini_dense_exl3_real_tensors_equal_the_oracle` | fp32 within 1.3e-6 to 1.1e-4 of the row rms of the oracle that rounds the rotated input to fp16 (what the kernels store); within 5.6e-4 to 9.2e-4 of the exact float64 oracle; bf16 output equal to the rounding of the fp32 output at every element, and within one bf16 step of the oracle's bf16 (0 to 0.4% of elements differ) |
| Row invariance of the dense path: decode windows of 1..7 rows equal the first rows of the 8-row window; 4,096 rows equal the 8-row window, lone rows (8, 127, 128, 1000, 2047, 4087) and the last window, bit for bit | `mini_dense_exl3_rows_are_invariant_at_every_shape` (random 6-bit tensors at all 10 Mini shapes) and the same check inside the real-tensor test (all 12 tensors) | bit-identical everywhere (unembed in fp32 output, the rest bf16) |
| The unembed epilogue is the BF16 linear's mode 3 on the same logits: `cap * tanh(z / cap)`, `-inf` from column 200,058, n = 201,088 | `mini_unembed_epilogue_equals_the_bf16_linear_softcap_mode` | bit for bit at (col_base 0, valid 200,058), (1,000, 200,058), (0, 201,088); 36k saturated logits reached; 5,150 columns masked per 5 rows |
| The EXL3 unembed end to end: fp32 logits (not rounded to bf16 first: NUMERICS U6, taken as RESOLVED), cap and mask, argmax | `mini_exl3_unembed_masks_caps_and_feeds_the_argmax` | capped logits within 3 fp32 ulps of `30 tanh(z/30)`, column 200,057 finite, 200,058 and up `-inf`, `topk` argmax = the first maximum of the unpadded columns |
| Message mix at 4 queries, 1..5 sources, fp32 unrounded out | `mini_message_mix_at_four_queries_equals_the_float64_reference_and_rows_are_invariant` | worst error 1.2e-7 to 6.3e-7 of unit scale against float64; a row alone and a reversed batch are bit-identical |
| Message mix on the pack's real queries (merged q, k, v, mlp) of the 12 core layers against the oracle's `mix_raw` | `mini_message_mix_equals_the_oracle_on_the_real_queries` | worst 4.2e-7 of unit scale |
| Loop controller stages (pair-norm, `down`, `gain`, `forget`, blend, norm), teacher forced against float64, at 1, 8, 70 rows; rows bit-identical at every row count | `mini_loop_controller_stages_equal_the_float64_reference_and_rows_are_invariant` | ok |
| The 48 real controllers (layers 6-17 x q, k, v, mlp) against the oracle's `blend` and the RMSNorm | `mini_loop_controllers_equal_the_oracle_on_the_real_weights` | 44 cases agree to 6e-7 of the blend's rms; 4 cases show one bf16 flip of `hdn` (blend within 1.05e-3 of rms, bf16 normed output within one step beyond a 5e-3 floor); each case also passes the teacher-forced stage check |
| The new ops under CUDA graph capture, with the PDL mask off and on (`PDL_DECODE_CHAIN | 3`) | `mini_dense_ops_graph_replay_equals_eager` | projection, message mix, controller and unembed replay bit-identical to eager |
| Router at E = 128, width 2,304, top 8 (reference, ties to the lower index, NaN logits, rows) | `mini_router_equals_the_reference_ties_to_the_lower_index_and_rows_are_invariant` | ok, 0 near-ties in 150 rows |
| Short conv at C = 2,304 (and 1,024), the AttnRes mix at hidden 2,304 (1..9 sources, 1 and 3 queries), the bf16 linears at the sink / weight_proj / controller shapes | the Small's `conv_*`, `mix_equals_*`, `linear_equals_*` tests, parameterised in the same file (`ops-all.txt`) | pass |
| Softcap constants and vocabulary sizes equal the Mini's `config.json` | `mini_softcap_constants_are_the_configs` | 30, 4, 25, 8, 200,058, 2,304; 201,088 = 1571 x 128 |
| Regression: the whole `inkling2_ops` binary (golden-hash test included) and `inkling2_dspark` | `ops-all.txt`, `dspark-all.txt` | 28 and 11 passed, 0 failed; the Mac's `families::inkling2::` stays 31 of 31 |

## What the numbers mean (findings)

* **The dense kernels follow the `exl3_fp16` emulation, not the exact float64 oracle.** They round the rotated input to fp16 (a design of the
  kernels, NUMERICS `exl3_fp16`): 5.6e-4 to 9.2e-4 of the row rms against the exact oracle (the generator prints the same). Against the
  oracle with that rounding emulated the residual is 1e-6 to 1.1e-4, the scatter of fp32 against float64 rotation at an fp16 rounding tie
  (about one flipped input a row, ~2e-5 of the rms each). The test bounds it at 2e-4 and prints the measured value per tensor.
* **Controllers flip against the oracle in 4 of 48 cases**: one `hdn` value of 64 rounds the other way (fp32 sum against float64), moving the
  blend by up to 1.05e-3 of its rms. This is the expected rate (about one per 10 cases) and is not an error; the engine is deterministic and
  row-invariant, the oracle is the tolerance anchor (NUMERICS).
* **UNKNOWN (audit 5) settled for the pack**: of 211,968 bias values in the 138 EXL3 groups with a bias (the vision block linears; the drafter
  shard is not in the scanned index), **3 are not exact in fp16** (|v| of 1.6e-6 to 6.3e-6, subnormal in fp16), worst error 2.98e-8 absolute:
  `layers.18.mlp.c_proj.bias[22]`, `layers.2.self_attn.k_proj.bias[915]`, `layers.3.self_attn.v_proj.bias[676]` of `model.visual.backbone`. The
  vision tower is step 4.7; `Exl3Weights::from_loaded` uploads biases as fp16 (`exl3_linear.rs`, not edited: the lead decides whether a bf16 bias
  upload is worth it).
* **Unembed output precision (spec UNKNOWN-6) is RESOLVED in NUMERICS**, so no choice was open: raw logits stay fp32, the cap is applied to the
  fp32 value, no bf16 round in between.
* Timings (`timing ...` lines of `ops-mini.txt`, random weights, L2 warm): unembed decode 1,466 us at 1 row = 237 GB/s of its 347 MB; `wq_du`
  18 us; `wo_ud` 20 us; the 5-launch controller 23 us at 1 row, 32 us at 8; the message mix 6 us; the epilogue 4 us at 1 row, 18 us at 8. Small
  tensors fit L2, so their GB/s is not a DRAM figure.

## Not proven / not done

* Vision `qkv_post` from three EXL3 outputs, `OUT_HIDDEN` 4,608 -> 2,304 and the drafter's EXL3 linears (plan 4.6, 4.7): the dense linear path
  and the 5 vision shapes are proven here, the vision kernels are not touched.
* The conv state of the K/V streams (12,288 B a slot) is the attention lane's; only the static conv kernel at C = 2,304 and 1,024 was run.
* Goldens cover 5-6 strips of each tensor (all 3 of `wr_du`), not every column; the invariance tests cover every column. A column-complete check
  of the unembed would need the CPU decode of 201,088 columns.
* No end-to-end loop assembly (step 4.4) and no timing under the real decode chain (graphs are tested for equality, not speed).

## How to reproduce

`NODE=<node> LAB_SRC=<engine worktree> scripts/ops_dev.sh test inkling2_ops mini_` (the pack is read from `REXL3_INKLING2_LOOP_DIR`, which
`ops_dev.sh` exports); goldens: `testdata/inkling2_loop/gpu/gen_goldens.py` (header documents the file layouts and the input recipe, which the
tests mirror in `inkling2_support`).
