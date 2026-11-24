# kernels-vision (plan step 4.7: the vision tower and the image path)

Gate `kernels-vision`. Node-2 (one DGX Spark, GB10 sm_121, TP1), engine worktree branch `inkling2-loop-vision` at
`41814cc4e724cb347fe541148c0b74f8955209c3` (clean; `dirty` false in `meta.json`), the pinned Mini pack staged on the node by the node-ops lane.
Raw outputs: `ops-cuda-vision.txt` (synthetic-weight tests), `ops-model-vision.txt` (the real pack: the gate), `ops-small-vision.txt` (the
Small's own vision tests), `mac-tests.txt` (host tests), `reference-gen.txt` (the reference generator), `node-status.txt`.

Engine commits (local, branch `inkling2-loop-vision` from `1203fe6`): the tower's EXL3 projections and `qkv_post3`; `stage`/`run`
(capturable launches); the reference goldens, their generator and the preprocessor parity test; the host encoder, features and splice;
the GPU tests; the release-notes fragment.

## What is proven (test, printed result)

| Claim | Test | Result |
|---|---|---|
| The Mini tower (23 blocks, the 140 EXL3 linears of the pack, 138 of them with a bias, BF16 biases, projector to 2,304) through the family loader and encoder equals HF transformers' fp32 tower on the pack's effective weights, per token, on 4 real images, as closely as HF's own bf16 tower does | `mini_vision_encoder_matches_the_reference_on_the_real_pack` | cosine min 0.9959 to 0.9992, mean 0.99959 to 0.99978 (HF bf16: min 0.9963 to 0.9990, mean 0.99964 to 0.99977); largest element error 0.165 to 0.352 of its row rms (HF bf16: 0.159 to 0.294). Per image in `meta.json` |
| A batch of all four images (361 token rows, one launch set) equals each image alone, bit for bit | `mini_vision_batches_and_bias_modes_on_the_real_pack` | identical |
| Graph replay equals eager: the staged launches of the real tower, captured once per image and replayed after another image's features took the buffer | same test | identical for all four images |
| Row invariance of the synthetic tower across the EXL3 decode path (25 rows), the large-row path (145 rows alone; 191 rows in a batch, whose q/k/v projections run in 63-row chunks and the MLP's in 47-row chunks) and every batch order, in both bias modes | `mini_vision_batches_are_row_invariant_across_the_exl3_paths` | identical bits |
| Graph replay equals eager on the 2-layer synthetic tower, decode path and large-row path | `mini_vision_graph_replay_equals_eager` | identical |
| `qkv_post3` (three fp32 matrices, any row stride) equals the Small's fused `qkv_post` bit for bit; a bias folded into the matrices and none given gives the same bits | `mini_vision_qkv_post3_equals_the_fused_post` | identical; rows outside the range untouched |
| The preprocessor's patch tensors equal HF's `Inkling2ImageProcessor` (transformers be48e7a0, the pack's `processor_config.json`) byte for byte on the 4 images (raw padded tensor, HF `pixel_values` reproduced from it, the bf16 tower input); the image-token count planned from the size equals HF's `image_grid_thw.prod()` (117, 96, 48, 100) | `gb-vision inkling2_loop_preprocess` (host) | 3 passed. The digests equal the Small's `expected.json` (fc00b5b3) for the same images |
| The Small is unchanged: its golden-hash test (synthetic weights, bit-exact), batch invariance and its real-weights test on the Small's staged pack (read only) pass; `families::inkling2::` stays 31 of 31 | `ops-small-vision.txt`, `mac-tests.txt` | pass; real-pack reduced-digest deviation 0.020 to 0.041 (limit 0.15) |

## The bias decision (step 3): upload the biases exactly

The pack's BF16 biases reach the activations as stored: `BiasMode::Exact` leaves the EXL3 projections without an epilogue bias, uploads each
bias as bf16 and lets the post kernels (`qkv_post3`, `resid`, `bias_gelu`) add it in fp32 to the unrounded fp32 projection before the one
bf16 rounding. `BiasMode::Epilogue` (fp16 bias inside the EXL3 launch, what `Exl3Weights::from_loaded` uploads) is kept as a documented
option for the tests. Evidence:

* **The 3 fp16-inexact values (worst absolute error 2.98e-8) change no output bit on these images.** The exact path with those 3 values
  rounded through fp16 (a bf16 value rounded to fp16 is a bf16 value again, so nothing else differs) gives the exact-bias bits at **0 of
  831,744** feature values over the four images (`fp16-rounded biases` line). By a rough estimate (not measured) a 3e-8 change flips the bf16 rounding of a given
  element with a probability near 4e-6 over a few thousand rows a layer, so a flip is possible on another image; the exact path costs nothing
  and is what is shipped.
* **The epilogue mode is not the same arithmetic**, and that, not the 3 values, is what moves its outputs: 749,576 of 831,744 bf16 feature
  values differ from the exact path (largest difference 3.3e-2, a quarter of an image's rms), while its error against the reference is the
  same within noise (cosine min 0.9957 to 0.9994, mean 0.99963 to 0.99978; largest error 0.149 to 0.368 of the row rms). The EXL3 launch
  computes `v * scale * svh + bias` as one fused operation, so it differs from a separate fp32 add in 55,688 of 262,144 values of one
  1,024 x 4,096 test layer by at most **1.98 ulp** of the larger operand (`mini_vision_epilogue_bias_differs...`); a change that survives the
  bf16 rounding of an activation is a whole bf16 step, and 23 blocks spread it. Any perturbation of the fp32 sums does this; bits of the tower
  are a contract with itself (row invariance, graph = eager), not with the reference.
* Cost of the exact path: none that matters (the post kernels already took a bf16 bias for the Small; the fp32 intermediates are the same
  buffers). The Mini tower encodes an image in 11 to 18 ms here (one run each, host-side, see notes).

## Findings

* **The reference** (`testdata/inkling2_loop/vision/`, at most 1.08 MB a file, 3.3 MB the four): HF's own tower at the pinned revision in fp32
  with the pack's EXL3 linears decoded to float64 effective weights, cast to fp32, on the engine's vLLM-normalised pixels. Regenerating it
  reproduces every file byte for byte (`reference-gen.txt`). It is not an emulation of the kernels' fp16 rounding of the rotated input
  (5.6e-4 to 9.2e-4 of the row rms a linear, DenseLane's record), so part of the engine's error against it is that rounding; the split
  between that and the bf16 activations was not measured.
* **The engine is at HF's own bf16 error, not below it**: mean cosine within 5e-5 of HF's bf16 tower on every image. Its largest single-element
  error is higher than HF bf16's on two images (480x320: 0.323 against 0.189; 150x1000: 0.352 against 0.258) and lower on the other two. The
  gate allows 2x HF's figure; the cause of the two larger outliers is not isolated.
* **Memory**: a loaded tower for 512-token images allocates 72.1 MiB of scratch for 2,049 rows (36.9 KB a row, EXL3 workspace included),
  `Inkling2VisionTower::scratch_allocated()`; at the server's 4,096 tokens (16,385 rows) that extrapolates linearly to about 577 MiB, against 541 MiB priced by
  `inkling2_loop::memory::vision()` (`native * per_native + merge`): 36 MiB short, inside the 5 GiB reserve. LoopRank may want the figure.
* Both bias modes and every chunking give the same bits per mode; the first draft sized the fp32 buffer for the 4,096-wide MLP (16 KB a
  row), now the Small's 1,024-wide chunk.

## Not proven / not done

* Wiring: `LoopEncoder`, `LoopSplice` and `load_tower` exist (`families::inkling2_loop::vision`, registered in `mod.rs`) but the engine build
  does not call them (`engine.rs` still refuses vision; `VISION_TOWER` is not in `BUILT_COMPONENTS`); the splice into the Mini rank's
  embedding and an end-to-end image prompt are LoopRank / LoopBackend's.
* Four images, one pack revision; no image above 1,000 pixels a side or at the 4,096-token cap was run. Encode times are not a benchmark.
* The Small's real-pack run has no per-token HF comparison here (it needs a reference directory generated for the Small pack); the Small's
  exact bits are held by the synthetic golden-hash test.
* No CPU emulation of the kernels' fp16 rotation for the tower: the reference cannot say how much of the residual is that rounding.

## How to reproduce

`NODE=<node> LAB_SRC=<engine worktree> [PKG=gb-model] scripts/ops_dev.sh test <bin> mini_vision` (the pack is read from
`REXL3_INKLING2_LOOP_DIR`, which `ops_dev.sh` exports). References: `testdata/inkling2_loop/vision/fetch_vision_pack.py` then
`gen_vision_ref.py` (header documents the venv: torch 2.14.1 CPU, transformers at `be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef`, numpy, pillow,
safetensors, httpx, torchvision).
