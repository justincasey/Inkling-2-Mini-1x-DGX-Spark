# kernels-draft (plan step 4.6, lane DrafterLane): the block drafter's kernels and Rust

Gate `kernels-draft`. Node-1 (one DGX Spark, GB10 sm_121), engine worktree branch `inkling2-loop-draft` (from `1203fe6`) at
`ca8af5148858db7ae19938b790d77bca9d229e1c` (clean: `dirty` false in `meta.json`), the pinned Mini pack staged on the node by the node-ops
lane; the Small's pack (`/home/spark/models/Inkling-2-Small-EXL3-4bpw-full`) was read, never written, for the Small's regression. Raw
outputs: `inkling2_dspark-all.txt` (the whole drafter test binary: the Mini's 10 tests and the Small's 11, every printed measurement),
`inkling2_ops-all.txt` (the other binary that links the changed kernels), `mac-tests.txt`, `node-status.txt`, `oracle-planes/`, `tools/`.

Engine commits (local, eight; the drafter is NOT wired into the rank and the engine: the tap capture of the target forward and the spec loop
follow `engine::build`):

| commit | title |
|---|---|
| `c2f9d76` | FEAT: Parameterised the block drafter kernels by model geometry |
| `b9c213e` | FEAT: Added the Mini block drafter over EXL3 linears with the Markov head folded into the unembed |
| `7786560` | TEST: Added Mini drafter tests with oracle planes, invariance, rollback and graph replay |
| `5dfa0ae` | FEAT: Added the Mini drafter host side: geometry check and loader for mtp.safetensors |
| `f178293` | TEST: Added the Mini norm-row check and case discovery to the drafter tests |
| `8ed7aa4` | TEST: Added golden hashes for the Mini drafter run on the pinned pack |
| `297faf3` | TEST: Added the Mini drafter golden hash file |
| `ca8af51` | TEST: Printed the Mini drafter's weight bytes and upload time in the timing test |

## What was built

* `csrc/inkling2_dspark.cu`: the Small's kernels take the model as template parameters dispatched per launch: width `HID` (4,608 / 2,304),
  taps (8 / 4), query heads a KV head `GQA` (6 / 3); `mixes`, `nq`, the tap stride are launch arguments. Kernels that read a projection take
  q / k / v, the context k and v, gate and up by pointer and stride (the Small passes three column ranges of one buffer, the Mini three EXL3
  outputs), and `add_norm` reads a bf16 or an fp32 branch. New: `markov_in` / `markov_gate` (the Markov step over an EXL3 `rnn_proj`) and
  `fold_pack`. The Small's instantiations are the old code: its golden hashes are unchanged (below).
* `src/inkling2/dspark.rs`: FFI and the Small's call sites only. `src/inkling2/dspark/mini.rs`: `MiniDraftWeights::upload`, `MiniKv`,
  `MiniDrafter::{commit, adopt_context, propose, stage_proposal + enqueue_proposal + finish_proposal, save_slot, load_slot,
  copy_plane_page, refuse}` and the memory plan (`planned_bytes` equals the allocation, asserted on every run). `enqueue_proposal` is
  launches only (no upload, sync or allocation), so the engine can capture it in a CUDA graph.
* `crates/model/src/families/inkling2_loop/draft.rs` (+ the `pub mod draft;` line in `mod.rs`): `DraftGeometry::from_config` (names the first
  value of `mtp_config` the kernels were not built for) and `draft::load(dir, inventory, bound drafter)` (BF16 vectors as bits, the 29 EXL3
  groups with fp16 scales, the Markov heads' bf16 biases).

## Geometry traps of the audit (rows 35 / 36), and how each was resolved

| trap | resolution |
|---|---|
| `PER = H / NT_ROW` 9 at 4,608, 4.5 at 2,304 | a norm row stays 9 columns a thread in order (`c = tid + NT j`), `NT = H / 9` = 512 (Small) or 256 (Mini). The row's sum is the same double fold: 9 columns in order, the warp shfl_xor tree, the warps in order (16 warps, 8 warps). The oracle sums in float64 front to back; the two double sums differ by rounding at 2^-53, which moves a bf16 output only when a value sits within that of a rounding tie. Measured: `f_rnn` (the Markov mix, a bare norm row) is bit-equal to the oracle at all 18,432 + 20,736 elements of the oracle cases, and `mini_norm_rows_equal_the_float64_reference` finds 0 of 5,529,600 outputs different |
| `TAPS` 8 -> 4, `MIXES` 6 -> 4 | `mix_norm<HID, TAPN>`, `mixes` an argument; `fuse_alpha` is the 4-wide row softmax |
| `GQA` 6 -> 3 | `attn_part<GQA>` / `attn_merge<GQA>`: `QV` 24, `PV_Q` 12, shared memory by `AttG<GQA>::SMEM`; the tiling (256 threads, two halves of 128) is unchanged |
| `LAYERS` 5, `LOCAL [0,1,3,4]`, global 2 | `LAYERS` 3, `LOCAL [0,2]`, `GLOBAL_LAYER` 1 in `mini.rs`; two rings a slot, layer 1 in the page plane |
| taps are visits, not layers | `DraftGeometry` checks every tap visit is below the 48 visits and the last equals 48 (the final norm); the kernels do not depend on the ids, the target forward does |
| `fold_kernel` needs a bf16 unembed | not used by the Mini: see UNKNOWN-17 below |

## Decisions

**UNKNOWN-17 (quantized Markov head): the engine builds the oracle's variant (a) from EXL3 launches; the question stays OPEN for the lead.**
`rnn_proj` is one EXL3 tensor over the whole 5,120-wide input, so its column halves are the same tensor applied to an input whose other half
is zero (the kernel is linear in the input): the context half `[0 | 0 | hidden | f_rnn]` once a block (rows `b * 7`), the step half
`[state | w1(prev) | 0 | 0]` each step; the gate adds them in the Small's order, `raw = fp32(y_step + fp32(y_ctx + bias))`, the bias being the
group's bf16 bias added in fp32 (not through the kernel's fp16 bias). The fold `fold_W = bf16(U . H_out)`, `fold_b = bf16(U . b_out)` is
built once at load by the EXL3 launches themselves: `hidden_out` on the identity gives `H_out^T`, the target's unembed on those 256 rows
(f32 in, f32 out) gives `fold_W^T`, and on `b_out` gives `fold_b`; `fold_pack` transposes and rounds to bf16. Nothing is dequantised; the
unembed stays EXL3 and untouched. Why not the alternatives: the spec's text and the oracle implement (a) and "drafts the same tokens as the oracle"
means (a); variant (c) (no fold: `hidden_out` then an EXL3 unembed launch each step) needs 7 more unembed launches a round, 1,466 us each at
one row (`kernels-dense`) = about 10 ms a round (derived, not measured here), against one 103 MB bf16 read a step. What is NOT done: the
experiment NUMERICS asks for ((a) / (b) / (c) against the target's greedy tokens on the fixture contexts) needs target fixtures (one exists, 40
positions); (b) and (c) are not implemented in the engine. The EXL3 launches round their rotated input to fp16, so the engine's fold is
within that rounding of the oracle's float64 fold (`kernels-dense` measured 5.6e-4 to 9.2e-4 of a row's rms per dense launch); the effect is
measured end to end in the table below (drafts equal the oracle's on 56 / 56 and 63 / 63 synthetic and real-feature steps on the int8 plane).

**UNKNOWN-21 (residual add): RESOLVED as NUMERICS says, and implemented that way**: the residual is stored `bf16(h + branch)` (one fp32 add of two
bf16 values) and the following norm reads that bf16 residual; with TP1 the branch is the one bf16 output of `o_proj` / `down_proj`.

**The global layer's plane.** Layer 1 keeps its K/V in the drafter's own plane indexed by the target's page ids, int8 at the target's page
price: 256 x (2,048 + 16) = 528,384 B a page (the target's K page 264,192 B + V page 264,192 B, `kernels-attn`), with the target's quantiser
(`exponent_i8`, `quant_i8`, exponents in [-24, 8]) and dequantisation in the attention's tile load; the bytes inside a page are the drafter's
(K and V rows of a position together, then their exponents) as in the Small, not the attention kernel's K page / V page split: the target never reads this
plane, only the page id and the size are shared. Layers 0 and 2 keep a 1,032-position ring a slot (4 KiB a position).

## What is proven (every number is printed in `inkling2_dspark-all.txt`)

Oracle planes: `oracle/gen_dspark_fixtures.py` at recipe `2d1f210`'s oracle sources, run on node-1 (CPU, the venv in the lab directory). `long` is
the generator's seeded synthetic case (1,100 positions, anchors 1, 600, 1023, 1024, 1025, 1031, 1032, 1100: the 1,024 window edges); the real-feature
case is `real_chat-capital-effort0` (`tools/gen_real_case.py`: the generator's own `real_features` / `run_case` on target fixture
`chat-capital-effort0.safetensors`, sha256 `317571c0...3cc926`, read from the node-ops lane's fixture folder: 40 positions, anchors 1, 5, 13, 24, 25, 26, 30, 33,
39). `mini_matches_the_oracle`, bf16 plane (int8 plane in parentheses). Errors: max bf16 steps of max(|want|, group rms), elements that differ, relative L2.

| plane | `long` | `real_chat-capital-effort0` |
|---|---|---|
| context K/V, 3 layers x positions in the window (global layer: all) | rel L2 9.25e-4 (6.21e-3), max 5.5 (10.9) steps, 10.0% (37%) differ | 8.84e-4 (6.17e-3), max 3.6 (11.8), 9.5% (36%) |
| `f_rnn` | bit-equal, 0 of 18,432 | bit-equal, 0 of 20,736 |
| block rows `hd` (8 x 2,304 a anchor) | 1.15e-2 (1.54e-2) | 6.32e-3 (9.69e-3) |
| Markov state / latent (teacher forced) | 1.23e-2 / 8.87e-3 (1.53e-2 / 1.15e-2) | 7.24e-3 / 4.97e-3 (1.02e-2 / 6.97e-3) |
| logits over 200,058 columns, teacher forced | max abs diff 0.375 (0.5), mean 3.2e-2 | 0.3125 (0.375), mean 2.1e-2 |
| padding columns >= 200,058 | -inf (asserted) | -inf |
| free-running drafts equal to the oracle's | 56 / 56 (56 / 56) | 60 / 63 (63 / 63) |
| teacher-forced picks equal | 56 / 56 (56 / 56) | 62 / 63, 1 flip at an oracle gap of 0.125 (63 / 63) |
| leading drafts equal to the target's next tokens | 0 everywhere (random features, random tokens) | `[0, 0, 1, 5, 4, 3, 5, 1, 1]`, the oracle's own acceptance on every anchor |

The errors are the dense EXL3 launch's own (it rounds the rotated input to fp16: 5.6e-4 to 9.2e-4 of a row's rms against the exact decode,
`kernels-dense`), accumulated over 3 layers; there is no linear in `f_rnn`, which is why it is exact. The tolerances in the test (hd, Markov 3e-2, logits 1.0,
K/V 2e-3 bf16 / 1.5e-2 int8) are 2 to 2.5 times the first measured run (set after it, not before).

| claim | test | result |
|---|---|---|
| rows are invariant: a request alone, in a reversed batch, repeated, equals its place in a batch of 10 (lengths 5, 700, 1023, 1024, 1025, 1031, 1032, 1033, 1100, 1300); drafts and every traced plane bit for bit; both plane formats | `mini_rows_are_invariant_across_windows` | pass |
| incremental commit equals one commit (2 chunks; 1,000 rows then 15 commits of at most 7), rollback equals replay (8 rows of other features committed at 1100 then 3 true rows rewritten; a rewind twice), the stored K/V of every layer and `f_rnn` equal position by position | `mini_incremental_commit_equals_one_commit_and_rollback_equals_replay` | pass, both planes |
| a restored slot (rings, `f_rnn`, book from a snapshot, plane pages copied) proposes the same bits | `mini_slot_snapshot_restores_the_bits` | pass |
| graph replay equals eager (free and forced, drafts, `hd`, Markov state; replayed twice), PDL mask 0 and 15, both planes | `mini_graph_replay_equals_eager` | pass |
| the int8 plane tracks the bf16 plane (5 lengths to 1,300) | `mini_int8_plane_tracks_the_bf16_plane` | free drafts agree 35 / 35, layer-1 K/V rel L2 1.05e-2, `hd` 1.34e-2 |
| norm rows at 2,304 (256 threads of 9 columns) equal float64 | `mini_norm_rows_equal_the_float64_reference` | 0 of 5,529,600 outputs differ |
| refusals: nothing committed, anchor past the vocabulary, a slot twice, a hole, too many requests, a short tap stride, a wrong geometry | `mini_refuses_what_it_cannot_draft` | pass |
| golden hashes of the Mini run (both planes) | `mini_golden_hashes`, `testdata/gpu/golden_inkling2_loop_dspark.txt` | equal |
| host geometry and loader (on a synthetic one-group file) | `families::inkling2_loop::draft` (3 tests, Mac) | pass |
| **Small regression**: the Small's drafter on the Small's real pack (TP4 emulated on one GPU), incl. byte-exact golden hashes `testdata/gpu/golden_inkling2_dspark.txt` (file unchanged), int8 plane, rollback, snapshot | the 11 Small tests of `inkling2_dspark` | 11 passed (+ 1 ignored benchmark); `inkling2_ops`: 28 passed (6 ignored); the Mac's `families::inkling2::` 31 of 31 |

## Measured cost (one request, 1,300 positions, int8 plane, GB10 at 2,190 MHz; one run, `mini_propose_timing`)

Propose 7.89 ms eager and 7.73 ms as a captured graph (one round of 7 drafts); commit of 8 rows 0.18 ms; weights on the device 339,613,248 B (the 29 linears,
BF16 parts and `markov_w1` as in the pack, plus the fold's `fold_w` + `fold_b` 103,359,232 B: 0.316 GiB against the plan's 0.220 GiB for the pack's drafter); state and scratch for
one slot, one request: 28,592,224 B; per slot `MiniKv::slot_bytes` 8,458,752 B; per page of the plane 528,384 B (int8). Upload with the fold 0.04 s. Where the 7.9 ms
goes is not measured (by bytes: the 8-row EXL3 unembed 347 MB, then seven `markov_logits` steps that each read the 103 MB fold, then the 21 EXL3 launches of the layers); the
launch is not tuned.

## Not proven / UNKNOWN / risks

* UNKNOWN-17 stays open (above). Only one real target fixture existed: `raw-counting`, `chat-sky-effort0.9` and `raw-long-window` were not available (the node-ops
  fixture job's `spec.log` ended, when read, with `AssertionError: chat-sky-effort0.9: cached decode differs from the full recompute`; not investigated here), so real
  features are tested at 40 positions, below the 1,024 window: the window edges are covered by the synthetic `long` case and by row invariance only. `mini_matches_the_oracle` runs every case in
  `REXL3_INKLING2_LOOP_DSPARK_ORACLE`, so more `real_*` cases can be added by running `tools/gen_real_case.py` per fixture.
* A free-running draft can differ from the oracle's at a near tie (3 of 63 on the real case's bf16 plane, one teacher-forced flip at gap 0.125): the engine is deterministic and
  row-invariant, the oracle is the tolerance anchor (as for the dense launches).
* The tolerances are those of one run of two cases. The Small's oracle comparison tests (`matches_the_oracle_*`) were not run: no Small oracle fixtures on the node.
* Not wired: the target forward must write the 4 taps (the input partial of visits 5, 23 and 41, bf16, before the visit's mix, and the final-norm output) at a row stride of
  `TAP_ROW` (9,216 elements) or more, the engine's spec loop must call `stage_proposal` / `enqueue_proposal` / `finish_proposal` (or `propose`), `commit` the accepted rows with their taps and
  `copy_plane_page` on a copy-on-write; `inkling2_loop_dspark` is not in `BUILT_COMPONENTS`, and `engine::build` refuses a drafter by name until then.
* The page plane's byte layout is the drafter's, not `Shape::MINI`'s K page / V page (the pool slot size is equal; see above). Never run against the engine's own page pool.
* Up to 16 requests a call run the EXL3 decode path (<= 128 rows); more run the large-row path, which gives each row the same bits by its contract (`kernels-dense`); tested to 10 requests.
* The oracle sources' sha256 differs between `long_meta.json` and the real case's (`80f7767c...` / `c7d4a6e8...`) only because `tools/gen_real_case.py` was added to the lab's `oracle/` directory between the two runs.

## Files and how to reproduce

`oracle-planes/`: the compact planes of both cases (drafts, hd, f_rnn, state, latent, anchors, tokens, meta; for the real case also feats and ctxkv; 2.1 MB); the big planes (`*_logits.npy`
45 MB and 50 MB, `long_feats.npy`, `long_ctxkv.npy`) stay on the node (`sha256.txt` and `sizes.txt` list every file's hash and size).

```sh
# oracle planes (node-1 lab, CPU only, niced; the venv is torch 2.14.1 cpu + numpy 2.5.5 from oracle/requirements.txt)
python oracle/gen_dspark_fixtures.py --pack $PACK --fixtures none --out fix/dspark --threads 8 --only long
python oracle/gen_real_case.py --pack $PACK --fixtures fix --out fix/dspark --name chat-capital-effort0 --anchors 1,5,13,24,25,26,30,33,39
# the gate (the Mini's 10 tests and the Small's 11 in one binary)
NODE=<node-1> LAB_SRC=<engine worktree> TEST_ENV="REXL3_INKLING2_DIR=<Small pack> REXL3_INKLING2_LOOP_DSPARK_ORACLE=<lab>/fix/dspark" OPS_LINES=800 OPS_GREP=<see meta.json> scripts/ops_dev.sh test inkling2_dspark ''
```

Node paths created (all under `/home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-draft/`, 2.0 GB; remove with `NODE=<node-1> LANE=rexl3-inkling2-loop-draft scripts/ops_dev.sh clean`):
`src/` (98 MB), `target/` (811 MB), `venv/` (750 MB, the oracle's CPU Python), `oracle/` (0.5 MB), `fix/` (165 MB: the copied target fixture, the oracle planes, the logs), `build.*`. No process left running; no server touched.


## Public copy

The float activation planes of `oracle-planes/` (`feats`, `ctxkv`, `hd`, `frnn`, `state`, `latent` of both cases) are computed on the pack's real tensors and are not in
this public repository; their sha256 are in `oracle-planes/sha256.txt`. The token, anchor and draft-id planes and the `*_meta.json` files are kept.
`oracle/gen_dspark_fixtures.py` regenerates them from the public pack.
