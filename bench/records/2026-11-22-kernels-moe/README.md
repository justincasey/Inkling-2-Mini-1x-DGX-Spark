# kernels-moe: mixed-K routed experts on one GB10 (plan step 4.3, audit lane A)

Gate `kernels-moe`, result **pass** (`meta.json`). Engine: branch `inkling2-loop-moe`, clean HEAD `b6a965455ebe21dfa5e44681b8e66fc1c9c5bd57`
(commits `b9bf8de` FEAT mul1 softcap prefill + band entries, `0c4d4fb` TEST mixed-width GPU tests, `441117e` TEST goldens + pack-backed
test, `b6a9654` TEST clock sampling), node-1, SM clock sampled 2177-2184 MHz under load.

## What is proven (raw outputs next to this file)

| Claim | Test (binary `inkling2_moe_mixed` unless noted) | Raw |
|---|---|---|
| mul1 prefill softcap gate/up at K = 4 and 5: decode == prefill, byte for byte per row AND per pair (`y_pairs`), single-block, parallel and fused plans, 1-300 rows, random / hot / bands-1+2-only / band-0-only routing, a row alone == in a batch; small shape (512 x 256, 32 experts) and the Mini's (2304 x 1152, 128 experts) | `mul1_bands_decode_equals_prefill_small`, `..._mini_shape`; `inkling2_moe`: `mul1_prefill_equals_decode_at_k4_and_k5` (the Small's single-band functions at mul1 K4, K5) | `inkling2_moe_mixed.txt`, `inkling2_moe.txt` |
| Band composition, decode (7 launches per band + 1 combine) and prefill: a mixed layer's pair for expert e equals pair e of a UNIFORM layer of e's class holding the same weights (the Small's single-band chain), at every batch position, decode and prefill; the layer output is the fp32 `fmaf` chain of those pairs in slot order, rounded once to bf16, computed on the host; two layouts (three bands, and two 1-expert bands) | `band_split_equals_single_band_small`, `..._mini_shape` | same |
| The bands entries on a uniform layer equal the Small's `run_softcap_partial` / `run_prefill_softcap_partial` (mul1 K4, K5, mcg K4) | `band_entries_on_a_uniform_layer_equal_the_single_band_functions` | same |
| Poisoned scratch (0xFF, 0x00): every pick's `y_pairs` row is written (no NaN, identical under both poisons, equal to the uniform layer's), decode and prefill | `every_pair_row_is_written_once_small`, `..._mini_shape` | same |
| Graph replay == eager over three bands (22 launches), decode and prefill | `graph_replay_equals_eager_over_bands` | same |
| Refusals: mul1 prefill at 3 bits, with no instantiation, is refused instead of computed | `bands_entries_refuse_what_they_do_not_run` | same |
| Mini-shape golden on REAL tensors against the oracle's expert FFN, layers 12 (bands 107 / 17 / 4 experts), 22 (125 / 2 / 1: the all-5-bit single-expert band) and 23 (127 / 1: 5-bit down, 4-bit gate and up): cosine >= 0.9999 and max error <= 3 % of the row RMS (the Small's gate) on `out_exact` and `out_fp16in`, three rows' pairs; decode == prefill on the golden's 16 rows and on 300 rows cycling them; each row of the 300 equals the golden row alone | `real_mini_layers_equal_the_oracle` | `inkling2_moe_mixed.txt` (cosine 0.9999980-0.9999982 against `exact`, 0.9999998 against `fp16in`, max 0.0137-0.0171 of the row RMS, the same maxima the oracle's own `exact` and `fp16in` variants differ by (golden README, check b); pairs: cosine 0.9999994-0.9999996, max 0.0045-0.0073 of the pair RMS) |
| Small regression (byte-exact): the Small's committed golden hashes still match; its other MoE GPU tests pass; lib tests 31 of 31 | `inkling2_moe`: `golden_inkling2_moe_hashes`, `partial_equals_the_float64_reference`, `prefill_equals_decode_and_rows_are_invariant`, `prefill_with_tight_class_grids_equals_decode`, `four_rank_banks_sum_to_the_tp4_result`, `graph_replay_equals_eager`; Mac: `families::inkling2::` | `inkling2_moe.txt`, `lib-families-inkling2.txt` |

## What is NOT proven

- The Small real-pack test (`real_layers_equal_the_reference_and_the_oracle`) did not run: the Small pack is not on node-1 (it prints "not run" and passes).
  The Small's bytes are covered by its committed golden hashes (unchanged) and its synthetic tests, not by a real-pack run.
- The goldens cover layers 12, 22 and 23 (16 rows each). Layers 6-11 and 13-17 (up to 68 all-5-bit experts) were not run against the oracle;
  layer 22 and 12 carry the same classes.
- A pick outside `0..experts` is not tested for the bands entries (its pair row is not written; the combine skips it).
- No server-level or multi-layer run: this lane is the MoE kernels and their host composition only.
- Only the first-occurrence band order of `from_groups` is exercised; band count is bounded by (bits_gu, bits_d, codebook) triples (at most 3 in the Mini).

## Audit UNKNOWN 2: a decode token's MoE cost over bands

`bench.txt` (test `bench_decode_cost_of_bands`; medians of 200 calls, microseconds per layer, one decode token = 1 row, L2 flushed before each call, CUDA events;
eager = 7 launches per band + 1 combine from the host, graph = the engine's mode). Real layers read the pack's tensors; "synthetic" are random trellises at the Mini's
shape with layer 12's class mix.

| Layer | Bands (gate/up bits, down bits, experts) | rows 1 eager | rows 1 graph | rows 4 eager | rows 4 graph |
|---|---|---|---|---|---|
| real 0 | (4,4,128) | 172.9 | 148.4 | 545.1 | 538.5 |
| real 23 | (4,5,1) (4,4,127) | 201.6 | 162.5 | 591.7 | 559.8 |
| real 22 | (4,4,125) (4,5,2) (5,5,1) | 230.0 | 173.7 | 619.6 | 575.3 |
| real 12 | (4,4,107) (5,5,4) (4,5,17) | 242.6 | 193.1 | 646.0 | 602.8 |
| synthetic 1 band | (4,4,128) | 171.9 | 148.2 | 554.8 | 534.3 |
| synthetic 3 bands | (4,4,73) (4,5,37) (5,5,18) | 261.0 | 217.8 | 668.5 | 622.2 |

Reading: a graph-replayed decode token costs +14 us for layer 23's second band (one expert, usually no pick), +25 us for layer 22's third, +45 us for layer 12 (against the one-band layer 0).
Part of layer 12's and the synthetic layer's extra is bytes (5-bit tensors are 17-25 % larger per expert); the rest is the extra bands' launches and tails. Derived, not measured end to end:
pricing the 48 passes of the schedule with these medians (layers 0-5 and 20 at layer 0, 18/19/21/23 at layer 23, 22 at layer 22, layers 6-17 x 3 at layer 12) gives 8,814 us of MoE a decode token
against 7,123 us if every layer were layer 0 (`meta.json` `estimate.*`; layers 16 and 17 hold more 5-bit experts than layer 12, so the sum is a lower bound).

Clock: `nvidia-smi` sampled while measuring: SM 2177-2184 MHz, 13 samples, 55-96 % utilisation (max 3003, application graphics 2418). The test does not set or check a clock lock;
NodeOps reports 2,200 MHz locked. A sampler thread polls `nvidia-smi` every 0.5 s during the measurement (a small host-side perturbation of the eager numbers).

## Other facts

- Layer 23 has no all-5-bit class (census: 127 x (4,4,4), 1 x (4,4,5)); the plan's step text named one. The all-5-bit 1-expert band is in layer 22; goldens: `testdata/inkling2_loop/gpu/moe_layer{12,22,23}.safetensors` (engine tree), generator `gen_moe_golden.py` and `golden-gen.log` here.
- The answer to audit UNKNOWN 10 (is prefill's second remap + plan per band needed): no. The band entries remap and plan once per band and both GEMMs read that plan; decode == prefill per pair holds (tests above).
- Node paths created: `/home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-moe/` (lab tree, 1.1 GB; `ops_dev.sh clean` removes it). Nothing else.
