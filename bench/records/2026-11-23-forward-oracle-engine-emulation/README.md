# forward-oracle-engine-emulation: the forward-oracle gate with the oracle that models the dense kernels

Supersedes `2026-11-23-forward-oracle` (same engine commit, same dumps; that record stays as it is: `fail`, because the oracle then had no
emulation of the dense EXL3 linears). Gate `forward-oracle`, node-3, engine `aeaaa4f94944672578bac340379729ba87f85d8a` (clean, the engine tree
was not touched), oracle at this recipe branch's commit in `meta.json`: `Emulation.dense_fp16` added (`oracle/inkling2_ref.py`, NUMERICS.md,
`oracle/README.md`), `teacher_force.py`'s attention variants gain `dense_fp16` and `kv8+attn_bf16_p+dense_fp16`, the gate's default is the
latter (with `moe_fp16`). **No limit was changed**: the Small's, as before. 175 oracle tests pass (`oracle/.venv/bin/python -m pytest -q oracle/tests`).

## Result: the stock gate passes, 48 of 48 passes on both prompts (`oracle/*-teacher_force_report.json`, `"pass": true`, no wrapper)

| Metric (stock limit) | sky, 26 tokens | counting, 35 tokens |
|---|---|---|
| attention core `rel_med`, max over passes (2e-3) | 9.31e-4 | 7.04e-4 |
| attention output `rel_med`, max over passes (6e-3) | 2.18e-3 | 1.61e-3 |
| attention core worst row (`rel_max`, no limit) | 3.07e-3 | 1.91e-3 |
| MoE input mix `rel_max` (2e-3) | 6.4e-4 | 7.9e-4 |
| `msg` / `attn_in` `rel_max` (2e-3) | 1.5e-7 / 8.5e-4 | 1.8e-7 / 9.0e-4 |
| MoE output `rel_med` max, `moe_fp16` (1.5e-3) | 5.9e-4 | 5.9e-4 |
| router logit error / sum abs (1e-6) | 3.2e-7 | 2.9e-7 |
| router picks identical, flips | 1.000 in all 48 passes, 0 | same, 0 (smallest margin 7.0e-6) |
| `restart_mismatch` / `residual_mismatch` (elements) | 0 / 0 | 0 / 2 |
| attention noise floor (oracle vs a second fp32 implementation), core `rel_med` max | 1.4e-4 | 3.1e-4 |

Before the emulation (first record): core medians 1.7e-3 to 4.3e-3, 37 and 26 passes over 2e-3. The engine did not change. The attention
output limit (6e-3) was at the edge for the counting prompt (6.000e-3) and is now 1.6e-3. No re-measured limit is proposed: every metric is
inside the Small's limits (core by 2.1x to 2.8x, output by 2.8x to 3.7x); two prompts are too few to tighten them.

## What `dense_fp16` covers

Every dense EXL3 linear of the target the oracle evaluates: attention q, r, gate, k, v, `wo`, and the unembed (the logits plane of
`gen_fixtures` and the dump). Exact still: BF16 tensors (router, `attn_sink_proj`, `weight_proj` taps, controllers, convs, queries) and the
embedding. The drafter oracle (`dspark_ref.py`) and the vision tower are untouched. Tests: `oracle/tests/test_dense_fp16.py` and the
changed expectations in `test_teacher_force.py` (the synthetic engine dump is made with the pack's weights and `Emulation.engine()`; a new
test shows the old variant misses a dump that carries the rounding).

## Engine checks (unchanged, from the first record's `raw/`)

Determinism (byte-identical 90,056,120 / 121,229,024-byte dumps), chunk-width invariance, decode == prefill row, window commit/rollback, and
1,119 tokens past the window (chunks of 129 and 300, decode): all bit-exact; see that record's README.

## Not compared, and the cost of more

- A prompt past 64 tokens against the oracle. The 26- and 35-token comparisons took 1,905 s together (two runs in parallel, 6 expert workers
  each, 2 threads, `nice 19`): the expert decode dominates and grows linearly with tokens, so 128 tokens is about 1.5 to 2 hours and 1,119
  tokens about 15 hours or more on this node. Not started.
- Positions past 8,192 (log scaling), the logits plane (the dump carries it; `teacher_force.py` does not compare it), the vision tower, the drafter.

## Node footprint

`/home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-rank/` (as in the first record; plus `out/v2_*.json|log`). No process left running.
