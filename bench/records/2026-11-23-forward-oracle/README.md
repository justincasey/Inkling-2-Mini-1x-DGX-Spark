# forward-oracle (plan steps 4.4, 4.5 per-visit state, 6.3 dump hooks): the 48-visit forward of the Mini rank against the oracle

Gate `forward-oracle`. Node-3 (one DGX Spark, GB10 sm_121), engine worktree branch `inkling2-loop-rank` at
`aeaaa4f94944672578bac340379729ba87f85d8a` (clean, `dirty` false; `raw/engine-commit.txt`), the pinned Mini pack staged on the node.
Result in `meta.json`: **fail against the stock gate, on one metric**: see "Findings". The engine is bit-exact on every invariance and
determinism check.

## What was built (engine commits, local)

- `e6d2c62 FEAT: Added the Mini rank's weight loader and 48-visit forward`: `inkling2_loop/device.rs` (the bound pack onto one device:
  embedding, six 6-bit EXL3 attention tensors a layer, BF16 sink and dynamic-conv tap projections, experts as bands, controllers and
  merged message queries, EXL3 unembed; 13.553 GiB loaded in 7.7 to 10.2 s, files evicted from the page cache), `inkling2_loop/model.rs`
  (`Inkling2LoopRank`: per-visit state on `VisitState`, the loop's restart from the prelude with a buffer pool for the blocks and the
  message bank, prefill chunk and window paths, `commit`, `window_logits`, the dump). `changes/unreleased/inkling2-loop-rank.md`.
- `c038999`, `aeaaa4f TEST:` `crates/model/tests/inkling2_loop_rank.rs`: `forward_oracle_gate`, `long_context_gate`.
- Dump: `REXL3_INKLING2_LOOP_DUMP=<dir>` (one `prefill{N}.safetensors` a prefill chunk: `hidden` `[49,T,2304]` bf16, `core_out`, `h_final`,
  `logits` `[T,200058]` f32) and `REXL3_INKLING2_LOOP_DUMP_OPS=1` (the planes of the docstring of `oracle/teacher_force.py`: `attn_core`,
  `attn`, `mlp_in`, `router_logits`, `router_idx`, `router_w`, `moe` (f32 holding the bf16-rounded sum), `msg`, `attn_in`).

## Proven on node-3 (`raw/gate-*.txt`; every `LOOP_` line is a measured value)

| Claim | Test | Result |
|---|---|---|
| Two prefills of the same prompt write byte-identical dumps (all 11 planes) | `forward_oracle_gate` | sky (26 tokens): 90,056,120 bytes, sha256 `c25f9876...c45c0`, both runs; counting (35 tokens): 121,229,024 bytes, sha256 `0773323c...f43d`. The sky dump's hash is the same in the run that fed the oracle and in the final run |
| Chunk-width invariance | same | the prompt as one chunk and as chunks of 7 and 11 rows: the logits planes are byte-identical (sky: 20,806,032 bytes, sha256 `a105f245...f41a`) |
| Decode == prefill row | same | one-row decode windows at the last 4 positions: each row's 200,058 logits equal that position's in the prefill, bit for bit |
| Window commit and rollback | same | a window of 3 rows committed at 2, then a window continuing at position +2: all 6 rows equal the prefill's rows |
| Candidates | same | the best candidate of the last row is the first maximum of its logits (sky 200,001 at 26.82; counting 220 at 29.41) |
| Past the 1,024 window | `long_context_gate` | 1,119 tokens: one chunk, chunks of 129 and 300 rows (decode and large-row GEMM paths, MoE prefill), and a prefill plus 4 decode windows give the same top-64 candidates (ids and values) of the last row, bit for bit |

## The oracle comparison (`oracle/`; `teacher_force.py --engine <dump> --pack <pack>`, nothing changed in the oracle)

Each pass is fed the engine's own inputs; the numbers are the report's per-pass row statistics (`rel_med`: median over rows of
|oracle - engine| / |engine|). Stock report = the oracle's `kv8+attn_bf16_p` attention variant, `moe_fp16` experts, the Small's limits.

| Metric (gate limit) | sky, 26 tokens | counting, 35 tokens |
|---|---|---|
| attention core `rel_med` max over 48 passes (2e-3) | **3.81e-3** (37 passes over; worst 11, 23, 35, 5, 17); min 1.78e-3 | **4.26e-3** (26 over); min 1.65e-3 |
| attention output after `attn_sconv` (6e-3) | 5.4e-3 | **6.000e-3** (at the limit) |
| MoE input mix `rel_max` (2e-3) | 6.4e-4 | 7.9e-4 |
| `msg` / `attn_in` `rel_max` (2e-3) | 1.5e-7 / 8.5e-4 | 1.8e-7 / 9.0e-4 |
| MoE output `rel_med` max, fp16 inputs (1.5e-3) | 5.9e-4 | 5.9e-4 |
| router logits error / sum abs (1e-6) | 3.2e-7 | 2.9e-7 |
| router picks identical / flips | 1.000 in all 48 passes / 0 | 1.000 / 0 (smallest selection margin 7.0e-6) |
| `restart_mismatch` (pass restarts equal the prelude's state) | 0 | 0 |
| `residual_mismatch` (elements) | 0 | 2 |

With the same dump and the engine's `exl3_fp16` emulation on the oracle's dense EXL3 linears (`tools/teacher_force_exl3_fp16.py`, a
wrapper that only patches `Inkling2LoopRef.lin`, `--no-moe`): attention core `rel_med` max **9.3e-4** (sky) and **7.0e-4** (counting), attention
output 1.9e-3 and 1.6e-3 (the `wo` projection is not patched), and the gate passes (`oracle/*-exl3_fp16-report.json`).

## Findings

- **The attention-core miss is the oracle, not the engine.** The dense EXL3 kernels store the rotated input as fp16 (`exl3_fp16`, NUMERICS
  "Engine emulation"; the dense lane measured 5.6e-4 to 9.2e-4 of the row rms against the exact oracle), `teacher_force.py` has no variant that
  emulates it for the attention linears (`ATTN_VARIANTS` has kv8 and attn_bf16_p only), and a 1-ulp bf16 difference in q, k, v shifts the core by
  about 2e-3 (28 percent of elements differ). The patched oracle brings it to 0.7e-3 to 0.9e-3 with no engine change. The gate's limits are
  the Small's and say "to be re-measured": 2e-3 for the core median is below the Mini's measured 3.8e-3 to 4.3e-3 without the emulation.
- Router, MoE, mix, messages, controllers (`attn_in`), the restart of the three core passes and the residual are inside the limits on all 48 passes
  of both prompts; the engine's loop (bank, blends, restart) agrees with the oracle's.
- Not measured: the global attention at positions past 8,192 (log scaling) through the whole forward; a prompt past 64 tokens against the oracle
  (the oracle ran 26 and 35 tokens; 1,119 tokens are checked for self-consistency only); speed.

## Node footprint

`/home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-rank/` (7.1 GiB): `src` and `target` (the build), `oracle/` with its
`.venv` (5.2 GiB, `uv`, Python 3.12, `requirements.txt`), `out*/` (dumps, ids, reports). Remove with
`NODE=<node-3> LANE=rexl3-inkling2-loop-rank scripts/ops_dev.sh clean`. No process left running.

**Superseded** by `2026-11-23-forward-oracle-engine-emulation` (same engine commit and dumps; the oracle now carries `Emulation.dense_fp16`
and the stock gate passes on both prompts). This record is kept as it was.
