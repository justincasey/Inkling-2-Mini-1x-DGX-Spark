# oracle-cacheddiff: why the cached decode differed from the full recompute (gate `oracle-cacheddiff`)

Result **pass** (`meta.json`). Node: node-1 (idle, CPU only, the GPU unused), pack at the pinned revision.

What happened: the fixture job's `--verify-recompute` demanded bit-identical logits and stopped at `chat-sky-effort0.9`.

| Question | Answer (file) |
|---|---|
| How big? | One logit of 49 x 200,058 (row 28, token 64928) differed by 1.19e-7 = **1 fp32 ulp** (7.2e-9 of the row rms). Argmax and top-2 order equal, hidden states of all 48 passes, routes, bank partials identical (`logs/real_diff.log`, `logs/chat-sky-effort0.9.report.json`, `logs/analyse.log`) |
| Which op? | The unembed of a decode row. Replaying the stored hidden states: the float64 product of the 1-row and the 49-row operand differ by 4.66e-15 and straddle the midpoint between two fp32 values; both stored logits reproduced bit for bit (`logs/element.log`) |
| Noise or semantics? | Noise: summation order (BLAS gemv vs gemm). The rate on the real unembed is 9 flips in 64,225,280 logits, always 1 ulp (`logs/rate.log`). The prefill rows (same kernel) match; only decode rows can flip |
| Could the unit tests see it? | No: the tiny looped model's cached run is bit-identical to the recompute at six lengths and H = 256, 1152, 2304, on the Mac and node-1 (`logs/synth_node1.log`, `tools/synth*.py`): the flip needs ~10^7 logits |
| Fix | `gen_fixtures.py`: `--verify-recompute` = bf16 planes and routes bit for bit, logits and routing weights within 2 fp32 ulp, argmax and top-2 order equal; `--verify-exact` keeps the exact comparison; a failure writes the fixture, records it in the manifest and does not stop the job. `tests/test_verify_recompute.py` (12 tests); the suite: 187 passed (`logs/mac-tests.txt`) |
| Re-run with the fix | `raw-long-window` (1,119 + 8), `chat-sky-effort0.9`, `raw-counting` all verified: 0, 1, 0 differing logits (`logs/spec-manifest.json`, `logs/long.log`, `logs/short.log`). `raw-long-window` continued the count past the 1,024 window: ` 561 562 563 564` |
| Invalid artifacts? | None for `emulate none`: the oracle is bit-identical before and after LoopRank's change (`logs/old-vs-new-oracle.txt`). Engine-emulating fixtures made before `dense_fp16` (numerics_version 1) are stale for the engine gate |
| Timings | in `meta.json` and `oracle/README.md` ("Fixtures on the real pack") |

Lab paths appear only as `lab/oracle-cacheddiff/` on node-1 (the venv of the drafter lane was used read-only). Fixtures written there are not in the repo (1.2 GB for the long prompt).
