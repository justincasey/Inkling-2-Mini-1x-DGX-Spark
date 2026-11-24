# Eval matrix

The bring-up gates of `docs/exec-plans/completed/inkling2-mini-rexl3.md` Phase 6 (6.1-6.10), in order. Stop at the first failure.
Bring-up order: `doctor` -> `stop-other` -> `fetch` -> `verify` -> `build` -> `start` -> `status` / `logs` on failure.

Column "State" names the record that closes the gate at the pinned engine (`bench/records/2026-11-23-final-<gate>/`, written by `scripts/final_run.sh`; its figures are in
the README's Results table and in `docs/REXL-PR-NOTES.md`'s evidence index, both generated from the records). The ladder (`./autoresearch.sh`) judges a gate by the newest record of its
gate name (`scripts/check_record.py`).

| Gate | Command | State | Pass condition | Proves |
|---|---|---|---|---|
| 6.1 boot | `start`, `status`, `memwatch begin\|report`, `record` | **pass** at the pin: `final-boot`, `final-capacity`, `final-memory` (the guard held with four cold prompts of the window at once) | `/health` answers 200; ready `MemAvailable`, the guard's minimum and the engine's KV capacity line recorded against `docs/FIT.md` | the stack loads and the guard held |
| 6.2 smoke | `smoke` | **pass** at the pin: `final-smoke` (chat, completion, tool call, reasoning on and off, image), `final-vision-smoke` | chat, completion, tool call, reasoning separated (effort high and none), image (skipped only with `VISION=off`) | the serving path is not silently wrong |
| 6.3 oracle | `oracle/compare.py` and the fixtures | **run**: op-level teacher forcing `forward-oracle-engine-emulation`, the engine-to-oracle gap `oracle-gap` (`docs/ORACLE-GAP.md`), the end-to-end greedy check `greedy-vs-oracle` | op-level teacher forcing within the documented tolerances; free-running drift measured against the documented limits | the engine computes what the model computes |
| 6.4 byte-exactness | `bench/tools/exactness.py` and the `spec`, `cache`, `chunk`, `vision` gates of the controller | **pass** at the pin: `final-exact-basic` (concurrent == solo, chunk width, boot to boot), `final-spec` (drafted == serial), `final-cache` and `final-cache-images` (cached == uncached), `final-pin` (`bench/reference.json` and `reference-vision.json` pinned, reproduced byte for byte by a fresh boot) | drafted == serial; cached == uncached; concurrent == solo; chunk width changes no byte; boot to boot | every speed change is byte-exact |
| 6.5 needles | `bench/tools/needle.py` | **pass** at the pin: `final-needle` (the shipping window, 10% and 50% depth); the 1M-token window on an earlier engine: `needle-window` | code found at 10% and 50% depth at the profile's window | long-context recall at the window |
| 6.6 unknowns | none: the engine's own logs are the input | **done as analysis**: route statistics (`docs/C4-DECODE-PLAN.md`), per-position acceptance and the policy (`spec-speed`, `markov-variants`); decided: ship `SPEC=dspark`, prose is the one kind that loses (`docs/DECISIONS.md`) | route log (expert overlap across consecutive tokens and across loop passes), per-position drafter acceptance, 5-bit expert route frequency; decide the MTP depth and policy, or ship serial | whether the drafter pays on an expert-bound model |
| 6.7 quality, tools | `bench/quality_gate.py`, `bench/toolgate.py`, `scripts/loop_probe.py` | **fail** at the pin: `final-quality-tools` (the 60 exact-answer questions are right; the tool gate passes 25 of 28 cases, the three misses call `run_sql` first; no invalid or destructive call); `final-loop-probe` pass | baselines recorded: fixed exact-answer set; schema-valid calls; loop probe; the base card's destructive-action warning | quantization did not break answers; the agent surface works |
| 6.8 api-test, probe | `tests/api_conformance.py`, `bench/probe.py` | **pass** at the pin: `final-api-test` (0 FAIL, probe included) | 0 FAIL; models, tokenize, usage, reasoning, tools, spec metrics | client contract; later measurements are meaningful |
| 6.9 speed | `bench/tools/bench.py`, `bench/rexl3/sweep_decode.py`, `sweep_prefill.py`, `profile_report.py` | **run** at the pin: `final-speed-baseline`, `final-speed-decode`, `final-speed-prefill`, `final-speed-c4`; the full context sweeps (4K to 1M) were not run. `decode_c4_tok_s` is the span to the last reply (the tail runs one stream with the drafter) and `decode_c8_tok_s` is 8 requests on 4 slots, two waves, not an 8-stream batch; the sustained c=4 number is `bench rolling` (`decode_c4_rolling_tok_s`, gate `speed-c4`, `docs/HARNESS-NEXT.md` section 7) | decode (1, 2, 4, 8 streams; 4K to 1M context), prefill (8K to 1M), TTFT, cached TTFT, against `docs/ROOFLINE.md` | throughput, as raw numbers first, then against a record |
| 6.10 longctx, sixcat | `bench/longctx.py`, SixCat pinned by `SIXCAT_REPO` / `SIXCAT_COMMIT` as RED-SNOW does | **not run** (open): the tools exist (`bench longctx`, `bench sixcat`); the 1M needles and the 4 x 1M lanes are in `needle-window` and `profile-max` | PASS at every length up to the window; score recorded | recall and prefill curve; cross-recipe comparability |

Gates outside Phase 6 that the same controller runs:

| Check | Command | State | Pass condition |
|---|---|---|---|
| static | `bash tests/test_recipe.sh`, `python3 tests/test_records.py` | pass (the last line of each says `0 failed`) | last line `test_recipe: N passed, 0 failed` |
| host tests | `engine-test host` | **pass** at the pin: `final-engine-tests` | every suite runs at least one test, none fails or is skipped |
| golden kernel hashes | `engine-test goldens` | **pass** at the pin: `final-engine-tests`. These are the **Small's** golden files (attn, ops, moe, vision, dspark): the Small's regression evidence. The Mini's own goldens are in the next row | every golden passes; a golden that is "not run" fails |
| Mini GPU tests and goldens | `scripts/ops_dev.sh test` over the nine GPU test binaries (gb-cuda `inkling2_attn`, `_attn_oracle`, `_ops`, `_dspark`, `_moe`, `_moe_mixed`, `_loop_vision`; gb-model `inkling2_vision`, `inkling2_loop_vision`) | **pass** at the pin, on node-3: `engine-gpu-tests-pin` (both Mini goldens, `golden_inkling2_loop_attn.txt` and `golden_inkling2_loop_dspark.txt`, included). Eight of the tests print "not run" and pass for want of oracle fixtures that exist only in the lanes' labs (listed in the record's README) | every test passes; a test that cannot find its GPU or pack fails (`REXL3_REQUIRE_GPU=1`) |
| sampled and reasoning decode | `bench/tools/sampled.py run\|summary\|compare`, `scripts/loop_probe.py run --temperature 1.0 --seed 1` | **measured** at the pin, on node-2: `sampled-decode` (drafter on against off, same seeds, temperature 0 and 1.0, effort low and high; drafted == serial in 64 of 64 pairs) | the measurement completed with every request answered; a drafted != serial pair is reported as the metrics `ab_identical_t0` and `ab_identical_t1`, not hidden |
| process tests | `scripts/ops_dev.sh test process ""` (gb-server) | **pass** at the pin: `process-tests-linux` | every test passes on Linux |
| node | `doctor` | exists | GPU idle, lock free, MemAvailable >= `MIN_AVAIL_GIB`, pack at full size, installed engine equals `bin/rexl3-serve.commit` |
| pack | `verify` | exists | sha256 of every file equals `intake/manifest.tsv` |

The speed-loop harness of `docs/HARNESS-NEXT.md` (`target_score`, `REQUIRED_GATES`) was designed and not written: the ladder and the records took its place, and `scripts/final_run.sh` runs
the gates above in one command at the pin (`docs/HARNESS-NEXT.md` section 8).

Records: `bench/records/<date>-<name>/meta.json` written by `./inkling2mini.sh record NAME` or by the final run (recipe commit, pack, engine
commit and sha256, `engine.lab_build`, config, serve command, results).

Warm-up: one short and one long request before any timed run. Discard one 20 s decode pass after boot.
Never count SSE chunks for tokens; use `usage.completion_tokens`.
