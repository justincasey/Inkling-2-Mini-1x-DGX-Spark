# Notes for a later submission into REXL

Status (2026-11-24): the owner asked for the issue to be opened. The series head is the branch `inkling2-mini` of the owner's fork `justincasey/REXL3` and is offered to REXL as a `[finding]` issue (its URL is in the README); no pull request, no tag. The notes below are the analysis written before
filing (2026-11-23, at the owner's request: "capture any notes we can use later for opening a PR into REXL"), kept as written except for these corrections, which came from reading the guide and the maintainers' replies at filing time:

- the whole text diff is 3,001,584 bytes, 46 times GitHub's 65,536-character body limit (section 3 counted lines); #189 linked a gist and #141 a fork compare link; the issue uses the compare link and inlines the shared hunks;
- upstream `main` is now `e496c51b`, 5 commits past the base; the overlap (18 files) and the 5 conflicted files are the same as at `aa63b03d`;
- topics 1 to 17 of the series are #189's family (code-equal to `a73ff7a` apart from comments and release-note fragments); the Mini is topics 18 to 46;
- the maintainers' replies on #95 and #141 show they use a patch as a reference and write their own implementation, crediting the author, and that they ask for small separable diffs first and shared-kernel changes last: the issue's reading order follows that;
- no fork tag was pushed (the guide's Never list is not scoped to a repository); 8 of the 185 commit subjects do not satisfy `tools/check_pr_rules.py`'s title rule and were left as they are.

Series mechanics live in `docs/ENGINE_EXPORT.md`; the numbers live in `bench/records/` and are indexed in section 4.

## 1. How REXL takes a contribution (read from the engine repo's own files)

Sources: `CONTRIBUTING.md`, `AGENTS.md` and `.github/ISSUE_TEMPLATE/tester-fix-or-improvement.yml` in `~/projects/REXL3` (upstream `main`).

- **Two repositories.** `REXL3/REXL3` is the public production repo: sealed runtime trees, the policy tools, and a *snapshot* of the engine
  source (`crates/`, `third_party/`, `testdata/`, `Cargo.toml`, `Cargo.lock`, ...) that changes only by a maintainer-requested sync from dev.
  `REXL3/REXL3-dev` holds the engine, templates, profiles and every qualification gate; it is private (an unauthenticated fetch of its
  `CONTRIBUTING.md` returns 404), and production's `CONTRIBUTING.md` says to follow that repo's rules for engine work. We have not seen them.
- **Testers do not open pull requests.** Invited testers (and the maintainers) open a **`[finding]` issue** on `REXL3/REXL3` with the
  template `tester-fix-or-improvement.yml` and paste a **complete, tested `git diff`** in it; the maintainers apply the patch in
  `REXL3/REXL3-dev`, run the qualification gates and promote it. No bare feature requests: a coded, locally validated patch is required.
- **A new model or family** reaches production only through the dev repo's approval and `tools/promote_runtime.py`, in a
  maintainer-opened `release/v<X.Y.Z>` cut. An agent prepares a model branch or registry entry only on an explicit, quoted maintainer
  instruction. CODEOWNERS is `* @am423 @vcruz305`; the Mini pack (`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`) is vcruz305's.
- **The template's required parts:** an "Open issues and KNOWN_ISSUES.md checked" line (the search you ran and the `#N` you read); Scope
  (engine-wide or one family); Category; "Release tag, engine commit, runtime bundle, and profile tested"; **1. WHY**; **2. WHAT** (files,
  crates, shared or one family, bit-exact or not); **3. HOW TO FIX** (the `git diff`); **4. FULL-CONTEXT VALIDATION EVIDENCE**; four
  confirmation boxes (searched, validated at the full window, no secrets, not a vulnerability). Before filing, run
  `gh issue list -R REXL3/REXL3 --state all --search "<terms>"` and read what matches; comment on an open issue instead of duplicating it.
- **Validation standard (AGENTS.md sections 3 and 4):** at the profile's full context window, never 32K; speculative decoding and the
  prefix cache **on**; exactness (`drafted == serial`, `chunked == whole`, test pass counts); the median of three runs of decode tok/s,
  prefill 50% and 100% tok/s, acceptance rate and free GPU GiB, before and after. For a new family there is no "before": say so, and show
  the first-boot serial record next to the final one, and the Small's unchanged results as the regression side.
- **House rules that bind our text and commits:** titles `TYPE: Past-tense summary` (FIX, FEAT, BUILD, CI, TEST, BUMP, DOCS), at most 90
  characters, no trailing period, one logical commit per branch; a `changes/unreleased/` fragment for behaviour changes; GitHub noreply
  identity; no secrets, personal e-mail or internal project names; **SAGE is described only as "SAGE is our own mixed-precision
  quantization method for EXL3. The method is not published."**; AGPL-3.0-only, and only MIT, Apache-2.0 or BSD-3-Clause code may be copied
  or re-expressed, with attribution (AGPL, GPL or non-commercial code is ideas only); no model weights or derivatives in the repo.
  (`PROVENANCE:` lines per commit are this effort's own convention, from the plan and the CYBERFROST precedent, not a REXL rule.)
- **Never (any instruction):** push to `main`, create or edit a release or tag, edit a sealed tree or the engine snapshot by hand, change
  repository settings, approve or merge a PR, weaken a check in the same change it guards.

## 2. Base drift: the series must be rebased before anything is filed

- Our base is `02ef8cb` (upstream `main` on 2026-11-18: engine source tree v0.1.1, `.rexl3_commit` 75031112).
- Checked 2026-11-23 with `git fetch upstream` (read only): `main` is now `aa63b03d`, four commits on, among them
  `FEAT: Updated the engine source tree to the v0.1.3 engine commit 0c8bc43c (#192)`. Since our base upstream changed 235 files
  (`crates/model` 100, `crates/cuda` 68, `crates/server` 14, `SOURCE_IDENTITY.json`, `tools`, `crates/exl3`, `crates/engine`, ...); our series
  touches 285 (series head `fb5e3a7b`, rechecked at the end of the work); **18 files overlap** (the same 18 as at the first check).
- Dry run `git merge-tree --write-tree` of the series head onto `aa63b03d` (no ref or worktree touched; rechecked at the end): **5 conflicted files**:
  `crates/cuda/csrc/moe_prefill.cu`, `crates/cuda/src/moe_prefill.rs`, `crates/model/src/serve.rs`, `crates/server/src/http.rs`,
  `crates/server/src/main.rs`. The other 13 overlapping files merge cleanly (clean textual merge is not proof they are right).
- The rebase changes the engine bytes, so it invalidates every record: after it, rerun the gates on the rebased build (section 8). It was
  not done in this wrap-up on purpose: the shipped, tested build stays the one the records name.
- Recommended order when the owner says go: `git fetch upstream`; rebuild the series on the new base with the export build script
  (`docs/ENGINE_EXPORT.md`), resolving the conflicts above once; build and run the gates on a node; only then draft the finding.

## 3. What the series changes

- Size of the series head `fb5e3a7b` against base `02ef8cb`: 285 files, +62,348 / -254 lines (42 modified, 243 added): `crates/cuda` +26.2k (57 files), `crates/model`
  +21.7k (48), `testdata/inkling2` +5.4k and `testdata/inkling2_loop` +3.9k (84 files of goldens and fixtures), `crates/server` +2.5k,
  `crates/engine` +1.1k, `crates/vision` +0.5k, `crates/spec` +0.2k, 55 changelog fragments, three tool files. Regenerate with
  `git diff --numstat 02ef8cb engine/inkling2-export-work`; `docs/ENGINE_EXPORT.md` has the same totals.
- **GitHub caps an issue or comment body at 65,536 characters.** A 66k-line diff (65,679 lines of text diff: `docs/ENGINE_EXPORT.md`) cannot be pasted into a `[finding]` as the template asks.
  Ask the maintainers how they want a patch of this size delivered (a link to a fork branch or an attached patch set; or one finding per
  topic) before drafting anything.
- **42 files outside the new family directory are modified, not added** (`git diff --name-status base..head | awk '$1=="M"'`). They are
  the risk to other families and the part a reviewer reads first: shared CUDA kernels and runtime (`crates/cuda/csrc/exl3_big.cu`,
  `moe_experts.cu`, `moe_prefill.cu`, `pdl.cuh`, `tp_oneshot.cu`; `crates/cuda/src/{graph,lib,moe,moe_prefill,nccl,rt}.rs`), the server
  (`template.rs`, `chat_format.rs`, `reasoning*.rs`, `http.rs`, `main.rs`, `field_policy.rs`, `formats/mod.rs`), `crates/model/src/{family,
  fingerprint,serve,window_graph}.rs`, `families/{mod.rs,glm/inventory.rs,glm/spec.rs,nemotron_h_omni/spec.rs}`, `crates/engine/src/lib.rs`,
  `crates/spec/src/{mode,stats}.rs`, `crates/vision/src/{lib,rgb_decode,webp}.rs`, `tools/switch_registry.json`, and five server test files.
  `docs/ENGINE_EXPORT.md` gives each one's reason and topic. **Evidence for the other families does not exist here:** the only regression
  evidence is the Small's (the 31 `families::inkling2::` host tests and the byte-identical GPU golden hashes of `engine-test goldens`).
  The maintainers' own qualification chain has to run for GLM, Nemotron, MiniCPM and the rest wherever a shared kernel or the server changed.
- Attribution: the family's semantics were read from the Hugging Face reference and vLLM's model definition (Apache-2.0), pinned by sha in
  the spec (`spec/`), with a `PROVENANCE:` line per commit; the export doc says whether `THIRD_PARTY_NOTICES.md` needs a line.
- **Pin and filing are two commits of one tree.** The pin and every record name the dev head H (`inkling2-loop`, after the lane merges and one scrub
  commit of comments, docs and fragments). What would be filed is the series head (`engine/inkling2-export-work`: stacked `engine/<topic>` branches,
  history rewritten to the rules). They differ by test data only: 25 files (10.8 MB) of goldens that are slices of real expert tensors or values
  computed on them, which CONTRIBUTING section 8 says not to add. `.export/check.sh` proves the relation and `docs/ENGINE_EXPORT.md` has the
  one-liner that shows it. **Never push the dev branch publicly** (it holds those 25 files); push only the series.
- **Shared behaviour changes** (file these separately; `docs/ENGINE_EXPORT.md`, "What shared changes can alter other families"): the stop-string fix in
  `http.rs` (it changes `usage.completion_tokens` and the returned ids for every family); `return_token_ids` accepted on non-streamed completions
  (its `TEST:` commit replaces one assertion, so file the two as one unit); block-order changes in `exl3_big.cu` and `moe_prefill.cu`. None was run
  on another family. The only kernel evidence is the Small's golden hashes (`engine-test goldens`).
- **Licence lines.** The export doc reports that `THIRD_PARTY_NOTICES.md` would need a line for vLLM PR #67325, `transformers` and the Hugging Face
  chat-template environment (the series edits no upstream file for this). The dense-pipe kernel carries a TensorFold and ExLlamaV3 MIT attribution
  that upstream's notices already use.

## 4. Evidence index

Generated from `bench/records/*/meta.json` (newest record per gate); regenerate with `python3 scripts/records_index.py`. No number in
this file is typed by hand.

<!-- records-index:start -->
| gate | record | node | engine | result | tests | metrics |
|---|---|---|---|---|---|---|
| `api-test` | `2026-11-23-final-api-test` | node-1 | `59d08a9c5e9f` | pass | 34 passed, 0 failed | api_checks_failed = 0<br>api_checks_passed = 17<br>boot_s = 16<br>probe_cached_tokens_repeat = 2352 |
| `boot` | `2026-11-23-final-boot` | node-1 | `59d08a9c5e9f` | pass | 2 passed, 0 failed | boot_s = 21<br>engine_ready_t_s = 15.381<br>gpu_process_mib = 46690<br>guard_loading_min_mib = 71447 |
| `cache` | `2026-11-23-final-cache` | node-1 | `59d08a9c5e9f` | pass | 18 passed, 0 failed | gate_prompts_equal = 7<br>gate_shared_cached_tokens_1k = 2027<br>gate_shared_cached_tokens_70k = 73192<br>gate_warm_cached_tokens_70k = 73104 |
| `cache-images` | `2026-11-23-final-cache-images` | node-1 | `59d08a9c5e9f` | pass | 8 passed, 0 failed | cache_images_after_text_warm_cached = 3411<br>cache_images_evictions = 0<br>cache_images_hits = 11<br>cache_images_image_first_warm_cached = 3399 |
| `cache-speed` | `2026-11-23-prefix-cache-speed` | node-3 | `de2d4a30e423` | pass | 16 passed, 0 failed | counter_evictions = 50<br>counter_hits = 67<br>counter_inserts = 66<br>counter_lookups = 121 |
| `capacity` | `2026-11-23-final-capacity` | node-1 | `59d08a9c5e9f` | pass | 1 passed, 0 failed | free_after_gib = 101.27<br>kv_bytes_per_token = 18576<br>kv_gib = 18.16<br>kv_tokens = 1048576 |
| `engine-build` | `2026-11-23-final-engine-tests` | node-1 | `59d08a9c5e9f` | pass | 196 passed, 0 failed | build_seconds = 115<br>engine_test_goldens_seconds = 70<br>engine_test_host_seconds = 76<br>golden_tests_passed = 6 |
| `engine-export` | `2026-11-23-engine-export` | - | `-` | pass | 292 passed, 0 failed | - |
| `engine-gpu-tests` | `2026-11-24-engine-gpu-tests-pin` | node-3 | `59d08a9c5e9f` | pass | 122 passed, 0 failed | binaries = 9<br>tests_failed = 0<br>tests_ignored = 11<br>tests_not_run_printed = 8 |
| `engine-gpu-tests-fixtures` | `2026-11-24-engine-gpu-tests-fixtures` | node-3 | `59d08a9c5e9f` | pass | 8 passed, 0 failed | still_not_run = 0<br>tests_failed = 0<br>tests_passed = 8 |
| `exact-basic` | `2026-11-23-final-exact-basic` | node-1 | `59d08a9c5e9f` | pass | 20 passed, 0 failed | exactness_replies_identical = 4<br>max_prompt_tokens = 8334<br>new_tokens = 128<br>prompts = 4 |
| `exact-fuse` | `2026-11-23-fuse-exact` | node-2 | `24a04afe9fda` | pass | 306 passed, 0 failed | exactness_records_identical = 4<br>specedge_cases = 65<br>specedge_failures = 0 |
| `exact-graph` | `2026-11-23-graph-exact` | node-2 | `9fa93ea6e8f1` | pass | 139 passed, 0 failed | exactness_records_identical_to_start_engine = 4<br>live_pin_replies_equal_to_start_engine = 5<br>long_context_top1 = 157<br>long_context_top1_logit = 23.62496 |
| `exact-prefill` | `2026-11-23-prefill-dense-pipe-exact` | node-3 | `fd6ef616be70` | pass | 247 passed, 0 failed | chunk_digests_equal = 33<br>exactness_replies_identical = 4<br>golden_exl3_big_lines_identical = 42<br>graph_windows_equal = 37 |
| `expert-arena` | `2026-11-23-expert-arena` | node-1 | `0a6b7e26e6c4` | pass | 152 passed, 0 failed | bootstrap_after_cache_gib = 0.0<br>bootstrap_after_err = 415<br>bootstrap_after_gpu = 33588<br>bootstrap_after_guard_min = 84482 |
| `forward-oracle` | `2026-11-23-forward-oracle-engine-emulation` | node-3 | `aeaaa4f94944` | pass | 105 passed, 0 failed | attn_core_rel_max_max = 0.003068<br>attn_core_rel_med_max = 0.000931<br>attn_core_rel_med_max_counting = 0.000704<br>attn_core_rel_med_max_sky = 0.000931 |
| `fp8-numerics` | `2026-11-23-fp8-numerics` | node-4 | `-` | fail | 5 passed, 3 failed | argmax_flips_vs_exact_A_of_139_rows = 0<br>argmax_flips_vs_exact_f8_row_of_139_rows = 7<br>hidden_added_mean_f8_row_chat_capital = 0.0841<br>hidden_added_mean_f8_row_chat_sky = 0.0653 |
| `fp8-potential` | `2026-11-23-fp8-potential` | node-4 | `ba571a9e007f` | fail | 3 passed, 0 failed | bound_gain_32k_roofline_pct_derived = 19.1<br>e4m3_vs_fp16_rel_l2_down = 0.0267<br>gate_a_required_gain_pct = 25<br>projected_gain_32k_gemm_only_pct_derived = 6.8 |
| `greedy-oracle` | `2026-11-23-greedy-vs-oracle` | node-4 | `3287e80d1636` | pass | 16 passed, 0 failed | fixtures = 1<br>min_oracle_margin = 0.1034<br>near_tie_positions_matched = 2<br>tokens_compared = 16 |
| `kernels-attn` | `2026-11-23-kernels-attn` | node-3 | `dbfa89a04f7a` | pass | 105 passed, 0 failed | mini_global_past_the_floor_vs_f64_worst_max_rel_error = 0.004543<br>mini_global_vs_f64_int8_ref_worst_max_rel_error = 0.006141<br>mini_global_vs_oracle_kv8_min_cosine = 0.9999911<br>mini_global_vs_oracle_kv8_worst_max_rel_error = 0.007812 |
| `kernels-dense` | `2026-11-23-kernels-dense` | node-4 | `867486ea4c35` | pass | 52 passed, 0 failed | controller_5_launches_us_1_row = 22.9<br>controller_blend_vs_oracle_median_case_of_rms = 3e-07<br>controller_blend_vs_oracle_worst_of_rms = 0.00105<br>controllers_checked_vs_oracle = 48 |
| `kernels-draft` | `2026-11-23-kernels-draft` | node-1 | `ca8af5148858` | pass | 128 passed, 0 failed | commit_8_rows_ms = 0.18<br>f_rnn_bit_equal_to_oracle = True<br>fold_device_bytes = 103359232<br>int8_vs_bf16_plane_free_drafts_agree = 35/35 |
| `kernels-moe` | `2026-11-22-kernels-moe` | node-1 | `b6a965455ebe` | pass | 50 passed, 0 failed | estimate.decode_token_moe_us_graph_rows1 = 8814.1<br>estimate.decode_token_moe_us_if_every_layer_were_layer0 = 7123.2<br>real_layer_0.rows1.eager_median_us = 172.9<br>real_layer_0.rows1.eager_min_us = 161.6 |
| `kernels-vision` | `2026-11-23-kernels-vision` | node-2 | `41814cc4e724` | pass | 49 passed, 0 failed | batch_of_four_equals_each_alone_bitwise = True<br>bias_values_not_exact_in_fp16 = 3<br>bias_values_total = 211968<br>encode_ms_la_150x1000 = 17 |
| `loop-probe` | `2026-11-23-final-loop-probe` | node-1 | `59d08a9c5e9f` | pass | 86 passed, 0 failed | greedy_identical_to_earlier_record = 34<br>greedy_prompts_compared = 34<br>loop_agentic_destructive_calls = 0<br>loop_agentic_prompts = 6 |
| `markov-variants` | `2026-11-23-markov-variants` | node-2 | `761b14388d12` | pass | 7 passed, 0 failed | accepted_code_a = 1266<br>accepted_code_c = 1266<br>accepted_reasoning_a = 1074<br>accepted_reasoning_c = 1074 |
| `memory` | `2026-11-23-final-memory` | node-1 | `59d08a9c5e9f` | pass | 8 passed, 0 failed | avail_ready_mib = 69301<br>bench_all_mem_min_gib = 67.6<br>floor_mib = 8192<br>free_after_gib = 101.27 |
| `memory-breakdown` | `2026-11-23-profile-overhead` | node-4 | `918e6854bb9d` | pass | 5 passed, 0 failed | decode_c1_graphs_off_tok_s = 41.5<br>decode_c1_graphs_on_tok_s = 53.4<br>graphs_overlap_mib = 174<br>kv_bytes_per_token = 16512 |
| `memory-plan` | `2026-11-23-memory-plan` | node-1 | `abc681365c70` | pass | 141 passed, 0 failed | after-bootstrap_gpu_process_mib = 39680<br>after-bootstrap_guard_min_mib = 66097<br>after-bootstrap_predicted_mib = 65905<br>after-bootstrap_prediction_error_mib = -342 |
| `moe-pipe-bounds` | `2026-11-23-prefill-moe-pipe-bounds` | node-3 | `4f939a689b8c` (dirty) | pass | - | best_case_layer_drop_pct = 30<br>gate_layer_drop_needed_pct = 15<br>layer0_ms_compute_only = 13.3<br>layer0_ms_epilogues_free = 13.43 |
| `needle` | `2026-11-23-final-needle` | node-1 | `59d08a9c5e9f` | pass | 2 passed, 0 failed | needles_found = 2<br>needles_total = 2<br>prefill_s_depth_0_1 = 191.788<br>prefill_s_depth_0_5 = 191.8 |
| `oracle-cacheddiff` | `2026-11-23-oracle-cacheddiff` | node-1 | `-` | pass | 199 passed, 0 failed | argmax_equal = True<br>differing_logits = 1<br>failing_prompt = chat-sky-effort0.9<br>fp32_ulps = 1.0 |
| `oracle-fixtures` | `2026-11-23-oracle-fixtures` | node-1 | `-` | pass | 5 passed, 0 failed | dspark_bytes = 298693516<br>dspark_fixtures = 5<br>dspark_seconds = 45<br>engine_bytes = 1319550544 |
| `oracle-gap` | `2026-11-23-oracle-gap` | node-1 | `7da8c6134484` | pass | 138 passed, 0 failed | argmax_equal_fraction = 0.9976284584980237<br>greedy_tokens_equal = 64<br>greedy_tokens_total = 64<br>hidden_rel_l2 = 0.259781 |
| `pin` | `2026-11-23-final-pin` | node-1 | `59d08a9c5e9f` | pass | 11 passed, 0 failed | pin_replies = 7<br>pin_replies_reproduced_at_pin = 0<br>reference_replies = 7<br>reference_vision_replies = 7 |
| `process-tests` | `2026-11-23-process-tests-linux` | node-4 | `59d08a9c5e9f` | pass | 14 passed, 0 failed | cap_proof_lines = 2<br>process_tests_failed = 0<br>process_tests_passed = 14 |
| `quality-tools` | `2026-11-23-final-quality-tools` | node-1 | `59d08a9c5e9f` | fail | 148 passed, 3 failed | loop_agentic_destructive_calls = 0<br>loop_agentic_prompts = 6<br>loop_reasoning_loops = 0<br>loop_reasoning_prompts = 12 |
| `review-fixes-a` | `2026-11-23-review-fixes-a` | node-1 | `87c39494e8ad` | pass | 182 passed, 0 failed | bootstrap_defaults_measured_ready_mib = 71557<br>bootstrap_defaults_process_mib = 46663<br>max_cacheauto_guard_min_mib = 12271<br>max_cacheauto_measured_ready_mib = 12452 |
| `review-fixes-b` | `2026-11-23-review-fixes-b` | node-2 | `2e684eb7109f` | pass | 547 passed, 0 failed | drafter_million_positions_commit_seconds = 5.0<br>drafter_million_positions_partials = 257<br>drafts_accepted_on_chat_prompt_16_rounds = 24<br>quiet_window_median_ms = 24.1 |
| `roofline` | `2026-11-23-roofline` | node-2 | `-` | pass | - | bf16_tflops_256x1152x2304_mean3 = 50.8<br>bf16_tflops_256x2304x1152_mean3 = 56.8<br>bf16_tflops_4096_mean3 = 89.1<br>bf16_tflops_4096x2304x3072_mean3 = 80.7 |
| `row-probe` | `2026-11-23-row-probe` | node-1 | `-` | pass | 104 passed, 0 failed | chat_gap_error_abs_median_capital = 0.135<br>chat_gap_error_abs_median_sky = 0.086<br>chat_gap_error_abs_p90_capital = 0.481<br>chat_gap_error_abs_p90_sky = 0.332 |
| `sampled-decode` | `2026-11-24-sampled-decode` | node-2 | `59d08a9c5e9f` | pass | 228 passed, 0 failed | dspark_c4_aggregate_tok_s = 122.558<br>dspark_c4_median_tok_s = 44.004<br>serial_c4_aggregate_tok_s = 83.209<br>serial_c4_median_tok_s = 29.85 |
| `skip-study` | `2026-11-23-skip-study` | node-2 | `-` | pass | 275 passed, 0 failed | bytes_dropped_one_row_d0p2_renorm = 0.099<br>bytes_dropped_one_row_d0p3_renorm = 0.2232<br>bytes_dropped_one_row_t0p15_renorm = 0.0722<br>bytes_dropped_one_row_t0p2_plain = 0.1128 |
| `smoke` | `2026-11-23-final-smoke` | node-1 | `59d08a9c5e9f` | pass | 6 passed, 0 failed | smoke_checks_passed = 6<br>smoke_checks_skipped = 0<br>think_off_reasoning_tokens = 105<br>think_on_reasoning_tokens = 188 |
| `spec` | `2026-11-23-final-spec` | node-1 | `59d08a9c5e9f` | pass | 11 passed, 0 failed | drafted_acceptance = 0.4765<br>drafted_accepted = 711<br>drafted_drafted = 1492<br>drafted_rounds = 432 |
| `spec-cache` | `2026-11-23-spec-cache` | node-2 | `761b14388d12` | pass | 59 passed, 0 failed | drafter_slot_bytes = 8458752<br>prefix_cache_reach_tokens = 393216<br>snapshot_slot_bytes_with_drafter = 179294720<br>specedge_cases = 50 |
| `spec-final` | `2026-11-23-spec-final` | node-2 | `e9eeea9683ed` | pass | 10 passed, 0 failed | c4_drafted_tok_s = 111.37<br>c4_serial_tok_s = 100.52<br>c4_speedup = 1.108<br>speedup_chat = 0.899 |
| `spec-images` | `2026-11-23-spec-images` | node-2 | `e9eeea9683ed` | pass | 569 passed, 0 failed | cache_images_hits = 11<br>cache_images_replies_compared = 28<br>cache_images_tokens_saved = 40689<br>gates_spec_pairs_equal = 9 |
| `spec-speed` | `2026-11-23-spec-speed` | node-2 | `761b14388d12` | pass | 16 passed, 0 failed | c4_drafted_tok_s = 110.48<br>c4_serial_tok_s = 99.92<br>breakeven_tokens_per_round_4k = 1.96<br>c4_speedup = 1.106 |
| `speed-baseline` | `2026-11-23-final-speed-baseline` | node-1 | `59d08a9c5e9f` | pass | 8 passed, 0 failed | decode_c1_tok_s = 91.574<br>decode_c4_tok_s = 112.977<br>decode_c8_tok_s = 135.764<br>decode_ctx128k_tok_s = 67.673 |
| `speed-c4` | `2026-11-23-final-speed-c4` | node-1 | `59d08a9c5e9f` | pass | 108 passed, 0 failed | decode_c4_rolling_tok_s = 142.329<br>decode_c4_steady_tok_s = 143.553<br>decode_c4_tok_s = 113.175<br>decode_c4_rolling_requests = 32 |
| `speed-c4-baseline` | `2026-11-23-c4-baseline` | node-4 | `1d53344d874b` | pass | 35 passed, 0 failed | decode_c4_rolling_tok_s = 131.502<br>decode_c4_steady_tok_s = 131.834<br>decode_c4_tok_s = 101.822<br>drafted_decode_c1_tok_s = 81.246 |
| `speed-c8-slots` | `2026-11-23-c8-slots` | node-4 | `1d53344d874b` | pass | 9 passed, 0 failed | drafted8_decode_c1_tok_s = 81.462<br>drafted8_decode_c4_rolling_tok_s = 131.656<br>drafted8_decode_c4_tok_s = 102.233<br>drafted8_decode_c8_steady_tok_s = 225.927 |
| `speed-decode` | `2026-11-23-final-speed-decode` | node-1 | `59d08a9c5e9f` | pass | 5 passed, 0 failed | decode_c1_4k_tok_s = 98.068<br>decode_c4_aggregate_tok_s = 113.175<br>decode_c8_aggregate_tok_s = 134.764<br>decode_code_tok_s = 129.754 |
| `speed-fuse-baseline` | `2026-11-23-fuse-baseline` | node-2 | `1d53344d874b` | pass | 7 passed, 0 failed | decode_c1_4k_tok_s = 51.55<br>decode_c1_short_tok_s = 53.64<br>decode_c2_short_tok_s = 75.71<br>decode_c4_short_tok_s = 100.733 |
| `speed-graph` | `2026-11-23-graph-decode` | node-2 | `9fa93ea6e8f1` | pass | 2 passed, 0 failed | baseline_bench_decode_c1_tok_s = 37.238<br>baseline_decode_c1_4k_tok_s = 36.158<br>baseline_decode_c1_short_tok_s = 37.232<br>baseline_decode_c2_short_tok_s = 56.246 |
| `speed-graph-baseline` | `2026-11-23-graph-baseline` | node-2 | `51f2993c1549` | pass | - | bench_decode_c1_tok_s = 37.238<br>bench_decode_c4_tok_s = 80.407<br>bench_decode_c8_tok_s = 83.623<br>decode_c1_4k_tok_s = 36.158 |
| `speed-prefill` | `2026-11-23-final-speed-prefill` | node-1 | `59d08a9c5e9f` | pass | 13 passed, 0 failed | decode_after_128k_tok_s = 100.216<br>decode_after_224k_tok_s = 76.603<br>decode_after_32k_tok_s = 87.578<br>decode_after_8k_tok_s = 106.633 |
| `speed-prefill-baseline` | `2026-11-23-prefill-baseline` | node-3 | `7da8c6134484` | pass | 3 passed, 0 failed | decode_after_128k_tok_s = 34.36<br>decode_after_224k_tok_s = 27.603<br>decode_after_32k_tok_s = 46.263<br>decode_after_8k_tok_s = 50.541 |
| `speed-round-baseline` | `2026-11-23-round-baseline` | node-1 | `1d53344d874b` | pass | 7 passed, 0 failed | decode_c4_aggregate_tok_s = 101.38<br>decode_code_tok_s = 122.61<br>decode_json_tok_s = 88.177<br>decode_mean4_tok_s = 81.632 |
| `speed-serial-chain-nogo` | `2026-11-23-serial-chain` | node-4 | `f658ec6e8073` | pass | 11 passed, 0 failed | serial_c1_ahead0_tok_s = 53.41<br>serial_c1_ahead1_tok_s = 53.32<br>serial_c4_ahead0_tok_s = 100.57<br>serial_c4_ahead1_tok_s = 100.17 |
| `tools-analysis` | `2026-11-23-tools-analysis` | node-4 | `918e6854bb9d` | pass | 27 passed, 4 failed | invalid_calls = 0<br>lookup_replay_pass_greedy = 2<br>lookup_replay_pass_t1 = 6<br>lookup_replay_total_t1 = 9 |
| `vision-smoke` | `2026-11-23-final-vision-smoke` | node-1 | `59d08a9c5e9f` | pass | 11 passed, 0 failed | gpu_process_mib_vision_on = 46690<br>image_prompt_tokens_max = 72<br>image_prompt_tokens_min = 70<br>image_replies_wall_s_max = 0.877 |
<!-- records-index:end -->

## 5. Limits to state up front (from our own records and docs)

- The pack is not validated by REXL: it serves only with `--allow-unvalidated-pack`; the maintainers' validation and quality gates are theirs.
- Byte-exactness is against our own oracle's numerics (`docs/ORACLE-GAP.md`, `docs/NUMERICS.md`), not against a reference GPU engine; the
  "A lane" is bit-identical to itself across chunk widths, graphs, drafting and the prefix cache, and that is what the exactness records show.
- Stretch targets were **not** met on the byte-exact lane (records and `docs/DECODE-MEAN4-PLAN.md`, `docs/C4-DECODE-PLAN.md`,
  `docs/PREFILL-SPEED.md`): cold 32K prefill over 3,000 tok/s, mean decode over 100 tok/s, c=4 aggregate over 200 tok/s. The fp8 prefill lane
  is a measured no-go (`bench/records/2026-11-23-fp8-*`); expert skipping failed the offline gate on every row and is not justified further
  (`bench/records/2026-11-23-skip-study`). Do not describe any of these as achieved.
- The tool-call gate of the final run is a recorded **fail** (`bench/records/2026-11-23-final-quality-tools/`): the 60 exact-answer questions are right, 25 of 28 tool schema cases pass (the three
  misses call `run_sql` before `create_order`; no invalid call and no unrequested destructive call), and in raw greedy text continuation 17 of 34 probe prompts loop (in reasoning mode none of 12).
  Sampled and reasoning chat decode is recorded once (`bench/records/2026-11-24-sampled-decode/`: drafted equals serial in 64 of 64 pairs, drafted 81 to 92 against about 53 tok/s serial on 16 prompts, four requests a cell); the all-chat c=4 mix is not re-measured.
- The chat template needed a server change (the Jinja `generation` tag, `crates/server/src/template.rs`): a generic server change, tested
  against Hugging Face's environment (`docs/` and the chat tests), not against the other families' templates beyond their existing tests.
- Open items in the plan's risk table: the fixed-cache memory prediction is optimistic by 1.35 GiB at the largest window with the drafter and
  the tower; a pack without the vision tower is refused; `SPEC=auto` is untested; the vision block's bias values are not exactly
  representable in fp16 for 3 of 211,968 values (worst absolute error 2.98e-8).
- Host-build facts: on a host stub build `minicpmv4_7_boot` does not link (`posix_fadvise` does not exist on macOS; identical at the base `02ef8cb`,
  not ours). `pdl_order`, added by the series, did not link either and now carries `#![cfg(not(gb_no_cuda))]`. gb-server's `process` test
  `the_server_caps_its_malloc_arenas_...` failed at `crates/server/tests/process.rs:425` on macOS because it reads `/proc/<pid>/environ` (the host is
  Darwin); the Linux run of the whole `process` test binary (node-4, at the pin) is the record `process-tests-linux` in section 4.
- The Mini's own GPU tests and goldens ran at the pin on one node (`engine-gpu-tests-pin`: nine test binaries, no failure); eight of those tests print "not run" for want of oracle fixtures that exist only in the lanes'
  labs, and the final run's `engine-test goldens` are the Small's golden files, so the Mini's kernels have no fixture-complete run at the pin.
- The GPU SM clock cap of 2,200 MHz (hardware protection, see `AGENTS.md`) has been enforced by the tools since `5c34e19`, and every record of the final run carries its proof line (`raw/cap-proof.txt`). Older
  records do not all: `python3 scripts/records_index.py --cap-audit` classifies every GPU-node record (at close: the proof line in 25, only a clock sample taken when the record was written in 51, neither in 22;
  no sample is above 2,200 MHz). Records without the proof line rest on that sample or on the lane's own report, not on a lock proof; a few early lab runs of some lanes had no proof either and were discarded,
  not recorded.

## 6. Decisions the owner has to make before filing

1. **Route.** The rules say a `[finding]` issue with a diff, filed by an invited tester; a family is a maintainer-opened release cut. The Mini
   pack is vcruz305's and CODEOWNERS are am423 and vcruz305: ask them first whether they want the series, in what form, and who files.
2. **Delivery of a large patch** (section 3): fork-branch link, attached patch set, or one finding per topic.
3. **Publishing.** Fork branches (`justincasey/REXL3`) are public once pushed; the CYBERFROST precedent pushed a head branch and a tag to the
   fork, this series has not (no push, no tag). Decide whether, when and what; the dev branch never.
4. **Rebase onto v0.1.3** (section 2) and the full gate rerun that follows.
5. **Goldens.** As built, none of the 25 weight-derived files joins the filed series; the alternative (`EXPORT_INCLUDE_GOLDENS=1`) puts 9.2 MB
   compressed into it, against CONTRIBUTING section 8. The other test data (hash goldens, fixtures) joins; the snapshot is a sealed,
   maintainer-synced tree, so the maintainers decide what they take.
6. **Scope.** The family as a whole, or the separable fixes first as their own findings (the stop-string fix and `return_token_ids`, one purpose
   each; the kernel block-order changes together with the Inkling kernel topics, asking that the Small's golden hashes be rerun in their chain).
7. **Licence lines** in `THIRD_PARTY_NOTICES.md` (a maintainer-owned file) and the wording of the scrubbed comments (146 comment and doc lines in
   66 files no longer name recipe paths).
8. **Topic granularity.** Topics follow the lanes; a one-commit topic stands alone. Merge neighbours if fewer, larger issues are wanted.
9. **Naming and description**: the family name `inkling2_loop`, the pack name, and the SAGE sentence in every text we write.
10. **Whether to file anything.** The series is local.

## 7. Filing checklist (when the owner says go)

1. `git fetch upstream`; rebase per section 2; `.export/check.sh` all PASS; `git diff --check`; address, host and project-name scans PASS.
2. Build and run the gates on a node at the rebased head (section 8); fill section 4 from the new records.
3. `gh issue list -R REXL3/REXL3 --state all --search "inkling"` and `"mini"`; read every hit; comment instead of duplicating.
4. Draft the four sections from the series doc and the records, as the template wants them, with the head SHA, commands and printed output
   (an exit code is not evidence).
5. Present the filing as a decision packet to the owner (action, target repo, head SHA, evidence, reversibility, the single command), and
   run exactly that after a yes.

## 8. Commands

```bash
# series (see docs/ENGINE_EXPORT.md for the exact script names)
cd ~/projects/rexl3-inkling2-loop-export && .export/check.sh                       # PASS lines per check
# host tests, engine worktree
REXL3_NO_CUDA=1 cargo +1.89.0 test -p gb-model --lib --locked inkling2             # Small 31 + Mini families::inkling2
# node work always through the recipe's tools (they apply the 2,200 MHz cap first)
./inkling2mini.sh build <rev>; ./inkling2mini.sh engine-test host; ./inkling2mini.sh engine-test goldens
./inkling2mini.sh start; ./inkling2mini.sh bench all; ./inkling2mini.sh stop
NODE=<addr> LAB_SRC=<engine worktree> PKG=gb-server scripts/ops_dev.sh test process ""   # the Linux run of the process tests
python3 scripts/check_record.py <gate> --need-engine --same-engine <exact gate>          # a record is a pass or it is not
# the pin and the series differ by test data only: this prints nothing (H = the pinned dev head)
cd ~/projects/rexl3-inkling2-loop-export && git diff --stat H engine/inkling2-export-work -- . $(sed 's|^|:(exclude)|' .export/excluded-paths.txt)
```
