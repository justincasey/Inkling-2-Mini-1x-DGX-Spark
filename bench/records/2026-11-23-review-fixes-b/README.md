# review-fixes-b: the findings in the rank, the streams, the kernels, the drafter and the spec policy (gate `review-fixes-b`)

Engine `2e684eb7109f11b6bccc2987de17c967663e3430` built and run on node-2 (lab build; branch `inkling2-loop-graph`, the integration head `6f227ee` plus
the fixes; the commits after it, `216e2e4` and the final head, change tests only), TP1, the shipping defaults for the live part (graphs, the
drafter, the prefix cache, the vision tower), GPU clock locked at 2,200 MHz. Every GPU run below sets `REXL3_REQUIRE_GPU=1`.

## One row a fix: the test, and what it did on the unfixed code

The unfixed runs are the same tests built over a copy of the tree with only the fix reverted (`raw/unfixed/new-tests-on-the-unfixed-code.log`).

| # | Fix (engine commit) | Test | Unfixed | Fixed |
|---|---|---|---|---|
| 1 | `flush` syncs the drafter's staging (`d3148c1`) | `inkling2_loop_spec::flush_releases_the_drafters_staging_when_nothing_proposes`: 40 windows with commits, a flush each, nothing proposes; the staged host buffers after every flush | FAILED: "the drafter's staging was not released by flush" | pass: 0 staged after every flush |
| 2 | the two-launch wrappers return the real launch code (`89a6a54`) | `inkling2_loop_graphs::a_capture_refused_inside_the_two_launch_wrappers_runs_eager`: a capture invalidated right before the prologue launch and before the split top-k (900-series codes), counting `capture_failures`, replies equal | FAILED: "a refused capture failed the window: CUDA error 719 (cudaErrorLaunchFailure)" | pass: code 901 reaches the cache as a refusal, 1 failure a shape, the replies equal the eager ones |
| 3 | `partials_for` bounds a sliding row's 5 chunks (`89a6a54`) | `a_graph_captured_early_replays_late_at_a_small_context`: `--max-model-len` 1,024 and 2,048, a 5-row window captured at p = 520 and replayed at every position of its key up to 995, byte equal to eager | FAILED at `--max-model-len` 1,024: "the graph window differs from the eager one (partials past the scratch?)" | pass: 188 replays at each context, equal |
| 4 | the tower's copies are stream-ordered, `LoopEncoder` non-blocking, the rank's streams at the greatest priority (`c05b801`) | `inkling2_loop_stall::decode_windows_keep_their_latency_during_an_image_encode`: 4 slots decoding while a 4,096-token image is encoded on another thread | measured: one window waited 2,861 ms of a 3,188 ms encode (blocking encoder stream); see below | pass: the slowest window 152 ms during a 3,671 ms encode |
| 5 | `set_overlap` forgets the graphs (`d3148c1`) | `changing_the_overlap_forgets_the_captured_graphs` | FAILED: "a graph captured with the lanes replayed after the lanes were switched off" | pass |
| 6 | commits that cannot be applied are `StateLost` (`d3148c1`) | `commits_that_cannot_be_applied_are_a_fatal_error` (an injected retryable failure, code 9, after the drafter was fed) | FAILED: the error was not fatal | pass |
| 7 | kernels: `block_sums` strides, `enqueue_proposal` refuses 0 / too many requests, the `ctx_kv` ring check, the split prefill smem, `Args` `offsetof` mirrors, the boot's prefill-width check (`89a6a54`, `d3148c1`) | `inkling2_ops::mix_equals_the_float64_reference...` at hidden 256 (9 sources, 3 queries: 36 totals in a block of 32 threads); `inkling2_dspark::mini_enqueue_proposal_refuses_a_batch_the_scratch_cannot_hold` (0, 5 and 9 requests at `max_reqs` 4); `device::a_band_the_prefill_kernel_lacks_is_refused_by_layer_and_width` (host); `args_layout_matches_the_kernel_struct` (every offset, host; the CUDA side `static_assert`s the same table and compiled) | mix at hidden 256: FAILED (uninitialised totals); the enqueue test was not run on the unfixed code (5 requests would write past the scratch) | pass |
| 8 | the learnt class prior fades in rounds that draft nothing (`98e5b2e`, SpecPolicyFix) | `gb-engine` `tests/class_prior.rs::code_traffic_gets_its_drafts_back_after_prose_has_collapsed_the_class_prior` (60 prose requests then 40 code, Plan/Margin/Scored): before the fix code stayed at 1.00 tokens a round (a fresh engine: 6.90), after it 6.90; live check below | see the engine commit | pass |
| 9 | tests | slow-side lane mutants `a_slow_lane_is_waited_for_and_a_dropped_join_is_seen` (K, V and MoE band lane: a lane that spins 2M cycles before its join gives the sequential bits; the join dropped gives other bits: all three differ); `a_capture_that_fails_after_the_fork_leaves_the_lanes_usable`; `inkling2_moe_mixed::an_arena_bank_equals_the_from_groups_bank`; Mini drafter at p = 4,095 / 4,096 / 4,097 / 5,000 in ONE launch of 4 requests on both planes (alone == reversed == repeated == batch; the partial counts 2 and 3), a million positions (257 partials, 256 commits of 4,096 rows over one buffer: 5.0 s), `mini_graph_replay_follows_restaged_positions`; `drafts_are_accepted_on_a_chat_prompt` (24 drafts accepted in 16 rounds); `LOOP_SPEC_replay_vs_prefill_context` asserted; the sky dump and logits hashes asserted in `forward_oracle_gate`; the graph gate asserts a replay per window shape; `prefix_snapshot_costs` and a derive-only test deleted | the lane test fails if a join is missing (the dropped-join arm); the rest are new coverage | pass |

## The stall of item 4, measured (`raw/stall-before-and-after.txt`, `raw/spec-mix/stall-final.txt`)

Four slots decode one row each (24 ms a window alone); another thread encodes a 4,096-token image on the tower (about 3.2 s alone). Windows started during the encode:

| Encoder stream and tower | Slowest window | Typical window | Encode |
|---|---:|---:|---:|
| blocking stream, legacy copies (before) | **2,861 ms** | 34 ms (5 windows in all) | 3,188 ms |
| non-blocking stream, legacy copies | 416 ms | 413 ms (10 windows) | 3,253 ms |
| final tower, rank streams at the default priority | 517 ms | 408 ms (10 windows) | 3,220 ms |
| **final tower, rank streams at the greatest priority (the default now)** | **152 ms** | **145 ms (30 windows)** | 3,671 ms |

The stall was real: with a blocking encoder stream one window waited for nearly the whole encode. Past it the windows are slowed by the device they share
with the tower (17x at the default priority, 6x with the rank's streams at the greatest priority, at the price of an encode 13% longer). One mistake on
the way is recorded in the file: a pageable async read-back enqueued before the host waited parked its thread inside `cudaMemcpyAsync` for the whole
encode and held up the engine thread's calls (2.9 s again); the tower syncs first.

## Nothing moved (live, shipping defaults: graphs + drafter + prefix cache + vision; `raw/live/`)

- `bench.py --ref` against the replies pinned on the starting engine `51f2993` (`2026-11-23-graph-exact`): `PASS pin`; `exactness.py` record `identical: 4
  replies`, replies sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d`; `exactness.py concurrent` pass.
- `bench spec` (`gates.py spec`): `PASS spec`; `specedge.py`: `PASS: 65 cases, 0 failures, 0 warnings`; `bench cache`: `PASS cache`; `bench vision`: `PASS
  vision-record` (there is no pinned `reference-vision.json` in this tree: the replies were recorded, not compared).
- Rank gates (`raw/gpu-tests-2e684eb.log`), graphs on and `REXL3_INKLING2_LOOP_GRAPHS=0`: the sky dump sha256 `c25f9876...c45c0` and the chunk-width logits
  `a105f245...f41a` (now asserted by the test), `long_context_gate` equal; the 37 graph windows' logits hash `6d0a555b...c06b`; prefix tests 7 passed, vision 2
  (model) and 4 (cuda), `inkling2_attn_oracle` 5, spec tests 9 passed.
- Kernel batteries (full binaries): `inkling2_ops` 29 passed, `inkling2_moe` 8, `inkling2_moe_mixed` 13 (with the arena test), `inkling2_attn` `mini::` 14 and
  `small::` 14, `inkling2_dspark` 28 (the Mini's, with the new ones, and the Small's that need no pack; the Small's real-weight tests need the Small's pack
  and skip here as before), `inkling2_attn_oracle` 5. Host: the Small's 31, `families::inkling2_loop::` 100, `window_graph::` 6, `gb-spec` 77, `gb-sched` 97,
  `gb-engine` 202.

## Item 8 live (`raw/spec-mix/`)

`specmix.py` on a freshly started server: 12 prose requests (1.84 tokens a round, 53 tok/s: the class never fell to zero drafts on this pack's prose) then 12
code requests: **5.02 to 5.12 tokens a round and 103 tok/s**, against 5.447 on a second fresh server that served code only (92% to 94%; the tool's verdict `PASS
drafting follows the traffic`). The collapse itself, and the recovery from it, are shown by the engine's simulated trace test (code at 1.00 tokens a round
before the fix, 6.90 after); this pack's prose did not collapse the live class, so the live run shows that the fade leaves a healthy class alone.

## Notes for the lead

- `scripts/ops_dev.sh` callers should pass `REXL3_REQUIRE_GPU=1` in `TEST_ENV` (it is not edited here); the Small's real-weight tests (`REXL3_INKLING2_DIR`) are
  deliberately outside it.
- Not run: the unfixed counterpart of the split top-k's refusal (the prologue's failed first and stopped the test; both wrappers have the same change),
  the enqueue refusal on the unfixed code (it would write past the scratch), memory at ready.
