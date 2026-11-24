# review-fixes-a: the engine's control-path findings of the three reviews (gate `review-fixes-a`)

Result **pass** (`meta.json`). Engine commits on branch `inkling2-loop-backend`, head `87c39494e8adf195468473793a8d0ffa4fdd97a7` (clean):
the fixes in order: (1) `56df1dc`, (2) `3572a7a` the server's spec default, (3, 4, 5) `1ce625b`
config constants, env knobs, `--draft`, loader-bytes error, unembed columns and top-k partials priced, (6) `13e32a1`, (7) `964215b` docs, (8) `3a78e44` and `87c3949` tests. Node-1, lab build of the head, boots through the controller with its guard; GPU gate through `scripts/ops_dev.sh`.

| # | Fix | Test that fails before and passes after |
|---|---|---|
| 1 | an explicit `--prefix-cache-mib N` was subtracted from the memory twice in `predicted_ready_available_gib` and the floor check (`refuse_below_floor`, now removed: after the fix it could not fire for any request, the planner already refuses a box that cannot hold the floor, the reserve and the pool) | `a_fixed_cache_at_max_with_the_drafter_boots_and_is_not_subtracted_twice` (4 x 1,048,576, drafter, `Mib(12288)`: 16,388 pages, cache 12.00 GiB, prediction = floor + reserve + vision + spare; it fails with the old subtraction put back: 10,737,418,240 against 13,826,230,356), `a_fixed_cache_on_a_box_one_gib_too_small_is_refused_by_name` (the smallest box that holds the cache boots, one GiB less is refused by the planner), `an_auto_cache_leaves_the_floor_and_the_variable_part` |
| 2 | `--spec auto` was coerced to `none`; the default is now the block drafter, and an explicit `--spec auto|copy|dspark|none` or `--no-drafts` reaches the engine untouched (`spec_given` tracks the flag) | `the_minis_spec_default_is_the_block_drafter_and_every_explicit_mode_passes_through`: eight command lines through `parse_from`, `apply_family_defaults` and `serve_options` |
| 3 | the kernels' hard-coded constants and the target geometry are checked: 20 keys (heads, KV heads, head width, `d_rel`, `rel_extent`, hidden, experts, top-k, expert width, both vocabulary sizes, log scaling floor and alpha, both clips, the three softcaps, the route scale, `eos_token_id`), refused by name with both values, before any weight is read | `check_kernel_constants_accepts_the_shipped_config_and_refuses_each_value_changed_alone` (each key alone, the message pinned), `the_pack_is_checked_against_the_kernels_before_the_load`; `dims`' 32-head test is renamed: parsing a config is arithmetic, accepting it is this check |
| 4 | the spec env knobs and the prefix config are validated in `refuse_unsupported` (before the load); `REXL3_INKLING2_LOOP_SPEC_LEARN` garbage is refused; the tower's `check_config` and the drafter's geometry run before the load; `--draft DIR` is refused by name | `the_spec_environment_knobs_are_validated_strictly_before_the_load`, `a_draft_directory_is_refused_by_name` |
| 5 | a loader that holds more than the plan priced + the arena's block is a boot error, not a warning; `logits` priced at the 201,088 columns the rank allocates, `topk_partials` priced | `memory_the_plan_prices_exactly_the_listed_buffers` (the named items, `64 x 98 x 64 x 8` partials, `201,088`), the three-profile prediction test repinned (-3 MiB, the two new items) |
| 6 | the tower-less refusal no longer advises `--no-vision` (the bind claims `model.visual.*` unconditionally, so a pack without the tower is refused there in any case): the advice and the changelog claim are dropped, not the bind changed | `a_pack_missing_either_tower_anchor_refuses_images_by_name` (one key at a time) |
| 7 | stale comments and fragments reworded (`backend.rs`, `mod.rs`, `draft.rs`, `memory.rs`, `dims.rs`, the four fragments); `dims::device_constant_gaps` and its test deleted; `inkling2_attn.rs` is cuda: left to GraphLane | |
| 8 | tests that pinned nothing deleted or rewritten, the gap tests added: window limits, commit validation, wrong row count, a fatal step, two recoverable errors, `FailRestoreLate`, a prefill failing after the slot began, the drafter plan (18,576 B/token), the prefill and window clamps, a pool below one context, admission at 1,048,576 with 16,388 pages, `stop_in_batch` at the third token of a round with `reasoning_tokens` | listed in `meta.json` |

## Nothing moved

* `exactness.py` on a boot with the shipping defaults (`--spec dspark`, cache on, vision on): `identical: 4 replies` against the `graph-exact` pin,
  replies sha256 (sorted keys, compact separators) `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d`; `concurrent --max-tokens 64` `"pass": true`.
* `forward_oracle_gate`: dump sha256 `c25f9876...c45c0` twice, chunk widths 7 and 11 logits `a105f245...f41a` (`raw/oracle-gate.txt`).
* Host: `families::inkling2` 146 passed with the Small's 31 unchanged; `gb-server` bin 12, `stop_in_batch` 6, `inkling2_loop_chat` 19, `inkling2_format` 18.

## Boots (`runs/`)

| Boot | Plan predicts at ready | Measured at ready | Guard minimum | Process |
|---|---:|---:|---:|---:|
| bootstrap, defaults (dspark, cache on, vision on) | not read (the events line is cut) | 71,557 MiB | 71,557 MiB | 46,663 MiB |
| max, `--prefix-cache-mib 12288` (drafter, vision on) | 12.08 GiB | 11,021 MiB | 10,856 MiB | 106,220 MiB |
| max, `--prefix-cache auto` | 12.28 GiB | 12,452 MiB | 12,271 MiB | 104,991 MiB |

The `max` + `--prefix-cache-mib 12288` boot, which the double subtraction refused, now boots: 16,388 pages, cache 12.00 GiB, drafter and tower loaded.

## Not proven or not done

* **The fixed-cache prediction is 1.35 GiB optimistic at this configuration** (12.08 GiB against 11,021 MiB; the cache-auto boot is within 0.12 GiB). Both boots take the same stages (the
  `mem_stage` lines are in `runs/`); the fixed one used about 1.2 GiB more, the size of the cache difference between the two. The cause was not found; it is above the 1 GiB the plan test pins
  for the measured profiles, and no profile with the drafter, the tower and a cache has a measured pin yet.
* Not done: the `config.rs`/`draft.rs`/`bind.rs` weak-guard tests of ReviewLogic's list beyond the new ones (only the kernel-constants and the tower tests were rewritten), `bind.rs` unclaimed-group cases,
  the `REXL3_REQUIRE_GPU` helper (the GPU tests each define their own `pack_dir()`: there is no shared helper to extend, and those files are GraphLane's), the `vision.rs:247` one-liner (given to GraphLane:
  it is only safe with their cuda change).
* The tower-less pack is refused, not served (the bind was not made optional).
