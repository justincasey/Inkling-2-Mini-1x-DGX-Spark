# Engine export: the Inkling 2 REXL3 changes as a clean, rule-compliant series

Built 2026-11-23 from the local development branch (`inkling2-loop` @ `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` in the REXL3 clone) as 46 stacked `engine/<topic>` branches on
upstream's `02ef8cb` (`REXL3_UPSTREAM_BASE`), following the CYBERFROST precedent (`~/recipes/CYBERFROST-3.8-1x-DGX-Spark/docs/ENGINE_EXPORT.md`).
**Published on 2026-11-24 at the owner's instruction, and only this much:** the series head is the branch `inkling2-mini` of the owner's fork `justincasey/REXL3` (no tag, no topic branches) and is offered to REXL as a `[finding]` issue; the development branch `inkling2-loop` is not pushed and must never be (it holds the 25 withheld files). The public copy of the recipe pins the series head (its README, "The engine pin in this copy"). **Pin and evidence in this lab tree: the dev head H** (after the lane merges and the one scrub commit; `REXL3_COMMIT` and every node run are H, because the node tests read the excluded goldens and the ladder needs the pin to be an
ancestor of the dev head). **Filing: the series head** `fb5e3a7bbe590db3dc7ac039dd44fe751b92e3fc` (branch `engine/inkling2-export-work`, not `engine/export-work`: that is the CYBERFROST export worktree's branch). **They differ by test data only:** 25 files, 10.8 MB,
listed in `.export/excluded-paths.txt` (below). `inkling2mini.sh build REV` stages a commit with `git archive` from `REXL3_SRC`, no remote needed. The raw
development branch (36 lane merges, measurement builds and their reverts, subjects outside the rules) stays in `~/projects/REXL3` as `inkling2-loop`.

REXL3's rules want a tester to submit a complete tested `git diff` in a `[finding]` issue and never a pull request; the maintainers apply it in the private dev
repository, and a new model family reaches production only through a maintainer-opened release cut. The engine tree of the production repository is a sealed snapshot
(`CONTRIBUTING.md` section 2 and the Never list). This series is the clean form of the change for that route and the exact source of the final node run.

> **Never push the dev branch `inkling2-loop` publicly.** It holds 25 files (10.8 MB) derived from the pack's tensors (real expert tensor slices and values computed on real tensors); the
> sealed snapshot and the licence rules do not allow them. Push only this series (`engine/inkling2-export-work` and the topic branches), which excludes them: `check.sh` proves it.

## The series

Base: `02ef8cb`. Head: `engine/inkling2-loop-draft-slim` = `engine/inkling2-export-work` = `fb5e3a7bbe590db3dc7ac039dd44fe751b92e3fc`. Worktree: `~/projects/rexl3-inkling2-loop-export` (git, `.export/`, a host
`target/`). Source of truth for the content: `inkling2-loop` @ `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (199 non-merge commits, 44 lane merges). The commits are **unsigned** (the dev commits are signed with the
owner's key; a signature carries a timestamp, so signing here would make every rebuild a different set of hashes; sign when filing).

### Branches in stacking order (each contains all the previous ones)

| # | branch | commits | tip | what it does |
|--:|--------|--------:|-----|--------------|
| 1 | `engine/inkling2-family-host` | 2 | `054e4a25` | Inkling 2 Small family host half (probe, strict config parser, inventory, bind, per-rank memory model) and the `inkling2` chat format (pack template, self-framed replies, JSON tool calls). |
| 2 | `engine/inkling2-token-ids` | 2 | `3328dc4a` | Non-streamed `/v1/completions` return the generated token ids (`return_token_ids`). |
| 3 | `engine/inkling2-kernels` | 2 | `a9bc08c3` | Non-attention kernels of the Small (bf16 linear, sigmoid router, softcapped MoE, residual mix, short convolution, unembed top-k), row-invariant and checked against float64 references. |
| 4 | `engine/inkling2-attention` | 1 | `e501b1bd` | Attention of one TP rank (prologue, window/sink attention, int8 KV write), fixed absolute chunks so every execution mode gives the same bits. |
| 5 | `engine/inkling2-vision` | 2 | `49f17388` | Vision path (checkerboard-alpha preprocessing, 23-layer tower, merge projector, encoder, splice), bit-for-bit against Hugging Face's preprocessing. |
| 6 | `engine/inkling2-dspark-drafter` | 3 | `8e476e33` | DSpark drafter of one TP rank, its golden hashes, its int8 layer-2 plane and slot snapshots. |
| 7 | `engine/inkling2-rank-engine` | 4 | `7de442a5` | Four-rank engine (rank forward, lockstep group, int8 KV plan, split pack identity) and two group fixes (follower entry point, peer timeout). |
| 8 | `engine/inkling2-group-wiring` | 1 | `faceded9` | Drafter, prefix cache and vision splice wired through the four-rank group (`--spec dspark`). |
| 9 | `engine/inkling2-group-diagnostics` | 4 | `2fbc0987` | Plan memory-view logging, u64 call counts on the wire, `REXL3_INKLING2_DUMP_OPS` oracle dump. |
| 10 | `engine/inkling2-draft-budget` | 1 | `879abf7d` | Draft budget from a measured round-cost curve (`BudgetPolicy::Curve`). |
| 11 | `engine/inkling2-prefill-lanes` | 4 | `eba312d3` | Prefill all-reduce as reduce-scatter plus all-gather, multi-block MoE planner, two-lane prefill (comm stream under compute), logits-only dump. |
| 12 | `engine/inkling2-linear-moe-tiles` | 7 | `5e4629fc` | Thin-tile narrow linear, cp.async wide linear with grouped row blocks, A fragments one k step ahead in the 4-bit expert prefill; all bit-identical. |
| 13 | `engine/inkling2-decode-pdl-warm` | 5 | `00da1d0a` | PDL on the decode chains (default on) and L2 warming of `w_q` and the router gate during the all-reduces. |
| 14 | `engine/inkling2-moe-prefill-groups` | 3 | `a42d6cf3` | MoE prefill launches sized by fillable blocks and one fused list of expert groups. |
| 15 | `engine/inkling2-markov-narrow-kernels` | 3 | `1f61b9f1` | Drafter Markov walk kernels with wider loads and PDL, twelve-chunk loads in the narrow linear, new benches. |
| 16 | `engine/inkling2-launch-fusion` | 3 | `fb71645f` | K/V projections fused into the `w_q` launch, conv-state commits in three launches a round, PDL on the drafter chain. |
| 17 | `engine/inkling2-collectives` | 5 | `08449a5f` | bf16 layer partials, RoCE mesh decode collectives, bf16 reduce-scatter and all-gather, and the rank-consistent gather choice fix. |
| 18 | `engine/inkling2-loop-host` | 7 | `1f8dc81a` | Inkling 2 Mini family host half (config, 48-visit schedule, sizes from the config, one-rank memory plan, pack bind with per-tensor EXL3 width, registration). |
| 19 | `engine/inkling2-loop-dense-kernels` | 3 | `a892fedc` | Mini dense EXL3 projection and unembed epilogue, raw AttnRes mix at four queries, loop controllers, with oracle-golden tests. |
| 20 | `engine/inkling2-loop-attention-kernels` | 3 | `618227f3` | Attention kernels at the Mini's 24 query and 8 KV heads, int8 KV page header for any head count, attention battery at both shapes. |
| 21 | `engine/inkling2-loop-chat` | 3 | `d7071154` | `{% generation %}` tag in the chat template engine and the Mini's template goldens. |
| 22 | `engine/inkling2-loop-moe-kernels` | 4 | `eb9b8d3f` | mul1 softcap prefill gate/up kernel and MoE entries over width bands (mixed 4- and 5-bit experts), GPU tests and real-layer goldens. |
| 23 | `engine/inkling2-loop-backend-rank` | 4 | `da00cafe` | Serving backend, serve defaults, weight loader and 48-visit forward of the Mini rank, engine boot. |
| 24 | `engine/inkling2-loop-vision-tower` | 9 | `990fb5e7` | Mini vision tower (dense EXL3 projections, staged upload plus capturable launches, encoder, splice) with references, parity and real-pack tests. |
| 25 | `engine/inkling2-loop-drafter-kernels` | 8 | `eeabd55a` | Mini block drafter over EXL3 linears (Markov head folded into the unembed), geometry-parameterised kernels, host side and tests. |
| 26 | `engine/inkling2-loop-vision-wiring` | 3 | `212fbe52` | Mini image tower connected to the engine and prefill, preprocessor in the family, tower scratch priced from its measured allocation. |
| 27 | `engine/inkling2-loop-rank-gate` | 4 | `2b0a437d` | Rank oracle-dump, determinism, chunk and long-context gates; forward marked built; exact weight pricing and the Mini's reasoning levels. |
| 28 | `engine/inkling2-loop-prefix-cache` | 3 | `00a0e675` | Snapshot, restore and prefix cache of the Mini engine, with bit-for-bit restore tests. |
| 29 | `engine/inkling2-loop-decode-graphs` | 3 | `12bc4c2e` | Decode windows replayed as CUDA graphs, pinned equal to eager windows, graph counters through a mock rank. |
| 30 | `engine/inkling2-loop-decode-fuse` | 5 | `6b91bace` | Split top-k, MoE bands and K/V branches on forked streams, relative-bias kernel. |
| 31 | `engine/inkling2-loop-drafter-wiring` | 12 | `ddc54445` | Drafter taps, state and proposals wired into the rank; unfolded Markov head switch; draft budget policy knobs, learning switch and default; their tests and fragments. |
| 32 | `engine/inkling2-loop-stop-strings` | 3 | `ab68166d` | A reply ends at the token that completes a stop string inside a drafted round (server fix) and its tests. |
| 33 | `engine/inkling2-loop-cache-images` | 4 | `a580fdf1` | Prefix cache serves prompts with images; the integration commit that serves the block drafter with images too. |
| 34 | `engine/inkling2-loop-memory-plan` | 5 | `43efeae1` | Million-token timeout defaults, device-memory logging at each boot stage and while layers load, measured fixed overhead priced and a plan under the floor refused. |
| 35 | `engine/inkling2-loop-prefill-2tile` | 3 | `871d6f2a` | 4-bit MoE prefill GEMMs as 2-tile blocks, with prefill timing tests. |
| 36 | `engine/inkling2-loop-expert-arena` | 3 | `2b4cde4b` | Expert buffers carved from one arena; weights priced at the arena's unused block; plan readings repinned. |
| 37 | `engine/inkling2-loop-review-backend` | 6 | `08db84df` | Review fixes of the backend: prefix cache subtracted once, drafter default and explicit `--spec`, refusals before the weights load, reworded comments, replaced tests. |
| 38 | `engine/inkling2-loop-class-prior` | 1 | `23cf1dbe` | Learnt class prior fades in rounds that draft nothing so it can recover (engine crate). |
| 39 | `engine/inkling2-loop-review-rank` | 5 | `b3a16ec0` | Review fixes of the rank and kernels: real launch codes, image encode off the decode thread, drafter synced in flush, pinned lane joins. |
| 40 | `engine/inkling2-loop-chained-round` | 5 | `ef2a3303` | Round trace, drafted round's proposal and verify window chained on one stream, unchained A/B switch, served-window round costs, slow retry. |
| 41 | `engine/inkling2-loop-drafter-fuse` | 6 | `046fb841` | Drafter draft pick on the split top-k, staged attention partial, proposal graph per request count. |
| 42 | `engine/inkling2-loop-round-costs` | 5 | `5a894351` | Chained round with the per-count proposal graph, one-stream window cost line, proposal priced behind the graph. |
| 43 | `engine/inkling2-loop-prefill-chunks` | 6 | `075180aa` | 8,192-row prefill chunks, split MoE item order, 5-bit 2-tile blocks, `--prefill-chunk` cap of the Mini's family. |
| 44 | `engine/inkling2-loop-dense-pipeline` | 3 | `54cd7d39` | Pipelined dense EXL3 prefill projections of the Mini and the prefill attention projections through the dense pipe, with the pipe lab test. |
| 45 | `engine/inkling2-loop-copy-hybrid` | 2 | `605ef482` | Per-stream choice between a copy of the stream's own text and the block drafter in the Mini's plan round (net of a windows-ahead experiment that was added and removed, which is dropped). |
| 46 | `engine/inkling2-loop-draft-slim` | 5 | `fb5e3a7b` | Slimmer Markov pick and finer attention partials of the Mini drafter, its golden hashes as their own TEST commit, drafter plan pricing fix, comment scrub. |

Total: 185 commits: 181 of the 199 non-merge dev commits, kept in dev order within their topic (2 of them split in two: its test edit follows as its own `TEST:` commit), and 2 integration commits; 18 dev commits are left out
(measurement builds and their reverts, with the reason in `.export/drop.txt`).

## Tree identity: `.export/dev-scrub.patch`

**The series head's tree is not byte-identical to the dev head's tree, and the difference is stated, not silent.** `git diff 59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a fb5e3a7bbe590db3dc7ac039dd44fe751b92e3fc` is exactly two things, both proven by
`check.sh`:

1. `.export/dev-scrub.patch` (0 text files, a patch against the dev head): the address and name scrubs, the SAGE sentence, two whitespace lines, six fragment headings the
   release-notes assembler refuses, and the 21 fragments the export adds for the Small's topics. Apply it to the dev branch as its last commit (`git apply .export/dev-scrub.patch`, then commit)
   and the dev tree and the series tree differ only by item 2. `build.sh` rewrites the patch against each new dev head; when the dev branch carries it, `classify` recognises that commit by patch-id and
   puts it in `drop.txt`.
2. The 25 large goldens listed in `.export/excluded-paths.txt` (10.8 MB, next section), which exist only in the dev tree and **stay there** (the node tests read them; nothing removes them from dev). `check.sh` proves:
   *dev head + dev-scrub.patch - excluded-paths = the series head's tree*, tree for tree.

   **The one-liner that proves it at the end**, with H the pinned dev head (after `apply-dev-scrub.sh`):
   `git diff --stat H engine/inkling2-export-work -- . $(sed 's|^|:(exclude)|' .export/excluded-paths.txt)` prints nothing (run from the export worktree; it lists only the excluded goldens without the exclusion).

   The alternative is to let the goldens join: `EXPORT_INCLUDE_GOLDENS=1 .export/build.sh rebuild` (tested: all checks pass, dev head + dev-scrub.patch then equals the series tree with nothing removed). That
   adds 9.2 MB compressed to every clone and to the sealed snapshot, and is an owner decision (below).

## How the dev history became the series

The 199 non-merge commits were grouped into topics by path and by the lane they came from; the topic order follows the dev order wherever two commits touch the same lines, and only commits that
apply without a conflict were moved across topics.

1. **Lane merges dropped, history linearised.** 9 commits conflict when replayed in the topic order (the registration lists in `family.rs`, `inkling2/mod.rs`, `cuda/src/lib.rs`, the Mini's `engine.rs` and
   `mod.rs`, one README); each is resolved once (13 file resolutions: most are the content of the lane merge commits, the rest by hand), stored in `.export/resolutions/` and replayed by `build.sh`; a conflict that exists only because
   earlier commits were scrubbed is merged again from the scrubbed parent and commit versions (`scrub_merge`).
2. **Two integration commits** carry content that exists only in a lane merge (`e9eeea9`, whose message says the drafter with images is no longer refused):
   `FEAT: Served the Mini's block drafter with images` (in `inkling2-loop-cache-images`) and `FEAT: Logged the device memory after the Mini's drafter loads` (in `inkling2-loop-memory-plan`);
   patches and messages in `.export/patches/`.
3. **Measurement builds dropped.** `99fa7fb`/`aead824`, `8292bec`/`d66c395`, `226deca`/`f3bf927`, `8a6d225`/`5fbf68a`, `2c16af4`/`23eb6f3`, `ba8eb9a`/`f280307`, `c209012`/`c2809a4`: each LAB or PERF commit is
   undone by its revert, so the pair nets to nothing (the tree check proves it).
4. **Subjects.** 58 subjects were rewritten to `TYPE: Past-tense summary` of at most 90 characters (the long originals, which carry the measurements, stay as the first paragraph of the body);
   26 of them also had a TYPE outside FIX FEAT BUILD CI TEST BUMP DOCS (`PERF` and `DUMP` became `FEAT`, `DOC` became `DOCS`). 58 dev subjects were over 90 characters (the longest 567), including the eleven of the earlier scan.
5. **PROVENANCE.** 52 messages had no `PROVENANCE:` line (including `761b143`); each now ends with `PROVENANCE: REXL3-original; no outside code.`
6. **Identity and dates.** Every commit is authored and committed by the GitHub noreply identity with the dev author date as both dates; a rebuild gives the same hashes (run twice and compared).
7. **Scrubs, line for line** (table in `rules.SCRUB`): messages: `spark@<address>` and `spark-NN` host names of measurement lines became `GB10` (34 messages). Files: a lab rank address in two usage comments
   (`crates/cuda/tests/nccl_oneshot_quad.rs`, `roce_latency_pair.rs`), the Small's recipe repository name in one comment (`crates/model/src/family.rs`), a lab user's home directory in one usage line
   (`testdata/inkling2/vision/gen_vision_ref.py`), and the sentence that describes SAGE in `testdata/inkling2_loop/gpu/README.md`, which now reads exactly the sentence the repository allows. Recipe pointers: 0 comment and documentation lines in 0 files named files of the recipe repository (`docs/PORT-SPEC.md`, `docs/NUMERICS.md`, `docs/FIT.md`, `oracle/*.py`, `bench/records/...`, `bench/tools/...`,
   `scripts/ops_dev.sh`, `intake/...`); they now describe the thing ("the port spec", "the numerics notes", "a lab record", "the lab test script"), in comments and prose only (code lines are never touched, the text after `//` or `#` is); the
   bare word "recipe" in other senses (a layer recipe, a seeded input recipe) is left. Whitespace: the
   trailing space of `// gb-nvcc: --fmad=false ` in `inkling2_linear.cu` (commit `457d637`; the build splits that line on whitespace) and the blank line at the end of
   `changes/unreleased/inkling2-loop-family-host.md` (commit `964215b`).
8. **Fragments.** Every dev fragment is in the series (55); the series adds 21 for the Small's topics, which had none (`.export/fragments/`; each states only what its commits' messages and code say, and
   the numbers in them are copied from those messages, not measured again). 0 dev fragments used headings the release-notes assembler refuses (`Performance`, `Internal`, `Fixes`) and are mapped to
   its headings (`New features and improvements`, `Bug fixes`).

## Test data: what joins the series and what stays in the dev branch

The engine tree is a sealed snapshot in the production repository, and CONTRIBUTING section 8 says not to add model weights or anything derived from them: the Mini MoE layer goldens are slices of real
expert tensors and the other goldens are values computed on real tensors. So no large binary blob joins the series. **Excluded (25 files, 10.8 MB, about 9.2 MB compressed in git):**

| Path | Files | Size | Read by |
|---|--:|--:|---|
| `testdata/inkling2_loop/gpu/*.bin`, `*.safetensors` | 17 | 5.4 MB | `mini_dense_exl3_real_tensors_equal_the_oracle`, `mini_loop_controllers_equal_the_oracle_on_the_real_weights`, `mini_message_mix_equals_the_oracle_on_the_real_queries` (`crates/cuda/tests/inkling2_ops.rs`), `real_mini_layers_equal_the_oracle` (`inkling2_moe_mixed.rs`) |
| `testdata/inkling2_loop/vision/*.ref.f32` | 4 | 3.3 MB | the two real-pack tests of `crates/model/tests/inkling2_loop_vision.rs` |
| `testdata/inkling2/attn_oracle_*.bin`, `testdata/inkling2_loop/attn_oracle_*.bin` | 4 | 2.1 MB | every test of `crates/cuda/tests/inkling2_attn_oracle.rs` |

All of these tests are GPU and real-pack tests (none is in the host test set). They stay in the series as source and run from the dev branch; to run one against a series checkout, restore the goldens into
the worktree without committing them: `git restore --source=inkling2-loop --worktree -- testdata/inkling2_loop/gpu testdata/inkling2_loop/vision testdata/inkling2/attn_oracle_global.bin
testdata/inkling2/attn_oracle_local.bin testdata/inkling2_loop/attn_oracle_global.bin testdata/inkling2_loop/attn_oracle_local.bin`. Everything small joins: the vision PNGs of the Small (the only
binary files of the series; the preprocessing tests decode them and compare against `expected.json`), the chat goldens and templates, configs, tensor inventories, golden hash text files and the generator scripts.

## Shared-code touch list

Every file the series changes outside `crates/model/src/families/inkling2/**`, `crates/model/src/families/inkling2_loop/**`, the new tests, `testdata/` and
`changes/unreleased/`: the 40 files that exist at the base (`M`) and the new files outside the family directories (`A`). "Topic" is the earliest topic that
touches the file (later ones are listed when several do). "Other families" says whether the change can alter what a family other than Inkling 2 and the Mini does.
Evidence is what is in the tree or was run on the host: **no golden hash and no kernel was run on a node for this series**, so every "bit-identical" below is the
dev commit message's claim plus the tests named, not a measurement of this export.

**Changes that can alter other families (each is also in the decisions list):**

| # | where | what changes for other families | evidence |
|--:|---|---|---|
| 1 | `crates/server/src/http.rs`, topic `inkling2-loop-stop-strings` | the reply task now pushes a batch token by token: a stop string or a closed tool-call block that completes in the middle of a drafted or MTP batch ends the reply at that token. The text bytes are the same; `usage.completion_tokens` and the returned `token_ids` now count only up to that token (before: the whole batch). Every family, streamed or not | `crates/server/tests/stop_in_batch.rs` (host tests, real `Engine`, scripted backend, GLM's format); no node run |
| 2 | `crates/server/src/field_policy.rs`, `tests/request_budget.rs`, topic `inkling2-token-ids` | a non-streamed `/v1/completions` request with `return_token_ids` used to get a 400 and now gets a 200 with `choices[0].token_ids`; streamed requests are still refused. `request_budget.rs` was edited in the same commit to pin the new behaviour (CONTRIBUTING golden rule 7 puts a test edit like this in its own TEST change) | the edited test |
| 3 | `crates/cuda/csrc/exl3_big.cu`, topic `inkling2-loop-prefill-chunks` | the shared large-row EXL3 GEMM (`gb_exl3_linear_big`, also used by `mla_prefill.rs` and the GLM/DeepSeek paths) walks its blocks in another order (one row group's blocks run together, the layer's trellis is read from L2); the arithmetic is the same | `testdata/gpu/golden_exl3_big.txt` is untouched and was not rerun |
| 4 | `crates/cuda/csrc/moe_prefill.cu`, topics `inkling2-loop-moe-kernels`, `-linear-moe-tiles`, `-moe-prefill-groups`, `-loop-prefill-2tile`, `-loop-prefill-chunks` | the shared routed-expert prefill entries (`gb_moe_pf_gateup`, `_down`, `_plan_parallel`, `_combine_f32`) keep their signatures and call the old code with the new template parameters at their defaults (`ACT=0`, `MB=1`, `SPL=1`). Two instantiation changes reach any user of them: the A fragments of the 4-bit MCG experts are loaded one k step ahead, and 4- and 5-bit `mul1` experts run as 2-tile split blocks; both are claimed bit-identical | `four_bit_mcg_prefill_equals_the_decode_path` (GLM dimensions, `crates/cuda/tests/moe_prefill.rs`) and the unchanged `glm_shaped_prefill_equals_the_decode_path`; the fused plan is opt-in (`use_fused_groups`, default off) and `moe_plan_parallel_cpu.rs` keeps the legacy-plan bit-identity tests; no node run |
| 5 | `crates/engine/src/lib.rs`, `crates/spec/src/stats.rs`, topics `inkling2-loop-class-prior`, `-chained-round` | `SpecStats.class` (fading class counters) is filled for every family; the budget reads it only when learning is on (`REXL3_SPEC_LEARN=1`, off by default; MiniCPM-V's engine also passes `learn`), so other families' draft counts can change after a round that drafts nothing; replies never change (the verify window decides). `chained_round` and `with_idle_relax` default to `None`. `round_trace` marks run in every family's decode round (one relaxed atomic load when `REXL3_ROUND_TRACE` is unset) | `crates/engine/tests/class_prior.rs` and `chained_round.rs` (host); the unchanged `tests/budget.rs`; a unit test pins the class prior equal to `conditional_prior` on lifetime counters until a zero-draft round |
| 6 | `crates/spec/src/mode.rs`, `crates/server/src/main.rs`, `crates/model/src/serve.rs`, `glm/spec.rs`, `nemotron_h_omni/spec.rs`, `tools/switch_registry.json`, topic `inkling2-group-wiring` | `--spec dspark` is a new accepted value for every family (default stays `auto`); a family that does not list `dspark` in its plan's drafters refuses it by name in `refuse_unoffered_drafter`; the two other families' `match SpecMode` get the extra arm (`Dspark \| None => Choice::None`) | the arms are unreachable after the refusal; no test beyond the registry lists |
| 7 | `crates/model/src/serve.rs`, topic `inkling2-rank-engine` | the follower entry point accepts every rank above 0 of a group (it accepted only `tp.rank == want_rank`, so ranks 2 and 3 of a larger group were refused); the family's `max_tp_ranks` is still enforced first; rank 0 alone serves | a toy-family test of ranks 1 to 3 of 4 in the same file |
| 8 | `crates/model/src/serve.rs`, topic `inkling2-loop-prefill-chunks` | `check_options` takes the family's prefill-chunk cap: the Mini's family allows 8,192, every other family keeps the old cap; the family is probed first, so a probe error now precedes a chunk error | `a_family_cap_widens_the_prefill_chunk_check` (the old cap still rejects 8,192) |
| 9 | `crates/cuda/src/moe_prefill.rs`, topic `inkling2-loop-prefill-chunks` | `MAX_ROWS` 4,096 to 8,192 (the kernels' own limit; `moe_owned.rs` checks against it; GLM and Nemotron price their own scratch width in `glm/alloc.rs`, so they are not widened) | none beyond the above |
| 10 | `crates/cuda/csrc/pdl.cuh`, `tp_oneshot.cu`, topic `inkling2-decode-pdl-warm` | PDL mask bit 8 (`kInk2`) is new: `REXL3_GLM_PDL` values of 8 or more, which were masked off, now switch it on, and `gb_tp_sum_slots_f32` (the shared NCCL one-shot sum) launches through `launch_pdl`, so it takes PDL for every family under such a value. Default mask 0: unchanged | `pdl.cuh` says the griddepcontrol calls are no-ops without the attribute |
| 11 | `crates/cuda/src/graph.rs`, topic `inkling2-loop-decode-graphs` | `Capture::end` makes one more `cudaGraphGetNodes` call on every capture (error cleared, count 0) | none needed: the result of capture and launch is unchanged |
| 12 | `crates/vision/src/rgb_decode.rs`, `webp.rs`, topic `inkling2-vision` | new `AlphaPolicy::Checkerboard`; `apply()` takes `(x, y)`; the old `Drop` and `CompositeWhite` policies ignore the position and take the same arithmetic | the old policies' tests are kept; a unit test pins `webp::oriented` against `orient` |
| 13 | `crates/model/src/fingerprint.rs`, topic `inkling2-rank-engine` | a pack staged split across ranks (an index that lists shards missing from the folder) gets a new form (`rexl3-pack-fingerprint-split-v1`); a whole pack or one without an index keeps the old form byte for byte | `a_pack_staged_split_has_one_identity_on_every_rank` and the whole-pack form assertion |
| 14 | `crates/model/src/families/glm/inventory.rs`, topic `inkling2-family-host` | `PackInventory::from_table` knows the `mcg` marker (PARTS 5 to 6, the second loop walks PARTS); a table without `mcg` rows lays out as before; a table with `mcg` rows used to lose them silently | no `*.tsv` table of the base contains an `mcg` row (grep), so no committed inventory table changes |
| 15 | `crates/engine/src/lib.rs`, `crates/sched/src/lib.rs`, `crates/spec/src/stats.rs`, `crates/server/src/spec_metrics.rs`, topic `inkling2-loop-copy-hybrid` | the per-stream copy-or-block-drafter choice. Gates: `Engine.copy_hybrid` is `None` unless `with_copy_hybrid` is called, and only the Mini's engine builder calls it, and only with the `plan` budget and its block drafter loaded; `REXL3_INKLING2_LOOP_SPEC_COPY=off` turns it off and `_COPY_MATCH` sets the minimum match (default per the engine). `gb_sched::plan_round_free` is new; the old `plan_round` and `plan_round_charged` now call the shared `plan_with` with an empty `free` list, so every family's plan goes through changed code that must (and is claimed to) give the same plan: the tree's unit test `free_drafts_do_not_pay_the_draft_round` pins `plan_round_free` with nothing free equal to `plan_round_charged` to the bit. `SpecStats.copy` is filled and `/metrics` gains `rexl3_spec_copy_*` lines for every family (rows are emitted only for non-zero counts; whether the HELP and TYPE lines still appear for an empty family was not read) | the sched unit tests and `crates/engine/tests/copy_hybrid.rs` (host); none from a node |

**Registrations and additions that cannot change another family** (pure additions, defaults that keep the old path, or test-only):

| file | topic | what | note |
|---|---|---|---|
| `crates/model/src/family.rs` | family-host and eleven more | strings appended to `BUILT_COMPONENTS`, two ids added to test lists | `bf16_linear_bf16_in`, `router_sigmoid_bias_f32`, `moe_softcap_swiglu` are generic names; no base plan lists any of them as required (grep of the base) |
| `crates/model/src/families/mod.rs` | family-host | `Inkling2Family` and `Inkling2LoopFamily` appended to `FAMILIES` | last in the list, so an earlier family wins any probe collision; they claim `Inkling2ForConditionalGeneration` and `Inkling2LoopForConditionalGeneration` (+ `model_type`s) and refuse non-EXL3 packs |
| `crates/model/src/window_graph.rs` | inkling2-loop-decode-graphs | `WindowGraphs<G, K = GraphKey>` with a generic key and `from_env`; `new()` keeps its environment semantics | defaults keep every existing use compiling |
| `crates/server/src/chat_format.rs`, `reasoning.rs`, `reasoning_caps.rs` | family-host, rank-gate | `ChatFormat::reply_framing` (default `None`), `ReplySplit` trait, `ReasoningCount::reopening` (default `None`), optional `effort` fields skipped when `None` | other families' descriptors serialise as before |
| `crates/server/src/formats/mod.rs`, `tests/chat_format.rs`, `family_formats.rs`, `reasoning_capabilities.rs` | family-host, loop-host | registers the `inkling2` format last; test lists extended | |
| `crates/server/src/template.rs` | inkling2-loop-chat | `{% generation %}` / `{% endgeneration %}` lowered to `{% with %}` / `{% endwith %}` before compiling; comments, `{% raw %}` and strings are skipped; every other template takes the unchanged `Cow::Borrowed` path | the old engine refused any template with that tag, so none that rendered before can render differently; no template in the tree uses it besides the two Inkling ones; three new unit tests; `tests/template.rs` goldens untouched |
| `crates/server/src/main.rs` | rank-engine, group-wiring, backend-rank, rank-gate, memory-plan, review-backend | defaults and aliases under `if family.id() == "inkling2"` or `"inkling2_loop"` only (slots, model aliases, million-token timeouts, `peer_timeout`, `--spec` default `dspark` for the Mini, `REXL3_KV_DTYPE` set to `int8` if unset for these two) | no flag added to any other family |
| `crates/cuda/csrc/moe_experts.cu` | inkling2-kernels | `ACT` and `<WT,BF16OUT>` template parameters and two new entries; the old entries instantiate `<0>` and `<half,false>` with the same operation order | the existing decode goldens are untouched |
| `crates/cuda/src/moe.rs` | rank-engine, expert-arena | host arithmetic mirror of `default_plan` under `gb_no_cuda`; `from_groups` goes through `OwnAlloc` (same `DevBuf::from_slice`); `from_groups_arena` new | |
| `crates/cuda/src/nccl.rs`, `rt.rs`, `lib.rs`, `crates/vision/src/lib.rs`, `crates/engine/src/round_trace.rs` (A) | various | only additions (no deleted line in `nccl.rs`, `rt.rs`) | |
| `crates/cuda/src/moe_prefill.rs` (rest), `tests/moe_prefill.rs`, `tests/moe_plan_parallel_cpu.rs` | moe-prefill-groups, linear-moe-tiles | fused-plan FFI and `class_grid`, off by default; tests added | |
| New kernels and modules: `crates/cuda/csrc/pdl_probe.cu`, `roce_mesh.cu`, `roce_mesh_abi.h`, `roce_mesh_shim.cu`, `crates/cuda/src/roce_mesh.rs`, `crates/server/src/formats/inkling2.rs`, `crates/vision/src/inkling2.rs`, `tools/inkling2_attn_oracle.py`, `tools/inkling2_loop_attn_oracle.py` | various | new, used by Inkling 2 only (the RoCE mesh is used when `REXL3_INKLING2_MESH` is not 0, by the Small's group) | |

The Small's own evidence in the series: the 31 `families::inkling2::` host tests pass at every prefix from the one that adds them and at the head
(`.export/prefix-check-output.txt`); the Small's byte-identical golden hashes (`testdata/gpu/golden_inkling2_*.txt`) were **not** rerun on a node by this lane, and the shared
kernel changes in rows 3, 4 and 10 are the ones they would test.

## New flags, environment variables and test targets

**Command-line flags.** None is new. `--spec` takes a new value, `dspark` (`crates/spec/src/mode.rs`, `crates/server/src/main.rs`), parsed for every family; the default stays `auto`. A family
that does not list `dspark` among its plan's drafters refuses it by name. `apply_family_defaults` gains two blocks gated by the family id: for `inkling2` and `inkling2_loop` it sets
4 slots when none was asked, adds the model aliases (`inkling2`, `inkling-2-small`, `Inkling-2-Small-EXL3-4bpw`, `0xSero/Inkling-2-Small-EXL3-4bpw`, `default`; `inkling2-mini`, `inkling-2-mini`,
`Inkling-2-Mini-SAGE-REXL3-4.23bpw`, `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`, `default`), sets `REXL3_KV_DTYPE` to `int8` when unset, and above a 524,288-token window raises the request, prefill-budget and
decision timeouts (1,800 s) and the queue timeout (300 s) to 14,400 s when they are at their defaults; the Small raises the group's peer timeout to 14,400 s; the Mini's `--spec` defaults to `dspark`.

**New environment variables (44 names starting `REXL3_`; `git diff` of the series against the base, names absent from the base):**

| group | names | read by | default |
|---|---|---|---|
| engine (every family) | `REXL3_ROUND_TRACE` | `crates/engine/src/round_trace.rs` | off; a lab timeline of the host clock between a drafted round's phases, changes no launch or token |
| Small, serving | `REXL3_INKLING2_PART_BF16`, `_RS_BF16`, `_AG_BF16`, `_FUSE_KV`, `_COMMIT_LAYERS`, `_WARM_ROUTER`, `_MESH` | `families::inkling2`, `crates/cuda` | on; `0` turns each off (`_WARM_MB` below) |
| Small, tuning and checks | `REXL3_INKLING2_WARM_MB` (a size in MB), `_PART_CHECK` (a lab guard that traps on a partial that is not bf16-valued) | `families::inkling2::model` | 16; off |
| Small, dumps (oracle comparison) | `REXL3_INKLING2_DUMP`, `_DUMP_OPS`, `_DUMP_LOGITS_ONLY` | `families::inkling2::model` | off unless set |
| Mini, serving | `REXL3_INKLING2_LOOP_GRAPHS`, `_GRAPH_AT`, `_OVERLAP`, `_PRIORITY`, `_CHAIN` | `families::inkling2_loop` | graphs, overlap, priority, chain on (`0`, `off`, `false`, `no` turn off); `_GRAPH_AT=0` captures at first sight (default off) |
| Mini, speculation | `REXL3_INKLING2_LOOP_SPEC_TOKENS` (drafts a stream offers, 1 to 7), `_SPEC_POLICY` (`fixed`, `curve`, `plan`, `margin`, `scored`), `_SPEC_LEARN`, `_SPEC_RELAX` (a factor in (0, 1]) | `families::inkling2_loop::engine` | 7, `plan`, on, 0.99; a bad value is refused by name before the weights load |
| Mini, lab | `REXL3_INKLING2_LOOP_TIMING`, `_DUMP`, `_DUMP_OPS`, `_MARKOV` (the unfolded Markov head, variant `c`) | `families::inkling2_loop`, `crates/cuda` | off |
| tests only | `REXL3_INKLING2_DIR`, `_LOOP_DIR` (pack folders), `_ORACLE`, `_RANK`, `_DSPARK_ORACLE`, `_LOOP_DSPARK_ORACLE`, `_VISION_REF`, `REXL3_BENCH_UNFUSED`, `REXL3_MOE_BENCH`, `REXL3_PF_ITERS`, `_PF_LAYERS`, `_PF_ROWS`, `REXL3_QUAD_MESH`, `_QUAD_SIZES` | `crates/*/tests` | unset: the test reports "not run" |

Names the series reads that the base already had: `REXL3_GLM_PDL` (bit 8 is new, see the touch list), `REXL3_REQUIRE_GPU`, `REXL3_GOLDEN_OUT`, `REXL3_NCCL_*`, `REXL3_ROCE_*`, `REXL3_ALLOW_UNGATED_COMPONENTS` and `NCCL_MAX_NCHANNELS`.

**New test targets and what runs them** (a target that finds no device or pack returns as a pass, unless `REXL3_REQUIRE_GPU=1`, which makes it fail; the recipe's gate runs set it):

| crate | targets | needs |
|---|---|---|
| `crates/engine/tests` | `chained_round`, `class_prior` | host (mock backend) |
| `crates/server/tests` | `inkling2_format`, `inkling2_loop_chat`, `stop_in_batch` | host (committed `testdata`, mock tokenizer) |
| `crates/vision/tests` | `inkling2_preprocess`, `inkling2_loop_preprocess` | host (committed PNGs and `expected.json`) |
| `crates/model/src` (unit tests) | `families::inkling2::` (31), `families::inkling2_loop::` (115) | host |
| `crates/model/tests` | `inkling2_vision`, `inkling2_loop_graphs`, `_prefix`, `_rank`, `_spec`, `_spec_timing`, `_stall`, `_vision`, `_prefill_perf`, `_routelog` | a GPU and the Mini pack (`REXL3_INKLING2_LOOP_DIR`); `inkling2_vision` the Small's pack too; `_prefill_perf` and `_routelog` also an ids file; `_routelog` holds three host `#[test]`s |
| `crates/cuda/tests` | `inkling2_attn`, `inkling2_attn_oracle`, `inkling2_dspark`, `inkling2_loop_vision`, `inkling2_moe`, `inkling2_moe_mixed`, `inkling2_ops`, `pdl_order` | a GPU; `inkling2_dspark`, `_moe`, `_moe_mixed`, `_ops` also a pack; `_attn_oracle` and parts of `_ops`, `_moe_mixed` the excluded goldens |
| `crates/cuda/tests` | `nccl_oneshot_quad`, `roce_latency_pair` | a multi-process rig (4 or 2 ranks, `REXL3_NCCL_*`, `REXL3_ROCE_*`); skipped when the variables are unset |

## Scans (`check.sh` prints PASS or FAIL for each; the lists are regenerated at every run)

- **SAGE.** PASS: every whole-word `SAGE` in an added line or message is part of a pack id (`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`, in `crates/server/src/main.rs` (3 lines: the Mini's model aliases), the READMEs, `tensor_inventory.tsv`,
  `expected.json`, `fetch_vision_pack.py`, `gen_vision_ref.py`, and the PROVENANCE lines of some commits) or the one allowed sentence, which appears once, in `testdata/inkling2_loop/gpu/README.md` (it was
  worded differently in the dev branch and is scrubbed). No added line says how bits are chosen. `check.sh` prints every hit.
- **Internal and lab names.** PASS: none of the internal project names, lab host names, a lab address or a personal e-mail is in an added
  line or a message (`node-1` is used where a node is named). Not flagged by the scan, and kept: four comments name the recipe's lab script `scripts/ops_dev.sh`
  (`crates/cuda/tests/inkling2_loop_vision.rs:10`, `inkling2_ops.rs`, `crates/model/src/families/inkling2_loop/engine.rs`, `crates/model/tests/inkling2_loop_vision.rs:14`), and many comments point at the recipe's
  `docs/PORT-SPEC.md`, `docs/NUMERICS.md`, `oracle/*.py`, `bench/records/...`; the strings `glm53` in `crates/server/tests/family_formats.rs` and the `glm53_*` fixture names in `stop_in_batch.rs` are the
  base's own GLM family id and fixtures.
- **TensorFold** is named in the base itself as an MIT source (about 60 comment lines and `THIRD_PARTY_NOTICES.md`), so it is reported under the licence scan, not as an internal name (the brief listed it as one: say if that is wrong). At 4f939a6 the series adds no new mention of it (the dense-pipeline lane merged later does: `inkling2_dense_pipe.cu`).
- **Licence (report; AGPL-3.0-only, only MIT, Apache-2.0 and BSD-3-Clause code may be re-expressed).** PASS: no added line or message says code was copied or ported from vLLM, `transformers` or any other source;
  `.export/licence-scan.txt` lists the 239 lines that cite one. The sources: vLLM PR #67325 (`@1448919f`; the Inkling 2 `dspark.py`, `ops/dspark.py`, `speculator.py`, `ops/vision.py`, `processing.py`; and `loop.py @2b55f82b`),
  Hugging Face `transformers` (`@fc00b5b3`, `@be48e7a0`: semantics, and a runtime oracle in `gen_vision_ref.py` and the chat goldens), Pillow's `has_transparency_data` and RGB conversion (re-implemented), PyTorch (the
  oracle scripts only; one comment says the LayerNorm follows torch's fp32 statistics), NCCL (a linked library and a measured baseline), and `https://huggingface.co` (`fetch_vision_pack.py` downloads the pack's tensors at a
  pinned revision). The text says semantics read, rounding points, constants and numbers taken, with "our own fixed orders" and "REXL3-original". One more statement: `crates/cuda/src/inkling2/reference.rs` is
  "transcribed from the recipe's oracle", and that oracle (Apache-2.0 like the recipe) transcribes vLLM's code, also Apache-2.0. `THIRD_PARTY_NOTICES.md` at the base already has Pillow, `transformers` and vLLM
  notices in the form "behaviour re-expressed, no source included" (for other models and for the vLLM `dspark` speculator) and none for the Inkling 2 sources above: **a line would be needed** (report only; no
  upstream file was edited).

## Patch set

Size and checksums of the patch set at the series head (`.export/patches.sh` prints them again; a later check can prove the bytes against these):

```
series.diff:  285 files changed, 62348 insertions(+), 254 deletions(-)
series-text.diff:    65679 lines, 281 files, sha256 a7ad8f613f4304de1487a3c73b147c44ebf728d6b679cec3a3c947ca41d366c1
series.diff:    71770 lines, 285 files, sha256 5c3f087c8142d25721b2db3837c51bc9442fd6606abf456e697939fcb22781f0
patches:      185 mbox patches in series/, 46 topic folders, written to /tmp/engine-patches
```

Recipe (`.export/patches.sh [OUTDIR]` runs all of it and prints the sizes and checksums):

```sh
cd ~/projects/rexl3-inkling2-loop-export
out=${OUT:-/tmp/engine-patches}; rm -rf $out; prev=02ef8cba920e1a86b911c6e20c248c43724ddd66
for b in $(cat .export/branches.txt); do
  mkdir -p $out/$b
  git format-patch --no-signature -o $out/$b $prev..engine/$b      # one mbox patch per commit
  git diff --binary $prev engine/$b > $out/$b.diff                 # the topic as one diff against the branch before it
  prev=engine/$b
done
git format-patch --no-signature -o $out/series 02ef8cb..engine/inkling2-export-work   # global numbering, git am
git diff --binary 02ef8cb engine/inkling2-export-work > $out/series.diff              # git apply; one diff for a single finding
git diff 02ef8cb engine/inkling2-export-work -- . ':(exclude)*.png' > $out/series-text.diff   # plain text, without the 4 PNG fixtures
```

Verified on the host: `git am` of the 185 patches on `02ef8cb` gives tree `a6b4c684275569bdd2997d1defbe00ed13aecf90`, the series head's tree; `series.diff` applies on `02ef8cb` (and reverses); `series-text.diff` applies (the four binary PNG
fixtures of `testdata/inkling2/vision/` then have to be added separately, from the series). Per topic, apply the topic folders in the order of `branches.txt`.

## Checks

`.export/check.sh` re-runs every check (PASS or FAIL for each, exit 1 on any FAIL; output in `.export/check-output.txt`):

```
series head fb5e3a7bbe590db3dc7ac039dd44fe751b92e3fc (engine/inkling2-export-work); dev head 59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a; base 02ef8cba920e1a86b911c6e20c248c43724ddd66

branch table: # branch commits tip
   1 engine/inkling2-family-host 2 054e4a25
   2 engine/inkling2-token-ids 2 3328dc4a
   3 engine/inkling2-kernels 2 a9bc08c3
   4 engine/inkling2-attention 1 e501b1bd
   5 engine/inkling2-vision 2 49f17388
   6 engine/inkling2-dspark-drafter 3 8e476e33
   7 engine/inkling2-rank-engine 4 7de442a5
   8 engine/inkling2-group-wiring 1 faceded9
   9 engine/inkling2-group-diagnostics 4 2fbc0987
  10 engine/inkling2-draft-budget 1 879abf7d
  11 engine/inkling2-prefill-lanes 4 eba312d3
  12 engine/inkling2-linear-moe-tiles 7 5e4629fc
  13 engine/inkling2-decode-pdl-warm 5 00da1d0a
  14 engine/inkling2-moe-prefill-groups 3 a42d6cf3
  15 engine/inkling2-markov-narrow-kernels 3 1f61b9f1
  16 engine/inkling2-launch-fusion 3 fb71645f
  17 engine/inkling2-collectives 5 08449a5f
  18 engine/inkling2-loop-host 7 1f8dc81a
  19 engine/inkling2-loop-dense-kernels 3 a892fedc
  20 engine/inkling2-loop-attention-kernels 3 618227f3
  21 engine/inkling2-loop-chat 3 d7071154
  22 engine/inkling2-loop-moe-kernels 4 eb9b8d3f
  23 engine/inkling2-loop-backend-rank 4 da00cafe
  24 engine/inkling2-loop-vision-tower 9 990fb5e7
  25 engine/inkling2-loop-drafter-kernels 8 eeabd55a
  26 engine/inkling2-loop-vision-wiring 3 212fbe52
  27 engine/inkling2-loop-rank-gate 4 2b0a437d
  28 engine/inkling2-loop-prefix-cache 3 00a0e675
  29 engine/inkling2-loop-decode-graphs 3 12bc4c2e
  30 engine/inkling2-loop-decode-fuse 5 6b91bace
  31 engine/inkling2-loop-drafter-wiring 12 ddc54445
  32 engine/inkling2-loop-stop-strings 3 ab68166d
  33 engine/inkling2-loop-cache-images 4 a580fdf1
  34 engine/inkling2-loop-memory-plan 5 43efeae1
  35 engine/inkling2-loop-prefill-2tile 3 871d6f2a
  36 engine/inkling2-loop-expert-arena 3 2b4cde4b
  37 engine/inkling2-loop-review-backend 6 08db84df
  38 engine/inkling2-loop-class-prior 1 23cf1dbe
  39 engine/inkling2-loop-review-rank 5 b3a16ec0
  40 engine/inkling2-loop-chained-round 5 ef2a3303
  41 engine/inkling2-loop-drafter-fuse 6 046fb841
  42 engine/inkling2-loop-round-costs 5 5a894351
  43 engine/inkling2-loop-prefill-chunks 6 075180aa
  44 engine/inkling2-loop-dense-pipeline 3 54cd7d39
  45 engine/inkling2-loop-copy-hybrid 2 605ef482
  46 engine/inkling2-loop-draft-slim 5 fb5e3a7b
PASS branches stack in tips.txt order, each contains all previous
PASS head branch engine/inkling2-export-work equals the last topic tip
PASS no merge commits (the lane merges are linearised): 0
PASS 185 commits on the base, all inside the topic branches
PASS the series is based on the upstream base
PASS every one of the 199 non-merge dev commits is in the series or is a listed drop
  183 picked + 18 dropped (measurement builds and reverts) = 201; 2 synthetic integration commits
PASS excluded paths are only the dev branch's large goldens (25 files, 10.8 MB, absent from the series)
PASS the series tree equals the dev head's tree except fragments, the excluded goldens, the scrubs, the pdl_order gate and whitespace-only lines
  scrubs of the dev lines (lab addresses, recipe pointers, the SAGE sentence), line for line (0 files; the substitution table is rules.SCRUB):
  comment/whitespace-only differences (0 files):
  fragment differences: 0 files (A = only in the series, D = only in the dev branch, M = differs)
PASS every dev fragment is in the series
PASS the series adds no path the dev branch lacks apart from fragments
PASS git diff --check clean on every branch (each topic against the one before)
PASS git diff --check clean over the whole series
PASS every subject is `TYPE: Past-tense summary` of at most 90 characters (longest 89)
PASS every message has a PROVENANCE line
PASS author and committer are the GitHub noreply identity on every commit
PASS no commit carries a signature (an unsigned series rebuilds to the same hashes; sign when filing)
PASS no lab address or lab host name in any added line or message (node-1..node-4 are the allowed names)
PASS dev head + .export/dev-scrub.patch (0 files) - the 25 excluded goldens has exactly the series head's tree: tree a6b4c68427 vs a6b4c68427
PASS the word SAGE appears only in pack ids or in the one allowed sentence
    SAGE hit crates/server/src/main.rs: 3 (pack ids 3, allowed sentence 0)
    SAGE hit testdata/inkling2_loop/README.md: 2 (pack ids 2, allowed sentence 0)
    SAGE hit testdata/inkling2_loop/gpu/README.md: 2 (pack ids 1, allowed sentence 1)
    SAGE hit testdata/inkling2_loop/tensor_inventory.tsv: 1 (pack ids 1, allowed sentence 0)
    SAGE hit testdata/inkling2_loop/vision/expected.json: 1 (pack ids 1, allowed sentence 0)
    SAGE hit testdata/inkling2_loop/vision/fetch_vision_pack.py: 1 (pack ids 1, allowed sentence 0)
    SAGE hit testdata/inkling2_loop/vision/gen_vision_ref.py: 1 (pack ids 1, allowed sentence 0)
    SAGE hit commit 6f0f88ca: 1 (pack ids 1, allowed sentence 0)
    SAGE hit commit b7c1367a: 1 (pack ids 1, allowed sentence 0)
    SAGE hit commit 9c19df90: 1 (pack ids 1, allowed sentence 0)
    SAGE hit commit f210f569: 1 (pack ids 1, allowed sentence 0)
PASS no internal project or lab name (internal project names, lab host names, personal e-mail) in added lines and messages
PASS licence: no added line says code was copied or ported from vLLM, transformers or another source (239 citing lines in 71 places, listed in .export/licence-scan.txt)
PASS the series adds 55 changelog fragments, all well formed (name, headings, not empty)
PASS every topic with a FEAT or FIX commit adds or updates a changelog fragment
PASS the worktree HEAD is the series head

ALL PASS
```

`.export/prefix_check.sh` builds every prefix of the series on the host (detached checkout of each branch, in order; `REXL3_NO_CUDA=1 cargo check -p gb-model -p gb-cuda --lib --locked` and
`cargo test -p gb-model --lib --locked inkling2`):

```
1 inkling2-family-host check=ok tests: test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; 1408 filtered out; finished in 3.57s
2 inkling2-token-ids check=ok tests: test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; 1408 filtered out; finished in 3.61s
3 inkling2-kernels check=ok tests: test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; 1408 filtered out; finished in 3.63s
4 inkling2-attention check=ok tests: test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; 1408 filtered out; finished in 3.53s
5 inkling2-vision check=ok tests: test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 1408 filtered out; finished in 3.60s
6 inkling2-dspark-drafter check=ok tests: test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 1408 filtered out; finished in 3.61s
7 inkling2-rank-engine check=ok tests: test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.68s
8 inkling2-group-wiring check=ok tests: test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.60s
9 inkling2-group-diagnostics check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.53s
10 inkling2-draft-budget check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.54s
11 inkling2-prefill-lanes check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.53s
12 inkling2-linear-moe-tiles check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.51s
13 inkling2-decode-pdl-warm check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.50s
14 inkling2-moe-prefill-groups check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.49s
15 inkling2-markov-narrow-kernels check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.52s
16 inkling2-launch-fusion check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.60s
17 inkling2-collectives check=ok tests: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.56s
18 inkling2-loop-host check=ok tests: test result: ok. 64 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.56s
19 inkling2-loop-dense-kernels check=ok tests: test result: ok. 64 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.66s
20 inkling2-loop-attention-kernels check=ok tests: test result: ok. 64 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.63s
21 inkling2-loop-chat check=ok tests: test result: ok. 64 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.60s
22 inkling2-loop-moe-kernels check=ok tests: test result: ok. 64 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.61s
23 inkling2-loop-backend-rank check=ok tests: test result: ok. 76 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.65s
24 inkling2-loop-vision-tower check=ok tests: test result: ok. 80 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.93s
25 inkling2-loop-drafter-kernels check=ok tests: test result: ok. 83 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.50s
26 inkling2-loop-vision-wiring check=ok tests: test result: ok. 87 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.53s
27 inkling2-loop-rank-gate check=ok tests: test result: ok. 90 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.45s
28 inkling2-loop-prefix-cache check=ok tests: test result: ok. 104 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.48s
29 inkling2-loop-decode-graphs check=ok tests: test result: ok. 110 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.49s
30 inkling2-loop-decode-fuse check=ok tests: test result: ok. 110 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.44s
31 inkling2-loop-drafter-wiring check=ok tests: test result: ok. 122 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.61s
32 inkling2-loop-stop-strings check=ok tests: test result: ok. 122 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.60s
33 inkling2-loop-cache-images check=ok tests: test result: ok. 128 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.89s
34 inkling2-loop-memory-plan check=ok tests: test result: ok. 131 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.55s
35 inkling2-loop-prefill-2tile check=ok tests: test result: ok. 131 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.55s
36 inkling2-loop-expert-arena check=ok tests: test result: ok. 131 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.51s
37 inkling2-loop-review-backend check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.52s
38 inkling2-loop-class-prior check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.52s
39 inkling2-loop-review-rank check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.52s
40 inkling2-loop-chained-round check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.72s
41 inkling2-loop-drafter-fuse check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.53s
42 inkling2-loop-round-costs check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1409 filtered out; finished in 3.57s
43 inkling2-loop-prefill-chunks check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1410 filtered out; finished in 3.53s
44 inkling2-loop-dense-pipeline check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1410 filtered out; finished in 3.60s
45 inkling2-loop-copy-hybrid check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1410 filtered out; finished in 3.43s
46 inkling2-loop-draft-slim check=ok tests: test result: ok. 146 passed; 0 failed; 0 ignored; 0 measured; 1410 filtered out; finished in 3.45s
PREFIX CHECK 1..46: ALL PASS (branches 1-43 at dev head 4f939a6 in three worktrees, 44 after the append at 1ad011ff, 45-46 after the append at 59d08a9c)
```

Host verification at the series head (Mac, Rust 1.89.0, the two empty Nemotron fixture placeholders present and git-excluded):

| command (`REXL3_NO_CUDA=1`, `--locked`) | result |
|---|---|
| `cargo test -p gb-model --lib inkling2` | 146 passed, 0 failed (`families::inkling2::`: 31 passed, 0 failed; `families::inkling2_loop::`: 115 passed, 0 failed) |
| `cargo check -p gb-model -p gb-cuda --lib` | exit 0 |
| `cargo test -p gb-server --test inkling2_format` / `inkling2_loop_chat` / `stop_in_batch` | 18 passed, 0 failed / 19 passed, 0 failed / 6 passed, 0 failed |
| `cargo test -p gb-vision --test inkling2_preprocess` / `inkling2_loop_preprocess` | 2 passed, 0 failed / 3 passed, 0 failed |
| `cargo test -p gb-engine --test chained_round` / `class_prior` | 2 passed, 0 failed / 1 passed, 0 failed |

**Targets that do not link on a host stub build:** one, and it is the base's (`cargo build --workspace --tests --keep-going --locked`, Mac, at the base `02ef8cb`; at the first-pass head the series added a second, `pdl_order`, now fixed in the series).
- `gb-model` test `minicpmv4_7_boot`: **pre-existing at the base**, untouched by the series. `libc::posix_fadvise` and `libc::POSIX_FADV_DONTNEED` do not exist on macOS.
- `gb-cuda` test `pdl_order` (added by the series, topic `inkling2-decode-pdl-warm`) declared `extern "C"` `gb_pdl_probe_*` from `csrc/pdl_probe.cu`, which a stub build does not compile, and had no gate: `Undefined symbols`. The commit that adds it now carries
  `#![cfg(not(gb_no_cuda))]` like its neighbours (a one-line fixup in `.export/rules.py` `PREPEND`, the same line is in `dev-scrub.patch`, and `check.sh` proves the tree relation), and `cargo test -p gb-cuda --test pdl_order --no-run` builds on the host at the head. On a node (nvcc) the test is unchanged.

Not run (host only, no node): any GPU test, any golden hash, the `.cu` files (no nvcc on the Mac, so a kernel source error would not show), and the series built as a whole on a node. The node run of the final head
is the evidence for those.

## What shared changes can alter other families

For the owner and the maintainers. Most of the series is Inkling-only (new modules, new kernels, defaults gated by the family id). Three shared changes can change what another family does, and none of them was run on a node here:

1. **Stop strings in `crates/server/src/http.rs` (topic `inkling2-loop-stop-strings`).** The reply task now pushes a drafted or MTP batch token by token, so a stop string or a closed tool-call block that completes in the middle of a batch ends
   the reply at that token. The text is the same; `usage.completion_tokens` and the returned token ids now stop at that token for **every family**, streamed or not (before, the whole batch was counted and returned). Evidence: `crates/server/tests/stop_in_batch.rs`, host tests on GLM's format.
2. **`return_token_ids` on non-streamed `/v1/completions` (topic `inkling2-token-ids`).** A request that used to get a 400 now gets a 200 with `choices[0].token_ids`, for every family; streamed requests are still refused.
3. **Block order in `crates/cuda/csrc/exl3_big.cu` and `moe_prefill.cu`.** The shared large-row EXL3 GEMM (also used by the DeepSeek and GLM prefill paths) walks its blocks in another order, and the shared routed-expert prefill entries load the 4-bit MCG A fragments one step ahead and
   run 4- and 5-bit `mul1` experts as 2-tile blocks. The arithmetic and sums are claimed unchanged; this is timing and order, and the Small's golden hashes (`testdata/gpu/golden_*.txt`) on a node are what would show otherwise.

4. **The copy hybrid in `gb_sched`, `gb_engine`, `gb_spec` and `/metrics` (topic `inkling2-loop-copy-hybrid`).** Off for every other family (the engine option is `None`; only the Mini's builder sets it), but the shared planner `plan_with` was refactored to carry a `free` list, so other families' plans run through changed code (claimed identical, pinned by a unit test) and `/metrics` gains `rexl3_spec_copy_*` families.
5. **A deliberate numerics change of the DRAFTER, not of the model's outputs (topic `inkling2-loop-draft-slim`).** The drafter's attention is split into finer partials, which moves the last bf16 bits of `hd` (proposals only; every draft is verified by the target), so the four hash lines of `golden_inkling2_loop_dspark.txt` were regenerated in their own `TEST:` commit right after the feature.

Which to file separately as engine-wide `[finding]`s (the lead's view, which this lane shares): **the stop-string fix and `return_token_ids` are separable, one purpose each**, and each is small (a topic or two of this series), so file each on its own with its own
`git diff` (`git diff <branch before> engine/<topic>` of the topics above) and the host tests named. The block-order changes belong with the Inkling kernels, since the Mini and the Small use them first; the copy hybrid (row 15) goes with the Mini speculation topics; file them with the kernel topics and ask for the Small's golden hashes to be rerun. Everything else rides
with the Inkling 2 family finding(s).

## Decisions needed from the owner

1. **Which goldens join the filed series.** As built, none of the 25 large files joins; the dev branch (the pin) keeps them (decided), so never push `inkling2-loop`. The alternative, `EXPORT_INCLUDE_GOLDENS=1`, keeps them in the dev tree and puts 9.2 MB compressed in the series; CONTRIBUTING section 8 argues against the
   real-tensor ones.
2. **Applying `.export/dev-scrub.patch` to the dev branch** as one commit (`.export/apply-dev-scrub.sh DEV_WORKTREE`), after the last lane merge and before the pin, so the pin and the series head differ by test data only.
3. **Shared behaviour changes for other families** (rows 1 to 15 of the touch list): the stop-string fix of `http.rs` (usage counts and returned ids), the `return_token_ids` acceptance, the `exl3_big` and `moe_prefill`
   block orders. None was run on a node here; the Small's golden hashes are the evidence to run.
4. **`request_budget.rs`** is edited in its own `TEST:` commit right after the change it pins (the commit body says it replaces the one assertion that pinned the old refusal and weakens nothing else; between the two commits that assertion fails, so file them as one unit).
5. **`pdl_order.rs` gate.** The series adds `#![cfg(not(gb_no_cuda))]` to it (in the commit that adds the test; the same line is in `dev-scrub.patch`), so `cargo test --workspace` links on a stub build except for `minicpmv4_7_boot`, which is the base's.
6. **`THIRD_PARTY_NOTICES.md` needs a line** for vLLM PR #67325, `transformers` and the Hugging Face chat-template environment (above).
7. **Recipe pointers were scrubbed** (item 7 above); say if any wording should differ. The one-line `#![cfg(not(gb_no_cuda))]` gate on `pdl_order.rs` is a code change in the patch (item 5).
8. **Topic granularity.** Topics follow the lanes; a one-commit topic is a change that stands alone. Merge neighbours if fewer, larger issues are wanted.
9. **Whether to file anything.** The series is local.

## How to append (the three running lanes: decode proposal bytes, copy hybrid, dense pipeline)

The series is rebuilt or extended by `.export/build.sh` from `.export/commits.tsv` (one row per dev commit: sha, topic, subject, extras). The head branch must be checked out in the export worktree and the worktree clean.
`dev_head.txt` holds the dev head the series reflects; nothing reads a branch tip.

```sh
cd ~/projects/rexl3-inkling2-loop-export
NEW=<the new dev head>
.export/build.sh classify --topic <new-topic-name> --dev-head $NEW   # one row per new non-merge dev commit, in dev topological order
$EDITOR .export/commits.tsv    # fix each subject marked NEW SUBJECT NEEDED (TYPE: Past-tense summary, <= 90 chars); a `frag=slug` extra adds .export/fragments/slug.md
                               # a LAB or measurement commit and its revert go to .export/drop.txt instead; edit column 2 to split a topic or to put a row into an existing one
.export/build.sh append --dev-head $NEW    # replays the new rows on the head; rewrites branches.txt, tips.txt, progress.json, dev_head.txt, dev-scrub.patch
.export/check.sh                           # all PASS, or fix and rerun
```

`build.sh update --topic NAME --dev-head $NEW` does classify, append and check in one go when no subject needs editing. `classify` takes the dev commits of `base..$NEW` that are in neither `commits.tsv` nor `drop.txt`,
proposes the subject (`PERF` to `FEAT`, `DOC` to `DOCS`) or marks it, and recognises the dev-scrub commit by patch-id. `append` fails loudly instead of guessing: on a cherry-pick conflict it stops with the markers in the worktree;
resolve them, `git add` the files, run `.export/build.sh continue` (the resolution is stored in `.export/resolutions/`, per hunk, and replayed by every later rebuild). A user-visible commit (FEAT or FIX) needs a fragment in its
topic: either the dev commit adds one or the row names one. Merge commits are never picked: a lane merge that carries content of its own shows up as a difference in `check.sh`; fold it into a `patch:NAME` row
(`.export/patches/NAME.patch` and `.msg`, as the two integration commits are). `.export/build.sh rebuild` replays every row from the base and gives the same hashes. Tested on the host: the last topic was removed, the series rebuilt at
the earlier dev head (`4eec4b5`), then `classify`, a subject edit and `update` at `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` gave a head equal to this one, and the scrub-commit path was exercised on a scratch branch.
Appended commits become new branches on top of the head, so no existing hash changes; a rebuild after editing an earlier row changes every hash after it. The pin is the dev head H, not the series head; the series head is what is filed.
