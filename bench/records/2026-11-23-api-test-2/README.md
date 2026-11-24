# api-test-2: the OpenAI surface on the Mini after the three backend fixes (gate `api-test`)

Result **pass** (`meta.json`): `tests/api_conformance.py` **17 PASS, 0 FAIL** (`api-test.log`) and `bench/probe.py` all PASS incl. the new
`levels` check (`probe.log`, `probe.json`). Fresh lab build of the committed engine head `0ed77ffe9decda95f05b8c4c362655369eb00e6e`
(branch `inkling2-loop-backend`, clean) on node-4, `bootstrap` profile, `--spec none --prefix-cache off --no-vision`, **no**
`REXL3_ALLOW_UNGATED_COMPONENTS` (the family's `short_conv` and `forward` are built now), boot 16 s. `serve.log` has no `weights_differ`, no
`ungated_components`, no `level=error` or `level=fatal`. Supersedes `2026-11-23-api-test` (fail) by date.

## The three defects

1. **`errors`: `max_tokens` of 262,145 and 10,000,000 answered 200.** Not an engine defect. The server clamps `max_tokens` to the room left in the
   window for every family on purpose (`length_budget` in `crates/server/src/http.rs`; the test
   `a_huge_max_tokens_is_clamped_to_the_window_and_the_reply_ends_at_length` in `crates/server/tests/abuse.rs` requires HTTP 200 and a reply that
   ends at the window, for every family; that test is the contract and was not changed). The Mini's admission is correct: its limit is the
   window, not the 1,048,576-token pool, and it refuses a sequence of window + 1 tokens as `ContextTooLong` (host test
   `the_admission_limit_is_the_window_not_the_pool`, engine commit below). The claim that the Small answers 4xx has no record behind it
   (the Small's recipe has no such check) and the same generic code serves it. So the check is what changed: a `max_tokens` past the window
   passes on a 4xx *or* on a 200 whose prompt + completion fit the window and that ends `stop` or `length`. Printed: both requests 200
   clamped, 51 tokens, finish `stop` (a short prompt ends on its own, the model does not run to the window). **If the lead wants OpenAI's 4xx
   instead, that is a generic server change that moves every family and the abuse test, not a Mini fix.**
2. **`inkling2_loop_weights_differ planned=14552922948 loaded=14552899328`.** The 23,620 bytes are exact: 9,361 EXL3 groups x the 4-byte
   codebook marker the pack stores and the device does not keep (37,444, in the on-disk model `FIT.md` reproduces), less the f32 copies the
   loader makes of the embedding norm (4,608 beyond the bf16 the model counted) and of layer 0's attention norm (9,216). `engine::loaded_weight_bytes`
   now prices exactly that. Pinned by `the_plan_prices_what_the_loader_holds_to_the_byte` (the shipped widths: 14,552,899,328 and the 23,620
   difference) and by `the_real_packs_headers_price_to_the_loaders_total`, which reads the real pack's headers, binds them and prices them
   (run on node-4 through `scripts/ops_dev.sh test --lib`: 1 passed, 14,552,899,328). This boot's log has no `weights_differ`; the event now also
   prints `delta=` and names what to fix.
3. **The declared reasoning levels.** The table said all seven levels have `thinks: false`. By the descriptor's own definition ("the prompt
   opens a reasoning block at this level") that is correct and the server checks it against the template at boot (`verdict ok`); it is no
   statement about the model. What the table could not say, and what the probe, the smoke and the card care about, is what the model does.
   The Mini now has its own declaration (`mini_reasoning_capabilities`, selected by the pack's `config.json`; the Small's descriptor and its JSON
   are unchanged and tested so): each level carries the number the template renders (`effort`: none 0, minimal 0.1, low 0.2, medium 0.7,
   high 0.9, xhigh and max 0.99, read back from renders in a test) and `suppresses_reasoning: false` for every level.

## What `none` means for this model

`none` renders `Thinking effort level: 0` and the model still reasons: short thinking, not none (the model card says so, the float64 oracle
opens `<|content_thinking|>` on the same prompt, `fixtures/spec/chat-capital-effort0`). Measured here (`effort-table.txt`, `effort_probe.py`,
greedy, one request each): reasoning tokens at `none` were 39, 105, 30 and 378 on four prompts, against 39, 188, 45 and 405 at `high`. The
levels move the length only weakly and not monotonically (17*23 reasons 39 tokens at every level; the haiku 378 at none, 449 at minimal, 291
at low, 631 at medium, 405 at high, 723 at xhigh). So no client should expect `none` to switch reasoning off; a bounded `max_tokens` is the
only control, and a short answer needs room for the reasoning first (smoke's `chat` check, `max_tokens` 32, is cut in its reasoning:
`content ''`, finish `length`, and passes because it checks the reasoning separation, not an answer).
`bench/probe.py` `levels` now asserts the declaration (seven names, the template's numbers, no level suppresses reasoning) and that the
`chat` check's own reasoning_tokens at `none` agree (> 0): it printed `7 levels, none renders 0 and still reasons (reasoning_tokens 39 at none)`.
`bench/tools/smoke.py` `think-off` fails on an empty reasoning at `none` (a behaviour change to look at).

## Not proven

* One request per cell of the effort table: no repeat, no sampling at temperature above 0.
* The 17 checks cover the OpenAI surface, not the model's quality; the engine's forward is the oracle record's.
* The `levels` check reads the `/v1/models` declaration; it cannot prove the model reasons at `none` on every prompt (it proves four).
