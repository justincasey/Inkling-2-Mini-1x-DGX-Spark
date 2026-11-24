# Recipe decisions

One bullet per decision: what, why, evidence, risk. "User" marks a decision the owner made.

## Model and pack
- **Pack:** `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` @ `44c78c9d229b43dd45e41506071b9f5f939bed44` (user). Per-tensor K
  read from the trellis shapes: routed experts 4-bit (7,944 tensors) and 5-bit (1,272), attention / unembed / vision
  6-bit, drafter 4-bit, embedding BF16; mean K 4.238 (`docs/ROOFLINE.md`). Publisher's figures, unverified here: top-1
  agreement 98.33%, mean KLD 0.0088 against the source. SAGE is unpublished.
- **Config/overlay:** `config.json` says only `quant_method: exl3`. Nothing to overlay yet: K is per tensor, so a loader
  must read it from each trellis (the Small family does not).
- **Draft:** the pack's `mtp.safetensors` (3 layers, block 8, Markov RNN head, taps `[5, 23, 41, 48]` over the 48 passes):
  7 drafted tokens a round. **Decided 2026-11-23 (plan 4.6; records `2026-11-23-spec-speed`, `-spec-exact`, `-markov-variants`, `-spec-cache`):
  ship `SPEC=dspark` as the default, with the engine's `plan` budget and learnt priors; `none` and `copy` stay selectable.** The arithmetic
  (node-2, GPU locked at 2,200 MHz, greedy, median of 3): a serial token is 19.15 ms (52.2 tok/s); a drafted round is a verify window of 8
  rows (29.4 ms at a 4K context, 1.54 x one row: 8 consecutive tokens touch 28-33 distinct experts a visit against 8 for one, the route
  log, but the step is not expert-read bound) plus the proposal (8.3 ms) plus the drafter's commit, 45 ms measured end to end for reasoning
  (3.26 tokens a round at 72.3 tok/s), so a stream needs about 2.4 tokens a round to break even (2.0 from the rank's own times). Measured
  at one stream with the shipped budget: code 2.24 x, tool calls 2.37 x, structured JSON 1.65 x, a 6K-token question 1.56 x, reasoning 1.40 x
  (the model reasons first: most of its tokens), prose 0.90 x (0.98 x on a server that has only seen prose: acceptance 0.14-0.21 is below
  break-even); the median of reasoning, chat and code is 1.40 x against a noise of 0.01 (repeat runs); at four streams the aggregate is
  110.5 against 99.9 tok/s (1.106 x). The rule (win at c=1 on the median of the kinds that matter by more than the noise, no loss at c=4)
  is met. Depth: fixed 7 / 4 / 2 drafts give reasoning 1.39 / 1.43 / 1.31 and code 2.23 / 1.95 / 1.52, c=4 aggregate 1.04 / 1.13 / 1.13;
  the learnt `plan` budget takes depth 7 where acceptance is high and shortens or stops elsewhere (code 2.24, reasoning 1.43, c=4 1.105 on
  a fresh server), so no fixed depth is shipped. Prose is the one kind that loses (0.90 x with other traffic before it): a chat-only
  deployment should set `SPEC=none`. Variant (c) of the Markov head accepts exactly the same drafts as the shipped fold (a) and costs 7.4 ms
  a round: keep (a) (UNKNOWN-17). The drafter serves image prompts and runs with the prefix cache over images (records `spec-images`,
  `spec-final`): an image's rows are ordinary rows of the forward, their taps feed the drafter's context like a text row's, and a snapshot
  after an image carries the drafter's state; the shipping defaults are `SPEC=dspark PREFIX_CACHE=on VISION=on`. On the merged engine with
  all three on (bootstrap profile) the plan prices the drafter inside its stages and predicts 62.61 GiB of `MemAvailable` at ready against 63.57
  GiB measured (+0.96 GiB; the cache is the larger share of that gap: +0.37 without it).
  **Erratum 2026-11-23 (lead, from `docs/C4-DECODE-PLAN.md` sections 2 and 3, `bench/tools/c4_model.py`; the decision above stands, these figures do not):**
  the 8-row window "29.4 ms" and "7 ms of a round (11 ms at four streams) outside the rank's timers" were read against `timing.txt` of the rank's timing
  test, whose ids are a synthetic cycled list: its windows touch fewer distinct experts than real text (19.6 against 29.4 a visit for 1 stream x 8
  rows), so it understates real windows. A real 8-row window is about 38 ms, the real host residual of a drafted round is 2.8 ms (c=1) to 7.1 ms (c=4),
  and most of the former "gap" is expert bytes. The engine's `round_cost` grid (`WINDOW_MS`, built from the same table) prices 4 streams x 8 rows at 53 ms
  against about 89-90 ms real, so the learnt depth allotment over-drafts at four streams (an all-chat c=4 mix loses about 9% against serial). The ship/no-ship rule
  is unchanged (it was met on served measurements); the grid is to be rebuilt from real-id windows (plan 7.4, experiment E1).
- **Provenance:** the card's base revision `6e7c8248...` is not served by the base repo. `chat_template.jinja` and both
  tokenizer files equal base `main` `b85c3cfa` byte for byte; `config.json` differs in `dtype` and the quant block.

## Topology and engine
- **Topology:** TP1, one Spark (user). No fabric, no collectives.
- **Engine:** REXL3 with a new `inkling2_loop` family (user: build it in REXL3, start the working branch at the latest REXL3
  commit). Branch `inkling2-loop` and worktree `~/projects/rexl3-inkling2-loop` start at upstream `main` `02ef8cb` (fetched
  2026-11-22), which has no Inkling family; the Small family (`inkling2-a` @ `a73ff7a`, pin `f4e27ca`, development head
  `c4a3a26`) joins by merge (exec plan 4.1b). Refusals this pack hits are listed in `PLAN.md`.

## Memory
- Weights 14.000 GiB; int8 KV 16,512 B a token (+2,064 for the drafter's global layer); 4 slots x 262,144 is 18.14 GiB
  (`docs/FIT.md`). All unmeasured.
- Guard floors: carry the Small recipe's (2 GiB while loading, 1 GiB once ready, kill after 3 breaches) until a boot
  record says otherwise.

## Targets
- Proposals in `docs/ROOFLINE.md`, derived by rule from the census arithmetic and the siblings' measured efficiency. The
  owner has not set any number.

## Operations
- **Owning stack:** the Inkling-2-Small TP4 ring holds all four Sparks today (`inkling2.sh`, `~/recipes/Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark`).
  Not touched. Any node used here needs that group stopped first, and stopping one rank ends it.
- **Staging:** one copy, 14.0 GiB, to the chosen node (`./<controller> fetch` + `verify` against `intake/manifest.tsv`).

## Benchmarking
- Gate set and records follow the Small recipe (`EVAL-MATRIX.md`); no baseline exists yet.

## The drafted round's budget (RoundOverhead, `bench/records/*-round-overhead`)

- **The draft budget is priced at served windows** (engine `round_cost`): 18.9 ms for one row, 41 for eight, 4 streams x 1 row 37.7, 4 x 8 rows 92, the proposal 6.4 to 8.3 ms behind the drafter graph. The earlier table (synthetic ids) priced real text at 55 to 70 %,
  so a full-depth round looked 12 ms cheaper than it was (8 rows: 29.5 against 41). The break-even for one draft is 0.57 in both tables (7.3 ms of proposal and a two-row window of 22 ms against 19 ms serial); the error is in the depth, and prose (acceptance 0.15 to 0.3) was drafted at depth for 20 % less than serial. Measured: prose 41.8 -> 49.9 tok/s, c = 4 aggregate
  101.4 -> 109.0, structured 73.4 -> 79.0, code 122.6 -> 118.9, json 88.2 -> 88.2. The one-stream window is the straight line through the three kinds' served windows (3.15 ms a row); code's own windows are cheaper (39.8 ms at 8 rows) and json's dearer (43.6), so code
  and json lose 2 to 3 % to a trimmed last draft. A steeper line (3.4 ms a row) made it worse (code 117), a flatter one was not tried.
- **Idle relax 0.99** (`Engine::with_idle_relax`, `REXL3_INKLING2_LOOP_SPEC_RELAX`, Mini only): the curve of a stream the plan stopped drafting fades by 0.99 a round (retry after about 70 rounds) instead of 0.9 (about 10): a retry costs
  31 ms (a full-depth proposal and window) against 19 for a serial round. Prose 47.6 -> 50.9 tok/s. Other families keep the curve's own decay.
- **Learnt class priors stay on** (`REXL3_INKLING2_LOOP_SPEC_LEARN`): frozen priors measured worse on prose at two builds (47.1 against 47.6; 48.8 against 51.1) and no better elsewhere. The speed of a kind still depends on the traffic before it
  (the first `bench decode` run after a boot reads 1 to 2 below the next two): code after prose on a fresh server runs at 0.8 of its fresh tokens a round (specmix: 4.60 against 5.7).
- **Chained rounds are the default** (`REXL3_INKLING2_LOOP_CHAIN=0` turns them off): same bytes (`round-exact`), 0.4 to 0.7 ms a round, 3 to 5 % at four streams together with the prices.
