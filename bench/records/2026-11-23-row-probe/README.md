# row-probe: why the engine's argmax differs from both oracles on rows 5, 5 and 23 of the chat prompts

Gate `row-probe`, node-1, no engine build (the dumps are FixtureGate's lab build of engine `7da8c61`, `REXL3_INKLING2_LOOP_DUMP` + `_OPS`,
`--prefill-full`: the rows of the fixtures). Oracle at the recipe commit in `meta.json` (`dense_fp16` gate variant, `moe_fp16`).
New oracle tools (mine): `teacher_force.py --per-row`, `row_table.py`, `unembed_probe.py`, `row_gap.py`.

**Answer: no op stands out and the unembed is exact; the flips are the engine's accumulated final-norm noise at near-ties.**
"Row" is the absolute row of the prompt (the dump's row index); row 5 of `chat-capital` and of `chat-sky` is ONE position (the prompts share
their first 9 tokens, so the oracle's and the engine's values there are identical in both): the 3 flips are 2 independent positions of 1,265 rows.

## 1. The last op alone (`raw/unembed_rows.txt`): oracle logits from the engine's own `h_final`

| | fp16-rotation unembed (`dense_fp16`) | exact-numerics unembed |
|---|---|---|
| relative L2 to the engine's logits, median / max over all 1,265 rows | 3.8e-7 to 4.3e-7 / 3.6e-5 (row 406 of the long prompt) | 6.6e-5 to 1.5e-4 / 2.0e-4 |
| argmax differs | 0 rows | 0 rows |
| row 5 (`chat-capital`): rel; max abs logit diff; change of the top-2 gap | 1.4e-6; 9.5e-5; -1.1e-5 | 5.8e-5; 4.2e-3; +0.0007 |
| max abs change of the oracle's top-2 gap, chat prompts | 1.1e-4 (capital) | 0.0017 |

So the engine's EXL3 unembed, soft cap and mask equal the oracle's with the fp16 rotation to fp32 summation order at every row, rows 5 and 23
included (no neighbour is better or worse), and the rotation itself moves the logits by about 4e-3 and a top-2 gap by under 0.002: it cannot
explain gaps of 0.17 to 0.22 (`unembed_probe.py`, 4 prompts).

## 2. Every op per row, 48 passes, the two chat prompts (`raw/row_table_*.txt`, full per-row values in `raw/teacher_force_per_row_*.json.gz`)

Each pass fed the engine's inputs; per op and pass, a row's value over the median of the OTHER rows (`row_table.py`; the denominator
never below 1e-4 for bf16-level ops, 1e-8 for the router logit error, 1 for residual elements). Both reports pass the stock gate (`pass`: true).

| op | worst (pass, row) ratio, capital | worst, sky | rows 5 and 23: max ratio over passes (at pass) | rows 0 to 3: max ratio |
|---|---|---|---|---|
| attention core | 17.6 (17, 25; 3.6e-3) | 17.9 (4, 4; 2.8e-3) | capital 4.3@3 / 4.2@6; sky 3.1@41 / 5.8@6 | 5.2 (row 2) |
| attention output | 7.7 (18, 13) | 9.0 (18, 46) | capital 3.5 / 2.8; sky 4.9 / 3.2 | 3.3 (row 3) |
| MoE input mix | 9.5 | 7.1 | capital 3.7 / 4.5; sky 3.7 / 2.2 | 5.1 |
| `attn_in` | 8.5 | 12.1 | capital 5.9 / 0.4; sky 5.9 / 1.0 | 8.5 (row 3, pass 19) |
| router logit error | 1.7 | 1.7 | 1.2 / 1.4 | 1.7 |
| MoE output | 3.8 | 4.8 | capital 2.1 / 2.4; sky 2.2 / 2.0 | 2.6 |
| residual elements, `msg` | 1 element (pass 38 row 39); `msg` 1.8e-7 | 2 elements (rows 39, 44) | 0 | 0 |

Rows 5 and 23 are never the worst row of a pass for any op in more than 2 passes; the rows that recur as worst (router logit error rows 7, 13;
attention core rows 25, 32, 39) are rows with no flip. The first rows (0 to 3: short contexts, sink, conv warm-up) are not special either
(ratios at or below 5.2; row 0's medians are the lowest). The largest single deviation anywhere is attention core 3.6e-3 at (pass 17, row 25) and
2.8e-3 at (pass 4, row 4, shared by both prompts): both below the gate's 2e-3 median limit only because they are single rows; neither is at
rows 5 or 23, and neither changes an argmax. Pass 4 / row 4 and pass 17 / row 25 are the candidates if a row-level attention question is ever
asked; this probe does not pursue them.

## 3. Where the gap comes from (`raw/row_gap_rows.txt`: dump vs the oracle fixture with engine numerics, the floor = spec fixture vs it)

At row 5 the engine's logits are 0.0137 (relative L2) from the engine fixture against a floor of 0.0039 (3.5x, the FixtureGate figure);
its `h_final` is 3.4% from the fixture's. But the engine's error at the other chat rows is as large: capital rows 0, 2, 3, 4, 7 are 0.034, 0.034, 0.035,
0.025, 0.044 (median over rows 0.0206, floor 0.0108); row 5's floor is simply a small draw. What matters for the argmax is the change of the
top-2 gap: the engine's gap error at row 5 is -0.300 against a margin of 0.167, at sky row 23 -0.260 against 0.220: the same size as the
chat prompts' typical gap error (|gap error| median 0.135 / 0.086, 90th percentile 0.48 / 0.33; rows 5 and 23 sit at the 78th to 85th
percentile). Rows with a margin under 0.25 and a gap error that did not close it: capital 35 (0.036 vs -0.019), sky 15, 17, 18, 30 (0.007 to
0.104 vs 0.071, -0.027, 0.048, 0.006); the long prompt and the counting prompt have no such row with a gap error as large (counting |gap error|
median 0.015). In short, a gap error of 0.2 to 0.3 is ordinary for the chat prompts after 48 bf16 passes, and rows 5 and 23 are two of the
roughly 11 chat rows whose oracle margin is under 0.5.

## Not proven

- Teacher forcing removes accumulation: the end-to-end `h_final` divergence of rows 5 and 23 (3.4% and 3.6%) is not decomposed per op; each op is
  at the neighbours' level, so nothing singles out a cause, and the sum of ordinary per-op noise is what the numbers support.
- The expected-flip statistic (`row_gap.py` prints it: 3.3 and 2.8 for the chat prompts against 1 and 2 observed) assumes Gaussian gap errors
  with the rms over rows of margin above 1; it is a plausibility figure, not a test.
- Only the two chat prompts were teacher-forced (counting and long: unembed and gap only). Each run took 54 to 56 min on node-1
  (above the 40 min the lead named: the dumps hold 40 and 49 rows, more than the 25 and 26 of the prefill-only dumps).

## Node footprint

`/home/spark/inkling-2-mini-1x-dgx-spark/lab/rowprobe/` (12 MiB): `oracle/` (sources only) and `out/` (reports and logs). No process left running.
Remove with `rm -rf` of that directory.
