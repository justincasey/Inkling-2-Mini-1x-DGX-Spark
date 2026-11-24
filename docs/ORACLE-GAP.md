# Engine vs oracle on the Mini: the whole-forward gap, explained

Plan gate 6.3 (oracle), second half. The first half (`bench/records/2026-11-23-forward-oracle-engine-emulation/`) fed every oracle op the engine's
own inputs on two prompts of 26 and 35 tokens, 48 passes each, and the stock limits held (no op missed once the oracle emulates the dense
kernels' fp16 rotation). Three things were never compared: a prompt past the 1,024-token window, the logits plane, and the engine generating on
its own. This document does those, for the REXL3 family `inkling2_loop` at engine `7da8c61344842f4a2605591d22e97f5ad28d1a7e` (a lab build,
committed tree) on node-1, against the fp64 oracle fixtures of `bench/records/2026-11-23-oracle-fixtures/`: four prompts (25, 26, 35 and 1,119
tokens, 16 to 24 generated each), in two numerics. Evidence: `bench/records/2026-11-23-oracle-gap/`.

## Conclusion

**No defect found; the gap is the model amplifying fp noise, and it is no larger than the oracle's own two numerics are from each other.**

1. **Free-running, the engine generates the oracle's tokens.** 64 of 64 greedy tokens equal on all four prompts against both fixture sets
   (they hold the same tokens), among them the three generated positions whose oracle top-1 - top-2 margin is under 0.3 (0.069, 0.103,
   0.198). The 1,119-token prompt crosses the window and its 8 tokens are equal too.
2. **Whole-forward, the planes drift the way the Small's did, and the same amount as an oracle with other numerics.** The hidden state of the
   long prompt (engine against the engine fixture, relative L2 of the whole plane) is 4e-5 at the embedding (`hidden[0]`), 2.5e-3 after visit 0, 7.8e-3 after visit 1, and
   grows through the 48 visits to 0.26 at visit 36 (median row 0.097), then comes back to 0.07 at the end. The oracle with the engine's four lossy choices switched on (the *floor*) moves from the exact oracle
   by 0.27 at the same visit: the engine over the 48 visits is at 0.76 to 1.09 of the floor at every visit (long prompt, `curve`
   below). 27.4 % of the (visit, row) router pairs of the long prompt have another top-8 set than the engine fixture's (27.0 % against the spec
   fixture); the oracle's two numerics differ on 28.5 %. The rows where the engine is worst are the floor's worst rows (long prompt, logits relative L2: row 689 is
   1.04 against the engine fixture and 0.56 for the floor, row 541 is 0.56 against the spec fixture and 0.60 against the engine one; `raw-counting` row 41: 0.17, 0.10, floor 0.19).
3. **Argmax.** Over 1,265 rows the engine's top-1 differs from the oracle's on 3 rows, all prompt rows of the two chat prompts (rows 5, 5 and 23;
   row 5 is one position the two prompts share), at oracle top-1 - top-2 margins of 0.103 to 0.22 logits and gaps to the engine's pick of 0.10 to
   0.22; no generated row differs. The oracle's own two numerics differ on 0 of 1,265 rows. The stock `compare.py` spec gate (the Small's tie
   margin 0.1) therefore fails two prompts on 3 rows by 0.003 to 0.007 logits, and its engine gate (hidden cosine 0.999 on every token,
   KL 1e-3) fails as it did on the Small. Both are measurements, not verdicts; see "Limits proposed".
4. **The gate of record stays the op-level one** (`teacher_force.py`, passed in the first half), and the long prompt has no op-level
   comparison (see "Not measured").

## What the engine does differently from the exact oracle

Definitions: `docs/NUMERICS.md`, "Engine emulation". The oracle reproduces four lossy choices (`Emulation.engine()`: `kv8`, `attn_bf16_p`,
`moe_fp16`, `dense_fp16`):

- the 8 global visits keep K/V as int8 with a power-of-two scale per (token, KV head); the 40 sliding visits keep bf16;
- the attention kernel rounds the softmax probabilities to bf16 for `P V`, in absolute 512/256-key chunks;
- the EXL3 expert GEMMs take fp16 rotated inputs;
- the dense EXL3 linears (q, r, gate, k, v, `wo`, the unembed) take `fp16( (x (*) suh) H / sqrt(128) )`.

What it cannot reproduce is the engine's fp32 summation order in every GEMM and in the softmax. That order is a
perturbation of about 1e-3 per op (the first half's noise floor). Teacher-forced, one op at a time, nothing beyond it is left: all 96 passes of the
first half fit the stock limits. Free-running the 48 visits amplify it.

The spec fixtures are the exact float64 policy; the engine fixtures switch the four choices on (`--emulate engine`). Both fixtures hold the
same greedy tokens on all four prompts.

## Method

- **Dumps, one engine boot** (`inkling2mini.sh start` with `EXTRA_ENV=REXL3_INKLING2_LOOP_DUMP=<dir> REXL3_INKLING2_LOOP_DUMP_OPS=1`, `--spec none`
  `--prefix-cache off` `--no-vision`, prefill chunk 4,096): the rank's dump hooks write one file per prefill chunk, `hidden` [49, T, 2304], `core_out`,
  `h_final`, `logits` [T, 200058] and the ops planes. A live server was simpler than the rank test: `forward_oracle_gate` takes 8 to 64
  tokens, and the same boot serves the greedy runs. `greedy_oracle.py --prefill-full` sends, per fixture, the fixture's tokens but the last as
  one prompt (`max_tokens` 1), so the dump holds exactly the rows the fixture holds (40, 49, 50 and 1,126), then runs the greedy comparison.
  The dump of the long prompt is 3.9 GB (ops planes included): it stays on the node and is compared there.
- **Determinism and reproduction of the first half.** The dumps of the 26- and 35-token prompts are byte-identical to the first half's
  (node-3, engine `aeaaa4f9`, rank test): sha256 `c25f9876...` and `0773323c...`. The 1,119-token prompt prefilled twice gives byte-identical 3,875,832,200-byte files
  (`288f2376...`). The server at `7da8c61` on node-1 therefore writes the same bytes as the rank test of the earlier commit on another node.
- **Comparison** (`oracle/engine_gap.py`, CPU, the oracle venv, 8 threads, no oracle recompute: 20 s for all twelve): per prompt the engine dump
  against the engine fixture and against the spec fixture, and the two fixtures against each other (the floor). Per visit 0..48 of `hidden`: the
  relative L2 of the plane and of every row (median, maximum), the minimum and median cosine, max |e - o| over the row's RMS; the same for `core_out`
  and `h_final`; per logits row: cosine, relative L2, max |e - o| over the row's RMS, KL(oracle || engine), argmax equal, top-2 order equal, the oracle's
  margin; per (visit, row) the top-8 set equal or a flip, with the oracle's selection margin (and the engine's, from its dumped router logits and the
  pack's bias). Row groups: all, prompt, generated, and for the long prompt before the window (rows < 1,024), after it, the edge (rows 1,016 to 1,031)
  and the tail (the last 16). `oracle/tests/test_engine_gap.py`, `test_oracle_gap_summary.py` (20 tests) pin the statistics on synthetic planes.
- **Joined** by `bench/tools/oracle_gap_summary.py` (`raw/summary.json`, `raw/tables.md`).

## Evidence

Planes, per prompt and comparison (hidden = the worst visit's whole-plane relative L2, its index in brackets; logits relative L2 over the whole
plane, the median row and the worst row; KL in nats; flips of the (visit, row) pairs):

| prompt | rows | comparison | hidden rel L2 (worst visit) | h_final rel L2 | logits rel L2 plane / row median / row max | logits cos min | KL mean / max | argmax equal | top-2 equal | flips of pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| chat-capital-effort0 | 40 | engine vs engine fixture | 0.11 (47) | 0.0774 | 0.0298 / 0.0206 / 0.176 | 0.98646 | 0.00734 / 0.0822 | 39/40 | 36/40 | 238 / 1920 (12.4%) |
|  |  | engine vs spec fixture | 0.0637 (40) | 0.062 | 0.0259 / 0.0188 / 0.198 | 0.98454 | 0.00691 / 0.0795 | 39/40 | 36/40 | 245 / 1920 (12.8%) |
|  |  | floor: engine fixture vs spec fixture | 0.126 (47) | 0.0728 | 0.0253 / 0.0108 / 0.0776 | 0.99863 | 0.00277 / 0.0512 | 40/40 | 36/40 | 182 / 1920 (9.5%) |
| chat-sky-effort0.9 | 49 | engine vs engine fixture | 0.0463 (48) | 0.0506 | 0.0191 / 0.0137 / 0.0864 | 0.99724 | 0.00411 / 0.0678 | 47/49 | 44/49 | 213 / 2352 (9.1%) |
|  |  | engine vs spec fixture | 0.0489 (37) | 0.0462 | 0.017 / 0.0148 / 0.0943 | 0.99599 | 0.00493 / 0.0654 | 47/49 | 46/49 | 229 / 2352 (9.7%) |
|  |  | floor: engine fixture vs spec fixture | 0.0433 (40) | 0.0451 | 0.0166 / 0.00972 / 0.0952 | 0.99589 | 0.00246 / 0.0512 | 49/49 | 45/49 | 204 / 2352 (8.7%) |
| raw-counting | 50 | engine vs engine fixture | 0.178 (24) | 0.0491 | 0.0247 / 0.016 / 0.172 | 0.98772 | 0.000283 / 0.00348 | 50/50 | 50/50 | 420 / 2400 (17.5%) |
|  |  | engine vs spec fixture | 0.159 (36) | 0.0475 | 0.0239 / 0.0173 / 0.0994 | 0.99505 | 0.000268 / 0.0028 | 50/50 | 50/50 | 463 / 2400 (19.3%) |
|  |  | floor: engine fixture vs spec fixture | 0.168 (36) | 0.0459 | 0.0253 / 0.014 / 0.188 | 0.98747 | 0.000166 / 0.00152 | 50/50 | 50/50 | 407 / 2400 (17.0%) |
| raw-long-window | 1126 | engine vs engine fixture | 0.256 (36) | 0.111 | 0.103 / 0.08 / 1.04 | 0.87126 | 0.000718 / 0.0923 | 1126/1126 | 1112/1126 | 14827 / 54048 (27.4%) |
|  |  | engine vs spec fixture | 0.26 (36) | 0.0829 | 0.0786 / 0.0582 / 0.56 | 0.90058 | 0.000231 / 0.012 | 1126/1126 | 1110/1126 | 14589 / 54048 (27.0%) |
|  |  | floor: engine fixture vs spec fixture | 0.267 (36) | 0.106 | 0.0897 / 0.0735 / 0.561 | 0.88274 | 0.000681 / 0.016 | 1126/1126 | 1115/1126 | 15414 / 54048 (28.5%) |
The floor row is what changing the oracle's numerics by the four choices does to the same planes. The engine's rows sit at it, not below it:
that the oracle with the engine's numerics tracks the engine no better than the exact one (long prompt: hidden 0.256 against the engine fixture,
0.260 against the spec one) is the sign that what is left is the fp32 summation order, amplified, which the oracle does not reproduce.

Per visit, the engine's plane error over the floor's, the flips that recur when the oracle's numerics change instead, the smallest router margin
(the closest top-8 call in the fixture) and the largest weight difference on identical sets (routing weights sum to 8):

| prompt | engine / floor, per visit 1..48 (max at visit; min) | flips that recur in the floor (engine vs spec): share vs the floor's rate | smallest router margin | weights, max abs diff (identical sets) |
|---|---|---|---|---|
| chat-capital-effort0 | 2.19 (visit 8); 0.35 | 111/245 = 45.3% vs 9.5% | 1.31e-05 | 0.27 |
| chat-sky-effort0.9 | 1.29 (visit 13); 0.74 | 126/229 = 55.0% vs 8.7% | 3.13e-07 | 0.299 |
| raw-counting | 1.32 (visit 31); 0.58 | 289/463 = 62.4% vs 17.0% | 2.14e-08 | 0.552 |
| raw-long-window | 1.09 (visit 5); 0.76 | 11371/14589 = 77.9% vs 28.5% | 1.67e-07 | 1.12 |
On the 40- and 50-row prompts one or two rows decide a visit's plane (the first column reaches 2.19 and 1.58 there), so the per-visit curve is a
check of the long prompt only (1,126 rows: 0.76 to 1.09, no step at any visit, so no visit is a place where the engine departs from the oracle).

**Router.** A flip is a (visit, row) whose top-8 set differs. Flips grow with the visit (1 at visit 0, 6 at 1, 20 at 3, 351 at 19, 674 at 32 of the
1,126 rows) and with the number of differing inputs: teacher-forced they are zero (first half). Visit 0 reads the embedding only, bitwise equal on all but
5 rows of the long prompt; its 1 flip against the engine fixture (oracle margin 2.2e-7) and 9 against the spec fixture all sit on rows whose input is
equal, so the layer-0 ops' own noise (1e-3 relative) flips them, as it flips the floor's 10. Flips are near ties: by the oracle's
selection margin of the pair (long prompt, engine vs spec fixture):

| margin of the pair | [0, 1e-5) | [1e-5, 1e-4) | [1e-4, 1e-3) | [1e-3, 1e-2) | [1e-2, 0.1) | [0.1, inf) |
|---|---|---|---|---|---|---|
| pairs | 49 | 427 | 4,184 | 25,259 | 23,586 | 543 |
| flips | 32 (65 %) | 275 (64 %) | 2,340 (56 %) | 8,682 (34 %) | 3,257 (14 %) | 3 (0.6 %) |

The flip rate falls with the margin, and 11,371 of the 14,589 flips (78 %) also flip when only the oracle's numerics change (the floor flips 28.5 % of
the pairs): the pairs that flip are the same pairs. The largest margin of a flip is 0.135 to 0.157 (long prompt; the floor's 0.157) and 0.02 to 0.06 on the short
prompts; hidden planes differ by 0.05 to 0.26 there, so the router scores move by much more than 1e-3. The routing weights of identical sets differ by up to 0.27 (capital) to 1.12 (long) of the 8 (floor 0.21 to 1.90).

**The window.** Rows past 1,024 lose the first key from the sliding visits. Nothing happens there. Logits relative L2 against the spec fixture: 0.080 (rows < 1,024), 0.059
(rows 1,024 to 1,125), edge (rows 1,016 to 1,031) 0.069, tail (the last 16 rows) 0.050. Against the engine fixture (floor in brackets), the hidden plane at visit 36: 0.258, 0.234, 0.209, 0.200
(0.271, 0.234, 0.184, 0.203) and flips per pair 27.8 %, 23.8 %, 18.0 %, 24.2 % (28.9 %, 25.1 %, 23.4 %, 24.5 %) in the same four groups. Every one of the 1,126 argmax is equal
and the top-2 order equal on 1,112 (floor 1,115) of them.

**Which rows.** The worst rows by logits relative L2 recur between the engine and the floor (long prompt: row 689 is 1.04 against the engine fixture and 0.56 for the floor; row 541
is 0.56 against the spec fixture; counting: row 41 is 0.17 / 0.10 / floor 0.19), and the oracle's margin at rows 689 and 41 is 5.5 to 6.2 logits (row 541: 4.5): a row can be far
off in the logits plane and keep its argmax. `raw/row_probe.txt` has rows 3 to 7 of `chat-capital` and rows 5 and 23 of `chat-sky`.

## Free-running drift: the engine's greedy tokens

`greedy_oracle.py --prefill-full` against `fixtures/spec` and `fixtures/engine` (`raw/greedy-spec.json`, `raw/greedy-engine.json`; the oracle's top-1 - top-2
margin comes from the fixture's own logits):

| prompt | generated | engine tokens vs spec fixture's | vs engine fixture's | oracle top-1 - top-2 margin at the generated positions: min (position); positions under 0.3 |
|---|---|---|---|---|
| chat-capital-effort0 | 16 | equal | equal | 0.103 (11); 2 of 16 |
| chat-sky-effort0.9 | 24 | equal | equal | 0.0688 (5); 1 of 24 |
| raw-counting | 16 | equal | equal | 3.66 (8); 0 of 16 |
| raw-long-window | 8 | equal | equal | 2.41 (6); 0 of 8 |

No divergence, so no first-divergence position to excuse. Of the 64 generated positions three have an oracle margin under 0.3, and the engine took the oracle's token at all three.

**How often the oracle's own two numerics disagree on the greedy token, against the engine** (every row, the oracle's top-1 - top-2 margin of the row on
top, pooled over the four prompts, 1,265 rows):

| oracle margin of the row | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.5) | [0.5, 1) | [1, inf) |
|---|---|---|---|---|---|---|---|
| rows (spec fixture's margins, all prompts) | 0 | 3 | 7 | 2 | 6 | 5 | 1242 |
| argmax differs, engine vs engine fixture | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| argmax differs, engine vs spec fixture | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| argmax differs, floor: engine fixture vs spec fixture | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The oracle's two numerics disagree on 0 of 1,265 rows (0 of the 64 generated), the engine and either oracle on 3 (0 of the 64 generated, rows 5, 5 and
23, gaps 0.10 to 0.22). Of the 12 rows with an oracle margin between 0.05 and 0.3, the engine takes another token on 3 (25 %), the oracle's other numerics on none.
So at the near-tie band the engine is farther from the oracle than the oracle is from itself: its logits are 1 to 3 % (relative L2 0.0137 to 0.030 at those rows)
off at a margin of 0.1 to 0.2 logits. The 1.4 % of row 5 is 3.5x the floor's 0.39 % at the same row; at row 23 the engine's 3.0 % and the floor's 2.6 % are alike. It is one position, not a pattern, and not on the path of a greedy
token; it is the one place where the measurement does not put the engine inside the floor.

## Against the limits that exist

| limit | where | engine vs spec fixture | engine vs engine fixture | floor (spec vs engine fixture) |
|---|---|---|---|---|
| `compare.py` `GATES_SPEC` (hidden cosine median per visit >= 0.95, h_final median >= 0.99, KL mean <= 2.5e-2, KL max <= 1.0, top-1 >= 0.95, no disagreement above tie margin 0.1) | `oracle/compare.py` | counting and long pass; capital and sky fail `logits` only, on `top1_disagree_outside_ties` (1 and 2 rows; margins 0.103, 0.103, 0.107); every other check passes (hidden median min 0.9959, h_final median min 0.9977) | n/a | passes all four |
| `compare.py` `GATES_ENGINE` (hidden and h_final cosine >= 0.999 on every token, KL mean <= 1e-3, KL max <= 1e-2, top-1 >= 0.99, tie margin 0.05) | `oracle/compare.py` | n/a | fails hidden and h_final on all four (cosine minimum 0.37 to 0.98 and 0.78 to 0.98), logits on capital, sky and long (KL max 8.2e-2, 6.8e-2 and 9.2e-2; top-1 0.975 and 0.959); counting passes logits | n/a |
| `teacher_force.py` `GATES` (core 2e-3, attention output 6e-3, mix 2e-3, MoE 1.5e-3, router logit 1e-6) | op level, teacher-forced | not run on these dumps | not run on these dumps | first half: all 96 passes inside, margins 2.1x to 3.7x |

`GATES_ENGINE` fails for the same reason it failed on the Small (no fp32 implementation that sums in another order meets 0.999 on every token of a model that
amplifies 1e-3 into 0.1) and the oracle with other numerics fails it as well (the floor's hidden cosine minimum is 0.49 to 0.98). `GATES_SPEC` misses by 3 rows at 0.003 to 0.007
logits above its tie margin.

## Limits proposed

For the whole-forward drift of the free-running gate (`oracle/compare.py` `GATES_SPEC` and a new row-level set; none is edited here: `teacher_force.py`'s limits
are untouched, `compare.py` is not changed). Each limit is the worst value measured over the four prompts, engine against either oracle, times a headroom,
rounded up; the three the Small already had are kept. With four prompts these are fitted to the measurement: they are not independent evidence, the
floor comparisons below are.

| metric (per prompt, per oracle) | worst measured | proposed | arithmetic |
|---|---|---|---|
| hidden relative L2, whole plane, every visit | 0.2598 (long, spec) | <= 0.40 | 0.2598 x 1.5 = 0.39 |
| hidden relative L2, row median, every visit | 0.0972 (long, engine) | <= 0.15 | x 1.5 = 0.146 |
| hidden cosine, row median, every visit | 0.9957 (long, engine) | >= 0.95 | the Small's; 1 - 0.9957 = 0.0043 of the 0.05 allowed |
| `core_out` relative L2, whole plane | 0.1141 (long, spec) | <= 0.18 | x 1.5 = 0.171 |
| `h_final` relative L2, whole plane | 0.1108 (long, engine) | <= 0.17 | x 1.5 = 0.166 |
| `h_final` cosine, row median | 0.9958 (long, engine) | >= 0.99 | the Small's |
| logits relative L2, whole plane | 0.1025 (long, engine) | <= 0.16 | x 1.5 = 0.154 |
| logits relative L2, row median | 0.0800 (long, engine) | <= 0.12 | x 1.5 |
| logits cosine, worst row | 0.8713 (long, engine) | >= 0.80 | no Small limit; 0.8713 less 0.07 |
| KL(oracle \|\| engine), mean | 7.34e-3 (capital, engine) | <= 1.5e-2 | x 2 = 1.47e-2 |
| KL, max over rows | 0.0923 (long, engine) | <= 0.2 | x 2 = 0.185 |
| argmax equal, fraction of rows | 0.9592 (sky) | >= 0.95 | the Small's |
| argmax difference allowed when the oracle's top-1 logit exceeds the engine's pick by at most | 0.22 (sky row 23, engine fixture) | 0.3 | x 1.35; the Small's 0.1 (spec) and 0.05 (engine) are below the 0.103 to 0.22 measured |
| top-2 order equal, fraction of rows | 0.898 (sky) | >= 0.85 | 0.898 x 0.95 |
| router flips, fraction of (visit, row) pairs | 0.2743 (long, engine) | <= 0.41 | x 1.5 |
| greedy tokens | all equal | equal, or the first difference at an oracle margin under 0.3 | the same tie margin |

Floor-relative checks, which no fitted number decides (`oracle_gap_summary.py` `LIMITS`): the engine's flips over the floor's, pooled over the four prompts
<= 1.25 (measured 0.96 against the spec fixture, 0.97 against the engine fixture: 15,526 and 15,698 flips against 16,207) and per prompt <= 1.5 (measured 0.95 to 1.35: capital 245 and 238
against 182 flips); the per-visit curve of the long prompt <= 1.5 of the floor's (measured 1.09 and 1.10). Not limited: the per-token hidden cosine minimum (0.37 to 0.98: a flipped route
moves a single token's hidden state by percents, which the oracle's own other numerics do too: 0.49 to 0.98) and the worst logits row's relative L2.

**Verdict of the record** (`meta.json` `result`): every check above holds (138 of 138 per-prompt and pooled checks; `raw/summary.json`). That is a statement
about the fitted limits and about the floor, and about no more: the engine's whole-forward numbers are not those of the Small's `GATES_ENGINE`, and
`GATES_SPEC`'s tie margin is missed by 3 rows.

## Not measured

- **Positions past 8,192**, where log scaling `tau = 1 + 0.1 ln(pos / 8192)` bites through the whole forward of the global visits: the longest prompt is 1,126 rows. A fixture
  past it was not generated: the 1,119-token fixtures took 922 s (spec) and 929 s (engine) of oracle CPU on this node, and a prompt of more than 8,192 rows is more than 7x the rows
  with attention growing faster [INFERENCE: two hours or more per numerics, not measured].
- **The long prompt has no op-level comparison.** `forward_oracle_gate` takes at most 64 tokens; `teacher_force.py` on 1,126 rows needs the oracle's experts over all of
  them (the first half estimated about 15 hours or more); the dump with the ops planes is on the node (`lab/fixgate/dump/prefill00006.safetensors`) for it. What exists: the long prompt's planes against both fixtures, no step at any
  visit or at the window, and the engine's 1,119-token decode == prefill and chunk-width checks of the first half.
- **A perturbation-only oracle** (the emulating oracle with another fp32 summation order, no other change) would complete the floor: the floor here is the four
  lossy choices, not the summation order. The engine's whole-forward gap equalling the four choices' says the same thing from the other side but is not that measurement.
  Generating one is an oracle run of 10 to 16 minutes per prompt at the 25 to 50 rows of the short prompts (the fixtures job's own seconds); not started.
- **Isolating `kv8`** (UNKNOWN-20: kv8 against bf16 KV on a 32,768-token fixture): the floor switches all four at once.
- **Decode windows**: the dump covers prefill rows. Decode == prefill is bit-exact in the engine's tests and the greedy tokens (a decode path) equal the oracle's.
- **The vision tower and the drafter** are other gates (`kernels-vision`, `dspark` fixtures).
- Four prompts, 1,265 rows; three of the prompts are 25 to 50 tokens.

## Reproduce

```bash
# engine: lab build of the committed tree, one boot with the dump hooks, prefill dumps and greedy tokens
REXL3_SRC=~/projects/rexl3-inkling2-loop NODES=<node-1> ./inkling2mini.sh build 7da8c61344842f4a2605591d22e97f5ad28d1a7e
NODES=<node-1> PREFIX_CACHE=off SPEC=none VISION=off \
  EXTRA_ENV="REXL3_INKLING2_LOOP_DUMP=$BASE/lab/fixgate/dump REXL3_INKLING2_LOOP_DUMP_OPS=1" ./inkling2mini.sh start
FX=$BASE/fixtures
python3 bench/tools/greedy_oracle.py --ssh spark@<node-1> --dir $FX/spec   --url http://<node-1>:8000 --prefill-full --out greedy-spec.json
python3 bench/tools/greedy_oracle.py --ssh spark@<node-1> --dir $FX/engine --url http://<node-1>:8000 --out greedy-engine.json
NODES=<node-1> ./inkling2mini.sh stop
# on the node (oracle venv, CPU): per prompt, engine vs engine fixture, engine vs spec fixture, the floor; 20 s in all
oracle/engine_gap.py --oracle $FX/engine/NAME.safetensors --engine $BASE/lab/fixgate/dump --pack $PACK --stock-gates --json dump-vs-engine-NAME.json
oracle/engine_gap.py --oracle $FX/spec/NAME.safetensors   --engine $BASE/lab/fixgate/dump --pack $PACK --stock-gates --json dump-vs-spec-NAME.json
oracle/engine_gap.py --oracle $FX/spec/NAME.safetensors   --engine $FX/engine/NAME.safetensors --pack $PACK --stock-gates --json fixture-engine-vs-spec-NAME.json
# anywhere, stdlib only
python3 bench/tools/oracle_gap_summary.py --dir DIR --out summary.json --tables tables.md
```
(`raw/run_gap.sh` is the loop the record ran.)
