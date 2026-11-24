# fuse-decode: the drafted round, the windows by rows and the proposal after the fuse lane (gate `speed-decode`)

Engine `24a04afe9fda1f304865ed2af0f0c2e13bae7f08` (branch `inkling2-loop-fuse`: 3 engine changes over `1d53344`, below), node-2, clock locked at 2,200 MHz, shipping defaults
(`SPEC=dspark PREFIX_CACHE=on VISION=on`) for the drafted numbers, `SPEC=none PREFIX_CACHE=off VISION=off` for the serial ones. Baseline: `fuse-baseline` (same node, same tools, same day).
Later engine commits (`76ed8af`, `11823b3`) touch only tests (real-text round table, a pinned memory price).

## Result

| Measurement | Baseline `1d53344` | Now `24a04af` | Command |
|---|---:|---:|---|
| **`decode_mean4_tok_s`** (`./inkling2mini.sh bench decode`, mean of prose, code, structured, json at c = 1, drafter on; printed there as `decode_c1_tok_s`), median of 3 | 81.65 | **83.35** (83.35, 83.20, 83.36; +2.1%) | `raw/drafted-final-24a04af/bench-decode-{1,2,3}.log` after a discarded warm-up |
| prose / code / structured / json | 41.8 / 122.9 / 73.6 / 88.3 | 42.7 / 125.6 / 75.0 / 90.0 | same |
| `specbench.py` speed-ups: chat / code / json / longctx / tool / reasoning | 0.897 / 2.252 / 1.660 / 1.572 / 2.372 / 1.410 | **0.917 / 2.296 / 1.691 / 1.617 / 2.413 / 1.439** | `specbench.py --concurrency 4` (`raw/drafted-final-24a04af/specbench.out`; `spec-final`: 0.901 / 2.239 / 1.648 / 1.558 / 2.368 / 1.404) |
| four streams, drafted over serial aggregate | 1.109 (111.27 / 100.36) | **1.132** (113.79 / 100.50) | same |
| proposal, 4,096 positions, 1 / 2 / 4 requests (synthetic ids) | 8.450 / 8.935 / 9.975 ms | **6.690 / 7.161 / 8.208** (-1.76 ms, -21%) | `round_cost_table` (`raw/round-cost-table-synthetic-final.txt`) |
| proposal at 32,768 positions, 1 / 2 / 4 requests | (8.51 ms, `spec-speed`, older engine) | 7.330 / 8.438 / 11.369 | same |
| proposal with real texts, 4K and 32K, 1 / 2 / 4 requests | not measured | 6.900 / 7.538 / 8.931 and 7.524 / 8.807 / 12.017 | `LOOP_REAL_IDS_FILE=raw/real_ids.json` (`raw/round-cost-table-real-ids-final.txt`) |
| 8-row verify window, 4K, one stream: synthetic ids / **real texts** | 29.25 ms (synthetic) | 29.51 / **41.93** (window path unchanged) | same |
| serial decode c = 1 short / 4K, c = 4 aggregate | 53.64 / 51.55 / 100.73 | 53.51 / 51.56 / 100.83 (unchanged: no serial path was touched) | `raw/serial-and-gates-final/sweep.log` |

The windows table with real texts (1, 2, 4, 8, 16, 32 rows at 4K and 32K, 1, 2 and 4 streams), with the bytes and achieved GB/s, is in `docs/DECODE-SPEED.md` ("Windows by rows") and in
`raw/round-cost-table-real-ids-final.txt`. Real windows cost 1.4 to 1.8 times the synthetic ones: 4 x 1 / 4 x 4 / 4 x 8 = 39.6 / 73.4 / 96.1 ms, within 3 / 6 / 8% of `c4_model.py`
(38.7 / 69 / 89). The window is 73 to 87% of the 233.8 GB/s ceiling over its whole length (**INFERENCE**: bytes from `c4_model.py`, not logged).

## What changed (engine commits; each timed on the proposal alone, drafter tests after each)

| # | commit | proposal 4K, 1 request | what |
|---|---|---:|---|
| 0 | `1d53344` | 8.450 ms | |
| 1 | `1a84469` FEAT | 7.916 | the draft pick on the split top-k: the one-block `local_kernel` (90 us) and `merge_kernel` (75 us) a step become 19 + 4 us |
| 2 | `5a1c04b` FEAT | 6.831 | `attn_part_mini_kernel` (GQA 3 only): V rows staged by cp.async, query vectors read as float4, 3 query vectors x 4 columns a thread; same chains in the same order |
| 3 | `5f39463` FEAT | 6.632 | the proposal captured in a graph per request count (`draft_graphs`), replayed against the re-staged proposal |

Directions measured and stopped (details in `docs/DECODE-SPEED.md`): L2 warm-up lane (two variants: 19,011 / 19,119 us against 19,146 one row, 33.4 ms against 33.0 at 8 rows), gain/forget in one launch (19,078 / 19,139 us:
the noise), an in-kernel look-ahead in the MoE GEMV (20.6 ms, worse). None is committed.

## Against `docs/ROOFLINE.md` (arithmetic)

Serial ceiling at 4K 70.1 tok/s (14.27 ms), proposed target 62 (16.13): the engine is at 51.56 (19.40 ms): 74% of the ceiling, 83% of the target; this lane did not move it. The drafted
round at 4K, synthetic ids: window 29.3 + proposal 6.7 = 36.0 ms (was 37.7), break-even 1.88 tokens a round against 19.15 ms serial (was 1.97); with real texts the 8-row window is 41.9 ms,
the round 48.8 ms and the break-even **2.5** tokens a round: prose (1.5 to 1.9 a round at depth 7) still does not pay at full depth (chat 0.917 x, from 0.897), which is the policy's, not
the kernels', to solve. The gap to the serial ceiling (5.1 ms) is, by the profile: 12 dependent kernels a visit of glue and controllers (about 1.7 ms), MoE at 85% of the bus (1.0-2.3 ms),
the attention at twice its KV bound (0.8 ms), `wk`/`wv`/`wr` at 55-95 GB/s (0.5 ms): all but the first measured per visit, the totals extrapolated (**INFERENCE**).

## Not proven / not done

Gate `speed-decode` asks for more than 100 tok/s of mean decode: 83.35 is 17% short; this lane's kernel work is worth +2%, the rest is the round's host time and the policy (other lanes).
The proposal's global-layer attention (600 us, 37 us a tile where the arithmetic wants about 10), a fused argmax in the Markov logits kernel (about 0.17 ms), an L2 residency policy for the
folded table (about 0.3 ms) and the router chain fusion (about 0.3 ms a window) were not done. The drafter graph and RoundOverhead's chained round (`chain_round` enqueues the proposal
eagerly) are not integrated: both branches edit `propose`. Distinct experts per window were not logged.
