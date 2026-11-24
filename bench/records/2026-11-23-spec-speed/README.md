# spec-speed: does the block drafter pay on the expert-bound Mini, and the policy (gate `spec-speed`, plan 4.6 / 6.6)

Node-2, GPU locked at 2,200 MHz, TP1, greedy, 512 generated tokens a request (`bench/tools/specbench.py`: one warm-up pass discarded, the median of
3 timed passes, wall of the request with prefill, serial = the same server with `"draft": false`; every drafted reply equals its serial reply).
Engines: the shipped default is `761b143`; the experiments ran on `5e6df8d` (first run, fixed depth 7), `dfb05bb` (fixed 7 / 4 / 2, plan, Markov c)
and `4234dd7` (learnt plan, learnt curve), which differ from `761b143` only in the policy knobs' defaults. Every number below is in `raw/`.

## Decision: ship `dspark` as the default (`docs/DECISIONS.md`)
Serial token 19.15 ms (52.2 tok/s). A drafted round = 8-row verify + proposal + commit. Rank times (`timing.txt`, `inkling2_loop_spec_timing`,
median of 16 windows, drafter attached, 1K / 4K / 32K context): serial 1 row 19.06 / 19.27 / 21.36 ms, 8-row window 30.37 / 29.36 / 31.96 ms
(1.54 x one row), proposal 7.42 / 8.31 / 8.51 ms: a round 37.8 / 37.7 / 40.5 ms, break-even 1.98 / 1.96 / 1.90 tokens a round. End to end the round
also carries the engine's own time: reasoning 3.26 tokens a round at 72.3 tok/s = 45.1 ms, code 5.75 at 117.0 = 49.1 ms, so break-even there is
about 2.4 tokens a round. The rule (win at c=1 on the median of reasoning, chat and code by more than the noise; no loss at c=4 aggregate) is met:
median 1.404 x, noise 0.01 (three repeats of one configuration: chat 0.883 / 0.888 / 0.883, code 2.238 / 2.244 / 2.247, reasoning 1.402 / 1.388 / 1.402),
c=4 aggregate 1.106 x. Prose loses (0.90 x with other traffic before it, 0.98 x on a server that has seen only prose): its acceptance (0.14-0.21 of
drafts, 1.93 tokens a round) is below break-even.

## Shipped default (`761b143`: plan budget, learnt priors, depth 7, folded Markov head), `specbench-final.md`
| kind | prompt tok | serial tok/s | drafted tok/s | speedup | accepted/offered | tokens/round |
|---|---:|---:|---:|---:|---|---:|
| chat (prose) | 74 | 52.20 | 47.05 | 0.901 | 739/3594 (0.21) | 1.93 |
| code | 225 | 52.28 | 117.03 | 2.239 | 1266/1845 (0.69) | 5.75 |
| json (structured) | 251 | 51.59 | 85.01 | 1.648 | 1188/2363 (0.50) | 4.45 |
| longctx (6K prompt) | 6,047 | 40.29 | 62.77 | 1.558 | 1190/2200 (0.54) | 4.48 |
| tool call turn | 166 | 51.47 | 121.89 | 2.368 | 939/1146 (0.82) | 6.47 |
| reasoning | 150 | 51.51 | 72.32 | 1.404 | 1062/2793 (0.38) | 3.26 |
| c = 4 (chat, code, json, reasoning), aggregate | | 99.92 | 110.48 | 1.106 | | |

## Per-position acceptance (accepted / reached, fixed depth 7, `fixed7/specbench.md`)
| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 | tokens/round |
|---|---|---|---|---|---|---|---|---|
| chat | 0.56 | 0.44 | 0.40 | 0.56 | 0.29 | 0.25 | 0.00 | 1.97 |
| code | 0.91 | 0.93 | 0.91 | 0.91 | 0.92 | 0.79 | 0.82 | 5.75 |
| json | 0.79 | 0.74 | 0.85 | 0.88 | 0.98 | 0.92 | 0.98 | 4.49 |
| longctx | 0.89 | 0.82 | 0.84 | 0.81 | 0.80 | 0.81 | 0.66 | 4.61 |
| tool | 0.95 | 0.93 | 0.96 | 0.85 | 0.98 | 1.00 | 1.00 | 6.47 |
| reasoning | 0.76 | 0.80 | 0.68 | 0.60 | 0.66 | 0.60 | 0.67 | 3.35 |

## Depth and policy (speedup at one stream; c = 4 aggregate speedup)
| setting | chat | code | json | reasoning | c = 4 |
|---|---:|---:|---:|---:|---:|
| fixed depth 7 (`fixed7`; first run `5e6df8d`: 0.814 / 2.239 / 1.653 / 1.381, c4 1.044) | 0.816 | 2.233 | 1.651 | 1.385 | 1.040 |
| fixed depth 4 | 0.926 | 1.953 | 1.625 | 1.433 | 1.128 |
| fixed depth 2 | 1.001 | 1.519 | 1.388 | 1.308 | 1.127 |
| plan, frozen priors (`plan7`; longctx 1.567, tool 2.323) | 0.883 | 2.238 | 1.644 | 1.402 | 1.096 |
| **plan, learnt priors on a fresh server (`planL`)** | 0.984 | 2.239 | 1.659 | 1.427 | 1.105 |
| curve, learnt priors (`curveL`) | 0.978 | 2.190 | 1.675 | 1.403 | 1.109 |
| fixed 7 with the unfolded Markov head (`markovc`) | 0.705 | 1.955 | 1.454 | 1.198 | 0.949 |

The best at one stream by the median of the three kinds that matter is fixed depth 4 (1.433) but it gives up 13 % on code; the learnt plan keeps depth 7
where acceptance is high and shortens or stops elsewhere (median 1.427 on a fresh server, 1.404 behind other traffic), so it is the default and no fixed
depth is shipped. The learnt priors are the server's traffic: chat is 0.98 x when the server has seen prose first and 0.90 x behind code and JSON.

## Route log (`routelog.txt`, `inkling2_loop_routelog`: a prompt plus the model's own 640-token reply, one prefill chunk, the router's picks of every visit)
| text | consecutive-token expert overlap (random 0.0625) | distinct experts a visit in 8 consecutive tokens (1 token: 8; random 51.6) | expert-weight read ratio, 8 tokens / 1 token | core-layer overlap across loop passes | share of picks on 5-bit experts |
|---|---:|---:|---:|---:|---:|
| chat | 0.435 | 27.97 | 3.46 | 0.537 | 0.474 |
| code | 0.415 | 29.36 | 3.65 | 0.552 | 0.278 |
| reasoning | 0.432 | 28.12 | 3.44 | 0.553 | 0.538 |
| json | 0.347 | 32.97 | 4.08 | 0.479 | 0.408 |
Eight rows read 3.4-4.1 x the expert bytes of one and cost 1.54 x the time: the 1-row step is not bound by the expert reads alone (launch chains,
attention and glue), which is why an 8-row verify is cheap enough for the drafter to pay.

## Unexplained / not proven
* At four streams the serial round is 40 ms end to end (99.9 tok/s) against 29.0 ms of rank time for four rows (`timing.txt`): about 11 ms a round of
  engine-side time that this record does not attribute (one stream: 19.15 against 19.05 ms).
* Timing contexts are cycled ids of a synthetic list (a shape, not the served mix); the speedups use real prompts.
* The kinds are 512-token samples of 6 prompts; the learnt budget's speed depends on the traffic before it.

## Erratum (2026-11-23, lead, after `docs/C4-DECODE-PLAN.md`)
The "about 11 ms a round of engine-side time" above and the 29.4 ms window come from `timing.txt`, whose ids are a synthetic cycled list that touches fewer
distinct experts than real text: most of that difference is expert bytes, not host time (the real host residual is 2.8 ms at c=1 and 7.1 ms at c=4; a real
8-row window is about 38 ms). The measured speed-ups of this record are served measurements and stand.

## Commands
```sh
SPEC=dspark ./inkling2mini.sh start     # EXTRA_ENV="REXL3_INKLING2_LOOP_SPEC_POLICY=fixed|plan|curve REXL3_INKLING2_LOOP_SPEC_LEARN=1 REXL3_INKLING2_LOOP_SPEC_TOKENS=2|4 REXL3_INKLING2_LOOP_MARKOV=c" for the rows above
python3 bench/tools/specbench.py --out specbench.json --label LABEL --concurrency 4 [--kinds chat,code,json,reasoning]
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model OPS_GREP=LOOP_TIMING TEST_ENV='LOOP_TIMING=1 REXL3_INKLING2_LOOP_DIR=<pack> LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_spec_timing round_cost_table
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model OPS_GREP=ROUTE_ TEST_ENV='LOOP_ROUTE=1 REXL3_INKLING2_LOOP_DIR=<pack> LOOP_ROUTE_IDS_FILES=a.json,b.json,c.json,d.json LOOP_ROUTE_LABELS=chat,code,reasoning,json' scripts/ops_dev.sh test inkling2_loop_routelog route_log
```
