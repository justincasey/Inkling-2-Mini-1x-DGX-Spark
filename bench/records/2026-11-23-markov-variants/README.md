# markov-variants: the quantized Markov head, fold (a) against unfolded (c) (gate `markov-variants`, UNKNOWN-17)

(a) is what the engine ships: `fold_W = bf16(U . hidden_out)` and `fold_b` made once at load through the EXL3 launches (a 103 MB bf16 read a step).
(c) is a lab switch (`REXL3_INKLING2_LOOP_MARKOV=c`, engine `45d83f8`): each of the 7 steps applies `hidden_out` (EXL3, bias added in fp32, one bf16 rounding) and then
the target's EXL3 unembed, seven more unembed launches a round. Variant (b) (the fold without the bf16 store) was not built.

**Live, against the target's greedy tokens** (`specbench.py`, 512 tokens, fixed depth 7, same prompts, the per-position counters of `/metrics`; `fixed7/` and `markovc/`,
both on `dfb05bb`): (a) and (c) offer and have accepted exactly the same drafts at every position, so (c)'s acceptance gain is zero:

| kind | accepted / offered, (a) | accepted / offered, (c) | speedup (a) | speedup (c) |
|---|---|---|---:|---:|
| chat | 753 / 5,397 | 753 / 5,397 | 0.816 | 0.705 |
| code | 1,266 / 1,854 | 1,266 / 1,854 | 2.233 | 1.955 |
| json | 1,191 / 2,370 | 1,191 / 2,370 | 1.651 | 1.454 |
| reasoning | 1,074 / 3,153 | 1,074 / 3,153 | 1.385 | 1.198 |
| c = 4 aggregate | | | 1.040 | 0.949 |

The per-position conditional acceptance tables of the two runs are identical to the printed digit. Round time (`markov-synthetic.txt`, `inkling2_dspark` tests, one request at 1,300 positions, eager medians of 20): (a) 7.862 ms, (c) 15.304 ms, +7.443 ms a
round (the derived estimate was about 10 ms); four requests 9.951 against 17.292 ms. On synthetic features (a) and (c) pick the same draft at all 7 positions of 7 anchors; the largest logit difference is 0.125;
(c) is deterministic and graph-safe on both plane formats. **Decision: keep (a)**: (c) buys no acceptance and costs 5 to 25 % of the end-to-end speed.

Not done: the oracle fixtures' comparison against the oracle's three variants (no oracle planes on node-2); only the live target tokens decide here.

## Commands
```sh
EXTRA_ENV="REXL3_INKLING2_LOOP_SPEC_POLICY=fixed [REXL3_INKLING2_LOOP_MARKOV=c]" ./inkling2mini.sh start; python3 bench/tools/specbench.py --kinds chat,code,json,reasoning --concurrency 4 --out OUT.json
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-cuda OPS_GREP=MARKOV_VARIANT scripts/ops_dev.sh test inkling2_dspark mini_markov_variant_c mini_golden_hashes mini_graph_replay_equals_eager
```
