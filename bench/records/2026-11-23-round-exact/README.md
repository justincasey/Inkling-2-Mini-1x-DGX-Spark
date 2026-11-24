# round-exact: the chained round, the budget prices and the idle relax change no byte (gate `spec`)

Engine `5cc464e7b0faaa267ad42f9ee9bf68b9e036b972` (branch `inkling2-loop-round`; the same commit as `2026-11-23-round-overhead`), node-1, TP1, GPU at 2,200 MHz, greedy, shipping defaults
`SPEC=dspark PREFIX_CACHE=on VISION=on`, no ungated override. "Before" is the pinned replies of `2026-11-23-graph-exact` (engine `51f2993`) and the engine `1d53344`.
The proofs ran on one fresh server (`raw/*`, driver `../2026-11-23-round-overhead/tools/run_final.sh`) and as test runs on the node (`raw/gpu/*.log`, `tools` of the lead's `scripts/ops_dev.sh`).

| # | Claim | Evidence |
|---|---|---|
| 1 | drafted == serial with the chained round on (the default): short, medium, long, xlong, 73K solo and four concurrent | `gates.py spec --max-tokens 128`: 9 of 9 equal (`raw/gates-spec.out`; 1,479 drafted, 732 accepted, 411 rounds) |
| 2 | the prefix cache with drafts: cached == uncached | `gates.py cache --max-tokens 64`: 7 of 7 equal, cached tokens 469 to 73,190 (`raw/gates-cache.out`) |
| 3 | the edge cases of a block (EOS, stop string, max_tokens, ring wrap, chunking, full rejection, concurrent mix) | `specedge.py`: **65 cases, 0 failures, 0 warnings** (`raw/specedge.out`) |
| 4 | the engine answers byte for byte as the pinned engine | `exactness.py record` then `diff` against `graph-exact/raw/graph-9fa93ea/exact.json`: `identical: 4 replies`; the `replies` part has sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` (`raw/exact-record.out`, `raw/exact.json`) |
| 5 | `bench.py --phases warm,decode,conc,prefill-32k --ref` the pinned replies of `51f2993` (sha256 `5ca5bab6...fbdec`) | `PASS pin` (`raw/bench-ref.out`): each reply, and each concurrent reply at c = 4 and 8, equal |
| 6 | concurrent == solo (ids and prompt logprobs) | `exactness.py concurrent --max-tokens 64`: `"pass": true` (`raw/concurrent.out`) |
| 7 | drafted == serial on the speed runs themselves | `specbench.py`: six kinds and c = 4, every pass `equal True`; `homo_c4.py`: 4 x prose, code, structured, json and the mixed four, drafted ids equal serial ids; `specmix.py`: `PASS drafting follows the traffic` (`../2026-11-23-round-overhead/raw/final/`) |
| 8 | the rank's own gates | `inkling2_loop_rank forward_oracle_gate long_context_gate`: 2 passed (the test asserts the sky dump sha256 `c25f9876...c45c0` and logits `a105f245...f41a`, the long context against its own prefill); `inkling2_loop_graphs graph_replay_equals capture_refusal`: 2 passed; `inkling2_loop_prefix`: 7 passed (the cuts); `inkling2_loop_spec`: 9 passed (`raw/gpu/`) |
| 9 | the 37 graph windows | `LOOP_GRAPH_equal windows 37 logits_sha256 44294f72e147c7e0ccd317d3aa3d1dda2b27534b50189915abb2c065f6ad90e1` on engine `5cc464e` and on `1d53344`, the same ids file (`expert-arena-ids/ids_long.json`, sha256 `293794b8...2970`: not the file of the `6d0a555b...c06b` runs, which this node does not hold; equal before and after is the claim) |
| 10 | kernel batteries (no kernel of this lane changed; the head carries DecodeFuse's, which have their own record) | `inkling2_dspark` 28 passed (its golden hash file equal), `inkling2_ops` 29, `inkling2_moe` 8, `inkling2_moe_mixed` 13, `inkling2_attn` 28 (both golden hashes) (`raw/gpu/c-*.log`) |
| 11 | host tests | `REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked families::inkling2`: 146 passed (the Small's and the Mini's, the chained round's backend bookkeeping included); `gb-engine` test `chained_round` (a backend that chains equals the engine's own path in replies, rounds, drafted and accepted, solo and at three streams, and a backend that declines falls back) |

What this does not prove: the chained round with images (the drafter serves them through the same window; `gates.py vision` was not run), at 1M of context, at TP>1, and sampled (temperature above 0) streams (the chain is offered
only under the `Plan` budget; a sampled round takes it too, its draft counts being fixed before the draft: not exercised by a gate here). `inkling2_loop_stall`, `_vision`, `_routelog` and `_prefill_perf` were not run.
The pins of the recipe's `bench/reference.json` are the lead's.

## Commands (node-1 stands for the node's address)

```sh
./inkling2mini.sh build 5cc464e7b0faaa267ad42f9ee9bf68b9e036b972 && ./inkling2mini.sh start
python3 bench/tools/exactness.py record exact.json --max-tokens 128 --label final;  exactness.py diff <graph-exact>/raw/graph-9fa93ea/exact.json exact.json;  exactness.py concurrent --max-tokens 64
python3 bench/tools/bench.py --url http://node-1:8000 --model inkling-2-mini-1x-dgx-spark --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref <graph-exact>/raw/pinned-replies-engine-51f2993.json
python3 bench/tools/specedge.py --out specedge.json;  gates.py spec gates-spec.json --max-tokens 128;  gates.py cache gates-cache.json --max-tokens 64
NODE=node-1 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='LOOP_LONG_IDS_FILE=<ids file>' scripts/ops_dev.sh test inkling2_loop_rank forward_oracle_gate long_context_gate   # and the others of raw/gpu
```
