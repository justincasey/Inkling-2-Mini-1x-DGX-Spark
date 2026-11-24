# graph-exact: the decode speed work changed no byte (gate `exact-graph`)

Engine `9fa93ea6e8f1671ec618ac1ee91ac42ee13749a6` (branch `inkling2-loop-graph`, seven commits on the integration head `51f2993`), node-2, TP1,
GPU clock locked at 2,200 MHz. Everything below is greedy. "Before" is the integration head `51f2993c1549f927a632c2c502de3d20dc261ec3`,
built and booted for this record (`raw/pinned-replies-engine-51f2993.json` was written by that engine).

## What was proved, and how

| # | Claim | Evidence (raw files under `raw/`) |
|---|---|---|
| 1 | Live: a graph server (graphs and side streams on, the default) answers byte for byte as the starting engine | `bench/tools/bench.py --phases warm,decode,conc,prefill-32k --pin` on `51f2993` pinned 5 replies (4 prompt types x 512 tokens, one 32K-token prefill + 128 tokens; each concurrent reply at c = 4 and c = 8 equal to its solo reply: `raw/base-51f2993/bench.out`); the same command with `--ref` on `9fa93ea`: `PASS pin` (`raw/graph-9fa93ea/bench.out`). Pinned file sha256 `5ca5bab6a2ba8540933cb6d8c2f8bc11d73a58199b6eec04c3c307198a2fbdec` |
| 2 | Live: a graph-off, overlap-off server (`REXL3_INKLING2_LOOP_GRAPHS=0 REXL3_INKLING2_LOOP_OVERLAP=0`) does too | same `--ref` run on `9fa93ea` with the two switches: `PASS pin` (`raw/eager-9fa93ea/bench.out`) |
| 3 | Live: 4 prompts of about 30, 400, 1,500 and 6,000 tokens, 128 greedy tokens and the prompt-logprobs top-5 (ids and values), solo, are identical on the three engines | `bench/tools/exactness.py record` then `diff` against the starting engine: "identical: 4 replies" for `9fa93ea` graph, `9fa93ea` eager and (before the bias kernel) `3df40c1` graph and eager. The `replies` part of all five records has the same sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` (canonical JSON, sorted keys, compact separators; the files differ only in their `meta`: label, address, timings) |
| 4 | Live: concurrent equals solo on the graph server (4 different prompts alone, then at once; ids and logprobs) | `exactness.py concurrent --max-tokens 64`: `"pass": true` (`raw/graph-9fa93ea/concurrent.out`, and on the other four engines) |
| 5 | The rank's own gates are byte-identical before and after, graphs on and off | `inkling2_loop_rank` `forward_oracle_gate` (prompt "sky", 26 tokens): the dump of 90,056,120 bytes has sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0` twice (runs a and b) with graphs on and with `REXL3_INKLING2_LOOP_GRAPHS=0`, the same hash as `2026-11-23-forward-oracle` (engine `aeaaa4f`, before the graph work); chunk widths 7 and 11: logits 20,806,032 bytes, sha256 `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a` (also the record's); decode rows 22-25 equal the prefill's rows; the 3-row window committed at 2 and continued equal. `long_context_gate` (1,119 synthetic ids, `ids_long.json` sha256 `e2d8af93f90f54bbcff57c75be94580f0ad721196f0621e2b09bf469b65c0cf6`, the first run's list is not kept): top-1 157, logit 23.62496, chunks of 129 and 300 and the decode row equal, at `d8a6915` (graphs off, the old top-k and bias) and at `9fa93ea` with graphs on and off (`raw/rank-gates-d8a6915.log`, `raw/gates-9fa93ea.log`) |
| 6 | A replayed graph, an overlapped eager window and a sequential eager window agree bit for bit on every shape | `inkling2_loop_graphs::graph_replay_equals_eager`: 37 windows (1-row steps across the 511/512 and 1,023/1,024 chunk boundaries, 8-row windows committed at 5, four slots of one row, a 3+2 window; a slot concurrent equals the same slot alone), each run sequential-eager, overlapped-eager and 3 x graph, logits of every row and top-64 candidates. The sha256 of the sequential-eager logits of all 37 windows is `6d0a555b92e02953c9e26efecdd9dece0ab211030ba8947cbe03b8407846c06b` at `d8a6915`, `52055ce`, `7465607`, `3df40c1` and `9fa93ea`: the split top-k, the side streams and the bias kernel each left the bytes of the original eager path as they were |
| 7 | A refused capture runs the shape eagerly and is counted | `capture_refusal_runs_eager_and_counts`: a host sync inside the capture (CUDA error 900): the window still answers equal, `capture_failures` 1, the shape never captures again, another shape captures and replays |
| 8 | The kernels changed are bit-exact on their own | `inkling2_ops`: `split_topk_equals_the_local_scan_and_merge` (tie-heavy values, NaN, -inf, both zeros, padding columns, 1-8 rows, k 1-64, a shard narrower than k), `sharded_argmax...`, `unembed_softcap...`, the golden hashes; `inkling2_attn` `mini::` 14 passed and `small::` 14 passed (each with its golden hash file, so the bias kernel's tables equal the prologue's); `inkling2_moe` golden and graph, `inkling2_moe_mixed` graph |
| 9 | Host tests | `REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked`: `families::inkling2::` 31 passed (the Small's, unchanged), `families::inkling2_loop::` 61 passed, `window_graph::` 6 passed |

`raw/gates-9fa93ea.log` also holds two lines that are not failures of the engine: a `decode_step_timing` run whose parameters did not fit the test rank (an assertion on the test's own arithmetic; re-run in `2026-11-23-graph-decode`) and a `golden` filter that matches no test of `inkling2_moe_mixed` (its `graph` test passed).

Not proved: the graph path at 1M of context, drafted windows (the drafter is off), images (vision is off here), TP>1. The family is not
oracle-qualified (`REXL3_ALLOW_UNGATED_COMPONENTS=1`): "equal" here means equal to the engine as it was, not correct against the oracle.

## Commands (node-2 stands for the node's address)

```sh
REXL3_SRC=<engine worktree> ./inkling2mini.sh build <40-hex>          # 51f2993c1549f927a632c2c502de3d20dc261ec3, then 9fa93ea6e8f1671ec618ac1ee91ac42ee13749a6
EXTRA_ENV="REXL3_ALLOW_UNGATED_COMPONENTS=1 [REXL3_INKLING2_LOOP_GRAPHS=0 REXL3_INKLING2_LOOP_OVERLAP=0]" ./inkling2mini.sh start
python3 bench/tools/bench.py --url http://node-2:8000 --model inkling-2-mini-1x-dgx-spark --max-model-len 262144 \
  --phases warm,decode,conc,prefill-32k --pin pinned.json      # on 51f2993;  --ref pinned.json  on 9fa93ea
python3 bench/tools/exactness.py record OUT.json --max-tokens 128 --label LABEL;  exactness.py diff base.json OUT.json;  exactness.py concurrent --max-tokens 64
./inkling2mini.sh stop
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='LOOP_LONG_IDS_FILE=<lab>/ids_long.json [REXL3_INKLING2_LOOP_GRAPHS=0]' scripts/ops_dev.sh test inkling2_loop_rank forward_oracle_gate long_context_gate
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_graphs graph_replay_equals capture_refusal
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-cuda scripts/ops_dev.sh test inkling2_ops split_topk sharded_argmax unembed_softcap golden
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-cuda scripts/ops_dev.sh test inkling2_attn "mini::" "small::"
```
