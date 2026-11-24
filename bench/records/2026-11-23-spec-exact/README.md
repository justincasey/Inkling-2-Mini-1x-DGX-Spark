# spec-exact: a drafted reply equals the serial one (gate `spec`, plan 4.6 / 6.4)

Engine `761b14388d12d0cecf3a0186876f6952c35a9c7c` (branch `inkling2-loop-draftwire`, on the integration head `918e685`), node-2, GPU locked at
2,200 MHz, TP1, `bootstrap` profile (4 x 262,144), `VISION=off`, **no `REXL3_ALLOW_UNGATED_COMPONENTS`** (`dspark_inkling2_loop` is in
`BUILT_COMPONENTS`; the family boots without the override). Every request is greedy. The drafter proposes; the target's verify window decides
every committed token. "Equal" = the same token ids (the EOS of a stop included), the same finish reason and, for stop strings, the same text.
The two GPU test binaries below ran on the lab tree of commit `437ca4b` (761b143 plus one test-only fix).

| # | Property | Evidence (`raw/`) | Result |
|---|---|---|---|
| 1 | short (16), medium (521), long (2,030), xlong (8,334) and a 73,154-token prompt: drafted == serial; the four first ones at once == their solo serial | `spec-dspark.out/json` (`gates.py spec`): 9 of 9 equal; drafted 1,708, accepted 742, 401 rounds | pass |
| 2 | a `SPEC=none` server's replies == the `SPEC=dspark` server's, byte for byte (two boots) | `spec-serial.json`, `spec-diff.out`: `identical` | pass |
| 3 | the same 4 prompts' 128-token replies and top-5 prompt logprobs: serial boot == drafted boot == the engine before the drafter (`graph-exact`) | `exactness-*.json`; `replies` sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` in all three; `exactness-diff.out` identical; `exactness.py concurrent`: pass on both | pass |
| 4 | EOS inside a block (6 short-answer chats), stop string inside a block (10 cases), `max_tokens` 1-10, 15, 17 on two prompts (24), prompts of 1,016 / 1,023 / 1,024 / 1,030 / 1,031 / 1,032 / 1,040 / 2,060 tokens crossing the 1,024-window and 1,031/1,032 rings (8), a 12,345-token prompt on the 4,096 grid (1), four concurrent mixed requests x 3 rounds (12), four prompt kinds for the rollback histogram (4) | `specedge.out/json` (`bench/tools/specedge.py`): 65 cases, 0 failures. The rounds' accepted-count histogram over 911 rounds: `[285, 179, 106, 74, 66, 31, 26, 144]`: fully rejected blocks (285) and fully accepted ones (144) both happened, so rollback after a fully rejected block was exercised | pass |
| 5 | `SPEC=copy` (the copy drafter alone) | `spec-copy.out`: 9 of 9 equal | pass |
| 6 | taps: the four planes of every row equal the oracle dump's `hidden[5]`, `hidden[23]`, `hidden[41]` and `h_final` byte for byte (700 rows x 4 x 2,304 elements, 0 mismatched), whole or cut at 311 | `gpu-spec-tests.txt` `LOOP_SPEC_tap*`, `LOOP_SPEC_chunks` | pass |
| 7 | a rank with the drafter computes the same window bits as one without (logits and top-64, 8-row and 1-row windows, 6 runs); graph replay == eager with taps on (taps, logits, candidates; 6 windows, 4 captures, 4 replays) | `LOOP_SPEC_no_drafter_bits`, `LOOP_SPEC_graph_taps` | pass |
| 8 | the drafter's length follows prefill, commits of 1, 3, 8 and a rollback; its proposals do not depend on chunking or on the batch; a drafter fed by window taps proposes what one fed by a prefill of the same ids does | `LOOP_SPEC_follow`, `LOOP_SPEC_propose`, `LOOP_SPEC_replay_vs_prefill_context` | pass |
| 9 | host: the round bookkeeping over a scripted mock rank (fully rejected, partly and fully accepted blocks, EOS, stop token and `max_tokens` inside a block, four concurrent streams, a stream that ends early): 11 tests; the family's host tests 91; the server's stop-string-inside-a-round tests 5 | `host-tests-model.txt`, `host-tests-server.txt` | pass |
| 10 | **the serial path is byte-identical to the engine before the drafter**: `forward_oracle_gate` dump 90,056,120 B sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0` twice, chunk-width logits `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a`, long-context top-1 157 / 23.62496, the 37 graph windows' logits `6d0a555b92e02953c9e26efecdd9dece0ab211030ba8947cbe03b8407846c06b` (the hashes of `graph-exact`) | `serial-rank.txt`, `serial-graphs.txt` | pass |

A defect the gate found and fixed: a stop string completed in the middle of a drafted round (the round commits up to 8 tokens) still reported the
whole round's ids and token count (text and finish reason were right; the first `specedge` run failed 7 of its 10 stop cases by 1 to 5 trailing
ids). The server's reply task now pushes a round's tokens one at a time and ends the reply at the token that completes the stop string (engine
`6b81ff7`, tests `stop_in_batch`); the serial reply is unchanged. The first run's output is `first-run-before-stop-fix/`.

Not proven: images with a drafter (refused by name: `--spec dspark` with images), TP, the oracle (drafted == serial says nothing about the oracle),
the drafter's quality against the oracle's drafts on real fixtures (no oracle planes on node-2; the kernels' own record is `kernels-draft`).

## Commands
```sh
./inkling2mini.sh build <40-hex>   # REXL3_SRC=<engine worktree>, engine 761b14388d12d0cecf3a0186876f6952c35a9c7c
SPEC=dspark PREFIX_CACHE=off VISION=off ./inkling2mini.sh start   # EXTRA_ENV empty: no ungated component
python3 bench/tools/gates.py spec spec-dspark.json --max-tokens 128; same with SPEC=none --serial-boot; gates.py spec-diff
python3 bench/tools/specedge.py --out specedge.json
python3 bench/tools/exactness.py record OUT.json --max-tokens 128; exactness.py diff A B; exactness.py concurrent --max-tokens 64
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='REXL3_INKLING2_LOOP_DIR=<pack> LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_spec taps_equal no_drafter graph_replay the_drafter_follows draft_equals
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_rank forward_oracle_gate long_context_gate
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_graphs graph_replay_equals capture_refusal
cargo test -p gb-model --lib families::inkling2_loop; cargo test -p gb-server --test stop_in_batch
```
