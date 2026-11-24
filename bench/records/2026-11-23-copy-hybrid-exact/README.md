# copy-hybrid-exact: the per-stream copy / block drafter choice changes no byte (gate `spec`)

Engine `efcb6ed25153766f1d35b664cdbec662f06eb6ee` (the commit of `2026-11-23-copy-hybrid`), node-4, TP1, shipping defaults `SPEC=dspark PREFIX_CACHE=on VISION=on`, the copy hybrid at the engine's default (on), greedy, one fresh server for the battery (`tools/run_exact.sh`, `raw/`).

| # | Claim | Evidence |
|---|---|---|
| 1 | the engine answers byte for byte as the pinned engine | `exactness.py record` then `diff` against `2026-11-23-graph-exact/raw/graph-9fa93ea/exact.json`: `identical: 4 replies`; the `replies` part has sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` (`raw/exact-record.out`, `raw/exact-sha.txt`) |
| 2 | `bench.py --phases warm,decode,conc,prefill-32k --ref` the pinned replies of `51f2993` | `PASS pin` (`raw/bench-ref.out`) |
| 3 | concurrent == solo (ids and prompt logprobs) | `exactness.py concurrent --max-tokens 64`: `"pass": true` (`raw/concurrent.out`) |
| 4 | drafted == serial, four concurrent included | `gates.py spec --max-tokens 128`: pass, 1,537 drafted, 727 accepted, 416 rounds (`raw/gates-spec.out`) |
| 5 | cached == uncached | `gates.py cache --max-tokens 64`: pass (`raw/gates-cache.out`) |
| 6 | the edge cases of a block and, new, of a copy draft | `specedge.py`: **125 cases, 0 failures, 1 warning** (`raw/specedge.out`): the 65 of before and the new property `copy_hybrid` (60 cases: four prompts (structured, json, code, medium) in full at 320 tokens, `max_tokens` 1 to 8, and stop strings picked from tokens 60-140, 140-220 and 220-300 of the reply, each drafted == serial, ids, finish reason and text); the `/metrics` copy counters around every drafted request: 79 copy rounds in that property (412 in the battery), **rejected at position 1: 8 rounds, in the middle (positions 2 to 6): 12, 12, 3, 5, 0 (positions 2 to 6)**, 12 stop-string requests ran a copy round and ended on the stop string |
| 7 | host tests | `REXL3_NO_CUDA=1 cargo test -p gb-engine -p gb-spec -p gb-sched --locked`: all pass, the new `copy_hybrid` (6: every mix of copy and block drafts equals serial at 1, 3 and 5 streams, chained and not, three block-draft prices; sampled streams equal the run without the hybrid; a stop token at every position of a reply; the length limit at every offset 1 to 39; the choice follows the block drafter's price), `gb-sched` `free_drafts_do_not_pay_the_draft_round`; `gb-server` `the_copy_hybrids_rounds_have_series_of_their_own`; `-p gb-model --lib families::inkling2`: 146 passed |

Not proven: **a 7-token copy draft rejected at its LAST (7th) token was not seen** in the live run (warning of `specedge.py`: `no 7-token copy draft was rejected at its last token`; the counters show 0 at position 7); first-token and middle rejections were (8 and 12+12+3+5). The engine-level equality tests
(`crates/engine/tests/copy_hybrid.rs`) reject at positions the mock model decides and pass for every mix, but they do not pin the last position either. The kernel batteries and the rank gates were not rerun: this commit changes no kernel, no rank code and nothing under `crates/cuda` (the one backend hunk lets
`chained_round` accept requests that carry host drafts when they ask for none from the drafter). `gates.py vision`, 1M context, TP > 1 and sampled streams over HTTP were not run (sampled streams are covered by the host tests: the drafts are verified against the keyed sample).

## Commands

```sh
./inkling2mini.sh build efcb6ed25153766f1d35b664cdbec662f06eb6ee && bash tools/run_exact.sh raw     # NODES=<node-4>
```
