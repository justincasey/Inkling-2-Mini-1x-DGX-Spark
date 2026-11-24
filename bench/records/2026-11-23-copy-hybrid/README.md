# copy-hybrid: a per-stream choice between the block drafter and a copy of the stream's own text (gate `speed-decode`)

Engine `efcb6ed25153766f1d35b664cdbec662f06eb6ee` (branch `inkling2-loop-hybrid`, from `4eec4b5`), node-4, TP1, bootstrap profile, `SPEC=dspark PREFIX_CACHE=on VISION=on`, greedy, the GPU SM clock
held at 2,200 MHz. One binary, two fresh servers, run the same afternoon: **off** = `REXL3_INKLING2_LOOP_SPEC_COPY=off` (the block drafter alone: the engine of `4eec4b5` in behaviour) and **on** (the engine's default).
Byte-exactness of this commit: `2026-11-23-copy-hybrid-exact`. Driver: `tools/run_speed.sh` (specmix, `bench decode` x3, `bench rolling` x3, specbench with c = 4, one fresh boot per configuration).

## What it is

`Engine::with_copy_hybrid` (generic, off for every other family; the Mini turns it on): each round, for every stream that may draft, the host finds the longest earlier match of the stream's own text (`gb_spec::CopyDrafter`, index over the first 8 tokens
of a match) and keeps it when it is at least 16 tokens long. The round is then planned for every choice of which streams copy (2^n, at most 6 eligible streams): a copy draft is made on the host and costs the draft round nothing
(`gb_sched::plan_round_free`: its rows are verify rows, but they do not count as a drafting stream nor add a head row to the proposal's price), so a round in which every stream copies skips the block drafter's proposal (6.4 ms at one stream); the choice is the one with the best
expected tokens per millisecond of the whole round, taken over the block drafter alone only when it wins by 5 %. The copy's acceptance is learnt by sampling mode and match-length bucket (16-23, 24-31, 32-63, 64+) and per stream, from the copy rounds
and, free of cost, from the copy drafts the block drafter's rounds test (row j's sample is the target's own token after the first j drafts, so it tests the copy's token j for every j up to the block's accepted run). A copy round feeds none of the block drafter's
statistics. Every draft is verified against the target's own sample: replies are the same bytes (see the exactness record). A copy draft rides the window as the host holds it: the Mini's `chained_round` runs the window of a stream that is not asked with the drafts the host gave it.
"A combined draft" (the copy extending the block's) is not built: a window holds at most 8 rows a stream (`SLOT_WINDOW_ROWS`, the slot's state) and the block already fills 7 drafts when it can.

## Sizing before building (the numbers that decided it)

C4Plan's offline hybrid bound (`bench/tools/c4_model.py copy_gains`, reproduced here: structured +6.2 %, json +7.8 % tokens a round) assumed the block's accepted count independent of the copy's and an oracle choice. The real joint was not known, so a shadow
run measured it first (a build that computed the copy draft of every round and logged it beside what the block drafter did; `bench decode` on the server of that build: 312 rounds with a match of 8 or more): where the copy matched, it was right more often than the block drafter:
copy accepted / block accepted of 7 drafts by match length: structured 4.9 / 2.8 (8-11), 3.3 / 2.3 (12-15), 5.9 / 3.5 (16-31), 6.8 / 5.1 (32+); json 5.4 / 4.0, 2.9 / 3.7, 5.5 / 4.4, 5.3 / 5.0; code 2.8 / 5.5 and 1.1 / 3.4 (8-15 only). Replacing the rounds by the model's
window and proposal prices (18.7 ms + 3.15 a row, 6.7 ms) gave an oracle ceiling of +6 to +14 % on structured and +4 to +10 % on json, and +0 on code; a rule "copy when the match is at least L" +8 to +14 % (L = 8) and +9 % (L = 32) on structured, +4 to +9 % and +1 % on json, and a loss on code at L = 8.
The shadow build is not in the engine (the counters of the final one are in `/metrics`, `rexl3_spec_copy_*`). Real gain on mean4 was therefore expected above the +1.0 bar and was measured.

## Result (`bench decode`, median of three run means, fresh server per column, `raw/off`, `raw/on`)

| metric | off | **on** | change |
|---|---:|---:|---:|
| **`decode_mean4_tok_s`** (the mean of the four kinds, c = 1) | 83.92 (83.9, 83.8, 85.0) | **85.41** (85.4, 85.7, 85.4) | **+1.49** |
| prose | 49.67 | 49.65 | 0.0 |
| code | 117.75 | 119.83 | +1.8 % (the off runs spread 116.7-121.4: no loss) |
| structured | 79.51 | 82.42 | +3.7 % |
| json | 89.23 | 90.60 | +1.5 % |
| `bench decode` c = 4 aggregate | 107.49 | 108.88 | +1.3 % (spread 106-110: noise) |
| `decode_c4_rolling_tok_s` (`bench rolling`, 3 runs) | 134.42 (134.4, 134.3, 134.9) | 134.76 (134.8, 134.3, 135.1) | +0.3 (noise); steady 135.28 / 136.31 |
| specbench c = 4 aggregate (drafted) | 114.87 | 115.40 | |

specbench, one stream (`raw/*/specbench.err`; every drafted reply equals its serial reply): chat 52.82 / 53.03, code 117.22 / 116.30 (-0.8 %), json 85.33 / 86.49 (+1.4 %), longctx 65.95 / 65.14 (-1.2 %), tool 126.80 / 127.03, reasoning 76.43 / 76.83.
specmix on a fresh server (`chat, code, reasoning, json, chat`, 8 requests a phase, 256 tokens; `raw/*/specmix.out`, tok/s off / on): 55.16 / 55.52, 103.5 / 103.7, 83.09 / 83.55, 84.19 / **86.24** (+2.4 %), 55.45 / 55.59; tokens a round equal in every phase but json (4.039 / 3.977:
the copy rounds trade tokens a round for rounds without a proposal). `PASS drafting follows the traffic` on both. Prose and code do not lose by more than 1 % in any of these (code in specbench -0.8 %, the only negative number past noise).
Copy rounds in the on specbench and bench runs: 1,538 sequence-rounds in all (`raw/on/copy-metrics.txt`: per-position reached / accepted).

## What was tried on the way (kept as `raw/on-v1-min8`, `raw/off-v1`)

The first policy took a copy from 8 matched tokens: mean4 85.91 against 84.36 (the off runs of that session), structured +7.3 %, json +4.8 %, but **code -2.7 %** (117.99 against 121.34: the rounds of code with a match of 8 to 15 tokens were chosen on a prior pooled with
structured and json and the copy was right 1.1 to 2.8 of 7 times there). Its gain was larger and its loss on code is what the owner's rule forbids, so the shipped minimum is 16 tokens (this record), which costs 0.5 on mean4 and gives code back. A short sweep of the margin (1.15 to 1.5) and the stream's
prior weight (4 to 1.5) at min 8 changed no mean past the run-to-run spread (the first `bench decode` after a boot reads 2 to 4 % low on code in every configuration, including the off ones; numbers of that sweep are not in a record).

## What is not proven / risks

* Gain is on bench.py's four texts and specbench's six; the copy hybrid helps text that repeats itself (JSON, YAML, a document being edited or quoted). On free prose and on code with short matches it never chooses a copy (prose 49.67 against 49.65).
* The one-stream line of the window price (`round_cost`) prices a copy draft's rows like the block's; a copy round at c = 4 saves the proposal only when every stream copies (4 x structured/json is where it would show; not measured).
* The cold-start acceptances (0.90 to 0.97 a position by match length) come from the shadow run on these same four texts; the learnt numbers replace them after 12 tests of a position.
* The shipped minimum (16) and margin (5 %) were chosen once, after seeing the min-8 run; the off baseline of this record was measured after both, on the same binary.

## Commands (node-4 stands for the node's address)

```sh
NODES=<node-4> REXL3_SRC=<engine worktree> ./inkling2mini.sh build efcb6ed25153766f1d35b664cdbec662f06eb6ee
bash tools/run_speed.sh on  raw/on        # EXTRA_ENV=REXL3_INKLING2_LOOP_SPEC_COPY=on, then specmix, bench decode x3, bench rolling x3, specbench --concurrency 4
bash tools/run_speed.sh off raw/off       # REXL3_INKLING2_LOOP_SPEC_COPY=off
```
