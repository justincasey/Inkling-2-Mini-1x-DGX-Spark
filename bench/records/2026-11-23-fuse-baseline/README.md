# fuse-baseline: serial decode, the windows and the proposal of engine `1d53344` before the fuse lane (gate `speed-fuse-baseline`)

Engine `1d53344d874b3a578152e35c500a2718599f5b61` (the integration head `inkling2-loop`, lab build), node-2, TP1, `bootstrap` profile, GPU clock locked at 2,200 MHz
(`./inkling2mini.sh gpu-clocks lock`), graphs and side streams on (defaults). Greedy; token counts are `usage.completion_tokens`; the first pass of every tool is a discarded warm-up.

## Result

| Measurement | Value | Command |
|---|---:|---|
| serial decode c = 1, short prompts, median of 3 | **53.64 tok/s** (53.64, 53.64, 53.63) | `SPEC=none PREFIX_CACHE=off VISION=off ./inkling2mini.sh start`, `./inkling2mini.sh bench decode` (warm-up), then `bench/rexl3/sweep_decode.py --phase both --ctxs 4096 --conc 1,2,4 --modes serial --depth-modes serial --reps 3 --depth-reps 3` (`raw/sweep-decode-baseline.log`) |
| serial decode c = 1, a 4,096-token prompt, median of 3 | **51.55 tok/s** (51.39, 51.55, 51.55) | same |
| aggregate c = 2 / c = 4, short prompts | 75.71 / 100.73 tok/s | same (the depth tool's c = 2 and 4 runs are flagged as in `graph-decode`: it streams `return_token_ids`) |
| `bench decode` serial, mean of 4 kinds / c = 4 / c = 8 | 53.63 / 100.83 / 101.12 | `raw/bench-decode-serial-baseline.log` |
| one-row step, 3,800 positions, graph replay, median of 100 | **19,146 us** (rows 1 / 2 / 3 / 4 / 6 / 8: 19.12 / 22.53 / 24.83 / 26.72 / 30.29 / 32.99 ms) | `LOOP_PROFILE_STEPS=100 LOOP_PROFILE_ROWS=R LOOP_PROFILE_MODE=graph ops_dev.sh test inkling2_loop_graphs decode_step_timing` (`raw/rows-step-timing.txt`) |
| 8-row window / proposal at 4,096 positions (synthetic ids) | **29.25 ms / 8.45 ms** (4 streams x 8 rows 52.3; proposal at 2 / 4 requests 8.94 / 9.98) | `LOOP_TIMING=1 LOOP_TIMING_CTX=4096 ops_dev.sh test inkling2_loop_spec_timing round_cost_table` (`raw/round-cost-table-baseline.txt`) |
| drafted speed, shipping defaults (`SPEC=dspark PREFIX_CACHE=on VISION=on`), `bench decode` mean of 4 kinds, 3 runs | **81.65 tok/s** (81.65, 81.77, 81.65; warm-up 81.22) = prose 41.8, code 122.9, structured 73.6, json 88.3 | `raw/drafted-baseline-1d53344/` (`session.sh`: exactness record, `bench decode` x4, `specbench.py --concurrency 4`) |
| `specbench.py` speed-ups (median of 3) | chat 0.897, code 2.252, json 1.660, longctx 1.572, tool 2.372, reasoning 1.410, c = 4 aggregate 1.109 | `raw/drafted-baseline-1d53344/specbench.out` (reproduces `spec-final`: 0.901 / 2.239 / 1.648 / 1.558 / 2.368 / 1.404 / 1.106) |

## Profile

`profile/` holds the `nsys` tables of 20 replayed windows at 3,800 positions for 1, 2, 4 and 8 rows (`kernels-rows*.txt`: per-kernel table and stage sums), one window's
timeline (`timeline-1row-one-window.txt`) and the proposal's (`timeline-proposal-baseline-one-request.txt`, `proposal-kernels-baseline.txt`). The per-visit kernel table, the
rows table and the analysis are in `docs/DECODE-SPEED.md` ("Second pass"). Tools: `tools/nsys_run.sh NAME ROWS`, `prof_visit.py`, `prof_stages.py`, `nsys_propose.sh`, `prof_visit.py`'s
`timeline N` mode. Kernel durations of PDL launches include the wait for their predecessor.

## Not measured

A baseline of the proposal and the windows with real texts on `1d53344` itself (the real-text tables are in `fuse-decode`, measured on the final engine; the window path is identical),
the proposal at 32K on `1d53344` (`spec-speed` has 8.51 ms for an older engine), distinct experts per window (the byte figures in the docs come from `c4_model.py`).
