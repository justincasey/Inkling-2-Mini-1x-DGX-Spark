# graph-decode: decode of the Mini on one Spark with graphs and overlap (gate `speed-graph`)

Engine `9fa93ea6e8f1671ec618ac1ee91ac42ee13749a6` (branch `inkling2-loop-graph`, lab build), node-2, TP1, `bootstrap` profile (262,144 x 4 slots),
`--spec none --prefix-cache off --no-vision`, `REXL3_ALLOW_UNGATED_COMPONENTS=1`, **graphs and side streams on (the defaults)**, GPU clock locked at
2,200 MHz. Baseline: `2026-11-23-graph-baseline` (engine `51f2993`, same node, same tools). Greedy; the token count is `usage.completion_tokens`;
the first pass of each tool is a discarded warm-up; medians of 3. Every output byte equals the baseline's: `2026-11-23-graph-exact`.

## Result

| Measurement | Baseline `51f2993` | Now `9fa93ea` | Command |
|---|---:|---:|---|
| decode c = 1, short prompts, median of 3 | 37.232 tok/s | **53.386** (53.369, 53.386, 53.393) | `bench/rexl3/sweep_decode.py --phase both --ctxs 4096 --conc 1,2,4 --modes serial --depth-modes serial --reps 3 --depth-reps 3` after `./inkling2mini.sh bench decode` (the warm-up) |
| decode c = 1, a 4,096-token prompt, median of 3 | 36.158 | **51.318** (51.182, 51.318, 51.341) = 19.5 ms a token | same |
| aggregate c = 2 / c = 4, short prompts | 56.246 / 80.351 | **75.358 / 100.195** | same |
| `bench decode` (4 prompt types x 512 tokens, mean) | 37.238 | 53.394 (prose 53.08, code 53.86, structured 53.47, json 53.18) | `./inkling2mini.sh bench decode` |
| aggregate c = 4 / c = 8 (`bench decode`) | 80.407 / 83.623 | 100.385 / 103.708 | same |
| decode after a cold 32K prompt (one pass) | 33.46 tok/s | 46.29 | `bench.py --phases warm,decode,conc,prefill-32k` (`raw/bench-warm-decode-conc-prefill32k.out`) |
| cold 32K prefill (one pass; **not a goal**) | 27.25 s (1,203 tok/s) | 14.47 s (2,265 tok/s) | same |
| nodes of a decode window | 2,830 launches | 2,883 nodes in one graph | `event=inkling2_loop_graph_captured nodes=` |

Serial decode is **x1.43 at short context and x1.42 at 4K**, the c = 4 aggregate x1.25. Graph counters of the server after the whole run
(`raw/graph-metrics-after-bench.txt`, the server's `/metrics`): 13,747 replays, 10 captures (0.105 s of capture time), 10 eager windows (each
shape's first sighting), 0 capture failures, 0 evictions. The 32K prefill got faster because the bias tables (see the table below) were the
prefill prologue's bottleneck too; that is a side effect (measured once, not isolated), prefill speed was not the assignment.

Not measured: c = 2 and c = 4 with 4K prompts (the depth tool streams `return_token_ids`, which the engine refuses: HTTP 400), 1M, drafted
windows (no drafter in this configuration), memory at ready (`memwatch`) and the device memory of the 10 graphs and the lane scratch: the
rank's own buffers grew by 8.2 MB in the test rank (slots 2, prefill 64, window 8: 481,099,924 B to 489,257,116 B), the production figure was not
measured (it sits in the plan's 5 GiB reserve, `docs/FIT.md`).

## What each change did

Step time = one window of one row at 3,800 positions of context, median of 100 (`crates/model/tests/inkling2_loop_graphs.rs::decode_step_timing`
through `ops_dev.sh`, same node, same pack; the whole window including the upload, the wait and the download). Server tok/s only where a server was run.

| # | Change (engine commit) | Step, graph replay | Step, eager | Server c = 1 short / 4K tok/s | Verdict |
|---|---|---:|---:|---:|---|
| 0 | the starting engine (`51f2993`; eager equals `1f3d8d9` with graphs off) | - | 29,136 us | 37.23 / 36.16 | |
| 1 | window graphs (`1f3d8d9`, `d8a6915`) | 26,734 us | 29,136 us | 40.90 / 39.61 (`d8a6915`; same build graphs off: 37.18 / 36.13) | kept: -2.4 ms, the host enqueue (16.1 ms) was hidden behind a 25.1 ms device time |
| 2 | split top-k, 2 kernels over blocks of 2,048 columns (`52055ce`) | 23,483 us | 25,838 us | - | kept: -3.25 ms (the single-block scan took 3.1 ms) |
| 3 | MoE bands on side streams (`7465607`) | 22,444 us | - | - | kept: -1.04 ms |
| 4 | K and V input branches on two lanes (`3df40c1`) | 20,429 us | 23,545 us | - | kept: -2.0 ms |
| 5 | gate on lane 0, sink on lane 1 (not committed) | 20,427 us | - | - | **non-gain 1**, reverted |
| 6 | PDL on the dense EXL3 launches (`REXL3_GLM_PDL` 13 and 15, env only) | 20,469 / 20,495 us | - | - | **non-gain 2** (before the lanes mask 13 saved 0.18 ms: 23,490 to 23,309 us), left off |
| 7 | the relative-bias tables in their own kernel (`9fa93ea`) | 19,179 us | - | 53.39 / 51.32 | kept: -1.25 ms; skipping the table's writes altogether had measured -1.42 ms, so this took 88% of the possible |
| final | `9fa93ea`, three runs of the test | **19,137 us** | 22,064 us | | |
| | the same with `REXL3_INKLING2_LOOP_OVERLAP=0` | 22,018 us | 24,409 us | | the lanes are worth 2.9 ms |

Stopped directions: the gate/sink placement and the PDL mask each gave no measured gain twice (5, 6; the PDL one gave 0.2 ms without lanes and
nothing with them), so neither is pursued. Not tried: fusing the controller / mix / router chain, a faster `select`, the prologue's remaining
5 us, overlapping the attention's three kernels, MoE at more than 80% of the read bandwidth.

## Where the 19.5 ms go now, against `docs/ROOFLINE.md`

`profile/nsys-4-*`: `nsys profile -t cuda --cuda-graph-trace=node` of 20 replayed windows at 3.8K of context (`tools/nsys_run.sh`, `tools/prof_an.py`; 2,886
kernels a window; the profile's own window is 20.5 ms; durations of kernels launched with the programmatic-dependent-launch attribute include their wait for
the predecessor, so read spans, not shares). One core visit on the timeline (`profile/nsys-4-timeline-one-visit-9fa93ea.txt`), microseconds:

| Phase of a visit | Now | Bandwidth floor of its bytes at 237 GB/s |
|---|---:|---:|
| controllers q/k/v, projections q, k, v, r, gate, sink on three lanes | about 115 (37 for the controllers' chain, 78 for the projections) | about 62 (14.7 MB) |
| prologue 5, bias 6, attention + merge 28 | 34 | about 18 (the 1,024-key ring, 4.2 MB) |
| `wo` | 26 | 22 |
| conv, mix, MLP controller, router logits, select, remap | about 35 | - |
| MoE: 1 to 3 bands side by side, the longest band sets the time (the gate/up GEMV about 190 GB/s) | 160-190 | 142 (33.8 MB) |
| per visit | about 370-440 | 228 |

Arithmetic only. The serial ceiling at 4K with int8 KV is 70.1 tok/s (14.27 ms; ROOFLINE) and the proposed target 62 tok/s (16.13 ms): the
engine is at 51.32 tok/s (19.49 ms), **73% of the ceiling and 83% of the target**; the gap is 5.2 ms to the ceiling and 3.4 ms to the target. By
the table above, per-visit sums x 48 (extrapolated from one visit's timeline: **INFERENCE**, not measured visit by visit), the gap is made of: the MoE
GEMVs at 80% of the read bandwidth (about 1.0-2.3 ms), the controller / mix / router chain of about a dozen dependent 2-10 us kernels
(about 1.7 ms), the attention at twice its KV bound (about 0.8 ms), the projections at 80% of theirs (about 0.8 ms), the unembed 1.48 ms against 1.40.
Nothing here is the host: a replay costs the host 5 us.

## Switches and how to reproduce

- `REXL3_INKLING2_LOOP_GRAPHS=0` (off, false, no): every window eager, the same launches. `REXL3_INKLING2_LOOP_OVERLAP=0`: no side streams.
  `REXL3_INKLING2_LOOP_TIMING=1`: a line every 64 windows with the host microseconds of the uploads, the enqueue and the wait.
  `REXL3_INKLING2_LOOP_GRAPH_AT=0`: capture at the first sighting. Both defaults are on; see `docs/DECODE-SPEED.md`.
- Server: `EXTRA_ENV='REXL3_ALLOW_UNGATED_COMPONENTS=1' ./inkling2mini.sh start`, then the commands in the table; `tools/measure.sh` and `tools/live.sh` are the
  two drivers used (addresses replaced by node-2); `tools/nsys_run.sh` profiles the lab test binary under the GPU lock.
