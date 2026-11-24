# round-baseline: the drafted round of engine `1d53344`, where its milliseconds go (gate `speed-round-baseline`)

Engine `1d53344d874b3a578152e35c500a2718599f5b61` (the integration head), node-1, GPU locked at 2,200 MHz, TP1, bootstrap profile, shipping defaults `SPEC=dspark PREFIX_CACHE=on VISION=on`,
greedy. Raw outputs under `raw/`. The round trace and the nsys timeline were taken on engine `579aa8a` (`1d53344` plus `REXL3_ROUND_TRACE`, which adds only inert marks when off).

## The number: `bench decode`, the four kinds, c = 1, 512 tokens (`./inkling2mini.sh bench decode`, three runs, each after the phase `warm`)

| kind | run 1 | run 2 | run 3 | median tok/s | tokens/round (median) |
|---|---:|---:|---:|---:|---:|
| prose | 44.764 | 41.727 | 41.836 | 41.836 | 1.55 |
| code | 120.002 | 123.142 | 122.610 | 122.610 | 5.75 |
| structured | 73.763 | 73.374 | 73.237 | 73.374 | 3.35 |
| json | 88.000 | 88.418 | 88.177 | 88.177 | 4.45 |
| **mean of the four (`decode_c1_tok_s`)** | 81.632 | 81.665 | 81.465 | **81.632** (median of the run means; mean of the medians 81.50) | |

c = 4 aggregate 100.56 / 101.38 / 103.00 (median 101.38); c = 8 124.4 / 122.7 / 113.0. The note of the task (about 93, derived from `spec-final` speed-ups) overstated it: the
bench's prose request runs behind a `warm` phase and drafts on prose (0.15 acceptance), 41.8 tok/s against 52.3 serial.

## `specbench.py`, seven-kind table at c = 1 and four streams (`raw/base-specbench.md`; median of 3 after a discarded pass; drafted == serial on every pass)

| kind | serial tok/s | drafted tok/s | speed-up | tokens/round |
|---|---:|---:|---:|---:|
| chat (prose) | 52.31 | 50.98 | 0.974 | 1.51 |
| code | 52.30 | 117.27 | 2.242 | 5.69 |
| json | 51.60 | 85.68 | 1.660 | 4.49 |
| longctx (6K) | 40.62 | 63.83 | 1.571 | 4.53 |
| tool call | 51.51 | 122.21 | 2.373 | 6.47 |
| reasoning | 51.47 | 72.28 | 1.404 | 3.27 |
| c = 4 aggregate | 100.19 | 110.92 | 1.107 | |

(The specbench server had not seen the `bench decode` runs' prose first; prose reads 0.974 here and 0.80 in `bench decode`: the learnt budget's speed depends on the traffic before it.)

## Where a drafted round goes (`tools/roundtrace.py`, `REXL3_ROUND_TRACE=1`, host clock, ms; median of 3 requests; `raw/trace-base.json`)

| kind | tok/round | plan | drafter feed + commit launches | propose (stage, launch, wait, read) | alloc + prepare + launch | **wait for the window** | read | accept + tail | gap outside `decode_round` | round period |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prose (0 drafts: serial rounds) | 1.00 | 0.00 | 0.10 | 0 | 0.14 | 19.06 | 0.02 | 0.00 | 0.03 | 19.28 |
| code | 5.75 | 0.00 | 0.07 | 7.41 | 0.35 | 39.76 | 0.02 | 0.00 | 0.02 | 47.34 |
| structured | 3.35 | 0.00 | 0.07 | 7.26 | 0.03 | 38.84 | 0.02 | 0.00 | 0.02 | 46.29 |
| json | 4.45 | 0.00 | 0.07 | 7.42 | 0.03 | 43.61 | 0.02 | 0.00 | 0.02 | 51.15 |

**The host is not the gap.** Everything the engine thread does between two device waits (plan, acceptance, sampling, the drafter's commit launches, the window's uploads and
launch, the server loop, streaming, `snapshot`) is 0.1 to 0.5 ms a round (under 1 %); the engine waits on the device for 98 % of every round. Serial decode shows the same (19.06 ms of 19.28).
The "7 ms outside the rank's timers" of the earlier records is device time: the verify window of real text costs more than the table the budget uses
(`engine.rs::round_cost`, measured on synthetic rows): by rows in the window, one stream, from every traced round (`raw/trace-all.log`):

| rows | 1 | 3 | 4 | 5 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| real ms | 19.0 | 26.2 | 29.3 | 33.0 | 38.9 | 42.6 |
| `round_cost` ms | 19.05 | 23.4 | 25.0 | 25.8 | 27.6 | 29.5 |

## The device timeline of a drafted round (`tools/nsys_rounds.py` on `nsys profile -t cuda,nvtx --cuda-graph-trace=node` of the server; `raw/nsys-round2-table.txt`; code and json, profiler overhead inflates every figure by about 10 %)

- The round is two device chains: the proposal (drafter kernels, median span 7.7 ms, busy 7.2) and the window graph (2,887 nodes; span 43.2 ms at 7 to 8 rows, 19.0 at one row).
  MoE GEMVs are 36.3 of the window's kernel milliseconds at 8 rows (a kernel every 144 us, 252 a window) against 12.3 at one row: the extra rows cost experts (more distinct experts per layer), not the host.
- Idle device between the last proposal kernel and the first window node: **median 1.6 ms a round** (3 % of a drafted round, 7 % of a serial one): the host reads the proposal back,
  plans, uploads and calls `cudaGraphLaunch` (0.98 ms of host time for 2,887 nodes) before any window kernel can start. The wake after the window's last kernel is 23 us, the commit and feed launch at once.
- Between the windows the GPU also idles 0.45 ms inside the drafter's chain (launch gaps of its 190 small kernels).

This is what the lane removes: the chain of the proposal and the window with no host wait between them (`chained round`, `docs/DECODE-SPEED.md`).
