# serial-chain: windows of one row run ahead of the host: NO-GO (gate `speed-serial-chain-nogo`)

Question (RoundOverhead's finding, and the brief): a serial window (a stream the budget gave no draft, or `SPEC=none`) idles about 1.5 ms between windows; can the next window be launched on the device behind the
current one (its token id copied by the stream from the first candidate, the commit recorded by the rank, the candidates read back through pinned memory and an event) and win it?

Engine `f658ec6e8073ecab9725947ab5a659e3822323b9` (built on node-4; `rexl3-serve` sha256 `c2a502a0...75bf`): the rank's `window_ahead`, `Backend::step_ahead` with a per-stream hint, the backend absorbing the engine's commit; switch `REXL3_INKLING2_LOOP_AHEAD=0|1`.
Node-4, bootstrap profile, GPU cap 2,200 MHz applied by `scripts/gpu_cap.sh` before each run, fresh server per column, `bench decode` x3 and `bench rolling` x3. Driver `tools/run_serial.sh`.

## Where the time goes (`raw/none-ahead0/round-trace.txt`: `REXL3_ROUND_TRACE=1`, `SPEC=none`, one stream, 12,384 rounds, medians in microseconds from the round's start)

`w_prepared` 18, `w_launched` 21 (the graph launch returns at once), `w_synced` **18,636**, `verified` 18,650, `accepted` 18,651, the gap to the next round 5. A serial round is 18.65 ms and all but about 0.05 ms of it is the host waiting for the device:
the host's turnaround between two windows is 0.03 to 0.06 ms (the commit and drafter-feed launches are 9 us), not 1.5 ms. What a device-fed chain can take is therefore at most that, 0.3 %. The 1.5 ms of the earlier note was the profiler's view
(a window under nsys), not a gap of the served window; what is inside the 18.6 ms is the window's own kernels and the launch gaps between its 2,887 graph nodes, which no host-side chain reaches.

## A/B (medians of three; `raw/*/decode-N.txt`, `rolling-N.txt`)

| configuration | AHEAD=0 | AHEAD=1 |
|---|---:|---:|
| `SPEC=none` c = 1, mean of the four kinds | 53.41 | 53.32 |
| `SPEC=none` `bench decode` c = 4 aggregate | 100.57 | 100.17 |
| `SPEC=none` `decode_c4_rolling_tok_s` | 100.61 | 100.28 |
| `SPEC=dspark` (shipping, copy hybrid on) mean4 (three runs) | 85.0 (84.4, 85.0, 85.7) | 84.8 (84.8, 82.7, 85.3) |
| `SPEC=dspark` prose | 50.13 | 49.52 |

No gain anywhere (serial c = 1 -0.2 %, c = 4 -0.4 %, rolling -0.3 %: the cost of the read-back through an event and the staged-buffer syncs); the drafted prose, which is the case that was supposed to gain 0.9 on mean4, did not move. `dspark-ahead1` has no rolling runs
(the battery was stopped on the verdict). **Verdict: no-go.** The code was removed from the engine branch (`f4922eb` reverts `f658ec6` and its test fix; nothing of it ships).

## What this does not prove

The GPU equality test of the chain (`inkling2_loop_ahead`, in `f658ec6`) failed in its setup (a slot begun twice, fixed in the test afterwards) and was not rerun; byte-exactness of the chain is therefore NOT proven, and no exactness record exists for it. The A/B stands without it: a
chain that gains nothing is not worth proving. The host anatomy is for one stream; at four streams the host turnaround is also under a tenth of a millisecond of a 37.7 ms window by the same trace shape (not tabulated).
