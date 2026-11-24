# round-overhead: the drafted round's host side and budget, against the baseline (gate `speed-decode`)

Engine `5cc464e7b0faaa267ad42f9ee9bf68b9e036b972` (branch `inkling2-loop-round`: the integration head `189cbbc` + chained round + budget prices + idle relax), node-1,
GPU locked at 2,200 MHz, TP1, bootstrap profile, `SPEC=dspark PREFIX_CACHE=on VISION=on`, greedy, one fresh boot for everything below (`raw/final/`, driver `tools/run_final.sh`).
Baseline: `2026-11-23-round-baseline` (engine `1d53344`). Byte-exactness of this commit: `2026-11-23-round-exact`.

## Result

| metric | baseline `1d53344` | integration head `189cbbc` (DecodeFuse, from its record) | **this engine** |
|---|---:|---:|---:|
| **`decode_mean4_tok_s`** (`bench decode`, the four kinds, c = 1, median of the three run means) | 81.63 | 83.35 | **84.01** (84.01, 84.42, 83.96) |
| prose / code / structured / json (median of 3) | 41.84 / 122.61 / 73.37 / 88.18 | 42.7 / 125.6 / 75.0 / 90.0 | 49.90 / 118.88 / 79.03 / 88.19 |
| `bench decode` c = 4 aggregate (median) | 101.38 | | 108.97 (109.15, 108.97, 108.37) |
| `bench decode` c = 8 aggregate (median) | 122.67 | | 127.04 |
| specbench c = 4 aggregate (chat, code, json, reasoning) | 110.92 | | 116.46 (serial 100.17) |

**The target of 100 tok/s is not reached: 84.01.** This lane's own share over the integration head is +0.7 (chain, budget prices, idle relax net of each other, see the table below); the head's 83.35 already carries DecodeFuse's gain.

### specbench, seven kinds at c = 1 (median of 3 after a discarded pass, `raw/final/specbench.md`; every drafted reply equals its serial reply)

| kind | serial | baseline drafted | **drafted** | speed-up | tokens/round (base -> now) |
|---|---:|---:|---:|---:|---|
| chat (prose) | 52.11 | 50.98 (0.974x) | **53.59** | **1.028** | 1.51 -> 1.65 |
| code | 52.18 | 117.27 | 112.95 | 2.164 | 5.69 -> 5.22 |
| json | 51.57 | 85.68 | 86.19 | 1.671 | 4.49 -> 4.30 |
| longctx (6K) | 40.64 | 63.83 | 65.81 | 1.619 | 4.53 -> 4.34 |
| tool call | 51.48 | 122.21 | 126.75 | 2.462 | 6.47 -> 6.36 |
| reasoning | 51.47 | 72.28 | 75.75 | 1.472 | 3.27 -> 3.09 |

Prose no longer loses (1.03x here; 0.96x in `bench decode`, where its request follows the `warm` phase's code and json: the learnt class prior is the server's traffic). Code and json lose 2 to 4 %
to the budget's trimmed depth (below). **Prose, code, prose on a fresh server** (`tools`' `specmix.py --mix chat,code,chat --per-phase 8 --max-tokens 256`, `raw/final/specmix.out`): 1.58, 4.60, 1.98 tokens a round,
55.4 / 103.8 / 55.5 tok/s, `PASS drafting follows the traffic`.

### Four identical streams (`tools/homo_c4.py`, E2 of the lead; aggregate tok/s, drafted == serial on every pass)

| 4 x | serial | drafted | speed-up |
|---|---:|---:|---:|
| prose | 128.97 | 139.18 | 1.079 |
| code | 123.48 | 259.34 | 2.100 |
| structured | 122.76 | 176.61 | 1.439 |
| json | 114.31 | 186.11 | 1.628 |
| bench's four kinds mixed | 99.99 | 105.86 | 1.059 |

4 x code spans 259 tok/s (28 rows a window): well above the 190 the union model needs.

## What the host side was, and what removing it bought (E0 of the lead too)

1. **The host gap does not exist.** Between two device waits the engine thread spends 0.1 to 0.5 ms (`2026-11-23-round-baseline`: plan, acceptance, feed and commit launches, uploads, launch, server loop,
   streaming, `snapshot`); it waits on the device 98 % of a round. The "7 ms outside the rank's timers" is the window of real text costing 41 ms at 8 rows, not the 29.5 of the synthetic table
   (`raw/cost-*.json`: served windows by streams and rows: 1x1 18.7, 1x3 25.5, 1x5 31.8, 1x7 37.1, 1x8 42.4, 2x8 46.9, 2x14 58.5, 3x21 75.3, 4x4 (serial) 37.7, 4x24 84.7, 4x32 94.3 ms; the proposal 7.3 / 7.9 / 8.7 / 9.3 ms
   for 1 to 4 streams before the drafter graph, 6.95 / 7.7 / 8.3 after). A served 4-row window (4 streams, serial) is 37.7 ms: the model of C4Plan stands, the host pool is under 0.5 ms.
2. **Chained round** (`1e35bf6`, `b74e4ae`): the proposal, the window's draft rows (a device-to-device copy from the proposal's buffer) and the window are enqueued on one stream; the host waits once per round
   for both. nsys (baseline record) had shown the device idle 1.6 ms between the proposal's last kernel and the window's first (profiler-inflated); the host trace of the chained server (`raw/trace-chain.json`, engine `1e35bf6`):
   code 47.34 -> 46.91 ms a round, structured 46.29 -> 45.67, json 51.15 -> 50.48: **0.4 to 0.7 ms (about 1 %)**; the window starts as the proposal ends (`enqueue 0.36 ms + launch 0.47 ms` are hidden behind the proposal's 7 ms).
3. **The budget's prices** (`daec449`, `a76035f`, `54bfaaa`): the table priced real windows at 55 to 70 % of their cost, so the plan drafted prose at depth (acceptance 0.15 to 0.3 against a break-even of 0.56 for
   one draft) and over-drafted at four streams. With served costs prose draws fewer drafts. The straight line through the three kinds' served windows (3.15 ms a row at one stream) still trims code's and json's last draft
   (their windows cost 39.8 and 43.6 ms at 8 rows, the line 41): code -3 % and json -2 % against the head. A table per kind is not possible (the plan does not know the kind).
4. **Idle relax** (`4f07e0d`): a stream the plan stopped drafting forgets by 0.99 a round instead of 0.9, so it retries after about seventy rounds instead of ten (`Engine::with_idle_relax`, default off for every
   other family; `REXL3_INKLING2_LOOP_SPEC_RELAX`): prose 47.6 -> 50.9 tok/s (serial 52.3).

### Every build, in order (`bench decode`, medians of the runs listed; raw under `raw/steps/`)

| step | engine | what | mean4 | prose | code | structured | json | c=4 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `1d53344` | | 81.63 | 41.84 | 122.61 | 73.37 | 88.18 | 101.38 |
| chain + real window prices (learnt priors on) | `daec449` | 3 runs | 82.39 | 47.64 | 117.0 | 75.4 | 89.7 | 107.6 |
| same, frozen priors (`SPEC_LEARN=0`) | `daec449` | 3 runs | 81.90 | 47.11 | 116.9 | 74.4 | 89.3 | 107.7 |
| + idle relax 0.99 (learnt priors) | `4f07e0d` | 2 runs | 82.34 (81.10, 83.58) | 51.10 | 115.9 | 76.1 | 86.3 | 108.2 |
| same, frozen priors | `4f07e0d` | 2 runs | 82.15 | 48.8 | 116.4 | 76.0 | 87.4 | 107.9 |
| + DecodeFuse's drafter graph (integration), flat one-stream line, proposal 6.4 ms | `54bfaaa` | 3 runs | 84.53 (82.78, 84.63, 84.53) | 50.2 | 122.0 | 79.2 | 87.2 | 109.3 |
| final (changelog fragment only) | `5cc464e` | 3 runs | **84.01** | 49.90 | 118.88 | 79.03 | 88.19 | 108.97 |

Two directions stopped after measuring no gain: frozen class priors (`SPEC_LEARN=0`: prose 2 tok/s worse at both builds, nothing better elsewhere) and a one-stream line steeper than the served mean
(it trims code's last draft). The run-to-run spread of one build is 0.5 to 1.8 on mean4 (the first run after a boot, whose class prior comes from `warm`, reads lowest: `54bfaaa` 82.78 against 84.6).

## What remains, with numbers

- Every drafted round is the proposal (6.4 to 7 ms, 8 at four streams) plus the window (19 ms for 1 row, 41 for 8): 45 to 50 ms for 3.2 to 5.4 tokens. The host adds under 0.5 ms; the chain removed 0.4 to 0.7 of the 1.6 ms the
  profiler saw. **The rest is the device: the window's MoE GEMVs (36 ms of kernel time at 8 rows against 12 at one) and the proposal's markov steps.** For mean4 to reach 100 with prose at its serial 52, the three drafted kinds must
  average 116 tok/s (now 95): a 22 % shorter round at today's acceptance, or more tokens a round, neither of which the host can give.
- A zero-draft round (prose served serially) still idles the device 1.5 ms between windows (7 % of 19 ms): the next window's token is the previous window's argmax, which a device-side greedy chain could feed
  without the host (not built: it needs the engine to run one window ahead and roll back with `commit` on a late stop; worth about +3.5 tok/s on prose, +0.9 on mean4).
- Code and json are 2 to 3 % under the integration head because of the one-stream window line; a per-kind price is not available to the plan.

## Commands

```sh
./inkling2mini.sh build 5cc464e7b0faaa267ad42f9ee9bf68b9e036b972     # REXL3_SRC=<engine worktree inkling2-loop-round>
tools/run_final.sh: ./inkling2mini.sh start; specmix.py --mix chat,code,chat --per-phase 8 --max-tokens 256; ./inkling2mini.sh bench decode (x3);
  specbench.py --concurrency 4; homo_c4.py; exactness / gates (see round-exact); ./inkling2mini.sh stop
EXTRA_ENV="REXL3_ROUND_TRACE=1 REXL3_INKLING2_LOOP_CHAIN=0" ./inkling2mini.sh start; bench/tools/roundcost.py --url http://node-1:8000 --ssh spark@node-1 --log serve.log --out cost.json   # the window prices
```
