# prefill-speed: cold prefill after the prefill lane (gate `speed-prefill`)

Result **pass** (a measurement; the proposed 25% targets of `docs/ROOFLINE.md` are met, the 40% stretch is not). Engine `1a36951e0b27` (branch `inkling2-loop-prefill` on the integration
head `7da8c61`: three commits, one of them a kernel change: the mul1 4-bit MoE GEMMs as 2-tile blocks), a lab build, node-3 alone on the GPU, SM clock locked at 2,200 MHz (2,197 MHz read),
`SPEC=none PREFIX_CACHE=off VISION=off`, profile `bootstrap` (262,144 x 4, chunk 4,096), int8 KV, decode graphs on. Method as `2026-11-23-prefill-baseline`: `./inkling2mini.sh bench prefill-sweep`,
per size four passes (the first discarded, each a different prompt, `"prefix_cache": false`), the rate is `usage.prompt_tokens` over the server's TTFT, the decode rate follows the 128-token reply of the
same request. The same prompts as the baseline run, so the replies compare (`2026-11-23-prefill-exact`, row 11). One boot, one sweep, `raw/sweep-final/`.

| Size | Kept passes (tok/s) | **Median tok/s** | **TTFT (s)** | baseline tok/s (TTFT) | change | decode after it (3 passes) | median | baseline |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 8K | 2493.3 / 2491.2 / 2495.9 | **2,493.3** | 3.286 | 2,436.7 (3.362) | **+2.3%** | 50.209 / 50.122 / 50.433 | 50.209 | 50.541 |
| 32K | 2301.2 / 2304.4 / 2306.8 | **2,304.4** | 14.220 | 2,254.8 (14.533) | **+2.2%** | 46.210 / 46.012 / 45.979 | 46.012 | 46.263 |
| 128K | 1754.9 / 1754.9 / 1755.5 | **1,754.9** | 74.688 | 1,725.5 (75.960) | **+1.7%** | 34.169 / 34.093 / 34.417 | 34.169 | 34.360 |
| 224K | 1415.2 / 1415.2 / 1414.9 | **1,415.2** | 162.085 | 1,394.9 (164.435) | **+1.5%** | 27.382 / 27.169 / 27.393 | 27.382 | 27.603 |

(The warm-up passes: 2,492.8 / 2,308.0 / 1,754.0 / 1,414.5.) Decode after the prompt is 0.5-0.8% under the baseline's at every size: no decode kernel changed; the difference is inside the spread of the
runs (the three passes of one size span up to 1%) `[INFERENCE]`, and the live `bench.py` on the same boot gave decode c = 1 53.407 tok/s and c = 4 100.13, c = 8 104.86 (the graph-decode record: 53.4 and 100.2).
Lowest MemAvailable over the boot and the sweep: 77,932 MiB (`raw/memwatch-report.txt`).

## Against `docs/ROOFLINE.md` (arithmetic only)

| Size | Measured | bf16 ceiling | % of it | baseline % | 25% target | 40% stretch | what is left to the stretch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K | 2,493.3 | 9,525 | 26.2% | 25.6% | 2,381 | 3,810 | x 1.53 |
| 32K | 2,304.4 | 8,481 | 27.2% | 26.6% | 2,120 | 3,392 | x 1.47 |
| 128K | 1,754.9 | 5,894 | 29.8% | 29.3% | 1,474 | 2,358 | x 1.34 |
| 224K | 1,415.2 | 4,519 | 31.3% | 30.9% | 1,130 | 1,808 | x 1.28 |

The 224K ceiling is derived here (`ROOFLINE.md` has no row): 8.90 + 0.50 + 49,152 x 229,376 / 1e9 = 20.67 GFLOP a token at 93.4 TFLOP/s. The 25% targets are met at every size (they are proposals, not the owner's).
The gap that is left and its causes are in `docs/PREFILL-SPEED.md` (the per-stage table of the baseline record, the five lab removals of the MoE layer, the seven directions measured as no gain): the
GEMMs sit at 40-47% of the fp16 peak with 8 or 16 warps an SM and a sequence of decode / mma / reduce phases, the global attention at 37% of the bf16 peak, the glue at its byte floors; none of it is launch-bound
(idle gaps 0.4% of a chunk). Those causes are `[INFERENCE]` where the document says so.

## What changed (one change)

`1a36951` runs the 3-4-tile list of the mul1 4-bit MoE gate/up and down GEMMs as blocks of two 16-row tiles, two to a group (grid z): half the accumulators, two blocks an SM. One MoE layer of 4,096 rows
on the real pack: 16.46 -> 15.43 ms (layer 0), 17.07 -> 16.41 (12), 17.19 -> 16.10 (22), 17.23 -> 17.46 (17: 68 of 128 experts at 5 bits keep their kernels); the 16K prompt of the prefill test 2,387 -> 2,441 tok/s
before the server was involved. The profile of the changed chunk (8K prompt, depth 0): gate/up 409.9 ms (448.5 before), down 197.7 (213.5), span 1,610.0 ms (1,671.7).

## Commands (node-3 stands for the node's address)

```sh
./inkling2mini.sh build 1a36951e0b27715bf8e34b77130107ebd4ca282e && ./inkling2mini.sh start     # .env: SPEC=none PREFIX_CACHE=off VISION=off
./inkling2mini.sh bench prefill-sweep                                                         # METRIC lines in raw/sweep-final/metrics.txt
python3 bench/tools/bench.py --url http://node-3:8000 --model inkling-2-mini-1x-dgx-spark --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref <pinned file>   # decode check, PASS pin
./inkling2mini.sh memwatch begin ... report; ./inkling2mini.sh stop
```
