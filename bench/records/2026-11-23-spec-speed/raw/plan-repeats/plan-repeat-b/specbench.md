# specbench 2026-11-23T09:33:52Z - fixed2: REXL3_INKLING2_LOOP_SPEC_POLICY=fixed REXL3_INKLING2_LOOP_SPEC_TOKENS=2

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.20 | 46.09 | 0.883 | 753/3999 (0.19) | 0.97 | 1.97 | True |
| code | 225 | 512 | 52.08 | 117.01 | 2.247 | 1266/1854 (0.68) | 4.74 | 5.75 | True |
| json | 251 | 512 | 51.58 | 84.85 | 1.645 | 1188/2367 (0.50) | 3.44 | 4.45 | True |
| reasoning | 150 | 512 | 51.54 | 72.26 | 1.402 | 1062/2802 (0.38) | 2.25 | 3.26 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.56 | 0.44 | 0.40 | 0.56 | 0.36 | 0.25 | - |
| code | 0.91 | 0.93 | 0.91 | 0.91 | 0.92 | 0.79 | 0.82 |
| json | 0.79 | 0.74 | 0.85 | 0.89 | 0.98 | 0.90 | 0.98 |
| reasoning | 0.76 | 0.79 | 0.69 | 0.59 | 0.66 | 0.60 | 0.50 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 116 | 81 | 38 | 11 | 10 | 3 | 1 | 0 |
| code | 8 | 6 | 7 | 6 | 6 | 12 | 8 | 36 |
| json | 24 | 24 | 11 | 6 | 2 | 5 | 2 | 41 |
| reasoning | 39 | 25 | 29 | 26 | 13 | 10 | 14 | 1 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.01 | 2048 | chat 25.04 (512), code 25.02 (512), json 25.00 (512), reasoning 25.04 (512) |
| drafted | 109.24 | 2048 | chat 27.31 (512), code 53.67 (512), json 43.91 (512), reasoning 36.21 (512) |

speedup 1.092

PASS drafted == serial
