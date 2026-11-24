# specbench 2026-11-23T11:36:53Z - merged engine default: dspark plan learnt, cache on, vision on

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.30 | 47.00 | 0.899 | 745/3624 (0.21) | 0.95 | 1.95 | True |
| code | 225 | 512 | 52.38 | 117.56 | 2.244 | 1264/1843 (0.69) | 4.70 | 5.71 | True |
| json | 251 | 512 | 51.69 | 84.59 | 1.636 | 1186/2370 (0.50) | 3.42 | 4.43 | True |
| longctx | 6047 | 512 | 40.72 | 63.87 | 1.568 | 1191/2176 (0.55) | 3.48 | 4.49 | True |
| tool | 166 | 369 | 51.50 | 122.54 | 2.379 | 939/1141 (0.82) | 5.49 | 6.47 | True |
| reasoning | 150 | 512 | 51.70 | 72.74 | 1.407 | 1063/2779 (0.38) | 2.26 | 3.27 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.55 | 0.43 | 0.38 | 0.59 | 0.29 | 0.00 | - |
| code | 0.91 | 0.94 | 0.89 | 0.93 | 0.90 | 0.80 | 0.79 |
| json | 0.79 | 0.74 | 0.85 | 0.89 | 0.98 | 0.90 | 1.00 |
| longctx | 0.89 | 0.81 | 0.83 | 0.82 | 0.79 | 0.77 | 0.63 |
| tool | 0.95 | 0.93 | 0.96 | 0.90 | 0.95 | 0.98 | 1.00 |
| reasoning | 0.75 | 0.79 | 0.71 | 0.60 | 0.66 | 0.64 | 0.33 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 121 | 82 | 41 | 9 | 11 | 2 | 0 | - |
| code | 8 | 5 | 8 | 5 | 7 | 11 | 12 | 33 |
| json | 24 | 24 | 11 | 6 | 2 | 5 | 2 | 41 |
| longctx | 13 | 19 | 14 | 12 | 12 | 10 | 22 | 12 |
| tool | 3 | 4 | 2 | 5 | 2 | 1 | 3 | 37 |
| reasoning | 39 | 25 | 27 | 26 | 14 | 9 | 15 | 1 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.52 | 2048 | chat 25.14 (512), code 25.16 (512), json 25.14 (512), reasoning 25.14 (512) |
| drafted | 111.37 | 2048 | chat 27.84 (512), code 53.51 (512), json 44.36 (512), reasoning 36.65 (512) |

speedup 1.108

PASS drafted == serial
