# specbench 2026-11-23T18:23:39Z - copy=on

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.09 | 52.88 | 1.015 | 591/1760 (0.34) | 0.63 | 1.63 | True |
| code | 225 | 512 | 52.21 | 115.06 | 2.204 | 1241/1772 (0.70) | 4.25 | 5.26 | True |
| json | 251 | 512 | 51.54 | 88.83 | 1.724 | 1132/2182 (0.52) | 2.82 | 3.83 | True |
| longctx | 6047 | 512 | 40.62 | 66.89 | 1.647 | 1165/1842 (0.63) | 3.17 | 4.17 | True |
| tool | 166 | 369 | 51.51 | 128.07 | 2.486 | 924/1082 (0.85) | 4.97 | 5.95 | True |
| reasoning | 150 | 512 | 51.55 | 76.13 | 1.477 | 1025/2070 (0.50) | 2.02 | 3.02 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.55 | 0.44 | 0.48 | 0.29 | 1.00 | - | - |
| code | 0.90 | 0.92 | 0.89 | 0.91 | 0.89 | 0.82 | 0.82 |
| json | 0.71 | 0.61 | 0.89 | 0.96 | 0.95 | 1.00 | 1.00 |
| longctx | 0.90 | 0.77 | 0.85 | 0.82 | 0.74 | 0.62 | 0.00 |
| tool | 0.91 | 0.87 | 0.88 | 0.81 | 1.00 | 1.00 | 0.93 |
| reasoning | 0.77 | 0.77 | 0.73 | 0.62 | 0.50 | 1.00 | 0.00 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 216 | 64 | 37 | 11 | 1 | 1 | 0 | 0 |
| code | 10 | 7 | 9 | 6 | 8 | 15 | 23 | 18 |
| json | 38 | 19 | 5 | 2 | 2 | 0 | 2 | 19 |
| longctx | 11 | 20 | 10 | 12 | 20 | 17 | 8 | 0 |
| tool | 3 | 4 | 3 | 6 | 2 | 1 | 1 | 13 |
| reasoning | 39 | 29 | 27 | 35 | 32 | 3 | 2 | 0 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.23 | 2048 | chat 25.12 (512), code 25.11 (512), json 25.08 (512), reasoning 25.09 (512) |
| drafted | 115.89 | 2048 | chat 28.97 (512), code 52.13 (512), json 44.96 (512), reasoning 37.46 (512) |

speedup 1.156

PASS drafted == serial
