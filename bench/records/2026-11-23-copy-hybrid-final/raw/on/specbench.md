# specbench 2026-11-23T21:43:37Z - copy=on

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.05 | 53.05 | 1.019 | 628/1901 (0.33) | 0.69 | 1.70 | True |
| code | 225 | 512 | 51.33 | 113.98 | 2.220 | 1247/1794 (0.70) | 4.36 | 5.37 | True |
| json | 251 | 512 | 51.16 | 85.57 | 1.673 | 1148/2257 (0.51) | 2.98 | 3.99 | True |
| longctx | 6047 | 512 | 40.90 | 66.09 | 1.616 | 1167/1900 (0.61) | 3.19 | 4.20 | True |
| tool | 166 | 369 | 51.43 | 126.59 | 2.462 | 929/1079 (0.86) | 5.13 | 6.12 | True |
| reasoning | 150 | 512 | 51.44 | 76.72 | 1.491 | 1035/2116 (0.49) | 2.08 | 3.08 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.56 | 0.45 | 0.48 | 0.44 | 0.33 | - | - |
| code | 0.91 | 0.90 | 0.90 | 0.91 | 0.89 | 0.82 | 0.83 |
| json | 0.77 | 0.71 | 0.78 | 0.93 | 0.97 | 0.94 | 1.00 |
| longctx | 0.89 | 0.78 | 0.87 | 0.80 | 0.80 | 0.78 | 0.40 |
| tool | 0.93 | 0.90 | 0.92 | 0.84 | 1.00 | 1.00 | 0.96 |
| reasoning | 0.77 | 0.80 | 0.65 | 0.62 | 0.33 | - | - |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 199 | 65 | 39 | 11 | 3 | 1 | 0 | - |
| code | 9 | 9 | 8 | 6 | 8 | 15 | 22 | 19 |
| json | 33 | 22 | 13 | 3 | 1 | 4 | 2 | 30 |
| longctx | 14 | 23 | 11 | 15 | 20 | 16 | 19 | 2 |
| tool | 3 | 4 | 3 | 6 | 3 | 1 | 1 | 22 |
| reasoning | 40 | 26 | 36 | 32 | 33 | 3 | 0 | 0 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.24 | 2048 | chat 25.09 (512), code 25.12 (512), json 25.07 (512), reasoning 25.06 (512) |
| drafted | 114.55 | 2048 | chat 28.64 (512), code 53.65 (512), json 44.84 (512), reasoning 37.28 (512) |

speedup 1.143

PASS drafted == serial
