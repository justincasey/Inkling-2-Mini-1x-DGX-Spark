# specbench 2026-11-23T17:57:31Z - copy=off

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.04 | 52.78 | 1.014 | 622/1893 (0.33) | 0.68 | 1.69 | True |
| code | 225 | 512 | 52.17 | 116.61 | 2.235 | 1251/1805 (0.69) | 4.44 | 5.45 | True |
| json | 251 | 512 | 51.51 | 85.12 | 1.653 | 1161/2287 (0.51) | 3.12 | 4.13 | True |
| longctx | 6047 | 512 | 40.62 | 65.67 | 1.617 | 1178/1950 (0.60) | 3.32 | 4.33 | True |
| tool | 166 | 369 | 51.49 | 126.61 | 2.459 | 933/1038 (0.90) | 5.36 | 6.36 | True |
| reasoning | 150 | 512 | 51.47 | 76.33 | 1.483 | 1034/2128 (0.49) | 2.07 | 3.08 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.56 | 0.43 | 0.42 | 0.44 | 0.33 | 0.00 | - |
| code | 0.91 | 0.93 | 0.88 | 0.91 | 0.87 | 0.82 | 0.80 |
| json | 0.78 | 0.74 | 0.85 | 0.87 | 0.98 | 0.91 | 1.00 |
| longctx | 0.91 | 0.79 | 0.87 | 0.81 | 0.88 | 0.74 | 0.57 |
| tool | 0.95 | 0.93 | 0.94 | 0.89 | 1.00 | 1.00 | 1.00 |
| reasoning | 0.76 | 0.79 | 0.69 | 0.65 | 0.60 | 1.00 | 0.00 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 240 | 60 | 34 | 7 | 3 | 1 | 0 | - |
| code | 8 | 6 | 10 | 6 | 9 | 13 | 22 | 20 |
| json | 30 | 23 | 12 | 7 | 1 | 4 | 2 | 41 |
| longctx | 12 | 22 | 11 | 15 | 16 | 16 | 22 | 4 |
| tool | 3 | 4 | 3 | 6 | 3 | 1 | 0 | 38 |
| reasoning | 40 | 26 | 31 | 28 | 28 | 10 | 2 | 0 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.22 | 2048 | chat 25.15 (512), code 25.08 (512), json 25.08 (512), reasoning 25.05 (512) |
| drafted | 115.75 | 2048 | chat 28.94 (512), code 52.43 (512), json 44.62 (512), reasoning 37.28 (512) |

speedup 1.155

PASS drafted == serial
