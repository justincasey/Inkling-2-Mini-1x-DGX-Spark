# specbench 2026-11-23T19:54:10Z - copy=off

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.00 | 52.82 | 1.016 | 620/1886 (0.33) | 0.68 | 1.68 | True |
| code | 225 | 512 | 52.10 | 117.22 | 2.250 | 1251/1801 (0.69) | 4.44 | 5.45 | True |
| json | 251 | 512 | 51.44 | 85.33 | 1.659 | 1155/2270 (0.51) | 3.06 | 4.06 | True |
| longctx | 6047 | 512 | 40.55 | 65.95 | 1.627 | 1179/1928 (0.61) | 3.33 | 4.34 | True |
| tool | 166 | 369 | 51.40 | 126.80 | 2.467 | 932/1037 (0.90) | 5.36 | 6.36 | True |
| reasoning | 150 | 512 | 51.42 | 76.43 | 1.486 | 1033/2125 (0.49) | 2.07 | 3.07 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.55 | 0.45 | 0.43 | 0.44 | 0.33 | 0.00 | - |
| code | 0.92 | 0.93 | 0.88 | 0.90 | 0.88 | 0.78 | 0.75 |
| json | 0.80 | 0.73 | 0.83 | 0.89 | 0.98 | 0.93 | 0.97 |
| longctx | 0.91 | 0.79 | 0.87 | 0.81 | 0.82 | 0.79 | 0.40 |
| tool | 0.95 | 0.93 | 0.94 | 0.89 | 1.00 | 1.00 | 1.00 |
| reasoning | 0.76 | 0.79 | 0.69 | 0.65 | 0.60 | 1.00 | 0.00 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 172 | 71 | 44 | 11 | 3 | 1 | 0 | - |
| code | 8 | 6 | 10 | 7 | 9 | 16 | 28 | 12 |
| json | 30 | 25 | 13 | 6 | 1 | 4 | 5 | 38 |
| longctx | 12 | 23 | 11 | 15 | 19 | 17 | 21 | 2 |
| tool | 3 | 4 | 3 | 6 | 3 | 1 | 0 | 38 |
| reasoning | 40 | 26 | 31 | 28 | 28 | 10 | 2 | 0 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.11 | 2048 | chat 25.09 (512), code 25.05 (512), json 25.03 (512), reasoning 25.08 (512) |
| drafted | 114.87 | 2048 | chat 28.72 (512), code 53.54 (512), json 44.52 (512), reasoning 37.19 (512) |

speedup 1.147

PASS drafted == serial
