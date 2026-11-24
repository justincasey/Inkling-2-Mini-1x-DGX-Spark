# specbench 2026-11-23T19:11:00Z - copy=on

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 51.99 | 53.03 | 1.020 | 628/1903 (0.33) | 0.69 | 1.70 | True |
| code | 225 | 512 | 52.10 | 116.30 | 2.232 | 1247/1791 (0.70) | 4.36 | 5.37 | True |
| json | 251 | 512 | 51.46 | 86.49 | 1.681 | 1145/2230 (0.51) | 2.95 | 3.96 | True |
| longctx | 6047 | 512 | 40.56 | 65.14 | 1.606 | 1167/1911 (0.61) | 3.19 | 4.20 | True |
| tool | 166 | 369 | 51.44 | 127.03 | 2.469 | 929/1079 (0.86) | 5.13 | 6.12 | True |
| reasoning | 150 | 512 | 51.48 | 76.83 | 1.492 | 1035/2116 (0.49) | 2.08 | 3.08 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.55 | 0.46 | 0.45 | 0.44 | 0.33 | - | - |
| code | 0.91 | 0.90 | 0.90 | 0.91 | 0.89 | 0.82 | 0.83 |
| json | 0.77 | 0.71 | 0.78 | 0.93 | 0.97 | 0.94 | 1.00 |
| longctx | 0.89 | 0.78 | 0.87 | 0.80 | 0.80 | 0.78 | 0.40 |
| tool | 0.93 | 0.90 | 0.92 | 0.84 | 1.00 | 1.00 | 0.96 |
| reasoning | 0.77 | 0.80 | 0.66 | 0.60 | 0.75 | 0.83 | 1.00 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 202 | 64 | 40 | 10 | 3 | 1 | 0 | - |
| code | 9 | 9 | 8 | 6 | 8 | 15 | 22 | 19 |
| json | 33 | 22 | 13 | 3 | 1 | 4 | 2 | 30 |
| longctx | 14 | 23 | 11 | 15 | 20 | 16 | 19 | 2 |
| tool | 3 | 4 | 3 | 6 | 3 | 1 | 1 | 22 |
| reasoning | 39 | 25 | 35 | 33 | 25 | 4 | 4 | 1 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.15 | 2048 | chat 25.04 (512), code 25.09 (512), json 25.13 (512), reasoning 25.07 (512) |
| drafted | 115.40 | 2048 | chat 28.85 (512), code 53.69 (512), json 44.62 (512), reasoning 37.39 (512) |

speedup 1.152

PASS drafted == serial
