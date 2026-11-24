# specbench 2026-11-23T14:55:22Z - 1d53344 shipping defaults

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.31 | 50.98 | 0.974 | 517/1696 (0.30) | 0.51 | 1.51 | True |
| code | 225 | 512 | 52.30 | 117.27 | 2.242 | 1263/1833 (0.69) | 4.68 | 5.69 | True |
| json | 251 | 512 | 51.60 | 85.68 | 1.660 | 1191/2355 (0.51) | 3.48 | 4.49 | True |
| longctx | 6047 | 512 | 40.62 | 63.83 | 1.571 | 1194/2202 (0.54) | 3.52 | 4.53 | True |
| tool | 166 | 369 | 51.51 | 122.21 | 2.373 | 939/1149 (0.82) | 5.49 | 6.47 | True |
| reasoning | 150 | 512 | 51.47 | 72.28 | 1.404 | 1063/2809 (0.38) | 2.26 | 3.27 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.54 | 0.49 | 0.41 | 0.33 | - | - | - |
| code | 0.91 | 0.93 | 0.91 | 0.91 | 0.92 | 0.79 | 0.82 |
| json | 0.79 | 0.74 | 0.85 | 0.89 | 0.96 | 0.92 | 0.98 |
| longctx | 0.88 | 0.82 | 0.83 | 0.79 | 0.81 | 0.84 | 0.68 |
| tool | 0.95 | 0.93 | 0.96 | 0.90 | 0.93 | 1.00 | 1.00 |
| reasoning | 0.75 | 0.79 | 0.71 | 0.60 | 0.66 | 0.64 | 0.33 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 214 | 57 | 40 | 12 | 3 | - | - | - |
| code | 8 | 6 | 7 | 6 | 6 | 12 | 8 | 36 |
| json | 24 | 23 | 11 | 6 | 2 | 4 | 2 | 42 |
| longctx | 14 | 18 | 14 | 14 | 10 | 7 | 17 | 19 |
| tool | 3 | 4 | 2 | 5 | 3 | 0 | 2 | 38 |
| reasoning | 39 | 25 | 27 | 26 | 14 | 9 | 15 | 1 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.19 | 2048 | chat 25.05 (512), code 25.09 (512), json 25.12 (512), reasoning 25.06 (512) |
| drafted | 110.92 | 2048 | chat 27.73 (512), code 53.59 (512), json 43.99 (512), reasoning 36.55 (512) |

speedup 1.107

PASS drafted == serial
