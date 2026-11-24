# specbench 2026-11-23T16:04:09Z - final

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.11 | 53.59 | 1.028 | 603/1774 (0.34) | 0.65 | 1.65 | True |
| code | 225 | 512 | 52.18 | 112.95 | 2.164 | 1239/1780 (0.70) | 4.21 | 5.22 | True |
| json | 251 | 512 | 51.57 | 86.19 | 1.671 | 1176/2320 (0.51) | 3.29 | 4.30 | True |
| longctx | 6047 | 512 | 40.64 | 65.81 | 1.619 | 1179/1944 (0.61) | 3.33 | 4.34 | True |
| tool | 166 | 369 | 51.48 | 126.75 | 2.462 | 933/1040 (0.90) | 5.36 | 6.36 | True |
| reasoning | 150 | 512 | 51.47 | 75.75 | 1.472 | 1036/2131 (0.49) | 2.08 | 3.09 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.55 | 0.43 | 0.46 | 0.44 | 0.50 | - | - |
| code | 0.91 | 0.90 | 0.90 | 0.91 | 0.89 | 0.82 | 0.83 |
| json | 0.78 | 0.74 | 0.85 | 0.89 | 0.98 | 0.92 | 0.98 |
| longctx | 0.90 | 0.79 | 0.87 | 0.81 | 0.88 | 0.77 | 0.00 |
| tool | 0.95 | 0.93 | 0.94 | 0.86 | 1.00 | 1.00 | 1.00 |
| reasoning | 0.76 | 0.79 | 0.69 | 0.65 | 0.60 | 1.00 | 0.00 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 220 | 64 | 38 | 7 | 3 | 1 | 0 | 0 |
| code | 9 | 9 | 8 | 6 | 8 | 15 | 22 | 19 |
| json | 25 | 24 | 12 | 6 | 1 | 4 | 3 | 41 |
| longctx | 13 | 23 | 11 | 15 | 16 | 15 | 27 | 0 |
| tool | 3 | 4 | 3 | 7 | 4 | 1 | 0 | 37 |
| reasoning | 40 | 26 | 31 | 28 | 28 | 10 | 2 | 0 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.17 | 2048 | chat 25.06 (512), code 25.07 (512), json 25.06 (512), reasoning 25.11 (512) |
| drafted | 116.46 | 2048 | chat 29.11 (512), code 53.25 (512), json 44.31 (512), reasoning 37.24 (512) |

speedup 1.163

PASS drafted == serial
