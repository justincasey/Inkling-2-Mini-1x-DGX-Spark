# specbench 2026-11-23T09:15:35Z - plan7: REXL3_INKLING2_LOOP_SPEC_POLICY=plan

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.30 | 46.17 | 0.883 | 753/3999 (0.19) | 0.97 | 1.97 | True |
| code | 225 | 512 | 52.37 | 117.22 | 2.238 | 1266/1854 (0.68) | 4.74 | 5.75 | True |
| json | 251 | 512 | 51.62 | 84.89 | 1.644 | 1188/2367 (0.50) | 3.44 | 4.45 | True |
| longctx | 6047 | 512 | 40.48 | 63.42 | 1.567 | 1194/2196 (0.54) | 3.52 | 4.53 | True |
| tool | 166 | 369 | 51.52 | 119.72 | 2.323 | 939/1149 (0.82) | 5.49 | 6.47 | True |
| reasoning | 150 | 512 | 51.57 | 72.29 | 1.402 | 1062/2802 (0.38) | 2.25 | 3.26 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.56 | 0.44 | 0.40 | 0.56 | 0.36 | 0.25 | - |
| code | 0.91 | 0.93 | 0.91 | 0.91 | 0.92 | 0.79 | 0.82 |
| json | 0.79 | 0.74 | 0.85 | 0.89 | 0.98 | 0.90 | 0.98 |
| longctx | 0.88 | 0.82 | 0.83 | 0.79 | 0.81 | 0.84 | 0.66 |
| tool | 0.95 | 0.93 | 0.96 | 0.90 | 0.93 | 1.00 | 1.00 |
| reasoning | 0.76 | 0.79 | 0.69 | 0.59 | 0.66 | 0.60 | 0.50 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 116 | 81 | 38 | 11 | 10 | 3 | 1 | 0 |
| code | 8 | 6 | 7 | 6 | 6 | 12 | 8 | 36 |
| json | 24 | 24 | 11 | 6 | 2 | 5 | 2 | 41 |
| longctx | 14 | 18 | 14 | 14 | 10 | 7 | 17 | 19 |
| tool | 3 | 4 | 2 | 5 | 3 | 0 | 2 | 38 |
| reasoning | 39 | 25 | 29 | 26 | 13 | 10 | 14 | 1 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 99.84 | 2048 | chat 25.03 (512), code 25.01 (512), json 24.98 (512), reasoning 25.07 (512) |
| drafted | 109.44 | 2048 | chat 27.36 (512), code 53.30 (512), json 43.67 (512), reasoning 35.97 (512) |

speedup 1.096

PASS drafted == serial
