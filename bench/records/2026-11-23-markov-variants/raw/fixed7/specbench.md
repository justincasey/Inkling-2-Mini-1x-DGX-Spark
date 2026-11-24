# specbench 2026-11-23T09:48:24Z - fixed7: REXL3_INKLING2_LOOP_SPEC_POLICY=fixed

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.20 | 42.58 | 0.816 | 753/5397 (0.14) | 0.97 | 1.97 | True |
| code | 225 | 512 | 52.22 | 116.63 | 2.233 | 1266/1854 (0.68) | 4.74 | 5.75 | True |
| json | 251 | 512 | 51.56 | 85.11 | 1.651 | 1191/2370 (0.50) | 3.48 | 4.49 | True |
| longctx | 6047 | 512 | 40.40 | 63.15 | 1.563 | 1200/2307 (0.52) | 3.60 | 4.61 | True |
| tool | 166 | 369 | 51.20 | 119.82 | 2.340 | 939/1197 (0.78) | 5.49 | 6.47 | True |
| reasoning | 150 | 512 | 51.29 | 71.04 | 1.385 | 1074/3153 (0.34) | 2.34 | 3.35 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.56 | 0.44 | 0.40 | 0.56 | 0.29 | 0.25 | 0.00 |
| code | 0.91 | 0.93 | 0.91 | 0.91 | 0.92 | 0.79 | 0.82 |
| json | 0.79 | 0.74 | 0.85 | 0.88 | 0.98 | 0.92 | 0.98 |
| longctx | 0.89 | 0.82 | 0.84 | 0.81 | 0.80 | 0.81 | 0.66 |
| tool | 0.95 | 0.93 | 0.96 | 0.85 | 0.98 | 1.00 | 1.00 |
| reasoning | 0.76 | 0.80 | 0.68 | 0.60 | 0.66 | 0.60 | 0.67 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 116 | 81 | 38 | 11 | 10 | 3 | 1 | 0 |
| code | 8 | 6 | 7 | 6 | 6 | 12 | 8 | 36 |
| json | 24 | 23 | 11 | 7 | 1 | 4 | 1 | 43 |
| longctx | 13 | 18 | 13 | 13 | 11 | 8 | 12 | 23 |
| tool | 3 | 4 | 2 | 7 | 1 | 0 | 0 | 40 |
| reasoning | 38 | 23 | 29 | 25 | 13 | 10 | 5 | 10 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 99.81 | 2048 | chat 24.98 (512), code 24.96 (512), json 24.96 (512), reasoning 25.00 (512) |
| drafted | 103.79 | 2048 | chat 25.95 (512), code 50.81 (512), json 42.22 (512), reasoning 34.77 (512) |

speedup 1.040

PASS drafted == serial
