# specbench 2026-11-23T10:14:55Z - markovc: REXL3_INKLING2_LOOP_SPEC_POLICY=fixed REXL3_INKLING2_LOOP_MARKOV=c

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.18 | 36.80 | 0.705 | 753/5397 (0.14) | 0.97 | 1.97 | True |
| code | 225 | 512 | 52.24 | 102.16 | 1.955 | 1266/1854 (0.68) | 4.74 | 5.75 | True |
| json | 251 | 512 | 51.56 | 74.95 | 1.454 | 1191/2370 (0.50) | 3.48 | 4.49 | True |
| reasoning | 150 | 512 | 51.50 | 61.71 | 1.198 | 1074/3153 (0.34) | 2.34 | 3.35 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.56 | 0.44 | 0.40 | 0.56 | 0.29 | 0.25 | 0.00 |
| code | 0.91 | 0.93 | 0.91 | 0.91 | 0.92 | 0.79 | 0.82 |
| json | 0.79 | 0.74 | 0.85 | 0.88 | 0.98 | 0.92 | 0.98 |
| reasoning | 0.76 | 0.80 | 0.68 | 0.60 | 0.66 | 0.60 | 0.67 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 116 | 81 | 38 | 11 | 10 | 3 | 1 | 0 |
| code | 8 | 6 | 7 | 6 | 6 | 12 | 8 | 36 |
| json | 24 | 23 | 11 | 7 | 1 | 4 | 1 | 43 |
| reasoning | 38 | 23 | 29 | 25 | 13 | 10 | 5 | 10 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.03 | 2048 | chat 25.01 (512), code 25.03 (512), json 25.07 (512), reasoning 25.03 (512) |
| drafted | 94.92 | 2048 | chat 23.73 (512), code 48.31 (512), json 38.94 (512), reasoning 32.09 (512) |

speedup 0.949

PASS drafted == serial
