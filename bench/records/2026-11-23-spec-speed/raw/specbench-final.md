# specbench 2026-11-23T10:49:51Z - final default: plan, learnt priors, depth 7, folded Markov head

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.20 | 47.05 | 0.901 | 739/3594 (0.21) | 0.93 | 1.93 | True |
| code | 225 | 512 | 52.28 | 117.03 | 2.239 | 1266/1845 (0.69) | 4.74 | 5.75 | True |
| json | 251 | 512 | 51.59 | 85.01 | 1.648 | 1188/2363 (0.50) | 3.44 | 4.45 | True |
| longctx | 6047 | 512 | 40.29 | 62.77 | 1.558 | 1190/2200 (0.54) | 3.47 | 4.48 | True |
| tool | 166 | 369 | 51.47 | 121.89 | 2.368 | 939/1146 (0.82) | 5.49 | 6.47 | True |
| reasoning | 150 | 512 | 51.51 | 72.32 | 1.404 | 1062/2793 (0.38) | 2.25 | 3.26 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.55 | 0.44 | 0.39 | 0.55 | 0.33 | 0.00 | - |
| code | 0.91 | 0.94 | 0.89 | 0.93 | 0.90 | 0.80 | 0.79 |
| json | 0.79 | 0.74 | 0.85 | 0.89 | 0.98 | 0.90 | 0.98 |
| longctx | 0.89 | 0.81 | 0.83 | 0.81 | 0.80 | 0.77 | 0.62 |
| tool | 0.95 | 0.93 | 0.96 | 0.90 | 0.93 | 1.00 | 1.00 |
| reasoning | 0.76 | 0.79 | 0.69 | 0.59 | 0.68 | 0.60 | 0.50 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 124 | 81 | 41 | 10 | 10 | 2 | 0 | - |
| code | 8 | 5 | 8 | 5 | 7 | 11 | 12 | 33 |
| json | 24 | 24 | 11 | 6 | 2 | 5 | 2 | 41 |
| longctx | 13 | 19 | 14 | 13 | 11 | 10 | 21 | 13 |
| tool | 3 | 4 | 2 | 5 | 3 | 0 | 2 | 38 |
| reasoning | 39 | 25 | 29 | 26 | 13 | 10 | 14 | 1 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 99.92 | 2048 | chat 25.01 (512), code 25.03 (512), json 24.98 (512), reasoning 25.04 (512) |
| drafted | 110.48 | 2048 | chat 27.62 (512), code 52.86 (512), json 44.30 (512), reasoning 36.23 (512) |

speedup 1.106

PASS drafted == serial
