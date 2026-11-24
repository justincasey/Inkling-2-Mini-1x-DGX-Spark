# specbench 2026-11-23T10:25:42Z - planL: REXL3_INKLING2_LOOP_SPEC_POLICY=plan REXL3_INKLING2_LOOP_SPEC_LEARN=1

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.35 | 51.48 | 0.984 | 549/1761 (0.31) | 0.56 | 1.56 | True |
| code | 225 | 512 | 52.39 | 117.28 | 2.239 | 1262/1821 (0.69) | 4.66 | 5.67 | True |
| json | 251 | 512 | 51.68 | 85.74 | 1.659 | 1191/2355 (0.51) | 3.48 | 4.49 | True |
| reasoning | 150 | 512 | 50.10 | 71.50 | 1.427 | 1062/2801 (0.38) | 2.25 | 3.26 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.52 | 0.47 | 0.43 | 0.33 | - | - | - |
| code | 0.91 | 0.94 | 0.89 | 0.93 | 0.90 | 0.80 | 0.79 |
| json | 0.79 | 0.74 | 0.85 | 0.89 | 0.96 | 0.92 | 0.98 |
| reasoning | 0.76 | 0.79 | 0.69 | 0.59 | 0.68 | 0.60 | 0.50 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 253 | 52 | 33 | 10 | 3 | - | - | - |
| code | 8 | 5 | 8 | 5 | 7 | 11 | 12 | 33 |
| json | 24 | 23 | 11 | 6 | 2 | 4 | 2 | 42 |
| reasoning | 39 | 25 | 29 | 26 | 13 | 10 | 14 | 1 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.06 | 2048 | chat 25.07 (512), code 25.02 (512), json 25.09 (512), reasoning 25.04 (512) |
| drafted | 110.59 | 2048 | chat 27.65 (512), code 53.45 (512), json 44.36 (512), reasoning 36.55 (512) |

speedup 1.105

PASS drafted == serial
