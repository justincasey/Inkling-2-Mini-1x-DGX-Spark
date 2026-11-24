# specbench 2026-11-23T10:33:35Z - curveL: REXL3_INKLING2_LOOP_SPEC_POLICY=curve REXL3_INKLING2_LOOP_SPEC_LEARN=1

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.35 | 51.19 | 0.978 | 716/2254 (0.32) | 0.88 | 1.88 | True |
| code | 225 | 512 | 52.45 | 114.88 | 2.190 | 1255/1817 (0.69) | 4.51 | 5.53 | True |
| json | 251 | 512 | 51.21 | 85.77 | 1.675 | 1191/2354 (0.51) | 3.48 | 4.49 | True |
| reasoning | 150 | 512 | 51.70 | 72.52 | 1.403 | 1062/2797 (0.38) | 2.25 | 3.26 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|
| chat | 0.55 | 0.45 | 0.46 | 0.50 | - | - | - |
| code | 0.91 | 0.94 | 0.89 | 0.93 | 0.90 | 0.80 | 0.79 |
| json | 0.79 | 0.74 | 0.85 | 0.89 | 0.96 | 0.92 | 0.98 |
| reasoning | 0.75 | 0.79 | 0.71 | 0.60 | 0.66 | 0.60 | 0.67 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chat | 121 | 81 | 48 | 11 | 8 | 0 | 0 | 0 |
| code | 8 | 5 | 8 | 5 | 7 | 11 | 12 | 33 |
| json | 24 | 23 | 11 | 6 | 2 | 4 | 2 | 42 |
| reasoning | 39 | 25 | 27 | 26 | 14 | 10 | 13 | 2 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.16 | 2048 | chat 25.15 (512), code 25.10 (512), json 25.07 (512), reasoning 25.14 (512) |
| drafted | 111.12 | 2048 | chat 27.78 (512), code 53.26 (512), json 44.46 (512), reasoning 36.48 (512) |

speedup 1.109

PASS drafted == serial
