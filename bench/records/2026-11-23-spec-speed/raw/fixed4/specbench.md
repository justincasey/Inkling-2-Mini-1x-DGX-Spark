# specbench 2026-11-23T09:58:49Z - fixed4: REXL3_INKLING2_LOOP_SPEC_POLICY=fixed REXL3_INKLING2_LOOP_SPEC_TOKENS=4

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.18 | 48.32 | 0.926 | 741/3144 (0.24) | 0.94 | 1.94 | True |
| code | 225 | 512 | 52.25 | 102.03 | 1.953 | 1170/1452 (0.81) | 3.22 | 4.23 | True |
| json | 251 | 512 | 51.57 | 83.78 | 1.625 | 1110/1686 (0.66) | 2.62 | 3.63 | True |
| reasoning | 150 | 512 | 51.53 | 73.85 | 1.433 | 1023/2016 (0.51) | 2.01 | 3.01 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 | p3 | p4 |
|---|---|---|---|---|
| chat | 0.55 | 0.44 | 0.39 | 0.56 |
| code | 0.92 | 0.95 | 0.88 | 0.89 |
| json | 0.79 | 0.83 | 0.93 | 0.92 |
| reasoning | 0.76 | 0.80 | 0.67 | 0.63 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| chat | 120 | 80 | 39 | 11 | 14 |
| code | 10 | 6 | 13 | 10 | 82 |
| json | 29 | 19 | 7 | 7 | 79 |
| reasoning | 42 | 26 | 34 | 25 | 43 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 99.60 | 2048 | chat 24.90 (512), code 24.93 (512), json 24.98 (512), reasoning 24.97 (512) |
| drafted | 112.40 | 2048 | chat 28.10 (512), code 44.60 (512), json 39.43 (512), reasoning 35.18 (512) |

speedup 1.128

PASS drafted == serial
