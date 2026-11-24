# specbench 2026-11-23T10:06:46Z - fixed2: REXL3_INKLING2_LOOP_SPEC_POLICY=fixed REXL3_INKLING2_LOOP_SPEC_TOKENS=2

model inkling-2-mini-1x-dgx-spark, 512 greedy tokens, median of 3 timed passes (one warm-up discarded), wall of the request (prefill included)

| kind | prompt tok | gen tok | serial tok/s | drafted tok/s | speedup | acc/drafted | acc/round | tok/round | equal |
|---|---|---|---|---|---|---|---|---|---|
| chat | 74 | 512 | 52.22 | 52.26 | 1.001 | 678/1701 (0.40) | 0.79 | 1.80 | True |
| code | 225 | 512 | 52.29 | 79.41 | 1.519 | 981/1101 (0.89) | 1.78 | 2.78 | True |
| json | 251 | 512 | 51.61 | 71.64 | 1.388 | 942/1182 (0.80) | 1.59 | 2.60 | True |
| reasoning | 150 | 512 | 51.59 | 67.49 | 1.308 | 879/1299 (0.68) | 1.34 | 2.35 | True |

Conditional acceptance accepted/reached per draft position:

| kind | p1 | p2 |
|---|---|---|
| chat | 0.55 | 0.44 |
| code | 0.92 | 0.93 |
| json | 0.87 | 0.83 |
| reasoning | 0.77 | 0.77 |

Rounds by accepted-draft count (solo pass):

| kind | 0 | 1 | 2 |
|---|---|---|---|
| chat | 128 | 88 | 69 |
| code | 14 | 13 | 157 |
| json | 25 | 30 | 142 |
| reasoning | 52 | 39 | 127 |

Concurrency 4 (chat, code, json, reasoning), aggregate tok/s over the wall of the batch:

| arm | agg tok/s | tokens | per stream tok/s |
|---|---|---|---|
| serial | 100.02 | 2048 | chat 25.02 (512), code 25.02 (512), json 25.00 (512), reasoning 25.01 (512) |
| drafted | 112.73 | 2048 | chat 28.18 (512), code 36.38 (512), json 34.28 (512), reasoning 32.11 (512) |

speedup 1.127

PASS drafted == serial
