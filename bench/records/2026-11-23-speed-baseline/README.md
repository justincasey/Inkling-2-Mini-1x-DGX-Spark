# speed-baseline: the numbers the performance phase starts from (gate `speed-baseline`)

Result **pass** (a measurement was made; there is no pinned reference and no threshold). Engine `3287e80d` (lab build,
`REXL3_ALLOW_UNGATED_COMPONENTS=1`), `--spec none --prefix-cache off --no-vision`, profile `bootstrap` (262,144 x 4, chunk 4096), int8 KV, **no CUDA
graphs**, node-4 alone on the GPU, SM clock locked at 2,200 MHz (`gpu-clocks status` read 2197 MHz). One boot; the three passes ran back to back.

Commands (from this machine, no `bench/reference.json`, so `--raw`): `./inkling2mini.sh bench decode` three times (`decode-1/2/3`, each:
`warm` phase discarded, then `decode` at c = 1 over four prompt types with 512 greedy tokens each, then `conc` at c = 4 and c = 8) and
`./inkling2mini.sh bench prefill` three times (`prefill-1/2/3`, each: `warm` discarded, then cold 32,768-, 131,072- and 229,376-token prompts, 128 new
tokens each). Token counts are `usage.completion_tokens`; the prefill time is the server's `rexl3_request_ttft_seconds` (`bench/tools/bench.py`). Median of three below.

| Metric | Pass 1 / 2 / 3 | **Median** |
|---|---|---:|
| `decode_c1_tok_s` (mean of prose, code, structured, json) | 37.290 / 37.198 / 37.163 | **37.198** |
| `decode_c4_tok_s` (aggregate, 4 streams) | 80.253 / 80.180 / 80.063 | **80.180** |
| `decode_c8_tok_s` (aggregate, 8 requests on 4 slots: 4 queue) | 83.239 / 82.067 / 83.984 | **83.239** |
| `prefill_32k_tok_s` (cold, 32,768 tokens) | 1187.2 / 1200.8 / 1201.9 | **1200.8** (27.3 s) |
| `prefill_128k_tok_s` (cold, 131,072) | 1027.6 / 1032.8 / 1033.2 | **1032.8** (126.9 s) |
| `prefill_224k_tok_s` (cold, 229,376) | 904.1 / 905.4 / 905.1 | **905.1** (253.4 s) |
| `decode_ctx32k_tok_s` (decode after that prompt) | 33.397 / 33.348 / 33.476 | **33.397** |
| `decode_ctx128k_tok_s` | 26.732 / 26.779 / 26.823 | **26.779** |
| `decode_ctx224k_tok_s` | 22.198 / 22.267 / 22.280 | **22.267** |

`decode_tokens_per_round` is 1.002 (no drafter). The four prompt types decode within 1.5% of each other (36.95 to 37.61 tok/s). Every
`conc` reply equalled its solo reply (the tool's own check; no `FAIL` in `decode-*/bench-decode.out`). The 260K needle prompt of
`2026-11-23-needle` prefilled at 869.6 tok/s (260,019 tokens in 299.0 s), one data point beyond this grid.

## Against `docs/ROOFLINE.md` (arithmetic only)

| Metric | Measured (median) | Ceiling | % of ceiling | Target | % of target |
|---|---:|---:|---:|---:|---:|
| decode c = 1, serial (ceiling 75.4 tok/s weights only; 70.1 with int8 KV at 4K; the prompts here are a few hundred tokens) | 37.20 | 70.1 | 53.1% | 62 | 60.0% |
| decode c = 1 at 32K context | 33.40 | 61.5 | 54.3% | 54 | 61.8% |
| decode c = 1 at 128K context | 26.78 | 43.3 | 61.8% | 38 | 70.5% |
| decode aggregate, 4 streams | 80.18 | 113 | 71.0% | 89 | 90.1% |
| decode aggregate, 8 streams (4 slots) | 83.24 | 137 | 60.8% | 108 | 77.1% |
| prefill cold 32K | 1,200.8 | 8,481 | 14.2% | 2,120 | 56.6% |
| prefill cold 128K | 1,032.8 | 5,894 | 17.5% | 1,474 | 70.1% |

No ROOFLINE row exists for 224K or for 260K. 37.20 tok/s is 26.9 ms a token; at the 3.14 GB of weights one row reads (ROOFLINE "Decode"),
that is 116.8 GB/s of weight traffic, against the 237 GB/s the ceiling assumes. `docs/ROOFLINE.md` (Sibling calibration) says in its own words
that a first boot far below the siblings (serial under about 40 tok/s) "points at a bug, not a tuning gap" and marks that `[INFERENCE]`; this
record states the two numbers side by side and goes no further. The first-boot smoke saw 37.4 tok/s over a 128-token request, the same figure.

Not measured here: decode with CUDA graphs, any drafter, prefill with the prefix cache, the 1M window, the sweep grid (`bench sweep`).
