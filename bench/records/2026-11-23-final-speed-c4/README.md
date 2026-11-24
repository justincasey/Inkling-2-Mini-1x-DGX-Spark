# final-speed-c4: gate `speed-c4`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench all   (rolling phase)`
- `./inkling2mini.sh bench rolling   (g-pass2-rolling)`
- `./inkling2mini.sh bench rolling   (g-pass3-rolling)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/bench.py | g-all: 4 solo and 32 rolling replies, each equal to its solo reply (PASS pin) | 36 | 0 |
| bench/tools/bench.py | g-pass2-rolling: 4 solo and 32 rolling replies, each equal to its solo reply (PASS pin) | 36 | 0 |
| bench/tools/bench.py | g-pass3-rolling: 4 solo and 32 rolling replies, each equal to its solo reply (PASS pin) | 36 | 0 |

## Metrics

| metric | value |
|---|---:|
| `decode_c4_rolling_tok_s` | 142.329 |
| `decode_c4_steady_tok_s` | 143.553 |
| `decode_c4_rolling_span_s` | 115.114 |
| `decode_c4_tok_s` | 113.175 |
| `decode_c4_rolling_requests` | 32 |
| `decode_c4_rolling_short_replies` | 0 |
| `decode_c4_rolling_tok_s_min` | 141.817 |
| `decode_c4_rolling_tok_s_max` | 142.365 |
| `rolling_runs` | 3 |
| `gpu_clock_mhz` | 2184 |

## Notes

- Sustained c = 4: four workers in a closed loop, 32 requests of 512 greedy tokens, drafter dspark as booted (docs/HARNESS-NEXT.md section 7). Medians over 3 runs (g-all, g-pass2-rolling, g-pass3-rolling); the span number (decode_c4_tok_s) and the steady window figure are from the same runs. The ladder pairs this record with the newest passing `spec` record of the same engine commit.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/bench-decode.err`
- `g-all.d/bench-decode.json`
- `g-all.d/bench-decode.out`
- `g-all.d/bench-prefill.err`
- `g-all.d/bench-prefill.json`
- `g-all.d/bench-prefill.out`
- `g-all.d/cache.json`
- `g-all.d/cache.log`
- `g-all.d/capacity.log`
- `g-all.d/capacity.txt`
- `g-all.d/gates.txt`
- `g-all.d/memory.log`
- `g-all.d/memwatch-begin.txt`
- `g-all.d/memwatch.txt`
- `g-all.d/metrics.txt`
- `g-all.d/needle-0.1.json`
- `g-all.d/needle-0.5.json`
- `g-all.d/needle.log`
- `g-all.d/smoke.log`
- `g-all.d/spec.json`
- `g-all.d/spec.log`
- `g-all.d/vision-record.log`
- `g-all.d/vision.json`
- `g-all.d/vision.log`
- `g-all.out`
- `g-pass2-rolling.d/bench-rolling.err`
- `g-pass2-rolling.d/bench-rolling.json`
- `g-pass2-rolling.d/bench-rolling.out`
- `g-pass2-rolling.d/gates.txt`
- `g-pass2-rolling.d/metrics.txt`
- `g-pass2-rolling.out`
- `g-pass3-rolling.d/bench-rolling.err`
- `g-pass3-rolling.d/bench-rolling.json`
- `g-pass3-rolling.d/bench-rolling.out`
- `g-pass3-rolling.d/gates.txt`
- `g-pass3-rolling.d/metrics.txt`
- `g-pass3-rolling.out`
