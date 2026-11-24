# final-speed-decode: gate `speed-decode`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench all   (decode phase: warm, 1 stream over four kinds, 4 and 8 at once)`
- `./inkling2mini.sh bench decode   (g-pass2-decode)`
- `./inkling2mini.sh bench rolling   (g-pass2-rolling)`
- `./inkling2mini.sh bench decode   (g-pass3-decode)`
- `./inkling2mini.sh bench rolling   (g-pass3-rolling)`
- `./inkling2mini.sh bench decode-4k   (c = 1, cold 4,096-token prompt, 3 runs)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/bench.py | decode: four kinds x 5 runs, every reply equal to the pinned reply (PASS pin), concurrent == solo | 5 | 0 |

## Metrics

| metric | value |
|---|---:|
| `decode_mean4_tok_s` | 91.438 |
| `decode_prose_tok_s` | 52.023 |
| `decode_code_tok_s` | 129.754 |
| `decode_structured_tok_s` | 88.472 |
| `decode_json_tok_s` | 95.538 |
| `decode_c4_aggregate_tok_s` | 113.175 |
| `decode_c8_aggregate_tok_s` | 134.764 |
| `decode_tokens_per_round` | 2.441 |
| `decode_acceptance` | 0.543 |
| `decode_runs` | 5 |
| `decode_mean4_tok_s_min` | 90.021 |
| `decode_mean4_tok_s_max` | 92.281 |
| `decode_c1_4k_tok_s` | 98.068 |
| `gpu_clock_mhz` | 2184 |

## Notes

- Drafter dspark, prefix cache on, vision on (the shipping defaults), boot g (fresh boot after the pin boot), GPU capped at 2,200 MHz. Medians over every run of the phase: `bench all` and the later `bench decode` / `bench rolling` passes (each starts with its own discarded warm-up); decode_mean4_tok_s is the mean over prose, code, structured and json of the median run. decode_c4_aggregate_tok_s is the span number (all tokens over the span to the last reply), decode_c8_aggregate_tok_s is 8 requests on 4 slots. decode_c1_4k_tok_s is c = 1 on a cold 4,096-token prompt (256 tokens), median of 3.

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
- `g-decode-4k.d/decode-4k.json`
- `g-decode-4k.d/decode-4k.log`
- `g-decode-4k.d/gates.txt`
- `g-decode-4k.d/metrics.txt`
- `g-decode-4k.out`
- `g-pass2-decode.d/bench-decode.err`
- `g-pass2-decode.d/bench-decode.json`
- `g-pass2-decode.d/bench-decode.out`
- `g-pass2-decode.d/gates.txt`
- `g-pass2-decode.d/metrics.txt`
- `g-pass2-decode.out`
- `g-pass2-rolling.d/bench-rolling.err`
- `g-pass2-rolling.d/bench-rolling.json`
- `g-pass2-rolling.d/bench-rolling.out`
- `g-pass2-rolling.d/gates.txt`
- `g-pass2-rolling.d/metrics.txt`
- `g-pass2-rolling.out`
- `g-pass3-decode.d/bench-decode.err`
- `g-pass3-decode.d/bench-decode.json`
- `g-pass3-decode.d/bench-decode.out`
- `g-pass3-decode.d/gates.txt`
- `g-pass3-decode.d/metrics.txt`
- `g-pass3-decode.out`
- `g-pass3-rolling.d/bench-rolling.err`
- `g-pass3-rolling.d/bench-rolling.json`
- `g-pass3-rolling.d/bench-rolling.out`
- `g-pass3-rolling.d/gates.txt`
- `g-pass3-rolling.d/metrics.txt`
- `g-pass3-rolling.out`
