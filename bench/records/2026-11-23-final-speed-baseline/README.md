# final-speed-baseline: gate `speed-baseline`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench pin   (boot p: decode, c=4/c=8, cold 32K/128K/224K)`
- `./inkling2mini.sh bench all   (boot g: decode, c=4/c=8, rolling, cold 32K/128K/224K)`
- `./inkling2mini.sh bench decode   (g-pass2-decode)`
- `./inkling2mini.sh bench rolling   (g-pass2-rolling)`
- `./inkling2mini.sh bench decode   (g-pass3-decode)`
- `./inkling2mini.sh bench rolling   (g-pass3-rolling)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/bench.py | decode x6 runs: every reply produced and equal to the pinned/solo reply, concurrent == solo | 6 | 0 |
| bench/tools/bench.py | prefill x2 runs: cold 32K, 128K, 224K replies equal to the pinned replies | 2 | 0 |

## Metrics

| metric | value |
|---|---:|
| `decode_c1_tok_s` | 91.574 |
| `decode_c4_tok_s` | 112.977 |
| `decode_c8_tok_s` | 135.764 |
| `decode_tokens_per_round` | 2.472 |
| `decode_acceptance` | 0.542 |
| `prefill_32k_tok_s` | 2380.1 |
| `prefill_128k_tok_s` | 1789.55 |
| `prefill_224k_tok_s` | 1444.5 |
| `prefill_32k_s` | 13.7675 |
| `prefill_128k_s` | 73.2441 |
| `prefill_224k_s` | 158.791 |
| `decode_ctx32k_tok_s` | 88.671 |
| `decode_ctx128k_tok_s` | 67.673 |
| `decode_ctx224k_tok_s` | 63.2225 |
| `gpu_clock_mhz` | 2197 |
| `passes_decode` | 6 |
| `passes_prefill` | 2 |

## Notes

- Shipping defaults (SPEC=dspark PREFIX_CACHE=on VISION=on), profile bootstrap, GPU capped at 2,200 MHz. Medians over every run that measured the metric: decode over p-pin, g-all, g-pass2-decode, g-pass2-rolling, g-pass3-decode, g-pass3-rolling; prefill over p-pin, g-all (each run starts with its own discarded warm-up). The numbers are single-boot figures per boot; the spread is in the per-run metrics.txt.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/gates.txt`
- `g-all.d/metrics.txt`
- `g-all.out`
- `g-pass2-decode.d/gates.txt`
- `g-pass2-decode.d/metrics.txt`
- `g-pass2-decode.out`
- `g-pass2-rolling.d/gates.txt`
- `g-pass2-rolling.d/metrics.txt`
- `g-pass2-rolling.out`
- `g-pass3-decode.d/gates.txt`
- `g-pass3-decode.d/metrics.txt`
- `g-pass3-decode.out`
- `g-pass3-rolling.d/gates.txt`
- `g-pass3-rolling.d/metrics.txt`
- `g-pass3-rolling.out`
- `p-pin.d/gates.txt`
- `p-pin.d/metrics.txt`
- `p-pin.out`
