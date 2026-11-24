# final-smoke: gate `smoke`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh smoke   (boot p: shipping defaults)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/smoke.py | chat | 1 | 0 |
| bench/tools/smoke.py | completion | 1 | 0 |
| bench/tools/smoke.py | tool | 1 | 0 |
| bench/tools/smoke.py | think-on | 1 | 0 |
| bench/tools/smoke.py | think-off | 1 | 0 |
| bench/tools/smoke.py | image | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `smoke_checks_passed` | 6 |
| `smoke_checks_skipped` | 0 |
| `think_on_reasoning_tokens` | 188 |
| `think_off_reasoning_tokens` | 105 |

## Notes

- Boot p (the first boot of the final run), SPEC=dspark PREFIX_CACHE=on VISION=on: the image check ran (not skipped).

## Raw output (`raw/`)

- `cap-proof.txt`
- `p-smoke.out`
- `p-start.out`
