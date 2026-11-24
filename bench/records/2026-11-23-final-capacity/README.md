# final-capacity: gate `capacity`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench all   (gate capacity: the engine's capacity line against the profile)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| inkling2mini.sh | bench check capacity, profile bootstrap: lanes_at_max_context >= 4 and kv_tokens >= 4 x 262,144 | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `lanes_at_max_context` | 4 |
| `kv_tokens` | 1048576 |
| `kv_gib` | 18.16 |
| `free_after_gib` | 101.27 |
| `prefix_gib` | 9.47 |
| `spare_gib` | 69.41 |
| `tower_gib` | 0.79 |
| `slots` | 4 |
| `kv_bytes_per_token` | 18576 |

## Notes

- Profile bootstrap (4 slots x 262,144), drafter dspark, prefix cache on, vision on: the plan line of the boot g log (raw/capacity.txt).

## Raw output (`raw/`)

- `cap-proof.txt`
- `capacity.txt`
