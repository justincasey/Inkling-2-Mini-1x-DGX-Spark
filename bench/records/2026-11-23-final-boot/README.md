# final-boot: gate `boot`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh doctor`
- `./inkling2mini.sh start   (shipping defaults: SPEC=dspark PREFIX_CACHE=on VISION=on, profile bootstrap, 4 x 262,144)`
- `./inkling2mini.sh status`
- `./inkling2mini.sh memwatch report`
- `./inkling2mini.sh logs 500`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| inkling2mini.sh | doctor: node idle, MemAvailable, pack at full size, engine installed | 1 | 0 |
| inkling2mini.sh | start: /health 200, memory guard armed on the server pid, no level=fatal | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `boot_s` | 21 |
| `engine_ready_t_s` | 15.381 |
| `weights_gib` | 13.55 |
| `weights_load_s` | 6.3 |
| `mem_available_before_mib` | 116736 |
| `mem_available_at_ready_mib` | 71517 |
| `guard_min_mib` | 71447 |
| `guard_loading_min_mib` | 71447 |
| `gpu_process_mib` | 46690 |
| `kv_pages` | 4100 |
| `kv_tokens` | 1048576 |
| `kv_gib` | 18.16 |
| `kv_bytes_per_token` | 18576 |
| `slots` | 4 |
| `planned_free_after_weights_gib` | 99.27 |
| `planned_spare_gib` | 67.41 |
| `planned_prefix_gib` | 9.47 |
| `planned_tower_gib` | 0.79 |

## Notes

- First boot after the build of the pin, shipping defaults, drafter dspark, prefix cache on, vision on (tower loaded). boot_s is the controller's start, doctor included, until /health answered 200. mem_available_before_mib is the doctor's whole GiB x 1024.

## Raw output (`raw/`)

- `cap-proof.txt`
- `doctor.out`
- `p-clock-start.out`
- `p-memwatch-ready.out`
- `p-serve-boot.out`
- `p-start.out`
- `p-status-ready.out`
