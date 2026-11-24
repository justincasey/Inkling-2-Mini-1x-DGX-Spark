# final-engine-tests: gate `engine-build`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh doctor`
- `./inkling2mini.sh build   (no revision: the pin of config/pins.sh)`
- `./inkling2mini.sh engine-test host`
- `./inkling2mini.sh engine-test goldens   (the Small's kernel hash tests, on the GPU)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| host: gb-model families::inkling2 + inkling2_loop |  | 146 | 0 |
| host: gb-cuda inkling2 |  | 25 | 0 |
| host: gb-server formats::inkling2 |  | 1 | 0 |
| host: gb-server inkling2_format |  | 18 | 0 |
| goldens: golden attn |  | 2 | 0 |
| goldens: golden ops |  | 1 | 0 |
| goldens: golden moe |  | 1 | 0 |
| goldens: golden vision |  | 1 | 0 |
| goldens: golden dspark |  | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `host_tests_passed` | 190 |
| `golden_tests_passed` | 6 |
| `build_seconds` | 115 |
| `engine_test_host_seconds` | 76 |
| `engine_test_goldens_seconds` | 70 |

## Facts

- binary sha256 637d50b40f75378de92407eda48265d9f4dc912076a308c7c28dd1db37c7f7d2

## Notes

- Engine built from the pinned commit with `git archive` (cargo --release --locked, sm_121); `engine.lab_build` is false. The goldens run the Small's kernel hash tests (attn, ops, moe, vision, dspark) against the checked-in hashes: the drafter hash test reads the Small's pack read-only (its mtp.safetensors is bf16, the Mini's is exl3). The GPU cap line precedes each of build, host and goldens.

## Raw output (`raw/`)

- `build.out`
- `cap-proof.txt`
- `doctor.out`
- `engine-goldens.out`
- `engine-host.out`
