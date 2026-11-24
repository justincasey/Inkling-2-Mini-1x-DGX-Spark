# final-needle: gate `needle`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench all   (gate needle: bench/tools/needle.py --tokens 262144 --max-model-len 262144, depth 0.1 and 0.5)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/needle.py | 262K window, depth 0.1: the code OTTER-2624-OTTER found in the reply | 1 | 0 |
| bench/tools/needle.py | 262K window, depth 0.5: the code OTTER-2624-OTTER found in the reply | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `prefill_s_depth_0_1` | 191.788 |
| `prompt_tokens_depth_0_1` | 260019 |
| `wall_s_depth_0_1` | 203.553 |
| `prefill_s_depth_0_5` | 191.8 |
| `prompt_tokens_depth_0_5` | 260019 |
| `wall_s_depth_0_5` | 197.855 |
| `needles_found` | 2 |
| `needles_total` | 2 |
| `prompt_tokens` | 260019 |

## Notes

- Window of profile bootstrap (262,144 tokens), the shipping boot g; max_tokens 1024 (the pack reasons before it answers). The 1,048,576-token needle of profile max is not part of the final run (docs/FIT.md, record needle-window).

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/gates.txt`
- `g-all.d/needle.log`
- `needle-0.1.json`
- `needle-0.5.json`
