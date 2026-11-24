# final-cache: gate `cache`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench all   (gate cache: bench/tools/gates.py cache, prefix cache on, vision on, drafter on)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/gates.py cache | cached == uncached, cold and warm (5 prompts up to 73168 tokens, 2 shared-prefix pairs) | 7 | 0 |
| bench/tools/gates.py cache | 0 cached tokens on every uncached and cold reply | 5 | 0 |
| bench/tools/gates.py cache | the warm reply is served from the cache where the prompt reaches the smallest cut | 4 | 0 |
| bench/tools/gates.py cache | the second reply of a shared-prefix pair is served from the cache | 2 | 0 |

## Metrics

| metric | value |
|---|---:|
| `gate_warm_cached_tokens_70k` | 73104 |
| `gate_warm_cached_tokens_xlong` | 8284 |
| `gate_shared_cached_tokens_1k` | 2027 |
| `gate_shared_cached_tokens_70k` | 73192 |
| `gate_prompts_equal` | 7 |
| `prefix_cache_reach_tokens` | 393216 |

## Notes

- Boot g, shipping defaults (prefix cache on, 16 entries, vision on, drafter on). gates.py cache: per prompt an uncached reference, then cold and warm with the cache; two shared-prefix pairs; 128 greedy tokens each. The with-images cache gate is the record cache-images.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/cache.json`
- `g-all.d/cache.log`
- `g-all.d/gates.txt`
- `g-all.d/metrics.txt`
- `g-all.out`
