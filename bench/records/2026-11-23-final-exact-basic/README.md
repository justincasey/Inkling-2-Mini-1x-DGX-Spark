# final-exact-basic: gate `exact-basic`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `boot p: ./inkling2mini.sh bench exactness   (PREFILL_CHUNK=4096)`
- `boot g: ./inkling2mini.sh bench exactness   (fresh boot, 4096)`
- `boot c: PREFILL_CHUNK=1000 ./inkling2mini.sh start; PREFILL_CHUNK=1000 ./inkling2mini.sh bench exactness`
- `python3 bench/tools/exactness.py diff p g; diff p c   (local)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/exactness.py | concurrent==solo (boot p, chunk 4096, 4 prompts, 4 lanes) | 4 | 0 |
| bench/tools/exactness.py | concurrent==solo (boot g, chunk 4096, 4 prompts, 4 lanes) | 4 | 0 |
| bench/tools/exactness.py | concurrent==solo (boot c, chunk 1000, 4 prompts, 4 lanes) | 4 | 0 |
| bench/tools/exactness.py | chunk width: 4096 vs 1000 byte-equal (replies and finish reasons) | 4 | 0 |
| bench/tools/exactness.py | boot to boot: boot p vs boot g (4096) byte-equal | 4 | 0 |

## Metrics

| metric | value |
|---|---:|
| `prompts` | 4 |
| `max_prompt_tokens` | 8334 |
| `new_tokens` | 128 |
| `exactness_replies_identical` | 4 |
| `solo_xlong_s` | 5.821 |

## Notes

- Four fixed prompts (about 16, 521, 2,030 and 8,334 tokens), 128 greedy tokens, solo, one at a time; boots p and g with PREFILL_CHUNK 4096, boot c with 1000; shipping defaults otherwise. prompt_logprobs is refused by this engine, so the comparison rests on the generated ids and finish reasons (the tool says so). A record of this gate is taken first on each boot, before anything that reads the prefix cache.

## Raw output (`raw/`)

- `c-exact.out`
- `cap-proof.txt`
- `concurrent-c-chunk1000.log`
- `concurrent-g-chunk4096.log`
- `concurrent-p-chunk4096.log`
- `diff-boot-to-boot.txt`
- `diff-chunk-4096-vs-1000.txt`
- `exactness-c-chunk1000.json`
- `exactness-g-chunk4096.json`
- `exactness-p-chunk4096.json`
- `g-exact.out`
- `p-exact.out`
