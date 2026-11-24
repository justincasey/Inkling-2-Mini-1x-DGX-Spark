# spec-cache: the prefix cache carries the drafter (gate `spec-cache`, plan 4.5 x 4.6)

Engine `761b14388d12d0cecf3a0186876f6952c35a9c7c`, node-2, `SPEC=dspark PREFIX_CACHE=on VISION=off`, bootstrap profile (16 snapshot slots, 1,536 pages, reach
393,216 tokens), no ungated override. A snapshot now holds, after the rings and the conv states, the drafter's two rings and `f_rnn` (8,458,752 B: a slot is 179,294,720 B,
the memory model's figure, checked at boot); the drafter's plane lives in the target's pages, shared by reference, and a restore copies the plane page with the target's
partial last page. The restored slot keeps the drafter's book.

| claim | evidence (`raw/`) | result |
|---|---|---|
| `gates.py cache` with the drafter on: short, medium, long, xlong and 70K prompts uncached == cold == warm; two shared-prefix pairs (2,082 and 73,247 shared tokens); the warm replies served 471, 1,980, 8,284 and 73,104 cached tokens | `cache-dspark.out/json` | pass |
| the edge cases with the cache on: EOS, `max_tokens`, ring wraps, four mixed concurrent requests (50 cases) | `specedge-cache.out/json` | pass |
| at the rank: a restored slot proposes what an uncached slot does (cuts 512, 700 (a partial page: the plane page copy), 1,024, 1,100), the proposals stay equal through a verify round, and six draft-verify rounds of a restored slot equal the uncached slot's (committed tokens, proposals, window candidates) | `gpu-snapshot-tests.txt` (`LOOP_SPEC_snapshot`, `LOOP_SPEC_cached_decode`) | pass |

Not proven: eviction under drafted load beyond the stress of the cache record (`prefix-cache`), images (the combination is refused), speed with the cache + drafter together.

## Commands
```sh
SPEC=dspark PREFIX_CACHE=on VISION=off ./inkling2mini.sh start; python3 bench/tools/gates.py cache cache-dspark.json --max-tokens 128
python3 bench/tools/specedge.py --out specedge-cache.json --only eos_in_block,max_tokens_in_block,ring_wrap,concurrent_mixed
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='REXL3_INKLING2_LOOP_DIR=<pack> LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_spec a_snapshot_restores a_cached_decode_equals
```
