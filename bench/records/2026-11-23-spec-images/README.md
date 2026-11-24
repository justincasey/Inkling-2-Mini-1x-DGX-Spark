# spec-images: the drafter with images and with the cache over images, on the merged engine (gate `spec-images`, plan 4.6 x 4.5 x 4.7)

Engine `e9eeea9683ed9c897ece83e5eac103bad6e40ee4` = the integration head `e2b8414` (the cache with images, the staged memory plan, the 2-tile MoE
prefill) merged with the drafter lane (`85fb069` adds a changelog fragment only), node-2, GPU locked at 2,200 MHz, decode graphs on (the default), no ungated
override, bootstrap profile. Boot A is the shipping default `SPEC=dspark PREFIX_CACHE=on VISION=on`; boot B the serial reference `SPEC=none PREFIX_CACHE=off VISION=on`;
boot C `SPEC=dspark PREFIX_CACHE=off VISION=on` (the image cases without the cache). Greedy everywhere.

## What the code does with image rows (the refusal was lifted on this reading, and the runs below are its proof)
* Image rows are ordinary rows of the forward: `forward` writes the four tap planes for all `n` rows of a chunk, the splice rows included (the splice replaces the embedding
  of the rows before visit 0; taps 0-2 read the partial block entering visits 5, 23, 41, tap 3 the final norm, whatever put the row there).
* `prefill_chunk_mm` feeds the drafter for every segment of the chunk after the forward, with or without splices; `commit` uses the rows' taps and positions only, never token ids,
  and the Mini has no M-RoPE rows (positions are consecutive, one row an image position).
* A snapshot is taken between chunks (the engine never cuts inside an image: its cache key replaces an image's placeholder run with tokens derived from the image's content), so
  it holds the drafter's rings, `f_rnn` and book as of the cut, and a restore after an image puts the same state back; the plane pages of the image's rows are shared by reference like any page.
* What could have differed and did not: a proposal's anchor is the last prompt token (text), the drafter's `refuse` checks only its own length against the anchor position.

## Proofs
| # | Property | Evidence (`raw/`) | Result |
|---|---|---|---|
| 1 | drafted == serial (text, 5 prompts to 73K + 4 concurrent) with the cache and the tower loaded; the drafted boot's replies == the serial boot's | `A-spec.out`: 9 of 9 equal; `spec-diff.out`: `identical` | pass |
| 2 | image replies: 3 synthetic images and 4 text prompts, drafted + cache on == serial (`gates.py vision`) | `vision-diff.out` (A vs B): `identical`; without the cache `vision-diff-nocache.out` (C vs B): `identical` | pass |
| 3 | `cache_images.py` with the drafter on: after-text, other-image, text-first (an image under a cached text prefix), image-first, two-images, concurrent; every cached reply equals its uncached reference, and all 28 replies equal the serial boot's | `A-cache-images.out` pass (11 hits, 40,689 tokens saved, 0 evictions); `cache-images-compare.out`: 28 replies identical; without the cache `cache-images-compare-nocache.out`: 28 identical | pass |
| 4 | `gates.py cache` with the drafter and the tower on: uncached == cold == warm, the 70K prompt, two shared-prefix pairs | `A-cache.out` pass (reach 393,216 tokens) | pass |
| 5 | `specedge.py`: all 65 cases (EOS, stop string, `max_tokens`, ring wraps, 12,345-token prompt, rollback histogram, 4 concurrent mixed) | `A-specedge.out`: 65 cases, 0 failures | pass |
| 6 | `exactness.py`: the 4 prompts' 128-token replies and prompt logprobs: replies sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` on the drafted boot AND the serial boot; `diff` identical; `concurrent`: pass | `A-exactness.json`, `B-exactness.json`, `exactness-diff.out`, `A-exactness-concurrent.out` | pass |
| 7 | the serial path is byte-identical: dump sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0`, chunk-width logits `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a`, long-context top-1 157 / 23.62496, 37 graph windows `6d0a555b92e02953c9e26efecdd9dece0ab211030ba8947cbe03b8407846c06b` | `serial-rank.txt`, `serial-graphs.txt` | pass |
| 8 | the drafter's GPU tests on the merged tree: taps equal the oracle dump planes, a rank with a drafter computes the same bits as one without, graph replay == eager with taps on, the drafter follows prefill and commits, proposals independent of chunking, a snapshot restores the drafter's proposals (cuts 512 / 700 / 1,024 / 1,100), cached decode == uncached through windows | `gpu-spec-tests.txt` | pass (7 of 7) |
| 9 | host: the family's tests 100, the Small's 31 unchanged (`REXL3_NO_CUDA=1 ... families::inkling2::`), the server binary's unit tests 11, `stop_in_batch` 5; the recipe's `tests/test_recipe.sh` 154, `test_bench.py` 27, `test_gates.py` 69 | `host-tests-*.txt`, `recipe-tests.txt` | pass |

## Memory with everything on (bootstrap profile; `A-log.txt`, `A-memwatch.txt`, `docs/FIT.md` "Measured")
Capacity line: `weights_gib=13.87 free_after_gib=93.63 reserve_gib=4.72 state_mib_per_slot=174.4 kv_bytes_per_token=18576 drafter=dspark prefix_cache=on prefix_entries=16 prefix_pages=1536 prefix_gib=9.47 prefix_reach_tokens=393216 spare_gib=61.80 tower_gib=0.79 vision_reserve_gib=0.28 weights_alloc_padding_gib=6.70 step_buffers_gib=2.70 fixed_overhead_gib=9.42 predicted_ready_available_gib=62.61`.
`MemAvailable` at `mem_stage engine_ready` 65,095 MiB = 63.57 GiB: the prediction is 0.96 GiB low; the guard minimum after the gate runs is 63,000 MiB (61.52 GiB), 1.09 GiB under it.
Without the cache (boot C) the prediction is 73.08 against 73.45 GiB (+0.37); serial (boot B) 75.57 against 76.10 (+0.53). The drafter's stage took 3,142.6 MiB (3,817.5 with the cache's pages)
against 2.49 GiB (3.25) priced: the prediction was left as the plan computes it.

## Not proven
* A rank-level GPU test of the drafter over spliced image rows (the server-level gates above exercise it: the three synthetic images, two images in a prompt, an image under a cached prefix).
* Real photographs (the synthetic images are the gate's: a red square, a blue circle, green and yellow halves); the drafter's acceptance on image prompts is in `spec-final`.
* TP, the oracle (drafted == serial says nothing about it), eviction under drafted image load beyond `cache_images.py`'s counters.

## Commands
```sh
REXL3_SRC=<engine worktree> ./inkling2mini.sh build <40-hex>   # engine e9eeea9683ed9c897ece83e5eac103bad6e40ee4 (the merge of the integration head e2b8414 and the drafter lane)
SPEC=dspark PREFIX_CACHE=on VISION=on ./inkling2mini.sh start   # EXTRA_ENV empty
python3 bench/tools/gates.py spec A-spec.json --max-tokens 128; gates.py vision A-vision.json --max-tokens 64; cache_images.py --out A-cache-images.json; gates.py cache A-cache.json --max-tokens 128
python3 bench/tools/specedge.py --out A-specedge.json; exactness.py record A-exactness.json --max-tokens 128; exactness.py concurrent --max-tokens 64
SPEC=none PREFIX_CACHE=off VISION=on ... start; gates.py spec B-spec.json --serial-boot; gates.py spec-diff; gates.py vision B-vision.json; gates.py diff; cache_images.py --reference-only; cache_compare.py --images A B; exactness.py diff
SPEC=dspark PREFIX_CACHE=off VISION=on ... start; gates.py vision C-vision.json; gates.py diff C B; cache_images.py --reference-only; cache_compare.py --images C B
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='REXL3_INKLING2_LOOP_DIR=<pack> LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_spec taps_equal no_drafter graph_replay the_drafter_follows draft_equals a_snapshot_restores a_cached_decode_equals
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_rank forward_oracle_gate long_context_gate
NODE=node-2 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='LOOP_LONG_IDS_FILE=<lab>/ids_long.json' scripts/ops_dev.sh test inkling2_loop_graphs graph_replay_equals capture_refusal
REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked families::inkling2::; cargo test -p gb-model --lib families::inkling2_loop; cargo test -p gb-server --bin rexl3-serve; cargo test -p gb-server --test stop_in_batch
bash tests/test_recipe.sh; python3 tests/test_bench.py; python3 tests/test_gates.py
```
