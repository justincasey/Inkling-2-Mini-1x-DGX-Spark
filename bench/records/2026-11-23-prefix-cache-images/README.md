# prefix-cache-images: the Mini's prefix cache with the vision tower on, on the engine with CUDA graphs (gate `cache-images`)

Result **pass** (`meta.json`). Engine lab build `6b6b824431f1c4cb52566828f726555cdeec6ce2` (branch `inkling2-loop-state` after merging the
integration head `918e685`: the prefix cache, GraphLane's decode CUDA graphs and side-stream overlap, the vision wiring; clean, `dirty` false),
recipe `61ff31e`, node-3, GPU SM clock locked (2,197 MHz read), profile `bootstrap`. Boot A (everything on): `VISION=on PREFIX_CACHE=on
SPEC=none`, graphs and overlap at their default (on). Boot B (the reference): `VISION=on PREFIX_CACHE=off`, `REXL3_INKLING2_LOOP_GRAPHS=0
REXL3_INKLING2_LOOP_OVERLAP=0` (every window eager, one stream). The server is stopped and the node idle at the end.

## What image rows needed

Nothing, in the rank or the backend; the refusal was the only thing in the way. Read from the code and then proven below:

* **Image rows are ordinary rows of the forward.** `prefill_chunk_mm` runs the same `forward`; the only difference is that after the embedding
  and its norm, `LoopSplice::write` overwrites the chunk's placeholder rows of the step's first residual buffer with the feature rows (one
  host-to-device copy). Every K/V write, ring write and conv commit after that is the text path's, so the rings, conv state and int8 pages a
  snapshot copies after an image are what an uncut prefill holds (proven: `prefix_images_exact`).
* **No per-slot image state.** The rank keeps no feature buffer, pending-splice list or image bookkeeping between chunks (the splices are an
  argument of one call); the backend keeps none either: `run_chunk` builds `LoopSplice::from_span(span).clip(first, end)` for every span of the
  request, in prompt rows, for each chunk it runs. A restored slice starts at `done` (any row, off the chunk grid or not): spans that end before
  it clip to nothing, a span that straddles a chunk boundary clips to its own rows (`src`, `dst`, `n`), and the features stay alive in the
  request's `ImageSpan` for the whole request (the h2d copy reads them from the host). The engine never restores inside an image.
* **The key and the cut are the engine's generic rules** (`cache_tokens`, `outside_images`, `ImageSpan.key` = the encoder's `image_content_hash` of
  the pixel hash, the canvas and the preprocessing id): an image's placeholder run is replaced by tokens derived from its key, so the same text
  around another image is another prefix; a cut that falls inside an image moves to the image's start or is dropped.
* **What the policy does with a short image prompt (a finding, not a bug):** the end cut is 64 tokens before the end of the prompt. A prompt of
  text, one image and a one-line question has its image inside those 64 tokens, so its end entry stops before the image and the image is
  prefilled again each time (`after-text` below: 3,411 of 3,475 tokens cached, the image is in the rest). An image with text after it
  (`image-first`: 3,399 cached, `two-images`: 5,043 cached, past the first image) is cached and restored.

## Proofs

**Rank, real pack, synthetic features** (`raw/gpu-merged-engine-graphs-on.log`; seeded bf16 rows, 2,304 wide, values in [-2, 2], 64 and 100
placeholder rows at prompt rows 300 and 1,200):

1. `prefix_images_exact`: a 1,800-token prompt with two images; cuts at 250 (before the first image), 364 (right after it), 700 (between the two), 1,200
   (the second image's start: where the planner moves a cut that falls inside it) and 1,300 (after it), cold grid and resumed grid different
   (resumed chunks start at the cut, so the second image meets them at other rows: cut 700 with 129-row chunks puts 42 of its rows in the
   chunk [1,158, 1,287) and 58 in the next). At every cut the restored state at the cut (K/V of all positions, the rings' last 1,024 rows, all
   conv state), the last row after the rest of the prefill, the state after it, the first decode step's full logits plane and three more decode
   steps equal the uncut run bit for bit: `last_row_sha 2680fdf08f6b07ed state_sha 9111fada329906d6 decode_sha a938a63f4df7acbf` at all five.
2. `prefix_images_another_image_changes_the_reply` (the negative): the same text under another image's features gives another reply
   (`a_sha c3ca803ef0fbf925`, `b_sha 1539e10cec33681b`), so the equalities are not vacuous; and a restore from the snapshot taken after image A,
   continued with image B in the image list (before the restored length, so not spliced), gives A's reply and not B's: the snapshot carries
   the image's effect, the restore does not re-read it.
3. The merged engine reproduces the text proofs of `../2026-11-23-prefix-cache/` with the graphs on: `prefix_exact_at_cuts` (20 cuts: the digests
   are the same as before the merge, every one), `prefix_logits_planes_equal`, `prefix_restore_after_eviction`,
   `prefix_restore_beside_decoding_streams`, `prefix_long_prompt` all pass. The 20-cut run reports `LOOP_PFX_graphs captures 10 replays 140
   eager_windows 10 capture_failures 0`: the cold runs' decode steps are eager or the capture, the restored runs' are replays, and they are
   bit-equal. The image run: `captures 2 replays 36`.

**Host** (`raw/host/`): `families::inkling2_loop::` 85 passed, the Small's 31 of 31, `family::` 39, `plan::` 66. New: the real `Engine` over the
simulated rank whose image rows are hashes of their feature rows (a wrong source offset or the wrong image changes the state): an image before the
cut (restored after it, equal), a cut inside an image (it moves to the image's start: 300 of 404 tokens), the same text under another image (not
shared, its own reply, a different reply), an image prompt after a cached text-only prefix, a restore at row 856 (off the 128-row grid) that
skips image 1 and splices image 2 at its rows, on five grids; and the plan beside the tower (the tower and its reserve are paid from the
spare memory, the cache and the KV pool are the same).

**Live, boot A** (`raw/*-on.*`):
* `./inkling2mini.sh bench cache` PASS on the merged engine with the tower loaded (`raw/gate-cache/`): five prompts (31 to 73,169 tokens), warm
  hits 472, 1,981, 8,285 and **73,105** tokens, the shared-prefix pairs 2,028 and **73,193** tokens, 0 cached tokens on every uncached and cold reply.
* `bench/tools/cache_images.py` PASS (`raw/cache-images-on.json`, six scenarios, every cached reply equals its uncached one id for id; 0 on every
  uncached and cold reply; 14 lookups, 11 hits, 40,689 tokens saved, 0 restore failures): `after-text` (3,475-token chat prompt: warm 3,411
  cached); `other-image` (the same text and question under another image: 3,411 cached, its own reply, differs from the first image's);
  `text-first` (a text-only prompt cached, then the same filler with an image: 3,375 cached, equal); `image-first` (warm 3,399; another question
  about the same image and filler: 3,399); `two-images` (cold 3,411 shared with an earlier prompt, warm 5,043; the same prompt with only the second
  image changed: 5,043, equal); `concurrent` (two image prompts at once, 3,411 cached each, each equal to its uncached reply).
* `bench/tools/cache_stress.py` PASS (`raw/cache-stress-on.json`): 24 prompts over 16 slots, reverse re-ask 12 hits and 12 misses, 45
  evictions, every reply equal; 4 concurrent streams (3 restores, 1 cold) equal their solo runs. (Image prompts are mixed in by `cache_images.py`,
  not by this tool.)
* `gates.py vision`: three image replies (`Red square.`, `Blue circle.`, `Left: green; right: yellow.`, each naming its colour) and four text replies recorded.
* Graph counters at the end (`raw/metrics-on.txt`): 10,184 replays, 10 captures, 0 capture failures; cache counters 102 lookups, 38 hits, 53
  evictions, 0 restore failures, 16 entries.

**Boot A against boot B, byte for byte** (`raw/compare-*.txt`; B has no cache, no graphs, no overlap):
* `cache_compare.py --images`: 28 replies identical: every request of the image scenarios on A (cached or not) equals B's uncached reply of the
  same prompt (`raw/cache-images-off.json` is `--reference-only`).
* `cache_compare.py --gate` (`gates.py cache --nonce xboot --big-tokens 66000` on both; B's own run reports the cache assertions as FAIL because it
  has no cache, which is the point): 23 replies identical, among them the 68,896-token restore and the 8,275- and 1,971-token ones.
* `gates.py diff vision-on vision-off`: three image and four text replies identical.

## Memory with the tower and the cache on (`raw/start-on.log`, `raw/memwatch-*.txt`, `raw/status-*.txt`)

Capacity line, A: `slots=4 kv_pages=4100 kv_tokens=1048576 ... prefix_cache=on prefix_entries=16 prefix_pages=1536 prefix_gib=8.59
prefix_reach_tokens=393216 spare_gib=71.58 tower_gib=0.79`; B: the same with `prefix_cache=off ... prefix_gib=0.00 ... spare_gib=71.58`. The tower's 0.79
GiB and its feature reserve come out of the spare memory first; the cache is the same 16 slots and 1,536 pages as without the tower and the KV pool
is the same.

| | A: tower + cache + graphs | B: tower, no cache, no graphs | text-only, cache on (`../2026-11-23-prefix-cache-speed/`) |
|---|---:|---:|---:|
| server process on the GPU | 49,493 MiB | 40,501 MiB | 48,420 MiB |
| guard minimum `MemAvailable` over the window | 68,439 MiB (66.8 GiB) | 77,682 MiB | 69,417 MiB |

A less B is 8,992 MiB: the cache (8,798.6 MiB planned) and whatever the captured decode graphs and side lanes hold (the remaining ~190 MiB; not
separated here: that needs a graphs-on, cache-off boot). The tower adds 880 MiB to a cache-off server: 40,501 MiB against the 39,621 MiB of the
cache-off, text-only boot of `../2026-11-23-prefix-cache-speed/` (graphs off in both).

## Not proven

A restore that lands inside an image is not exercised (the engine never plans one; the rank's own tests restore only at the planner's cuts: before, at the end of, between, and at the
start of images); an image larger than 100 placeholder rows at the rank level (the live gate uses the pack's real tower at 49 to 58 rows); cached TTFT with images;
a drafter with the cache; the cache with graphs on a window of more than one row (no speculation runs here).
