# final-cache-images: gate `cache-images`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench cache-images   (bench/tools/cache_images.py: cached == uncached for prompts that carry images; prefix cache, vision and drafter on)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/cache_images.py | after-text: filler, image, question: cold, warm (text before the image served); every cached reply equals its uncached reference | 1 | 0 |
| bench/tools/cache_images.py | other-image: another image: its own reply, the image not served; every cached reply equals its uncached reference | 1 | 0 |
| bench/tools/cache_images.py | text-first: text cached first, then the same text with an image; every cached reply equals its uncached reference | 1 | 0 |
| bench/tools/cache_images.py | image-first: image first: cold, warm, another question; every cached reply equals its uncached reference | 1 | 0 |
| bench/tools/cache_images.py | two-images: two images: cold, warm, only the second changed; every cached reply equals its uncached reference | 1 | 0 |
| bench/tools/cache_images.py | concurrent: two image prompts at once; every cached reply equals its uncached reference | 1 | 0 |
| bench/tools/cache_images.py | the cache served 11 of 14 lookups and saved 40689 tokens | 1 | 0 |
| bench/tools/cache_images.py | no request error | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `cache_images_scenarios` | 6 |
| `cache_images_hits` | 11 |
| `cache_images_lookups` | 14 |
| `cache_images_tokens_saved` | 40689 |
| `cache_images_evictions` | 0 |
| `cache_images_after_text_warm_cached` | 3411 |
| `cache_images_image_first_warm_cached` | 3399 |
| `cache_images_two_images_warm_cached` | 5043 |

## Notes

- Boot g, shipping defaults (SPEC=dspark PREFIX_CACHE=on VISION=on): the drafter serves the image prompts and its state rides in the cache snapshots. Every cached reply equal to its uncached reference, id for id (48 greedy tokens).

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-cache-images.d/cache-images.json`
- `g-cache-images.d/cache-images.log`
- `g-cache-images.d/gates.txt`
- `g-cache-images.d/metrics.txt`
- `g-cache-images.out`
