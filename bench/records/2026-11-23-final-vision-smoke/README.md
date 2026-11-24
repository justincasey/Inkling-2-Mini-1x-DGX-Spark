# final-vision-smoke: gate `vision-smoke`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh smoke   (boot p: the image check)`
- `./inkling2mini.sh bench all   (boot g: gates.py vision against bench/reference-vision.json)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/gates.py vision | image red-square: the reply names the colour(s) | 1 | 0 |
| bench/tools/gates.py vision | image blue-circle: the reply names the colour(s) | 1 | 0 |
| bench/tools/gates.py vision | image green-yellow: the reply names the colour(s) | 1 | 0 |
| bench/tools/gates.py diff | image replies equal the pinned reply-vision.json replies (ids and finish reason) | 3 | 0 |
| bench/tools/gates.py diff | text replies equal the pinned reference-vision.json replies | 4 | 0 |
| bench/tools/smoke.py | image: a red square on white is named red (boot p) | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `image_prompt_tokens_min` | 70 |
| `image_prompt_tokens_max` | 72 |
| `image_replies_wall_s_max` | 0.877 |
| `vision_replies_identical` | 7 |
| `planned_tower_gib` | 0.79 |
| `planned_vision_reserve_gib` | 0.28 |
| `gpu_process_mib_vision_on` | 46690 |

## Notes

- VISION=on at the shipping defaults (drafter and prefix cache on as well). The reference-vision.json replies were pinned by boot p; boot g reproduced them byte for byte (token ids and finish reasons of three synthetic images and four text prompts).

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/gates.txt`
- `g-all.d/vision.json`
- `g-all.d/vision.log`
- `p-smoke.out`
