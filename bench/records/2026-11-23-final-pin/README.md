# final-pin: gate `pin`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench pin   (boot p: writes or reproduces bench/reference.json and bench/reference-vision.json)`
- `./inkling2mini.sh bench all   (boot g, a fresh boot: every reply compared with the pinned references)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/bench.py --pin | decode x4 and cold 32K/128K/224K replies pinned or reproduced (7; 0 reproduced byte for byte, added ['decode/code', 'decode/json', 'decode/prose', 'decode/structured', 'prefill/128k', 'prefill/224k', 'prefill/32k']) | 7 | 0 |
| bench/tools/bench.py | pin run: every concurrent (c=4, c=8) reply equal to its solo reply | 1 | 0 |
| bench/tools/bench.py --ref | boot g (fresh boot): decode x4, c=4/c=8/rolling and cold 32K/128K/224K replies equal the pinned replies: PASS pin x2 (decode run and prefill run) | 2 | 0 |
| bench/tools/gates.py diff | boot g: vision replies equal bench/reference-vision.json | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `reference_replies` | 7 |
| `reference_vision_replies` | 7 |
| `pin_replies` | 7 |
| `pin_replies_reproduced_at_pin` | 0 |

## Facts

- bench/reference.json sha256 e6277c895baa4aeaade4361b0ab9501273d418f57a2cab4cf9a998af2343c2e7: 7 replies (decode/code, decode/json, decode/prose, decode/structured, prefill/128k, prefill/224k, prefill/32k)
- bench/reference-vision.json sha256 b7f8283ba9c0839afe888c176e9e38f7d446730b54a4bc803d8f50e72b8d57b5: 3 image replies, 4 text replies
- pin run: 7 replies pinned, added ['decode/code', 'decode/json', 'decode/prose', 'decode/structured', 'prefill/128k', 'prefill/224k', 'prefill/32k']; 0 already pinned ones reproduced

## Notes

- Boot to boot: references written (or reproduced) by boot p of this run, then every pinned reply reproduced by boot g, a fresh boot of the same build and configuration. The references name the engine they were pinned on (meta.engine). bench/reference*.json are committed by the lead with this record.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/bench-decode.out`
- `g-all.d/bench-prefill.out`
- `g-all.d/gates.txt`
- `p-pin.d/bench-pin.err`
- `p-pin.d/bench-pin.json`
- `p-pin.d/bench-pin.out`
- `p-pin.d/gates.txt`
- `p-pin.d/metrics.txt`
- `p-pin.d/vision-record.log`
- `p-pin.d/vision.json`
- `p-pin.out`
- `reference-vision.json`
- `reference.json`
