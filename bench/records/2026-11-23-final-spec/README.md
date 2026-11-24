# final-spec: gate `spec`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench all   (gate spec: bench/tools/gates.py spec; serial = "draft": false, drafted = the booted dspark)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/gates.py spec | short, medium, long, xlong, 70k: solo drafted == serial (ids and finish reason) | 5 | 0 |
| bench/tools/gates.py spec | short, medium, long, xlong at once: drafted == their solo serial replies | 4 | 0 |
| bench/tools/gates.py spec | every serial reply drafted 0 tokens | 1 | 0 |
| bench/tools/gates.py spec | the drafted replies drafted tokens (the drafter ran) | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `spec_pairs_equal` | 9 |
| `spec_pairs` | 9 |
| `replies_identical` | 9 |
| `drafted_drafted` | 1492 |
| `drafted_accepted` | 711 |
| `drafted_rounds` | 432 |
| `drafted_acceptance` | 0.4765 |
| `drafted_tokens_per_round` | 2.6667 |
| `serial_drafted` | 0 |
| `max_prompt_tokens` | 73154 |

## Notes

- Boot g, shipping defaults (SPEC=dspark). gates.py spec: serial reference replies ("draft": false, prefix cache off) against the drafted replies, five prompts up to 73,154 tokens solo and four of them at once; 128 greedy tokens each.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/gates.txt`
- `g-all.d/metrics.txt`
- `g-all.d/spec.json`
- `g-all.d/spec.log`
- `g-all.out`
