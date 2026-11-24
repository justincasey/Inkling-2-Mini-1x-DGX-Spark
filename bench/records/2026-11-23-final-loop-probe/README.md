# final-loop-probe: gate `loop-probe`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench loop   (scripts/loop_probe.py run / reasoning / agentic)`
- `python3 scripts/loop_probe.py compare <earlier loop-greedy.json> loop-text.json`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| scripts/loop_probe.py | text, greedy 512 tokens: 34 prompts answered (loop flags are a baseline, counted not judged: 17 flagged) | 34 | 0 |
| scripts/loop_probe.py | reasoning: no loop; finished or cut by the budget | 12 | 0 |
| scripts/loop_probe.py | agentic: no destructive tool call | 6 | 0 |
| scripts/loop_probe.py | text greedy: 34 replies compared with the replies of record 2026-11-23-loop-probe (engine 918e685), byte for byte | 34 | 0 |

## Metrics

| metric | value |
|---|---:|
| `loop_text_prompts` | 34 |
| `loop_text_loops` | 17 |
| `loop_text_mean_distinct4` | 0.5191 |
| `loop_reasoning_prompts` | 12 |
| `loop_reasoning_loops` | 0 |
| `loop_reasoning_unfinished` | 0 |
| `loop_reasoning_tokens_mean` | 610.5 |
| `loop_reasoning_tokens_max` | 3466 |
| `loop_agentic_prompts` | 6 |
| `loop_agentic_destructive_calls` | 0 |
| `greedy_identical_to_earlier_record` | 34 |
| `greedy_prompts_compared` | 34 |
| `loops_greedy` | 17 |
| `prompts` | 34 |
| `mean_distinct4_greedy` | 0.5191 |

## Notes

- Boot g, shipping defaults, drafter on. Text mode is a baseline (the tool records loop flags and judges nothing); the sampled temperature-1.0 run of the earlier record is not part of the final run. Reasoning mode: 12 chat prompts at effort high, 4,096-token budget; agentic mode: 6 prompts with tools offered, a destructive one among them.

## Raw output (`raw/`)

- `cap-proof.txt`
- `compare-earlier-record.txt`
- `g-loop.d/gates.txt`
- `g-loop.d/loop-agentic.json`
- `g-loop.d/loop-probe.log`
- `g-loop.d/loop-reasoning.json`
- `g-loop.d/loop-text.json`
- `g-loop.d/metrics.txt`
- `g-loop.out`
- `loop-report.txt`
