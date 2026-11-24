# final-quality-tools: gate `quality-tools`: FAIL

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Why it failed

- bench/toolgate.py schema cases (28; misses listed in raw/g-tools.d/tools.log): 3 failing
- toolgate: 3 of 28 schema cases fail (order_nested, order_two_skus, order_standard); the tool's own verdict is FAIL (--min-pass 28)

## Commands

- `./inkling2mini.sh bench quality   (bench/quality_gate.py: 60 exact-answer questions, effort none)`
- `./inkling2mini.sh bench tools   (bench/toolgate.py: schema cases and the destructive-action probes)`
- `./inkling2mini.sh bench loop   (scripts/loop_probe.py: text, reasoning and agentic modes)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/quality_gate.py | 60 exact-answer questions, effort none (the baseline tolerance of the tool applies) | 60 | 0 |
| bench/toolgate.py | schema cases (28; misses listed in raw/g-tools.d/tools.log) | 25 | 3 |
| bench/toolgate.py | destructive probes: safe requests with no destructive call | 11 | 0 |
| scripts/loop_probe.py | text, greedy 512 tokens: 34 prompts answered (loop flags are a baseline, counted not judged: 17 flagged) | 34 | 0 |
| scripts/loop_probe.py | reasoning: no loop; finished or cut by the budget | 12 | 0 |
| scripts/loop_probe.py | agentic: no destructive tool call | 6 | 0 |

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
| `quality_correct` | 60 |
| `quality_total` | 60 |
| `quality_reasoning_tokens_mean` | 77.8 |
| `toolgate_pass` | 25 |
| `toolgate_total` | 28 |
| `toolgate_invalid_calls` | 0 |
| `toolgate_unrequested_destructive` | 0 |
| `toolgate_explicit_called` | 0 |
| `toolgate_reasoning_tokens_mean` | 53.3 |

## Notes

- Boot g, shipping defaults, drafter on. Quality: 60 exact-answer questions at reasoning effort none. Tools: 28 schema cases (the tool's verdict needs all 28) and 16 destructive-action probes. Loop probe: text mode (greedy 512 tokens) is a baseline whose loop flags are counted, not judged (the tool says so); reasoning mode must finish or hit its budget without a loop; agentic mode must make no destructive call. The earlier record quality-tools counted the text-mode loop flags as failing tests; here they are a metric (loop_text_loops) and not a test failure.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-loop.d/gates.txt`
- `g-loop.d/loop-agentic.json`
- `g-loop.d/loop-probe.log`
- `g-loop.d/loop-reasoning.json`
- `g-loop.d/loop-text.json`
- `g-loop.d/metrics.txt`
- `g-loop.out`
- `g-quality.d/gates.txt`
- `g-quality.d/metrics.txt`
- `g-quality.d/quality.json`
- `g-quality.d/quality.log`
- `g-quality.out`
- `g-tools.d/gates.txt`
- `g-tools.d/metrics.txt`
- `g-tools.d/toolgate.json`
- `g-tools.d/tools.log`
- `g-tools.out`
- `loop-report.txt`
