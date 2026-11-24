# final-api-test: gate `api-test`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench probe   (bench/probe.py: the OpenAI surface, the drafter on, one image)`
- `./inkling2mini.sh bench api   (tests/api_conformance.py)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| tests/api_conformance.py | health | 1 | 0 |
| tests/api_conformance.py | models | 1 | 0 |
| tests/api_conformance.py | tokenize | 1 | 0 |
| tests/api_conformance.py | chat | 1 | 0 |
| tests/api_conformance.py | render | 1 | 0 |
| tests/api_conformance.py | max_tokens | 1 | 0 |
| tests/api_conformance.py | stop | 1 | 0 |
| tests/api_conformance.py | streaming | 1 | 0 |
| tests/api_conformance.py | reasoning | 1 | 0 |
| tests/api_conformance.py | tools | 1 | 0 |
| tests/api_conformance.py | completions | 1 | 0 |
| tests/api_conformance.py | errors | 1 | 0 |
| tests/api_conformance.py | metrics | 1 | 0 |
| tests/api_conformance.py | spec | 1 | 0 |
| tests/api_conformance.py | prefix cache | 1 | 0 |
| tests/api_conformance.py | concurrency | 1 | 0 |
| tests/api_conformance.py | determinism | 1 | 0 |
| bench/probe.py | health | 1 | 0 |
| bench/probe.py | models | 1 | 0 |
| bench/probe.py | window | 1 | 0 |
| bench/probe.py | tokenize | 1 | 0 |
| bench/probe.py | chat | 1 | 0 |
| bench/probe.py | usage | 1 | 0 |
| bench/probe.py | reasoning | 1 | 0 |
| bench/probe.py | levels | 1 | 0 |
| bench/probe.py | effort system level 0 | 1 | 0 |
| bench/probe.py | effort effort none | 1 | 0 |
| bench/probe.py | effort effort high | 1 | 0 |
| bench/probe.py | tools | 1 | 0 |
| bench/probe.py | metrics | 1 | 0 |
| bench/probe.py | spec | 1 | 0 |
| bench/probe.py | spec serial | 1 | 0 |
| bench/probe.py | prefix-cache | 1 | 0 |
| bench/probe.py | vision | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `api_checks_passed` | 17 |
| `api_checks_failed` | 0 |
| `probe_checks_passed` | 17 |
| `probe_cached_tokens_repeat` | 2352 |
| `probe_max_model_len` | 262144 |
| `probe_reasoning_tokens_high` | 188 |
| `probe_reasoning_tokens_none` | 39 |
| `probe_spec_acceptance` | 0.4918 |
| `probe_spec_drafted` | 368 |
| `boot_s` | 16 |

## Notes

- Boot g, shipping defaults: the drafter is on, so the probe's spec and serial checks run in their drafter-on form (docs: probe.py --spec auto reads the state from /metrics).

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-api.d/api-test.log`
- `g-api.d/gates.txt`
- `g-api.d/metrics.txt`
- `g-api.out`
- `g-probe.d/gates.txt`
- `g-probe.d/metrics.txt`
- `g-probe.d/probe.json`
- `g-probe.d/probe.log`
- `g-probe.out`
