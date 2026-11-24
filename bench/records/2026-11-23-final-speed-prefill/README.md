# final-speed-prefill: gate `speed-prefill`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh bench prefill-sweep   (cold 8K, 32K, 128K, 224K; a unique prompt every pass; 4 passes, the first discarded; 128 tokens decoded after)`
- `./inkling2mini.sh bench all   (prefill phase: cold 32K, 128K, 224K compared with the pinned replies)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| bench/tools/prefill_sweep.py | cold prefill 8K/32K/128K/224K: 3 kept passes each, every pass answered | 12 | 0 |
| bench/tools/bench.py | bench all prefill phase: cold 32K, 128K, 224K replies equal to the pinned replies (PASS pin) | 1 | 0 |

## Metrics

| metric | value |
|---|---:|
| `prefill_8k_tok_s` | 2581.1 |
| `prefill_32k_tok_s` | 2380.7 |
| `prefill_128k_tok_s` | 1799.3 |
| `prefill_224k_tok_s` | 1445.6 |
| `prefill_8k_ttft_s` | 3.1738 |
| `prefill_32k_ttft_s` | 13.7639 |
| `prefill_128k_ttft_s` | 72.8472 |
| `prefill_224k_ttft_s` | 158.6772 |
| `decode_after_8k_tok_s` | 106.633 |
| `decode_after_32k_tok_s` | 87.578 |
| `decode_after_128k_tok_s` | 100.216 |
| `decode_after_224k_tok_s` | 76.603 |
| `prefill_32k_bench_all_tok_s` | 2377.9 |
| `prefill_128k_bench_all_tok_s` | 1779.5 |
| `prefill_224k_bench_all_tok_s` | 1444 |
| `gpu_clock_mhz` | 2184 |

## Notes

- Cold prefill over a unique prompt per pass (prefix cache off for the request), median of the 3 kept passes; drafter dspark booted (shipping defaults), GPU capped at 2,200 MHz. prefill_*_bench_all_tok_s are the single-pass figures of bench all's prefill phase on the same boot.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/bench-prefill.err`
- `g-all.d/bench-prefill.out`
- `g-all.d/metrics.txt`
- `g-all.out`
- `g-prefill-sweep.d/gates.txt`
- `g-prefill-sweep.d/metrics.txt`
- `g-prefill-sweep.d/prefill-sweep.json`
- `g-prefill-sweep.d/prefill-sweep.log`
- `g-prefill-sweep.out`
