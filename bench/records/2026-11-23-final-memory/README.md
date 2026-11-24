# final-memory: gate `memory`: PASS

Final run, node node-1, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, lab_build false), recipe `9adf2f0f7c0d40c7b0feb766cb12b3b62c3f4f7e`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); no number in this record was typed.

## Commands

- `./inkling2mini.sh memwatch begin; ./inkling2mini.sh bench all   (gate memory: the guard's minimum against the engine floor, memcheck.py)`
- `./inkling2mini.sh bench profile   (capacity line; GPU MiB and MemAvailable at ready; one cold 262,144-token prompt; 4 concurrent ones; a decode burst; node-side sampler)`
- `./inkling2mini.sh memwatch report   (after every gate of the boot: the guard's minimum since it armed, loading included)`

## Tests

| bin | filter | passed | failed |
|---|---|---:|---:|
| inkling2mini.sh | memwatch report after the whole boot: guard still running, minimum MemAvailable 69066 MiB >= the 8192 MiB floor | 1 | 0 |
| bench/tools/memcheck.py | bench all: guard minimum over the floor, guard running | 1 | 0 |
| bench/tools/memcheck.py | bench profile: guard minimum over the floor after one and four concurrent cold 262,144-token prompts and a decode burst | 1 | 0 |
| bench/tools/needle.py | profile: one cold prompt of the window (262,144 tokens), the code found | 1 | 0 |
| bench/tools/needle.py | profile: 4 concurrent cold prompts of the window, every code found | 4 | 0 |

## Metrics

| metric | value |
|---|---:|
| `guard_min_mib` | 69066 |
| `mem_available_min_mib` | 69066 |
| `floor_mib` | 8192 |
| `headroom_mib` | 60874 |
| `guard_loading_min_mib` | 71578 |
| `guard_ready_min_mib` | 69066 |
| `window_solo_prefill_s` | 191.874 |
| `window_solo_prompt_tokens` | 259915 |
| `window_lanes_found` | 4 |
| `window_lanes_prefill_s_max` | 768.263 |
| `gpu_ready_mib` | 47933 |
| `avail_ready_mib` | 69301 |
| `sampled_gpu_peak_mib` | 47933 |
| `sampled_avail_min_mib` | 69075 |
| `kv_tokens` | 1048576 |
| `lanes_at_max_context` | 4 |
| `kv_gib` | 18.16 |
| `free_after_gib` | 101.27 |
| `prefix_gib` | 9.47 |
| `mem_min_gib` | 67.45 |
| `bench_all_mem_min_gib` | 67.6 |
| `gpu_growth_mib` | 0 |

## Notes

- Boot g, shipping defaults (drafter, prefix cache 16 entries, vision tower), GPU capped at 2,200 MHz, the memory guard armed on the server pid for the whole boot. guard_min_mib is the guard's lowest MemAvailable sample since it armed (loading included) over everything the boot ran: the gates of bench all, the sweeps, the quality, tool and loop probes and the profile bracket; the engine's floor is 8 GiB (8,192 MiB). sampled_* are the node-side 5 s sampler over the profile only.

## Raw output (`raw/`)

- `cap-proof.txt`
- `g-all.d/gates.txt`
- `g-all.d/memory.log`
- `g-all.d/memwatch-begin.txt`
- `g-all.d/memwatch.txt`
- `g-all.d/metrics.txt`
- `g-all.out`
- `g-clock-end.out`
- `g-clock-start.out`
- `g-memwatch-ready.out`
- `g-memwatch.out`
- `g-profile.d/bench-decode.err`
- `g-profile.d/bench-decode.json`
- `g-profile.d/bench-decode.out`
- `g-profile.d/capacity.txt`
- `g-profile.d/gates.txt`
- `g-profile.d/memwatch-begin.txt`
- `g-profile.d/memwatch.txt`
- `g-profile.d/metrics.txt`
- `g-profile.d/profile.log`
- `g-profile.d/ready.txt`
- `g-profile.d/samples.txt`
- `g-profile.d/window-lanes.err`
- `g-profile.d/window-lanes.json`
- `g-profile.d/window-lanes.out`
- `g-profile.d/window-solo.err`
- `g-profile.d/window-solo.json`
- `g-profile.d/window-solo.out`
- `g-profile.out`
