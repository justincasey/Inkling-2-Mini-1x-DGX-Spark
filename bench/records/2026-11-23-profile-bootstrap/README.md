# profile-bootstrap: 4 x 262,144 on one Spark, measured (gate `memory`)

Result **pass** (`meta.json`). Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb` (`inkling2-loop`, clean), built on node-4 by `./inkling2mini.sh build`; `--spec none`, graphs and overlap lanes on (the head's default), GPU clock locked at 2,200 MHz, the memory guard armed on every boot, node-4 otherwise idle. **This build lacks the expert arena** (`bench/records/2026-11-23-expert-arena`) and the corrected plan (`2026-11-23-memory-plan`): those records, made by LoopBackend on later heads, supersede the memory figures below (they return about 6 GiB of allocation padding and price the plan within 0.6 GiB). The figures here are for this build and are kept as the measurement that found the padding.

Commands (per configuration, from the recipe worktree): `PROFILE=bootstrap PREFIX_CACHE=<on|off> VISION=<off|on> ./inkling2mini.sh start`, then `./inkling2mini.sh bench profile`
(`bench_profile` in the controller: memwatch window, the capacity gate, GPU MiB and MemAvailable at ready, a node-side sampler every 5 s, one cold
window-length prompt (needle at depth 0.5, seed 11), 4 concurrent cold prompts (seed 12, depths 0.2-0.8), a decode burst of 1, 4 and 8 streams, then
`memcheck.py` against the 8 GiB engine floor). `bench/tools/profile_table.py` does the arithmetic. Raw files are in the two subdirectories.

| | cache on (auto), vision off | vision on, cache off |
|---|---:|---:|
| capacity line | `kv_pages=4100 kv_tokens=1048576 kv_gib=16.14 lanes_at_max_context=4 prefix_entries=16 prefix_gib=8.59 spare_gib=72.66` | `... prefix_gib=0.00 tower_gib=0.79 spare_gib=71.58` |
| MemAvailable at ready | 69698 MiB (68.06 GiB) | 77331 MiB (75.52 GiB) |
| guard minimum (whole window, incl. the cold prefills and 4 lanes) | 69376 MiB (67.75 GiB) | 77220 MiB (75.41 GiB) |
| GPU MiB of the server process, at ready / sampled peak | 48479 / 48710 | 40497 / 40731 |
| overhead beyond weights, KV pages, slot state, cache, tower (`profile_table.py`) | 8.41 GiB | 8.42 GiB |
| cold prefill of 259,915 tokens, 1 request | 197.7 s | 197.3 s |
| 4 concurrent cold prompts of about 260,000 tokens (all first tokens together) | 789-792 s | 779-790 s |
| needles found (1 solo + 4 lanes) | 5 of 5 | 5 of 5 |
| decode, one pass after a discarded warm-up, 1 / 4 / 8 streams | 53.2 / 100.0 / 105.0 tok/s | 53.2 / 100.2 / 100.2 tok/s |

Verdict: **holds with a wide margin.** The guard minimum is 59-67 GiB above the engine's 8 GiB floor and the doctor's 100 GiB idle requirement is met
(116 GiB before boot). Memory does not move under load: the process holds the same GPU MiB at ready as at the peak of a 4-lane prefill, plus 231-234 MiB
that the decode burst allocates (graphs and overlap lanes, first use). Four concurrent cold prompts take four times one (the lanes are served in
turn): the prefill rate is 1,316 tok/s alone and 329 tok/s a lane at 4 lanes.

Deltas that are measured (config differences at the same boot): the prefix cache costs 8799 MiB (plan 8.59 GiB = 8,796 MiB); vision on costs 817 MiB at ready (tower 0.79 GiB planned +
0.28 GiB reserve, `2026-11-23-vision-smoke`). The graphs and overlap lanes add 174 MiB (graphs on 39,854 MiB vs off 39,684 MiB after the same decode phase, `../2026-11-23-profile-overhead`).

Not proven: the cold 260K prefill is not the bootstrap window's worst case for host memory under images; no 8-image burst was run; one boot per configuration.
