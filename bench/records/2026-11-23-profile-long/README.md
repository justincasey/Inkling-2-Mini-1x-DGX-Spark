# profile-long: 2 x 1,048,576 on one Spark, boot-only (gate `memory`)

Result **pass**. Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb` (`inkling2-loop`, clean), built on node-4; `--spec none`, graphs on, GPU clock locked at 2,200 MHz, guard armed. **This build lacks the expert arena**: `bench/records/2026-11-23-memory-plan` and `2026-11-23-expert-arena` (LoopBackend, later heads) supersede the memory figures here.

Configuration: `PROFILE=long PREFIX_CACHE=off VISION=on` (the shipping image configuration; no cache, so the memory is the plan's without the cache's share). By the lead's instruction this record is a boot,
the capacity gate and ONE cold prefill of 276707 tokens (needle at 0.5, `--tokens 262144`; the tool's sizing lands above the target here), not a 1M run: `profile-max` and `needle-window` cover the window and the lanes.
Commands: `PROFILE=long PREFIX_CACHE=off VISION=on ./inkling2mini.sh start`, `memwatch begin`, `bench check`, `needle.py --tokens 262144 --max-model-len 1048576 --max-tokens 1024 --depth 0.5 --seed 11`, `memwatch report`, `memcheck.py`.

| Fact | Measured |
|---|---|
| capacity line | `slots=2 max_context=1048576 kv_pages=8194 kv_tokens=2097152 kv_gib=32.26 lanes_at_max_context=2 weights_gib=13.55 free_after_gib=101.66 floor_gib=8.00 reserve_gib=5.00 state_mib_per_slot=166.3 kv_bytes_per_token=16512 kv_dtype=int8 drafter=none prefix_cache=off prefix_entries=0 prefix_pages=0 prefix_gib=0.00 prefix_reach_tokens=0 spare_gib=55.93 tower_gib=0.79 vision_reserve_gib=0.14` |
| MemAvailable at ready | 60226 MiB (58.8 GiB) |
| guard minimum (after the prefill) | 60097 MiB (58.69 GiB), 50 GiB over the 8 GiB floor |
| GPU MiB of the server process, ready / after the prefill | 57849 / 57869 |
| overhead beyond weights, KV pages, slot state and tower (`profile_table.py` arithmetic) | 9.57 GiB |
| cold prefill of 276707 tokens | 216.283 s (1279 tok/s), needle found: `AMBER-9411-OTTER` |

Verdict: **holds** (boot, one 277K-token prefill; the window itself is `profile-max`). Not measured at `long`: a 1M prefill, two concurrent 1M requests, the prefix cache, the decode burst (graphs and overlap lanes allocate on first use: about +0.2 GiB at the other profiles).
