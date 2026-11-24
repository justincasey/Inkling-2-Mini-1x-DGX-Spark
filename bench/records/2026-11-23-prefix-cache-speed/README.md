# prefix-cache-speed: what the cache buys and costs on one Spark (gate `cache-speed`)

Result **pass** (`meta.json`): every request that had to be a hit was one, every cold one was served 0 cached tokens, and decode with the
cache on is within noise of decode with it off. Same engine, node and configuration as `../2026-11-23-prefix-cache/` (engine lab build
`de2d4a30e423`, recipe `5c62b03`, node-3, GPU SM clock locked, 2,197 MHz read; profile `bootstrap`, `SPEC=none`, `VISION=off`); the
cache-off runs are the same binary booted with `PREFIX_CACHE=off`. Greedy, one server at a time, nothing else on the node; the server is
stopped and the node idle at the end. Timings are medians of three after a discarded warm-up; token counts are `usage.completion_tokens`.

## What the plan chose (profile `bootstrap`: 4 slots x 262,144)

`raw/serve-start-cache-on.log` (capacity line): `slots=4 kv_pages=4100 kv_tokens=1048576 kv_gib=16.14 ... prefix_cache=on prefix_entries=16
prefix_pages=1536 prefix_gib=8.59 prefix_reach_tokens=393216 spare_gib=71.66`. The window rule: 16 snapshot slots of 170,835,968 B and 1,536
pages of 4,227,072 B = 9,226,158,080 B, taken from the spare memory after the KV pool (which is the uncached plan's: 4,100 pages, the same
`kv_tokens`); reach 393,216 tokens, 1.5 x the window. `raw/serve-start-cache-off.log`: the same line with `prefix_cache=off`.

## Time to first token (`bench/tools/cache_ttft.py`, `raw/cache-ttft.json`)

One token generated, TTFT from the server's `rexl3_request_ttft_seconds` histogram (the prefill), median of 3 per cell, every cell its own
filler stream (no shared leading block). `cold`: cache off in the request. `first`: another prompt of the same size with the cache on, first
sighting (it prefills in full and leaves its snapshot). `shared`: the first prompt's first n - 64 tokens and 64 new ones, so the shared prefix is
restored from the first one's end cut and 64 tokens are prefilled (cached tokens 4,032 / 12,224 / 32,704 in every run). `warm`: the first
prompt again.

| prompt tokens | cold TTFT | first sighting, cache on | shared prefix, cached | warm repeat | cold prefill | shared / cold |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 3.266 s | 3.364 s (+0.099 s, +3.0 %) | **0.148 s** (4,032 restored) | 0.159 s | 1,254 tokens/s | 22.0 x faster |
| 12,288 | 9.930 s | 10.017 s (+0.087 s, +0.9 %) | **0.156 s** (12,224 restored) | 0.169 s | 1,237 tokens/s | 63.5 x |
| 32,768 | 27.401 s | 27.478 s (+0.077 s, +0.3 %) | **0.177 s** (32,704 restored) | 0.187 s | 1,196 tokens/s | 154.5 x |

What the cache costs on a cold prompt: 0.08 to 0.10 s of TTFT, not the copy (2.1 ms, below): the slice ends at the planned cut, so the last
64 rows run as a pass of their own and a pass reads the weights once whatever its rows. The same pass is what a hit pays: the shared-prefix
TTFT of 0.15 to 0.18 s is that pass over the 64 new tokens plus the restore. (A first measurement of this record's tool shared leading blocks
between sizes and read partial hits as cold ones; the tool now gives every cell its own stream and the run above is the second one.)

## What a snapshot and a restore cost (`prefix_snapshot_costs`, `raw/gpu-prefix_snapshot_costs-and-others.log`)

One snapshot slot is 170,835,968 B (rings 168,919,040 + conv 1,916,928), copied whole in 40 + 144 device copies; a restore also copies the
partial last page (8 visits x K and V, 4.03 MiB) and shares the full pages by reference. Median of 5 runs after a discarded one, the rank alone
(no server), `flush` after each:

| held length | snapshot | restore |
|---:|---:|---:|
| 300 | 2.132 ms | 2.242 ms |
| 1,500 | 2.134 ms | 2.237 ms |
| 4,096 | 2.151 ms | 2.149 ms |
| 12,000 | 2.142 ms | 2.235 ms |

Independent of the length, as the layout says (the pages are not copied): 171 MB read and written in 2.14 ms is 160 GB/s of the box's 273.

## Decode with the cache on and off (`raw/decode-cache-on/`, `raw/decode-cache-off/`; `./inkling2mini.sh bench decode`, 3 runs each)

The decode workload's requests carry `"prefix_cache": false`, so this measures what the cache's presence costs a decode step (nothing is
looked up or stored): the engine reads the cache's counters each step and holds 8.6 GiB the KV pool does not get.

| streams | cache on (3 runs) | median | cache off (3 runs) | median | on / off |
|---|---|---:|---|---:|---:|
| 1 (mean of 4 prompt types, 512 tokens) | 37.089, 37.096, 37.109 | 37.096 tok/s | 37.071, 37.044, 37.021 | 37.044 tok/s | +0.14 % |
| 4 | 80.013, 80.060, 79.970 | 80.013 tok/s | 79.999, 79.856, 79.864 | 79.864 tok/s | +0.19 % |
| 8 (4 lanes: the rest queue) | 79.975, 83.281, 86.801 | 83.281 tok/s | 83.538, 82.745, 83.481 | 83.481 tok/s | -0.24 % |

Within the run-to-run spread (c = 8 varies by 8.5 % between runs with the cache on; c = 1 by 0.05 % with it on and 0.14 % with it off, so the
0.14 % between the medians is as large as the off runs' own spread). Serial decode without CUDA
graphs, as in the first boot (37.4 tok/s there on another node).

## Memory with the cache on

| | cache on | cache off | difference |
|---|---:|---:|---:|
| the server process on the GPU (`nvidia-smi`) | 48,420 MiB | 39,621 MiB | **8,799 MiB** = the plan's 9,226,158,080 B (8,798.6 MiB) to the MiB |
| `MemAvailable` at ready (guard `ready_min`) | 69,417 MiB | 78,380 MiB | 8,963 MiB |
| guard minimum over the window (boot, gate, stress, TTFT, 3 decode runs) | **69,417 MiB** (67.8 GiB) | 78,380 MiB | the guard's floors are 2,048 (loading) and 1,024 MiB (ready) |
| swap | 507 MiB, unchanged | 507 MiB, unchanged | none |

The allocation equals the plan; the box keeps 67.8 GiB free at its lowest with the cache full of snapshots (`raw/memwatch-cache-on.txt`).
Counters after the runs (`raw/metrics-prefix-cache-on.txt`): 121 lookups, 67 hits (55 %), 827,196 prompt tokens not prefilled, 66 inserts, 50
evictions (all for snapshot slots, 0 for pages: 16 slots are the binding limit, the 1,536 pages hold 964), 0 restore failures, 16 entries holding
6.81 GB of the 9.23 GB budget.

## Not measured

Cached TTFT at a 128K or longer prefix; the cache with images, a drafter or graph replay; decode with a restore in flight (its correctness
is in `../2026-11-23-prefix-cache/`, its cost is the 2.2 ms copy plus the 64-row pass shown above).
