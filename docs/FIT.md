# Fit — vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw on one Spark (TP1)

Arithmetic from `intake/census.json`, `intake/manifest.tsv` and the pack config; `python3 scripts/roofline.py`
regenerates it. Written before any boot. The first boot's capacity line and `MemAvailable` supersede it.

## Pack

15,070,275,721 B (14.035 GiB) in 13 files (`intake/manifest.tsv`); the weights are 15,032,344,840 B = 14.000 GiB (two
trunk shards 14.795 GB of tensor data, `mtp.safetensors` 0.237 GB). One node holds all of it: nothing is sharded.

| Component | GiB |
|---|---:|
| routed experts (EXL3 4/5-bit) | 11.843 |
| attention (EXL3 6-bit) | 0.457 |
| embedding (BF16) | 0.863 |
| unembed (EXL3 6-bit) | 0.324 |
| vision tower | 0.226 |
| loop controllers, routers, glue | 0.067 |
| drafter (`mtp.safetensors`) | 0.220 |
| **total** | **14.000** |

## KV per token

Only the 8 global-attention passes (layers 5, 11 x3, 17 x3, 23) and the drafter's one global layer grow with context;
the 40 sliding passes keep a fixed 1,031-position ring a slot (0.157 GiB bf16).

| KV format | per token | + drafter plane | 1,048,576-token lane |
|---|---:|---:|---:|
| bf16 | 32,768 B | 36,864 B | 36.00 GiB |
| int8 + one exponent byte per (token, KV head) | 16,512 B | 18,576 B | 18.14 GiB |
| q4 (the card) | 8,704 B | 9,792 B | 9.56 GiB |

The card's q4 figures are reproduced exactly. The engine family this recipe starts from runs int8 only.

## KV budget (one node)

| Item | GiB |
|---|---:|
| `MemTotal` | 121.69 |
| OS, CUDA context, unaccounted (Inkling-2-Small FIT) | ~11 |
| `MemAvailable` floor kept under load | ~10 |
| engine budget | ~100.7 |
| weights | 14.00 |
| activations, graphs, scratch (CYBERFROST reserve) | ~5 |
| slot state, 4 slots | 0.63 |
| prefix cache | 8 |
| **KV pool** | **~73.1** |

## Profiles (`PROFILE_MEASURED=1` where a passing record exists)

Measured on engine `918e685` (lab build, `--spec none`, node-4), records `2026-11-23-profile-{bootstrap,long,max}` (gate `memory`) and
`-capacity` companions; `profile-overhead` has the breakdown. KV is int8 pages only (16,512 B a token): the drafter's plane (FIT's
18.14 GiB a lane) does not exist on this build, so the measured pools are 16.14 / 32.26 / 64.52 GiB, not 18.14 / 36.28 / 72.56.

| Profile | Window x lanes | kv_tokens, lanes_at_max_context | MemAvailable at ready | Guard minimum | GPU MiB (ready / peak) | Verdict |
|---|---|---|---:|---:|---:|---|
| bootstrap, cache on (8.59 GiB), vision off | 262,144 x 4 | 1,048,576, 4 | 68.1 GiB | 67.75 GiB | 48,479 / 48,710 | holds, 4 lanes of 260K cold prompts |
| bootstrap, vision on, cache off | 262,144 x 4 | 1,048,576, 4 | 75.5 GiB | 75.41 GiB | 40,497 / 40,731 | holds |
| long, vision on, cache off (boot + one 277K prefill) | 1,048,576 x 2 | 2,097,152, 2 | 58.8 GiB | 58.69 GiB | 57,849 / 57,869 | holds (boot-only) |
| max, cache 8,192 MiB, vision off | 1,048,576 x 4 | 4,194,304, 4 | 18.3 GiB | 17.92 GiB | 98,771 / 98,830 | holds: four 1M prompts resident, 9.9 GiB over the 8 GiB floor |
| max, cache `auto` (22.28 GiB) | 1,048,576 x 4 | 4,194,304, 4 | 3.9 GiB | 3.41 GiB | 113,201 / 113,356 | **fails the engine floor** (the plan's 5 GiB reserve is ~4.4 GiB short) |

Not measured: `max` with vision on, `long` with the cache or two 1M lanes, any profile with the drafter. The memory-plan and expert-arena records
(`2026-11-23-memory-plan`, `2026-11-23-expert-arena`, later heads) supersede the figures above for those heads: the arena returns about 6 GiB.

### Overhead: 5 GiB assumed, 8.4 to 9.7 GiB measured (`2026-11-23-profile-overhead`)

The process holds, beyond weights (13,875 MiB), KV pages, slot state and the priced cache and tower, **8.41 GiB at the bootstrap window and 9.55 to 9.74 GiB at
the 1M window**. Measured by difference: KV 16,512 B a token exactly, 161 MiB a slot, 0.411 MiB per prefill row (1,684 MiB at 4,096), ~6.9 GiB fixed;
decode graphs plus overlap lanes +174 MiB at first use (decode at one stream 41.5 -> 53.4 tok/s). The prefix cache costs its planned size (+8,799 MiB
for 8.59 GiB); vision on costs +817 MiB at ready. Cold prefill: 197 s for 260K (1,316 tok/s), 1,932 s for 1.04M (540 tok/s); four lanes are
served in turn (4 x the solo time for all four first tokens).

Ship `bootstrap`. Re-bracket when the drafter lands: `PROFILE=<p> ./inkling2mini.sh start && ./inkling2mini.sh bench profile`, then
`python3 bench/tools/profile_table.py bench/runs/<run>`.

## Measured

The first boot is in `bench/records/2026-11-23-first-boot/`. The prefix cache, measured on node-3 at the `bootstrap` profile
(`bench/records/2026-11-23-prefix-cache/`, `bench/records/2026-11-23-prefix-cache-speed/`; design and arithmetic in the first README):

| Item | Plan (code arithmetic) | Measured |
|---|---:|---:|
| snapshot slot (40 rings of 1,031 rows 168,919,040 B + conv state of 48 visits 1,916,928 B) | 170,835,968 B | 170,835,968 B (checked against the rank's layout at boot) |
| KV page held by reference (8 visits x K+V x 264,192 B) | 4,227,072 B | 4,227,072 B |
| cache at 4 x 262,144 (window rule: 16 slots + 1,536 pages, reach 393,216 tokens) | 9,226,158,080 B = 8.59 GiB | capacity line `prefix_gib=8.59`; GPU process +8,799 MiB (48,420 vs 39,621 MiB) |
| `MemAvailable` guard minimum, cache on vs off | | 69,417 vs 78,380 MiB (-8,963 MiB) |
| snapshot / restore copy | | 2.14 / 2.2 ms at any length (171 MB each way) |

Where it differs from this file's budget table above: the table's "prefix cache 8 GiB" was a lump; the engine sizes it by the window rule
(8.59 GiB here, the pages of one full window first, then `max(8, 4 x slots)` snapshot slots) out of `spare_gib` 71.66, so the KV pool is the
uncached plan's (4,100 pages) and 63 GiB stay unspent at `bootstrap`. "Slot state, 4 slots 0.63 GiB" is the rings alone; a slot's live state is
166.3 MiB (rings 161.1, conv 1.8, window records 3.4), and a snapshot is 162.92 MiB of it (no window records), fixed whatever the cut: at
`len` >= 1,024 the live part is 161.82 MiB (7 slack rows unread), below 1,024 it is `len x 160 KiB` of rings plus 1.8 MiB, and a compact
layout (not built) would save up to 87 % at a 128-token cut. A cold prompt pays 0.08 to 0.10 s of TTFT for the cut's extra 64-row pass, a
hit at 4K, 12K and 32K gives 0.15 to 0.18 s against 3.3, 9.9 and 27.4 s cold.

The block drafter (`SPEC=dspark`, node-2, `bootstrap` profile 4 x 262,144, `bench/records/2026-11-23-spec-exact/`; the plan prices every figure to the byte, a host test pins it):

| Item | Plan | Measured |
|---|---:|---:|
| drafter weights (the pack's 236,558,740 B less 29 group markers plus the f32 mixing weights plus the fold `[201,088][257]` bf16 103,359,232 B) | 339,917,888 B (0.317 GiB, not the 0.220 GiB of the table above: the fold is new) | 339,917,888 B (boot line `drafter_loaded`) |
| plane of the drafter's global layer (int8, the target's page ids) | 528,384 B a page, 2,064 B a token | 4,100 pages 2,166,374,400 B |
| a slot's rings (two local layers, 1,032 positions) and `f_rnn` | 8,458,752 B | rides in every snapshot: a snapshot slot is 179,294,720 B with the drafter |
| taps buffer (4 x 2,304 bf16 a row, 4,096 rows) | 75,497,472 B | |
| scratch for 4 proposals at 262,144 context (commit rows, block activations, attention partials, base logits, EXL3 workspace 5,308,416 B) | 60,029,312 B | |
| state and scratch at 4 slots and 4,100 pages | 2,335,736,192 B (2.175 GiB) | 2,335,736,192 B |
| capacity line with the drafter | `kv_bytes_per_token=18576`, `weights_gib=13.87`, `spare_gib=69.29`, KV 18.16 GiB | boot 16 s; `MemAvailable` guard minimum 73,628 MiB |

With the vision tower loaded as well (`bench/records/2026-11-23-prefix-cache-images/`, `VISION=on PREFIX_CACHE=on`, decode graphs on): the capacity
line is the text-only one plus `tower_gib=0.79` (`spare_gib` 71.58 after the tower's weights and its feature reserve), the cache is the same 16 slots
and 1,536 pages (8.59 GiB) and the KV pool is unchanged, so the plan prices the two independently: tower first, then the cache from what is
spare. Measured: the server process holds 49,493 MiB on the GPU against 40,501 MiB for the tower alone without a cache or graphs (the cache's
8,799 MiB and about 190 MiB the graphs and side lanes hold), and the guard's `MemAvailable` minimum is 68,439 MiB (66.8 GiB) against 77,682 MiB.
The tower alone adds 880 MiB to the process (40,501 against 39,621 MiB without it) while the plan prices 0.79 GiB of tower plus a 0.28 GiB
feature reserve, 1.07 GiB: consistent with the reserve being held back from the pool's room and not allocated up front [INFERENCE, not read from the allocator].

With the drafter, the cache and the vision tower all on (`bench/records/2026-11-23-spec-images/`, merged engine `e9eeea9`, node-2, `bootstrap` profile, `SPEC=dspark PREFIX_CACHE=on VISION=on`): the plan prices the drafter inside the staged plan and the capacity line says so (`weights_gib=13.87` with the drafter's 0.317, `kv_bytes_per_token=18576`, `prefix_gib=9.47` = the 8.59 of the text-only cache plus 16 slots of the drafter's 8,458,752 B and 1,536 pages of its plane, `step_buffers_gib=2.70` with the taps and the scratch, `fixed_overhead_gib=9.42`):

| configuration | `predicted_ready_available_gib` | `MemAvailable` at engine ready (`mem_stage engine_ready`) | difference | guard minimum after the gate runs |
|---|---:|---:|---:|---:|
| serial, vision on, cache off | 75.57 | 77,930 MiB = 76.10 GiB | +0.53 | 77,576 MiB |
| drafter, vision on, cache off | 73.08 | 75,214 MiB = 73.45 GiB | +0.37 | 74,535 MiB |
| drafter, vision on, cache on (everything) | 62.61 | 65,095 MiB = 63.57 GiB | +0.96 | 63,000 MiB = 61.52 GiB |

The drafter's stage (`mem_stage drafter_loaded`) took 3,142.6 MiB without the cache (priced 2.49 GiB: weights 339,917,888 B, plane for 4,100 pages, 4 slots, taps, scratch) and 3,817.5 MiB with it (5,636 pages, priced 3.25 GiB); the boot-to-boot `MemAvailable` difference of the drafter is 2.65 GiB against 2.49 GiB priced. The prediction stays within 1 GiB with everything on; the guard minimum after the gates is 1.09 GiB under it (graph captures and the pages the gates touched, the plan's 4.72 GiB variable part).
