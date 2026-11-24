# profile-bootstrap-capacity: the capacity line against docs/FIT.md (gate `capacity`)

Result **pass**. Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb` (`inkling2-loop`, clean), built on node-4 by `./inkling2mini.sh build`; `--spec none`, graphs and overlap lanes on (the head's default), GPU clock locked at 2,200 MHz, the memory guard armed on every boot, node-4 otherwise idle. **This build lacks the expert arena** (`bench/records/2026-11-23-expert-arena`) and the corrected plan (`2026-11-23-memory-plan`): those records, made by LoopBackend on later heads, supersede the memory figures below (they return about 6 GiB of allocation padding and price the plan within 0.6 GiB). The figures here are for this build and are kept as the measurement that found the padding.

bootstrap: `slots=4 max_context=262144 kv_pages=4100 kv_tokens=1048576 kv_gib=16.14 lanes_at_max_context=4` (4 x 262,144 x 16,512 B = 16.12 GiB; FIT.md 18.14 GiB with the drafter plane).
`bench_capacity` (`./inkling2mini.sh bench check`): lanes_at_max_context and kv_tokens at least the profile's (`SLOTS`, `SLOTS x MAX_MODEL_LEN`). The capacity lines are in `../2026-11-23-profile-bootstrap/` (`capacity.txt`).
FIT.md's KV figures carry the drafter's plane (18.14 GiB a 1,048,576-token lane); with `--spec none` the engine plans int8 pages only: 16,512 B a token, 16.12 GiB per
1,048,576 tokens, rounded up by one 256-token page a slot. The drafter plane is not in these figures.
