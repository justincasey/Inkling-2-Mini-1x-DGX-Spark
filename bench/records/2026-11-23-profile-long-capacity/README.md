# profile-long-capacity: the capacity line against docs/FIT.md (gate `capacity`)

Result **pass**. Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb` (`inkling2-loop`, clean), built on node-4; `--spec none`, graphs on, GPU clock locked at 2,200 MHz, guard armed. **This build lacks the expert arena**: `bench/records/2026-11-23-memory-plan` and `2026-11-23-expert-arena` (LoopBackend, later heads) supersede the memory figures here.

`long`: `slots=2 max_context=1048576 kv_pages=8194 kv_tokens=2097152 kv_gib=32.26 lanes_at_max_context=2` (`bench check` passed, run `PREFIX_CACHE=off VISION=on`; FIT.md's 36.28 GiB adds the drafter plane). Raw: `../2026-11-23-profile-long/boot/capacity.txt`.
