# profile-max-capacity: the capacity line against docs/FIT.md (gate `capacity`)

Result **pass**. Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb` (`inkling2-loop`, clean), built on node-4 by `./inkling2mini.sh build`; `--spec none`, graphs and overlap lanes on, vision off, GPU clock locked at 2,200 MHz, the memory guard armed. **This build lacks the expert arena** (`bench/records/2026-11-23-expert-arena`), the corrected plan (`2026-11-23-memory-plan`) and the Mini's long default timeouts: those records and the later head supersede the memory figures here (the arena returns about 6 GiB of allocation padding; the plan is priced within 0.6 GiB).

`max`: `slots=4 max_context=1048576 kv_pages=16388 kv_tokens=4194304 kv_gib=64.52 lanes_at_max_context=4` (4 x 1,048,576 x 16,512 B = 64.50 GiB plus page rounding). FIT.md's 72.56 GiB adds the drafter plane (not present with `--spec none`).
Raw: `../2026-11-23-profile-max/cache-8192/solo/capacity.txt`. Run with `PREFIX_CACHE=8192`; the auto cache at this profile fails the memory gate (see that record).
