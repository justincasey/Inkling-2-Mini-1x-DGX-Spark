# profile-max: 4 x 1,048,576 on one Spark (gate `memory`)

Result **pass for `PREFIX_CACHE=8192` (MiB), fail for the cache's `auto` size** (`meta.json` is the passing configuration; the failing boot is kept under `cache-auto-fails/`). Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb` (`inkling2-loop`, clean), built on node-4 by `./inkling2mini.sh build`; `--spec none`, graphs and overlap lanes on, vision off, GPU clock locked at 2,200 MHz, the memory guard armed. **This build lacks the expert arena** (`bench/records/2026-11-23-expert-arena`), the corrected plan (`2026-11-23-memory-plan`) and the Mini's long default timeouts: those records and the later head supersede the memory figures here (the arena returns about 6 GiB of allocation padding; the plan is priced within 0.6 GiB).

Configuration that passes: `PROFILE=max PREFIX_CACHE=8192 VISION=off EXTRA_FLAGS=--request-timeout 14400 --prefill-budget 14400 --decision-timeout 14400 --queue-timeout 14400`. The extra flags are needed on this build: its request deadline stays at 1,800 s for the Mini (the server raises it to
14,400 s only for the Small's family id, `crates/server/src/main.rs` `apply_family_defaults`), and a cold 1,042,547-token prefill takes 1931.979 s: the first attempt (`cache-auto-fails/window-solo.err`)
died with `request_timeout` at 1,800 s.

| | `cache-8192/` (solo, then 4 lanes) | `cache-auto-fails/` |
|---|---:|---:|
| capacity line | `kv_pages=16388 kv_tokens=4194304 kv_gib=64.52 lanes_at_max_context=4 prefix_entries=2 prefix_pages=1951 prefix_gib=8.00 spare_gib=16.28` | `... prefix_entries=16 prefix_pages=5013 prefix_gib=22.28 spare_gib=24.28` |
| MemAvailable at ready | 18744 MiB (18.3 GiB) | 4007 MiB (3.91 GiB) |
| guard minimum | 18352 MiB (17.92 GiB; solo boot 18600 MiB) | 3492 MiB (3.41 GiB): **under the 8 GiB engine floor** |
| GPU MiB of the process, ready / sampled peak | 98771 / 98830 (solo boot 98577 / 98751) | 113201 / 113356 |
| overhead beyond weights, KV pages, slot state, cache | 9.74 GiB | 9.55 GiB |
| cold prefill, 1 x 1,042,547 tokens | 1931.979 s (540 tok/s), needle at 0.5 found | timed out at 1,800 s |
| 4 concurrent cold prompts of about 1,042,900 tokens | first tokens at 7697.403-7708.167 s (4 x the solo time: served in turn), all four needles found (depths 0.2-0.8), wall 7755.096 s | timed out |
| decode 1 / 4 / 8 streams after the lanes | 53.3 / 100.6 / 100.6 tok/s | 53.4 / 100.3 / 105.2 tok/s |

Verdict: **`max` holds with `PREFIX_CACHE=8192`: guard minimum 17.92 GiB, 9.9 GiB above the engine floor, with four 1M-token requests resident at once. The cache's `auto` size does not**: the plan gave the cache 22.28 GiB of
the 24.28 GiB spare, but the process holds more than the plan's 5 GiB reserve covers (overhead 9.55 GiB at the 1M window against 8.4 GiB at the 262,144 window), so MemAvailable sat at 3.4-3.9 GiB for the whole run: under the engine's own 8 GiB floor, 2.4 GiB over the guard's kill floor (1 GiB).
Memory does not move during the prefills: GPU MiB is flat to within 60 MiB from ready to the end of the 4-lane run (the decode burst adds the graph and overlap buffers).
Overhead grows with the window (8.4 GiB at 262,144, 9.55-9.74 GiB at 1,048,576): [INFERENCE] unpriced per-context buffers. The `memory-plan` record explains the bulk of the overhead (6.4 GiB of 2 MiB rounding of 9,216 expert allocations, removed by the arena) and supersedes these figures.

Not measured: `max` with `VISION=on`, with the cache off, or with the drafter (none exists on this build); one boot per configuration.
