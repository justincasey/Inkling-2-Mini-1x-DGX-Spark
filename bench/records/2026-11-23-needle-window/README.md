# needle-window: needles at the 1,048,576-token window (gate `needle`)

Result **pass**. Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb`, node-4, profile `max` (4 x 1,048,576), `PREFIX_CACHE=8192` (the cache never answered:
every prompt is cold, `cached_tokens` 0), `VISION=off`, `--spec none`, graphs on, `EXTRA_FLAGS=--request-timeout 14400 --prefill-budget 14400 --decision-timeout 14400 --queue-timeout 14400` (the build's 1,800 s default
deadline kills a 1M prefill: see `../2026-11-23-profile-max`). `NEEDLE_MAX_TOKENS=1024` (the model reasons first), greedy, system line `Thinking effort level: 0`.

| Run | Prompt tokens | Prefill (first token) | Found | Reply |
|---|---:|---:|---|---|
| `./inkling2mini.sh bench needle`, depth 0.1 (`needle/needle-0.1.json`) | 1042827 | 1927.087 s | yes | `OTTER-2624-OTTER` (249 tokens) |
| same, depth 0.5 (`needle/needle-0.5.json`) | 1042827 | 1925.907 s | yes | `OTTER-2624-OTTER` (99 tokens) |
| solo profile prompt, depth 0.5, seed 11 (`solo/window-solo.json`) | 1042547 | 1931.979 s | yes | `AMBER-9411-OTTER` |
| 4 concurrent lanes, lane 0, depth 0.2 (`lanes/window-lanes.json`) | 1042932 | 7707.533 s | yes | `AMBER-4103-QUARTZ` |
| 4 concurrent lanes, lane 1, depth 0.4 (`lanes/window-lanes.json`) | 1042862 | 7697.403 s | yes | `VIOLET-3294-OTTER` |
| 4 concurrent lanes, lane 2, depth 0.6 (`lanes/window-lanes.json`) | 1042850 | 7707.828 s | yes | `MAGENTA-4948-NEBULA` |
| 4 concurrent lanes, lane 3, depth 0.8 (`lanes/window-lanes.json`) | 1042854 | 7708.167 s | yes | `QUARTZ-5092-SAFFRON` |

All 7 needles (a pass-code sentence at depth 0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.8 of a 1.04M-token prompt) were answered with the exact code in `content`, finish `stop`. Four 1M-token requests resident at once (the whole 4,194,304-token pool) were
admitted and answered; their prefills are interleaved, so all four first tokens come at about 4 x the solo time (7697.403-7708.167 s, wall 7755.096 s). Cold 1M prefill is 541 tok/s (the 260K prefill of the same build runs 1,316 tok/s: attention cost grows with the window).

Not proven: **each lane's reply equal to its solo run, byte for byte, was not run at 1M** (a solo replay of one lane costs 32 minutes; the `--lane-index` option of `needle.py` and the `reply_sha256` field exist for it: `needle.py --lanes 1 --lane-index K --depth D --seed 12`); no depth between 0.8 and 1.0 and none below 0.1; one prompt family (filler sentences plus one code), so this is recall of a single fact, not long-context reasoning (`longctx` and `sixcat` were not run). Memory during these runs: `../2026-11-23-profile-max`.
