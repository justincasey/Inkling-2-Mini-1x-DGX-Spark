# sampled-decode: gate `sampled-decode`: PASS

What decode speed an interactive, SAMPLED, reasoning chat session gets on one DGX Spark, with the drafter on (the shipping default `SPEC=dspark`, boot A) and off (`SPEC=none`, boot B). Node node-2, engine `59d08a9c5e9f9d3ce96ed9d2353c68b70169cc1a` (the pin, built by `./inkling2mini.sh build`, not a lab build), GPU capped at 2,200 MHz (`raw/cap-proof.txt`). Every number below comes from `raw/*.json` through `derive_meta.py` (`meta.json`, `raw/tables-*.md`); none is typed into `meta.json`. Result `pass`: all 160 requests (2 boots x (64 at c = 1 + 16 at c = 4)) and both 34-prompt loop probes were answered; the A/B exactness count is a metric, not a pass condition.

## Verdict

- **Drafted replies equal serial replies, token for token: 32 of 32 pairs at temperature 0 and 32 of 32 at temperature 1.0** (same seed, `ids_sha256` of the generated ids; 16 prompts x 2 efforts per temperature). The raw-text sampled loop probe agrees: 34 of 34 continuations identical (temperature 1.0, seed 1). Each boot's 16 c = 4 replies also equal its own c = 1 replies (16 of 16 for A, 16 of 16 for B). No difference was found: nothing to report as a correctness finding.
- **`SPEC=dspark` was at least as fast as `SPEC=none` in all 20 groups** (4 categories + all, x 2 efforts x 2 temperatures). Serial decode is flat (52.5 to 53.7 tok/s mean in every group, whatever the content or temperature); drafted decode depends on the content: chat 54.4 to 60.7 tok/s (A/B 1.015 to 1.130), reasoning problems 81.4 to 96.1 (1.549 to 1.829), code 86.6 to 98.7 (1.642 to 1.842), structured/JSON 100.6 to 114.5 (1.887 to 2.150). Over all 16 prompts: 91.6 / 84.7 / 89.9 / 81.2 tok/s drafted (low t0 / low t1.0 / high t0 / high t1.0) against 53.5 / 53.3 / 53.0 / 52.9 serial: A/B 1.714 / 1.589 / 1.698 / 1.533.
- **Sampling costs the drafter some speed, not its lead**: temperature 1.0 against 0, drafted all-prompt tokens per round 3.02 -> 2.46 (low) and 2.87 -> 2.49 (high), acceptance 0.557 -> 0.513 and 0.540 -> 0.498, tok/s 91.6 -> 84.7 and 89.9 -> 81.2; serial does not move.
- **Per request (c = 1, 64 pairs) the drafter was slower than serial in 2, both chat, at 0.996 x or better** (min of each category: chat 0.996, reasoning 1.159, code 1.385, structured 1.601). Chat replies are the one kind where drafting is near parity (tokens per round 1.14 to 1.59, acceptance 0.295 to 0.390), but it never cost more than the noise.
- **c = 4** (four different prompts at once, one per category, temperature 1.0, effort high, four waves of four): aggregate 122.6 tok/s drafted against 83.2 serial (A/B 1.473), per-request median 44.0 against 29.9 tok/s.
- **Loops when sampled**: raw text continuation (`scripts/loop_probe.py run`, 34 prompts, 512 tokens, temperature 1.0, seed 1): **4 of 34 flagged in both boots** (3 by a distinct 4-gram ratio below 0.5, 1 by a periodic tail; mean distinct 4-gram ratio 0.7857), against **17 of 34 greedy** (`bench/records/2026-11-23-final-loop-probe`, mean 0.5191). The 64 chat replies (reasoning included) of this run: 0 flagged at temperature 0 (0 of 32) and 0 at temperature 1.0 (0 of 32) in each boot, with the same flag.
- **Budgets bind before loops do**: at effort high, code replies spent 98.3 % (temperature 0) and 99.0 % (1.0) of their tokens reasoning and all 4 of 4 hit `max_tokens` 2048 at both temperatures; over all prompts 5 of 16 (t0) and 6 of 16 (t1.0) replies at effort high and 1 and 2 of 16 at effort low were cut by the limit (the same replies in both boots). A client should give a reasoning session a larger `max_tokens` than 2,048 for code or planning.
- **Is `SPEC=dspark` the right default for interactive use? On this evidence, yes**: with sampling and reasoning on, it won every group, a chat-only mix (4 prompts) was at parity (+1.5 % to +13 % of mean tok/s; two single requests 0.4 % slower). What is NOT shown: the claim of `docs/DECISIONS.md` that a chat-only deployment at four concurrent streams loses against serial (an all-chat c = 4 mix, about 9 % in its older measurement) was not remeasured: the c = 4 waves here mix the four categories. A deployment that serves only open chat to several users at once should measure that mix before keeping the default; at c = 1 nothing here supports `SPEC=none`.

## Conditions

- Boots: `./inkling2mini.sh start` with the shipping defaults (profile bootstrap, `PREFIX_CACHE=on`, `VISION=on`, 4 slots, 262,144-token window, int8 KV) for boot A; the same with `SPEC=none` for boot B (`--spec none` in `serve.boot_b` of `meta.json`, read back from the running process). Boot A took 17 s, boot B 16 s to ready. Full serve commands, pack revision and engine sha256 in `meta.json` (`raw/boot-a-skeleton.json`, `raw/boot-b-skeleton.json` are the controller's `record` output of each running boot).
- Requests: `bench/tools/sampled.py` (the prompt set is its `PROMPTS`: 16 original prompts, 4 open chat, 4 reasoning (arithmetic, logic, planning, debugging), 4 code, 4 structured extraction / JSON). Streamed `/v1/chat/completions`, `reasoning_effort` low (`max_tokens` 1024) and high (2048), `temperature` 0 and 1.0 (the pack's `generation_config.json`), `prefix_cache: false`, `return_token_ids`, `stream_options.include_usage`; `seed = 71000 + (prompt_index x 2 + effort_index) x 2 + temperature_index` (64 distinct seeds, identical in both boots). c = 1: all 64 combinations one at a time after one unrecorded warm-up request; then c = 4: wave w sends prompts w, w+4, w+8, w+12 together at temperature 1.0, effort high.
- Decode tok/s of a request = (completion_tokens - 1) / (time of the last token event - time of the first); the first token's time is the TTFT. Tokens, reasoning tokens and drafter counters come from `usage`, never from chunk counts. Tokens per round = completion tokens / rounds, acceptance = accepted / drafted, both pooled over the group's requests. Serial boot: the server still reports rounds (one token each, tokens per round 1.00) and drafts nothing (no acceptance). Aggregate c = 4 tok/s = all completion tokens over the span from the first start to the last end of a wave (the four waves' tokens over the sum of their spans in the `all` row).
- n = 4 requests per category cell, 16 per `all` cell: small cells, one prompt set. Means and medians are shown side by side; the serial boot is flat (52.5 to 53.7 tok/s in every cell), so the ratios move with the drafted numbers.

## Commands

```
scripts/gpu_cap.sh <node-2>                       # raw/cap-proof.txt
./inkling2mini.sh doctor ; ./inkling2mini.sh build              # raw/build.log (engine at the pin)
./inkling2mini.sh start ; ./inkling2mini.sh record skel-a       # boot A, SPEC=dspark (shipping)
python3 bench/tools/sampled.py run raw/boot-a-dspark.json
python3 bench/tools/sampled.py summary raw/boot-a-dspark.json
python3 scripts/loop_probe.py run raw/boot-a-loop-sampled.json --temperature 1.0 --seed 1
./inkling2mini.sh stop ; SPEC=none ./inkling2mini.sh start ; SPEC=none ./inkling2mini.sh record skel-b2   # boot B
python3 bench/tools/sampled.py run raw/boot-b-serial.json
python3 bench/tools/sampled.py summary raw/boot-b-serial.json
python3 scripts/loop_probe.py run raw/boot-b-loop-sampled.json --temperature 1.0 --seed 1
./inkling2mini.sh stop                                           # GPU idle afterwards
python3 bench/tools/sampled.py compare raw/boot-a-dspark.json raw/boot-b-serial.json
python3 derive_meta.py                                           # meta.json, raw/tables-*.md
```

## Boot A (SPEC=dspark), c = 1 by category x effort x temperature, and the c = 4 waves

| category | effort | temp | n | tok/s mean | tok/s median | tokens/round | acceptance | reasoning share | TTFT mean s | hit max_tokens | failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chat | low | 0.0 | 4 | 60.7 | 59.7 | 1.59 | 0.390 | 0.259 | 0.12 | 0 | 0 |
| chat | low | 1.0 | 4 | 54.4 | 53.9 | 1.14 | 0.295 | 0.222 | 0.12 | 0 | 0 |
| chat | high | 0.0 | 4 | 58.4 | 58.7 | 1.37 | 0.390 | 0.492 | 0.12 | 0 | 0 |
| chat | high | 1.0 | 4 | 56.0 | 56.7 | 1.24 | 0.354 | 0.358 | 0.12 | 0 | 0 |
| reasoning | low | 0.0 | 4 | 94.2 | 92.7 | 3.32 | 0.576 | 0.779 | 0.17 | 1 | 0 |
| reasoning | low | 1.0 | 4 | 81.9 | 81.6 | 2.64 | 0.506 | 0.767 | 0.16 | 1 | 0 |
| reasoning | high | 0.0 | 4 | 96.1 | 94.4 | 3.54 | 0.581 | 0.878 | 0.16 | 1 | 0 |
| reasoning | high | 1.0 | 4 | 81.4 | 78.7 | 2.78 | 0.515 | 0.937 | 0.16 | 2 | 0 |
| code | low | 0.0 | 4 | 98.7 | 100.4 | 3.67 | 0.557 | 0.390 | 0.13 | 0 | 0 |
| code | low | 1.0 | 4 | 89.5 | 90.0 | 3.14 | 0.493 | 0.533 | 0.14 | 1 | 0 |
| code | high | 0.0 | 4 | 90.8 | 93.4 | 3.31 | 0.518 | 0.983 | 0.14 | 4 | 0 |
| code | high | 1.0 | 4 | 86.6 | 86.7 | 3.13 | 0.491 | 0.990 | 0.13 | 4 | 0 |
| structured | low | 0.0 | 4 | 112.9 | 112.2 | 4.59 | 0.655 | 0.704 | 0.18 | 0 | 0 |
| structured | low | 1.0 | 4 | 113.2 | 105.6 | 4.63 | 0.685 | 0.695 | 0.18 | 0 | 0 |
| structured | high | 0.0 | 4 | 114.5 | 114.7 | 4.42 | 0.619 | 0.837 | 0.18 | 0 | 0 |
| structured | high | 1.0 | 4 | 100.6 | 99.7 | 3.82 | 0.564 | 0.802 | 0.18 | 0 | 0 |
| all | low | 0.0 | 16 | 91.6 | 92.7 | 3.02 | 0.557 | 0.539 | 0.15 | 1 | 0 |
| all | low | 1.0 | 16 | 84.7 | 84.1 | 2.46 | 0.513 | 0.588 | 0.15 | 2 | 0 |
| all | high | 0.0 | 16 | 89.9 | 90.8 | 2.87 | 0.540 | 0.862 | 0.15 | 5 | 0 |
| all | high | 1.0 | 16 | 81.2 | 79.5 | 2.49 | 0.498 | 0.854 | 0.15 | 6 | 0 |

| c=4 wave | requests | answered | tokens | span s | aggregate tok/s | per-request median tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 4 | 3616 | 32.2 | 112.2 | 41.0 |
| 1 | 4 | 4 | 5124 | 38.4 | 133.3 | 45.5 |
| 2 | 4 | 4 | 5078 | 40.6 | 124.9 | 57.0 |
| 3 | 4 | 4 | 5602 | 47.1 | 118.8 | 43.3 |
| all | 16 | 16 | 19420 | 158.5 | 122.6 | 44.0 |

## Boot B (SPEC=none)

| category | effort | temp | n | tok/s mean | tok/s median | tokens/round | acceptance | reasoning share | TTFT mean s | hit max_tokens | failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chat | low | 0.0 | 4 | 53.7 | 53.7 | 1.00 | - | 0.259 | 0.12 | 0 | 0 |
| chat | low | 1.0 | 4 | 53.6 | 53.7 | 1.00 | - | 0.222 | 0.12 | 0 | 0 |
| chat | high | 0.0 | 4 | 53.3 | 53.2 | 1.00 | - | 0.492 | 0.13 | 0 | 0 |
| chat | high | 1.0 | 4 | 53.1 | 53.2 | 1.00 | - | 0.358 | 0.12 | 0 | 0 |
| reasoning | low | 0.0 | 4 | 53.0 | 53.0 | 1.00 | - | 0.779 | 0.16 | 1 | 0 |
| reasoning | low | 1.0 | 4 | 52.8 | 52.8 | 1.00 | - | 0.767 | 0.16 | 1 | 0 |
| reasoning | high | 0.0 | 4 | 52.5 | 52.5 | 1.00 | - | 0.878 | 0.16 | 1 | 0 |
| reasoning | high | 1.0 | 4 | 52.5 | 52.3 | 1.00 | - | 0.937 | 0.16 | 2 | 0 |
| code | low | 0.0 | 4 | 53.6 | 53.6 | 1.00 | - | 0.390 | 0.13 | 0 | 0 |
| code | low | 1.0 | 4 | 53.4 | 53.4 | 1.00 | - | 0.533 | 0.13 | 1 | 0 |
| code | high | 0.0 | 4 | 52.8 | 52.8 | 1.00 | - | 0.983 | 0.14 | 4 | 0 |
| code | high | 1.0 | 4 | 52.8 | 52.8 | 1.00 | - | 0.990 | 0.13 | 4 | 0 |
| structured | low | 0.0 | 4 | 53.6 | 53.6 | 1.00 | - | 0.704 | 0.18 | 0 | 0 |
| structured | low | 1.0 | 4 | 53.5 | 53.6 | 1.00 | - | 0.695 | 0.18 | 0 | 0 |
| structured | high | 0.0 | 4 | 53.2 | 53.1 | 1.00 | - | 0.837 | 0.18 | 0 | 0 |
| structured | high | 1.0 | 4 | 53.3 | 53.2 | 1.00 | - | 0.802 | 0.18 | 0 | 0 |
| all | low | 0.0 | 16 | 53.5 | 53.5 | 1.00 | - | 0.539 | 0.15 | 1 | 0 |
| all | low | 1.0 | 16 | 53.3 | 53.4 | 1.00 | - | 0.588 | 0.15 | 2 | 0 |
| all | high | 0.0 | 16 | 53.0 | 53.0 | 1.00 | - | 0.862 | 0.15 | 5 | 0 |
| all | high | 1.0 | 16 | 52.9 | 53.0 | 1.00 | - | 0.854 | 0.15 | 6 | 0 |

| c=4 wave | requests | answered | tokens | span s | aggregate tok/s | per-request median tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 4 | 3616 | 49.7 | 72.8 | 30.7 |
| 1 | 4 | 4 | 5124 | 59.2 | 86.6 | 28.9 |
| 2 | 4 | 4 | 5078 | 60.8 | 83.5 | 30.6 |
| 3 | 4 | 4 | 5602 | 63.7 | 87.9 | 30.0 |
| all | 16 | 16 | 19420 | 233.4 | 83.2 | 29.9 |

## A/B (drafted / serial mean and median decode tok/s) and exactness

| category | effort | temp | A/B tok/s (means) | A/B tok/s (medians) |
|---|---|---:|---:|---:|
| chat | low | 0.0 | 1.130 | 1.112 |
| chat | low | 1.0 | 1.015 | 1.005 |
| chat | high | 0.0 | 1.096 | 1.104 |
| chat | high | 1.0 | 1.054 | 1.066 |
| reasoning | low | 0.0 | 1.777 | 1.747 |
| reasoning | low | 1.0 | 1.551 | 1.545 |
| reasoning | high | 0.0 | 1.829 | 1.801 |
| reasoning | high | 1.0 | 1.549 | 1.504 |
| code | low | 0.0 | 1.842 | 1.874 |
| code | low | 1.0 | 1.676 | 1.684 |
| code | high | 0.0 | 1.720 | 1.769 |
| code | high | 1.0 | 1.642 | 1.644 |
| structured | low | 0.0 | 2.107 | 2.095 |
| structured | low | 1.0 | 2.114 | 1.970 |
| structured | high | 0.0 | 2.150 | 2.159 |
| structured | high | 1.0 | 1.887 | 1.874 |
| all | low | 0.0 | 1.714 | 1.731 |
| all | low | 1.0 | 1.589 | 1.573 |
| all | high | 0.0 | 1.698 | 1.713 |
| all | high | 1.0 | 1.533 | 1.500 |

temperature 0.0: 32 of 32 replies identical id for id
temperature 1.0: 32 of 32 replies identical id for id

## Raw sampled loop probe

`raw/loop-report.txt`: both boots, 34 prompts, temperature 1.0, seed 1: 4 flagged each, 34 of 34 replies identical; mean tok/s of the probe (includes prefill) 73.1 drafted against 53.1 serial.

## Raw output (`raw/`)

- `boot-a-dspark.json`, `boot-b-serial.json`: every request (ids, texts, timings, counters); `*.log`: the live lines; `*-summary.md`, `tables-*.md`: tables
- `boot-a-loop-sampled.json`, `boot-b-loop-sampled.json`, `*.log`, `loop-report.txt`
- `boot-a-skeleton.json`, `boot-b-skeleton.json`, `build.log`, `cap-proof.txt`
