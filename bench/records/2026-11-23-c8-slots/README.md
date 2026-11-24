# c8-slots: eight slots, serial and drafted (gate speed-c8-slots), plan experiment E7

Node-4, GPU SM clock locked at 2,200 MHz (reads 2,197), one server at a time, a fresh boot per configuration, the memory guard on.
Engine `1d53344d874b3a578152e35c500a2718599f5b61` (lab build), recipe `9f3a87a`, `SLOTS=8` (profile bootstrap window 262,144; KV pool 2,097,152
tokens = 8 lanes at the full window), `PREFIX_CACHE=on VISION=on`. Serial: `SPEC=none`. Drafted: `SPEC=dspark`.

## Capacity and memory (the guard stayed on, no kill)

| boot | capacity line | MemAvailable at ready | guard minimum while benching |
|---|---|---|---|
| serial, 8 slots | `slots=8 kv_pages=8200 kv_tokens=2097152 kv_gib=32.28 lanes_at_max_context=8 weights_gib=13.55 prefix_entries=32 prefix_gib=13.15 spare_gib=54.86` | 52.2 GiB (53,406 MiB) | 51.6 GiB (52,816 MiB) |
| drafted, 8 slots | `slots=8 kv_pages=8200 kv_tokens=2097152 kv_gib=36.32 lanes_at_max_context=8 weights_gib=13.87 drafter=dspark prefix_entries=32 prefix_gib=14.41 spare_gib=50.32` | 46.2 GiB (47,293 MiB, loading minimum) | 43.8 GiB (44,837 MiB) |

The engine accepted the drafter, the prefix cache (32 entries, reach 524,288 tokens) and the vision tower with 8 slots: no refusal. Boot 16 s. MemAvailable
stayed above 43 GiB (floor asked: 10 GiB).

## Commands

```
SLOTS=8 SPEC=none   PREFIX_CACHE=on VISION=on ./inkling2mini.sh start     # then the passes, then stop
SLOTS=8 SPEC=dspark PREFIX_CACHE=on VISION=on ./inkling2mini.sh start
for pass in 0 1 2 3: python3 bench/tools/bench.py --url http://node-4:8000 --model inkling-2-mini-1x-dgx-spark \
    --max-model-len 262144 --phases warm,decode,conc,rolling --raw --out PASS.json   # pass 0 discarded
python3 bench/tools/exactness.py record exactness-8slots-drafted.json; exactness.py diff against exactness-B-serial.json of c4-baseline: identical: 4 replies
```

## Numbers (median of passes 1-3, range)

| metric | serial, 8 slots | drafted, 8 slots |
|---|---|---|
| decode_c1_tok_s | 53.24 (53.24-53.25) | 81.46 (80.55-81.51) |
| decode_c4_tok_s (span) | 100.42 (100.40-100.42) | 102.23 (100.73-102.41) |
| decode_c4_rolling_tok_s (4 workers on 8 slots) | 100.58 (100.53-100.59) | 131.66 (131.59-131.70) |
| **decode_c8_tok_s = c=8 SPAN, a real 8-stream batch** (two of each kind at once) | **137.01** (136.80-137.02) | **151.37** (147.52-158.23) |
| c=8 steady, all eight streams active (tokens inside [0, earliest end] / that window, computed from the per-request wall times) | 137.53 (137.37-137.54) | 225.93 (224.22-235.51) |
| discarded pass 0, c8 span | 137.21 | 147.56 |

c=8 steady is computed here, not printed by the tool: `conc` has one start time (all eight requests start together) and a wall time per request (raw JSON
`raw.conc8.replies[*].wall_s`); each reply's 512 tokens are spread evenly over its wall time (the tool's own convention for the steady window), the window
is [0, shortest wall time] (serial 29.6 s; drafted 13.5 s, i.e. while both code streams run), the prefill (about 0.2 s) is inside it. It is a lower-fidelity
number than the rolling steady rate: it covers the early half of the run.

c=8 drafted per-request wall times (pass 1, s): prose 27.7 / 27.8, code 13.6 / 13.2, structured 20.2 / 20.0, json 17.0 / 16.2; tokens per round prose 1.59, code 5.75,
structured 3.16, json 4.30-4.34. The span is set by the two prose streams; from 17 s on only the prose streams (and structured until 20 s) run. Serial: all eight end within 0.3 s (29.6-29.9 s).

## Against the model (`c4_model.py`)

| quantity | measured | model | residual |
|---|---|---|---|
| serial 8-stream batch (8 rows per window) | 137.01 | 132.6 (section 4, 8 streams, 60.3 ms step) | +3.3 % (step 58.4 ms) |
| serial c=4 rolling on 8 slots | 100.58 | 99.8 | +0.8 % |
| drafted c=4 rolling on 8 slots | 131.66 | 129.8 | +1.4 % |

Eight slots cost nothing at c=4: serial 100.58 against 100.50 and drafted 131.66 against 131.50 on four slots (c4-baseline).

## What it says

- Serial c=8 as a real batch is 137 tok/s, 1.31x the 4-slot c=4 serial (the 4-slot c=8 of the bench, two waves, is 104 tok/s). The model's byte-roofline call
  (132.6) holds within 3.3 %.
- Drafted c=8 is 151 tok/s on the span metric and 226 tok/s in the early window while all eight streams run: **the 200 line is crossed only in the
  all-active window, not in any sustained or span number.** What a sustained c=8 drafted rate is stays unmeasured:
  `bench.py` has no option for eight rolling workers (`ROLLING_WORKERS = 4` is a module constant; `run_rolling` takes a `workers` argument but no flag reaches it) and the tool was not edited. A tool change
  (a `--rolling-workers` flag) is the next action; the number is expected between the 151 span and the 226 early window (not measured).
- Drafted == serial at 8 slots: the exactness set on the 8-slot drafted boot is identical to the 4-slot serial boot's (`identical: 4 replies`); the serial 8-slot boot itself was not diffed (it was benched, not recorded).
