# loop-probe: greedy against sampled raw continuations (gate `loop-probe`, informational)

Result **pass** (informational: no threshold). Engine: lab build of the committed head `918e6854bb9d4cfe2a9857d92f592722fdf826eb`, node-4, profile `bootstrap`, `VISION=on PREFIX_CACHE=off --spec none`, graphs on, one boot.

Command: `python3 scripts/loop_probe.py run OUT.json` (greedy, as before) and `python3 scripts/loop_probe.py run OUT.json --temperature 1.0 --seed 1000` (new options; request seed = 1000 + the prompt's index, so a
rerun repeats). 34 raw prompts, 512 new tokens each, one request at a time, a loop is a distinct 4-gram ratio below 0.5 or a periodic tail. Sampling settings: the pack's `generation_config.json` is
`{"do_sample": true, "temperature": 1.0, "eos_token_id": 200006, "pad_token_id": 199999}`: **temperature 1.0 and nothing else** (no `top_p`, no `top_k`; the pack's `README.md` gives no sampling advice). That is what was applied.

| | loops (of 34) | by distinct-4-gram ratio alone | mean distinct 4-gram ratio | mean tok/s |
|---|---:|---:|---:|---:|
| greedy (`loop-greedy.json`) | **17** | 16 | 0.519 | 52.7 |
| sampled, temperature 1.0 (`loop-t1.json`) | **3** (`extra03` 0.403, `extra15` 0.242, `extra29` 0.479) | 3 | 0.769 | 52.4 |

Reading: **greedy reproduces the earlier record exactly** (`2026-11-23-quality-tools/loop/loop-text.json`, engine `3287e80d`): all 34 replies are token-identical (`loop_probe.py compare`: identical replies 34, same 17 loops, mean 0.519), now at 52.7 tok/s instead of 37.0 (decode graphs),
so the engine changes since (cache, graphs, overlap) did not move the greedy output. **At the pack's own sampling setting 3 of 34 loop, not half**: greedy decoding of raw continuation text loops on half of these prompts, sampled decoding on 9 %, and two of the three
sampled loops are prompts that also loop greedily (`extra15`, `extra29`; `extra29` is 0.079 greedy and 0.479 sampled). That is the behaviour of a base-style model (the card warns of repetitive loops), not a signal of an engine fault. It is not a proof of correctness:
the loop rate says nothing about the oracle's fp64 distribution, and one seed was run. The lead's oracle comparison of one looping prompt is **not** triggered by this probe.
Not run: reasoning and agentic modes (unchanged engine behaviour is covered by `2026-11-23-quality-tools`), sampled runs with other seeds, other temperatures.
