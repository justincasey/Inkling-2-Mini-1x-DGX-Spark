# spec-final: the shipping default on the merged engine, drafted against serial (gate `spec-final`)

Engine `e9eeea9683ed9c897ece83e5eac103bad6e40ee4` (the merged tree; `85fb069` adds a changelog fragment only), node-2, GPU locked at 2,200 MHz, TP1, bootstrap profile, graphs on, greedy,
`SPEC=dspark PREFIX_CACHE=on VISION=on` (the tower and the cache loaded, the engine's `plan` budget with learnt priors, depth 7, folded Markov head), no ungated override.
`bench/tools/specbench.py`, 512 generated tokens a request, one warm-up pass discarded, median of 3 timed passes, wall of the request with prefill; serial = the same server with
`"draft": false`; every drafted reply equals its serial reply. The server ran the gates of `spec-images` before this run (the learnt priors are its traffic).

| kind | prompt tok | serial tok/s | drafted tok/s | speedup | accepted/offered | tokens/round |
|---|---:|---:|---:|---:|---|---:|
| chat (prose) | 74 | 52.30 | 47.00 | 0.899 | 745/3624 (0.21) | 1.95 |
| code | 225 | 52.38 | 117.56 | 2.244 | 1264/1843 (0.69) | 5.71 |
| json (structured) | 251 | 51.69 | 84.59 | 1.636 | 1186/2370 (0.50) | 4.43 |
| longctx (6K prompt) | 6,047 | 40.72 | 63.87 | 1.568 | 1191/2176 (0.55) | 4.49 |
| tool call turn | 166 | 51.50 | 122.54 | 2.379 | 939/1141 (0.82) | 6.47 |
| reasoning | 150 | 51.70 | 72.74 | 1.407 | 1063/2779 (0.38) | 3.27 |
| c = 4 (chat, code, json, reasoning), aggregate | | 100.52 | 111.37 | 1.108 | | |

Images (`raw/imgspeed.py`: a synthetic image and "describe this picture in detail", `reasoning_effort` none, 256 tokens, prefix cache opted out, median of 3 after a discarded pass; ids equal on every pass):

| image | serial tok/s | drafted tok/s | speedup | accepted/offered | tokens/round |
|---|---:|---:|---:|---|---:|
| red-square | 52.03 | 59.87 | 1.151 | 141/398 (0.35) | 2.43 |
| blue-circle | 51.49 | 57.44 | 1.116 | 96/288 (0.33) | 2.32 |
| green-yellow | 52.09 | 71.31 | 1.369 | 169/409 (0.41) | 2.98 |

The numbers agree with the text-only default measured on the pre-merge engine (`spec-speed`: code 2.239, tool 2.368, JSON 1.648, long 1.558, reasoning 1.404, prose 0.901, c = 4 1.106): the merge,
the tower and the cache change the speed by under 1 %. The median of reasoning, chat and code is 1.407; prose stays the one kind that loses (0.90 x behind other traffic).

## Commands
```sh
SPEC=dspark PREFIX_CACHE=on VISION=on ./inkling2mini.sh start
python3 bench/tools/specbench.py --out A-specbench.json --label LABEL --concurrency 4
python3 raw/imgspeed.py --out A-imgspeed.json   # --url as the other tools
```
