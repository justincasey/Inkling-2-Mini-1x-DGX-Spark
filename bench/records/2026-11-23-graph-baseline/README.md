# graph-baseline: decode speed of the engine as it was (gate `speed-graph-baseline`)

Engine `51f2993c1549f927a632c2c502de3d20dc261ec3` (integration head of `inkling2-loop`, lab build, no CUDA graphs, serial launches), one
Spark (node-2), TP1, `bootstrap` profile (262,144 x 4 slots), `--spec none --prefix-cache off --no-vision`,
`REXL3_ALLOW_UNGATED_COMPONENTS=1`, GPU clock locked at 2,200 MHz (`./inkling2mini.sh gpu-clocks lock`). Pack revision `44c78c9d`.
Everything is greedy; the token count is `usage.completion_tokens`; the first pass of every tool is a discarded warm-up.

| Measurement | Command | Result |
|---|---|---|
| decode, 1 stream, short prompts (4 prompt types x 512 tokens) | `./inkling2mini.sh bench decode` (warm-up discarded) | **37.24 tok/s** (prose 37.09, code 37.55, structured 37.32, json 37.00) |
| the same at c = 4 and c = 8 (aggregate) | same | 80.41 and 83.62 tok/s |
| decode, 1 stream, short prompts, median of 3 | `bench/rexl3/sweep_decode.py --phase both --ctxs 4096 --conc 1,2,4 --modes serial --depth-modes serial --reps 3 --depth-reps 3` | **37.232** (37.232, 37.208, 37.236); c = 2: 56.246; c = 4: 80.351 aggregate |
| decode, 1 stream, a 4,096-token prompt, median of 3 | same (`depth` cells) | **36.158** (36.144, 36.180, 36.158) = 27.7 ms a token |
| prefill, cold | `./inkling2mini.sh bench prefill` | 32K: 27.36 s (1,198 tok/s), decode after it 33.46 tok/s; 128K: 127.33 s (1,029 tok/s), 26.96; 224K: 254.49 s (901 tok/s), 22.18 |

Not measured: the depth cells at c = 2 and c = 4 (`raw/sweep-decode.json`: the sweep's streamed requests ask for `return_token_ids`, which this
engine refuses on a streamed completion, HTTP 400; the short-prompt c = 2 and c = 4 rows above are the concurrent numbers), prefill speed beyond
this one pass, anything at 1M.

## Where the 26.7 ms of a token go (launch profile)

Two measurements on a build with the graphs switched off or on and nothing else changed (`d8a6915`, `REXL3_INKLING2_LOOP_GRAPHS=0` and
default, `REXL3_INKLING2_LOOP_TIMING=1`; the serve command is in `meta.json`; the eager path launches what `51f2993` launches):

| | host: uploads | host: enqueue of the launches | host: wait for the device | per window |
|---|---:|---:|---:|---:|
| eager (`profile/eager-window-timing.txt`) | 12 us | **16.1 ms** | 11.5 ms | 27.5 ms |
| one graph replay (`profile/graph-window-timing.txt`) | 10 us | 5 us | 25.1 ms | 25.1 ms |

- A window is **2,830 launches** (2,835 with the top-k): the capture reports the node count (`event=inkling2_loop_graph_captured nodes=2835`),
  about 59 a visit over 48 visits plus the unembed and its tail. A visit launches: the mix and the three controllers of a core pass (pair,
  three narrow linears, blend: five launches each), six dense EXL3 projections (each a `rot_in`, the GEMV and a `rot_out`), the sink and, on a
  global visit, two dynamic-tap linears, the prologue, the attention (and its merge) and the ring write, the output projection, two short convs,
  the router (logits, select), and the MoE (7 launches a band, 1 to 3 bands, one combine).
- The host enqueues 2,830 launches in 16.1 ms (5.7 us each) and the device finishes 11.5 ms after the host is done: the device, not the host,
  is the long pole. A replayed graph costs the host nothing (5 us) and the window takes 25.1 ms: **the device needs about 25 ms a token**; the
  graph recovers the 2.4 ms the eager path loses at the start of the window.
- `profile/nsys-kernels-graph-only.txt`: `nsys profile -t cuda --cuda-graph-trace=node` of 20 replayed one-row windows at 3.8K of context
  (tools in `tools/`), graphs only, before any other change. Per window: 28.3 ms in the profile; one kernel stands out, the **top-k's two
  single-block kernels, 3.1 ms**; the unembed 1.48 ms; per sliding visit about 410 us. (The durations of kernels launched with the
  programmatic-dependent-launch attribute include the time they wait for their predecessor, so the column sums exceed the span: read the
  span and the single kernels, not the share column.)

## Reproduce

```sh
REXL3_SRC=<engine worktree> ./inkling2mini.sh build 51f2993c1549f927a632c2c502de3d20dc261ec3
./inkling2mini.sh start && ./inkling2mini.sh bench decode && ./inkling2mini.sh bench prefill
INKLING2MINI_URL=http://node-2:8000 python3 bench/rexl3/sweep_decode.py --url http://node-2:8000 --model inkling-2-mini-1x-dgx-spark \
  --window 262144 --max-total 1048576 --phase both --ctxs 4096 --conc 1,2,4 --modes serial --depth-modes serial --reps 3 --depth-reps 3 --out sweep.json
./inkling2mini.sh stop
```
