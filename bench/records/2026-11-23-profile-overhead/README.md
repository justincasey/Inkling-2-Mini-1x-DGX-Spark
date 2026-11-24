# profile-overhead: where the ~9 GiB beyond weights, KV and slot state sit (gate `memory-breakdown`, informational)

Result **pass** (measurements, no threshold). Engine: lab build of `918e6854bb9d4cfe2a9857d92f592722fdf826eb` on node-4, bootstrap shape (4 x 262,144), `--spec none`, `--prefix-cache off`, `--no-vision`, the GPU
clock locked at 2,200 MHz, one boot per row. Each GPU figure is `nvidia-smi --query-compute-apps` MiB of the server process at ready (`regression.txt`; the boots are 20 s each, `stage.sh`-style sampling every 0.2 s in `boot-*/boot-samples.txt`).
**Superseded in part**: LoopBackend's `2026-11-23-memory-plan` (later heads, `mem_stage` events) attributes 6,366 MiB of the weights stage to 2 MiB allocation rounding of the 9,216 expert trellises, which the `expert-arena` record removes; this record measured the same total from outside.

| Boot (SLOTS, KV_TOKENS, PREFILL_CHUNK) | GPU MiB at ready | Reading |
|---|---:|---|
| 4, 1,048,576, 4096 (bootstrap) | 39,680 | baseline |
| 4, 262,144, 4096 | 27,296 | 786,432 fewer KV tokens = **12,384 MiB less = 16,512 B a token exactly** (the plan's `kv_bytes_per_token`) |
| 1, 262,144, 4096 | 26,801 | 3 fewer slots = 495 MiB less, of which 12 MiB are 3 KV pages: **161 MiB a slot** (plan 166.3) |
| 4, 1,048,576, 1024 | 38,417 | 3,072 fewer prefill rows = 1,263 MiB less: **0.411 MiB a row**, 1,684 MiB at 4,096 rows (priced step buffers) |
| 1, 1,048,576, 4096 | 39,185 | consistent with the two rows above |
| `--prefill-chunk 8192` | refused | `--prefill-chunk must be between 1 and 4096` |

Decomposition of the 39,680 MiB at the bootstrap shape (vision off, cache off): weights as loaded 13,875 + KV pages 16,527 (4,100 pages) + slot state 665 + **8,613 MiB (8.41 GiB) unaccounted**. Of that, 1,684 MiB scale with the
prefill chunk (priced `step_items`); the other **~6.9 GiB do not depend on slots, KV or chunk** (measured by difference). The sampler at 0.2 s (`boot-nographs-novision/boot-samples.txt`) shows the device at 1.8 GiB (CUDA context and the first buffers) before the weights ramp,
the ramp ending near 19.9 GiB while the loader counts 13.55 GiB of weights, then a jump of 19.7 GiB in about one second (KV 16.5, state 0.65, step buffers ~2.5). [INFERENCE] the fixed 6.9 GiB are mostly the weight allocation padding of the memory-plan record (6.4 GiB) plus the CUDA context (~0.5-1.8 GiB, not isolated); I did not instrument allocations.
The overhead is larger at the 1M window (9.55-9.74 GiB at 4 x 1,048,576 and 2 x 1,048,576, `profile-max`, `profile-long`): [INFERENCE] unpriced per-context buffers, `memory::UNPRICED` lists five (attention partials and row plans, short-convolution row descriptors, loop-controller message buffers, dense EXL3 input-rotation workspace, drafter scratch).

## Decode graphs and overlap lanes (what the lead asked to price)

Same bootstrap boot twice (vision off, cache off), then `bench decode` (warm-up, 1 stream, 4 and 8 streams): `REXL3_INKLING2_LOOP_GRAPHS=0 REXL3_INKLING2_LOOP_OVERLAP=0` against the default.

| | graphs and overlap off (`graphs-off/`) | on (`graphs-on/`) |
|---|---:|---:|
| GPU MiB at ready | 39,680 | 39,680 |
| GPU MiB after the decode phase | 39,684 | 39,854 |
| MemAvailable after, MiB (noise about 300) | 78,790 | 78,493 |
| decode 1 / 4 / 8 streams, tok/s | 41.5 / 87.5 / 92.0 | 53.4 / 100.4 / 103.1 |

**The graphs and overlap lanes cost 174 MiB of device memory, allocated at first use (ready is identical)**; the bracket runs show +231 to +254 MiB between ready and the sampled peak after four long lanes and the decode burst.
They decode 29 % faster at one stream. Logs: `graph_captured rows=3 / rows=2` events (`graphs-on/boot-events.txt` holds the boot's events).
