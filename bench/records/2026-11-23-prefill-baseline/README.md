# prefill-baseline: cold prefill and the decode after it on the merged engine, and where a 4,096-row chunk goes (gate `speed-prefill-baseline`)

Result **pass** (a measurement; no threshold). Engine `7da8c6134484` (branch `inkling2-loop`, the integration head the lane started from: rank, decode graphs, prefix
cache, vision tower), a lab build, node-3 alone on the GPU, SM clock locked at 2,200 MHz (`gpu-clocks status` read 2,197 MHz), `SPEC=none PREFIX_CACHE=off VISION=off`,
profile `bootstrap` (262,144 x 4 slots, chunk 4,096), int8 KV, decode CUDA graphs on (the default). One boot, one sweep: `./inkling2mini.sh bench prefill-sweep` (new in
this lane: `bench/tools/prefill_sweep.py`): per size four passes, the first discarded, each pass **a different filler stream** (block seed offset 1,000 x pass), 8,192,
32,768, 131,072 and 229,376 prompt tokens, ids, greedy, `"prefix_cache": false`; the rate is `usage.prompt_tokens` over the server's `rexl3_request_ttft_seconds` of a
solo request, the decode rate is `(completion_tokens - 1) / (e2e - ttft)` of the same request with 128 new tokens (`usage.completion_tokens`).
Median of the three kept passes:

| Size | Pass 1 / 2 / 3 (tok/s) | **Median tok/s** | **TTFT (s)** | decode after it (tok/s, 3 passes) | median |
|---|---|---:|---:|---|---:|
| 8K (8,192) | 2440.3 / 2436.7 / 2429.9 | **2,436.7** | 3.362 | 50.541 / 50.293 / 50.620 | **50.541** |
| 32K (32,768) | 2253.5 / 2254.8 / 2256.4 | **2,254.8** | 14.533 | 46.454 / 46.263 / 45.997 | **46.263** |
| 128K (131,072) | 1725.3 / 1725.5 / 1725.9 | **1,725.5** | 75.960 | 34.360 / 34.321 / 34.534 | **34.360** |
| 224K (229,376) | 1394.8 / 1395.3 / 1394.9 | **1,394.9** | 164.435 | 27.671 / 27.480 / 27.603 | **27.603** |

(`raw/sweep-final/`; an earlier sweep of the same boot with the same passes but the decode only on the last pass, `raw/sweep-first-no-decode-median/`, gave 2,438.3 / 2,259.7 /
1,724.8 / 1,394.6 tok/s: the runs agree within 0.3%.) The warm-up passes of the second sweep were within 0.5% of the kept ones; in the first sweep the very first 8K prompt of the boot was 2.9% slower (2,371.9 tok/s, the first launches of every kernel) and every other warm-up was within 0.2%. Lowest
MemAvailable over both sweeps 78,035 MiB (`memwatch report`, guard floor 1,024 MiB, engine floor 8 GiB).

## The one-off 14.5 s at 32K: reproduced and isolated

* The merged engine does 32,768 cold tokens in **14.53 s (2,254.8 tok/s)** every time (all four passes 14.49-14.54 s), not once.
* `9fa93ea` ("Made the Mini's relative-bias tables in a kernel of their own") is the only commit between `3df40c1` and its child that touches a kernel. `3df40c1` itself, built and
  booted here (`raw/isolation-3df40c1-32k-8k/`, `REXL3_ALLOW_UNGATED_COMPONENTS=1` because that commit still needs the gate switch, the same prompts and passes):
  32K in **27.30 s (1,200.4 tok/s)**, 8K in 6.556 s (1,249.5 tok/s), the numbers of the first boot. The head, with the bias kernel: 14.53 s and 3.362 s. So the 2x is the bias kernel
  (about 1.6 s a chunk: the prologue's two CTAs a row built the 24 x 1,024-entry bf16 tables, 33 ms a visit; the new grid of rows x 4 x 4 takes 1.1 ms a visit). Between `9fa93ea`
  and `7da8c61` only 17 lines of `inkling2_attn.rs` and `model.rs` changed in the engine's runtime (no kernel); `9fa93ea` itself was not booted.

## Decode after a long cold prompt

The first boot lost 10% after 32K (37.2 -> 33.4 tok/s, no graphs). With the graphs the decode after the **8K** prompt is 50.54 tok/s (the graph-decode record's short-context
figure is 53.4) and falls with the KV the step reads: 46.26 at 32K (-8.5% from 8K), 34.36 at 128K (-32%), 27.60 at 224K (-45%). These are the KV-read costs of 8 global visits
over 16,512 B a token, not a prefill side effect (`docs/DECODE-SPEED.md`, `ROOFLINE.md`: ceilings 61.5 / 43.3 tok/s at 32K / 128K); the prefill leaves nothing behind that slows
the next prompt (the four passes of a size agree within 1%).

## Where a 4,096-row chunk goes

`inkling2_loop_prefill_perf::prefill_curve` (new test, `PF_IDS_FILE` = the real ids of pass 0, 135,168 tokens, 33 chunks) under `nsys profile -t cuda` on the lab build of this
head (`tools/nsys_pf.sh`, `tools/prof_pf.py`: a chunk = the kernels between two embedding launches). Raw: `raw/profile-by-kernel-depths-0-8k-32k-128k.txt` (every kernel of the chunks at
context depth 0, 8,192, 32,768 and 131,072), `raw/pf-curve-nsys-135k.txt` (the host time of every chunk). The chunk at depth 0 spans **1,671.7 ms** (2,450 tok/s), union-busy
1,665.5 ms: **idle gaps are 6.3 ms (0.4%)**, 3,324 launches a chunk (mean 0.5 ms): the host is not in the way, and at 4,096 rows nothing is launch-bound. Milliseconds a chunk (the table of
`docs/PREFILL-SPEED.md` adds the work, the ceiling and the achieved fraction):

| Stage (launches a chunk) | depth 0 | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| MoE gate/up GEMM + softcap epilogue (378) | 448.5 | 439.2 | 429.0 | 448.5 |
| MoE down GEMM (378) | 213.5 | 206.1 | 205.8 | 211.3 |
| MoE input rotation `rot_in` (126) | 110.9 | 110.3 | 106.2 | 111.6 |
| MoE combine (48) | 74.0 | 74.4 | 74.2 | 74.2 |
| MoE plan / remap (631) | 8.1 | 7.9 | 7.7 | 7.8 |
| dense EXL3 GEMMs: q, gate, r, k, v, o (288) | 243.3 | 243.5 | 239.8 | 250.4 |
| dense input rotation `rot_in_frag` (288) | 48.6 | 48.2 | 48.0 | 50.0 |
| loop controllers: pair + blend/norm (192) | 118.0 | 119.4 | 117.2 | 118.6 |
| bf16 linears: controller hdn / gain / forget, dyn taps, sinks (400) | 96.7 | 97.4 | 89.8 | 95.8 |
| AttnRes mixes (120) | 64.6 | 65.5 | 64.5 | 64.8 |
| short convolutions (96) | 26.3 | 26.3 | 26.3 | 26.5 |
| relative-bias tables (48) | 53.5 | 53.7 | 52.9 | 53.7 |
| attention prologue (48) | 34.6 | 34.8 | 34.7 | 36.0 |
| attention core, 40 sliding visits | 90.5 | 98.4 | 102.5 | 99.8 |
| attention core, 8 global visits | 31.1 | 126.5 | 404.9 | 1,543.9 |
| ring write, conv commit, select, norm, embed (about 230) | 13.0 | 12.6 | 12.7 | 12.7 |
| **span of the chunk** | **1,671.7** | **1,760.6** | **2,013.2** | **3,210.1** |

Everything but the global attention is flat in the depth: 1,640 ms of a chunk at any depth, the global attention adds 31 / 127 / 405 / 1,544 ms. MoE (gate/up, down, rotation,
combine, plan) is 855 ms = 51% at depth 0. Chunk width (same prompt, 32K, `raw/chunk-width-sweep-32k.txt`): 1,024 rows 2,042 tok/s, 2,048 rows 2,219, 4,096 rows 2,273: wider is better
(a chunk's fixed cost about 44 ms, the weights' stream, `[INFERENCE]` from the three points), and the engine refuses 8,192; the last row's top-64 digest is the same at 1,024, 2,048 and 4,096.

## Commands (node-3 stands for the node's address)

```sh
git -C <engine worktree> rev-parse HEAD                                # 7da8c6134484 ... (40-hex in meta.json)
./inkling2mini.sh build 7da8c61 && ./inkling2mini.sh start             # SPEC=none PREFIX_CACHE=off VISION=off in .env
./inkling2mini.sh bench prefill-sweep                                  # twice; METRIC lines in raw/sweep-*/metrics.txt, rows in prefill-sweep.json
./inkling2mini.sh stop
NODE=node-3 LAB_SRC=<engine worktree> PKG=gb-model scripts/ops_dev.sh build inkling2_loop_prefill_perf
bench/records/2026-11-23-prefill-baseline/tools/nsys_pf.sh deep PF_TOKENS=135168    # then prof_pf.py deep.sqlite 1,3,9,33
NODE=node-3 LAB_SRC=<engine worktree> PKG=gb-model TEST_ENV='PF_IDS_FILE=<lab>/pf_ids.json PF_TOKENS=32768 PF_CHUNK=<n>' scripts/ops_dev.sh run inkling2_loop_prefill_perf prefill_curve
./inkling2mini.sh build 3df40c1 && EXTRA_ENV=REXL3_ALLOW_UNGATED_COMPONENTS=1 ./inkling2mini.sh start && PREFILL_SIZES=8192,32768 PREFILL_DECODE_AFTER=0 ./inkling2mini.sh bench prefill-sweep
```

The MoE layer experiments (`raw/moe-layer-experiments.txt`) are in the `prefill-speed` record's reading of them (`docs/PREFILL-SPEED.md`).
