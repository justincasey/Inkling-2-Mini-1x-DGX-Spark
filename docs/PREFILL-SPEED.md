# Prefill speed: where a 4,096-row chunk goes, and what moved it (engine branch `inkling2-loop-prefill`)

Plan Phase 7.2 (bit-preserving) and row 6.9. Numbers: `bench/records/2026-11-23-prefill-baseline` (the merged head `7da8c61`, the profile, the
experiments), `-prefill-speed` (the engine `1a36951`) and `-prefill-exact` (byte-exactness proofs). One Spark, TP1, GPU locked at 2,200 MHz, cold prompts
(a different one every pass), medians of three kept passes after a discarded one.

| Cold prefill | `7da8c61` | `1a36951` | change | % of the bf16 ceiling (`ROOFLINE.md`) | 25% target | 40% stretch |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 2,436.7 tok/s (3.362 s) | **2,493.3** (3.286 s) | +2.3% | 26.2% of 9,525 | 2,380 | 3,810 |
| 32K | 2,254.8 (14.533 s) | **2,304.4** (14.220 s) | +2.2% | 27.2% of 8,481 | 2,120 | 3,392 |
| 128K | 1,725.5 (75.960 s) | **1,754.9** (74.688 s) | +1.7% | 29.8% of 5,894 | 1,474 | 2,358 |
| 224K | 1,394.9 (164.435 s) | **1,415.2** (162.085 s) | +1.5% | 31.3% of 4,519 | 1,130 | 1,808 |

(224K: 49,152 x L + 9.40 GFLOP a token at L = 229,376 is 20.67 GFLOP, 4,519 tok/s at 93.4 TFLOP/s; the 25% and 40% columns apply the table's rule to it. Nothing here is
a measurement of the hardware: the ceilings are `ROOFLINE.md`'s arithmetic.) The proposed 25% targets are met at every size; the stretch is not. **The 14.5 s at 32K is
real and stable** (the head does it in every pass; its parent `3df40c1` booted here does 27.30 s): the relative-bias kernel of `9fa93ea` is the whole 2x
(`-prefill-baseline`). Decode after a cold prompt did not move (50.2 / 46.0 / 34.2 / 27.4 tok/s after 8K / 32K / 128K / 224K against 50.5 / 46.3 / 34.4 / 27.6: the
kernels are the same; the KV-read cost of the context is the loss, not the prefill). Outputs: every generated id of all 16 sweep passes and every pinned reply is equal.

## Where a chunk goes (`nsys -t cuda`, 4,096 rows, depth = tokens before the chunk; ms, baseline `7da8c61`)

| Stage | depth 0 | 8K | 32K | 128K | work a chunk | ceiling | achieved | bound |
|---|---:|---:|---:|---:|---|---:|---:|---|
| MoE gate/up GEMM | 448.5 | 439.2 | 429.0 | 448.5 | 16.70 TFLOP | 182 ms (fp16 91.9) | 41% | latency / occupancy |
| MoE down GEMM | 213.5 | 206.1 | 205.8 | 211.3 | 8.35 TFLOP | 91 ms | 43% | latency / occupancy |
| dense EXL3 (q, gate, r, k, v, o) | 243.3 | 243.5 | 239.8 | 250.4 | 10.55 TFLOP | 115 ms | 47% | latency / occupancy |
| MoE `rot_in` | 110.9 | 110.3 | 106.2 | 111.6 | 14.5 GB written | 62 ms (233.8 GB/s) | 56% | bandwidth (stores) |
| MoE combine | 74.0 | 74.4 | 74.2 | 74.2 | 16.3 GB | 70 ms | 94% | bandwidth |
| loop controllers (pair + blend) | 118.0 | 119.4 | 117.2 | 118.6 | 27-31 GB | 116-132 ms | at the floor | bandwidth |
| bf16 linears (hdn, gain, forget, taps, sink) | 96.7 | 97.4 | 89.8 | 95.8 | 10.8 GB written + read | about 47 ms | about 50% | bandwidth, small tiles |
| AttnRes mixes / short convs / dense `rot_in_frag` | 64.6 / 26.3 / 48.6 | 65.5 / 26.3 / 48.2 | 64.5 / 26.3 / 48.0 | 64.8 / 26.5 / 50.0 | bytes not derived | | | bandwidth `[INFERENCE]` |
| relative-bias tables | 53.5 | 53.7 | 52.9 | 53.7 | 9.7 GB written | 41 ms | 77% | bandwidth |
| attention prologue / ring and commit | 34.6 / 8.6 | 34.8 / 8.3 | 34.7 / 8.5 | 36.0 / 8.5 | | | | |
| attention core, 40 sliding visits | 90.5 | 98.4 | 102.5 | 99.8 | 2.06 TFLOP | 22 ms (bf16 93.4) | 22% | smem / latency `[INFERENCE]` |
| attention core, 8 global visits | 31.1 | 126.5 | 404.9 | 1,543.9 | 0.8 / 4.1 / 14.0 / 53.6 TFLOP | 9 / 44 / 150 / 574 ms | 28 / 35 / 37 / 37% | smem / latency `[INFERENCE]` |
| **span of the chunk** | **1,671.7** | **1,760.6** | **2,013.2** | **3,210.1** | | | | |

* **Not launch-bound.** 3,324 launches a chunk (mean 0.5 ms), idle gaps 6.3 ms (0.4%) at depth 0 and 12.4 ms at 128K; one stream, no host wait inside a chunk.
* **Not tensor-bound, not bandwidth-bound in the GEMMs.** Lab builds of the MoE layer (`bench_prefill_layers`, real layers, 4,096 rows, `raw/moe-layer-experiments.txt` of the baseline
  record) with one part of the GEMMs' work taken out, garbage outputs on purpose: no mma 16.5 -> 15.4 ms, no tile decode 16.5 -> 16.0, no warp reduction 16.5 -> 15.3, every A
  fragment the same row 16.5 -> 16.2, every weight tile the same tile 16.5 -> 15.8. Each part is worth 1 ms or less of a 16.5 ms layer. The GEMM kernels run 8 warps an SM (250 registers a thread,
  80 KB of shared memory for the gate/up block) with their loads, decode, mma and reduction in sequence, and 40% of the fp16 peak is what that costs (`[INFERENCE]` from the five removals and the
  register report: 251 / 249 / 232 registers). Rows scale with a fixed 3 ms (the weights, 0.57 GB at 233.8 GB/s) plus 3.9 ms a 1,024 rows.
* **Global attention is the only stage that depends on the depth** and the first stage at 128K (48% of a chunk): 38% of the bf16 peak with 255 registers (and 48-56 bytes of stack) a thread, 6 warps an SM.
* What a chunk costs apart from the global attention is 1,640 ms at any depth, so 32K = 8 chunks at 1,672-2,013 ms and 224K = 56 chunks up to 4.5 s.

## What changed (one change a build, each proven byte-identical before it was timed)

| # | Change (commit) | 8K | 32K | 128K | 224K |
|---|---|---:|---:|---:|---:|
| 0 | the merged head `7da8c61` | 2,436.7 | 2,254.8 | 1,725.5 | 1,394.9 |
| 1 | `1a36951`: the mul1 4-bit MoE gate/up and down GEMMs run the 3-4-tile list as blocks of two tiles, two to a group (grid z): half the accumulators (128 registers), two blocks an SM, 16 warps for 8. One layer of 4,096 rows 16.46 -> 15.43 ms (layer 0), 17.07 -> 16.41 (12), 17.19 -> 16.10 (22), 17.23 -> 17.46 (17: 68 of its 128 experts are 5-bit and keep their kernels); gate/up kernels 8-9% faster, down 10%; the 16K prefill test 2,387 -> 2,441 tok/s | 2,493.3 | 2,304.4 | 1,754.9 | 1,415.2 |

The commit is `FEAT: Ran the Mini's 4-bit MoE prefill GEMMs as 2-tile blocks for more warps an SM`; the Small's mcg widths and the 5-bit classes keep their kernels (the width test is `K2 == 8 && CB == 2`).

### Directions measured as no gain (each against the 16K prefill test, `prefill_curve`, same prompt; reverted, not committed)

| Direction | Result (tok/s; reference in the row) |
|---|---|
| dense EXL3 GEMM, 2-tile groups, 2 sweeps, 3 blocks an SM | 2,433-2,434 against 2,441: no gain |
| dense EXL3 GEMM, 2-tile groups, 1 sweep, 2 blocks an SM | 2,379-2,384 against 2,441: 2.3% slower |
| MoE, one tile a block (four to a group, three blocks an SM) | 2,201-2,203 against 2,441: 10% slower (the decode of each tile four times) |
| MoE, 2-tile blocks with the weight prefetch two steps ahead | 2,332-2,338 against 2,441: 4% slower |
| side streams in the prefill (K and V branches with their controllers, as the decode lanes: all visits / only the visits with controllers) | 2,397-2,409 / 2,401-2,405 against 2,444-2,450 with the switch off: 1.8% slower; scratch for two lanes at 4,096 rows costs about 260 MB (arithmetic) |
| `rot_in` and `rot_in_frag` assembled in shared memory and stored as contiguous 4 KB (instead of 32 partial sectors a warp) | 2,444 against 2,441: no change (110.3 against 110.9 ms and 51.4 against 48.6 ms in the profile): these kernels were not store-bound |
| chunk width | 1,024 rows 2,042, 2,048 rows 2,219, 4,096 rows 2,273 (32K); the engine refuses 8,192; about +1% from 8,192 `[INFERENCE]`, the weights' fixed cost per chunk is about 44 ms |

The non-gains are one direction each, stopped after two measurements (dense tiles, side streams) or after the number was far off (one tile, prefetch).

## What is left (arithmetic; the causes are `[INFERENCE]` unless a row above measured them)

At depth 0 the chunk is 1,610 ms after the change (`nsys`, 8K prompt). Its three GEMM families take 849 ms against 388 ms at the fp16 peak (461 ms over, 29% of the chunk), the sliding
attention 68 ms over its bf16 floor, the five byte-floored glue families (rotation, combine, controllers, bf16 linears, bias tables) about 115 ms over 344 ms of bytes, and about 180 ms
(mixes, short convs, prologue, dense rotation, commits) have no byte count here. Making the GEMMs occupy an SM better than 8 or 16 warps of sequential decode / mma / reduce phases (a
software-pipelined, warp-specialised kernel with the same accumulation order), a larger row block for the global attention (its fold state alone is 64 of the 255 registers a thread), and fusing
the controllers' gain / forget GEMV into the blend (two 37.7 MB fp32 arrays a launch, 192 launches) are the three places a further 10-20% is, none cheap; the numerics-changing options (a different
accumulation order, bf16 P changes) are Segment B and were not measured; fp8 prefill was (next section).

## Segment B: fp8 prefill measured, NO-GO (`bench/records/2026-11-23-fp8-potential`, `-fp8-numerics`; engine branch `inkling2-loop-fp8`, `ba571a9`)

The A lane's mul1 4-bit gate/up and down GEMMs, rebuilt as one template (byte-identical to the shipped kernels at fp16, asserted) and instantiated at e4m3 operands (`cvt.rn.satfinite.e4m3x2.f16x2` on the decoded
codebook values, `m16n8k32` mma, half the A traffic), timed in the same structure on the Mini's shapes (node-4, three runs, L2 flushed): the e4m3 / fp16 time ratio is **0.82-0.85 (gate/up), 0.92-0.93 (down), 0.88
(dense)**, 25-33% of the measured e4m3 roofline (195.1 TFLOP/s) where the A lane's kernels are at 47-57% of the fp16 one. This is what the removal experiments above predicted (the kernels are not mma-bound).
Projected onto the stage table (derived): -113 ms of a 1,610 ms chunk from the GEMMs (-193 ms with an optimistic halving of the rotation stages' writes), **+6.8% to +12.2% on the 32K cold prefill**, +19% at the
unreachable bound of both GEMM families at their measured rooflines, against the gate's +25%. The numerics side fails independently (`NUMERICS.md`, "B lane"). The lane was not built; the template and the
bench (`inkling2_fp8_bench`) stay on the branch as the instrument that re-measures the ratio if an A-lane kernel ever becomes mma-bound.


## Tools

`bench/tools/prefill_sweep.py` (`./inkling2mini.sh bench prefill-sweep`: cold 8K / 32K / 128K / 224K, unique prompts, warm-up discarded, TTFT from the server's histogram, the decode after each
prompt); engine test `inkling2_loop_prefill_perf::prefill_curve` (`PF_IDS_FILE`, `PF_TOKENS`, `PF_CHUNK`, `PF_EVERY=1` for a digest of every chunk's last row); `inkling2_moe_mixed::bench_prefill_layers`;
`bench/records/2026-11-23-prefill-baseline/tools/{nsys_pf.sh,prof_pf.py,nsys_moe.sh}` (profile a chunk by stage; `nsys -t cuda`, no counters: `ncu` is not installed on the node).

## The push for 3,000 tok/s at 32K (engine `ce650e3`, records `prefill-push` and `prefill-push-exact`)

Owner's target: cold prefill over 3,000 tok/s at 32K, byte-exact. **Not reached.** The shipping configuration (`SPEC=dspark PREFIX_CACHE=on VISION=on`, the prompts sent as a user sends them) does
**2,302.2 tok/s at 32K** (2,435.6 at 8K, 1,760.2 at 128K, 1,418.8 at 224K; `SPEC=none`: 2,344.7 / 2,542.6 / 1,777.4 / 1,429.5), against 2,268.0 / 2,399.9 / 1,739.0 before the push on the merged head: +1.5%, +1.5%, +1.2%
(`SPEC=none` +1.7 / +2.0 / +1.3 / +1.0%). 3,000 tok/s at 32K is eight chunks of 1,365 ms; a chunk averages 1,779 ms: 414 ms (23%) is missing.

### What moved (server medians are in the records; per-change screening is the 32K prompt of `prefill_curve`, tok/s, same prompt and digest every time)

| # | Change (commit) | 32K screen | what it did |
|---|---|---:|---|
| 0 | the merged head `1d53344` | 2,327 | |
| 1 | `76c02f8` chunk cap 8,192 and the dense large-row kernel's strips ordered fastest | 2,327 (chunk 4,096) / 2,364 (8,192) | the dense kernels re-read every row group's A once a strip; at 8,192 rows that no longer fits L2 and they took 1.9x as long a row. 4,096 rows: unchanged |
| 2 | `8ad2704` the two halves of a 2-tile MoE item side by side in the grid | 2,350 (+1.0%) | the halves were the grid's z axis, so the second halves ran after every first half and read the experts' weights from DRAM again: layer 0 15.43 -> 14.96 ms, 12 16.41 -> 15.82, 22 16.10 -> 15.59 |
| 3 | `acaf2cb` the 5-bit class split as well | 2,358 (+0.35%) | a layer with 68 5-bit experts 17.23 -> 16.62 ms |
| 4 | `7e6eab1` `--prefill-chunk` up to 8,192 for the Mini (serve.rs: the cap follows the family) | | the default stays 4,096 (below) |

### Findings (every one with its measurement; the first three are the argument that GEMM structure is where a further step is)

* **The MoE layer's main loop is 9.2 ms of 15.4.** A lab build whose GEMMs skip their whole k loop (garbage output; `bench_prefill_layers`) takes 6.3 ms a layer: rotation 1.9, combine 1.5, the two epilogues (the y writes alone are 302 MB) and
  the plan. The loop is 9.2 ms against 5.5 ms of tensor floor at 91.9 TFLOP/s: 60%. Whatever a restructured GEMM could do to it is at most 3 ms a layer (about 140 ms a chunk, 9%), in the
  best case.
* The weights' DRAM reads are not hidden by the compute: a layer costs a fixed 3 ms (the experts' 0.57 GB) plus 3.9 ms a 1,024 rows; schedule order moves it (finding 2 above: 3-3.6% of a layer). **A hint to L2 does not**: `prefetch.global.L2` of the strips
  the blocks one and two waves later will read, from the blocks themselves, made the layer 6% (distance 96 blocks) and 5.7% (36 blocks) slower; stopped.
* 2-tile blocks spill a little (128 registers: 40 B of stack in the gate/up kernel, 16 B in the down kernel); pointing the registers at a 3-block cap for the down kernel or dropping the A prefetch of the gate/up kernel did not help (16.07 / 16.17 against 15.43 / 16.41: noise).
* **Controller gain / forget / blend in one launch** (no fp32 gain and forget arrays; mma chains and the blend's sum tree reproduced, the digest unchanged): the fused kernel takes 1.37 ms against 1.42 ms for the three launches, 16 and 32 rows a block, with or
  without the `tanhf`: no gain (0.6% of a chunk at best), not kept. **Dense EXL3 with 2-tile groups**: a third time no gain (2,344 against 2,357). L2 prefetch, fused controllers, dense tiles: three directions closed.
* **8,192-row chunks** are exact (replies hash, 135,168-token digest, 12 sweep passes) and 1.4-1.8% faster at 8K / 32K / 128K shipping (2,479.3 / 2,334.9 / 1,789.3), but a decoding stream stands still behind a chunk: the longest gap of a stream while a 32K prompt
  prefills is **1.97 s at 4,096 rows and 3.80 s at 8,192**. The default stays 4,096; `PREFILL_CHUNK=8192` is there for an operator who wants prefill throughput over inter-token latency. The plan's step buffers grow with the chunk (`reserve_gib` 6.42 at 8,192).
* **Shipping against `SPEC=none`:** 4.2% at 8K, 1.8% at 32K, 1.0% at 128K on the same engine; the cache's 64-row tail pass at the cut is a whole pass over the experts (about 100 ms, `[INFERENCE]`), which is the 8K figure.

### What the arithmetic says is left

A chunk at depth 0 is 1,586 ms now (1,672 before). The measured floors (`ROOFLINE.md` rates): the three GEMM families 388 ms at the fp16 peak against 821 ms (gate/up 397.6, down 186.2, dense 237.6), the byte floors of the glue families 344 ms against about 460 ms,
sliding attention 22 ms against 88 ms, and about 180 ms of mixes, convs, prologue, rotation and commits whose bytes are not derived. To reach 1,365 ms a chunk needs 414 ms at 32K. If a pipelined, warp-specialised GEMM brought the MoE
loop from 60% to 80% of the tensor floor (about -140 ms) and the dense kernels from 47% to 65% (about -65 ms), and fusing the glue passes saved a quarter of its 115 ms excess (-30 ms), that is about -235 ms: **about 1,550 ms a chunk, roughly 2,600-2,700 tok/s at 32K,
byte-exact `[INFERENCE]`.** 3,000 byte-exact is out of reach of what was found; the fp8 lane is the path beyond. The attention is the other cost beyond 32K: the 8 global visits are 1,523 ms of a 3,096 ms chunk at 128K at 37% of the bf16 peak with 255 registers
and 56 bytes of stack a thread (6 warps an SM; a larger tile needs the fold state, 64 of the registers, elsewhere).

### MoePipe: a pipelined MoE GEMM with the rotation fused was measured before it was built, and is a no-go (record `prefill-moe-pipe-bounds`)

On the integration head, one real layer of 4,096 rows is 14.96 ms (layer 0) of which the k loops are 8.7 and the rest 6.3 (rotation 1.9, combine 1.5, epilogues and plan 2.9). The loop's compute path alone (every load an L1 hit) is 7.0 ms, 79% of the
5.5 ms tensor floor; its memory path alone 5.6 ms. So a perfect producer/consumer ring wins at most 1.7 ms (11%); the rotation is worth 1.4 ms free but one pass is 0.8 ms of ALU and a per-strip fusion recomputes it 4.5-9 times on the compute-bound path (3.6-7.2 ms);
the epilogues are worth at most 1.5 ms. All three at once: -30%; the realistic part 7-10%, under the 15% gate. Not built; the engine is unchanged.

## The dense pipe (engine `fd6ef61`, records `prefill-dense-pipe` and `prefill-dense-pipe-exact`)

The dense half of the same idea: the six EXL3 projections of a visit (q, gate, r, k, v, o) and their rotations took 289 ms of a 1,589 ms chunk (the rotation 49.8 ms, DRAM-bound at 0.16 ms a pass; the GEMMs 239 ms at 45-47% of the fp16 peak). **Byte-exact (every gate and live check of
`prefill-push-exact` reproduced); the speed gain is small: -39.8 ms of the dense stages and -48.0 ms (-3.0%) of a chunk at depth 0, +2.4% at 32K shipping (2,301.4 -> 2,357.4 tok/s, the same binary with `REXL3_INKLING2_LOOP_DENSE_PIPE=0` against the default), 2,498.9 at 8K, 1,791.3 at 128K, 1,439.5 at 224K;
`SPEC=none` 2,400.8 at 32K. The 3,000 tok/s target stays out of reach by this lane (-21%).** MoePipe's changes add to this.

| Stage, ms a 4,096-row chunk at depth 0 / 8K / 32K / 128K (`nsys`, pipe off -> on) | depth 0 | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| dense GEMMs (288 launches) | 239.3 -> 208.9 | 239.3 -> 210.6 | 237.9 -> 207.7 | 235.8 -> 207.3 |
| dense rotation (288 -> 172 launches) | 49.8 -> 40.5 | 49.4 -> 40.6 | 49.1 -> 41.0 | 49.4 -> 40.3 |
| span of the chunk | 1,589.0 -> 1,541.0 | 1,686.5 -> 1,642.2 | 1,959.4 -> 1,918.8 | 3,092.1 -> 3,058.8 |

What the lab found (`bench/records/2026-11-23-prefill-dense-pipe/`, 4,096 rows, real layer 11, L2 flushed): the shipped dense GEMM is not bound by one thing, but its cross-warp reduction through shared memory (eight sweeps behind two block barriers per 16-row tile) costs 0.30 ms of 1.24 ms by itself, and
the A fragments read a warp at a time through L1 cost as much as they save when the reduction is removed. The kept design: eight warps each owning a 16-column tile and the whole of K (the decode plan's 16 chains run in order by one warp and summed in registers in the same order: every output
element's fp32 sequence is the old one), the A fragments staged by `cp.async` into a four-stage shared-memory ring coupled by mbarriers, the rotation done once for the tensors that share an input (up to three outputs: q, gate, r; k and v on the visits whose inputs are q's). The GEMM goes from
47% to 61% of the fp16 peak (q 1.239 -> 1.024 ms); a visit's six projections from 5.898 to 5.015 / 4.942 ms (-15.0% / -16.2%: the go/no-go gate of 15% met at the line). The lead's arithmetic assumed the rotation could be recomputed in the GEMM's A producer; per 128-column strip that is 24 times for q and gate
(0.33 ms of shuffle pipe a GEMM against a 0.63 ms tensor floor), so the rotated copy stays materialised and the effort went into the GEMM. Measured as no gain or a loss and not kept: a producer warp with `cp.async.bulk` copies (ninth warp caps the registers at 96: spills, 1.39 ms), the rotation hosted in the GEMM's tail
(a visit 5.40 ms against 4.57 for the same GEMMs with no hosted work), the rotations of the visit on a second stream (-4.4% and 0.0% in two runs), a one-stage ring (1.30 ms). The pipe takes the Mini's six 6-bit shapes above 128 rows only, needs 49,280 B of shared memory a block (two blocks an SM) and +30 MiB of
workspace at 4,096 rows (`dense_pipe_rotated_inputs` in the memory plan).

What is left (`[INFERENCE]` unless a lab number says so): a visit's GEMMs 3.90 ms against a tensor floor of 2.4 ms (220 GFLOP at 91.9 TFLOP/s; the mma-only skeleton measured 0.68 ms of the q GEMM's 1.02 against its 0.63 ms floor, decode 0.16 more, coupling and the strip tile 0.18), and 0.9 ms of rotation a visit, DRAM-bound. A further 20 ms a chunk is
the rotation done by the kernel that produces x (the controllers' blend and norm), whose store is already bandwidth-bound; the dense lane alone does not reach the 3,000 tok/s asked.
