# prefill-moe-pipe-bounds: what a pipelined MoE GEMM with the rotation fused can win, measured before building it (gate `moe-pipe-bounds`)

Result: **no-go** for the MoePipe prototype (the lead's rule: build only if a layer of 4,096 rows can drop by 15%, 16.5 -> 14.0 ms or better). No kernel was written; nothing changes in the engine. The record is a set of lab
measurements on the current head, so it carries its own gate name and is NOT a `speed-prefill` record (it has no served number). Engine `4f939a6` (the integration head, itself the `ce650e3` tree plus the other lanes), every lab build
patched by one `-D` experiment macro that removes a part of the work (garbage output on purpose; the macros are in the commands below, not committed), node-3 alone, SM clock locked at 2,200 MHz,
`inkling2_moe_mixed::bench_prefill_layers` (`REXL3_MOE_BENCH=1`): real layers 0 (one 4-bit band) and 17 (three bands, 68 5-bit experts), 4,096 rows of random routing, L2 flushed, median of 9, ms for the whole layer
(rotation + gate/up + down + combine), 2-tile blocks as shipped.

## The numbers (ms a layer, 4,096 rows; the current kernels: 14.96 / 16.62)

| Lab variant (what is removed) | layer 0 | layer 17 | what it bounds |
|---|---:|---:|---|
| the shipped kernels | 14.96 | 16.62 | |
| **no k loop** (the GEMMs' loops skipped; rotation, epilogues, combine, plan stay) | 6.27 | 7.31 | the fixed part: 6.3 ms; **the loop is 8.7 ms** |
| **compute only**: every A and every weight tile read from one line (L1 hits), decode, mma, reduction real | 13.30 | 14.57 | the loop's compute path alone is **7.0 ms** (the tensor floor of the layer's 25.05 useful TFLOP at 91.9 is 5.5 ms: 79% already) |
| **memory only**: no mma, no decode, no reduction; the loads real | 11.87 | 13.08 | the loop's memory path alone is 5.6 ms |
| rotation free (`rot_in_kernel` returns at once) | 13.55 | 15.10 | fusing the rotation can save at most 1.4-1.5 ms (9%) |
| both epilogues free (gate/up and down return after their loops) | 13.43 | 15.19 | the epilogues (the 302 MB of fp32 y alone is 1.3 ms at 233.8 GB/s) are worth at most 1.5 ms (10%) |
| rotation compute only (no store, x from L1): `rot_in_kernel` | 0.81 ms a pass (nsys; 1.9 ms real) | | **one** rotation pass costs 0.8 ms of ALU |

## Why this is a no-go

1. **A pipelined mainloop is bounded by 1.7 ms of 14.96.** The loop is 8.7 ms; its compute path alone is 7.0 and its memory path alone 5.6. A producer/consumer ring that hid the memory completely would give max(7.0, 5.6) = 7.0 ms: -1.7 ms (11%),
   and that is the perfect case. The shipped loop already overlaps most of it (8.7 against a serial 12.6). The compute path is at 79% of the tensor floor, so moving the decode to other warps (it costs 0.5 ms when removed) cannot reopen the gap.
2. **Fusing the rotation into the GEMM's A producer recomputes it per strip.** A block owns one 128-column strip of the expert (9 strips a gate/up item, 18 for down) and needs the whole rotated row; the rotated row of a (pair, matrix)
   cannot stay resident for the 9 strips (32 rows x 2,304 x 2 B = 147 KB a matrix). One rotation pass is 0.8 ms of ALU, so the fused producers would do 9x (4.5x with the two halves of an item sharing) = 3.6-7.2 ms of ALU on the compute-bound path
   (7.0 ms), against the 1.4 ms it removes. Fused, the layer gets slower, not faster.
3. **Everything free at once** (pipelined loop -1.7, rotation -1.4, epilogues -1.5) is 14.96 -> about 10.4 ms (-30%); that needs the epilogue's 302 MB of y writes to cost nothing, the rotation to cost no ALU, and perfect overlap, three things the byte floors and (2) rule out.
   Capturing the realistic part (half of the loop's bound, the rotation's write moved from DRAM to L2, a tenth of the epilogue) is about -1.0 to -1.5 ms: 7-10%, below the 15% gate, for a kernel project of the size the lead described.
4. The lead's family estimate (772 ms at depth 0 -> 450-500 ms) needs the GEMM kernels at about 70% of the tensor floor *including* their epilogues and the rotation. The shipped gate/up and down kernels take 11.4 ms a layer (8.0 + 3.4,
   epilogues included, nsys) against the 5.5 ms floor: 48%; the loop's compute path alone is already at 79%, and of the 5.9 ms between 5.5 and 11.4, the loop's memory stalls are at most 1.7, the epilogues at most 1.5 and the rest is the skeleton around
   the mma. The achievable reduction of the family is about 40-70 ms a chunk, not 270-320.

## The design that was not built (for the successor)

Producer warps decode a trellis tile into a 3-stage smem ring of fp16 B fragments (512 B a tile, a pair of warps a K range, mbarrier full/empty per stage); consumer warps run today's mma sequence reading B with `ld.shared.v4`; 12-16 warps a block
(8 consumers, 4-8 producers) at 128 registers; smem = ring 48 KB + stage 32 KB + reduction 16 KB = 96 KB of the 101,376 B. Budget by arithmetic: at best the 1.7 ms of finding 1. The rotation would stay a separate kernel. What would change the verdict: a rotation
that is computed once per row and used by all strips (a persistent block per item looping over the 9 strips needs the rotated rows of two matrices resident: 294 KB for 32 rows, 147 KB for 16: more than the 101 KB of smem), or a part with more shared memory.

## Commands (node-3 stands for the node's address)

```sh
# the macros, applied to the head's csrc/exl3_grouped.cuh, csrc/exl3_linear_decode.cuh and csrc/moe_prefill.cu (the first line of moe_prefill.cu carries the -D):
#   EXP_NOMMA   ksweep: acc += a ^ b instead of the two mma16816      EXP_NODEC  decode_lane_at: b = the lane's words        EXP_NORED  reduce_sweep replaced by one store
#   EXP_ASAME   the A fragment address uses (kt & 1)                   EXP_WSAME  the weight tile address uses 0              EXP_NOLOOP group_sums returns after filling the stage
#   EXP_NOROT   rot_in_kernel returns first                            EXP_NOEPI  gate/up and down return after group_sums    EXP_ROTALU rot_in: x row & 7, the store guarded by a false test
NODE=node-3 LAB_SRC=<engine worktree> PKG=gb-cuda TEST_ENV='REXL3_MOE_BENCH=1 REXL3_PF_LAYERS=0,17' OPS_GREP=BENCH_PF scripts/ops_dev.sh test inkling2_moe_mixed bench_prefill_layers
bench/records/2026-11-23-prefill-baseline/tools/nsys_moe.sh rotalu REXL3_PF_LAYERS=0 REXL3_PF_ITERS=5
```
