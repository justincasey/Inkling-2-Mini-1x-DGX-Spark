# fp8-potential: is an e4m3 prefill GEMM worth building for the Mini? (gate `fp8-potential`, plan row 7.3 step 2)

Result **fail** (`meta.json`): the decision gate's condition (a) does not hold. "Fail" says the gate's condition is not met; the measurements are complete.
**NO-GO for the B lane**: in the A lane's own kernel structure an e4m3 GEMM is 8-18% faster than fp16, not 2x, which projects to +7% (GEMMs
only) to +12% (with an optimistic count of the halved A-operand traffic) on a 32K cold prefill against the gate's +25%; even with both GEMM families at their measured
roofline the fp8-specific gain is +19% at 32K. Condition (b) failed too (`2026-11-23-fp8-numerics`: 60-181x the A lane's error per op). Step 3 was not started.

## What was measured

`crates/cuda/tests/inkling2_fp8_bench.rs` (engine `ba571a9e007f`, branch `inkling2-loop-fp8`, node-4, GPU idle, SM clock 2,197 MHz sampled before and
after, L2 emptied by a 256 MB read before every timed launch, median of 9 timed launches after 2 discarded, three full runs of the test, the table
shows the median of the three runs and each run's ratio). No pack: random EXL3 trellis (the kernels' time does not depend on the values), the Mini's shapes:
128 experts, K = 2304 -> N = 1152 gate and up (one launch, two teams), 1152 -> 2304 down, mul1 4 bits, 4,096 rows of top-8 (256 pairs an expert) and of
top-16 (512 pairs an expert: the MoE chunk the engine cannot run, 4,096 rows is its limit), over the fused plan `run_prefill` uses; dense 4,096-row projections as one
expert of 256 tiles. Three kernels per shape:

* (a) the A lane's own (`gb_moe_pf_gateup_softcap`, `gb_moe_pf_down_sized`, the integration head's);
* (c) `csrc/inkling2_fp8_gemm.cu` at F8 = false: the A lane's structure (the 2-tile split blocks of `moe_prefill.cu`, `exl3_grouped.cuh`'s loop) as a template. The test asserts it equal to (a)
  **byte for byte** (gate/up output words and every `y` float of the down GEMM, both plans): the template IS the shipped kernel, and its time is within 2.5% of (a);
* (b) the same template at F8 = true: e4m3 A fragments (two 16-k tiles to a 16-byte lane word: half the A traffic), the decoded fp16 codebook values converted in registers
  (`cvt.rn.satfinite.e4m3x2.f16x2`), `mma.sync.m16n8k32` e4m3 with fp32 accumulation (half the mma instructions), and an epilogue that quantizes the down input rows to e4m3 with an
  fp32 scale a (row, 128 columns). A layout check in the test compares (b) with (c) on identical logical rows: the e4m3 down GEMM differs from fp16 by 2.67% relative L2 (the weights'
  rounding alone), the dense ones by 2.66-2.67%, the gate/up chain by 12.5% (random weights: the softcapped activation and two roundings); a layout error would be O(1).

What (b) is not: tuned. It uses the A lane's ring and registers unchanged (no spill was looked for). That is the point of the same-structure test: the previous lane's removal experiments
(`docs/PREFILL-SPEED.md`: taking the mma out saved 1 ms of a 16.5 ms MoE layer) said these kernels are limited by occupancy and phase sequencing, not by the mma, and this is what they predicted.

## Result

| shape (rows x K x N or experts) | (a) A-lane kernel ms | (c) template fp16 ms | (b) template e4m3 ms | ratio b / c (median; the 3 runs) | TFLOP/s a / c / b |
|---|---:|---:|---:|---|---|
| MoE gate/up, 256 pairs an expert | 8.018 | 7.840 | 6.524 | 0.846 (0.868, 0.811, 0.846) | 43.4 / 44.4 / 53.3 |
| MoE down, 256 pairs an expert | 3.716 | 3.763 | 3.500 | 0.931 (0.931, 0.932, 0.928) | 46.8 / 46.2 / 49.7 |
| MoE gate/up, 512 pairs an expert | 15.192 | 15.151 | 12.488 | 0.824 (0.837, 0.824, 0.815) | 45.8 / 45.9 / 55.7 |
| MoE down, 512 pairs an expert | 6.659 | 6.813 | 6.240 | 0.919 (0.919, 0.891, 1.011) | 52.2 / 51.1 / 55.8 |
| dense 4096x2304x3072 | (no A-lane kernel of this shape here) | 1.061 | 0.932 | 0.878 (0.876, 0.878, 0.878) | - / 54.7 / 62.2 |
| dense 4096x2304x1024 | (no A-lane kernel of this shape here) | 0.379 | 0.334 | 0.882 (0.886, 0.866, 0.882) | - / 51.0 / 57.9 |
| dense 4096x3072x2304 | (no A-lane kernel of this shape here) | 1.036 | 0.909 | 0.878 (0.879, 0.877, 0.878) | - / 56.0 / 63.8 |

Achieved rates (useful flops: pairs x 2 x K x N a matrix): the A lane's kernels run at 43-52 TFLOP/s (47-57% of the measured fp16 roofline 91.9); the e4m3 template at 50-56 TFLOP/s on the MoE shapes and 58-64 on
the dense ones, **25-33% of the measured e4m3 roofline 195.1**. The fp8 / fp16 time ratio is 0.82-0.85 (gate/up), 0.92-0.93 (down) and 0.88 (dense), never the roofline's 0.47.
The records' cuBLAS reference for the same shapes is `2026-11-23-roofline` (bf16: 56.8 TFLOP/s at 256 x 2304 x 1152, 50.8 at 256 x 1152 x 2304, 80.7-82.2 at 4096 x 2304 x 3072 / 4096 x 3072 x 2304): the small expert GEMMs
do not reach 91.9 even for a library, so the fp16 roofline column of the stage table is not a target an fp16 kernel could meet either.

## The projection (derived from the table above and `docs/PREFILL-SPEED.md`'s stage table; not a measurement)

Stage times of a 4,096-row chunk at depth 0 (baseline `7da8c61`): MoE gate/up 448.5 ms, MoE down 213.5, dense EXL3 243.3; the head `1a36951`'s chunk is 1,610 ms at depth 0 (its `nsys` of an 8K
prompt) and the 32K cold prefill is 14.220 s (2,304.4 tok/s), 8 chunks. Savings per chunk apply the measured medians (gate/up 0.846, down 0.931, dense 0.878 (the three shapes: 0.878, 0.882, 0.878)) to those stages:

| item | arithmetic | ms a chunk |
|---|---|---:|
| MoE gate/up | 448.5 x (1 - 0.846) | 69.1 |
| MoE down | 213.5 x (1 - 0.931) | 14.7 |
| dense EXL3 | 243.3 x (1 - 0.878) | 29.7 |
| **GEMMs, measured ratios** | sum | **113.5** |
| glue, optimistic and unmeasured: `rot_in` 110.9 ms and dense `rot_in_frag` 48.6 ms write half the bytes | 0.5 x (110.9 + 48.6) | 79.8 |
| **GEMMs plus glue** | sum | **193.2** |

| | chunk at depth 0 | cold 32K (8 chunks, the attention and the other glue unchanged) | gain | gate (a): +25% |
|---|---:|---:|---:|---|
| the head `1a36951` | 1,610 ms | 14.220 s, 2,304 tok/s | | |
| fp8, GEMMs only | 1,496.5 ms (-7.0%) | 13.312 s, 2,462 tok/s | **+6.8%** | not met |
| fp8, GEMMs plus the optimistic glue | 1,416.8 ms (-12.0%) | 12.674 s, 2,585 tok/s | **+12.2%** | not met |
| the unreachable bound: both GEMM families at the measured rooflines, fp8 182.5 ms against fp16 387.4 ms (35.6 TFLOP / 195.1 and / 91.9) plus the same glue | 1,325 ms (-17.7%) | 11.9 s | **+19.1%** | not met |

The last row is the best the number 2.1x can do beyond an fp16 kernel that already ran at its roofline (which none does: row above); what a restructured kernel buys beyond the A lane's 43-52 TFLOP/s is
a structure gain that an fp16 kernel takes as well (PrefillLane's lane, byte-exact), and fp8 only gets the ratio of the same structure on top: 8-18% of the GEMM stages. A faster A lane makes the fp8
share smaller, not larger, in absolute milliseconds (the GEMM stages shrink; the glue and the attention are untouched in both).

## Decision

* (a) the projected whole-prefill gain at 32K: **+6.8% to +12.2%** (derived above), bound +19.1%, against the gate's +25%. **Not met.**
* (b) the numerical deviation fp8 adds against the A lane's own (`2026-11-23-fp8-numerics`): 60-181x per op, 1.5-2.6x in the whole-forward hidden plane, 4-10x in KL, 7 of 139 rows (1 of 40 and 6 of 49 on the chat prompts) with another argmax than the exact oracle (the A lane: 0). **Not met.**

**NO-GO.** The B lane is not built: no switch, no cache-namespace change, no rank hook were written (the engine branch holds only this lab template and bench). If a later A lane kernel
makes the GEMMs mma-bound (a software-pipelined, warp-specialised structure, `docs/PREFILL-SPEED.md` "What is left"), the template here is the instrument that re-measures the fp8 / fp16 ratio on equal footing.

## Not measured

* The 5-bit classes (13.8% of the experts, the A lane keeps their kernels; the template is the 4-bit mul1 instantiation) and the real routing skew (random uniform picks: the least skewed case).
* A tuned e4m3 kernel (a different tile shape, a deeper ring, a per-row scale pass fused into `rot_in`): the removal experiments bound what such a kernel can gain (the mma and the A traffic are 1-3 ms of 16.5 per layer).
* The glue savings (`rot_in` writing e4m3) are an upper bound by arithmetic, not a kernel.

## Reproduce

```text
NODE=<node-4> LAB_SRC=<engine worktree at ba571a9e007f> scripts/ops_dev.sh test inkling2_fp8_bench fp8_gemm_bench     # 20 s a run; three runs in raw/
```
