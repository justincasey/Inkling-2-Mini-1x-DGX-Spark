# prefill-dense-pipe: the Mini's dense prefill projections pipelined (gate `speed-prefill`; byte-exactness in `2026-11-23-prefill-dense-pipe-exact`, same engine)

Result **pass** (a measurement). Engine `fd6ef616be704ad0863bd5dbebdc97408527d4bb` (branch `inkling2-loop-densepipe` on the integration head `4f939a6`: `15d8ad3` the kernels, `fd6ef61` the wiring; `b31cf9d` is the lab commit
the prototype numbers below come from), lab build on node-3, TP1, one GPU, SM clock locked at 2,200 MHz (the controller applies and checks the cap with `scripts/gpu_cap.sh`; the proof of each run is below), cold prompts
(a different one every pass), medians of three kept passes after a discarded one. **The owner's 3,000 tok/s at 32K is not reached by this lane: 2,357.4 tok/s shipping, +2.4% over the switch-off build, 21% short.**

| Cold prefill, served | `ce650e3` record | this build, pipe off (`=0`) | **this build, pipe on** | pipe on vs off |
|---|---:|---:|---:|---:|
| shipping (`SPEC=dspark PREFIX_CACHE=on VISION=on`) 8K | 2,435.6 | 2,442.2 | **2,498.9** (3.278 s) | +2.3% |
| 32K | 2,302.2 | 2,301.4 | **2,357.4** (13.900 s) | +2.4% |
| 128K | 1,760.2 | 1,759.6 | **1,791.3** (73.173 s) | +1.8% |
| 224K | 1,418.8 | 1,418.5 | **1,439.5** (159.343 s) | +1.5% |
| `SPEC=none PREFIX_CACHE=off VISION=off` 8K | 2,542.6 | 2,542.5 | **2,605.1** (3.145 s) | +2.5% |
| 32K | 2,344.7 | 2,346.2 | **2,400.8** (13.649 s) | +2.3% |
| 128K | 1,777.4 | 1,778.6 | **1,809.4** (72.441 s) | +1.7% |
| 224K | 1,429.5 | 1,430.2 | **1,450.2** (158.168 s) | +1.4% |

(`raw/{ship,nospec}-pipe-{on,off}/sweep/`: `./inkling2mini.sh bench prefill-sweep`, one fresh server per row group; "pipe off" is the same binary with `REXL3_INKLING2_LOOP_DENSE_PIPE=0`, which runs the shipped kernels; it reproduces the `ce650e3` numbers
of `2026-11-23-prefill-push` within 0.3%, so the A/B and the same-day baseline agree.) `bench.py` of the shipping server: cold 32K 2,374.2 tok/s (`raw`: `-exact/raw/ship-pipe-on/bench.out`).

## Where a 4,096-row chunk goes (`nsys -t cuda`, the same binary, pipe off against pipe on; ms a chunk; `raw/profile-by-kernel-pipe-{off,on}.txt`)

| depth | dense GEMMs (288 launches) | dense rotation | dense total | span of the chunk |
|---|---:|---:|---:|---:|
| 0 | 239.3 -> 208.9 | 49.8 (288 launches) -> 40.5 (172) | 289.1 -> **249.3 (-39.8)** | 1,589.0 -> **1,541.0 (-48.0, -3.0%)** |
| 8K | 239.3 -> 210.6 | 49.4 -> 40.6 | 288.7 -> 251.2 (-37.6) | 1,686.5 -> 1,642.2 (-44.2, -2.6%) |
| 32K | 237.9 -> 207.7 | 49.1 -> 41.0 | 287.0 -> 248.7 (-38.3) | 1,959.4 -> 1,918.8 (-40.7, -2.1%) |
| 128K | 235.8 -> 207.3 | 49.4 -> 40.3 | 285.2 -> 247.6 (-37.6) | 3,092.1 -> 3,058.8 (-33.2, -1.1%) |

The dense stages move by 38-40 ms a chunk; the span by 33-48 (the other stages moved 2-8 ms in either direction: the bf16 linears -3.8 ms, local attention -2.6 ms at depth 0, +4.4 ms at 128K: noise of the order of the
profile's run-to-run difference, not a claim). Rotation launches: 172 = 48 (q, gate, r) + 48 (o) + 76 (k and v: one pass on the 20 visits whose k and v inputs are q's, two on the 28 others). 8K / 32K / 128K tok/s of a chunk
change in proportion (table above). MoePipe's gains land on the same chunk and add; these are this lane's alone.

## Design note (what the lead's arithmetic became, and why)

**Shared inputs.** A visit projects through six EXL3 tensors with six *different* `suh` vectors: q (2,304 x 3,072), gate (2,304 x 3,072), r (2,304 x 384) on the q input; k and v (2,304 x 1,024) on the k and v inputs, which
are the q input on the 20 of 48 visits that have no controllers and no global attention (`distinct` false in `model.rs`), else their own; o (3,072 x 2,304) on the attention output. So a visit is **six rotations** (288 a chunk)
however the inputs are shared; sharing saves reads of x, not rotations. 220 GFLOP a visit: 2.39 ms at the fp16 peak of 91.9 TFLOP/s.

**Where a visit's 5.9 ms went (lab, real layer 11, 4,096 rows, L2 flushed before each launch, `raw/lab-prototype-bench.txt`, `bench-visit-final.txt`).** The rotation of one input takes 0.157-0.170 ms (0.226 for o's 3,072 columns): 37.7 MB
(18.9 MB of bf16 x read, 18.9 MB of fp16 fragments written) at 224 GB/s of the 233.8 GB/s the DRAM gives, i.e. exactly bandwidth-bound; a visit's rotations are 1.04-1.07 ms. The six GEMMs are 4.7 ms at 41-43 TFLOP/s (45-47%
of the peak). Knock-out builds of the shipped GEMM (garbage outputs on purpose, q, 1.239 ms): no mma 0.957, no cross-warp reduction 1.056, every A fragment the same address 1.130, every trellis tile the same tile 1.232, no decode 1.181,
no reduction and no decode 0.947, plus no mma 0.735, plus no L2 reads 0.337, no k loop at all (reduction kept) 0.397, neither 0.101: **the reduction machinery alone costs 0.30 ms of 1.24** (eight sweeps of 64 rows x 64
columns, each through shared memory behind two block barriers per 16-row tile), the k loop and the epilogue's skeleton 0.34, and every other part is worth 0.05-0.28 ms: nothing is bound by one thing, which is what the MoE lab found.
The L2 read ceiling (streaming reads of 4-16 MiB from 96-192 blocks): 1.4-1.9 TB/s.

**The producer-rotates-in-the-GEMM idea does not pay here, by arithmetic and by the lab.** Recomputing the rotated rows in each block's A producer repeats the rotation for every 128-column strip: 24 times for q and gate, 18 for o,
8 for k and v. One rotation is 4.8 M warp-instructions and 1.5 M shuffle-pipe cycles: 14 us of a 48-SM chip. Times 24 that is 0.33 ms of shuffle pipe against a 0.63 ms GEMM at the peak, more than half of its issue budget. Only
a rotated copy that is *materialised once and read by every strip* avoids that, which is the shipped design; its price is the DRAM pass. So the design keeps one rotation pass per input (up to three outputs: x is read once,
the shuffle work is trivial) and spends the effort on the GEMM, where 4.7 of the 5.9 ms are.

**The GEMM (`csrc/inkling2_dense_pipe.cu`, `gemm_kernel`).** 64-row groups x one 128-column strip as before (strip index fastest, so a row group's A fragments are read from L2 by its 24 strips back to back); but **eight warps, each
owning one 16-column tile and the whole of K**. The decode plan's chains (`plan_for`: 4 K splits x 4 warps, 9 or 12 k tiles each, every chain a fresh fp32 mma accumulation from zero) are the same 16 chains, run in order by
one warp; the old cross-warp sum `((c0 + c1) + c2) + c3` of a split happens in registers (`Z = c0; Z += c1; ...`), the splits' sums are added into the strip's fp32 tile in split order, and the epilogue is `finish` unchanged:
per output element the fp32 sequence and every rounding site are the old ones (proof: the tests below). Per stage of the k loop (2 k tiles): a warp waits on the stage's mbarrier, reads its 4 A fragments (LDS.128, 4 x 512 B) from the
shared-memory ring, loads its own trellis tile two steps ahead from global (`load_step`, ring of registers), decodes it (`decode_lane_at`, the old function), issues 8 mma, and after the stage's last k tile arrives on the
stage's `empty` barrier; the refill of a slot (cp.async of 4 x 1 KB, every thread one 16-byte chunk, `cp.async.mbarrier.arrive.noinc` on the stage's `full` barrier) runs two stages ahead and waits only for the readers of the
slot's previous stage. **Budget**: shared memory 49,280 B a block (ring 4 stages x 4 KB = 16 KB, the strip's fp32 tile 32 KB with a float2 XOR swizzle that makes the fragment stores and the epilogue's row reads conflict-free,
128 B of barriers), about 108 registers a thread on the lab builds of this loop (ptxas, no spills; 256 threads), so two blocks an SM (16 warps) by the shared memory and the registers, 1,536 blocks for q and gate. Expected by arithmetic at design time: the GEMM at 65% of the peak would be 3.7 ms a visit; measured 3.90.

## The prototype and the go/no-go (lab, `inkling2_dense_pipe`, 4,096 rows, L2 flushed, medians of 9; q-tensor GEMM ms unless noted)

| Design (engine `b31cf9d`, `raw/lab-prototype-bench.txt`) | q GEMM | note |
|---|---:|---|
| shipped GEMM, replica (bit-identical) | 1.239 | |
| warp-owned columns, A read per warp from global (4x the L1 traffic) | 1.380-1.413 | no cross-warp reduction, but A through L1 four times; with every address the same 0.932 |
| same, A staged by cp.async behind `__syncthreads` per 2 k tiles (`p_kernel`) | 1.117 | without the barrier (garbage) 0.975, without the strip-tile read-modify-write 1.060 |
| a producer warp issuing 1 KB bulk copies (`cp.async.bulk`, mbarrier complete_tx) | 1.388 | ptxas: 96 registers with 140 B of spills in the k loop (the ninth warp makes a block count as ten and caps the registers); the shapes that fit two blocks an SM were no better (1.34-1.94) |
| p_kernel with the swizzled strip tile | 1.069 | |
| **mbarrier-coupled cp.async ring (final)** | **1.024** | one-stage rings, ASTEP 1 / NS 8: 1.295 |
| rotation hosted in the GEMM's tail, or one chain-end slice | | the visit's chained sequence 5.40 ms against 4.57 for the same GEMMs with no hosted work (and 5.20 with standalone rotations): hosting costs more than the pass it replaces (with the loads from one row and no stores 4.86; the cause, L2 and DRAM contention plus a serialised tail, is `[INFERENCE]`) |
| rotations of gate / r / k / v on a second, lower-priority stream | | 5.29 against 5.54 (-4.4%) in one run, 5.36 against 5.36 (0.0%) in the next: not kept |

**Go/no-go** (gate: per-visit rotation + GEMMs at least 15% lower in the prototype): shipped six launches 5.898 ms; pipelined 5.015 ms with k and v on q's input (**-15.0%**), 4.942 ms with three inputs (**-16.2%**) in the final
run on the production code (`bench-visit-final.txt`, under the cap); the run before it, the same code, said -16.7% / -15.5% (5.823 / 4.848 / 4.922). Per tensor alone: q -14.9%, gate -14.4%, r -2.5%, k -3.5% (-12.3% in the
earlier run), v -24.3% (-11.4%), o -9.5%. **GO**, with the margin the 15% rule leaves: right at the line, in noise of 1-2 points. In situ the lane's change is 38-40 ms of the dense stages and 33-48 ms of a chunk (-3.0% at depth 0),
**less than the lead's -90 ms of the dense lane**: the GEMM went from 47% to 61% of the fp16 peak, not to 80%; the rotation stage fell by 9 ms (the x reads of the three-way and two-way shares), not by 49.

## What is left (arithmetic and the lab; `[INFERENCE]` where marked)

A visit after the change: GEMMs 3.90 ms (q 1.02, gate 1.02, r 0.14, k 0.37, v 0.37, o 0.98) and 0.9 ms of rotation. The GEMM's own knock-outs (`raw/lab-prototype-bench.txt`, v56-v66) say what stands between 1.02 ms and the tensor
floor of 0.55-0.63: the mma-only skeleton is 0.68, trellis decode on top of it 0.84 (v63, v64), and the final 1.02 adds 0.18 for the stage coupling, the strip tile's accumulation and the A reads together `[DERIVED]`. The decode (50 integer instructions a 16 x 16 tile a lane, once per
64-row group) is the largest piece left and shrinks only with taller row groups (128 rows needs 128+ accumulator registers a warp with the chain sums: one block an SM and 8 warps, measured slower in the lab). The rotation
is DRAM-bound at 0.16 ms a pass, 0.9 ms a visit, 43 ms a chunk, and the pass cannot be hidden by hosting it in a GEMM or on a side stream with what was measured; writing the rotated fragments from the kernel that produces x (the
controller's blend and norm, not this lane's kernel) would remove the x read and hide the write behind a bandwidth-bound kernel `[INFERENCE]`, about 20 ms a chunk at best.

## Not proven / risks

* Speed is measured for the Mini's six attention shapes at 4,096-row chunks (and the 8,192 shape in the exactness gates); the engine refuses the pipe for any other shape and below 129 rows (the shipped kernels run, tested).
* The pipe needs 2 blocks x 49,280 B of shared memory an SM (the GB10's 100 KB): a device with less runs one block an SM `[INFERENCE]`; `mbarrier` / `cp.async.mbarrier.arrive.noinc` need sm_80 or newer.
* The memory plan prices +30 MiB at 4,096 rows (+60 MiB at 8,192): the workspace holds three rotated inputs (`dense_pipe_rotated_inputs`, exact by arithmetic: 56,623,104 B against 25,165,824). A measured boot of the bootstrap profile
  (`raw/ship-pipe-on/boot-mem-stages.txt`): the four step stages together took 2,303.7 MiB against about 2,297 MiB the plan now prices; the three `engine.rs` pins of the plan's MemAvailable prediction moved by the same 30 MiB
  and were not re-measured on other profiles. The sizing does not depend on the switch.
* The runs before the lead's cap order of 2026-11-23 (lab knock-outs and hosting benches, the first nsys pair, the gates, the first shipping sweep) were made with the node reading 2,197 MHz and the task's statement that it was
  locked, not with a verified `-lgc` of mine. Their *ratios* are used in the design note; every number in the tables above is from a run after `scripts/gpu_cap.sh` / `-lgc 2200,2200` printed its success line
  (`raw/clock-lock.log`; the controller prints `GPU SM clock capped at 2200 MHz`), and the bit-exactness gates do not depend on the clock.

## Commands (node-3 stands for the node's address; `NODE` is exported)

```sh
git -C <engine worktree> rev-parse HEAD                                    # fd6ef616be704ad0863bd5dbebdc97408527d4bb
tools/gates.sh raw/gates-new.log                                           # ops_dev.sh on node-3: every gate of the exactness record
NODE=node-3 LAB_SRC=<engine worktree> TEST_ENV='DP_BENCH=1' scripts/ops_dev.sh run inkling2_dense_pipe bench_visit     # the per-visit bench
tools/nsys_pf.sh pipe-on PF_TOKENS=135168; tools/nsys_pf.sh pipe-off PF_TOKENS=135168 REXL3_INKLING2_LOOP_DENSE_PIPE=0   # then tools/prof_pf.py <sqlite> 1,3,9,33
./inkling2mini.sh build fd6ef616be704ad0863bd5dbebdc97408527d4bb && ./inkling2mini.sh start && tools/live.sh ship ship-pipe-on && CACHE=1 tools/live.sh sweep ship-pipe-on
EXTRA_ENV=REXL3_INKLING2_LOOP_DENSE_PIPE=0 ./inkling2mini.sh start && CACHE=1 tools/live.sh sweep ship-pipe-off      # likewise SPEC=none PREFIX_CACHE=off VISION=off ... tools/live.sh sweep nospec-pipe-{on,off}
```
