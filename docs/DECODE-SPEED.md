# Decode speed: graphs, side streams and the kernels behind them (engine branch `inkling2-loop-graph`)

Plan Phase 7.2, bit-preserving part. Every number is in `bench/records/2026-11-23-graph-baseline` (before), `-graph-decode` (after, with the
per-change table and the profile) and `-graph-exact` (byte-exactness proofs). Serial decode on one Spark: **37.2 to 53.4 tok/s** at short context and
**36.2 to 51.3 tok/s** at 4K (`ROOFLINE.md`: ceiling 70.1 at 4K, proposed 62), outputs byte-identical to the engine as it was.

## What a decode step is

A step (a window of 1 to 8 rows per slot) is **2,830 launches** over the 48 visits (about 59 a visit) plus the unembed and its top-k. The host needed
16.1 ms to enqueue them and the device 25 ms to run them: the device was the long pole, so the graph alone recovers only the 2.4 ms the eager path
lost, and the rest of the gain came from making the device chain shorter (the baseline record has the launch anatomy and the profile).

## Graphs (`families/inkling2_loop/graphs.rs`, `model.rs::run_window`)

- One graph per window shape: `(rows, row blocks of the attention plan, power-of-two bucket of the global attention's chunk bound, candidate count)`.
  The bucket's extra CTAs exit at their first line, so a replay's bits are the eager launch's. Everything else a launch reads (ids, rows, the plan, the
  conv descriptors, the block table) is in a device buffer that `prepare` writes before the replay; the commits of the previous window stay outside the
  graph as eager launches.
- Policy (`window_graph::WindowGraphs`, now generic over its key): the first sighting of a shape runs eagerly, the second captures and replays, every
  later one replays. A non-sticky capture failure runs the shape eagerly for good and is counted (`capture_failures`); a sticky device error is an error
  by name. The cache holds at most 64 graphs (LRU). Counters: `Backend::graph_counters`, the server's `/metrics` (`rexl3_graph_cache_*`).
- Switches (read once at boot): `REXL3_INKLING2_LOOP_GRAPHS=0` eager windows; `REXL3_INKLING2_LOOP_OVERLAP=0` no side streams;
  `REXL3_INKLING2_LOOP_PRIORITY=0` the rank's streams at the default priority instead of the device's greatest (see "Streams");
  `REXL3_INKLING2_LOOP_GRAPH_AT=0` capture at the first sighting; `REXL3_INKLING2_LOOP_TIMING=1` a host timing line every 64 windows. All are on by
  default. For an A/B, boot twice and compare `bench/tools/exactness.py record` files (`diff`) and the speed.

## Overlap inside a window (side streams, captured into the graph)

- MoE bands of a mixed layer run one chain each on their own scratch and lane, then one combine over all picks in slot order
  (`MoeExperts::run_softcap_bands_par`).
- The attention inputs of a layer: the K branch (its controller, `wk`, the dynamic taps) and the V branch (controller, `wv`, taps, the sink) run on two
  lanes beside the q branch, each with its own EXL3 workspace and controller scratch; they join before the prologue.
- Independent chains only: nothing is reordered inside a chain, so no bit moves (`inkling2_loop_graphs` compares sequential eager, overlapped eager and
  replayed windows).

## Kernels

- Split top-k (`gb_inkling2_vocab_topk_split`): 3.1 ms of one block's 64 rounds became two kernels over blocks of 2,048 columns. The order is total
  (value, then the lower index, NaN as -inf), so the selected pairs are the old kernel's.
- Bias kernel (`bias_kernel` in `inkling2_attn.cu`, the Mini's shape only): the relative-bias tables of a row, formerly two CTAs of the prologue (about
  30 us a visit), in a grid of rows x 4 x 4 with the same arithmetic. The side effect: the 32K prefill took 14.5 s instead of 27.3 s.

## Profiling a decode step

`bench/records/2026-11-23-graph-decode/tools/nsys_run.sh NAME` runs 20 replayed windows of the lab test binary under `nsys profile -t cuda
--cuda-graph-trace=node` (node-2; the GPU lock is taken by the command) and prints `prof_an.py`'s kernel table; `prof_tl2.py` prints a timeline. The A/B
loop of a change is `ops_dev.sh test inkling2_loop_graphs decode_step_timing` with `LOOP_PROFILE_STEPS=100` (one row, 3,800 positions, graph and eager).

## What is left (arithmetic and inference in the graph-decode record)

5.2 ms to the 4K ceiling: MoE GEMVs at about 80% of the read bandwidth, a dozen dependent 2-10 us kernels of controller / mix / router glue a visit,
attention at twice its KV bound, projections at 80%. Directions measured as no gain twice and stopped: gate and sink placement, PDL on the dense EXL3
launches.

## Streams, the image encoder and the review fixes (plan item ReviewFixesB; `bench/records/*-review-fixes-b`)

- **Streams.** The rank's window stream is blocking (legacy copies after a `stream.sync()` stay ordered), its lanes, the capture stream and the MoE band
  lanes are non-blocking, and all of them take the device's greatest priority. An image encode on a blocking-pool thread used to stall the engine
  thread: the tower's legacy `cudaMemcpy` synchronised with every blocking stream (one decode window waited 2.86 s of a 3.19 s encode of 4,096 tokens).
  The tower now stages and reads back with stream-ordered copies, `LoopEncoder` runs on a non-blocking stream, and the priority keeps a window beside an
  encode at about 150 ms against 410 ms (the encode shares the device; the numbers are in the record). A pageable async copy parks its caller until the
  copy is done, and a parked thread holds up the other threads' CUDA calls: the tower syncs before it reads its rows back.
- **Fixed in the rank**: `flush` syncs the drafter's staging (a stream that never proposed leaked a few host buffers a window); `set_overlap` forgets the
  graphs (a graph holds the structure it was captured with); commits that cannot be applied are fatal (`StateLost`); the boot refuses an expert band the
  prefill kernel lacks, by layer and widths. **In the kernels**: the two-launch wrappers return the real launch code (a refused capture runs eagerly);
  the partial scratch holds a sliding row's five chunks at any context (a graph baked in the partial mode overran it at `--max-model-len` 1,024);
  `block_sums` of the mix strides; the split prefill kernel sets its smem attribute; `struct Args` has `offsetof` mirrors.
- **Tests that pin something**: slow-side mutants for the K, V and MoE band lane joins (a lane spins before its join; with the join dropped the bits must
  differ), a capture that fails after the fork, an arena bank equal to a `from_groups` bank, Mini drafter contexts of 4,095 to 5,000 positions and a
  million (257 partials), a full 4-request launch, a graph re-staged at other positions, drafts accepted on a chat prompt, the sky dump hashes asserted.
  `REXL3_REQUIRE_GPU=1` makes a missing device or Mini pack a failure (set it in every gate run); the Small's real-weight tests keep skipping without
  the Small's pack.
- **The learnt class prior** (engine `Engine::priors`) fades in rounds that draft nothing, so code after prose is drafted again
  (`bench/tools/specmix.py`, the record).

## Second pass: where the step goes, the windows by rows, the proposal (lane `k-fuse`, engine branch `inkling2-loop-fuse`, records `fuse-baseline`, `fuse-decode`, `fuse-exact`)

Engine `1d53344` (the integration head) measured on node-2, clock locked at 2,200 MHz. Tools: `bench/records/2026-11-23-fuse-baseline/tools/` (`nsys_run.sh NAME ROWS`
profiles 20 replayed windows of `decode_step_timing` with `LOOP_PROFILE_ROWS`, `prof_visit.py` the per-kernel and timeline tables, `prof_stages.py` the stage sums,
`nsys_propose.sh` the drafter's proposal).

### What one serial step is made of (1 row, 3,800 positions, graph replay: 19,146 us in the test, 20,588 us span in the profile)

Kernel durations of PDL launches include their wait for the predecessor (a kernel launched early shows the predecessor's time too): read the spans of the timeline
(`profile/timeline-1row-one-window.txt`), not the shares. Launches a step, median microseconds, summed microseconds a step over all streams (side lanes overlap, so the
sum, 36.8 ms, exceeds the 20.6 ms span):

| kernel (launch shape) | what | launches | median us | sum us | achieved against its floor |
|---|---|---:|---:|---:|---|
| `moe_gemv_kernel` gate/up (9x6xN picks) and down (18x1xN), 4-bit and 5-bit bands | routed experts | 220 | 43-66 | 11,700 | bytes depend on the route: not derivable per launch (see the window table) |
| `down_combine_kernel`, `gateup_epilogue_kernel`, `combine_f32`, `group`, `own_remap` | MoE tail and bookkeeping | 546 | 1-33 | 6,800 | PDL waits included |
| `linear_kernel<12,2,4>` 24x4 (`wq`, `gate`) | dense EXL3, 5.31 MB | 96 | 25.5 | 2,564 | **208 GB/s** (237 ceiling) |
| `linear_kernel<12,2,4>` 8x4 (`wk`, `wv`) | 1.77 MB | 96 | 18.7 | 1,861 | **95 GB/s**: 32 CTAs, latency bound |
| `linear_kernel<12,2,4>` 18x4 (`wo`) | 5.31 MB | 48 | 25.7 | 1,251 | **207 GB/s** |
| `linear_kernel<12,2,4>` 3x4 (`wr`) | 0.66 MB | 48 | 12.0 | 608 | 55 GB/s: latency bound |
| `linear_kernel<12,2,8>` 1571 blocks (unembed) | 347.5 MB | 1 | 1,484 | 1,493 | **234 GB/s** |
| `rot_in`, `rot_out` of the EXL3 chain (3 launches a linear) | Hadamard rotations | 722 | 1.5-7 | 1,590 | |
| `attn_kernel` + `merge_kernel` + `prologue` + `bias` + `ring_write` | attention (sliding 40, global 8) | 328 | 6-61 | 3,754 | sliding visit 34-38 us for a 4.2 MB ring (18 us floor); global 61 us |
| controller `pair`, `down` (8 blocks), gain/forget (288 blocks), `blend` | loop controllers (24 core visits x 4) | 480 | 2.8-8.4 | 2,300 | |
| `mix`, `mix_raw`, `conv`, router logits, `select` | glue | 480 | 4-12 | 2,400 | |
| split top-k (2) + commits + embed | tail | | 53 + 83 | 174 | |

A core visit on the timeline: 388 us. Controller chain 34 us; projections 79 us (floor 62: the five linears' 14.8 MB); attention 38 (floor 18); `wo` 30 (22); conv, mix,
MLP controller, router, select, remap, group 36 us of twelve dependent 2-8 us kernels; MoE 167 (floor 142 for 33.8 MB); about 10 us of the MoE tail is the dependent hops
after the last weight byte. At the DRAM ceiling the step would be 14.27 ms (ROOFLINE): the engine is at 73% of it.

### Windows by rows (the engine as it was; the window path is not changed by the fuse lane)

`decode_step_timing`, one stream, 3,800 positions of the synthetic list, graph replay, median of 100 (`raw/rows-step-timing.txt`): **1 / 2 / 3 / 4 / 6 / 8 rows = 19.12 /
22.53 / 24.83 / 26.72 / 30.29 / 32.99 ms**. Stage sums from `nsys` (microseconds a step summed over streams, `profile/kernels-rows*.txt`):

| rows | span ms | MoE (gate/up, down, epilogues, combine, group) | dense EXL3 | glue | attention | unembed | tail |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.59 | 18,446 | 8,223 | 4,720 | 3,754 | 1,493 | 174 |
| 2 | 23.54 | 20,389 | 8,021 | 4,669 | 3,715 | 1,486 | 223 |
| 4 | 27.73 | 23,576 | 8,102 | 4,726 | 3,893 | 1,495 | 285 |
| 8 | 33.81 | 28,605 | 8,498 | 4,960 | 4,110 | 1,526 | 422 |

From 1 to 8 rows the span grows 13.2 ms and the MoE sum 10.2 ms; dense, glue, attention and unembed together grow 1.2 ms (attention +0.36, tail +0.25): **the window's
growth is the experts**.

With **real texts** (`LOOP_REAL_IDS_FILE`: the prompts and the pinned replies of `prose`, `code`, `structured`, `json` as slots 0 to 3, each slot its own text, consecutive
tokens a window; final engine, windows identical to `1d53344`; `raw` of `fuse-decode`) the windows cost a good deal more than with the synthetic cycled list, because rows and
streams of the synthetic list share many more experts than real routes do. Rank times, `round_cost_table`, median of 16:

| streams x rows | rows | 4K real ms | 4K synthetic ms | 32K real ms | window GB (c4_model) | GB/s whole window | expert GB/s (non-expert at 233.8) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 x 1 | 1 | 19.28 | 19.13 | 21.48 | 3.28 | 170 | 134 |
| 1 x 4 | 4 | 31.99 | 24.77 | 34.06 | 5.61 | 175 | 159 |
| 1 x 8 | 8 | 41.93 | 29.51 | 46.76 | 7.50 | 179 | 168 |
| 2 x 1 | 2 | 26.74 | 23.22 | 30.89 | 4.88 | 183 | 163 |
| 2 x 4 | 8 | 50.86 | 30.32 | 53.01 | 9.12 | 179 | 170 |
| 2 x 8 | 16 | 68.39 | 37.64 | 71.70 | 12.35 | 181 | 174 |
| 4 x 1 | 4 | 39.65 | 29.02 | 47.26 | 7.89 | 199 | 190 |
| 4 x 4 | 16 | 73.37 | 41.17 | 79.59 | 14.95 | 204 | 200 |
| 4 x 8 | 32 | 96.15 | 52.76 | 102.65 | 19.51 | 203 | 200 |

The window bytes and the achieved bandwidths are **INFERENCE**: `bench/tools/c4_model.py window_bytes` (the route statistics of other texts, kinds chat / code / reasoning / json
for slots 0 to 3) divided into the measured ms; the distinct experts of this run were not logged. They say the windows run at 170 to 204 GB/s over the whole window
(73 to 87% of 233.8), the expert stage at about 200 GB/s once there are 16 rows or more. The 4x1, 4x4, 4x8 real windows are within 3%, 6% and 8% of `c4_model.py`'s 38.7 / 69 / 89 ms.
The lab table with synthetic ids understates the windows by 1.4 to 1.8 times: anything priced from it (the engine's `WINDOW_MS` grid) is low.

### Proposal (the drafter, one request, 4,096 positions)

Baseline timeline (`profile/timeline-proposal-baseline-one-request.txt`), 8.45 ms: layer 0 (433 us), **layer 1, the global layer, 1,238 us of attention** (`attn_part_kernel`: 16 tiles
of 256 keys one after the other in one CTA, 77 us a tile, a dependent V load a key), layer 2 (330 us of attention), the unembed 1,503 us (234 GB/s), then seven Markov steps of
**about 560 us**: `markov_logits_kernel` 446 us (the 103 MB folded table, 231 GB/s: at its floor), the single-block `local_kernel` 90 us plus `merge_kernel` 75 us, and 30 us of hops.

| step | proposal ms, 4K, 1 / 2 / 4 requests (synthetic ids) | change |
|---|---|---|
| `1d53344` | 8.450 / 8.935 / 9.975 | |
| draft pick on the split top-k (`1a84469`) | 7.916 / 8.401 / 9.477 | -0.55 ms: 165 us a step become 22 |
| attention partial with a staged P V (`5a1c04b`) | 6.831 / 7.313 / 8.395 | -1.09 ms: global layer 1,238 -> 600 us, local layers 330 -> 143 |
| the proposal in a graph per request count (`5f39463`) | 6.632 / 7.104 / 8.159 | -0.20 ms |
| final engine (`24a04af`, a later run) | 6.690 / 7.161 / 8.208 | **-1.76 ms (-21%)** at one request; 32K: 7.33 / 8.44 / 11.37 |

Real texts: 6.90 / 7.54 / 8.93 ms at 4K and 7.52 / 8.81 / 12.02 at 32K (final engine). What is left of 6.7 ms at one request: seven Markov steps 3.4 ms (the table read is at its
ceiling), the unembed 1.5, attention 0.9 (the global layer still takes 37 us a tile where the arithmetic wants about 10: not understood without a kernel profiler), layers 0.7.

### Changes that gained, and directions stopped (step time = median of 100, graph replay, 3,800 positions, one stream; proposals as above)

| # | change | 1 row | 8 rows | verdict |
|---|---|---:|---:|---|
| 0 | baseline `1d53344` | 19,146 us | 32,985 us | |
| 1 | L2 warm-up lane: a side stream loads the next visit's `wq wk wv wr gate` trellises into L2 while the DRAM idles between the MoE and the projections | 19,011 (warm off 19,132) | 33,414 (off 32,911) | **non-gain 1**: the projections drop from 79 to 35 us but the controller chain behind the prefetch slows from 32 to 77 us (its small weight reads queue behind 14.8 MB of traffic) |
| 2 | the same with the three controllers' matrices first, 96 / 32 / 16 blocks | 19,119 / 19,529 / 23,191 | 33,214 | **non-gain 2**, reverted |
| 3 | controllers' gain and forget in one launch (`lin_narrow_pair_kernel`, blockIdx.z picks the matrix) | 19,078 / 19,139 | 32,939 | **non-gain** (within the 40 us noise: the K/V lanes hide the saved launch, the MLP one saves 2 us); reverted |
| 4 | `ldcg` look-ahead of 6 k steps inside `moe_gemv_kernel` | 20,616 | 36,992 | **worse** (+1.5 ms, +4 ms); reverted. `prefetch.global.L2` was measured as a no-op on this device by the Small's lane |

Stopped: L2 prefetch of weights (three variants, one lane, one in-kernel), controller launch fusion (one variant). Not tried: the router chain (select, remap, group, rot_in: 11 us a visit,
about 0.3 ms), the MoE kernel's own efficiency (gate/up and down at roughly 80 to 90% of the ceiling), a faster global attention in the drafter.

**The serial step is bandwidth and dependency bound.** The DRAM floor is 14.3 ms, the step 19.1; what is left is five to six us of dependent hops a kernel pair, forty-odd pairs a
visit, and kernels that cannot fill the bus on their own (`wk`/`wv` at 95 GB/s, `wr` at 55); the measured remedies that move bytes earlier (L2 warm-up) trade one wait for another, and the
fusions that remove a launch gain less than the noise. Serial decode is unchanged by this lane: 53.51 / 51.56 tok/s (short / 4K), c = 4 100.83.

## The drafted round: host side, chain and budget (plan item RoundOverhead; `bench/records/*-round-baseline`, `-round-overhead`, `-round-exact`)

`decode_mean4_tok_s` (c = 1 mean of `bench decode`'s prose, code, structured, JSON; drafter on) went from 81.63 (`1d53344`) to **84.01** (`5cc464e`); the 100 target is not met. Where a drafted round goes
(`bench/tools/roundtrace.py`, `REXL3_ROUND_TRACE=1`: one `event=round_trace` line a round with host-clock offsets of each phase; `bench/tools/nsys_rounds.py` on an nsys export of the server,
started with `SERVE_WRAP='nsys profile -t cuda,nvtx --cuda-graph-trace=node --delay 80 --duration 13 --kill none -f true -o NAME' ./inkling2mini.sh start`):

| phase of a c = 1 code round (`1d53344`) | ms |
|---|---:|
| plan, acceptance, sampling, accounting | < 0.05 |
| drafter feed and conv commit launches | 0.07 |
| proposal: stage, launch, wait, read | 7.4 |
| the window's flush, uploads, launch | 0.35 |
| **wait for the window** (8 rows of real text) | 39.8 |
| candidates read, server loop, streaming, snapshot | 0.04 |
| **round** | 47.3 |

- **The host is under 0.5 ms of a round.** The 7 ms outside the rank's timers were never host time: a window of real text costs 41 ms at 8 rows (synthetic ids: 29.5), 18.7 at one. Windows by
  streams x total rows, served: 1x1 18.7, 1x3 25.5, 1x5 31.8, 1x8 42.4, 2x8 46.9, 2x14 58.5, 3x21 75.3, 4x4 (serial) 37.7, 4x24 84.7, 4x32 94.3. Each extra row at one stream is 3.2 to 3.6 ms (experts), and the window's MoE
  GEMVs go from 12 to 36 kernel-ms between 1 and 8 rows. `bench/tools/roundcost.py` measures the table (server with `REXL3_ROUND_TRACE=1 REXL3_INKLING2_LOOP_CHAIN=0`).
- **Chained round** (`Backend::chained_round`, `Inkling2LoopRank::chain_round`): with the `Plan` budget the draft counts are fixed before the draft, so the proposal (the drafter graph of its request count), a
  device-to-device copy of its tokens into the window's draft rows, and the window graph are enqueued on one stream, and the host waits once. Same launches in the same stream order as
  propose-then-window, same bytes; the gap it removes (the proposal's read-back, plan, uploads and `cudaGraphLaunch` of 2,887 nodes between the two chains) is 0.4 to 0.7 ms a round (1 %). `REXL3_INKLING2_LOOP_CHAIN=0` runs the
  two-step path for an A/B. The pageable uploads of the window did not park the thread behind the running proposal (`prepare` 0.01 ms with the proposal running).
- **Budget prices** (`engine.rs::round_cost`): the served window table above and the proposal at 6.4 / 7.0 / 7.7 / 8.3 ms for 1 to 4 requests behind the drafter graph. `Engine::with_idle_relax(0.99)` (Mini only,
  `REXL3_INKLING2_LOOP_SPEC_RELAX`): a stream the plan stopped drafting retries after about 70 rounds, not 10. Prose: 41.8 -> 49.9 tok/s in `bench decode` (serial 52.3), 1.03x in specbench; the c = 4 aggregate
  101.4 -> 109.0. Code and json give back 2 to 3 % to the one-stream line's last draft.
- What is left is device time: the window (19 ms for one row, 41 for eight) and the proposal (6.4 to 8.3 ms). Prose, served serially, was thought to idle 1.5 ms a round between windows; the round trace of the serial path says 0.05 ms (record `serial-chain`: a built device-side greedy chain gained nothing and was removed).

### The copy hybrid (CopyHybrid lane)

Per round and stream the plan chooses between the block drafter's draft and a copy of the stream's own text (longest earlier match of 16 tokens or more, `gb_spec::CopyDrafter`): a copy draft is made on the host and costs the draft round nothing (`gb_sched::plan_round_free`), so a round in which every stream copies skips
the proposal. Acceptance of the copy is learnt by sampling mode, match-length bucket and per stream (including, free, from the copy drafts the block's rounds test: row j's sample tests the copy's token j up to the block's accepted run). On by default for the Mini (`REXL3_INKLING2_LOOP_SPEC_COPY=off`,
`..._COPY_MATCH` for the minimum); `/metrics` has `rexl3_spec_copy_*`. Mean4 +1.49 (83.92 -> 85.41), byte-exact (records `copy-hybrid`, `copy-hybrid-exact`). The serial-window chain was built and measured and gained nothing (`serial-chain`).

## Draft slim: the proposal's seven Markov steps without the 103 MB table (lane `k-slim`, engine branch `inkling2-loop-slim`, records `draft-slim`, `draft-slim-exact`)

### Design note (written before the build; the go decision)

**What a Markov step reads and computes, per request** (4,096 positions, one request; `fuse-baseline/profile/timeline-proposal-final-one-request.txt`, 6.65 ms: three draft layers 1.74 ms, the
unembed 1.51 ms, then seven steps of about 485 us). Step `i` takes the 3-launch EXL3 `rnn_proj` over `[state | markov_w1(prev)]` (2 MB read, 11 us), the gate kernel (9 us: `state`, `latent`),
then `markov_logits_kernel` (446-462 us): for **every one of the 200,058 rows** `v` the fp32 dot of `latent` (256) with `fold_w[v]` (the whole 102.9 MB table: 231 GB/s, its floor),
`bias = bf16(dot + fold_b[v])`, `x = bf16(base_i[v] + bias)`, `logit = bf16(30 tanh(x / 30))`, written as f32 (0.8 MB), then the split top-1 (22 + 4 us) and a copy. `latent` is the step's own output of
the gate, so **nothing of the table read can be hoisted or shared between steps** (each step's `latent` needs the previous step's argmax through `markov_w1(prev)`): idea (b), "read the table once", is
impossible for this recurrence, and with seven steps the bytes are seven reads. Only the argmax of a step is used: that is what is avoidable.

**Ideas, in the order tested, with their data** (numpy on the dump of `fold_w`, `fold_b`, the base rows and the latents of 77 real-feature positions x 7 steps = 539 steps, `draft-slim/tools/a1.py` .. `a4.py`;
the engine's own lab test `slim_picks_the_full_heads_winner` then checks 1,197 anchors x 7 steps = 8,379 steps in the kernels themselves):

| idea | evidence | verdict |
|---|---|---|
| (a) a candidate set with a **proof** (Cauchy-Schwarz `|latent . fold_w[v]| <= ||fold_w[v]||_2 ||latent||_2`, or the L1 bound with `|latent| <= 1`) | the rows' L2 norms are 1.18 (median), their L1 norms 15.1, `|dot|` reaches 9.5 on the maximum; **199,737 of 200,058 rows survive** (median, step 1; 200,050+ at later steps); even against the true maximum as the threshold, 242 to 15,896 rows (medians) | **No.** The worst-case bound is about 16x looser than a 256-wide dot of a latent that is not adversarial against `fold_w`. A bound from a low-rank proxy plus the exact residual norm (rank 96) still leaves 1,652 rows (median) and 61,395 (max). No rule with a proof is cheap |
| (a') a **probabilistic** candidate set: score every row with a cheap proxy of the dot, evaluate the best few exactly | the maximum is isolated: rows within 1.0 of it (in `x`) are 1 (median), 6 (p99); the rank-16 principal-axes proxy has a median error 0.36 (p99 1.6) and puts the true winner at proxy rank 0 in 90% of steps and at most 41 (of 539 steps; rank 54 at rank 8) | **Yes.** The proxy is 6.4 MB (`fold_w` projected on its 16 principal axes, bf16), which the L2 holds from step to step: 7 x 0.45 ms of table reads become 7 scans of an L2-resident table plus about 130 exact rows |
| (c) L2 residency of the hot rows of the full table | 24 MiB against 103 MB; no hot set (the winner is a different row every step) | subsumed by (a'): what is L2-resident is the proxy |
| (d) fused argmax; the global-layer attention | the pick replaces `markov_logits` + split top-1 + copy by two launches; the global layer's `attn_part_mini_kernel` walks 16 tiles in one CTA (24 CTAs on 48 SMs, 598 us at 4K; a local layer 132 us in one CTA a head) | **Yes**: a finer partition of the context (a different summation order: the block rows `hd` move in their last bits, drafts may differ, replies not) |
| (e) the unembed-sized read (348 MB, 1.51 ms of 1.49 at its floor) | `base[v] = hd . U[v]` over all rows is what the logits are made of; a proxy of `U` plus an exact row evaluation of the EXL3 trellis for the candidates would remove it but needs a **bit-exact per-row EXL3 kernel** (the rounding of the fp16-rotated input, the accumulation order of the split-K launch) | **Not built.** It is the next lever (about -1.2 ms), and a kernel project of its own |

Arithmetic of the go decision: at one request the seven `markov_logits` + top-1 steps cost 3.3 ms of the 6.65; a scan (about 9 us) and a pick (about 35 us) replace 485 us minus the 40 us of EXL3 hops and gate:
**about -3.0 ms by arithmetic** (bar: 1.5 ms), plus -0.45 ms from the attention partition. Both were then built and measured.

**What is exact and what is not.** The pick evaluates a candidate with the full kernel's arithmetic (same fmaf chain, same shfl_xor tree, same epilogue: its logit is the full kernel's bits) and keeps the best value
with the lower index on a tie, so *given the candidate set* the winner is the full argmax. The set is the rows scoring at least the 128th largest of the blocks' two best proxy scores (always at least 128 rows,
a few hundred on real steps); the full argmax is outside it only when the true winner's proxy rank is past the threshold. **Measured: 1 step of 8,379** (0.012%; `slim_picks_the_full_heads_winner`), the draft of
1 anchor of 1,197 differs from the full head's (2 tokens of 8,379). The drafter's finer attention partition moves `hd` by bf16 rounding: the oracle battery (`inkling2_dspark mini_matches_the_oracle`, tolerances
against the float64 reference) holds and its golden hashes are regenerated. Replies are unchanged by construction (every draft is verified against the target's argmax): the exactness gates are in `draft-slim-exact`.
`REXL3_INKLING2_LOOP_DRAFT_SLIM=0` selects the full read (the proxy is then not built; the attention partition has no switch).

### Result (engine `aa2f9ad` on the integration head `be0ba1e`; records `draft-slim`, `draft-slim-exact`; node-1, GPU capped at 2,200 MHz, medians of 3, fresh server)

| | baseline `4f939a6` | DraftSlim off (`55f4c1d`, the switch) | lane |
|---|---:|---:|---:|
| proposal at 4K, 1 / 2 / 4 requests (lab, ms; the lab build's switch off, which keeps the finer attention) | | 6.50 / 7.30 / 8.99 | **3.31 / 3.78 / 4.78** |
| proposal at 32K, 1 / 2 / 4 requests (lab, ms; same) | | 7.28 / 8.71 / 11.79 | **4.07 / 5.25 / 7.76** |
| `decode_mean4_tok_s` | 83.99 | 84.02 | **90.08** (+7.2 %) |
| prose / code / structured / json tok/s | 49.7 / 119.2 / 79.2 / 87.8 | 49.7 / 118.7 / 79.7 / 88.0 | 51.7 / 128.8 / 86.3 / 93.3 |
| `decode_c4_rolling_tok_s` | 134.39 | 135.39 | **141.80** (+5.5 %) |
| specbench c=4 aggregate (drafted, tok/s) | 115.4 | 115.6 | 121.2 |

Where the 3.2 ms went: seven Markov steps from 485 us to about 55 us (scan 9.5 us, pick 16 us, the EXL3 hops and the gate 30 us), plus 0.45 ms from the attention partition (the global layer 598 -> 165 us,
a local layer 132 -> 44 us). What the proposal is made of now (4K, one request, 3.34 ms span): three draft layers 1.3 ms, the unembed 1.5 ms (at its floor), seven steps 0.4 ms, the rest hops.
**Next lever, not built:** the unembed read (1.5 ms of 3.3) needs a bit-exact per-row EXL3 evaluation for the candidates to be replaced by a proxy scan; the layers' gate and up projections (two EXL3
launches of 54 us on 72 CTAs each, concurrent-able) are worth about 0.13 ms.
Each ms off the proposal was worth about +1.9 on `decode_mean4_tok_s` here (3.2 ms for +6.1; the c=1 plan's +0.7 per ms was low [INFERENCE: not decomposed]). The tokens a round per kind did not drop (code 5.28 -> 5.33, structured 3.2, json 4.0 -> 3.97; prose's counter moves between runs, 1.17 to 1.27 in the baseline's own runs, 1.11 here).
The engine's budget prices of the proposal (`round_cost`, 6.4 to 8.3 ms) were not changed and are stale.
