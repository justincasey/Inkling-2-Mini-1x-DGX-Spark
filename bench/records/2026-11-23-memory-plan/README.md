# memory-plan: the Mini's plan priced from the boot's measured memory (gate `memory-plan`)

Result **pass** (`meta.json`). Node-1, GPU free, vision off and `--spec none` unless a row says otherwise, window and slots by profile
(`bootstrap` 4 x 262,144, `long` 2 x 1,048,576, `max` 4 x 1,048,576), the guard on at every boot. Engines: `91c05af` and `04d4d60` (the old plan
with `event=mem_stage` instrumentation only) for "before", `abc681365c70e40050b3d10345f51a3e284d202a` (the new plan, clean) for "after".
`runs/<label>/` holds each boot's `serve-and-mem.txt` (the whole server log, `mem_stage` lines included), `status-*.txt`, `memwatch.txt`.
Node-1 carried LoopRank's CPU-heavy oracle job during these boots (other processes' memory is in the baseline each plan read at `boot_start`).

## Plan against measurement

"Predicted" is what the plan says MemAvailable will be at ready; "measured" is what the controller read once the server was ready; the
old plan predicted nothing, so its column is the memory it left unspent: floor + (5 GiB reserve - the 1.125 GiB of buffers it priced) + spare - cache.

| Boot | Old plan, predicted MiB | measured MiB | error | New plan, predicted MiB | measured MiB | error | guard minimum after the first requests MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| bootstrap, cache off | 74,276 | 65,072 | **+9,204** | 65,905 | 66,247 | -342 | 66,097 |
| long, cache off | 57,073 | 47,309 | **+9,764** | 48,568 | 48,406 | +162 | 48,339 |
| max, cache off | 24,735 | 15,997 | **+8,738** | 15,196 | 15,722 | -526 | 15,612 |
| max, `--prefix-cache auto` | 13.9 GiB (from the lead's numbers: prefix 22.28, spare 24.28) | 4.0 GiB (guard min 3.41 GiB, process 113,201 MiB: ProfileLane's measurement, not repeated here) | +9.9 GiB | 11,735 | 12,243 | -508 | 11,928 (process 94,872 MiB; the cache is 4.38 GiB) |
| bootstrap, vision on, cache on | n/a | n/a | | 57,324 | 57,923 | -599 | (see `runs/after-bootstrap-vision-cache`) |

The new plan is within 0.6 GiB at all five boots (the ask: about 1 GiB). Run to run the same configuration moves MemAvailable by a few hundred
MiB (bootstrap cache-off read 65,072 before and 66,247 after; the new plan's host test pins the older readings and holds to 1 GiB).
`--prefix-cache auto` at `max` now leaves 11.9 GiB at its guard minimum, above the floor (8 GiB) plus the 2 GiB variable part; before it left 3.41 GiB.

## Where the memory went (`mem_stage`, bootstrap, MiB; cache off, vision off)

| Stage | Took | The plan's figure |
|---|---:|---|
| weights (`weights_loaded`) | **20,745** | 13,879 counted by the loader; the rest is the **allocation padding**: 6,366 MiB of 2 MiB rounding of the 9,216 expert trellises (each is its own `cudaMalloc`: a 1.27 MiB 4-bit tensor takes 2 MiB) and 500 MiB of other rounding. Per layer of 384 expert tensors the stage shows about 810 MiB against 514 MiB of trellis bytes |
| `visit_state_kv_pool_rings` | 17,569 | KV pool 16,528 + rings and state 659 requested, 382 of slack (at `max` the same stage took 66,148 MiB, 75 above its requested bytes: the slack is not a constant, see below) |
| `lane_scratch`, `step_buffers`, `attention_and_controller_scratch`, `moe_scratch_...` | 172 + 593 + 744 + 758 = 2,267 | 2,219 priced item by item (the loop's destination and message buffers, the attention's partials, the controllers', the MoE's decode scratch x 3 bands, ...) + 48 for the EXL3 workspaces |
| attention + controller scratch at a 1,048,576 window | 1,937 (long) | grows with the window: `Scratch::partials_for(window rows, max_context)` |
| vision tower | 1,120 to 1,202 | 810 priced (weights 0.226 GiB + scratch 568.5 MiB): about 0.4 GiB short, covered by the variable part |
| first graph capture | +275, +760, -393, -7 (varied) | held by the 2 GiB variable part (`ACTIVATION_RESERVE`, now no blanket) |

The first boot's "about 9 GiB beyond weights + KV + state" is therefore 6.7 GiB of allocation padding in the weights and about 2.3 GiB of step buffers,
which the old plan covered with a 5 GiB blanket it also spent on nothing it listed.

## What changed in the plan

`memory.rs` prices the padding (`weight_allocation_padding`), the loop's, the attention's, the controllers' and the MoE's buffers by the rank's
own size functions (so they follow the window) and the measured small items; `ACTIVATION_RESERVE` is 2 GiB, the variable part (largest measured
swing from ready to after the first requests: 1.55 GiB at `max`, once; it did not recur in the second boot), and the planner's reserve is the priced buffers plus it.
`engine.rs` prints `weights_alloc_padding_gib`, `step_buffers_gib`, `fixed_overhead_gib` (padding + buffers + a window's records: 9.29 GiB
bootstrap, 10.44 GiB at a 1M window) and `predicted_ready_available_gib` in the capacity line, and refuses a plan that would be ready under the
floor plus the variable part, by name.

## Findings for the lead

* **6.2 GiB of the box is padding the loader could give back**: `MoeExperts::from_groups` uploads each expert tensor as its own `DevBuf`;
  `gb_cuda::arena::Arena` (and `MoeExperts::from_region`) carve many tensors from a few large blocks. Using one for the Mini's experts would
  return the padding (about 6.2 GiB; speed not measured) and is a loader change in shared kernel code, not made here: the Small's loader and its byte-exact tests are in that code.
* The `visit_state` slack is 382 MiB at bootstrap and about 0 at `max` (the stage took 66,148 MiB for 66,073 MiB of KV + rings requested...);
  the plan keeps the bootstrap figure, which is why `max` reads 0.5 GiB more than predicted. Unexplained, not chased.
* Node-1's swap grew from 311 to 2,022 MiB over these boots (the oracle job on the same node); the guard never acted. Boots at `max` leave
  little memory by design.

## Not proven

* One node, one pack, one measurement per cell; the 2 MiB granularity is inferred from the per-layer growth, not read from the driver.
* The variable part (2 GiB) is the largest swing seen in a handful of requests, not under sustained load, concurrency, graphs for many shapes or long prompts.
* `--prefix-cache auto` at `max` was measured once with the new plan (4.38 GiB cache); the cache's behaviour under eviction is not part of this gate.
* Vision at `long` and `max` was not booted; the tower's 0.4 GiB under-pricing is inside the margin at bootstrap only.
