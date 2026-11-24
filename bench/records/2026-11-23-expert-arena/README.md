# expert-arena: the Mini's experts carved from an arena, no byte moved (gate `expert-arena`)

Result **pass** (`meta.json`). Node-1, TP1, GPU clock locked at 2,200 MHz. Engine commits (branch `inkling2-loop-backend`): `7746bee` (the arena),
`0a6b7e2` (the plan priced at the arena's one unused block), `f8bcdea` (test constants only). The servers booted are `0a6b7e26e6c4`; the GPU proofs
ran from the committed tree `f8bcdea` (clean). The record's engine `commit` is the installed server's.

## What changed

`MoeExperts::from_groups` is now `from_groups_in(.., placer)`: the Small calls it with a placer that does `DevBuf::from_slice` (one `cudaMalloc` each,
exactly as before), the Mini through `MoeExperts::from_groups_arena` with a placer that carves each buffer from a `gb_cuda::arena::Arena` (64 MiB blocks;
every carve is 256-byte aligned, the alignment `cudaMalloc` guarantees, and the kernels are untouched). Only `inkling2_loop/device.rs` calls the arena
variant; `Arena` itself is unchanged. Why: the driver rounds each allocation to 2 MiB, and the 9,216 expert trellises (1.27 and 1.58 MiB) held 6.4 GiB
of it (`memory-plan` record). The dense groups and the small tensors were left alone: with the experts in the arena the weights stage took 13,699 to
14,313 MiB of the device against 13,879 MiB loaded (the spread is the box's other tenants), so no padding remains to hand back.

## Byte for byte (raw files under `raw/`)

| Proof | Result |
|---|---|
| `forward_oracle_gate` (prompt "sky", 26 tokens, 48 visits) dump, twice | 90,056,120 bytes, sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0` both times: the hash of `graph-exact` and `forward-oracle`; top-1 200001, logit 26.821056 (`raw/oracle-gate.txt`) |
| chunk widths 7 and 11, logits | 20,806,032 bytes, sha256 `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a`, the pinned one |
| `inkling2_loop_graphs graph_replay_equals_eager` (37 windows, graph == eager == overlapped), `capture_refusal` | pass; logits sha256 `44294f72...90e1` over 37 windows, 12 captures, 213 replays, 0 failures. **The ids are mine** (`raw/ids_long.json`, 1,500 ids, sha256 `293794b8...2970`, the pinned file is not on this node), so this hash is not comparable to `graph-exact`'s `6d0a555b...`; the proof is equality within the run |
| the MoE batteries: `inkling2_moe` (golden hashes of the Small, partial, TP4 banks, prefill == decode, tight grids, graph), `inkling2_moe_mixed` (mul1 bands, band split, real Mini layers against the oracle: cosine 0.9999980, layer 12 bands 107/17/4, graph), `inkling2_ops golden` | all pass (`raw/gates-graphs-and-moe.log`) |
| live: `exactness.py record` (4 prompts of 16, 521, 2,030 and 8,334 tokens, 128 greedy ids and the prompt-logprobs top-5, solo) diffed against `graph-exact`'s pinned `graph-9fa93ea/exact.json` | `identical: 4 replies`; the `replies` canonical-JSON sha256 is `d0cf6b3c5b83d5b733d5fc25387600124bb348bbd6f1daa352846b47d10c3bc2` for both (my canonicalisation: `json.dumps(replies, sort_keys=True)`) |
| `exactness.py concurrent --max-tokens 64` | `"pass": true` |
| host: `families::inkling2` | 119 passed (the Small's 31 unchanged) |

The Small's loader was not changed (its `from_groups` call is the plain placer), and its goldens `golden_inkling2_moe_hashes` and `golden_inkling2_ops_hashes` pass.
The first gates attempt (`raw/gates-first-attempt-no-ids.log`) shows `graph_replay_equals` "not run": it needs `LOOP_LONG_IDS_FILE`; the second run supplies it.

## Memory before and after (`runs/`, `arena-table.json`)

Before = the `memory-plan` record's boots (`abc6813`); after = `0a6b7e2`. Vision off and cache off unless the row says so. MiB.

| Boot | weights stage | process (nvidia-smi) | predicted at ready | measured at ready | error | guard minimum | cache |
|---|---:|---:|---:|---:|---:|---:|---|
| bootstrap, before | 20,133 | 39,680 | 65,905 | 66,247 | -342 | 66,097 | off |
| bootstrap, after | 14,114 | 33,588 | 84,992 | 84,577 | +415 | 84,482 | off |
| long, before | 20,700 | 57,031 | 48,568 | 48,406 | +162 | 48,339 | off |
| long, after | 13,837 | 50,927 | 67,656 | 67,549 | +107 | 67,416 | off |
| max, cache off, before | 21,269 | 90,386 | 15,196 | 15,722 | -526 | 15,612 | off |
| max, cache off, after | 14,313 | 84,294 | 34,294 | 33,545 | +749 | 33,422 | off |
| max, cache auto, before | 21,088 | 94,872 | 11,735 | 12,243 | -508 | 11,928 | 4.38 GiB |
| max, cache auto, after | 14,116 | 106,295 | 12,288 | 11,797 | +491 | 11,605 | **21.49 GiB** |
| bootstrap, vision on, cache on, before | 20,809 | 49,296 | 57,324 | 57,923 | -599 | 57,867 | 8.59 GiB |
| bootstrap, vision on, cache on, after | 14,058 | 43,206 | 75,387 | 75,380 | +7 | 75,259 | 8.59 GiB |

* The process's device memory falls by 6,092, 6,104, 6,092 and 6,090 MiB (about 6.0 GiB) at bootstrap, long, max cache-off and bootstrap with vision and cache:
  the clean figure, since nvidia-smi does not move with the other tenants. The weights stage falls by 5.9 to 6.8 GiB (it also moves with the box's other memory).
* MemAvailable at ready is **not** comparable across the columns: LoopRank's CPU job on node-1 ended between the two sets, and `boot_start` MemAvailable rose from about 107 GiB to 119 GiB.
  Against each boot's own baseline the same boots used 6.1 GiB less at bootstrap (40,436 to 34,235 MiB) and 5.3 GiB less at max cache-off.
* At `max` with `--prefix-cache auto` the plan gave the saving to the cache: 21.49 GiB instead of 4.38 GiB (reach and entries in `runs/arena-max-cacheauto/events.txt`),
  and the box still predicts, and measures, 11.7 to 12.0 GiB at ready, above the floor plus the 2 GiB variable part.
* The plan's padding is 0.06 GiB (`weights_alloc_padding_gib`, one arena block) and the prediction stays within 0.75 GiB at every boot; the host test that
  pins three profiles was repinned to these readings (`engine::tests::the_plan_predicts_memavailable_at_ready_within_a_gib_at_three_profiles`).

## Not proven

* No A/B of speed: the arena changes where weights sit, not what the kernels read, but decode time was not measured here (the loads are the same pointer-table gathers).
* The graph-equality hash is not comparable with `graph-exact`'s (my ids file); the live exactness diff and the oracle dump hash are the cross-record proofs.
* The unused tail of the last arena block was priced, not measured; the weights stage is within the box's noise of the loaded bytes.
* `long` and `max` with vision on, and images, were not re-booted.
