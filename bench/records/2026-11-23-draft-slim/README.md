# DraftSlim: the drafter's Markov steps without the 103 MB table (lane `k-slim`)

Gate `speed-decode`. Engine `aa2f9ad0a2d9c32662fcec205e66dc67e4900f3a` (lane branch `inkling2-loop-slim`, merged on the integration head `be0ba1e`), baseline engine `4f939a6` (the integration head
before the lane; decode is unchanged by the later integration commits, which touch prefill and comments), node-1, GPU SM clock capped at 2,200 MHz for every run
(`raw/*/gpu-cap.txt`: the success line of the cap, applied by `scripts/gpu_cap.sh` through `tools/session.sh` before each server start):

- `base`: GPU clocks set to "(gpuClkMin 2200, gpuClkMax 2200)" for GPU 0000000F:01:00.0 / All done.
- `slim`: GPU clocks set to "(gpuClkMin 2200, gpuClkMax 2200)" for GPU 0000000F:01:00.0 / All done.
- `slimoff`: GPU clocks set to "(gpuClkMin 2200, gpuClkMax 2200)" for GPU 0000000F:01:00.0 / All done.
- `confirm`: GPU clocks set to "(gpuClkMin 2200, gpuClkMax 2200)" for GPU 0000000F:01:00.0 / All done.

Design note, the ideas tested and what is exact: `docs/DECODE-SPEED.md`, "Draft slim". Exactness proofs: record `draft-slim-exact` (same engine commit).

## Result (median of 3 after a discarded warm-up, fresh server per configuration, shipping defaults `SPEC=dspark PREFIX_CACHE=on VISION=on`)

| metric | baseline `4f939a6` | lane build `55f4c1d`, DraftSlim off (`REXL3_INKLING2_LOOP_DRAFT_SLIM=0`) | lane `aa2f9ad` (confirm) | confirm vs baseline |
|---|---:|---:|---:|---:|
| **decode_mean4_tok_s** (`bench decode`, printed as `decode_c1_tok_s`) | 83.99 | 84.019 | 90.079 | +7.2 % |
| prose | 49.724 | 49.684 | 51.697 | +4.0 % |
| code | 119.203 | 118.66 | 128.83 | +8.1 % |
| structured | 79.214 | 79.688 | 86.324 | +9.0 % |
| json | 87.814 | 88.007 | 93.345 | +6.3 % |
| **decode_c4_rolling_tok_s** (`bench rolling`) | 134.386 | 135.388 | 141.796 | +5.5 % |
| decode_c4_steady_tok_s | 135.391 | 136.189 | 142.695 | +5.4 % |
| c=4 span (`bench decode`) | 109.271 | 108.139 | 115.044 | +5.3 % |
| c=8 span (2 waves) | 128.521 | 128.491 | 127.33 | -0.9 % |
| specbench c=4 aggregate (chat, code, json, reasoning) | 115.352 | 115.577 | 121.214 | +5.1 % |

The DraftSlim-off column is the lane's earlier build with the switch off (it still has the finer draft attention partition): the A/B of the pick alone.

Tokens a round per kind (`bench decode`, each kind's own `spec` counters, median of 3; the policy decides how many rounds draft, so prose, which rarely drafts,
moves from run to run):

| kind | baseline | lane |
|---|---:|---:|
| prose | 1.27 | 1.106 |
| code | 5.278 | 5.333 |
| structured | 3.2 | 3.2 |
| json | 4.0 | 3.969 |

specbench (512 greedy tokens, median of 3; drafted tok/s, tokens a round):

| kind | baseline tok/s | lane tok/s | baseline tok/round | lane tok/round |
|---|---:|---:|---:|---:|
| chat | 52.859 | 56.759 | 1.701 | 1.6935 |
| code | 116.49 | 126.768 | 5.4468 | 5.4662 |
| json | 85.053 | 90.744 | 4.0421 | 4.0315 |
| longctx | 66.11 | 70.736 | 4.3268 | 4.3268 |
| tool | 126.75 | 135.99 | 6.3257 | 6.3621 |
| reasoning | 76.636 | 83.435 | 3.0843 | 3.103 |

## The proposal (lab: `inkling2_dspark slim_proposal_timing`, synthetic features, int8 plane, median of 20 `propose` calls, capped)

| positions | requests | DraftSlim off ms | on ms | saved ms |
|---:|---:|---:|---:|---:|
| 4096 | 1 | 6.504 | 3.313 | 3.191 |
| 4096 | 2 | 7.303 | 3.78 | 3.523 |
| 4096 | 4 | 8.993 | 4.781 | 4.212 |
| 32768 | 1 | 7.281 | 4.07 | 3.211 |
| 32768 | 2 | 8.709 | 5.25 | 3.459 |
| 32768 | 4 | 11.793 | 7.757 | 4.036 |

The off column has the finer attention partition (0.45 ms of it); the full-table path before the lane measured 7.08 / 7.73 / 9.08 ms at 4K (first lab run).
nsys timeline of one proposal at 4,096 positions: `profile/timeline-proposal-final.txt` (3.34 ms span; the pick kernel about 16 us, the scan 9.5 us, a step about 55 us).
`slim_picks_the_full_heads_winner` (1,197 anchors of the real-feature cases, 7 steps each): SLIM_AGREE real cases (stride 1): 1197 anchors, 8379 steps; steps whose pick is not the full argmax 1; drafts that differ from DraftSlim off 1 (2 tokens)

## Commands

- `NODES=<node-1> REXL3_SRC=<engine worktree> ./inkling2mini.sh build REV` for REV in `4f939a6` (baseline), `55f4c1d`, `aa2f9ad`; `tools/ab.sh` and `tools/chain.sh` drive the sessions.
- `tools/session.sh TAG MODE` (cap, `start`, `exactness.py record`, `bench decode` x4, `bench rolling` x4, `specbench.py --concurrency 4`; MODE full adds the gates of the exactness record), `tools/summarize.py` the tables, `tools/lab_final.sh` the lab numbers, `tools/make_records.py` this record.
- Analysis behind the design note: `tools/a1.py` .. `a4.py` (numpy on a dump of `fold_w`, the base rows and the latents of 77 real positions), `tools/nsys_slim.sh`, `tools/tl.py`.

## Notes

- A first A/B (lab tests, one base and one slim session and a started third) had no proof that the cap was in force; it was discarded and none of its numbers is used anywhere.
  Everything above is from runs started after the cap was applied and its success line logged.
- The finer draft attention partition changes the summation order of the drafter's context attention (the block rows `hd` move in their last bf16 bits); the Mini's golden hashes in
  `testdata/gpu/golden_inkling2_loop_dspark.txt` were regenerated and the oracle tolerances hold.
- Not built (the next lever): a bit-exact per-row EXL3 evaluation of the unembed for the candidates, which would replace the 348 MB unembed read (1.5 ms of the proposal's 3.3) by a proxy scan.
- The engine's budget prices (`round_cost`, proposal 6.4 / 7.0 / 7.7 / 8.3 ms) are unchanged and now stale; re-pricing them is an engine-policy change not made here.
