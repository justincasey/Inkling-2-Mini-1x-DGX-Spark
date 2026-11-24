# prefill-dense-pipe-exact: the dense pipe changed no byte (gate `exact-prefill`)

Engine `fd6ef616be704ad0863bd5dbebdc97408527d4bb` (branch `inkling2-loop-densepipe` on the integration head `4f939a6`; the same commit as `2026-11-23-prefill-dense-pipe`), node-3, TP1, greedy throughout. The "before" values are the ones
`2026-11-23-prefill-exact` and `-prefill-push-exact` pinned: this engine must reproduce them. The change is kernel scheduling and sharing only (design and argument in the engine header of `csrc/inkling2_dense_pipe.cu` and in the speed record):
per output element the same fp32 mma chains in the same order, the same cross-warp and cross-split sums, the same epilogue, and the rotated activation the same fp32 products, butterflies, scale and fp16 rounding as `rot_in_frag_kernel`. The
gates ran on the committed tree (`tools/gates.sh` of the speed record, `raw/gates-new.log`), the live checks on a server built from this commit in the shipping configuration, under the verified 2,200 MHz cap. Every line is in `raw/`.

| # | Claim | Evidence |
|---|---|---|
| 1 | The pipe equals the shipped kernels bit for bit, output and rotated input | `inkling2_dense_pipe` (3 tests): the rotation pass (1 to 3 outputs from one x) and each GEMM equal `exl3_project`'s rotated input (the shipped workspace after the call, byte for byte) and output on random 6-bit `mul1` tensors of the six Mini shapes (the o tensor with a bias) at 4,096, 1,000, 129, 257, 4,080 and 8,192 rows, in the groups of a visit (q+gate+r, k+v, k, v, o and mixed orders), with scaled-up, zeroed and ordinary rows; on the pack's real tensors of layers 11, 0 and 23 at 4,096 and 1,001 rows; and rows up to 128 and a foreign shape fall back to the shipped path and equal it |
| 2 | The rank's oracle gate | `inkling2_loop_rank::forward_oracle_gate`: dump sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0` (90,056,120 bytes) twice; chunk widths 7 and 11: logits sha256 `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a`; decode rows 22-25 and the 3-row window rollback equal. With `REXL3_INKLING2_LOOP_DENSE_PIPE=0` the same two hashes (`raw/gates-pipe-off-rank.txt`) |
| 3 | A prompt past the window | `long_context_gate` on the 1,119 ids: top-1 149,897 at 29.695383, chunks of 129 and 300 and the decode row equal |
| 4 | Decode windows and graphs | `inkling2_loop_graphs`: 37 windows, logits sha256 `8e214bd03306e8f0e3e1f203566b2d4d5057d7ce503b15fad57f741f2c50c2b7` (the value of `prefill-exact` / `prefill-push-exact`), 24 captures 189 replays, the refused capture passed |
| 5 | The prefix cache, 20 cuts | `inkling2_loop_prefix` 4 tests: the 20 cuts' last-row / state / decode digests `identical` (20 of 20), the logits planes, the restore after eviction, a 12,000-token cut |
| 6 | 135,168 tokens, every chunk | `prefill_curve PF_EVERY=1`: the chain of the 33 chunk digests `f684af750c222810a48e72c536fc5385c0ff68f0f6316af66db127a3a709bcb3`, the value of `prefill-push-exact`; at chunk 8,192 the last row's digest `823553ce2d0494d0b2b72a11725c5a58bb952d5e866d800a1fba1470144f140d`, also the 4,096-row run's (the two nsys runs, pipe on and off, print it as well) |
| 7 | The kernels | `inkling2_moe` 8, `inkling2_moe_mixed` 10, `inkling2_attn` 28 (`mini::` and `small::`), `inkling2_ops` 9 (the Mini dense goldens on real tensors against the oracle, row invariance at every shape, unembed, split top-k, the Small's golden hashes: unchanged), `exl3_big` 4 and its `#[ignore]`d golden run explicitly (`golden_exl3_big_hashes`: 42 lines, identical to `testdata/gpu/golden_exl3_big.txt`, `raw/exl3-big-golden-ignored.txt`), the drafter's `inkling2_dspark` 28 |
| 8 | Host tests | `REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked`: `families::inkling2::` 31 passed, `families::inkling2_loop::` 115 passed (the plan's pinned MemAvailable predictions and the item list moved by the +30 MiB `dense_pipe_rotated_inputs` item: `memory.rs`, `engine.rs`) |
| 9 | Live, shipping configuration (pipe on, the default) | `exactness.py record` vs the `7da8c61` record of `prefill-exact`: **identical: 4 replies**, replies sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` (the value of `prefill-push-exact`); `bench.py --ref pinned-replies-engine-51f2993.json`: **PASS pin** (4 x 512 decode tokens with the drafter, c = 4 / 8 equal to solo, a cold 32K prompt); `specedge.py`: **PASS: 65 cases, 0 failures, 0 warnings**; `exactness.py concurrent`: `"pass": true` (`raw/ship-pipe-on/`) |
| 10 | Live, the cap | the controller's `start` printed `GPU SM clock capped at 2200 MHz on node-3` before the server of `raw/ship-pipe-on` (the rerun that is recorded; its sweep is in the speed record); the other servers of the speed record are preceded by `-lgc 2200,2200` success lines (`../2026-11-23-prefill-dense-pipe/raw/clock-lock.log`) |

Not proved: images (the shipping boot has the tower; no image was sent and `inkling2_loop_vision` was not run: image rows are ordinary rows of the text path, which is what the gates cover), the 1M window, TP > 1; the pipe for any
shape but the Mini's six; "equal" is equal to the engine as it was, not correct against the oracle. The first run of the live checks (before the cap order of the lead) gave the same results (identical, PASS pin, 65 of 65) and was
discarded and rerun under the verified cap. The gates (`raw/gates-new.log`) ran before the order too; they are bit-exactness proofs and do not depend on the clock.

## Commands (node-3 stands for the node's address)

```sh
NODE=node-3 ../2026-11-23-prefill-dense-pipe/tools/gates.sh raw/gates-new.log                     # 11 test binaries, the PF_EVERY=1 and chunk-8192 prompts
REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked families::inkling2::; families::inkling2_loop::
./inkling2mini.sh build fd6ef616be704ad0863bd5dbebdc97408527d4bb && ./inkling2mini.sh start && ../2026-11-23-prefill-dense-pipe/tools/live.sh ship ship-pipe-on
```
