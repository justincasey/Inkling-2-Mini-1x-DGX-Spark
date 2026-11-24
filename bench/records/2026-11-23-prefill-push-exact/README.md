# prefill-push-exact: the prefill push changed no byte (gate `exact-prefill`)

Engine `ce650e376cdeb8eff6b4d63f6ae63facc95a0d85` (branch `inkling2-loop-prefill` on the integration head `1d53344`; the same commit as `2026-11-23-prefill-push`), node-3, TP1, GPU clock locked at 2,200 MHz, greedy throughout. The
"before" values are the ones `2026-11-23-prefill-exact` pinned on `7da8c61` and `1a36951` (the rank's dump, chunk-width logits, 37-window hash, 20 prefix cuts, 33 chunk digests, the replies hash): this engine must reproduce them. The
engine changes of the push are kernel scheduling and widths only: 2-tile MoE blocks at 4 and 5 bits with the halves adjacent in the grid, the dense large-row kernel's strip-fastest order, and the chunk cap of 8,192 rows
(`--prefill-chunk`, default stays 4,096). The gates ran on the final tree (`tools/gates.sh new`), the live checks on a server built from this commit in the shipping configuration. Every line is in `raw/`.

| # | Claim | Evidence |
|---|---|---|
| 1 | The rank's oracle gate | `inkling2_loop_rank::forward_oracle_gate`: dump sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0` (90,056,120 bytes) twice; chunk widths 7 and 11: logits sha256 `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a`; decode rows 22-25 and the committed 3-row window equal. Same lines as `prefill-exact` (`raw/gates-new.log`) |
| 2 | A prompt past the window | `long_context_gate` on the lane's 1,119 ids (sha256 `d5191e8d...e1896`): top-1 149,897 at 29.695383, chunks of 129 and 300 and the decode row equal; the value of `prefill-exact` |
| 3 | Decode windows and graphs | `inkling2_loop_graphs`: 37 windows, logits sha256 `8e214bd03306e8f0e3e1f203566b2d4d5057d7ce503b15fad57f741f2c50c2b7` (the value of `prefill-exact`; `6d0a555b...` belongs to the id list of `graph-exact`, which is not kept), the refused capture passed |
| 4 | The prefix cache, 20 cuts | `inkling2_loop_prefix` 4 tests; the 20 cuts' last-row / state / decode digests equal the table of `2026-11-23-prefix-cache` (20 of 20) |
| 5 | 135,168 tokens, every chunk | `prefill_curve PF_EVERY=1`: the 33 chunk digests equal the 33 of `prefill-exact` (sha256 of the lines `41f0df8bafd5c1c7...`), chain `f684af750c222810a48e72c536fc5385c0ff68f0f6316af66db127a3a709bcb3`; at chunk 8,192 the last row's digest `823553ce2d0494d0b2b72a11725c5a58bb952d5e866d800a1fba1470144f140d` is the 4,096-row run's; 32K at 1,024 / 2,048 / 4,096 / 8,192 rows: `8aca0412...` |
| 6 | The kernels | `inkling2_moe` 8, `inkling2_moe_mixed` 10 (decode == prefill per pair over bands, 4 and 5 bits, the real layers 12 / 22 / 23 against the oracle), `inkling2_attn` 28, `inkling2_ops` 4, `exl3_big` 4 (the dense kernel with its new block order), the drafter's `inkling2_dspark` 28 (taps are written in prefill) |
| 7 | Host tests | `REXL3_NO_CUDA=1`: `families::inkling2::` 31, `families::inkling2_loop::` 115, the new `serve::tests::a_family_cap_widens_the_prefill_chunk_check` |
| 8 | Live, shipping configuration | `exactness.py record` vs the `7da8c61` record: "identical: 4 replies", replies sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` (the `graph-exact` value); `bench.py --ref pinned-replies-engine-51f2993.json` (sha256 `5ca5bab6...fbdec`, 4 x 512 decode tokens with the drafter, c = 4 / 8 equal to solo, a cold 32K prompt): `PASS pin`; `specedge.py`: `PASS: 65 cases, 0 failures, 0 warnings`; `exactness.py concurrent`: `"pass": true` (`raw/ship/`) |
| 9 | Live, `SPEC=none` | all 16 passes of the sweep (8K .. 224K, the same prompts as the baseline sweep of `7da8c61`): generated ids equal, 16 of 16, sha256 of all ids `afe69323e08332948a4aad80d74f9b95f65c2dd833d98a3773bd32d1409defb8` (`raw/nospec/`) |
| 10 | Live, 8,192-row chunks | a shipping server at `--prefill-chunk 8192`: `exactness.py diff` vs `7da8c61`: identical, replies sha256 `f7975a6c...`; the 12 passes of its sweep equal the 4,096 server's 12 (`raw/c8192/`) |

Not proved: the 1M window, images (the shipping boot has the tower; no image was sent here, `inkling2_loop_vision` was not run), TP > 1; "equal" is equal to the engine as it was, not correct against the oracle.
`golden_exl3_big` is `#[ignore]` and was not run. `raw/gates-8192-only-1d53344.log` is an earlier run of the same gates on the tree with only the 8,192 change; `raw/prev-8ce953b/` holds the live checks and sweeps of the
commit before the chunk-cap change (the kernels are the same; every number there agrees with the final ones within noise and is superseded).

## Commands (node-3 stands for the node's address)

```sh
tools/gates.sh new raw/gates-new.log                         # ops_dev.sh on node-3: 10 test binaries, the PF_EVERY=1 and chunk-8192 prompts
REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked families::inkling2::; families::inkling2_loop::; serve::tests::a_family_cap
./inkling2mini.sh build ce650e376cdeb8eff6b4d63f6ae63facc95a0d85 && ./inkling2mini.sh start && tools/live.sh ship
SPEC=none PREFIX_CACHE=off VISION=off ./inkling2mini.sh start && tools/live.sh nospec;  PREFILL_CHUNK=8192 ./inkling2mini.sh start && tools/live.sh c8192
```
