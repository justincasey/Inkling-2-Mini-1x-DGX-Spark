# fuse-exact: the byte-identity proofs of the fuse lane (gate `exact-fuse`)

Engine `24a04afe9fda1f304865ed2af0f0c2e13bae7f08` against `1d53344d874b3a578152e35c500a2718599f5b61`, node-2. The lane changed three things, all in the drafter (the target's window
path is untouched): the draft pick (split top-k), the attention partial (a Mini-only kernel, the Small's `attn_part_kernel<6>` unchanged), the proposal in a graph.

| # | proof | result |
|---|---|---|
| 1 | Live: `exactness.py record` (4 prompts of 16, 521, 2,030 and 8,334 tokens, 128 greedy tokens and the prompt-logprobs top-5, solo, drafter on) on `1d53344` and on `24a04af`, then `diff` | "identical: 4 replies"; `replies` sha256 (canonical JSON, sorted keys, compact separators) **`f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d`** both times, the hash of `graph-exact` |
| 2 | Live: `exactness.py concurrent --max-tokens 64` (concurrent == solo) | PASS |
| 3 | Live: `bench.py --phases warm,decode,conc,prefill-32k --ref graph-exact/raw/pinned-replies-engine-51f2993.json` with the drafter on | **PASS pin** (4 prompt types x 512 tokens, 32K prefill + 128, c = 4 and c = 8 replies equal to solo) |
| 4 | Live: `specedge.py` | PASS: 65 cases, 0 failures, 0 warnings |
| 5 | Live: `gates.py spec` (drafted == serial, concurrent == solo) and `gates.py cache` (cached == uncached, 1K and 70K shared prefixes) | spec: pass (1709 drafted, 745 accepted); cache: pass |
| 6 | Live: `specbench.py` every drafted reply equals its serial reply, all six kinds and the c = 4 batch | `equal True` x 6, `PASS drafted == serial` |
| 7 | Rank gates, graphs on and off: sky dump sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0` (90,056,120 bytes, twice), chunk-width logits `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a` (widths 7 and 11; asserted by the test), 37-window graph == eager logits `6d0a555b92e02953c9e26efecdd9dece0ab211030ba8947cbe03b8407846c06b` | identical (`raw/gates-24a04af/`) |
| 8 | Drafter kernel battery on the real `mtp.safetensors` (oracle match, row invariance, rollback, int8 plane, graph replay == eager) | 17 passed |
| 9 | Spec (9), prefix cache (7), attention (28), ops (4, with the Small's golden), MoE (2 + 1) batteries | all passed |
| 10 | Host tests: `families::inkling2::` (31), `families::inkling2_loop::` (115, after re-pinning the drafter's priced state by +3,104 bytes: the split top-k's candidate buffer), `window_graph::` (6) | 152 passed |

The gates ran on the tree of `5f39463` (cumulative over the three changes; `24a04af` differs from it by a changelog fragment). The target kernels and the Small's kernels were not changed, so the Small's
golden hashes (`inkling2_ops golden`, `inkling2_moe golden`) are unchanged. Each change was tested alone (drafter battery, spec tests) before its timing was kept; the full battery ran once over the three.
Not run: the Small's real-weight tests (they skip without the Small's pack), a drafted 32K or 1M proof beyond what `specedge` and the gates cover.

Tools: `tools/mini_gates.sh OUTDIR` (needs `NODE`, `LAB_SRC`; runs the engine gates through `ops_dev.sh`), `tools/gate_summary.py`.
