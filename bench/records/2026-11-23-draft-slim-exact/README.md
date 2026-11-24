# DraftSlim exactness (gate `spec`, engine `aa2f9ad0a2d9c32662fcec205e66dc67e4900f3a`)

Companion of `draft-slim` (same engine commit, the same `confirm` session, GPU capped at 2,200 MHz: `raw/confirm/gpu-cap.txt`). Drafts may differ from the full head's; replies may not.

| proof | result |
|---|---|
| the target's rank dump (`inkling2_loop_rank forward_oracle_gate`, graphs on and off) | sha256 `c25f987627a65408965a272922706185032d796bfdfcdd4da8163353e27c45c0` (expected `c25f9876...c45c0`) |
| chunk-width logits (7 and 11 rows) | sha256 `a105f245c9792da01102b4eaf820ef783117cf027476bb20ead59dc48d97f41a` (expected `a105f245...f41a`) |
| graph replay == eager, 37 windows of node-1's `ids_long` copy | logits sha256 `44294f72e147c7e0ccd317d3aa3d1dda2b27534b50189915abb2c065f6ad90e1` (an ids file other lanes may not share: not comparable with their hash) |
| live replies, shipping defaults: `exactness.py record`, `diff` against the baseline `4f939a6` session | identical: 4 replies (base vs confirm) |
| `exactness.py concurrent`, `specedge.py` (65 of 65), `gates.py spec` and `cache`, `specmix.py`, `bench.py --ref` PASS pin | `raw/confirm/*.out`, the `tests` list of `meta.json` |
| the Mini drafter battery (`inkling2_dspark mini_`; the Small's tests need the Small pack and do not run here) | drafts within the float64 oracle's tolerances, graph replay == eager, row invariance, rollback, snapshots |
| `slim_picks_the_full_heads_winner` | SLIM_AGREE real cases (stride 1): 1197 anchors, 8379 steps; steps whose pick is not the full argmax 1; drafts that differ from DraftSlim off 1 (2 tokens) |

Not proven: DraftSlim is not bit-identical to the full head's draft (1 of 8,379 real steps picks another token, 1 anchor of 1,197 drafts another sequence). The target verifies every draft, so
replies are unchanged by construction and by the gates above; the drafter's own golden hashes changed (regenerated) because of the finer attention partition.
