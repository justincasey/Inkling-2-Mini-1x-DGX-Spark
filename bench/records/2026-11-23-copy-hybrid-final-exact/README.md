# copy-hybrid-final-exact: the exactness battery of `copy-hybrid-exact` on the final engine head (gate `spec`)

Engine `f4922eb462ef2d2b27e20024e0fc39bb7673cb11`, node-4, shipping defaults, one fresh server (`../2026-11-23-copy-hybrid-exact/tools/run_exact.sh`, `raw/`). Replies `identical: 4 replies`, sha256 `f7975a6c1dae761ef24508177ce06da5e0607d94f1e81c1c7a5a72b9e105433d` (the same as the pinned engine); `bench.py --ref`: `PASS pin`;
`exactness.py concurrent`: pass; `gates.py spec`: pass (1,512 drafted, 729 accepted, 414 rounds); `gates.py cache`: pass; `specedge.py`: 125 cases, 0 failures, 1 warning (as in `copy-hybrid-exact`: no 7-token copy draft was rejected at its last token). Host tests of the final tree: `gb-model --lib families::inkling2` 146 passed,
`gb-engine` `copy_hybrid` 6 and `chained_round` 2 passed. The kernel and rank batteries were not rerun on this head.
