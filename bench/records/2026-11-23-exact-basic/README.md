# exact-basic: byte-exactness that needs no drafter and no prefix cache (gate `exact-basic`, plan 6.4 part)

Result **pass** (`meta.json`). Engine `3287e80d` (lab build, `REXL3_ALLOW_UNGATED_COMPONENTS=1`), `--spec none --prefix-cache off --no-vision`,
profile `bootstrap` (262,144 x 4), node-4. Boots A1, A2 at `PREFILL_CHUNK=4096`, boot B at `PREFILL_CHUNK=1000`
(`meta.json` `config` and `serve.command` are boot A's; boot B's command differs by `--prefill-chunk 1000` only, checked with
`PREFILL_CHUNK=1000 ./inkling2mini.sh print-serve`). Greedy, 128 new tokens, four fixed prompts tokenized by the server:
short 16, medium 521, long 2,030 and xlong 8,334 tokens (`bench/tools/exactness.py`; 128 tokens each, all `finish: length`).

| Property | Command | Result |
|---|---|---|
| concurrent == solo (boot A1) | `exactness.py concurrent --max-tokens 128` | the four prompts alone, then all four at once: generated ids equal on all four (`concurrent.json`). The four at once took 15.0 s each against 3.5 / 4.0 / 5.1 / 10.3 s alone |
| concurrent == solo (boot B, chunk 1000) | same | equal on all four (`concurrent-chunk1000.json`) |
| chunk width changes no byte | `exactness.py record` on A1 and on B, `exactness.py diff A1 B` | `identical: 4 replies` (`diff-chunk.txt`); 8,334 tokens is 3 chunks at 4096 and 9 chunks at 1000 |
| boot to boot | `exactness.py record` on A1 and on A2 (stop, start, record), `diff A1 A2` | `identical: 4 replies` (`diff-boot-to-boot.txt`) |

Limits, stated: this server refuses `prompt_logprobs` as unsupported (the Mini's engine has no full-vocabulary score path), so the tool's
`logprobs/*` lines in `concurrent.json` compare two `unsupported` markers and prove nothing; every claim above rests on the generated ids
and finish reasons alone. The longest prompt is 8,334 tokens. **Not run:** drafted == serial (needs the drafter, plan 4.6) and cached ==
uncached (needs the prefix cache, plan 4.5). The family is not yet qualified by the oracle comparison, so these equalities say the engine is
self-consistent, not that it is right (see `2026-11-23-greedy-vs-oracle`).
