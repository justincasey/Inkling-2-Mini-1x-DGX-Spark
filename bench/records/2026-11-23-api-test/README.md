# api-test: the OpenAI surface on the Mini, gate 6.8 (gate `api-test`)

Result **fail**: 16 PASS, 1 FAIL of `tests/api_conformance.py`; `bench/probe.py` passes (15 PASS lines). Same boot as `2026-11-23-smoke-tool`
(engine `3287e80d`, lab build, `REXL3_ALLOW_UNGATED_COMPONENTS=1`, `--spec none --prefix-cache off --no-vision`, node-4, window 262,144).

Commands: `./inkling2mini.sh bench probe` (`probe.log`, `probe.json`), `./inkling2mini.sh bench api` twice.

## The first run: 14 PASS, 3 FAIL (`api-test-run1-before-test-fix.log`)

| Check | Message | Where the fault is (`localise.py`, `localise.out`) |
|---|---|---|
| render | `usage.prompt_tokens 25 != render count 27` | **the test**: it asked `/v1/chat/completions/render` without `reasoning_effort` but the chat with `reasoning_effort: none`; the two renders differ by 2 tokens (22 vs 20 for a one-line prompt). With the same field on both calls render and usage agree: 20 = 20 and 22 = 22 |
| stop | `the stop string is in the reply: 'ALPHA BRAVO '` | **the test**: the content is `ALPHA BRAVO ` (cut before CHARLIE, `finish_reason: stop`); the string CHARLIE is in the model's *reasoning* text, which the engine does not scan (generation went on through the reasoning, then the stop cut the content). The test now checks the visible content only |
| errors | `not a 4xx client error: {'max_tokens over context': 200}` | **the engine**: `max_tokens` of 262,145 and of 10,000,000 are accepted with HTTP 200 and answered (the Small's engine and OpenAI answer 4xx). Nothing wrong comes back; the request is not refused |

The two test defects were fixed in `tests/api_conformance.py` (commit `Gates: api conformance sends reasoning_effort to render, stop is
checked on the visible reply`; no other change; `tests/test_bench.py` 27 and `tests/test_gates.py` 68 still pass). The second run
(`api-test-run2.log`): 16 PASS, 1 FAIL, the `errors` check above. The gate stays `fail` until the engine refuses `max_tokens` beyond the window
(an engine item for the lead, not investigated here) or the lead decides that clamping is the contract and relaxes the check.

## Probe (`probe.log`)

`PASS`: health, models (`max_model_len` 262144, `kv_pool_tokens` 1048576), window, tokenize (`hello world` = 2 tokens), chat (17*23 -> 391),
usage (adds up, `spec` block present and zero), reasoning (619 chars, 188 reasoning tokens), effort at system level 0, `none` and `high` accepted,
tools (`get_weather {"city":"Paris"}`), metrics, spec (drafter off: N/A as the plan says), spec serial, prefix cache off (0 cached tokens).
`probe.json` also says: `pack_validated` "no (--allow-unvalidated-pack)", and its reasoning levels table lists `none` and `minimal` as
`thinks: false` while the engine does think at `none` (smoke-tool: 105 reasoning tokens): a mismatch between the declared level table and the behaviour.

The other `api_conformance` checks passed: health, models, tokenize, chat, max_tokens, streaming, reasoning, tools (streamed identically,
`tool_choice` none, result answered), completions, metrics, spec, prefix cache, concurrency, determinism (64 identical ids).
