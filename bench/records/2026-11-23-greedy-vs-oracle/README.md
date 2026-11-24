# greedy-vs-oracle: the engine's 16 greedy tokens against the oracle's (gate `greedy-oracle`, plan 6.3 end-to-end check)

Result **pass**: 16 of 16 tokens equal on the one fixture that exists, `chat-capital-effort0`. Engine `3287e80d` (lab build,
`REXL3_ALLOW_UNGATED_COMPONENTS=1`: the family is not yet qualified by the op-level oracle comparison), `--spec none --prefix-cache off --no-vision`,
node-4, same boot as the quality and loop runs of this session.

Command: `python3 bench/tools/greedy_oracle.py --ssh spark@node-2 --dir /home/spark/inkling-2-mini-1x-dgx-spark/fixtures/spec --out greedy-oracle.json`
(new tool of this commit series: it streams `stlite.py` and a reader to `python3 -` on node-2, which prints the fixture's prompt ids, its own greedy
ids and the top-1 / top-2 logit margin per generated position from the fixture's `logits`; nothing is copied; the engine is then asked through
`/v1/completions` with the 25 prompt token ids, temperature 0, 16 tokens). Printed: `chat-capital-effort0: equal (16 tokens, prompt 25, 1 requests, 0.5s)`.

| Item | Value |
|---|---|
| prompt | 25 ids from the fixture's `tokens` (`[200002, 200004, 34182, ..., 200010]`): the first 10 ids are this server's render of the system line `Thinking effort level: 0` (checked with `/v1/chat/completions/render` in this session), then a 12-token user turn; the user text itself is not in the manifest and was not recovered (the oracle's thinking says the user asks for the capital of France in one word) |
| oracle tokens | `[200001, 200008, 371, 150409, 171960, 149887, 149876, 151451, 149879, 151194, 11, 150011, 151290, 13, 170289, 150893]` = `<\|message_model\|><\|content_thinking\|>The user asks for the capital of France, one word. Need answer` |
| engine tokens | identical, all 16 positions |
| oracle top-1 - top-2 margin per position | 27.17, 22.33, 11.73, 15.52, 4.30, **0.198**, 0.988, 14.78, 13.16, 18.74, 0.921, **0.103**, 9.24, 6.51, 1.03, 4.72 |
| oracle token is the oracle's own top-1 | all 16 |

Two positions are near-ties (5: 0.198 and 11: 0.103 logit units, a tenth to a fifth of a logit) and the engine took the oracle's token at both, so the
match is not a trivially easy one. What it does not show: 16 generated tokens of a short prompt, one fixture (the others were still being generated on
node-2 when this ran: the manifest listed one entry), thinking text only, no tool, no long context, no op-level comparison. `greedy-oracle.json` has the
prompt ids, both token lists and the margins; the fixture itself stays on node-2.
