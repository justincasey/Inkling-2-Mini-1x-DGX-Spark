# smoke-tool-2: gate 6.2 again, on a fresh build with no ungated components (gate `smoke`)

Result **pass** (`meta.json`). `./inkling2mini.sh smoke` against a fresh boot of the lab build of the committed engine head
`0ed77ffe9decda95f05b8c4c362655369eb00e6e` on node-4 (`--spec none --prefix-cache off --no-vision`, `bootstrap` profile, no
`REXL3_ALLOW_UNGATED_COMPONENTS`). Printed (`smoke.txt`): `smoke: pass (chat ok, completion ok, tool ok, think-on ok, think-off ok)`,
image skipped (`VISION=off`). Supersedes `2026-11-23-smoke-tool` by date; the replies are the same token for token (chat 30 reasoning
tokens, completion ` 13, 14, 15, 16, 17, `, `get_weather {"city":"Paris"}`, think-on 188 reasoning tokens, think-off 105).

`think-off` now asserts what the Mini does at `reasoning_effort: none`: a **short, non-empty reasoning** (105 reasoning tokens, 352 characters,
then `9 sheep are left.`). `none` renders `Thinking effort level: 0`; it is not "no thinking" for this model (see `2026-11-23-api-test-2`
for the measured table and the declared levels). An empty reasoning at `none` fails the check. `serve-log-tail.txt`: no `level=error` or
`level=fatal`, no `weights_differ`.
