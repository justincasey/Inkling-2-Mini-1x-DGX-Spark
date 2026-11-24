# smoke-tool: gate 6.2 with the real tool (gate `smoke`)

Result **pass** (`meta.json`). `./inkling2mini.sh smoke` (`bench/tools/smoke.py`) against a fresh boot of the lab build of the
engine integration head `3287e80d` (`--spec none --prefix-cache off --no-vision`, profile `bootstrap`, 262,144 x 4, chunk 4096,
`EXTRA_ENV=REXL3_ALLOW_UNGATED_COMPONENTS=1`: bring-up only, the family is not yet qualified by the oracle comparison). Node-4,
GPU clock locked at 2,200 MHz (nvidia-smi reads 2197). Supersedes `2026-11-23-smoke` (hand-written curl scripts) by date.

Printed result (`smoke.txt`): `smoke: pass (chat ok, completion ok, tool ok, think-on ok, think-off ok)`.

| Check | What the tool saw |
|---|---|
| chat | 37 prompt, 32 completion tokens (finish `length`): content empty, all 30 reasoning tokens in `reasoning`; the tool passes it |
| completion | `1, 2, 3, ... 12,` continued ` 13, 14, 15, 16, 17, `, 16 tokens |
| tool | `finish_reason: tool_calls`, `get_weather` with `{"city":"Paris"}` |
| think-on (`reasoning_effort: high`) | 188 reasoning tokens, content `**9 sheep.** ...` |
| think-off (`reasoning_effort: none`) | content `9 sheep are left.`, **105 reasoning tokens**: the pack still reasons at effort none (the engine separates it into `reasoning`; it does not suppress it; same as the first-boot record) |
| image | **skipped** (`--skip-image`: `VISION=off`, the vision tower is plan step 4.7) |

`usage.spec` is `drafted 0` everywhere (no drafter). `serve-log-tail.txt`: the last 60 lines of the server log after the run, no
`level=error` or `level=fatal`. The `tests` array lists the five checks the tool ran; none is an oracle comparison.
