# smoke: first replies of the Mini on one Spark (gate `smoke`)

Result **pass** (`meta.json`): the same boot as `2026-11-23-first-boot` (engine `c0cf5a1b...`, ungated components named at boot,
`--spec none --prefix-cache off --no-vision`), requests by `smoke_curl.py` and `extra.py` (python urllib; `bench/tools/smoke.py`
of the BenchTools lane replaces them when it merges), greedy, replies in `replies/*.json` (request, full reply, wall seconds).
Judged by eye: none of this is an oracle comparison.

| Request | What came back |
|---|---|
| chat, `reasoning_effort: none` ("capital of France, one word") | content `Paris`, finish `stop`; the reasoning text still arrives in `reasoning_content` (the pack thinks at effort none: `reasoning_tokens` 30 of 37) |
| chat, effort `high` ("17 * 23, think it through") | `reasoning_content` works the sum (340 + 51 = 391, checks it by 460 - 69), content `17 × 23 = ... = 391.\n\n391`, 145 tokens, 3.96 s |
| raw completion ("The three primary colors are") | `:\n\nA. Red, yellow, and blue\nB. Red, green, and blue\nC. Red,` (24 tokens, finish `length`) |
| tool call (`get_weather(city)`) | `finish_reason: tool_calls`, `get_weather` with `{"city":"Paris"}`, parsed; content null, reasoning present |
| 128-token completion ("Once upon a time") | fluent English prose; 3.42 s for 128 tokens: **37.4 tokens a second** over the whole request (serial, no graphs, a 4-token prompt) |
| long prompt, 12,610 tokens (`extra.py`) | 11.1 s wall for prefill + 32 tokens, about 1,150 prompt tokens a second over four 4,096-row chunks; the reply repeats the sentence (a continuation, not a summary: not judged) |
| four concurrent 64-token completions of the same prompt | all four byte-equal to the solo reply; 2.70 s for four against 1.72 s solo |

No wrong reply seen; no `level=error` or `level=fatal` in `../2026-11-23-first-boot/serve.log` after the requests. Untested: images
(vision is off), streaming, effort levels other than none and high, prompts past 12.6K, any long-context retrieval.

The `tests` array of `meta.json` lists the seven checks of the table above (each judged by eye from the reply files, none against the oracle); the lead added it so the ladder's record checker can count them.
