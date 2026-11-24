# needle: a pass-code at 10% and 50% depth of a 260K prompt (gate `needle`, plan 6.5)

Result **pass** at an answer budget of 1,024 tokens; the tool's default budget of 128 tokens fails for a reason that is not retrieval (below).
Engine `3287e80d` (lab build, `REXL3_ALLOW_UNGATED_COMPONENTS=1`), `--spec none --prefix-cache off --no-vision`, profile `bootstrap`
(window 262,144, chunk 4096, 4 slots), node-4, cold prompts, effort 0 system line (`Thinking effort level: 0`, the tool's default).

| Run | Command | Depth 0.1 | Depth 0.5 |
|---|---|---|---|
| 1 (`run1-128-tokens/`) | `./inkling2mini.sh bench needle` (tool default `--max-tokens 128`) | prompt 260,916 tokens, prefill 300.3 s, **empty content**, finish `length`, the code is in the reasoning (`found_in_reasoning`) | same: prefill 300.6 s, empty content, finish `length`, code in the reasoning |
| 2 (`run2-1024-tokens/`) | `python3 bench/tools/needle.py --tokens 262144 --max-model-len 262144 --max-tokens 1024 --depth D` | prompt 260,019 tokens, prefill 299.0 s, answer `OTTER-2624-OTTER`, finish `stop`, 476 reasoning tokens, total 322.1 s | prompt 260,019 tokens, prefill 299.1 s, answer `OTTER-2624-OTTER`, finish `stop`, 268 reasoning tokens, total 312.2 s |

What changed between the runs, and why: the pack reasons before it answers even at effort none, and on a 260K prompt that reasoning took
268 to 476 tokens, so 128 new tokens end the reply inside the reasoning (`content` empty, the pass-code present in the reasoning text). The
needle tool counts that as a miss. Run 2 only raises `--max-tokens` to 1,024 (the prompt shrinks by the 896 extra tokens the tool reserves
under the window: 260,019 against 260,916). The controller's `bench needle` now passes `NEEDLE_MAX_TOKENS` (default 1024, `.env.example`).
No chunk or time-out change was needed: the tool's per-request time-out is 7,200 s and a 260K prompt takes about 300 s cold.

Observations: prefill of 260K tokens at 4,096-token chunks runs at about 870 tokens/s (260,019 / 299 s). The two prompts are the same text with the needle moved; both runs
used one request at a time. Not run: 224K at depth 0.9, 1M windows (profile long/max), four lanes (`--lanes 4`), the vision-off baseline is the
only configuration. Guard minimum MemAvailable over the boot and both runs: 78,638 MiB (`memwatch report`, window from `start`), far above the
8 GiB floor. No `level=error` or `level=fatal` in the server log.
