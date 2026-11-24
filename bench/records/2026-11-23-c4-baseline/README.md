# c4-baseline: the sustained c=4 metric, first measurements (gate speed-c4-baseline)

Plan experiments E3 (rolling metric, serial and drafted) and E4 (why bench.py's drafted c=4 is about 101, not 111) of
`docs/C4-DECODE-PLAN.md` section 7. Node-4, GPU SM clock locked at 2,200 MHz (`gpu-clocks lock`; `nvidia-smi` read 2,197 MHz
at the record), one server at a time, a fresh boot per configuration, the memory guard on.

Engine `1d53344d874b3a578152e35c500a2718599f5b61` (lab build, `build 1d53344d...`), recipe `9f3a87a`, profile bootstrap (262,144 x 4 slots,
`--prefill-chunk 4096`, int8 KV), `PREFIX_CACHE=on VISION=on`. Boot A: `SPEC=dspark` (the shipping defaults). Boot B: `SPEC=none`,
everything else equal. Boot is 16 s each.

## Commands

```
./inkling2mini.sh start        # boot A (SPEC=dspark PREFIX_CACHE=on VISION=on), then boot B with SPEC=none
for pass in 0 1 2 3: python3 bench/tools/bench.py --url http://node-4:8000 --model inkling-2-mini-1x-dgx-spark \
    --max-model-len 262144 --phases warm,decode,conc,rolling --raw --out PASS.json   # pass 0 discarded
python3 bench/tools/exactness.py record exactness-A-drafted.json   (boot A)  /  exactness-B-serial.json (boot B)
python3 bench/tools/exactness.py diff exactness-A-drafted.json exactness-B-serial.json    -> identical: 4 replies
```

`--raw` because `bench/reference.json` does not exist (the replies are not pinned against a file); every conc and rolling reply was
held to its solo reply of the same invocation (no FAIL). Pass 0 is the discarded warm-up pass (kept as `*-pass0.*`); the numbers
below are the median of passes 1-3 with the range. `usage.completion_tokens` is every token count.

## Numbers (median of passes 1-3, range)

| metric | drafted (boot A) | serial (boot B) |
|---|---|---|
| decode_c1_tok_s (mean of four kinds) | 81.25 (81.18-81.33) | 53.20 (53.18-53.20) |
| c1 prose / code / structured / json | 41.44 / 122.52 / 72.96 / 88.03 | 52.90 / 53.60 / 53.25 / 53.00 |
| decode_tokens_per_round, acceptance | 2.977, 0.382 (prose 1.55, code 5.75, structured 3.35, json 4.45 tok/round) | 1.002, - |
| decode_c4_tok_s (SPAN) | 101.82 (101.44-101.99) | 100.42 (100.28-100.43) |
| decode_c4_rolling_tok_s (THE number) | **131.50** (131.49-131.64) | **100.50** (100.46-100.54) |
| decode_c4_rolling_span_s | 124.59 (124.46-124.61) | 163.03 (162.97-163.09) |
| decode_c4_steady_tok_s | 131.83 (131.73-131.89) | 100.63 (100.59-100.67) |
| decode_c4_rolling_short_replies | 0 | 0 |
| decode_c8_tok_s (8 requests on 4 slots, two waves) | 115.82 (115.82-123.09) | 104.34 (100.46-104.55) |
| discarded pass 0: span / rolling / c8 | 100.74 / 130.49 / 112.46 | 100.30 / 100.46 / 105.76 |

The c=8 drafted figure is the least stable (two waves; the second wave's mix depends on who finishes first): 115.8 here against 125.9 in
`review-fixes-b` (engine 2e684eb).

## Residuals against `python3 bench/tools/c4_model.py` (measured / model - 1)

| quantity | measured | model | residual |
|---|---|---|---|
| c=1 serial step (53.195 tok/s = 18.80 ms) | 18.80 ms | 18.81 ms | -0.1 % |
| c=4 serial span | 100.42 | 99.94 | +0.5 % |
| c=4 serial rolling | 100.50 | 99.8 | +0.7 % |
| c=4 serial steady | 100.63 | 103.4 | -2.7 % |
| c=8 serial, 4 slots, two waves | 104.34 | 101.72 | +2.6 % |
| c=4 drafted rolling | 131.50 | 129.8 (engine rule), 128.5 (optimum) | +1.3 % / +2.3 % |
| c=4 drafted span | 101.82 | 108.0 (engine rule), 109.7 (optimum), 111.4 (specbench) | -5.7 % / -7.2 % / -8.6 % |
| c=4 drafted steady | 131.83 | 154.8 | -14.8 % |

Serial: inside 3 % everywhere (E3's gate: serial within 3 % of 99.8: **met**, +0.7 %). Drafted rolling is +1.3 % over 129.8 (E3's gate within
8 %: **met**). The steady residual (-14.8 %) is large, and the span residual is -5.7 %: both come from the traffic, not the round cost
[INFERENCE for the cause]: the model's mix is specbench's chat/code/json/reasoning, bench.py's is prose/code/structured/json. Bench prose accepts
less than specbench chat (1.55 tokens/round, 331 solo rounds, against 1.94 and 264) so it is the slowest stream. The rolling number landing
at the model is then partly a coincidence of two offsetting differences, not a validation of the drafted round cost; the steady-state figure of the model
(154.8, 4 kinds at once) is not what the rolling window holds, since in the closed loop the four requests in flight are rarely one of each kind.

Per-request timelines (raw JSON `raw.rolling.requests`, pass 1 drafted): per request the elapsed time is prose 27.4-29.6 s, code 7.9-9.5 s, structured 13.4-15.5 s,
json 10.5-11.8 s. All four workers end within 0.5 s of each other (124.1, 124.5, 124.6, 124.1 s): no stream finishes early and no slot idles
(four requests are in flight from the first start to the last end), 0 short replies. Serial: every request takes 20.0-20.7 s whatever the kind, the four workers end
at 162.9-163.0 s.

## The tail is real

Span 101.82 against rolling 131.50 is a 29 % difference (the plan said 27 %). In the c=4 span run (drafted, pass 1) the replies end at
prose 20.08 s, structured 14.01 s, json 11.23 s, code 9.30 s: from 14 s on one or two streams run, and prose runs alone from 14 s to 20 s
(about 6 s of a 20 s span). Serial has no tail (span 100.42, rolling 100.50, all four end within 0.1 s).

## E4: bench.py drafted c=4 on the merged engine

Three passes: **101.82 (101.44-101.99)**, discarded pass 0 100.74. This is the 101 of `2e684eb` (101.1), **not** specbench's 111.4, and it is stable
across passes (±0.3 %). So it is not the build (the merged engine reproduces the old engine's number) and not the learnt budget state
(the discarded first pass after the boot gives 100.7, the later passes 101.4-102.0). The difference is the prompts: the bench's
`prose` request accepts 1.55 tokens/round (1.585 inside the c=4 run), against specbench's `chat` at 1.94, so it needs 323 rounds for 512 tokens against 264 and ends at 20.1 s against
18.4 s; the other streams agree (code 89 rounds/5.75, structured 162/3.16 against reasoning 160/3.21, json 115/4.45 against 115/4.45). The span is the last
stream's finish, so the ratio of the spans (20.19/18.39 = 1.10) is the ratio 111.4/101.4. The bench's number is the right one for its own prompts; it
is also the number most affected by the tail (the slowest stream sets it).

## Exactness

`exactness.py record` on boot A (drafted, prefix cache on, vision on) and on boot B (serial, same flags): `exactness.py diff` prints
`identical: 4 replies` (generated ids, finish reasons and prompt_logprobs top-5 of the short, medium, long and xlong prompts, 64 tokens each).
The tool did not need the cache off. drafted == serial holds byte for byte. The bench's own conc and rolling equality checks (every concurrent reply == its solo reply) passed in every pass.

## Verdict

Plan: rolling about 129 drafted / 99.8 serial is right (131.5 / 100.5). The tail is real (29 %). The metric the ladder reads, drafted rolling, is 131.5 tok/s today; the owner's 200 is
1.52x away on this traffic.
