# decode_mean4 plan: what 100 tok/s of mean c=1 decode needs, and where it can and cannot come from

Addendum to `docs/C4-DECODE-PLAN.md` (lane `k-c4plan`; desk work, no node run). Same discipline: every figure carries a source file, or is DERIVED (a formula on sourced numbers) or
[INFERENCE]. The model is `bench/tools/c4_model.py` (section 12 of its output is every table below; `--validate` gates the four per-kind rates and their mean at 10 %;
`tests/test_c4_model.py`, 10 tests). Measured inputs: record `fuse-decode` (engine `24a04af`, node-2, shipping defaults, medians of 3), `fuse-decode/raw/round-cost-table-real-ids-final.txt`
(DecodeFuse's windows with real texts), `drafted-final-24a04af/bench-decode-{1,2,3}.log` (per-kind rates and the spec counters of bench.py's own four kinds).

**Verdict.** `decode_mean4_tok_s` = mean(prose, code, structured, json) = **83.3** today (42.7 / 125.6 / 75.0 / 90.0). **100 is not reachable on the A lane at the estimates**: with every
lever delivered at its estimate the model gives **93.7** (bound, every lever at its maximum, **106.7**). The gap at the estimate is x1.07; it closes with **either** 8 % more tokens per round
on code / structured / json (x1.079 gives 100.0), **or** a changed drafter that reads its Markov table once (proposal 4.5 ms: 96.2, plus x1.046 tokens a round), **or** one B-lane lever:
experts skipped below about 0.25 x the top weight (about 8 % of the routing mass) or a 3.8 bpw pack. Reaching 100 inside the A lane needs the bound on the proposal,
the host, the glue and the bandwidth **at once**: it is a stretch, not a plan. (The lead's rough read, 98-102, had the proposal at 4.5 ms: with the same drafter arithmetic its floor is 5.6 ms, below.)

## 0. Correction to `C4-DECODE-PLAN.md`: the proposal is at its byte floor

The c=4 plan said the proposal's bytes are 0.34 GB (1.45 ms) so 6-7 ms of it is latency. That was wrong. `weights_bytes = 339,917,888` (`timing.txt`) is the drafter's 0.236 GB **plus the folded
Markov table** (201,024 x 256 bf16 = 102.9 MB: 236.6 + 102.9 = 339.5 MB [INFERENCE: sums to within 0.1 %]); the seven Markov steps each re-read that table (it does not fit the 24 MiB L2:
`markov_logits_kernel`, 446 us a step at 231 GB/s, `DECODE-SPEED.md`), and the head reads the unembed (0.348 GB, 1.5 ms) once. Bytes of a proposal: 0.340 + 6 x 0.103 + 0.348 =
**1.306 GB = 5.58 ms at 233.8 GB/s**. The proposal measures 6.69 ms (synthetic ids) / 6.90 ms (real texts) after DecodeFuse's three changes: **within 1.3 ms of its bytes**. DecodeFuse's
remaining items (fused argmax in the Markov logits 0.17 ms, L2 policy for the folded table about 0.3 ms, the global-layer attention 600 us against about 200 [INFERENCE]) sum to about 0.9 ms: 5.8 ms.
Going lower changes the drafter's arithmetic (Markov logits that do not re-read the table each step, or a quantised table): the outputs stay byte-exact (any draft is verified), the drafts and so
the acceptance change, which needs a measurement (M4). The lead's 4.5 ms target needs that. The same correction applies to the c=4 plan's A1: at four requests the floor is about 5.6 ms of
8.9 and the lever is x1.033-1.036 with the same drafter, not x1.05-1.10 (`C4-DECODE-PLAN.md` is patched where it said so; its all-A ladder used half the proposal and is about 1.5 % optimistic).

## 1. The model at c=1 per kind

A drafted round of one stream = window(1 + d rows) + proposal (when d > 0) + residual. For each bench kind the shipping policy's depth distribution comes from `fuse-decode`'s
`specbench.json` (`per_position.offered`: code 97 % of rounds at depth 7, json 94 % at 7, reasoning 81 % at 6; chat 49 % at 4 and 42 % at 6), tokens per round from the counters of
bench.py's own runs. Data notes: bench's **code and json prompts are specbench's** (`specbench.py` imports `DECODE_TEXTS`), and the counters are identical (89 rounds, 5.7528 tokens a round;
115 rounds, 4.4522); **structured's counters (153 rounds, 358 accepted, 3.3464) equal specbench's reasoning at fixed depth 7 to the digit**, which is why the plan's mapping of structured to reasoning is
kept; prose has no per-position record: its depth is the point mass at its mean drafted depth (3.65), [INFERENCE].

Two window sources are compared, both gated:

* **DecodeFuse's real-id one-stream table** (1 / 4 / 8 rows = 19.28 / 31.99 / 41.93 ms at 4K; slot 0's text for every kind), plus the real-text proposal 6.90 ms;
* **the byte model** of `C4-DECODE-PLAN.md` (O = 4.72 ms, eff = 0.997 fitted on the serial c=1/2/4 steps; kind-specific routes), plus 6.90 ms.

**12. decode_mean4_tok_s at c=1 (bench decode: prose, code, structured, json; one stream; drafter on; fuse-decode, engine 24a04af)**
| kind       | tok/s (median of 3) | tokens/round | rows/round | round ms | window ms (byte model) | proposal ms | residual ms (host) |
|------------|---------------------|--------------|------------|----------|------------------------|-------------|--------------------|
| prose      |                42.7 |         1.54 |       4.65 |     36.1 |                   30.4 |         6.9 |               -1.2 |
| code       |               125.6 |         5.75 |       7.93 |     45.8 |                   37.7 |         6.9 |               +1.2 |
| structured |                75.0 |         3.35 |       6.90 |     44.6 |                   35.2 |         6.9 |               +2.5 |
| json       |                90.0 |         4.45 |       7.89 |     49.4 |                   41.0 |         6.9 |               +1.6 |

mean4 = 83.32; the residual (host, commits, drafter commit) at c=1 is 1.8 ms on code / structured / json (prose: -1.2: the model is high there)

prediction of the four rates (tokens per round from the counters; the round = window + proposal over the shipping policy's depth distribution):

| window source                            |         prose |           code |    structured |          json |          mean4 |
|------------------------------------------|---------------|----------------|---------------|---------------|----------------|
| DecodeFuse real-id table (kind-agnostic) | 38.8 (-9.1 %) | 118.2 (-5.9 %) | 70.9 (-5.4 %) | 91.6 (+1.8 %) | 79.88 (-4.1 %) |
| byte model (kind routes)                 | 41.3 (-3.3 %) | 128.9 (+2.7 %) | 79.5 (+6.1 %) | 93.0 (+3.3 %) | 85.69 (+2.8 %) |

byte roofline of a drafted round at c=1 (window bytes + drafter weights at 233.8 GB/s, best depth, kind acceptance from the fixed-depth-7 histograms):

| kind       | best depth | tok/s at the byte floor | measured / floor |
|------------|------------|-------------------------|------------------|
| prose      |          1 |                    79.7 |             54 % |
| code       |          7 |                   167.7 |             75 % |
| structured |          3 |                   108.4 |             69 % |
| json       |          7 |                   119.1 |             76 % |

mean4 at the byte floor: 118.7 tok/s (prose at its serial floor 71.4: 116.6); today 83.3 = 70 % of it; 100 needs 84 %

single levers (each alone, on the measured rounds): estimate / bound

| lever                                                                        | mean4 estimate (delta) | mean4 bound (delta) |
|------------------------------------------------------------------------------|------------------------|---------------------|
| prose never drafts (serial 52.5 with probe rounds / 53.5)                    |          85.77 (+2.45) |       86.03 (+2.70) |
| proposal 6.9 -> 5.8 ms (DecodeFuse's named items) / 5.59 ms (its byte floor) |          85.41 (+2.09) |       85.84 (+2.51) |
| proposal with a changed drafter: 4.5 ms / 2.94 ms (table read once)          |          88.02 (+4.70) |       91.38 (+8.05) |
| host residual halved / removed                                               |          84.65 (+1.33) |       86.04 (+2.71) |
| O: -0.85 ms / -1.7 ms (half / all of DecodeFuse's 1.7 ms glue chain)         |          84.93 (+1.61) |       86.60 (+3.28) |
| bandwidth 237 GB/s probe / 258 GB/s stream                                   |          84.29 (+0.96) |       89.23 (+5.90) |
| depth re-optimised on the model window                                       |          83.34 (+0.02) |       83.34 (+0.02) |
| copy hybrid, half / oracle gain (structured +6 %, json +8 % tokens a round)  |          84.79 (+1.47) |       86.25 (+2.93) |

cumulative (levers added in this order):

| + lever                                           | mean4 estimate | mean4 bound |
|---------------------------------------------------|----------------|-------------|
| prose never drafts (serial 52.5 with probe rounds |          85.77 |       86.03 |
| proposal 6.9 -> 5.8 ms (DecodeFuse's named items) |          87.53 |       88.14 |
| host residual halved                              |          88.92 |       91.01 |
| O: -0.85 ms                                       |          91.02 |       95.50 |
| bandwidth 237 GB/s probe                          |          92.14 |      103.13 |
| depth re-optimised on the model window            |          92.16 |      103.14 |
| copy hybrid, half                                 |          93.74 |      106.68 |

the proposal's byte floor with the same drafter arithmetic is 5.58 ms (drafter weights 0.34 GB + the folded table re-read 6x 0.62 GB + the unembed 0.35 GB = 1.31 GB at 233.8 GB/s); a drafter that reads the table once would floor at 2.94 ms. With the changed drafter: mean4 96.2 at 4.5 ms (every other lever at its estimate), 113.4 at 2.94 ms (every other lever at its bound)

per kind after all A levers: estimate prose 55.7, code 134.6, structured 84.3, json 100.3; bound prose 63.7, code 149.2, structured 98.1, json 115.7

with every other lever at its estimate, mean4 = 100 needs: a proposal of 2.64 ms (estimate 5.8, byte floor 5.58), or the window's fixed part cut by 3.06 ms (estimate 0.85), or an effective bandwidth of 258 GB/s (estimate 237)

acceptance needed if the A lane stops at the estimate: tokens per round of code / structured / json x f:

| tokens per round | mean4 with the A estimate | mean4 with only the prose policy |
|------------------|---------------------------|----------------------------------|
| x1.00            |                      93.7 |                             85.9 |
| x1.02            |                      95.3 |                             87.4 |
| x1.04            |                      96.9 |                             88.8 |
| x1.06            |                      98.5 |                             90.3 |
| x1.10            |                     101.7 |                             93.2 |

B lane at c=1 (on top of the A estimate / the A bound; window-byte saving of the 1 x 8 counting-text fixture window, indicative):

| lever                            | routing mass dropped | mean4 on the estimate | mean4 on the bound |
|----------------------------------|----------------------|-----------------------|--------------------|
| skip below 0.15 x top weight     |                1.7 % |                  95.7 |              109.0 |
| skip below 0.2 x top weight      |                4.3 % |                  98.4 |              112.1 |
| skip below 0.3 x top weight      |               12.7 % |                 105.7 |              120.7 |
| skip below 0.5 x top weight      |               34.6 % |                 126.6 |              145.5 |
| experts at 3.5 bpw (4.238 today) |                    - |                 104.0 |              118.7 |
| experts at 3.0 bpw (4.238 today) |                    - |                 112.3 |              128.6 |

Reading it. **Residuals** (prediction / measured - 1): table -9.1 / -5.9 / -5.4 / +1.8 % (mean4 -4.1 %), byte model -3.3 / +2.7 / +6.1 / +3.3 % (mean4 +2.8 %); all within 10 % (`--validate`).
The table is kind-agnostic (slot 0's text at 4K context, bench's contexts are 0.3-0.8K) and over-predicts the round by 2.5-3 ms on code and structured; the byte model, kind by kind, leaves a
**residual of +1.2 / +2.5 / +1.6 ms on code / structured / json (prose -1.2): the host, commits and the drafter's commit at c=1 are about 1.8 ms**, not the 2.8 ms of the c=4 plan (that figure was read against
the old 8.3 ms proposal). Consequences: (1) the served 8-row window is already close to the byte model plus O (code: window 37.7 + proposal 6.9 + 1.2 = 45.8 ms measured); DecodeFuse's
"window cost cut" has nothing left beyond O and the bandwidth levers; its stage table (MoE +10.2 of the +13.2 ms from 1 to 8 rows) is for synthetic ids (29.5 ms at 8 rows) while a real
window grows +22.6 ms, which is the experts (4.3 GB more); (2) the break-even for the real grid is **2.53 tokens a round at 8 rows and 2.02 at 4 rows** ((window + 6.9) / 19.28): prose (1.54) never pays
at any depth, code (5.75), json (4.45) and structured (3.35) pay at 7, 7 and 3-6.

**Byte roofline per kind** (window bytes + the drafter's 1.31 GB per proposal, 233.8 GB/s, best depth, acceptance from the fixed-depth-7 histograms): code 168, json 119, structured 108 (depth 3), prose 80
(or its serial floor 71.4); **mean4 at the byte floor 118.7 (116.6 with prose serial)**. Today is 70 % of it; 100 needs 84 %; the A lane's estimate reaches 79 %, its bound 90 %.

## 2. Ceilings per lever (mean4 deltas, each alone, then added in the order of the table)

(See the two tables in section 12 above.) Ranked by what each is worth at its estimate:

| # | lever | mechanism | estimate | bound | evidence | who carries it |
|---|---|---|---:|---:|---|---|
| 1 | proposal, same drafter | DecodeFuse's named items (fused argmax, L2 policy, global attention): 6.9 -> 5.8 ms; floor 5.58 | +2.1 | +2.5 | section 0 | DecodeFuse (items listed as "not done" in `fuse-decode`) |
| 2 | prose never drafts | the policy's price: with the real grid the break-even is 2.5 tokens a round, prose has 1.54 at every depth (chat hist at depth 1: 49.7 tok/s drafted vs 51.9 serial); serial 52.5 (with probe rounds [INFERENCE]) / 53.5 | +2.45 | +2.70 | `c1_best_depth('prose') = 0` (tested) | **nobody**: the engine's `round_cost` grid (`ilp/engine.rs` `WINDOW_MS`, lab ids) and proposal prices still say draft; DecodeFuse produced the real table, adopting it is an engine change (RoundOverhead's branch edits the same file: +8 lines) |
| 3 | bandwidth | EXL3 / MoE at the 237 GB/s probe, 258 GB/s plain stream | +0.96 | +5.90 | `kernels-moe`: 215 GB/s on expert bytes | a kernel lane (none today at c=1; c=4 plan A4) |
| 4 | fixed part O of a window | DecodeFuse's 1.7 ms glue chain: half / all | +1.61 | +3.28 | `fuse-decode` README | DecodeFuse |
| 5 | copy hybrid (structured, json) | oracle gain structured +6.2 %, json +7.8 % tokens a round (copy alone 1.32 / 1.74 vs block 3.34 / 4.48); half | +1.47 | +2.93 | c=4 plan 5.3 | **nobody** (`auto` runs copy only when the block drafter returns nothing) |
| 6 | host residual | chained round (RoundOverhead): halved / removed | +1.33 | +2.71 | residual +1.2 / +2.5 / +1.6 ms | RoundOverhead; `chain_round` and `draft_graphs` are not integrated |
| 7 | depth per kind from the real grid | code and json 7, structured 6 are already the optimum (model: +0.02) | +0.02 | +0.02 | `c1_best_depth` | none needed |
| 8 | changed drafter (table read once / quantised) | proposal 4.5 ms / 2.94 ms | +4.70 | +8.05 | section 0; acceptance unmeasured | **nobody** (M4) |

Cumulative, exact drafter: 83.3 -> 85.8 (prose) -> 87.5 (proposal) -> 88.9 (host) -> 91.0 (O) -> 92.1 (bandwidth) -> **93.7** (copy) at the estimates; **106.7** at the bounds. The work items already
named by the lanes (proposal 5.8, host halved, O -0.3, 237 GB/s, prose policy) alone give **90.8** (+7.4). With the changed drafter (4.5 ms): 96.2; at its 2.94 ms floor with every other lever
at its bound: 113.4. **What the lead's read missed**: the proposal cannot reach 4.5 ms with the same drafter (-2.6 on the mean), the host pool is 1.8 ms not 2.8 (-0.7), and the depth lever is nil.

**Result of M4 (DraftSlim, records `draft-slim`, `draft-slim-exact`, engine `aa2f9ad`):** the proposal went from 6.5 / 7.3 / 9.0 ms to **3.3 / 3.8 / 4.8 ms** (1 / 2 / 4 requests, 4K; 4.1 / 5.3 / 7.8 at 32K), below the
5.6 ms "floor" of this section because the Markov steps no longer read the table: a rank-16 principal-axes proxy of the folded table (6.4 MB, L2-resident) picks about 130 candidates a step and only those are evaluated
exactly (a rule with a proof does not exist: the Cauchy-Schwarz bound leaves 199,737 of 200,058 rows; the measured rule misses the full argmax in 1 of 8,379 real steps, tokens a round unchanged), plus a finer
partition of the drafter's attention. `decode_mean4_tok_s` 83.99 -> **90.08** (+7.2 %), `decode_c4_rolling_tok_s` 134.4 -> **141.8** (+5.5 %), replies byte-identical (rank dump, `exactness.py`, `specedge.py` 65/65).
The lever was worth about +1.9 mean4 a ms, not the +0.7 of the table above [INFERENCE: not decomposed]. What is left of the proposal (3.3 ms): the unembed read 1.5 ms (its floor; a proxy scan plus a bit-exact per-row
EXL3 evaluation would remove most of it, not built), the three layers 1.3 ms, the steps 0.4 ms. The engine's `round_cost` proposal prices (6.4 to 8.3 ms) are stale and were not changed.

## 3. Is mean4 >= 100 reachable on the A lane?

* **At the estimates: no, 93.7** (x1.067 short). At the named items alone 90.8.
* **At the bounds: yes, 106.7**, but only with all of: proposal at its 5.58 ms floor, host 0, glue -1.7 ms, a 258 GB/s stream (the plain-stream probe; the EXL3 kernels' own ceiling is 233-237), the copy hybrid at its oracle gain
  and prose never drafted. The single largest term is bandwidth (+5.9), which no lane carries at c=1.
* **What acceptance would have to be** if the A lane stops at the estimate: tokens per round of code, structured and json **x1.079** (code 5.75 -> 6.2, json 4.45 -> 4.8, structured 3.35 -> 3.6); with the changed
  drafter at 4.5 ms x1.046. The block cap binds on code (40 % of rounds accept all seven) and json (38 %), so the gain would have to come from structured / prose / the mid-acceptance rounds: a better drafter
  (`DRAFTER-SURVEY.md`: only the vendor's exists), not a policy.
* **B-lane margin** (on top of the A estimate, 1 x 8 window bytes of the counting-text fixture, indicative): skip below 0.15 x the top weight (1.7 % of the routing mass) 95.7; 0.2 (4.3 %) 98.4;
  0.3 (12.7 %) 105.7; a 3.5 bpw pack 104.0; 3.0 bpw 112.3. **100 needs a window-byte saving of about 11 % (a threshold near 0.25, about 8 % of the routing mass) or a 3.8 bpw pack.** The gates are those of
  `C4-DECODE-PLAN.md` section 6: the recipe's quality set 60/60, the tool gate, the needle, KL / top-1 against the A lane (target: KL below the pack's own 0.0088), measured first offline (B1, oracle
  forward with picks dropped) before any engine work; skipping changes the tokens the model produces, so it is the owner's decision and never a default on its own.

## 4. Ranked experiments (cheap first)

| # | experiment | owner | node, time | gate | notes |
|---|---|---|---|---|---|
| M1 | **real grid in the engine's `round_cost`** (`WINDOW_MS` from `round-cost-table-real-ids-final.txt`, proposal 6.9 ms) so the plan stops drafting prose; `bench decode`, three repeats; the `spec` exactness gate | engine / RoundOverhead (same file) | node-2, 30 min | prose >= 51 tok/s, code / structured / json within 1 % of today, mean4 >= 85.5; drafted == serial | +2.45, no kernel work; also stops all-chat c=4 losing 9 % (`C4-DECODE-PLAN.md` A5) |
| M2 | rank `wait_us` per window of a served drafted c=1 run (`REXL3_INKLING2_LOOP_TIMING=1`) against the client's round | lead | any node, 15 min | residual <= 1 ms: stop `chain_round` (<= +1.3 mean4); >= 2.5 ms: continue | the model's residual is +1.2 / +2.5 / +1.6 ms |
| M3 | the proposal items: fused argmax, folded-table L2 policy, global-layer attention; then the drafter graph inside `chain_round` | DecodeFuse, then RoundOverhead | node-2, 1-2 days | proposal <= 5.8 ms at one request, bit-exact drafts | +2.1 mean4; the two branches both edit `propose` |
| M4 | **drafter arithmetic**: Markov logits with the table read once (candidate-restricted) or quantised; acceptance on the recorded replies against the exact drafter (drafts only; outputs unchanged by construction) | DecodeFuse | node-2, 1 day (drafter fixtures in `oracle/`, engine spec tests) | acceptance within 1 % of the exact drafter and proposal <= 4.5 ms | +4.7 mean4 if it holds; this is the one lever that moves the proposal below 5.6 ms |
| M5 | glue chain: router / controller / mix fusion (stage table in `fuse-decode`) | DecodeFuse | node-2, 1 day | window -0.85 ms or more | +1.6 |
| M6 | MoE and EXL3 bandwidth at 1-8 rows of one stream with real routes (extend `bench_decode_cost_of_bands`, which runs 1 and 4 rows) | kernel lane | node-3, 1 day | experts >= 215 GB/s at 8 rows | +1.0 to +5.9 |
| M7 | copy hybrid in `SPEC=auto` with a selection rule (copy when its match is longer than the block's expected acceptance) | engine | node-2, 1-2 days | structured / json tokens per round +3 % with no loss on code | +1.5; low priority |
| B1 | skip-threshold quality offline (oracle forward, KL and top-1 vs the A lane, thresholds 0.2 / 0.25 / 0.3) | quality lane | node-1 CPU, 16 min a threshold and fixture | KL < 0.0088 at the threshold that gives 100 | needs the owner's approval; same B1 as the c=4 plan |
| B3 | a 3.8 / 3.5 bpw SAGE pack | owner | GPUs, days | quality gates vs the 4.23 bpw pack | owner |

**Already carried**: DecodeFuse (proposal items, glue stage table), RoundOverhead (`chain_round`, host), `-fp8` closed NO-GO (prefill; not a decode lever), PrefillLane (not at c=1). **Carried by nobody**: M1 (engine
price), M4 (drafter arithmetic), M6 (kernel bandwidth at c=1), M7 (copy hybrid), the B lane. **Sequence**: M1 and M2 first (an hour, +2.45 and the size of the host pool), M3 and M5 in parallel, M4 as soon as the
proposal items land (it is the only way past 94), B1 from day one if the owner approves.

## 5. What the owner must decide

1. Accept that 100 on `decode_mean4_tok_s` is a **stretch at the bound on the A lane** (estimate 93.7) and say which of the three closes it: a better drafter / M4 (acceptance), one B-lane lever (about 8 % of the routing
   mass skipped, or 3.8 bpw), or the target lowered to 90-95.
2. Whether M4 (a changed drafter arithmetic, outputs byte-exact, drafts not) belongs to the A lane: the verification makes it so, but the pack's drafter is no longer the one that ships.
3. The B-lane approval (B1 first, offline, no engine change).

## Result note: the copy hybrid and the serial-window chain (CopyHybrid lane, 2026-11-23)

* **Copy hybrid (lever 5, M7): built, +1.49 on mean4.** `decode_mean4_tok_s` 83.92 -> 85.41 (median of three, same binary, `REXL3_INKLING2_LOOP_SPEC_COPY=off|on`, node-4; record `copy-hybrid`, exactness `copy-hybrid-exact`, same engine commit). Structured +3.7 %, json +1.5 %,
  prose and code unchanged; c = 4 aggregate and the rolling metric unchanged within noise (134.4 -> 134.8). The offline bound (+1.47 half / +2.93 oracle) was within reach only because the real joint was better than the independence assumption: where a copy of the stream's own text matched 16 tokens or more it was
  right more often than the block drafter (shadow measurement in the record). A first policy (copy from 8 matched tokens) gave +1.55 but lost 2.7 % on code, so the shipped minimum is 16. The model's `copy=0.5` term stays at its estimate; the lever is now worth +1.5, not more: structured/json only.
  A combined draft (copy extending the block's) is impossible at 8 rows a slot; the proposal is skipped only when every stream copies, so at c = 4 it matters little.
* **Serial-window chain (lever 6 residual): no-go.** The round trace of a serial stream shows 18.65 ms of which 0.05 ms is host turnaround; the 1.5 ms idle of the earlier note was a profiler artefact. A built device-fed chain measured serial c = 1 53.41 -> 53.32, c = 4 100.57 -> 100.17, drafted prose
  50.13 -> 49.52 (record `serial-chain`); it was removed. The +0.9 on mean4 attributed to it in `docs/DECODE-SPEED.md` is struck.

