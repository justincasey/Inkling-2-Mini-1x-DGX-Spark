# oracle-gap: the engine against the fp64 oracle fixtures, whole forward and free-running (gate `oracle-gap`, plan 6.3)

Result **pass** (`meta.json`): all 138 checks hold (17 `tests` entries: per prompt the hidden planes, the logits plane, the router, the greedy tokens; and the pooled router
check). Read what the pass means before relying on it: the limits are *proposed in this record* from the same four prompts (`docs/ORACLE-GAP.md`, "Limits proposed"), the floor-relative checks are not
fitted, and the Small's stock `compare.py` gates are **not** met: `GATES_ENGINE` fails on hidden and h_final cosine on all four prompts and on the logits of three (as on the Small), `GATES_SPEC` fails
`logits` of `chat-capital-effort0` and `chat-sky-effort0.9` on `top1_disagree_outside_ties` (3 rows, oracle margins 0.103, 0.103, 0.107 against its 0.1). No router flip is individually explained by a near
tie: flips are 25.6% of the pairs, as many as the oracle's own two numerics produce (26.7%); the explanation is statistical (`docs/ORACLE-GAP.md`, "Evidence").

Engine `7da8c61344842f4a2605591d22e97f5ad28d1a7e` (lab build of the committed tree, `engine.lab_build` true, no override), node-1, pack `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` at the pinned revision,
`--spec none --prefix-cache off --no-vision`, chunk 4,096, the dump hooks `REXL3_INKLING2_LOOP_DUMP` and `_OPS=1` (a live server, one boot), GPU clock locked at 2,200 MHz.

## What was done

1. One boot with the dump hooks; per fixture (`/home/spark/inkling-2-mini-1x-dgx-spark/fixtures/{spec,engine}`: 25, 26, 35 and 1,119 prompt tokens, 16, 24, 16 and 8 generated, identical tokens in both numerics)
   `greedy_oracle.py --prefill-full` prefilled the fixture's tokens less the last (40, 49, 50, 1,126 rows; the dump) and ran the engine greedy on the prompt. `raw/dump-and-serve.txt`: the twelve dump files
   (names, rows, sha256, bytes), the server log without HTTP lines, the memory guard's minimum. The dumps of the 26- and 35-token prompts are byte-identical to the dumps of the forward-oracle record (sha256
   `c25f9876...`, `0773323c...`); the 1,119-token prompt prefilled twice gives the same 3,875,832,200 bytes (`288f2376...`).
2. `raw/run_gap.sh` (on node-1, oracle venv, CPU, 20 s for all twelve comparisons): `oracle/engine_gap.py` per prompt for engine dump vs engine fixture, engine dump vs spec fixture, engine fixture vs spec fixture
   (the *floor*); reports in `raw/reports/*.json.gz`; `bench/tools/oracle_gap_summary.py` joined them (`raw/summary.json`, `raw/tables.md`).
3. `raw/row_probe.txt`: rows 3 to 7 of `chat-capital-effort0`, rows 5 and 23 of `chat-sky-effort0.9` (`raw/row_probe.py`).

## Headline numbers (`meta.json` `metrics`)

| metric | value |
|---|---|
| `prompts_compared` | 4 |
| `long_prompt_tokens` | 1119 |
| `rows_compared` | 1265 |
| `hidden_rel_l2` | 0.259781 |
| `logits_rel_l2` | 1.043139 |
| `logits_rel_l2_plane` | 0.102544 |
| `kl_mean_worst` | 0.007344170425440772 |
| `argmax_equal_fraction` | 0.9976284584980237 |
| `oracle_numerics_argmax_equal_fraction` | 1.0 |
| `router_flips` | 15526 |
| `router_flips_vs_engine_fixture` | 15698 |
| `router_flips_floor` | 16207 |
| `router_pairs` | 60720 |
| `router_flip_fraction` | 0.25569828722002635 |
| `router_flip_fraction_floor` | 0.2669137022397892 |
| `visit_curve_vs_floor_max` | 1.1029 |
| `greedy_tokens_equal` | 64 |
| `greedy_tokens_total` | 64 |

`hidden_rel_l2` is the worst whole-plane relative L2 of any visit's `hidden`, over the four prompts and both oracles; `logits_rel_l2` the worst single row's (the long prompt's row 689 against the engine fixture), `logits_rel_l2_plane` the worst whole plane's;
`argmax_equal_fraction` the lower of the two oracles' pooled fractions over 1265 rows (3 differ against each; the oracle's two numerics differ on 0: `oracle_numerics_argmax_equal_fraction`); `router_flips` the flips of the
engine dump against the spec fixture, `router_flips_floor` those of the engine fixture against the spec fixture; `greedy_tokens_*` the generated positions against the spec fixtures (the engine fixtures' tokens are the same).

## Tables (`raw/tables.md`)

| prompt | rows | comparison | hidden rel L2 (worst visit) | h_final rel L2 | logits rel L2 plane / row median / row max | logits cos min | KL mean / max | argmax equal | top-2 equal | flips of pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| chat-capital-effort0 | 40 | engine vs engine fixture | 0.11 (47) | 0.0774 | 0.0298 / 0.0206 / 0.176 | 0.98646 | 0.00734 / 0.0822 | 39/40 | 36/40 | 238 / 1920 (12.4%) |
|  |  | engine vs spec fixture | 0.0637 (40) | 0.062 | 0.0259 / 0.0188 / 0.198 | 0.98454 | 0.00691 / 0.0795 | 39/40 | 36/40 | 245 / 1920 (12.8%) |
|  |  | floor: engine fixture vs spec fixture | 0.126 (47) | 0.0728 | 0.0253 / 0.0108 / 0.0776 | 0.99863 | 0.00277 / 0.0512 | 40/40 | 36/40 | 182 / 1920 (9.5%) |
| chat-sky-effort0.9 | 49 | engine vs engine fixture | 0.0463 (48) | 0.0506 | 0.0191 / 0.0137 / 0.0864 | 0.99724 | 0.00411 / 0.0678 | 47/49 | 44/49 | 213 / 2352 (9.1%) |
|  |  | engine vs spec fixture | 0.0489 (37) | 0.0462 | 0.017 / 0.0148 / 0.0943 | 0.99599 | 0.00493 / 0.0654 | 47/49 | 46/49 | 229 / 2352 (9.7%) |
|  |  | floor: engine fixture vs spec fixture | 0.0433 (40) | 0.0451 | 0.0166 / 0.00972 / 0.0952 | 0.99589 | 0.00246 / 0.0512 | 49/49 | 45/49 | 204 / 2352 (8.7%) |
| raw-counting | 50 | engine vs engine fixture | 0.178 (24) | 0.0491 | 0.0247 / 0.016 / 0.172 | 0.98772 | 0.000283 / 0.00348 | 50/50 | 50/50 | 420 / 2400 (17.5%) |
|  |  | engine vs spec fixture | 0.159 (36) | 0.0475 | 0.0239 / 0.0173 / 0.0994 | 0.99505 | 0.000268 / 0.0028 | 50/50 | 50/50 | 463 / 2400 (19.3%) |
|  |  | floor: engine fixture vs spec fixture | 0.168 (36) | 0.0459 | 0.0253 / 0.014 / 0.188 | 0.98747 | 0.000166 / 0.00152 | 50/50 | 50/50 | 407 / 2400 (17.0%) |
| raw-long-window | 1126 | engine vs engine fixture | 0.256 (36) | 0.111 | 0.103 / 0.08 / 1.04 | 0.87126 | 0.000718 / 0.0923 | 1126/1126 | 1112/1126 | 14827 / 54048 (27.4%) |
|  |  | engine vs spec fixture | 0.26 (36) | 0.0829 | 0.0786 / 0.0582 / 0.56 | 0.90058 | 0.000231 / 0.012 | 1126/1126 | 1110/1126 | 14589 / 54048 (27.0%) |
|  |  | floor: engine fixture vs spec fixture | 0.267 (36) | 0.106 | 0.0897 / 0.0735 / 0.561 | 0.88274 | 0.000681 / 0.016 | 1126/1126 | 1115/1126 | 15414 / 54048 (28.5%) |

| prompt | engine / floor, per visit 1..48 (max at visit; min) | flips that recur in the floor (engine vs spec): share vs the floor's rate | smallest router margin | weights, max abs diff (identical sets) |
|---|---|---|---|---|
| chat-capital-effort0 | 2.19 (visit 8); 0.35 | 111/245 = 45.3% vs 9.5% | 1.31e-05 | 0.27 |
| chat-sky-effort0.9 | 1.29 (visit 13); 0.74 | 126/229 = 55.0% vs 8.7% | 3.13e-07 | 0.299 |
| raw-counting | 1.32 (visit 31); 0.58 | 289/463 = 62.4% vs 17.0% | 2.14e-08 | 0.552 |
| raw-long-window | 1.09 (visit 5); 0.76 | 11371/14589 = 77.9% vs 28.5% | 1.67e-07 | 1.12 |

| prompt | generated | engine tokens vs spec fixture's | vs engine fixture's | oracle top-1 - top-2 margin at the generated positions: min (position); positions under 0.3 |
|---|---|---|---|---|
| chat-capital-effort0 | 16 | equal | equal | 0.103 (11); 2 of 16 |
| chat-sky-effort0.9 | 24 | equal | equal | 0.0688 (5); 1 of 24 |
| raw-counting | 16 | equal | equal | 3.66 (8); 0 of 16 |
| raw-long-window | 8 | equal | equal | 2.41 (6); 0 of 8 |

| oracle margin of the row | [0, 0.05) | [0.05, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.5) | [0.5, 1) | [1, inf) |
|---|---|---|---|---|---|---|---|
| rows (spec fixture's margins, all prompts) | 0 | 3 | 7 | 2 | 6 | 5 | 1242 |
| argmax differs, engine vs engine fixture | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| argmax differs, engine vs spec fixture | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| argmax differs, floor: engine fixture vs spec fixture | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Not proven

Positions past 8,192; the long prompt op by op (its ops dump is on the node); `kv8` alone; an oracle that differs only in fp32 summation order; decode windows (the dump is prefill); the vision tower and the drafter. Four prompts, 1,265 rows.
See `docs/ORACLE-GAP.md`, "Not measured".

## Node footprint

`/home/spark/inkling-2-mini-1x-dgx-spark/lab/fixgate/` on node-1: `dump/` (12 files, 12.4 GB, the dumps above), `oracle/` (a copy of this repository's `oracle/` at `engine_gap.py`'s version), `out/` (the reports), `run_gap.sh`, `row_probe.py`, `gap.log`;
the engine installed under `/home/spark/inkling-2-mini-1x-dgx-spark/{bin,src,target}` by `inkling2mini.sh build`. The server is stopped; nothing of the Small was touched.
