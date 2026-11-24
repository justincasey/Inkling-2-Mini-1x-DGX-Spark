# HARNESS-NEXT: the segment-3 `autoresearch.sh` for the Mini

Design only. Nothing here runs: no engine serves the model yet, so there is nothing to benchmark and the harness is not
written. Segment 1 of `autoresearch.sh` (today) is the Mac-only bring-up ladder; segment 2 is the node bring-up by
hand (`docs/EVAL-MATRIX.md`, plan Phase 6); this document is segment 3, the speed loop of plan Phase 7. The lead writes
the script after plan step 4.8 (the engine pinned as its own byte-exact reference) and after a first baseline record.
The model is the Small's `autoresearch.sh` (333 lines, 11 gates, one score), one node instead of four ranks.

## 1. What it measures

One candidate engine commit (the HEAD of `CANDIDATE_SRC`, default the engine worktree), one fixed workload, printed as
`METRIC name=value` lines, plus one `PASS <id>` or `FAIL <id>` line per gate. The primary metric, higher is better:

```
target_score = sqrt(decode_c1_tok_s / T_d  x  prefill_32k_tok_s / T_p)  x  min(1, lanes_at_max_context / L)
```

It prints 0 unless every required gate printed its exact whole line `PASS <id>`; `target_score_raw` keeps the number.
`bench/tools/score.py` implements exactly this (`--metrics`, `--gates`, `--required`, `--screen`, and the three targets
as options); `tests/test_bench.py` pins its behaviour.

| Symbol | Value | Source | Metric of the run |
|---|---:|---|---|
| `T_d` | 62 tok/s | `docs/ROOFLINE.md` "Targets": decode c = 1, 4K, serial (95% of the f = 1.08 figure) | `decode_c1_tok_s`: mean of the four per-type rates of `bench.py` decode (prose, code, structured, json; 512 greedy tokens each; the drafter as booted), rates from `usage.completion_tokens` and the server's own `rexl3_request_ttft_seconds` / `rexl3_request_e2e_seconds` |
| `T_p` | 2,120 tok/s | `docs/ROOFLINE.md`: prefill cold, 32K (25% of the bf16 ceiling 8,481) | `prefill_32k_tok_s` = prompt tokens / server TTFT of one cold 32,768-token prompt |
| `L` | 4 lanes | `docs/ROOFLINE.md` / `docs/FIT.md`: profile `max`, four lanes of 1,048,576 | `lanes_at_max_context` from the engine's capacity line (`event=..._plan`) |

All three are proposals until the owner confirms the targets (plan 7.1; decision D3, targets, is open); `score.py` carries them as
constants and options, so confirming them is a one-line change plus the pinned test. Two honest caveats:

- `T_d` is defined at 4K and serial in `ROOFLINE.md`; `bench.py`'s four decode prompts are a few hundred tokens and run
  with the drafter on when `SPEC` is not `none`. The segment-3 harness either keeps the Small's definition (the metric
  moves with acceptance) or boots serial for the scored decode: decide at the first baseline, then freeze it. The 4K
  point of `bench/rexl3/sweep_decode.py` is the like-for-like number for the record either way.
- `L` can only be 4 when the harness boots profile `max` (`docs/FIT.md`: 72.56 GiB KV, 0.5 GiB under the pool). A harness
  that boots `bootstrap` (262,144 x 4) sees lanes 4 as well but at a quarter of the window; the Small's harness scored
  the full-window profile, and so should this one. Needles and the capacity gate then run at 1,048,576.

Secondary metrics printed with it: every raw metric of `bench.py`, `boot_s`, `kv_tokens`, `kv_gib`, `free_after_gib`,
`prefix_gib`, `mem_min_gib` (and `_norm` when a recorded prefix size is given), `gpu_clock_mhz_*`,
`min_mem_available_gib_sampled`.

## 2. Required gates

`REQUIRED_GATES="pin spec cache chunk vision needle capacity memory smoke host goldens"` (the Small's ids, plan 7.1;
`score.REQUIRED_GATES` holds the same list). Each maps to a Phase 6 gate of the plan and has one tool behind it, all of
them already built (plan 5.5) and callable today as `./inkling2mini.sh bench PHASE` or directly.

| Gate id | Phase 6 | Tool and command | Pass condition |
|---|---|---|---|
| `pin` | 6.4 (boot to boot) | `bench/tools/bench.py --phases ... --ref bench/reference.json` | every reply (decode x4, concurrent x4 and x8, cold 32K / 128K / 224K) equals the pinned reply byte for byte, ids and finish reason; the prompt hash matches; concurrent == solo. The reference was written by one boot, so every later boot reproducing it is the boot-to-boot check |
| `spec` | 6.4 (drafted == serial) | `bench/tools/gates.py spec OUT` (`--serial-boot` when `SPEC=none`) | serial (`"draft": false`) and drafted replies equal on short, medium, long, xlong and a 70K prompt; the four of them at once equal their solo serial replies; serial replies drafted 0; the drafted ones drafted > 0 |
| `cache` | 6.4 (cached == uncached) | `bench/tools/gates.py cache OUT` | cold and warm cached replies equal the uncached ones; 0 cached tokens on every uncached and cold reply; the warm and shared-prefix replies served from the cache where the prompt reaches the smallest cut |
| `chunk` | 6.4 (chunk width) | `bench/tools/exactness.py record` on boot A (`PREFILL_CHUNK=4096`) and on boot B (`PREFILL_CHUNK=1000`), then `exactness.py diff` | the two records' replies identical |
| `vision` | 6.2 and 6.4 | `bench/tools/gates.py vision OUT` then `gates.py diff bench/reference-vision.json OUT` | text and image replies equal the pinned vision reference byte for byte (a `VISION=off` boot compares text alone and says so) |
| `needle` | 6.5 | `bench/tools/needle.py --tokens $MAX_MODEL_LEN --max-model-len $MAX_MODEL_LEN --depth 0.1` and `0.5` | the code is found at both depths at the profile's window |
| `capacity` | 6.1 | the capacity line of the boot (`bench_capacity` in the controller) | `lanes_at_max_context` >= the profile's lanes and `kv_tokens` >= `KV_TOKENS`, recorded against `docs/FIT.md` |
| `memory` | 6.1 | `./inkling2mini.sh memwatch begin` before the run, `report` after, `bench/tools/memcheck.py` | the guard's lowest MemAvailable since it armed (loading included) >= the engine floor (8 GiB, or `MEMORY_FLOOR_GIB`); the guard still running |
| `smoke` | 6.2 | `./inkling2mini.sh smoke` (`bench/tools/smoke.py`) | chat, completion, tool call, reasoning on and off separated, image (not with `VISION=off`) |
| `host` | engine tests | `./inkling2mini.sh engine-test host` | every suite runs at least one test, none fails or is skipped |
| `goldens` | engine tests | `./inkling2mini.sh engine-test goldens`, after `testdata/gpu` is shown byte-identical to the pinned engine's | every golden passes; "not run" fails |

Phase 6 gates that are **not** required by the score, and why: 6.3 the oracle (op-level teacher forcing on the node:
`oracle/compare.py`, `bench/tools/greedy.py`; it belongs to bit-changing work, like the Small's `oracle` gate, and is
run by hand per `docs/ORACLE-GAP.md`); 6.6 the unknowns (route logs and acceptance: analysis, decides the MTP policy);
6.7 quality and tools (`bench/quality_gate.py`, `bench/toolgate.py`, `scripts/loop_probe.py`: baselines, recorded once
per pack and again only when numerics change); 6.8 `tests/api_conformance.py` and `bench/probe.py` (client contract);
6.9 the sweeps (`bench/rexl3/sweep_decode.py`, `sweep_prefill.py`, `profile_report.py`: records against
`docs/ROOFLINE.md`, not a loop gate); 6.10 `bench/longctx.py` and SixCat. A change that moves numerics (segment B)
re-runs 6.3 and 6.7 by hand before its pin is accepted. If the loop later needs one of them as a gate, add its id to
`REQUIRED_GATES` and to `score.py` together, with a test in `tests/test_bench.py`.

## 3. Phases

`PHASES` is a comma list; unset means a full run: every phase below except `prefill-1m`. Order matters (the chunk record
comes before anything that reads the prefix cache, so boot A's is as cold as boot B's).

| Phase | Runs | Needs a boot |
|---|---|---|
| `host`, `goldens` | `engine-test host|goldens` with the engine stopped, from the installed source | no |
| boot A | `PREFILL_CHUNK` as in `.env.example` (4096); `drop_caches` first so the capacity plan reads the same free memory; capacity line; `memwatch begin`; a 5 s sampler of GPU clock, utilisation and MemAvailable | yes |
| `smoke` | the smoke tool | A |
| `chunk` (first half) | `exactness.py record` at 4096 | A |
| `warm`, `decode`, `conc`, `rolling`, `prefill-32k`, `prefill` | `bench.py` (see its header): warm-up discarded, c = 1 over four prompts, c = 4 (the span number) and 8 (8 requests on 4 slots), the rolling c = 4 (section 7), cold 32K, then 128K and 224K | A |
| `vision`, `spec`, `cache` | `gates.py` | A |
| `needle` | `needle.py` at the window | A |
| `prefill-1m` | one cold 1M prompt, about 25 minutes on the Small: never part of a full run, `PHASES=prefill-1m`; a screen | A |
| boot B | only for `chunk`: `PREFILL_CHUNK=1000`, `exactness.py record`, diff against boot A's | B |

**A screen** is any run with `PHASES` set: only those phases run, only their gates print, `target_score` is 0 (the raw
number is printed with `score.py --screen`), and a screen never promotes a candidate in `scripts/safeloop.sh`. The
quick screens: `PHASES=warm,decode` (decode only), `PHASES=prefill-32k` (the cold 32K prefill alone). `PIN=1` is not a
screen but a pin run: it needs the unmodified pinned engine (`REXL3_COMMIT`), writes or extends `bench/reference.json`
and `bench/reference-vision.json` (an entry already pinned must be reproduced byte for byte or the file stays unchanged),
and prints no score. `./inkling2mini.sh bench pin` is the same operation for the interactive case.

## 4. Fixed settings and rules (as the Small's)

- The serve configuration is `.env.example` only: the script copies it over `.env` and drops every variable of the
  caller's shell, so runs stay comparable. An experiment that changes the configuration changes that file or the engine.
- The GPU SM clock cap (`GPU_CLOCK_CAP_MHZ`, 2,200 MHz, `config/pins.sh`) is hardware protection and non-negotiable: `start`, `build` and every
  engine test apply it (`sudo -n nvidia-smi -lgc`, success line required) and the harness samples the clock during the run; a sample above the
  cap, or no sample, fails the run. No mode skips, releases or changes the cap.
- The node must be free: `stop-other` first if the Small owns it, `restore-other` when the session ends; the harness
  stops only its own engine, before and after. `scripts/safeloop.sh run` wraps a candidate so the last known-good engine is
  restored after any outcome; `safeloop.sh cmd` does the same for lab commands (`scripts/logits_check.sh`,
  `scripts/loop_probe.sh`).
- The candidate is built with `./inkling2mini.sh build <sha>` from a clean engine tree (the build ships the commit),
  unless the installed engine already is that commit.
- The script body is functions and one call on the last line, so replacing it during a run cannot change that run.
- Off limits to the loop: `oracle/`, `bench/reference*.json` once pinned, the thresholds in `bench/tools/gates.py`, the
  pins in `config/pins.sh`, the memory guard's floors (`scripts/memguard.sh`, `GUARD_*`, `MEMORY_FLOOR_GIB`), the targets
  in `bench/tools/score.py`, and `docs/ROOFLINE.md`.

## 5. Reading a run

Each run writes `bench/runs/autoresearch-<stamp>/` (git-ignored): `gates.txt` (the `PASS`/`FAIL` lines), `metrics.txt`,
one log per gate id, `bench-<label>.json` (every reply and timing), `samples.txt`, `server.log` (last 300 lines). The last
output lines are the `METRIC` lines and an `ASI engine=<sha12> boot_s=<n> failed_gates=<ids|none> run=<dir>` line; exit 0
when every gate that ran passed. A passing run worth keeping becomes a record through `./inkling2mini.sh record NAME`
(plan Phase 6 rules: engine commit clean, `dirty` false).

## 6. Open items before it is written

1. Confirm `T_d`, `T_p`, `L` (D3) and whether the scored decode is serial or drafted (section 1).
2. 4.8: the engine pinned at `REXL3_COMMIT`, then `bench pin` writes `bench/reference.json` and `reference-vision.json`.
3. The capacity line's field names (`lanes_at_max_context`, `kv_tokens`, `prefix_gib`) are the Small family's; confirm
   them in `inkling2_loop` or adjust `bench_capacity` and `memcheck.py`.
4. Whether the dump variables of `scripts/logits_check.sh` (`REXL3_INKLING2_DUMP`, `..._DUMP_LOGITS_ONLY`) exist in the
   loop family; the script fails ("no logits file") rather than pass without them.
5. Wall time of a full run is unknown until a baseline exists; set the per-gate timeouts from it (the Small's: 900 s
   smoke, 7,200 s spec, 10,800 s cache, 7,200 s needle) rather than from this document.

## 7. The c=4 record (gate `speed-c4`)

Why a second c=4 number: `decode_c4_tok_s` of `bench.py` divides all tokens by the span to the last reply, so with the
drafter the last stretch runs one stream (`docs/C4-DECODE-PLAN.md` 1.2: the specbench mix finishes at 9.6 / 11.5 / 14.0 /
18.4 s, span 111 tok/s against about 152 steady). `decode_c4_rolling_tok_s` is the sustained figure (plan 1.3); the span
number stays for continuity. The ladder reads `decode_c4_rolling_tok_s` from the newest `speed-c4` record and pairs it
with the newest passing `spec` exactness record of the same engine commit (`--same-engine spec`), so a speed figure is
never taken from an engine whose drafted output was not shown equal to serial.

What the phase is (`bench.py` header has the definitions): four workers in a closed loop, worker w running the kinds
prose, code, structured, json back to back from kind w, twice (8 requests a worker, 32 in all, 512 greedy tokens,
`prefix_cache: false`), the next request issued the instant the previous one ends. Every rolling reply must equal the
solo reply of its kind (a difference fails, as in `conc`); a short reply (finish reason not `length`, or tokens other
than 512) is a `FAIL` in a `--ref` or `--pin` run and a warning in a `--raw` run, and is counted in
`decode_c4_rolling_short_replies`. `decode_c4_steady_tok_s` covers the window in which all four workers are active and is
left out (with a note) when that window is under 10 s. The client does not stream, so the window's start uses each first
request's start plus the solo ttft of its kind from the `decode` phase of the same invocation; the raw JSON says which.
The plan's expected values (99.8 tok/s serial, about 129-130 rolling drafted) are its derived model figures, not measurements.

Making the record (node window, engine at the commit under test, shipping defaults `SPEC=dspark PREFIX_CACHE=on VISION=on`,
GPU clock locked, nothing else on the node; warm-up first, one pass discarded after boot):

```
./inkling2mini.sh bench decode                 # same boot: the span number, decode_c4_tok_s, and the solo replies
./inkling2mini.sh bench rolling                # warm, decode, rolling: the rolling metrics, raw JSON in bench/runs/<stamp>-rolling/
./inkling2mini.sh record speed-c4              # creates bench/records/<UTC date>-speed-c4/ (meta.json started, README.md to write)
python3 bench/tools/finalize_record.py bench/records/<UTC date>-speed-c4 SPEC.json
```

`SPEC.json` (the keys `finalize_record.py` requires: `gate result tests metrics commands notes`):

```
{"gate": "speed-c4", "result": "pass",
 "tests": [{"bin": "bench/tools/bench.py", "filter": "warm,decode,rolling: 4 solo and 32 rolling replies, each equal to its solo reply", "passed": 36, "failed": 0}],
 "metrics": {"decode_c4_tok_s": <span number, bench decode>, "decode_c4_rolling_tok_s": <bench rolling>,
             "decode_c4_steady_tok_s": <bench rolling; omit the key if the run printed none>,
             "decode_c4_rolling_requests": 32, "decode_c4_rolling_short_replies": 0},
 "commands": ["./inkling2mini.sh bench decode", "./inkling2mini.sh bench rolling"],
 "notes": "<SPEC, PREFIX_CACHE, VISION, profile, GPU clock; the steady window and its first-token estimate; any pass discarded>"}
```

`result` is `pass` only when `bench rolling` printed `bench rolling: pass` (no short reply, every reply equal to its solo
one); the threshold on `decode_c4_rolling_tok_s` belongs to the ladder rung, not to the record. Copy the printed metric
values into `metrics` as numbers; put the `bench/runs/<stamp>-rolling/` files the README quotes into the record directory
(`bench-rolling.out`, `bench-rolling.json`) so every figure has its raw output. No record exists yet: none was made
(this section is desk work).

## 8. The final run (`scripts/final_run.sh`)

The last engine head is pinned (`REXL3_COMMIT`), every node gate is re-run on that one build, and the build is left for the owner to test.
`scripts/final_run.sh` does all of it from this machine, at any pin and on any node:

```
scripts/final_run.sh [--continue] [--resume RUN_DIR]      # NODES and NODE_NAME (node-1..node-4) in .env; a clean recipe tree
```

It refuses (before touching the node) without a node, a record node name, a pinned `REXL3_COMMIT` that is in the engine repository
(`REXL3_SRC`), or a clean tracked tree; it refuses to continue when the engine installed after `build` is not the pin (read from `doctor`
before every boot). Phases, each with its wall time in `bench/runs/final-<stamp>/stages.tsv`:

| Phase | Commands (all through `./inkling2mini.sh`, which applies the GPU cap and requires its success line) | Records |
|---|---|---|
| build | `doctor`, `build` (the pin), `engine-test host`, `engine-test goldens` | `final-engine-tests` (gate `engine-build`) |
| boot p (pin) | `start`, `bench exactness`, `smoke`, `bench pin` (writes `bench/reference.json` and `reference-vision.json` when absent, reproduces them when present) | `final-boot`, `final-smoke` |
| boot g (gates) | a fresh `start`; `bench exactness`; `bench all`; `bench cache-images`; `bench prefill-sweep`; `bench decode-4k`; `FINAL_PASSES - 1` more `bench decode` and `bench rolling`; `bench probe`, `api`, `quality`, `tools`, `loop`, `profile`; `memwatch report` | `final-spec`, `-cache`, `-vision-smoke`, `-needle`, `-capacity`, `-pin`, `-cache-images`, `-speed-prefill`, `-speed-baseline`, `-speed-decode`, `-speed-c4`, `-api-test`, `-quality-tools`, `-loop-probe`, `-memory` |
| boot c (chunk) | `PREFILL_CHUNK=1000`: `start`, `bench exactness`; diffs against boots p and g | `final-exact-basic` |

The records are written by `bench/tools/final_records.py` from what the run kept (nothing is typed): `meta.json` (schema of the ladder's
records; `date` is a full UTC timestamp so the newest-record rule sorts them after any record of the same day), `README.md`, `raw/` (every output the
verdict rests on, lab addresses replaced by the node name, and `cap-proof.txt`: the `GPU clocks set to "(gpuClkMin 2200, gpuClkMax 2200)"` lines of the
node; a gate without that proof is a fail). A gate that fails is recorded `result: fail` with the cause in `notes`; the run stops at it unless `--continue`
is given. The run always ends by stopping the server and printing `PASS gate` / `FAIL gate` / `NOT RUN gate` per gate and a total; exit 0 only when all
nineteen gates pass. `--resume RUN_DIR` keeps finished phases and restarts an unfinished boot.

Running it, and the traps (learned in the rehearsal at 1ad011ff on node-3 and the real run at 59d08a9c on node-1):

- From a worktree of the recipe whose tracked files are committed; `.env` (git-ignored) holds `NODES` (the node's address: the tools build URLs from it, an ssh alias does not resolve) and `NODE_NAME` (node-1..node-4).
  Start it detached (`nohup scripts/final_run.sh --continue > run.out 2>&1 < /dev/null &`) and read `run.out` or `bench/runs/final-<stamp>/`; the run takes about 1 h 25 min (build 2 min, bench pin 6 min, bench all 20 min, prefill sweep 17 min, two extra decode and rolling passes 8 min, quality/tools/loop 7 min, memory bracket 18 min).
- Use `--continue`: `quality-tools` FAILs honestly (toolgate 25/28: three `order_*` cases call `run_sql` first), and without `--continue` the run would stop there, before the memory bracket and boot c.
- Nobody else may use the node or the server while it runs: any other request on the server (a `bench` of another lane, a stray client) takes a slot and spoils the 4-lane memory bracket (a fifth stream makes one lane wait for a slot: HTTP 503 after the default `--queue-timeout` 300 s) and the histogram checks of `bench.py`.
- Never `pkill`/`killall` on the Mac or a node (other lanes' clients share the Mac); a stage you must stop: `kill` the PID of this script only (its children end with it; a python client may survive it: check `ps` and kill that PID by number).
- `--resume RUN_DIR` needs the recipe at the same commit as the run started; after a tool fix start a new run or write the missing record by hand with `python3 bench/tools/final_records.py --run RUN_DIR --node-name node-N --force GATE`.
- The first run on a tree without `bench/reference*.json` writes them (the pin); commit them with the records. A later run reproduces them (a difference fails the `pin` gate).
- The goldens need the Small's pack on the node (`SMALL_MODEL_DIR`, default `/home/spark/models/Inkling-2-Small-EXL3-4bpw`): the drafter hash test reads its `mtp.safetensors`, `model.llm.embed*` and `unembed` tensors, read-only; the Small's server need not run (it must not be touched).
- The end state: server stopped, GPU idle, the cap in place; start the server for the owner afterwards with `./inkling2mini.sh start` (shipping defaults).
