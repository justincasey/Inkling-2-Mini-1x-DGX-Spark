# Inkling-2-Mini-1x-DGX-Spark

Serve **vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw** @ `44c78c9d229b43dd45e41506071b9f5f939bed44` on **one DGX Spark** (GB10, TP1) behind an
OpenAI-compatible endpoint: text, images, reasoning, tool calls, a block drafter and a prefix cache, on the REXL3 engine family `inkling2_loop`.
Every optimisation that ships is **byte-exact**: replies equal the serial, uncached, single-stream replies bit for bit (drafted == serial, cached == uncached,
concurrent == solo, chunk width changes no byte, boot to boot).

> **Status (2026-11-24): served on one Spark. The final run at the pin passes 18 of its 19 gate records; the tool-call gate scores 25 of 28 and is recorded as a fail**
> (`bench/records/2026-11-23-final-quality-tools/`). **The owner's three speed stretch targets were not met** (table below). `./autoresearch.sh` is the lab's bring-up ladder:
> 52 of its 55 rungs stand there, and the three that do not are those targets. It checks each node record's engine commit against the lab's private development checkout of the
> engine (`ENGINE_SRC`), whose commits are not in the public series, so the rungs that make that check cannot pass outside the lab; the records and `scripts/records_index.py`
> are the evidence a clone can read. Every measured number in the Results table is read from a record in `bench/records/` by `scripts/records_index.py`; the bounds quoted in the targets table are arithmetic from the documents it cites.

## Results

<!-- headline:start -->
| measure | value | record | engine |
|---|---:|---|---|
| Cold prefill, 8K prompt | 2,581.1 tok/s | `2026-11-23-final-speed-prefill` | `59d08a9c5e9f` |
| Cold prefill, 32K prompt | 2,380.7 tok/s | `2026-11-23-final-speed-prefill` | `59d08a9c5e9f` |
| Cold prefill, 128K prompt | 1,799.3 tok/s | `2026-11-23-final-speed-prefill` | `59d08a9c5e9f` |
| Cold prefill, 224K prompt | 1,445.6 tok/s | `2026-11-23-final-speed-prefill` | `59d08a9c5e9f` |
| Decode, one stream, mean of prose, code, structured and JSON (drafter on) | 91.438 tok/s | `2026-11-23-final-speed-decode` | `59d08a9c5e9f` |
| Decode, one stream, prose | 52.023 tok/s | `2026-11-23-final-speed-decode` | `59d08a9c5e9f` |
| Decode, one stream, code | 129.754 tok/s | `2026-11-23-final-speed-decode` | `59d08a9c5e9f` |
| Decode, one stream, structured | 88.472 tok/s | `2026-11-23-final-speed-decode` | `59d08a9c5e9f` |
| Decode, one stream, JSON | 95.538 tok/s | `2026-11-23-final-speed-decode` | `59d08a9c5e9f` |
| Decode, four streams in a closed loop (drafter on) | 142.329 tok/s | `2026-11-23-final-speed-c4` | `59d08a9c5e9f` |
| Decode, four streams, wall time to the last reply (drafter on) | 113.175 tok/s | `2026-11-23-final-speed-c4` | `59d08a9c5e9f` |
| Boot to ready | 21 s | `2026-11-23-final-boot` | `59d08a9c5e9f` |
| Weights resident | 13.55 GiB | `2026-11-23-final-boot` | `59d08a9c5e9f` |
| KV pool | 1,048,576 tokens | `2026-11-23-final-capacity` | `59d08a9c5e9f` |
| Lanes at the full window | 4 | `2026-11-23-final-capacity` | `59d08a9c5e9f` |
| Needle test at the window, prompt length | 260,019 tokens | `2026-11-23-final-needle` | `59d08a9c5e9f` |
| Needle test at the window, needles found | 2 | `2026-11-23-final-needle` | `59d08a9c5e9f` |
| Needle test at the window, needles planted | 2 | `2026-11-23-final-needle` | `59d08a9c5e9f` |
| Lowest MemAvailable while four cold 260K prompts run at once | 69,066 MiB | `2026-11-23-final-memory` | `59d08a9c5e9f` |
| Decode, one stream, serial | 53.195 tok/s | `2026-11-23-c4-baseline` | `1d53344d874b` |
| Decode, four streams in a closed loop, serial | 100.499 tok/s | `2026-11-23-c4-baseline` | `1d53344d874b` |
| KV pool, long profile | 2,097,152 tokens | `2026-11-23-profile-long-capacity` | `918e6854bb9d` |
| Lanes at the full window, long profile | 2 | `2026-11-23-profile-long-capacity` | `918e6854bb9d` |
| Needle test at the long window, prompt length | 1,042,827 tokens | `2026-11-23-needle-window` | `918e6854bb9d` |
| Needle test at the long window, needles found | 7 | `2026-11-23-needle-window` | `918e6854bb9d` |
| Needle test at the long window, needles planted | 7 | `2026-11-23-needle-window` | `918e6854bb9d` |
<!-- headline:end -->

Shipping configuration: `SPEC=dspark PREFIX_CACHE=on VISION=on`, profile `bootstrap` (4 slots x 262,144), int8 KV, prefill chunk 4,096, the GPU clock capped at
2,200 MHz. The `engine` column is the engine commit a record was measured at; the pin is `REXL3_COMMIT` in `config/pins.sh`. Decode figures in this table are greedy
(temperature 0) on `bench/tools/bench.py`'s four fixed prompts; sampled and reasoning chat decode has its own record (`2026-11-24-sampled-decode`, see Limits), not rows here.

The four decode kinds are fixed greedy continuations of 512 tokens, and none is a repetition loop by `scripts/loop_probe.py`'s definition (a distinct 4-gram ratio under 0.5, or a periodic tail):
`python3 scripts/loop_probe.py decode bench/records/2026-11-23-final-speed-decode/raw/g-all.d/bench-decode.json` prints each kind's ratio and verdict from the stored replies, so repetition
does not carry the mean (json is the kind nearest the line).

### The owner's stretch targets (all byte-exact)

| target | outcome | where it ended |
|---|---|---|
| cold prefill over 3,000 tok/s at 32K | **not met** | `docs/PREFILL-SPEED.md`: the byte-exact ceiling is about 2,600-2,700; the fp8 lane is a measured no-go (`bench/records/2026-11-23-fp8-*`) |
| mean decode over 100 tok/s (c=1, four kinds, drafter on) | **not met** | `docs/DECODE-SPEED.md`, `docs/DECODE-MEAN4-PLAN.md` (bound about 107 by arithmetic) |
| c=4 aggregate decode over 200 tok/s | **not met** | `docs/C4-DECODE-PLAN.md`: the byte roofline is about 196 at today's acceptance; expert skipping fails its offline gate (`bench/records/2026-11-23-skip-study`) |

## Quick start

The node is reached over ssh with a key; the controller runs from any machine. Copy `.env.example` to `.env` and set `NODES` (one address or ssh name).

```bash
./inkling2mini.sh doctor        # read-only: GPU idle, MemAvailable, pack at full size, engine matches its commit
./inkling2mini.sh setup         # first time: preflight, toolchain, the 13 pack files at the pinned revision (sha256-verified)
./inkling2mini.sh build         # the pinned engine (REXL3_COMMIT), built on the node
./inkling2mini.sh start         # applies the GPU cap, arms the memory guard, boots, waits for /health
./inkling2mini.sh smoke         # chat, completion, tool call, reasoning, image
./inkling2mini.sh status        # pid, guard, MemAvailable, GPU, last log line, /v1/models
./inkling2mini.sh stop
```

The API (model id `inkling-2-mini-1x-dgx-spark`, port 8000, no API key: keep it on a private network):

```bash
curl -s http://<node>:8000/v1/models | python3 -m json.tool                       # window, KV pool, reasoning levels
curl -s http://<node>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-1x-dgx-spark",
  "messages": [{"role": "user", "content": "Write a haiku about unified memory."}],
  "reasoning_effort": "low", "max_tokens": 512}'
# add "stream": true for server-sent events; the reply carries "reasoning_content" beside "content"
```

- **Reasoning:** `reasoning_effort` is one of `none`, `minimal`, `low`, `medium`, `high` (the default), `xhigh`, `max`. The model reasons briefly even at `none`.
- **Images:** a content part `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` in a user message.
- **Tools:** the standard OpenAI `tools` array; the call comes back in `tool_calls`.
- **Other profiles:** `PROFILE=long ./inkling2mini.sh start` (2 lanes x 1,048,576) or `max` (4 lanes x 1,048,576, little memory headroom: `docs/FIT.md`). Both were measured on the older engine `918e6854bb9d`
  with the drafter off (records `profile-long` and `profile-max`, the `engine` column above); neither has been booted at the pin with the shipping defaults, and the memory plan is optimistic by about 1.35 GiB
  at `max` with the drafter and the vision tower (Limits). `SPEC=none` turns the drafter off. `PREFIX_CACHE=off`, `VISION=off`: `.env.example` documents every key.
- **Logs and measurements:** `./inkling2mini.sh logs`, `./inkling2mini.sh bench help` (`bench all` is the gate set; each prints `PASS id` or `FAIL id`).
- **The other stack on this hardware:** `./inkling2mini.sh restore-other` stops this server, locks the Small's four ranks at the GPU cap with the Small's own controller and starts the Inkling-2-Small TP4 group
  (it needs all four Sparks idle with MemAvailable of 100 GiB or more each, so it cannot run while this server holds a node).

## The engine pin in this copy

`REXL3_COMMIT` in `config/pins.sh` is `fb5e3a7bbe590db3dc7ac039dd44fe751b92e3fc`, the head of the clean `engine/<topic>` series (46 stacked topics, 185 commits on
upstream `02ef8cba`) that `docs/ENGINE_EXPORT.md` describes; it is on the fork `https://github.com/justincasey/REXL3` as branch `inkling2-mini`.
**Every record in `bench/records/` was measured at `59d08a9c5e9f`**, a private development head. Its tree equals the pinned commit's except for 25
weight-derived test files (10.8 MB: slices of real expert tensors and values computed on real tensors) that cannot be redistributed (REXL3's
`CONTRIBUTING.md` section 8: no model weights or derivatives): `git diff --stat 59d08a9c5e9f <pinned commit> -- . ':(exclude)<each of the 25 paths>'` prints nothing
(`docs/ENGINE_EXPORT.md`, "Tree identity" and "Test data", has the check and lists the 25 paths by directory). The pinned commit itself has not been built or run on a node by this recipe.

- `./inkling2mini.sh build` stages the engine with `git archive` from a local clone named by `REXL3_SRC`
  (default `$HOME/projects/rexl3-inkling2-loop`): `git clone https://github.com/justincasey/REXL3.git "$REXL3_SRC"` (the pin is the head of its branch `inkling2-mini`).
- `./inkling2mini.sh engine-test goldens` needs the withheld files and fails in a public clone (it fails closed); `engine-test host` does not need them.
- `bench pin` compares against replies recorded at `59d08a9c5e9f`.
- Not published here for the same reason: `bench/records/2026-11-23-kernels-attn/real/attn_real.bin` and the float planes of
  `bench/records/2026-11-23-kernels-draft/oracle-planes/` (their sha256 are in the records), listed in `NOTICE`.

## The GPU clock cap

The SM clock is capped at 2,200 MHz (`GPU_CLOCK_CAP_MHZ` in `config/pins.sh`): hardware protection, in place whenever any work runs on any node. The controller applies
it before every `start`, `build` and `engine-test` and requires the success line of `nvidia-smi -lgc`; nothing releases, raises or changes it
(`gpu-clocks` has `lock` and `status` only), and `restore-other` locks the Small's ranks at the same value before it starts them (the Small's own `start` does not). The tools enforce this from commit
`5c34e19` on, and every record of the final run carries the proof line (`raw/cap-proof.txt`). Earlier records do not all: `python3 scripts/records_index.py --cap-audit` lists, for every GPU-node record,
whether it carries the proof line, only a clock sample taken when the record was written, or neither, and fails if a sample is above the cap. No sample is above it; for a record without the proof line that is
a sample, not proof of a lock.

## Lane

| | |
|---|---|
| pack | `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` (EXL3 mixed K: experts 4/5-bit, attention / unembed / vision 6-bit, drafter 4-bit) |
| base model | `thinkingmachines/Inkling-2-Mini` (Apache-2.0): 26.4B total, 3.4B active, looped 24-layer MoE, vision, block drafter |
| topology | TP=1, one Spark; the Inkling-2-Small TP4 group is stopped while a node serves this recipe (`docs/CLEANUP.md`, "Services stopped") |
| engine | REXL3, family `inkling2_loop`, local branch `inkling2-loop` on upstream `main` `02ef8cb`; pin `REXL3_COMMIT`; the clean `engine/<topic>` series for a submission: `docs/ENGINE_EXPORT.md`, notes for REXL: `docs/REXL-PR-NOTES.md` |
| exact pack size | 15,070,275,721 B in 13 files; weights 14.000 GiB (`intake/manifest.tsv`, `intake/census.json`) |
| memory | the guard (`scripts/memguard.sh`) is mandatory: `start` fails and kills the server unless it armed on the server's pid |

## Limits (what was not measured, and known weak spots)

- **Sampled and reasoning decode** has one record, `bench/records/2026-11-24-sampled-decode/` (the pin, 16 original prompts, `reasoning_effort` low and high, temperature 0 and 1.0 (the pack's own default),
  the drafter on and off with the same seeds). Drafted replies equal serial replies token for token in all 64 pairs (32 at each temperature). The mean decode speed over the 16 prompts is 81 to 92 tok/s drafted
  against about 53 serial (1.53 to 1.71 times; sampling at 1.0 costs the drafter some speed, not its lead). Chat replies are the one kind near parity (1.015 to 1.130 times). At effort high, code replies spend
  98 to 99 % of their tokens reasoning and all four reach `max_tokens` 2,048, and over all prompts 5 of 16 (temperature 0) and 6 of 16 (1.0) are cut at the limit: such clients need a larger `max_tokens`.
  Four requests per category cell, one prompt set, one node: small cells.
- **Drafting does not pay on raw prose continuation**: the bench's prose kind decodes at 52.0 tok/s drafted (`final-speed-decode`), about the serial rate (52.5 to 53.7 in every cell of `sampled-decode`); chat replies
  at c=1 are at or above serial. An all-chat mix at c=4 was estimated to lose about 9 % against serial (`docs/DECISIONS.md`, `docs/C4-DECODE-PLAN.md` A5) and has not been re-measured: measure your own mix before
  choosing `SPEC=none`. Code, structured output, JSON and tool calls gain the most.
- **DraftSlim's drafter pick is not proven equal to the full argmax**: it picks the full head's winner in all but 1 of 8,379 real-feature steps (`bench/records/2026-11-23-draft-slim-exact/`). It changes only what the
  drafter proposes and the target verifies every draft, so replies are unchanged and drafted equals serial, but the draft itself is not bit-identical to the full head's.
- The pack is not validated by REXL: it serves with `--allow-unvalidated-pack` (the server says so in `/v1/models`).
- **The tool-call gate is a recorded fail**: the 60 exact-answer questions are all right, but the tool gate passes 25 of its 28 schema cases. The three misses
  (`order_nested`, `order_two_skus`, `order_standard`) call `run_sql` before `create_order`; there was no invalid call and no unrequested destructive call
  (`bench/records/2026-11-23-final-quality-tools/`). In raw greedy text continuation 17 of 34 probe prompts loop (a baseline the loop probe counts and does not judge; the base model's card warns of it);
  in reasoning mode none of 12 does. The Inkling-2-Small recipe's probe of its own model found 5 (bf16) to 8 (int8) of 34, so whether the Mini's higher rate comes from the model, the quantization or this
  engine (its int8 KV or its loop state) is not established.
- The oracle (an fp64 CPU reference with a bit-exact EXL3 decoder) agrees with the engine on the greedy token in nearly every row, not in all (`docs/ORACLE-GAP.md`).
- The memory plan is optimistic by about 1.35 GiB at the largest window with the drafter and the vision tower together; `SPEC=auto` is untested; a pack without the
  vision tower is refused (plan risk table).
- Other families were not re-qualified for the shared engine changes (kernels, server, scheduler): `docs/REXL-PR-NOTES.md` section 3.

## Files

| Path | What |
|---|---|
| `inkling2mini.sh`, `.env.example`, `config/` | the one-node controller, its configuration, the pins (`config/pins.sh`), the profiles |
| `scripts/` | `memguard.sh` (the mandatory guard), `gpu_cap.sh` and `ops_dev.sh` (node work under the cap), `check_record.py`, `records_index.py`, `final_run.sh` with `bench/tools/final_records.py` (the final procedure in one command: `docs/HARNESS-NEXT.md` section 8) |
| `bench/tools/`, `bench/records/` | the benchmark and gate tools; one directory a record (`meta.json`, README, raw output) |
| `autoresearch.sh` | the bring-up ladder (`bringup_gates`); `scripts/check_spec.py` and `scripts/check_record.py` decide its spec and node rungs |
| `tests/test_recipe.sh`, `tests/test_*.py` | the static tests (no GPU, no node); run before any hardware run |
| `oracle/` | the fp64 CPU reference forward, the EXL3 decoder, fixtures and teacher-forcing tools |
| `docs/exec-plans/` | the complete plan, phases, decisions, risks and log (`active/` until closed, then `completed/`) |
| `docs/DECODE-SPEED.md`, `PREFILL-SPEED.md`, `C4-DECODE-PLAN.md`, `DECODE-MEAN4-PLAN.md` | where decode and prefill time goes, and what moved it |
| `docs/ROOFLINE.md`, `FIT.md`, `NUMERICS.md`, `ORACLE-GAP.md`, `PORT-SPEC.md` | ceilings, memory fit, numerics, the oracle gap, the port spec |
| `docs/DRAFTER-SURVEY.md` | which drafters exist for this model |
| `docs/ENGINE_EXPORT.md`, `REXL-PR-NOTES.md` | the engine change as a clean series, and how REXL takes a contribution |
| `docs/CLEANUP.md` | everything this work put on disk, with the commands that remove it |
| `EVAL-MATRIX.md`, `PLAN.md` | the gates in order; the lane summary |

## Credits and licenses

- [Thinking Machines Lab](https://huggingface.co/thinkingmachines/Inkling-2-Mini): Inkling-2-Mini, Apache-2.0.
- [vcruz305](https://huggingface.co/vcruz305): the SAGE EXL3 4.23 bpw pack. SAGE is our own mixed-precision quantization method for EXL3. The method is not published.
- [turboderp](https://github.com/turboderp-org/exllamav3), exllamav3: the EXL3 format.
- [REXL3](https://github.com/REXL3/REXL3): the engine this recipe builds on (AGPL-3.0).
- This repository's scripts and documentation: Apache-2.0 (`LICENSE`). The model files are not in the repository.
