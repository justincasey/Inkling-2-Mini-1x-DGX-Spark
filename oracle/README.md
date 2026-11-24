# Inkling 2 Mini oracle

The oracle is an independent, deterministic CPU reference forward of the Inkling-2-Mini target text model (`inkling2_loop`:
24 layers whose middle 12 run three times = 48 layer passes a token) over the pack
`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw@44c78c9d229b43dd45e41506071b9f5f939bed44`, plus the 3-layer block drafter
(`mtp.safetensors`), an EXL3 decoder, and the tools that turn them into fixtures and gates for the REXL3 `inkling2_loop`
engine family. It was adapted from the Inkling-2-Small oracle (our own Apache-2.0 code) to the looped architecture, mixed-K
EXL3 and TP1.

Status (plan step 3.3, wave 1): the code, the synthetic tests and the EXL3 decode on **real tensors of this pack** are done.
No fixture from the real pack exists yet: that is wave 2 ("Fixtures on the real pack" below). Nothing here ever needed the
whole 14 GiB pack.

## Sources (pinned)

What the model computes (semantics) comes from these; no kernel or engine code is taken from them.

| Key | Source | Pin |
|---|---|---|
| `HF` | huggingface/transformers `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`, `configuration_inkling2_loop.py`, `conversion_mapping.py`, `core_model_loading.py` | `be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef` (main when fetched, 2026-11-22). `modeling_inkling2_loop.py` is byte-identical (sha256 `1c8e65b4…2e9f28`) at `046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`, the last commit that touched it ("Add Inkling2 Mini (#51050)", 2026-11-21) |
| `V` | vllm-project/vllm `vllm/models/inkling2/` (`nvidia/loop.py`, `nvidia/dspark.py`, `speculator.py`, `configs.py`) | `2b55f82b085a765e4c813691d505563b397b4173`; those four files are byte-identical at `8c17e4692a7cf393e3d1dacf6404fa4d48d993a1`, which the drafter and loop tests cite |
| `V-Small` | the same module for the Small drafter | `1448919ffc267a321c65b32bd51b334075d9a8f0`; a diff against `V` shows an unchanged drafter module flow (only the argmax helper, a derived-weight refresh and the loop config checks changed) |
| pack | `config.json`, the safetensors headers, `intake/census.json` of this recipe | the revision above |

HF has no drafter; the drafter's semantics are `V` (`dspark.py`, `speculator.py`). HF and `V` agree on the loop (checked by
reading both: the message bank, the controllers, the per-pass state, the taps over passes).

## What the loop is (map from the sources to the code)

- **Schedule** (`Inkling2LoopConfig.visits()`, HF `Inkling2LoopTextModel.forward`): passes 0-5 = layers 0-5; passes 6-17,
  18-29, 30-41 = layers 6-17 (core pass 0, 1, 2); passes 42-47 = layers 18-23. Global (full-attention) passes are 5, 11, 17,
  23, 29, 35, 41, 47 (`Visit.global_ordinal` 0-7); the other 40 are sliding-window (1,024). Pass k has its own K/V and its
  own four conv streams (`Cache.layers[k]`; HF appends copies of the core's cache layers per extra pass).
- **AttnRes state**: a layer with `layer % 6 == 0` (0, 6, 12, 18) opens a block (it appends its input P). Every core pass
  after the first restarts from the state after layer 5 (P and the blocks). The coda continues the last pass.
- **Messages** (core passes 1 and 2 only): bank = the previous pass's blocks after the one its first layer opened (here: P
  entering layer 12) plus that pass's final P (`ForwardOut.core_out`). Per core layer four learned queries
  (`message_queries()`: the pack stores `middle_cycle_cross_pass_queries` [24,2304] as (attn, mlp) per core layer and
  `…_kv_queries` [24,2304] as (k, v); HF `Inkling2LoopMergeMessageQueries` merges them to (attn, k, v, mlp)) read the bank
  with the same normalized-value AttnRes read as every other mix (`mix_raw`).
- **Controllers** (`blend`; pack `layers.N.controllers.{attn,k,v,mlp}.{down_weight [64,4608], gain_weight [2304,64],
  forget_weight [2304,64]}`; HF `{q,k,v,mlp}_message_gate`): `h = down . [rms(dest), rms(msg)]`,
  `out = (1 - tanh(forget h)) dest + tanh(gain h) msg`, then the sublayer's input norm. `dest` is the fp32 AttnRes mix. On a
  local layer the single mix is blended three times (q, k, v); on a global layer the three mixes are.
- **Drafter** (`dspark_ref.py`): 3 layers (layer 1 global), taps `target_layer_ids [5, 23, 41, 48]` over the 48 passes = P
  entering passes 5, 23, 41 and the final-norm output (`Inkling2LoopRef.drafter_features`); `fuse_weight` [4,4]; Markov head
  `rnn_hidden` (rank 256).
- No tensor parallelism (TP1): the Small oracle's per-rank roundings are single sums.

## Layout

| Path | What |
|---|---|
| `inkling2_ref.py` | the model: `Inkling2LoopConfig`, `Visit`, rounding, ops, `Cache`, `Inkling2LoopRef.forward/layer/blend`, `generate`, `greedy`, `Emulation` |
| `exl3.py` | EXL3 decode in numpy: codebooks (`3inst`, `mcg`, `mul1`), K read from the trellis (`bits_from_trellis`), trellis unpack, Hadamard rotations, dense decode, column slabs (`decode_cols`), layer forward |
| `exl3dq/` | Rust CLI over REXL3's CPU decoder `gb-exl3` (path dependency on `../../../REXL3/crates/exl3` (a clone of the engine repository, `https://github.com/justincasey/REXL3`, checked out at `REXL3_COMMIT`, next to this recipe; read-only; AGPL-3.0-only, see `NOTICE`)): `table`, `groups`, `wq`, `dense`, `strip`, `forward`; used for cross-checks only |
| `pack.py` | pack folder as weight and expert sources: index-driven positional reads, EXL3-dense weights decoded on read (`PackWeights`, optional LRU `cache_bytes`), `PackIndex(files=…)` for the drafter shard, `PackExperts` with an optional worker pool |
| `st.py` | minimal safetensors reader and writer |
| `gen_fixtures.py` | target fixture generator over a pack folder |
| `dspark_ref.py`, `gen_dspark_fixtures.py` | the drafter reference and its fixture generator |
| `compare.py` | engine dump vs oracle fixture: free-running metrics and the drift gates |
| `engine_gap.py` | an engine prefill dump (a server's `REXL3_INKLING2_LOOP_DUMP` directory, chunks found by token ids) or any fixture against a fixture, free of teacher forcing: per visit and row group the hidden / `core_out` / `h_final` error, the logits plane (cosine, relative L2, KL, argmax and top-2 equality, margins), router flips with their margins; `--stock-gates` also runs `compare.py`'s gates; docstring has the contract (`docs/ORACLE-GAP.md`) |
| `teacher_force.py` | engine sub-op dump vs the oracle fed the engine's own inputs, op by op and per layer pass: the correctness gate; the module docstring is the dump contract |
| `prompts.jsonl` | default fixture prompts (chat-rendered and raw text, one past the 1,024 window) |
| `synthetic.py` | tiny looped config (8 layers, core [2,6) x 3 = 16 passes), random weights, random EXL3 packs (mixed K, `mul1`), tiny drafter |
| `tools/verify_exl3_real.py` | fetches real tensors of the pack by HTTP range and checks the numpy decode against gb-exl3 bit for bit |
| `records/exl3-real-verify.json` | the result of that check |
| `tests/` | pytest suite; `tests/hf_transcription.py` is an independent float64 transcription of the HF eager looped model, `tests/vllm_dspark_transcription.py` of the vLLM drafter module flow; `tests/golden_exl3.json` holds gb-exl3's digests |

## Setup and tests (the Mac, no GPU)

```bash
cd oracle && uv venv .venv && uv pip install -r requirements.txt      # uv picked Python 3.12.11 here; the system python is
                                                                      # 3.15: `uv venv --python 3.12 .venv` pins it
cd .. && oracle/.venv/bin/python -m pytest -q oracle/tests            # from the repo root: the ladder's command
```

The suite needs `cargo` (toolchain 1.89, pinned in `exl3dq/rust-toolchain.toml`) for the live Rust cross-checks: the first
run builds `exl3dq`. `.venv`, `.cache` and `exl3dq/target` are git-ignored. Every test is deterministic (fixed seeds), uses
no network and no pack, and none is skipped.

Last full run for this document: `168 passed in 17.47s` (Mac, Python 3.12.11, torch 2.14.1).

What the tests cover (file: point):

- **`test_model.py`**
  - With rounding off (`Rounding.exact()`) the oracle equals the independent float64 HF transcription to 1e-9 in logits
    and in the P entering every one of the 16 passes (plus the last): on the tiny model for two lengths, and on a second
    shape whose core has three AttnRes blocks and runs twice (a two-block message bank).
  - Rounded passes track unrounded passes (every pass; the first, global, local and message-carrying ones).
  - AttnRes schedule: which queries, how many blocks, the 4-query message read; block bank contents; restart from the prelude.
  - Per-pass state: ring sizes, separate K/V and conv state per pass, prefill + decode == recompute at many splits,
    cached == uncached.
  - Schedule layout of the real config, `from_hf` on the real `intake/files/config.json`, its refusals.
- **`test_ops.py`**
  - Conv streaming == batch (static, dynamic), tap/lag orientation, relative-bias distances, window masks, sink and gate
    against an explicit softmax, router ties/bias/weights, softcapped activation, tau, numpy vs torch bf16 incl. ties.
  - The AttnRes read, the message-query layout, the controller formula and its limits and rounding points.
- **`test_exl3.py`, `test_pack_exl3.py`, `test_real_record.py`**
  - Codebooks and K = 4, 5, 6 (`mul1`) bit-identical to gb-exl3 (live) and to its recorded digests (`golden_exl3.json`).
  - An independent bitstream packer round trip, column slabs, wrong-marker refusal.
  - The EXL3-dense weight source, and the record file's coverage and bit-exactness.
- **`test_pack.py`**
  - A synthetic pack in the Mini layout: K per tensor from the trellis (mixed 4/5-bit inside a layer and inside one expert).
  - The whole model through the pack folder equals the in-memory twin bit for bit; worker pool == in-process; the decoded
    weight cache changes nothing; error paths.
- **`test_dspark.py`**
  - Drafter: incremental commits == one commit and rollback == replay; local window edges; global layer; the block is
    non-causal; f_rnn; equals the vLLM-flow transcription to 1e-9; the EXL3 drafter path equals the dense one bit for bit.
  - `from_hf` checks; taps over passes end to end.
- **`test_emulation.py`, `test_teacher_force.py`**
  - The engine-emulation options; teacher forcing per pass on synthetic dumps (a clean dump passes, each altered plane fails in
    its own pass only; `blocks_for`, `bank_for`); `compare.py`; `gen_fixtures.py --verify-recompute` on a tiny pack.

## Rounding points

The policy is the Small oracle's: every op between two storage points runs in float64 from the stored values, and a value is
rounded exactly where the engine stores one. The docstring of `inkling2_ref.py` and the code are what the tests pin; the spec
lane's `PORT-SPEC.md` and `NUMERICS.md` are the prose for the same points. The rows that are new for the loop:

| Value | Rounding |
|---|---|
| attention input of a pass after the first (per stream) | `bf16(rmsnorm_w(blend(dest, msg), norm))`; `dest` and `msg` are fp32 values (the unrounded AttnRes mixes) |
| controller input pair `[rms(dest), rms(msg)]` and `h = down . pair` | bf16 stores; `gain`, `forget` = `h . W^T` stored as fp32; `tanh` in fp32; the blend result fp32 |
| message queries | `H^-0.5` folded once in fp32 (as the AttnRes queries) |
| `core_out`, the P entering a pass, the layer output | bf16 (`bf16(f32(part + m))`) |
| dense (EXL3) linears | `bf16(x . W_eff^T)` with `W_eff` the float64 effective weight (not rounded to bf16), float64 accumulation |
| expert down output, weighted sum over experts | float64 then one bf16 store (no all-reduce at TP1) |

HF computes the experts' down outputs and the attention probabilities in bf16; the oracle (like the Small oracle) keeps them
in float64 until the engine's own storage point. `Emulation` reproduces the engine's three lossy kernel choices (int8 KV on the
global passes, bf16 P, fp16 expert GEMM inputs); its tile and chunk sizes are the Small kernels' (`inkling2_attn.cu`) and must
be re-checked against the Mini kernels when they exist.

## EXL3 decode and the real-tensor check

`exl3.py` is a numpy port of the CPU reference `gb-exl3` (`crates/exl3/src/lib.rs` of the engine tree `inkling2-loop`):
`unpack` reproduces `gb_exl3::unpack` bit for bit (W_q as fp16 bits), `decode_weight` reproduces `gb_exl3::decode_weight`
bit for bit (the same butterflies in the same order), `decode_cols` the columns `[c0, c1)` of it from the trellis column tiles
only (bit-identical to `gb_exl3::decode_strip`), and `forward` the float64 layer
`y = ((((x . suh) H_K) W_q) H_N) . svh`. K is read per tensor from the trellis shape (`16 * K` words a tile): routed experts 4
or 5, attention, unembed and vision 6, drafter 4; every codebook is `mul1`. The marker tensor must hold `0x83DCD12D`.

`tools/verify_exl3_real.py` fetched real tensors by HTTP range (no whole file) and compared them with gb-exl3 through `exl3dq`
(`records/exl3-real-verify.json`, UTC 2026-11-23, the Mac): **21 entries, all bit-exact** (W_q fp16 bits and float64 dense
bits equal), 11 at K=4, 4 at K=5, 6 at K=6, every marker `mul1`, 50,169,044 bytes fetched in 376 requests, 54.5 s, 1,337 MiB
peak RSS. Covered: expert triples in layers 12 and 23 (K=4, K=5, and layer 23 expert 0 with a 5-bit `down_proj` beside
4-bit gate/up), attention `wq_du` of global layer 11 and `wo_ud` of layer 0, two unembed column slabs (columns 0:1024 and
199936:201088, the second holds the real-vocab edge at token 200058), drafter `layers.1.self_attn.q_proj` and
`markov_head.{hidden_out,rnn_proj}`, vision `proj_out` and layer 0 `q_proj`. Run it again with
`oracle/.venv/bin/python oracle/tools/verify_exl3_real.py`; the record's `entries[]` carry `tensor`, `k`, `bytes` (W_q plus dense
bytes compared) and `match`.

Measured on the Mac with real groups (one process, no cache): reading and unpacking one K=4 expert (three 2304 x 1152 groups)
0.07 s, an expert FFN on 1 or 16 rows well under 0.01 s afterwards, the effective weight of a K=6 2304 x 3072 attention
linear 0.40 s.

## Fixtures on the real pack (wave 2: running; the first fixtures and the timings are in `bench/records/`)

The pack is not downloaded in wave 1. In wave 2 it is staged once on one node (plan step 2.4) and the oracle runs there on the
CPU only, niced, in a node window under the protocol of the execution plan (the Small TP4 group is production; nothing here
may start on a node outside a window). Memory: the pack's tensors are read by positional reads, nothing is mmapped; one
expert and one dense layer are resident at a time, plus whatever `--weight-cache-gib` holds.

```bash
# environment on the node (CPU-only torch wheels; the venv of this directory)
uv venv oracle/.venv --python 3.12 && uv pip install --python oracle/.venv/bin/python -r oracle/requirements.txt
PACK=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw        # staged and verified against intake/manifest.tsv
nice -n 19 oracle/.venv/bin/python oracle/gen_fixtures.py --pack "$PACK" --out fixtures/spec \
    --workers 3 --threads 4 --weight-cache-gib 12 --verify-recompute              # the spec fixtures (4 prompts)
nice -n 19 oracle/.venv/bin/python oracle/gen_fixtures.py --pack "$PACK" --out fixtures/engine \
    --emulate engine --workers 3 --threads 4 --weight-cache-gib 12                 # the engine-emulating fixtures
nice -n 19 oracle/.venv/bin/python oracle/gen_dspark_fixtures.py --pack "$PACK" --fixtures fixtures/spec \
    --out fixtures/dspark --threads 16 --weight-cache-gib 10
```

Each emulation goes to its own output directory (the metadata records `emulate`). The four prompts are in `prompts.jsonl`:
two chat-rendered (thinking effort 0 and 0.9), a counting prompt and `raw-long-window` (1,119 tokens with the pack's
tokenizer, past the 1,024 window).

### The three stages of the node-1 job

| Stage | Command (above) | Output | Cost |
|---|---|---|---|
| `spec` | `gen_fixtures.py --verify-recompute` | `fixtures/spec/` + `manifest.json` | measured on node-1: `chat-sky-effort0.9` 964 s + 620 s verification, `raw-counting` 581 s + 315 s, `raw-long-window` 922 s + 718 s; `chat-capital-effort0` 765 s generation (node-2, first run) |
| `engine` | `gen_fixtures.py --emulate engine` (numerics_version 2: includes `dense_fp16`) | `fixtures/engine/` | measured on node-2 with the emulation of version 1 (no `dense_fp16`): `chat-capital-effort0` 834 s, `chat-sky-effort0.9` 1,035 s, about 9 to 10% above spec; the version-2 emulation and the other two prompts are not measured yet |
| `dspark` | `gen_dspark_fixtures.py` (one case per spec fixture, 8 anchors each, plus the synthetic `long`) | `fixtures/dspark/` | not measured on the Mini; `--weight-cache-gib 10` keeps the 3.7 GB float64 unembed, which every proposal streams |

`--workers 3 --threads 4` is the setting measured on node-1 (three expert workers used about 5.5 to 6 cores each on its 20
cores, `ps` during the diagnostic run of `bench/records/2026-11-23-oracle-cacheddiff/`); other settings were not compared.
The job is `fixtures/run-fixtures.sh` on the node (nohup setsid, nice 19; `progress.txt`: one line per stage, `done` last).
`oracle/finalize_fixtures_record.py` verifies the finished job from the Mac and writes `bench/records/<UTC date>-oracle-fixtures/`
(gate `oracle-fixtures`).

`gen_fixtures.py` prefills each prompt, greedy-decodes from the cache, and writes `OUT/NAME.safetensors`:

| Tensor | Shape | Meaning |
|---|---|---|
| `tokens` | I64 [N] | prompt + generated tokens |
| `logits` | F32 [N-1, 200058] | row t is the soft-capped prediction for token t+1 |
| `hidden` | BF16 [49, N-1, 2304] | `[k]` = P entering layer pass k (0-47), `[48]` = P after the last pass. Drafter features are `hidden[5, 23, 41]` and `h_final` |
| `h_final` | BF16 [N-1, 2304] | final mix + norm output (drafter tap 48) |
| `core_out` | BF16 [2, N-1, 2304] | P leaving the last core layer of core passes 0 and 1: the message bank partials (`hidden` cannot hold them: the next pass restarts) |
| `router_idx` / `router_w` | I32 / F32 [48, N-1, 8] | selected experts in selection order and their weights |
| `router_margin` | F32 [48, N-1] | 8th minus 9th selection score |

Metadata: `prompt_len`, `pack_config_sha256`, `oracle_source_sha256`, `numerics_version`, `emulate`, `name`. `OUT/manifest.json`
holds each fixture's sha256, the generated ids and text, timings and the result of `--verify-recompute` (see "Cached decode vs
recompute" below).

`gen_dspark_fixtures.py` takes the drafter features from the target fixtures (`F[t] = stack(hidden[5], hidden[23],
hidden[41], h_final)[t]`) and writes, per case C (`real_raw-counting`, `real_chat-sky-effort0.9`: 8 anchors each; `long`:
1,100 seeded synthetic rows, anchors around the 1,024 window), `C_meta.json` and
`C_{feats,tokens,anchors,drafts,hd,frnn,state,latent,logits,ctxkv}.npy` (bf16 as uint16 bits; feats [N,4,2304], ctxkv
[3,N,2,8,128], logits f32 [P,7,200058]). It needs `mtp.safetensors` and the target fixtures.

Measured on node-1 (20 cores, otherwise idle, `--workers 3 --threads 4 --weight-cache-gib 10..12`, record
`bench/records/2026-11-23-oracle-cacheddiff/`): the cost is the expert decode, not the arithmetic.

| Run | Rows | Seconds |
|---|---|---|
| `chat-sky-effort0.9`, prefill (first call: also fills the weight cache) | 26 | 496 |
| `chat-sky-effort0.9`, one decode step | 1 | 20.1 to 21.0 (the first 10 of 23 steps in the log; the whole run's 967 s is consistent) |
| `chat-sky-effort0.9`, whole cached run | 26 + 23 steps | 967 |
| `chat-sky-effort0.9`, full recompute | 49 | 623 |
| `raw-long-window`, whole cached run (1,119 prompt tokens + 8 generated) | 1,119 + 7 steps | 922 (about 775 of it the prefill: 922 minus 7 steps of ~21 s) |
| `raw-long-window`, full recompute | 1,126 | 718 |

So a prompt crossing the 1,024 window costs about 15 minutes of prefill and 12 more for `--verify-recompute`; a 16-token
continuation adds about 5 minutes (15 decode steps). The older figure in this README of 0.6 s for a decode step was an extrapolation from the
Mac and is wrong by a factor of 30. `raw-long-window` generated ` 561 562 563 564` after a prompt that counts to 560: the window
logic is right at 1,119 tokens. The dense weights are 6 matrices a layer, about 215 MB of float64 each layer, and the unembed is
200,058 x 2,304 x 8 B = 3.7 GB; `--weight-cache-gib 12` holds both, so only the first forward decodes them. `--workers 3`
divides the expert decode; watch the resident set against the node's budget during the window.

## Cached decode vs recompute (`--verify-recompute`, `--verify-exact`)

The check runs the whole token sequence once more without a cache and compares it with the cached prefill + decode that wrote the
fixture. The first version demanded bit-identical logits and stopped the job at the second real prompt. Cause, measured on
node-1 (`bench/records/2026-11-23-oracle-cacheddiff/`): of 49 x 200,058 logits of `chat-sky-effort0.9`, ONE differed, by one fp32
ulp (1.19e-7 absolute, 7.2e-9 of the row's rms); the hidden state of every pass, the bank partials, the routes, `h_final` and the
argmax were bit-identical. That logit is `unembed(h)` of a decode row. The float64 product of a 1-row operand and the rows of a
batch sums in a different order (BLAS gemv against gemm): the two float64 values were 4.7e-15 apart and the exact value
(long double) sat 4e-15 from the midpoint of two fp32 numbers, so they rounded to different fp32 values. Replaying the stored
hidden states reproduced both stored logits bit for bit. Measured rate on the real unembed (9 flips in 64,225,280 logits, always
exactly 1 ulp): 1.4e-7 per logit, about one flip per 7 million logits, and only where a row is computed alone: the 25 decode rows
of `chat-sky-effort0.9` were enough. A prefill of 1,119 rows against a recompute of 1,126 rows (same gemm kernel) was identical.
Every other stored value is bf16 (spacing 2^-8 relative), which absorbs 1e-14 of noise almost always: in this run no bf16 plane differed.

- `--verify-recompute` (the default check): hidden states, bank partials and `h_final` bit for bit, routes equal, logits and routing
  weights within `gen_fixtures.LOGIT_ULP_BOUND` = 2 fp32 ulp (1 ulp is what the noise gives; the bound is that plus one), argmax and
  top-2 order of every row equal. The maxima that were measured go into `manifest.json` (`recompute_check`).
- `--verify-exact`: the logits and routing weights must be bit for bit equal as well; expect a failure on a run with many decode rows.
- A failed check no longer throws away hours of CPU: the fixture is written first, the manifest says `verified_recompute: false`, the
  other prompts still run, and the exit status is 1.

The unit tests could not see this: with a vocabulary of 250 and hidden sizes up to 2,304 the tiny looped model's cached run was
bit-identical to the recompute at six prompt lengths on both the Mac and node-1 (0 of 6 x 3 sizes differ): the flip needs ~10^7
logits. `tests/test_verify_recompute.py` pins the comparison rules.

## Tolerances

`compare.py` and `teacher_force.py` carry the Small oracle's limits, marked in the code as **"Small oracle limits, to be
re-measured on the Mini fixtures"**; the Mini's 48 passes (three times
the Small's routing depth per core layer) amplify differences about as much as the Small's did. The whole-forward comparison is written up in `docs/ORACLE-GAP.md` (plan step 6.3, record `2026-11-23-oracle-gap`): the limits above are
not edited; the document proposes row-level limits for the free-running planes from the measurement and reports what the existing sets say
(`GATES_ENGINE` fails as it did on the Small, `GATES_SPEC` misses its tie margin 0.1 by 0.003 to 0.007 on three rows).

- **Correctness gate: `teacher_force.py`** on a dump of an engine boot (contract in its docstring), against the emulating
  oracle: every op fed the engine's inputs, per layer pass; the engine's picks equal to the oracle's except within their fp32
  bound (`GATES`). Exit 1 on failure.
- **Drift: `compare.py ORACLE ENGINE [--json report.json] [--gates engine|none]`** on a free-running fixture: tokens, KL, top-1,
  hidden and h_final cosines per pass, routing overlap (informative). Gate set `engine` for an emulating oracle, `none` for the
  spec oracle; the drift gates measure and the teacher-forced gate decides.
- **Unit tests** compare with `Rounding.exact()` to 1e-9 against the HF transcription, and bit for bit where two paths are the
  same arithmetic (pack vs in-memory, cached vs recompute, EXL3 decode vs gb-exl3).

## Open items

- Fixtures on the real pack (above) and the first engine comparison.
- The vision tower is out of scope: the oracle refuses image placeholder ids.
- `Emulation` parameters (attention tile and chunk sizes) are the Small kernels'.
- HF's experts use a fused, interleaved `gate_up_proj`; the pack stores `gate_proj` / `up_proj` / `down_proj` as separate EXL3
  groups and the oracle reads those (the same arithmetic, so the layouts do not meet).


## Engine emulation (`Emulation`, `--emulate`)

Flags: `kv8`, `attn_bf16_p`, `moe_fp16`, `dense_fp16` (`engine` = all four; `none` = the exact float64 policy, the default). `dense_fp16`
rounds the rotated input `(x * suh) H` of every dense EXL3 linear the target oracle evaluates to fp16 before the product with the decoded
weight, as the engine's dense kernels do: the attention q, r, gate, k, v and `wo` projections and the unembed. It needs a weight source
that holds the group (`pack.PackWeights.exl3_linear`, `DictWeights(exl3=...)`); BF16 tensors (router, sink, tap projections, controllers,
convs, queries) and the embedding stay exact. The drafter oracle (`dspark_ref.py`) is unchanged. `teacher_force.py` evaluates the attention
variants `none`, `attn_bf16_p`, `kv8`, `kv8+attn_bf16_p`, `dense_fp16` and `kv8+attn_bf16_p+dense_fp16`; the gate's default is the last
(with `moe_fp16`). Numerics version 2 (`inkling2_ref.NUMERICS_VERSION`, recorded in fixtures' metadata): `dense_fp16` was added, nothing
else changed.
