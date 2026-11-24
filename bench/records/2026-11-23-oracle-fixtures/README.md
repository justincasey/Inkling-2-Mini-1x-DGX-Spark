# oracle-fixtures: the real-pack fixtures of the oracle (gate `oracle-fixtures`)

Result **pass** (`meta.json`, date 2026-11-23, node node-1, recipe commit `ac8a389cb349`). Each verification is a `tests`
entry of `meta.json`; what failed, if anything, is in `notes`. Pack `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` at the pinned
revision. The fixtures are CPU float64 oracle runs (no GPU); they stay on the node; this record holds their manifests, the job's
progress file, logs and run script, and the drafter case metas.

## The three stages

| Stage | What it is | Count | Size | Seconds |
|---|---|---|---|---|
| `spec` | `gen_fixtures.py` with no emulation: the oracle's own numerics, each prompt prefilled and greedy-decoded from its cache, checked against a full recompute (`--verify-recompute`: bf16 planes and routes bit for bit, logits within 2 fp32 ulp) | 4 | 1.23 GiB | 4886 (sum of the manifests' generation and verification seconds) |
| `engine` | the same prompts with `--emulate engine`: the lossy choices of the REXL3 engine (int8 KV on the global passes, bf16 P in attention, fp16 expert GEMM inputs, fp16 rotated inputs of the dense EXL3 linears); numerics_version 2 | 4 | 1.23 GiB | 3322 |
| `dspark` | `gen_dspark_fixtures.py`: the block drafter on the features of the spec fixtures (`stack(hidden[5], hidden[23], hidden[41], h_final)`), 8 anchors per case, plus one synthetic 1,100-row case (`long`): drafts, hd, f_rnn, Markov state/latent, logits, the context K/V | 5 cases | 284.9 MiB | 45 |

What a fixture holds (`oracle/README.md`, "Fixtures on the real pack"): `tokens`, `logits` [N-1, 200058], `hidden` [49, N-1, 2304]
(P entering each of the 48 layer passes, then after the last), `h_final`, `core_out` [2, N-1, 2304] (the message bank partials),
`router_idx` / `router_w` / `router_margin` for every pass; metadata `prompt_len`, `pack_config_sha256`, `oracle_source_sha256`,
`numerics_version`, `emulate`. The drafter files are named in `dspark/*_meta.json`.

### spec (`spec-manifest.json`)

| Prompt | Prompt tokens | Size | Seconds | sha256 | numerics_version |
|---|---|---|---|---|---|
| `raw-long-window` | 1119 | 1,174,554,088 B | 922 | `60c1629c51a4` | 2 |
| `chat-sky-effort0.9` | 26 | 51,113,808 B | 964 | `0dd087a94e45` | 2 |
| `raw-counting` | 35 | 52,156,920 B | 582 | `efc73ed44c17` | 2 |
| `chat-capital-effort0` | 25 | 41,725,728 B | 765 | `317571c029c8` | 1 |

`chat-capital-effort0` was written by the earlier job on another node with numerics_version 1; for `emulate none` the oracle is
bit-identical to version 2 (`bench/records/2026-11-23-oracle-cacheddiff/logs/old-vs-new-oracle.txt`). The other three were written
on node-1 by the verified generator of that record. The spec manifest is the merge of those runs.

### engine (`engine-manifest.json`)

| Prompt | Size | Seconds | sha256 |
|---|---|---|---|
| `chat-capital-effort0` | 41,725,728 B | 778 | `97fedc4b3151` |
| `chat-sky-effort0.9` | 51,113,808 B | 984 | `8d900b347c9f` |
| `raw-counting` | 52,156,920 B | 629 | `eeb450b1bb21` |
| `raw-long-window` | 1,174,554,088 B | 929 | `17a90fc24e74` |

## Where, how to regenerate

On node-1: `<base>/fixtures/{spec,engine,dspark}/`, base `/home/spark/inkling-2-mini-1x-dgx-spark`; the job is `fixtures/run-fixtures.sh`
(`run-fixtures.sh` here), progress in `progress.txt` (`{"spec": {"rc": 0, "seconds": 4886, "ended": "2026-11-23T08:25:04Z"}, "engine": {"rc": 0, "seconds": 3322, "ended": "2026-11-23T09:20:26Z"}, "dspark": {"rc": 0, "seconds": 45, "ended": "2026-11-23T09:21:11Z"}}`). Regenerate a stage by running its command from
`run-fixtures.sh` with the pack staged and the oracle (`oracle/` of this repository) copied to `<base>/oracle`; every command is
niced and takes about the seconds above. `spec` needs `--verify-recompute` and one run per prompt group (see `oracle/README.md`).
`python3 oracle/finalize_fixtures_record.py --ssh <node>` rebuilds this directory.
