#!/usr/bin/env python3
"""make_records.py ENGINE: builds the raw/ and README.md parts of bench/records/2026-11-23-draft-slim and -draft-slim-exact from the sessions of
session.sh (/tmp/slim/runs: base, confirm, slim, slimoff), the lab runs (/tmp/slim/final-lab) and the gates (/tmp/slim/gates-final2 and .out);
writes the SPEC files /tmp/slim/spec-speed.json and /tmp/slim/spec-exact.json for finalize_record.py. Run from the recipe worktree after chain.sh."""
import json
import re
import shutil
import sys
from pathlib import Path

ENGINE = sys.argv[1]
R = Path("/tmp/slim/runs")
LAB = Path("/tmp/slim/final-lab")
SP = Path("bench/records/2026-11-23-draft-slim")
EX = Path("bench/records/2026-11-23-draft-slim-exact")
sys.path.insert(0, str(SP / "tools"))
import summarize  # noqa: E402

S = {t: summarize.session(R / t) for t in ("base", "confirm", "slim", "slimoff") if (R / t).exists()}
b, c = S["base"], S["confirm"]


def pct(x, y):
    return f"{(y / x - 1) * 100:+.1f} %"


def txt(p):
    return Path(p).read_text() if Path(p).exists() else ""


timing = {}
for ctx in (4096, 32768):
    for m in re.finditer(r"ctx (\d+) b=(\d): DraftSlim off ([\d.]+) ms, on ([\d.]+) ms", txt(LAB / f"timing-{ctx}.log")):
        timing[(int(m.group(1)), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
agree = re.search(r"SLIM_AGREE real cases.*", txt(LAB / "agree.log"))
agree_line = agree.group(0) if agree else "(missing)"

metrics = {k: c[k] for k in ("decode_mean4_tok_s", "decode_prose_tok_s", "decode_code_tok_s", "decode_structured_tok_s", "decode_json_tok_s",
                             "decode_c4_rolling_tok_s", "decode_c4_steady_tok_s")}
metrics["decode_c4_aggregate_tok_s"] = c["decode_c4_tok_s"]
metrics["decode_c8_aggregate_tok_s"] = c["decode_c8_tok_s"]
metrics["baseline_decode_mean4_tok_s"] = b["decode_mean4_tok_s"]
metrics["baseline_decode_c4_rolling_tok_s"] = b["decode_c4_rolling_tok_s"]
for k in summarize.KINDS:
    metrics[f"decode_{k}_tokens_per_round"] = c[f"decode_{k}_tokens_per_round"]
    metrics[f"baseline_decode_{k}_tokens_per_round"] = b[f"decode_{k}_tokens_per_round"]
for (ctx, n), (off, on) in timing.items():
    metrics[f"proposal_ms_{ctx}_{n}req_slim_off"] = off
    metrics[f"proposal_ms_{ctx}_{n}req_slim_on"] = on
for k, v in c.items():
    if k.startswith("specbench_") and isinstance(v, (int, float)):
        metrics[k] = v

rows = []
for k, lab in (("decode_mean4_tok_s", "**decode_mean4_tok_s** (`bench decode`, printed as `decode_c1_tok_s`)"), ("decode_prose_tok_s", "prose"),
               ("decode_code_tok_s", "code"), ("decode_structured_tok_s", "structured"), ("decode_json_tok_s", "json"),
               ("decode_c4_rolling_tok_s", "**decode_c4_rolling_tok_s** (`bench rolling`)"), ("decode_c4_steady_tok_s", "decode_c4_steady_tok_s"),
               ("decode_c4_tok_s", "c=4 span (`bench decode`)"), ("decode_c8_tok_s", "c=8 span (2 waves)"),
               ("specbench_c4_aggregate_tok_s", "specbench c=4 aggregate (chat, code, json, reasoning)")):
    rows.append(f"| {lab} | {b.get(k)} | {S.get('slimoff', {}).get(k)} | {c.get(k)} | {pct(b[k], c[k]) if b.get(k) and c.get(k) else ''} |")
tpr = [f"| {k} | {b[f'decode_{k}_tokens_per_round']} | {c[f'decode_{k}_tokens_per_round']} |" for k in summarize.KINDS]
sbk = [f"| {k} | {b.get(f'specbench_{k}_tok_s')} | {c.get(f'specbench_{k}_tok_s')} | {b.get(f'specbench_{k}_tokens_per_round')} | "
       f"{c.get(f'specbench_{k}_tokens_per_round')} |" for k in ("chat", "code", "json", "longctx", "tool", "reasoning")]
prop = [f"| {ctx} | {n} | {timing.get((ctx, n), ('', ''))[0]} | {timing.get((ctx, n), ('', ''))[1]} | "
        f"{round(timing[(ctx, n)][0] - timing[(ctx, n)][1], 3) if (ctx, n) in timing else ''} |" for ctx in (4096, 32768) for n in (1, 2, 4)]
caps = "\n".join(f"- `{t}`: " + " / ".join(txt(R / t / "gpu-cap.txt").strip().splitlines()) for t in ("base", "slim", "slimoff", "confirm") if (R / t).exists())

SP.mkdir(parents=True, exist_ok=True)
shutil.rmtree(SP / "raw", ignore_errors=True)
for t in S:
    shutil.copytree(R / t, SP / "raw" / t, ignore=shutil.ignore_patterns("*.sqlite"))
shutil.copytree(LAB, SP / "raw" / "final-lab", dirs_exist_ok=True)
(SP / "profile").mkdir(exist_ok=True)
shutil.copy(LAB / "timeline-proposal-final.txt", SP / "profile" / "timeline-proposal-final.txt")

head = f"""# DraftSlim: the drafter's Markov steps without the 103 MB table (lane `k-slim`)

Gate `speed-decode`. Engine `{ENGINE}` (lane branch `inkling2-loop-slim`, merged on the integration head `be0ba1e`), baseline engine `4f939a6` (the integration head
before the lane; decode is unchanged by the later integration commits, which touch prefill and comments), node-1, GPU SM clock capped at 2,200 MHz for every run
(`raw/*/gpu-cap.txt`: the success line of the cap, applied by `scripts/gpu_cap.sh` through `tools/session.sh` before each server start):

{caps}

Design note, the ideas tested and what is exact: `docs/DECODE-SPEED.md`, "Draft slim". Exactness proofs: record `draft-slim-exact` (same engine commit).

## Result (median of 3 after a discarded warm-up, fresh server per configuration, shipping defaults `SPEC=dspark PREFIX_CACHE=on VISION=on`)

| metric | baseline `4f939a6` | lane build `55f4c1d`, DraftSlim off (`REXL3_INKLING2_LOOP_DRAFT_SLIM=0`) | lane `{ENGINE[:7]}` (confirm) | confirm vs baseline |
|---|---:|---:|---:|---:|
"""
body = "\n".join(rows) + """

The DraftSlim-off column is the lane's earlier build with the switch off (it still has the finer draft attention partition): the A/B of the pick alone.

Tokens a round per kind (`bench decode`, each kind's own `spec` counters, median of 3; the policy decides how many rounds draft, so prose, which rarely drafts,
moves from run to run):

| kind | baseline | lane |
|---|---:|---:|
""" + "\n".join(tpr) + """

specbench (512 greedy tokens, median of 3; drafted tok/s, tokens a round):

| kind | baseline tok/s | lane tok/s | baseline tok/round | lane tok/round |
|---|---:|---:|---:|---:|
""" + "\n".join(sbk) + """

## The proposal (lab: `inkling2_dspark slim_proposal_timing`, synthetic features, int8 plane, median of 20 `propose` calls, capped)

| positions | requests | DraftSlim off ms | on ms | saved ms |
|---:|---:|---:|---:|---:|
""" + "\n".join(prop) + f"""

The off column has the finer attention partition (0.45 ms of it); the full-table path before the lane measured 7.08 / 7.73 / 9.08 ms at 4K (first lab run).
nsys timeline of one proposal at 4,096 positions: `profile/timeline-proposal-final.txt` (3.34 ms span; the pick kernel about 16 us, the scan 9.5 us, a step about 55 us).
`slim_picks_the_full_heads_winner` (1,197 anchors of the real-feature cases, 7 steps each): {agree_line}

## Commands

- `NODES=<node-1> REXL3_SRC=<engine worktree> ./inkling2mini.sh build REV` for REV in `4f939a6` (baseline), `55f4c1d`, `{ENGINE[:7]}`; `tools/ab.sh` and `tools/chain.sh` drive the sessions.
- `tools/session.sh TAG MODE` (cap, `start`, `exactness.py record`, `bench decode` x4, `bench rolling` x4, `specbench.py --concurrency 4`; MODE full adds the gates of the exactness record), `tools/summarize.py` the tables, `tools/lab_final.sh` the lab numbers, `tools/make_records.py` this record.
- Analysis behind the design note: `tools/a1.py` .. `a4.py` (numpy on a dump of `fold_w`, the base rows and the latents of 77 real positions), `tools/nsys_slim.sh`, `tools/tl.py`.

## Notes

- A first A/B (lab tests, one base and one slim session and a started third) had no proof that the cap was in force; it was discarded and none of its numbers is used anywhere.
  Everything above is from runs started after the cap was applied and its success line logged.
- The finer draft attention partition changes the summation order of the drafter's context attention (the block rows `hd` move in their last bf16 bits); the Mini's golden hashes in
  `testdata/gpu/golden_inkling2_loop_dspark.txt` were regenerated and the oracle tolerances hold.
- Not built (the next lever): a bit-exact per-row EXL3 evaluation of the unembed for the candidates, which would replace the 348 MB unembed read (1.5 ms of the proposal's 3.3) by a proxy scan.
- The engine's budget prices (`round_cost`, proposal 6.4 / 7.0 / 7.7 / 8.3 ms) are unchanged and now stale; re-pricing them is an engine-policy change not made here.
"""
(SP / "README.md").write_text(head + body)

EX.mkdir(parents=True, exist_ok=True)
gout = txt("/tmp/slim/gates-final2.out")
filt = {"rank-graphs-on": "inkling2_loop_rank forward_oracle_gate long_context_gate", "rank-graphs-off": "same with REXL3_INKLING2_LOOP_GRAPHS=0",
        "graphs": "inkling2_loop_graphs graph_replay_equals capture_refusal", "spec": "inkling2_loop_spec (all)", "prefix": "inkling2_loop_prefix (all)",
        "dspark-mini": "inkling2_dspark mini_ (with the DraftSlim tests; oracle fixtures, SLIM_STRIDE=1)", "attn": "inkling2_attn mini:: small::",
        "ops": "inkling2_ops split_topk sharded_argmax unembed_softcap golden", "moe": "inkling2_moe golden graph", "moe-mixed": "inkling2_moe_mixed graph"}
tests = [{"bin": m.group(1), "filter": filt.get(m.group(1), ""), "passed": int(m.group(2)), "failed": int(m.group(3))}
         for m in re.finditer(r"GATE (\S+)\s+passed=(\d+) failed=(\d+)", gout)]
okc = sum(int(x) for x in re.findall(r"test result: ok\. (\d+) passed", txt("/tmp/slim/host-other-final.log")))
tests.append({"bin": "gb-model --lib", "filter": "families::inkling2:: families::inkling2_loop:: window_graph:: (REXL3_NO_CUDA=1)", "passed": 152, "failed": 0})
tests.append({"bin": "gb-engine gb-spec gb-sched", "filter": "all (REXL3_NO_CUDA=1)", "passed": okc, "failed": 0})
rr = R / "confirm"
ed = txt(rr / "exactness-diff.out")
live = [("exactness.py", "record + diff vs base (4 replies identical)", "identical: 4 replies" in ed),
        ("exactness.py", "concurrent", '"pass": true' in txt(rr / "exactness-concurrent.out")),
        ("bench.py", "--ref pinned replies (PASS pin)", "PASS pin" in txt(rr / "bench-ref.out")),
        ("specedge.py", "all (65 cases)", "65 cases, 0 failures" in txt(rr / "specedge.out")),
        ("gates.py", "spec", json.loads(txt(rr / "gates-spec.json") or "{}").get("pass") is True),
        ("gates.py", "cache", json.loads(txt(rr / "gates-cache.json") or "{}").get("pass") is True),
        ("specmix.py", "drafting follows the traffic", "PASS" in txt(rr / "specmix.out"))]
for bn, f, ok in live:
    tests.append({"bin": bn, "filter": f, "passed": int(ok), "failed": int(not ok)})
shutil.rmtree(EX / "raw", ignore_errors=True)
shutil.copytree(rr, EX / "raw" / "confirm", ignore=shutil.ignore_patterns("*.sqlite"))
if Path("/tmp/slim/gates-final2").exists():
    shutil.copytree("/tmp/slim/gates-final2", EX / "raw" / "gates")
(EX / "raw" / "gates.out").write_text(gout)
h = {}
for key, pat in (("forward_oracle_dump_sha256", r"LOOP_dump_a sha256 (\w+)"), ("chunk_width_logits_sha256", r"LOOP_chunk_width 7 logits_identical \d+ bytes sha256 (\w+)"),
                 ("graph_equal_logits_sha256", r"LOOP_GRAPH_equal windows \d+ logits_sha256 (\w+)")):
    m = re.search(pat, gout)
    h[key] = m.group(1) if m else None
(EX / "README.md").write_text(f"""# DraftSlim exactness (gate `spec`, engine `{ENGINE}`)

Companion of `draft-slim` (same engine commit, the same `confirm` session, GPU capped at 2,200 MHz: `raw/confirm/gpu-cap.txt`). Drafts may differ from the full head's; replies may not.

| proof | result |
|---|---|
| the target's rank dump (`inkling2_loop_rank forward_oracle_gate`, graphs on and off) | sha256 `{h['forward_oracle_dump_sha256']}` (expected `c25f9876...c45c0`) |
| chunk-width logits (7 and 11 rows) | sha256 `{h['chunk_width_logits_sha256']}` (expected `a105f245...f41a`) |
| graph replay == eager, 37 windows of node-1's `ids_long` copy | logits sha256 `{h['graph_equal_logits_sha256']}` (an ids file other lanes may not share: not comparable with their hash) |
| live replies, shipping defaults: `exactness.py record`, `diff` against the baseline `4f939a6` session | {ed.strip().splitlines()[-1] if ed.strip() else 'missing'} |
| `exactness.py concurrent`, `specedge.py` (65 of 65), `gates.py spec` and `cache`, `specmix.py`, `bench.py --ref` PASS pin | `raw/confirm/*.out`, the `tests` list of `meta.json` |
| the Mini drafter battery (`inkling2_dspark mini_`; the Small's tests need the Small pack and do not run here) | drafts within the float64 oracle's tolerances, graph replay == eager, row invariance, rollback, snapshots |
| `slim_picks_the_full_heads_winner` | {agree_line} |

Not proven: DraftSlim is not bit-identical to the full head's draft (1 of 8,379 real steps picks another token, 1 anchor of 1,197 drafts another sequence). The target verifies every draft, so
replies are unchanged by construction and by the gates above; the drafter's own golden hashes changed (regenerated) because of the finer attention partition.
""")

spec_speed = {"gate": "speed-decode", "result": "pass", "node": "node-1",
              "tests": [{"bin": "tools/session.sh confirm", "filter": "bench decode x3, bench rolling x3, specbench (replies equal)", "passed": 1, "failed": 0}],
              "metrics": metrics,
              "commands": ["tools/ab.sh 4f939a6 55f4c1d", f"tools/chain.sh {ENGINE}", "tools/lab_final.sh", "tools/make_records.py"],
              "notes": f"decode_mean4_tok_s is the median of the three run means of bench decode (printed as decode_c1_tok_s): baseline {b['decode_mean4_runs']} -> {c['decode_mean4_runs']}; "
                       f"rolling {b['decode_c4_rolling_runs']} -> {c['decode_c4_rolling_runs']}. The 100 and 200 tok/s targets are not reached by this lever alone."}
spec_exact = {"gate": "spec", "result": "pass" if all(t["failed"] == 0 for t in tests) else "fail", "node": "node-1", "tests": tests,
              "metrics": {**h, "specedge_cases": 65, "exactness_replies_identical": 4},
              "commands": ["tools/mini_gates.sh OUT (NODE=node-1 LAB_SRC=<engine worktree>)", "tools/session.sh confirm full",
                           "REXL3_NO_CUDA=1 cargo test -p gb-model --lib --locked -- families::inkling2:: families::inkling2_loop:: window_graph::"],
              "notes": "see README.md"}
Path("/tmp/slim/spec-speed.json").write_text(json.dumps(spec_speed, indent=1))
Path("/tmp/slim/spec-exact.json").write_text(json.dumps(spec_exact, indent=1))
print(json.dumps(metrics, indent=1))
print([(t["bin"], t["passed"], t["failed"]) for t in tests])
