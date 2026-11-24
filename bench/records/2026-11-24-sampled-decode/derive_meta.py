#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes meta.json and the tables (raw/tables-*.md) of this record from the saved outputs in raw/: every number comes from
the request records of bench/tools/sampled.py (boot A: SPEC=dspark, boot B: SPEC=none) and the loop probes; none is typed.

    python3 derive_meta.py
"""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
raw = here / "raw"
recipe = here.parents[2]
sys.path.insert(0, str(recipe / "bench" / "tools"))
sys.path.insert(0, str(recipe / "scripts"))
import loop_probe  # noqa: E402
import sampled  # noqa: E402

A = json.loads((raw / "boot-a-dspark.json").read_text())
B = json.loads((raw / "boot-b-serial.json").read_text())
LOOP = {"a": loop_probe.summarize(json.loads((raw / "boot-a-loop-sampled.json").read_text()))}
if (raw / "boot-b-loop-sampled.json").exists():
    LOOP["b"] = loop_probe.summarize(json.loads((raw / "boot-b-loop-sampled.json").read_text()))
greedy_loops = json.loads((recipe / "bench/records/2026-11-23-final-loop-probe/meta.json").read_text())["metrics"]

(raw / "tables-a-dspark.md").write_text(sampled.tables(A) + "\n")
(raw / "tables-b-serial.md").write_text(sampled.tables(B) + "\n")
(raw / "tables-compare.md").write_text(sampled.compare_report(A, B) + "\n")

cmp = sampled.compare_runs(A, B)
m = {}
tag = lambda t: f"t{int(t)}"
for boot, run in (("dspark", A), ("serial", B)):
    for (cat, effort, t), s in sampled.summarize_run(run).items():
        p = f"{boot}_{cat}_{effort}_{tag(t)}"
        m[f"{p}_tok_s_mean"] = round(s["tok_s_mean"], 3)
        m[f"{p}_tok_s_median"] = round(s["tok_s_median"], 3)
        m[f"{p}_reasoning_share"] = round(s["reasoning_share"], 4)
        m[f"{p}_ttft_mean_s"] = round(s["ttft_mean_s"], 3)
        m[f"{p}_hit_max_tokens"] = s["hit_max_tokens"]
        if s["tokens_per_round"] is not None:  # the serial boot reports rounds (one token each) but drafts nothing: no acceptance
            m[f"{p}_tokens_per_round"] = round(s["tokens_per_round"], 3)
        if s["acceptance"] is not None:
            m[f"{p}_acceptance"] = round(s["acceptance"], 4)
    tot = sampled.c4_totals(run)
    m[f"{boot}_c4_aggregate_tok_s"] = round(tot["aggregate_tok_s"], 3)
    m[f"{boot}_c4_median_tok_s"] = round(tot["median_tok_s"], 3)
    m[f"{boot}_c4_answered"] = tot["answered"]
for (cat, effort, t), r in cmp["ratios"].items():
    m[f"ratio_{cat}_{effort}_{tag(t)}"] = round(r["mean"], 3)
m["c4_aggregate_ratio"] = round(m["dspark_c4_aggregate_tok_s"] / m["serial_c4_aggregate_tok_s"], 3)
for t, e in cmp["exact"].items():
    m[f"ab_pairs_{tag(t)}"] = e["pairs"]
    m[f"ab_identical_{tag(t)}"] = e["identical"]
# the c = 4 reply of a request against its c = 1 reply (temperature 1.0, effort high), per boot
for boot, run in (("dspark", A), ("serial", B)):
    solo = {r["name"]: r for r in run["c1"] if r["effort"] == "high" and r["temperature"] == 1.0 and not r.get("error")}
    c4 = [r for r in run["c4"] if not r.get("error")]
    m[f"{boot}_c4_equals_c1"] = sum(1 for r in c4 if r["name"] in solo and solo[r["name"]]["ids_sha256"] == r["ids_sha256"])
    m[f"{boot}_c4_compared"] = len(c4)
# request by request (c = 1, both boots answered, both have a rate): the drafted rate over the serial rate
bm = {sampled.request_key(r): r for r in B["c1"]}
per = {}
for r in A["c1"]:
    o = bm.get(sampled.request_key(r))
    if o and not r.get("error") and not o.get("error") and r.get("tok_s") and o.get("tok_s"):
        per.setdefault(r["category"], []).append(r["tok_s"] / o["tok_s"])
for cat, xs in per.items():
    m[f"request_ratio_{cat}_min"] = round(min(xs), 3)
    m[f"request_ratio_{cat}_max"] = round(max(xs), 3)
    m[f"request_ratio_{cat}_slower_than_serial"] = sum(1 for x in xs if x < 1.0)
m["requests_compared"] = sum(len(xs) for xs in per.values())
m["requests_slower_than_serial"] = sum(1 for xs in per.values() for x in xs if x < 1.0)
# loop flags (scripts/loop_probe.py is_loop: distinct 4-gram ratio < 0.5 or a periodic tail) over the chat replies of each boot
for boot, run in (("dspark", A), ("serial", B)):
    for t in sampled.TEMPERATURES:
        rows = [r for r in run["c1"] if r["temperature"] == t and not r.get("error")]
        m[f"{boot}_reply_loops_{tag(t)}"] = sum(1 for r in rows if loop_probe.is_loop(r["ids"]))
        m[f"{boot}_reply_loops_{tag(t)}_of"] = len(rows)
for key, s in LOOP.items():
    m[f"sampled_loop_text_loops_{key}"] = s["loops"]
    m[f"sampled_loop_text_prompts_{key}"] = s["prompts"]
    m[f"sampled_loop_text_mean_distinct4_{key}"] = s["mean_distinct4"]
m["greedy_loop_text_loops"] = greedy_loops["loops_greedy"]
m["greedy_loop_text_prompts"] = greedy_loops["prompts"]
if "b" in LOOP:  # raw text, sampled, seed 1: drafter on against drafter off
    ra = json.loads((raw / "boot-a-loop-sampled.json").read_text())["replies"]
    rb = json.loads((raw / "boot-b-loop-sampled.json").read_text())["replies"]
    m["sampled_loop_text_identical_ab"] = sum(1 for k in ra if k in rb and ra[k]["generated"] == rb[k]["generated"])
    m["sampled_loop_text_pairs_ab"] = sum(1 for k in ra if k in rb)
    m["sampled_loop_text_tok_s_mean_a"] = round(statistics.fmean(r["tok_s"] for r in ra.values()), 3)
    m["sampled_loop_text_tok_s_mean_b"] = round(statistics.fmean(r["tok_s"] for r in rb.values()), 3)

# whole-run counts: every request of both boots answered (the pass criterion)
tests = []
for label, run in (("boot A (SPEC=dspark)", A), ("boot B (SPEC=none)", B)):
    c1 = run["c1"]
    c4 = run["c4"]
    for part, rows in (("c=1: 16 prompts x 2 efforts x 2 temperatures", c1), ("c=4: 16 requests in four waves of four", c4)):
        bad = sum(1 for r in rows if r.get("error"))
        tests.append({"bin": "bench/tools/sampled.py", "filter": f"{label}, {part}: requests answered ({len(rows)} sent)",
                      "passed": len(rows) - bad, "failed": bad})
for key, label in (("a", "boot A (SPEC=dspark)"), ("b", "boot B (SPEC=none)")):
    if key in LOOP:
        tests.append({"bin": "scripts/loop_probe.py", "filter": f"{label}, text sampled at temperature 1.0, seed 1: {LOOP[key]['prompts']} prompts answered "
                      f"(loop flags are a baseline, counted not judged: {LOOP[key]['loops']} flagged)",
                      "passed": LOOP[key]["prompts"], "failed": 0})
expected = 2 * (len(sampled.PROMPTS) * len(sampled.EFFORTS) * len(sampled.TEMPERATURES) + len(sampled.PROMPTS))
sent = sum(t["passed"] + t["failed"] for t in tests if t["bin"].endswith("sampled.py"))
failed = sum(t["failed"] for t in tests)

engine = Path.home() / "projects/rexl3-inkling2-loop"
git = lambda *a, c=engine: subprocess.run(["git", "-C", str(c), *a], capture_output=True, text=True, check=True).stdout.strip()
skel = json.loads((raw / "boot-a-skeleton.json").read_text())
build = (raw / "build.log").read_text()
built = re.search(r"built ([0-9a-f]{40}): rexl3-serve sha256 ([0-9a-f]{64})", build)
if not built or built.group(1) != skel["engine"]["commit"] or built.group(2) != skel["engine"]["sha256"]:
    sys.exit("the build log and the booted engine disagree")
meta = {
    "name": here.name.split("-", 3)[3],
    "gate": "sampled-decode",
    "date": here.name[:10],
    "result": "pass" if failed == 0 and sent == expected else "fail",
    "node": "node-2",
    "recipe": {"base_commit": git("rev-parse", "autoresearch/session-20261122", c=recipe)},
    "pack": skel["pack"],
    "engine": {**skel["engine"], "dirty": bool(git("status", "--porcelain")) or git("rev-parse", "HEAD") != skel["engine"]["commit"]},
    "gpu_clock_sm_max": skel["gpu_clock_sm_max"],
    "config": {"boot_a": skel["config"], "boot_b": {**skel["config"], "SPEC": "none"}},
    "serve": {"boot_a": skel["serve"]["command"], "boot_b": skel["serve"]["command"].replace("--spec dspark", "--spec none")},
    "tests": tests,
    "metrics": m,
    "notes": "Boot A is the shipping default (SPEC=dspark, PREFIX_CACHE=on, VISION=on, profile bootstrap), boot B the same with SPEC=none, on node-2 under the "
             "2,200 MHz cap; both run bench/tools/sampled.py with the same seeds and prompts and scripts/loop_probe.py run --temperature 1.0 --seed 1. "
             "Generated by derive_meta.py from raw/*.json; ab_identical_t0 / ab_identical_t1 count the replies equal id for id between the boots "
             "(a difference is a finding, not a failed measurement).",
}
(here / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(meta["result"], f"{sent}/{expected} requests", f"ab t0 {m['ab_identical_t0']}/{m['ab_pairs_t0']} t1 {m['ab_identical_t1']}/{m['ab_pairs_t1']}")
