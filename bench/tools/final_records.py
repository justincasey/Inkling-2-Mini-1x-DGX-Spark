#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The records of the final run (scripts/final_run.sh): one record per node gate, read from what the run kept.

  final_records.py --run RUN_DIR --node-name node-N [--root RECIPE] [--engine ENGINE_REPO] [--force] GATE

RUN_DIR (bench/runs/final-<stamp>) holds, per phase NAME: NAME.out (the controller's output), stages/NAME.{rc,sec,end} and, for a
`bench` phase, NAME.d/ (the controller's bench/runs/<stamp>-PHASE directory: metrics.txt, gates.txt, logs and JSON). This tool never runs
a node command except `./inkling2mini.sh record NAME`, which reads the installed engine. For GATE it writes
bench/records/<UTC date>-final-<name>/ (meta.json, README.md, raw/: every output the verdict rests on, lab addresses replaced by the
node name), then runs scripts/check_record.py on it as the ladder would. A number is read from a file, never typed; a gate that did not
pass is recorded `result: fail` with the cause in `notes`; nothing is edited afterwards.

Exit 0: the record passes. Exit 1: the record was written and the gate FAILED (or its record does not satisfy the ladder's checks).
Exit 2: no record was written (a usage error, a missing phase that never ran, an existing record directory; --force replaces an
untracked one). The GPU cap: every gate needs the nvidia-smi success line of the cap in the output of the phase that started the
server (or built, or tested); a gate without it is a fail.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import finalize_record as fr  # noqa: E402

LAB = re.compile(r"192\.168\.\d+\.\d+|10\.100\.\d+\.\d+")
SHIPPING_ENV = {"SPEC": "dspark", "PREFIX_CACHE": "on", "VISION": "on", "PROFILE": "bootstrap", "PREFILL_CHUNK": "4096",
                "MAX_MODEL_LEN": "", "SLOTS": "", "KV_TOKENS": "", "EXTRA_FLAGS": "", "EXTRA_ENV": "", "MEMORY_FLOOR_GIB": ""}
CAP_LINE = re.compile(r'GPU clocks set to "\(gpuClkMin (\d+), gpuClkMax (\d+)\)"')


class NoRecord(Exception):
    """Exit 2: nothing was written."""


# ── what the run kept ──
class Run:
    def __init__(self, path: Path):
        self.path = path
        if not (path / "stages").is_dir():
            raise NoRecord(f"{path} is not a run directory of scripts/final_run.sh")
        self.pin = (path / "pin").read_text().strip()
        self.recipe_head = (path / "recipe_head").read_text().strip()
        self.stages = {}
        for line in (path / "stages.tsv").read_text().splitlines() if (path / "stages.tsv").exists() else []:
            n, rc, sec = line.split("\t")
            self.stages[n] = (int(rc), int(sec))

    def ran(self, name):
        return name in self.stages

    def rc(self, name):
        return self.stages[name][0] if name in self.stages else None

    def sec(self, name):
        return self.stages[name][1] if name in self.stages else None

    def out(self, name) -> str:
        p = self.path / f"{name}.out"
        return p.read_text(errors="replace") if p.exists() else ""

    def d(self, name):
        p = self.path / f"{name}.d"
        return p if p.is_dir() else None

    def dfile(self, name, rel) -> str:
        d = self.d(name)
        p = d / rel if d else None
        return p.read_text(errors="replace") if p and p.exists() else ""

    def extra(self, rel) -> str:
        p = self.path / rel
        return p.read_text(errors="replace") if p.exists() else ""

    def metrics(self, name) -> dict:
        """The METRIC lines of a bench phase (its metrics.txt), name -> float."""
        out = {}
        for line in self.dfile(name, "metrics.txt").splitlines():
            m = re.match(r"METRIC (\S+)=(\S+)$", line.strip())
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
        return out

    def log_metrics(self, name, rel) -> dict:
        """METRIC lines of a log of a bench phase (quality, tools and probe print theirs in their own log, not in metrics.txt)."""
        out = {}
        for line in self.dfile(name, rel).splitlines():
            m = re.match(r"METRIC (\S+)=(\S+)$", line.strip())
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
        return out

    def gates(self, name) -> dict:
        """PASS/FAIL lines of a bench phase's gates.txt: id -> True/False (a FAIL anywhere wins)."""
        g = {}
        for line in self.dfile(name, "gates.txt").splitlines():
            m = re.match(r"(PASS|FAIL) (\S+)", line.strip())
            if m:
                g[m.group(2)] = g.get(m.group(2), True) and m.group(1) == "PASS"
        return g

    def start_stages(self, group):
        return [n for n in self.stages if n == f"{group}-start" or n.startswith(f"{group}-restart-")]

    def cap_proof(self, names) -> list[str]:
        lines = []
        for n in names:
            for line in self.out(n).splitlines():
                m = CAP_LINE.search(line)
                if m:
                    lines.append(f"{n}: {line.strip()}")
        return lines


def scrub(text: str, node: str) -> str:
    return LAB.sub(node, text)


def num(x, nd=None):
    if isinstance(x, float) and nd is not None:
        return round(x, nd)
    return x


def median(values):
    return statistics.median(values) if values else None


def counts(text: str) -> tuple[int, int]:
    """passed, failed of a `test result: ok. N passed; M failed` line."""
    m = re.search(r"(\d+) passed; (\d+) failed", text)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# ── a record under construction ──
class Rec:
    def __init__(self, name, gate, run: Run, stages, cap_stages):
        self.name, self.gate, self.run = name, gate, run
        self.stages = stages                # every phase this record rests on (timings, raw outputs)
        self.cap_stages = cap_stages        # phases whose output must carry the cap proof
        self.tests, self.metrics, self.commands, self.problems = [], {}, [], []
        self.notes, self.files, self.extra_meta, self.facts = [], {}, {}, []
        self.min_tests, self.need_metrics, self.same_engine = 0, [], []

    def test(self, bin_, filt, passed, failed=0):
        self.tests.append({"bin": bin_, "filter": filt, "passed": int(passed), "failed": int(failed)})
        if failed:
            self.problems.append(f"{bin_} {filt}: {failed} failing")

    def need(self, cond, why):
        if not cond:
            self.problems.append(why)
        return bool(cond)

    def metric(self, k, v):
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if v is not None:
            self.metrics[k] = v

    def keep(self, dest, src):
        """Copy a run file into raw/ (text, scrubbed). src: a Path or a str of content."""
        self.files[dest] = src

    def keep_stage(self, name, also_dir=True):
        if (self.run.path / f"{name}.out").exists():
            self.files[f"{name}.out"] = self.run.path / f"{name}.out"
        d = self.run.d(name)
        if d and also_dir:
            for f in sorted(d.iterdir()):
                if f.is_file() and f.stat().st_size < 6_000_000:
                    self.files[f"{name}.d/{f.name}"] = f


def stage_ok(rec: Rec, names, what=None):
    for n in names:
        if not rec.run.ran(n):
            raise NoRecord(f"phase {n} never ran: nothing to record for {rec.gate}")
        if rec.run.rc(n) != 0:
            rec.problems.append(f"phase {n} exited {rec.run.rc(n)}" + (f" ({what})" if what else ""))


def seconds_note(rec: Rec):
    return ", ".join(f"{n} {rec.run.sec(n)} s" for n in rec.stages if rec.run.ran(n))


# ── writing ──
def readme(rec: Rec, result: str, meta: dict, node: str) -> str:
    eng = meta["engine"]
    L = [f"# final-{rec.name}: gate `{rec.gate}`: {result.upper()}", "",
         f"Final run, node {node}, engine `{eng['commit']}` (the pin, lab_build {str(eng['lab_build']).lower()}), "
         f"recipe `{meta['recipe']['commit']}`. Written by `bench/tools/final_records.py` from the outputs `scripts/final_run.sh` kept (raw/); "
         "no number in this record was typed.", ""]
    if rec.problems:
        L += ["## Why it failed" if result == "fail" else "## Problems", ""] + [f"- {p}" for p in rec.problems] + [""]
    L += ["## Commands", ""] + [f"- `{c}`" for c in rec.commands] + [""]
    L += ["## Tests", "", "| bin | filter | passed | failed |", "|---|---|---:|---:|"]
    L += [f"| {t['bin']} | {t['filter']} | {t['passed']} | {t['failed']} |" for t in rec.tests] + [""]
    if rec.metrics:
        L += ["## Metrics", "", "| metric | value |", "|---|---:|"] + [f"| `{k}` | {v} |" for k, v in rec.metrics.items()] + [""]
    if rec.facts:
        L += ["## Facts", ""] + [f"- {f}" for f in rec.facts] + [""]
    L += ["## Notes", ""] + [f"- {n}" for n in rec.notes] + [""]
    L += ["## Raw output (`raw/`)", ""] + [f"- `{k}`" for k in sorted(rec.files)] + [""]
    return "\n".join(L)


def validate(rec: Rec, args, root: Path) -> str:
    cmd = [sys.executable, str(root / "scripts/check_record.py"), rec.gate, "--root", str(root), "--engine", args.engine,
           "--need-engine", "--min-tests", str(rec.min_tests)]
    for m in rec.need_metrics:
        cmd += ["--metric", m]
    for g in rec.same_engine:
        cmd += ["--same-engine", g]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else f"check_record exited {r.returncode}"


def write(rec: Rec, args, root: Path) -> int:
    run = rec.run
    node = args.node_name
    name = f"final-{rec.name}"
    # the cap: the nvidia-smi success line of the phases that ran this gate's work
    proof = run.cap_proof(rec.cap_stages)
    cap = re.search(r"^GPU_CLOCK_CAP_MHZ=(\d+)", (root / "config/pins.sh").read_text(), re.M).group(1)
    ok_proof = [p for p in proof if f"gpuClkMin {cap}, gpuClkMax {cap}" in p]
    if not ok_proof:
        rec.problems.append(f"no GPU cap proof ('GPU clocks set to \"(gpuClkMin {cap}, gpuClkMax {cap})\"') in the output of {', '.join(rec.cap_stages)}: "
                            "a run without the cap is not recorded as a pass")
    if proof and len(ok_proof) != len(proof):
        rec.problems.append("a cap line with another clock than the pin")
    rec.keep("cap-proof.txt", "\n".join(proof) + "\n" if proof else "(none)\n")
    # the record skeleton from the controller (engine, pack, configuration, serve command)
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    d = root / "bench/records" / f"{day}-{name}"
    if d.exists():
        if not args.force:
            raise NoRecord(f"{d} exists (--force replaces an untracked one)")
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(d)], capture_output=True)
        if tracked.returncode == 0:
            raise NoRecord(f"{d} is tracked: refusing to replace it")
        shutil.rmtree(d)
    env = {k: v for k, v in os.environ.items() if k not in SHIPPING_ENV}
    env.update(SHIPPING_ENV)
    r = subprocess.run([str(root / "inkling2mini.sh"), "record", name], cwd=root, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not d.exists():
        raise NoRecord(f"inkling2mini.sh record {name} failed: {(r.stdout + r.stderr).strip()[-400:]}")
    meta = json.loads((d / "meta.json").read_text())
    eng = meta["engine"]
    if eng["commit"] != run.pin:
        rec.problems.append(f"the installed engine {eng['commit'][:12]} is not the pin {run.pin[:12]}")
    if eng.get("lab_build"):
        rec.problems.append("the installed engine is a lab build")
    result = "fail" if rec.problems else "pass"
    meta["recipe"] = {"commit": run.recipe_head, "dirty_files": [l[3:] for l in subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.splitlines()]}
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    # raw/
    raw = d / "raw"
    raw.mkdir()
    for dest, src in rec.files.items():
        p = raw / dest
        p.parent.mkdir(parents=True, exist_ok=True)
        text = src if isinstance(src, str) else Path(src).read_text(errors="replace")
        p.write_text(scrub(text, node))
    notes = " ".join(rec.notes + ([f"Problems: {'; '.join(rec.problems)}."] if rec.problems else []))
    (d / "README.md").write_text(scrub(readme(rec, result, meta, node), node))
    spec = {"gate": rec.gate, "result": result, "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tests": rec.tests, "metrics": rec.metrics, "commands": rec.commands, "notes": scrub(notes, node), **rec.extra_meta}
    fr.finalize(d, spec, args.engine, node)
    line = validate(rec, args, root)
    print(f"{rec.gate}: {result} -> bench/records/{day}-{name}: {line}")
    if result == "pass" and line.startswith("record: FAIL"):
        print(f"{rec.gate}: the record does not satisfy the ladder's check; result stays pass in the file, the run reports FAIL")
        return 1
    return 0 if result == "pass" else 1


# ── the gates ──
def g_engine_build(run: Run) -> Rec:
    rec = Rec("engine-tests", "engine-build", run, ["doctor", "build", "engine-host", "engine-goldens"], ["build", "engine-host", "engine-goldens"])
    rec.min_tests = 60
    stage_ok(rec, ["build", "engine-host", "engine-goldens"])
    rec.commands = ["./inkling2mini.sh doctor", "./inkling2mini.sh build   (no revision: the pin of config/pins.sh)",
                    "./inkling2mini.sh engine-test host", "./inkling2mini.sh engine-test goldens   (the Small's kernel hash tests, on the GPU)"]
    for stage, want in (("engine-host", 4), ("engine-goldens", 5)):
        suites = re.findall(r"^ok   (.+?): (test result: ok\..*)$", run.out(stage), re.M)
        rec.need(len(suites) == want, f"{stage}: {len(suites)} suites reported, expected {want}")
        rec.need(f"engine-test {stage.split('-')[1]}: pass" in run.out(stage), f"{stage}: no 'engine-test ...: pass' line")
        for label, res in suites:
            p, f = counts(res)
            rec.test(f"{'host' if stage == 'engine-host' else 'goldens'}: {label}", "", p, f)
            rec.need(p > 0, f"{label}: no test ran")
    host = sum(t["passed"] for t in rec.tests[:4]) if len(rec.tests) >= 4 else 0
    rec.metric("host_tests_passed", host)
    rec.metric("golden_tests_passed", sum(t["passed"] for t in rec.tests[4:]))
    for s, k in (("build", "build_seconds"), ("engine-host", "engine_test_host_seconds"), ("engine-goldens", "engine_test_goldens_seconds")):
        rec.metric(k, run.sec(s))
    m = re.search(r"built (\w{40}): rexl3-serve sha256 (\w{64})", run.out("build"))
    if rec.need(m, "the build did not print its 'built SHA: rexl3-serve sha256 ...' line"):
        rec.need(m.group(1) == run.pin, f"the build produced {m.group(1)[:12]}, not the pin")
        rec.facts.append(f"binary sha256 {m.group(2)}")
    rec.notes.append("Engine built from the pinned commit with `git archive` (cargo --release --locked, sm_121); `engine.lab_build` is false. The goldens run the Small's "
                     "kernel hash tests (attn, ops, moe, vision, dspark) against the checked-in hashes: the drafter hash test reads the Small's pack read-only "
                     "(its mtp.safetensors is bf16, the Mini's is exl3). The GPU cap line precedes each of build, host and goldens.")
    for s in rec.stages:
        rec.keep_stage(s)
    return rec


def log_field(text, key, cast=float):
    m = re.search(rf"\b{key}=([0-9.]+)", text)
    return cast(m.group(1)) if m else None


def g_boot(run: Run) -> Rec:
    rec = Rec("boot", "boot", run, ["doctor", "p-start"], ["p-start"])
    rec.need_metrics = ["boot_s"]
    stage_ok(rec, ["p-start"])
    out, log = run.out("p-start"), run.extra("p-serve-boot.out")
    status, memw = run.extra("p-status-ready.out"), run.extra("p-memwatch-ready.out")
    rec.commands = ["./inkling2mini.sh doctor", "./inkling2mini.sh start   (shipping defaults: SPEC=dspark PREFIX_CACHE=on VISION=on, profile bootstrap, 4 x 262,144)",
                    "./inkling2mini.sh status", "./inkling2mini.sh memwatch report", "./inkling2mini.sh logs 500"]
    m = re.search(r"^boot_s=(\d+)$", out, re.M)
    if rec.need(m, "start printed no boot_s"):
        rec.metric("boot_s", int(m.group(1)))
    rec.need("start: ready at" in out, "start did not report ready")
    for pat, key in ((r"boot phase=ready t=([0-9.]+)", "engine_ready_t_s"), (r"boot phase=weights_loaded t=[0-9.]+ gib=([0-9.]+)", "weights_gib")):
        mm = re.search(pat, log)
        rec.metric(key, float(mm.group(1)) if mm else None)
    loads = re.findall(r"event=inkling2_loop_load layers=\d+ gib=[0-9.]+ s=([0-9.]+)", log)
    rec.metric("weights_load_s", float(loads[-1]) if loads else None)
    mm = re.search(r"MemAvailable (\d+) GiB", out)
    rec.metric("mem_available_before_mib", int(mm.group(1)) * 1024 if mm else None)
    mm = re.search(r"MemAvailable (\d+) MiB now", status)
    rec.metric("mem_available_at_ready_mib", int(mm.group(1)) if mm else None)
    mm = re.search(r"\bmin_mib=(\d+)", memw)
    rec.metric("guard_min_mib", int(mm.group(1)) if mm else None)
    mm = re.search(r"loading_min_mib=(\d+)", memw)
    rec.metric("guard_loading_min_mib", int(mm.group(1)) if mm else None)
    mm = re.search(r"processes: \d+, \S+, (\d+) MiB", status)
    rec.metric("gpu_process_mib", int(mm.group(1)) if mm else None)
    plan = next((l for l in log.splitlines() if re.search(r"event=\w*_plan( |$)", l) and "prefix_cache_plan" not in l), "")
    for key, name, cast in (("kv_pages", "kv_pages", int), ("kv_tokens", "kv_tokens", int), ("kv_gib", "kv_gib", float), ("kv_bytes_per_token", "kv_bytes_per_token", int),
                            ("slots", "slots", int), ("free_after_gib", "planned_free_after_weights_gib", float), ("spare_gib", "planned_spare_gib", float),
                            ("prefix_gib", "planned_prefix_gib", float), ("tower_gib", "planned_tower_gib", float)):
        rec.metric(name, log_field(plan, key, cast))
    rec.need("kv_tokens" in rec.metrics, "no capacity line (event=inkling2_loop_plan) in the boot log")
    rec.test("inkling2mini.sh", "doctor: node idle, MemAvailable, pack at full size, engine installed", 1)
    rec.test("inkling2mini.sh", "start: /health 200, memory guard armed on the server pid, no level=fatal", 1 if "start: ready at" in out and "level=fatal" not in log else 0,
             0 if "level=fatal" not in log else 1)
    rec.need("event=boot phase=ready" in log, "the boot log has no 'event=boot phase=ready'")
    rec.notes.append("First boot after the build of the pin, shipping defaults, drafter dspark, prefix cache on, vision on (tower loaded). boot_s is the controller's start, doctor "
                     "included, until /health answered 200. mem_available_before_mib is the doctor's whole GiB x 1024.")
    for f in ("p-serve-boot.out", "p-status-ready.out", "p-memwatch-ready.out", "p-clock-start.out"):
        if (run.path / f).exists():
            rec.keep(f, run.path / f)
    rec.keep_stage("doctor", False); rec.keep_stage("p-start", False)
    return rec


def g_smoke(run: Run) -> Rec:
    rec = Rec("smoke", "smoke", run, ["p-smoke"], ["p-start"])
    rec.min_tests = 4
    stage_ok(rec, ["p-smoke"])
    out = run.out("p-smoke")
    rec.commands = ["./inkling2mini.sh smoke   (boot p: shipping defaults)"]
    checks = re.findall(r"^(ok|FAIL|skip) +(\S+)", out, re.M)
    for st, n in checks:
        rec.test("bench/tools/smoke.py", n, 1 if st == "ok" else 0, 1 if st == "FAIL" else 0)
    rec.need(re.search(r"^smoke: pass", out, re.M), "smoke.py did not print 'smoke: pass'")
    rec.need({n for _, n in checks} >= {"chat", "completion", "tool", "think-on", "think-off", "image"}, "smoke did not run all six checks")
    rec.metric("smoke_checks_passed", sum(1 for s, _ in checks if s == "ok"))
    rec.metric("smoke_checks_skipped", sum(1 for s, _ in checks if s == "skip"))
    for key, pat in (("think_on_reasoning_tokens", r"reasoning_effort 'high': reasoning_tokens=(\d+)"), ("think_off_reasoning_tokens", r"reasoning_effort 'none': reasoning_tokens=(\d+)")):
        m = re.search(pat, out)
        rec.metric(key, int(m.group(1)) if m else None)
    rec.notes.append("Boot p (the first boot of the final run), SPEC=dspark PREFIX_CACHE=on VISION=on: the image check ran (not skipped).")
    rec.keep_stage("p-smoke", False)
    rec.keep_stage("p-start", False)
    return rec


def exact_counts(out: str) -> tuple[int, int]:
    """exactness.py concurrent: the `gen` lines that say equal / not (the prompt_logprobs lines are refused by this engine)."""
    eq = len(re.findall(r"^gen +\S+ +\(\d+ tokens\): equal", out, re.M))
    ne = len(re.findall(r"^gen +\S+ +\(\d+ tokens\): (?!equal)", out, re.M))
    return eq, ne


def g_exact_basic(run: Run) -> Rec:
    rec = Rec("exact-basic", "exact-basic", run, ["p-exact", "g-exact", "c-exact"], ["p-start", "c-start"])
    rec.min_tests = 4
    stage_ok(rec, ["p-exact", "g-exact", "c-exact"])
    rec.commands = ["boot p: ./inkling2mini.sh bench exactness   (PREFILL_CHUNK=4096)", "boot g: ./inkling2mini.sh bench exactness   (fresh boot, 4096)",
                    "boot c: PREFILL_CHUNK=1000 ./inkling2mini.sh start; PREFILL_CHUNK=1000 ./inkling2mini.sh bench exactness",
                    "python3 bench/tools/exactness.py diff p g; diff p c   (local)"]
    recs = {}
    for g, n in (("p", "4096"), ("g", "4096"), ("c", "1000")):
        d = run.d(f"{g}-exact")
        f = d / f"exact-chunk{n}.json" if d else None
        rec.need(f and f.exists(), f"{g}-exact kept no exact-chunk{n}.json")
        if f and f.exists():
            recs[g] = f
            rec.keep(f"exactness-{g}-chunk{n}.json", f)
        rec.keep_stage(f"{g}-exact", False)
        log = run.dfile(f"{g}-exact", "exact-concurrent.log")
        eq, ne = exact_counts(log)
        rec.keep(f"concurrent-{g}-chunk{n}.log", log)
        rec.test("bench/tools/exactness.py", f"concurrent==solo (boot {g}, chunk {n}, 4 prompts, 4 lanes)", eq, ne)
        rec.need(eq == 4 and ne == 0, f"concurrent == solo on boot {g}: {eq} equal, {ne} not")
    def diff(a, b, label, key):
        if a not in recs or b not in recs:
            return
        r = subprocess.run([sys.executable, str(HERE / "exactness.py"), "diff", str(recs[a]), str(recs[b])], capture_output=True, text=True)
        text = (r.stdout + r.stderr)
        rec.keep(f"diff-{key}.txt", text)
        m = re.match(r"identical: (\d+) replies", text)
        rec.test("bench/tools/exactness.py", label, int(m.group(1)) if (r.returncode == 0 and m) else 0, 0 if r.returncode == 0 else 1)
        rec.need(r.returncode == 0, f"{label}: {text.strip().splitlines()[0] if text.strip() else 'differs'}")
    diff("p", "c", "chunk width: 4096 vs 1000 byte-equal (replies and finish reasons)", "chunk-4096-vs-1000")
    diff("p", "g", "boot to boot: boot p vs boot g (4096) byte-equal", "boot-to-boot")
    if "p" in recs:
        d = json.loads(recs["p"].read_text())
        sizes = d["meta"].get("max_tokens"), [len(r["prompt_ids"]) for r in d["replies"]]
        rec.metric("prompts", len(d["replies"]))
        rec.metric("max_prompt_tokens", max(sizes[1]))
        rec.metric("new_tokens", sizes[0])
        rec.metric("exactness_replies_identical", len(d["replies"]))
        t = d["meta"].get("timing", {})
        rec.metric("solo_xlong_s", t.get("xlong", {}).get("gen_s"))
    rec.notes.append("Four fixed prompts (about 16, 521, 2,030 and 8,334 tokens), 128 greedy tokens, solo, one at a time; boots p and g with PREFILL_CHUNK 4096, boot c with 1000; "
                     "shipping defaults otherwise. prompt_logprobs is refused by this engine, so the comparison rests on the generated ids and finish reasons (the tool says so). "
                     "A record of this gate is taken first on each boot, before anything that reads the prefix cache.")
    return rec


def clock_mhz(run: Run, files):
    vals = []
    for f in files:
        m = re.search(r"^(\d+) MHz", run.extra(f), re.M)
        if m:
            vals.append(int(m.group(1)))
    return max(vals) if vals else None


def pooled(run: Run, stages, key):
    """Every value of metric `key` the stages measured (a run that did not measure it adds nothing)."""
    return [m[key] for m in (run.metrics(s) for s in stages if run.ran(s)) if key in m]


def bench_runs(run: Run, stages):
    """The bench.py stages among `stages` whose own gate line said PASS pin (replies equal to the pinned/solo replies) -> (good, bad)."""
    good, bad = [], []
    for s in stages:
        if not run.ran(s):
            continue
        g = run.gates(s)
        (good if run.rc(s) == 0 and g.get("pin") is True and not any(v is False for v in g.values()) else bad).append(s)
    return good, bad


def decode_stages(run):
    """Every stage that ran the decode phase: boot g's bench all and the later passes, boot p's pin run."""
    names = ["g-all"] + [n for k in range(2, 40) for n in (f"g-pass{k}-decode", f"g-pass{k}-rolling")]
    return [n for n in names if run.ran(n)]


def g_speed_decode(run: Run) -> Rec:
    ds = decode_stages(run)
    rec = Rec("speed-decode", "speed-decode", run, ["g-all"] + ds[1:] + ["g-decode-4k", "p-pin"], ["g-start"])
    rec.need_metrics = ["decode_mean4_tok_s"]
    rec.min_tests = 1
    stage_ok(rec, ["g-all", "g-decode-4k"])
    rec.commands = ["./inkling2mini.sh bench all   (decode phase: warm, 1 stream over four kinds, 4 and 8 at once)"] + \
                   [f"./inkling2mini.sh bench {'decode' if n.endswith('decode') else 'rolling'}   ({n})" for n in ds[1:]] + \
                   ["./inkling2mini.sh bench decode-4k   (c = 1, cold 4,096-token prompt, 3 runs)"]
    good, bad = bench_runs(run, ds)
    for s in bad:
        rec.problems.append(f"{s}: the run or its pin/solo comparison failed")
    meas = [s for s in ds if "decode_c1_tok_s" in run.metrics(s)]
    for s in meas:
        rec.keep_stage(s)
    rec.keep_stage("g-decode-4k")
    rec.test("bench/tools/bench.py", f"decode: four kinds x {len(meas)} runs, every reply equal to the pinned reply (PASS pin), concurrent == solo",
             len([s for s in meas if s in good]), len([s for s in meas if s in bad]))
    vals = lambda k: pooled(run, ds, k)
    for k, name in (("decode_c1_tok_s", "decode_mean4_tok_s"), ("decode_prose_tok_s", None), ("decode_code_tok_s", None), ("decode_structured_tok_s", None),
                    ("decode_json_tok_s", None), ("decode_c4_tok_s", "decode_c4_aggregate_tok_s"), ("decode_c8_tok_s", "decode_c8_aggregate_tok_s"),
                    ("decode_tokens_per_round", None), ("decode_acceptance", None)):
        rec.metric(name or k, round(median(vals(k)), 3) if vals(k) else None)
    rec.metric("decode_runs", len(vals("decode_c1_tok_s")))
    rec.metric("decode_mean4_tok_s_min", min(vals("decode_c1_tok_s")) if vals("decode_c1_tok_s") else None)
    rec.metric("decode_mean4_tok_s_max", max(vals("decode_c1_tok_s")) if vals("decode_c1_tok_s") else None)
    m4 = run.metrics("g-decode-4k")
    k4 = next((k for k in m4 if re.fullmatch(r"decode_default_c1_ctx4096_tok_s", k)), None)
    rec.metric("decode_c1_4k_tok_s", m4.get(k4) if k4 else None)
    rec.need(k4 is not None, "decode-4k printed no decode_default_c1_ctx4096_tok_s")
    rec.need(run.gates("g-decode-4k").get("decode-4k") is True, "gate decode-4k did not pass")
    rec.metric("gpu_clock_mhz", clock_mhz(run, ["g-clock-start.out", "g-clock-end.out"]))
    rec.need("decode_mean4_tok_s" in rec.metrics, "no decode_c1_tok_s measured")
    rec.notes.append("Drafter dspark, prefix cache on, vision on (the shipping defaults), boot g (fresh boot after the pin boot), GPU capped at 2,200 MHz. Medians over every run of "
                     "the phase: `bench all` and the later `bench decode` / `bench rolling` passes (each starts with its own discarded warm-up); decode_mean4_tok_s is the mean over "
                     "prose, code, structured and json of the median run. decode_c4_aggregate_tok_s is the span number (all tokens over the span to the last reply), "
                     "decode_c8_aggregate_tok_s is 8 requests on 4 slots. decode_c1_4k_tok_s is c = 1 on a cold 4,096-token prompt (256 tokens), median of 3.")
    return rec


def g_speed_c4(run: Run) -> Rec:
    ds = decode_stages(run)
    rec = Rec("speed-c4", "speed-c4", run, ["g-all"] + ds[1:], ["g-start"])
    rec.need_metrics = ["decode_c4_rolling_tok_s"]
    rec.min_tests = 1
    stage_ok(rec, ["g-all"])
    meas = [s for s in ds if "decode_c4_rolling_tok_s" in run.metrics(s)]
    rec.commands = ["./inkling2mini.sh bench all   (rolling phase)"] + [f"./inkling2mini.sh bench rolling   ({n})" for n in ds[1:] if n.endswith("rolling")]
    good, bad = bench_runs(run, ds)
    for s in meas:
        rec.keep_stage(s)
        ok = s in good
        rec.test("bench/tools/bench.py", f"{s}: 4 solo and 32 rolling replies, each equal to its solo reply (PASS pin)", 36 if ok else 0, 0 if ok else 1)
        sr = run.metrics(s).get("decode_c4_rolling_short_replies")
        rec.need(sr == 0, f"{s}: {sr} short replies")
        rec.need(run.metrics(s).get("decode_c4_rolling_requests") == 32, f"{s}: not 32 rolling requests")
    vals = lambda k: pooled(run, meas, k)
    for k in ("decode_c4_rolling_tok_s", "decode_c4_steady_tok_s", "decode_c4_rolling_span_s"):
        rec.metric(k, round(median(vals(k)), 3) if vals(k) else None)
    rec.metric("decode_c4_tok_s", round(median(pooled(run, ds, "decode_c4_tok_s")), 3) if pooled(run, ds, "decode_c4_tok_s") else None)
    rec.metric("decode_c4_rolling_requests", 32)
    rec.metric("decode_c4_rolling_short_replies", max(vals("decode_c4_rolling_short_replies")) if vals("decode_c4_rolling_short_replies") else None)
    rec.metric("decode_c4_rolling_tok_s_min", min(vals("decode_c4_rolling_tok_s")) if vals("decode_c4_rolling_tok_s") else None)
    rec.metric("decode_c4_rolling_tok_s_max", max(vals("decode_c4_rolling_tok_s")) if vals("decode_c4_rolling_tok_s") else None)
    rec.metric("rolling_runs", len(meas))
    rec.metric("gpu_clock_mhz", clock_mhz(run, ["g-clock-start.out", "g-clock-end.out"]))
    rec.need("decode_c4_rolling_tok_s" in rec.metrics, "no rolling run measured decode_c4_rolling_tok_s")
    for s in bad:
        rec.problems.append(f"{s}: the run or its comparison failed")
    rec.notes.append("Sustained c = 4: four workers in a closed loop, 32 requests of 512 greedy tokens, drafter dspark as booted (docs/HARNESS-NEXT.md section 7). Medians over "
                     f"{len(meas)} runs ({', '.join(meas)}); the span number (decode_c4_tok_s) and the steady window figure are from the same runs. The ladder pairs this record "
                     "with the newest passing `spec` record of the same engine commit.")
    return rec


def g_speed_prefill(run: Run) -> Rec:
    rec = Rec("speed-prefill", "speed-prefill", run, ["g-prefill-sweep", "g-all", "p-pin"], ["g-start"])
    rec.need_metrics = ["prefill_32k_tok_s"]
    rec.min_tests = 1
    stage_ok(rec, ["g-prefill-sweep", "g-all"])
    rec.commands = ["./inkling2mini.sh bench prefill-sweep   (cold 8K, 32K, 128K, 224K; a unique prompt every pass; 4 passes, the first discarded; 128 tokens decoded after)",
                    "./inkling2mini.sh bench all   (prefill phase: cold 32K, 128K, 224K compared with the pinned replies)"]
    m = run.metrics("g-prefill-sweep")
    for k in ("prefill_8k_tok_s", "prefill_32k_tok_s", "prefill_128k_tok_s", "prefill_224k_tok_s", "prefill_8k_ttft_s", "prefill_32k_ttft_s", "prefill_128k_ttft_s",
              "prefill_224k_ttft_s", "decode_after_8k_tok_s", "decode_after_32k_tok_s", "decode_after_128k_tok_s", "decode_after_224k_tok_s"):
        rec.metric(k, m.get(k))
    rec.need(all(k in m for k in ("prefill_8k_tok_s", "prefill_32k_tok_s", "prefill_128k_tok_s", "prefill_224k_tok_s")), "the sweep did not report all four sizes")
    sweep = run.dfile("g-prefill-sweep", "prefill-sweep.json")
    if sweep:
        d = json.loads(sweep)
        rec.test("bench/tools/prefill_sweep.py", "cold prefill 8K/32K/128K/224K: 3 kept passes each, every pass answered",
                 sum(len(v["passes"]) - 1 for v in d["sizes"].values()), 0)
    rec.need(run.gates("g-prefill-sweep").get("prefill-sweep") is True, "gate prefill-sweep did not pass")
    pf = run.metrics("g-all")
    for k in ("prefill_32k_tok_s", "prefill_128k_tok_s", "prefill_224k_tok_s"):
        rec.metric(f"{k[:-6]}_bench_all_tok_s", pf.get(k))
    good, bad = bench_runs(run, ["g-all"])
    rec.test("bench/tools/bench.py", "bench all prefill phase: cold 32K, 128K, 224K replies equal to the pinned replies (PASS pin)", len(good), len(bad))
    rec.metric("gpu_clock_mhz", clock_mhz(run, ["g-clock-start.out", "g-clock-end.out"]))
    rec.keep_stage("g-prefill-sweep"); rec.keep_stage("g-all", False)
    rec.keep("g-all.d/metrics.txt", run.dfile("g-all", "metrics.txt"))
    rec.keep("g-all.d/bench-prefill.out", run.dfile("g-all", "bench-prefill.out"))
    rec.keep("g-all.d/bench-prefill.err", run.dfile("g-all", "bench-prefill.err"))
    rec.notes.append("Cold prefill over a unique prompt per pass (prefix cache off for the request), median of the 3 kept passes; drafter dspark booted (shipping defaults), "
                     "GPU capped at 2,200 MHz. prefill_*_bench_all_tok_s are the single-pass figures of bench all's prefill phase on the same boot.")
    return rec


def jload(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def g_spec(run: Run) -> Rec:
    rec = Rec("spec", "spec", run, ["g-all"], ["g-start"])
    rec.min_tests = 8
    stage_ok(rec, ["g-all"])
    rec.commands = ["./inkling2mini.sh bench all   (gate spec: bench/tools/gates.py spec; serial = \"draft\": false, drafted = the booted dspark)"]
    d = jload(run.dfile("g-all", "spec.json"))
    rec.need(d, "g-all kept no spec.json")
    rec.need(run.gates("g-all").get("spec") is True, "gate spec did not pass in bench all")
    if d:
        names = [r["name"] for r in d["replies"]]
        rec.test("bench/tools/gates.py spec", f"{', '.join(names)}: solo drafted == serial (ids and finish reason)", sum(1 for r in d["replies"] if r["equal"]),
                 sum(1 for r in d["replies"] if not r["equal"]))
        rec.test("bench/tools/gates.py spec", f"{', '.join(r['name'] for r in d['concurrent'])} at once: drafted == their solo serial replies",
                 sum(1 for r in d["concurrent"] if r["equal"]), sum(1 for r in d["concurrent"] if not r["equal"]))
        a = d["acceptance"]
        rec.test("bench/tools/gates.py spec", "every serial reply drafted 0 tokens", 1 if a["serial"]["drafted"] == 0 else 0, 0 if a["serial"]["drafted"] == 0 else 1)
        rec.test("bench/tools/gates.py spec", "the drafted replies drafted tokens (the drafter ran)", 1 if a["drafted_all"]["drafted"] > 0 else 0, 0 if a["drafted_all"]["drafted"] > 0 else 1)
        rec.need(not d["errors"], f"errors: {d['errors']}")
        rec.metric("spec_pairs_equal", sum(1 for r in d["replies"] + d["concurrent"] if r["equal"]))
        rec.metric("spec_pairs", len(d["replies"]) + len(d["concurrent"]))
        rec.metric("replies_identical", sum(1 for r in d["replies"] + d["concurrent"] if r["equal"]))
        for k in ("drafted", "accepted", "rounds", "acceptance", "tokens_per_round"):
            rec.metric(f"drafted_{k}", a["drafted_all"][k])
        rec.metric("serial_drafted", a["serial"]["drafted"])
        rec.metric("max_prompt_tokens", max(r["prompt_tokens"] for r in d["replies"]))
    rec.metric("boot_s", None)
    rec.notes.append("Boot g, shipping defaults (SPEC=dspark). gates.py spec: serial reference replies (\"draft\": false, prefix cache off) against the drafted replies, five prompts up to "
                     "73,154 tokens solo and four of them at once; 128 greedy tokens each.")
    rec.keep_stage("g-all", False)
    for f in ("spec.json", "spec.log", "gates.txt", "metrics.txt"):
        rec.keep(f"g-all.d/{f}", run.dfile("g-all", f))
    return rec


def g_cache(run: Run) -> Rec:
    rec = Rec("cache", "cache", run, ["g-all"], ["g-start"])
    rec.min_tests = 8
    stage_ok(rec, ["g-all"])
    rec.commands = ["./inkling2mini.sh bench all   (gate cache: bench/tools/gates.py cache, prefix cache on, vision on, drafter on)"]
    d = jload(run.dfile("g-all", "cache.json"))
    rec.need(d, "g-all kept no cache.json")
    rec.need(run.gates("g-all").get("cache") is True, "gate cache did not pass in bench all")
    if d:
        ps, sh = d["prompts"], d["shared"]
        rec.test("bench/tools/gates.py cache", f"cached == uncached, cold and warm ({len(ps)} prompts up to {max(p['prompt_tokens'] for p in ps)} tokens, {len(sh)} shared-prefix pairs)",
                 sum(1 for p in ps + sh if p["equal"]), sum(1 for p in ps + sh if not p["equal"]))
        zero = sum(1 for p in ps if p["uncached"]["cached_tokens"] == 0 and p["cold"]["cached_tokens"] == 0)
        rec.test("bench/tools/gates.py cache", "0 cached tokens on every uncached and cold reply", zero, len(ps) - zero)
        hits = [p for p in ps if p.get("warm", {}).get("hit_expected")]
        ok_hits = sum(1 for p in hits if p["warm"]["cached_tokens"] > 0)
        rec.test("bench/tools/gates.py cache", "the warm reply is served from the cache where the prompt reaches the smallest cut", ok_hits, len(hits) - ok_hits)
        tail_ok = sum(1 for s in sh if s["cached"][1]["cached_tokens"] > 0)
        rec.test("bench/tools/gates.py cache", "the second reply of a shared-prefix pair is served from the cache", tail_ok, len(sh) - tail_ok)
        rec.need(not d["errors"], f"errors: {d['errors']}")
        by = {p["name"]: p for p in ps}
        for n in ("70k", "xlong"):
            if n in by:
                rec.metric(f"gate_warm_cached_tokens_{n}", by[n]["warm"]["cached_tokens"])
        for s in sh:
            rec.metric(f"gate_shared_cached_tokens_{s['name'].split('-')[1]}", s["cached"][1]["cached_tokens"])
        rec.metric("gate_prompts_equal", sum(1 for p in ps + sh if p["equal"]))
        rec.metric("prefix_cache_reach_tokens", d["meta"].get("prefix_cache_reach_tokens"))
    rec.notes.append("Boot g, shipping defaults (prefix cache on, 16 entries, vision on, drafter on). gates.py cache: per prompt an uncached reference, then cold and warm with the cache; two "
                     "shared-prefix pairs; 128 greedy tokens each. The with-images cache gate is the record cache-images.")
    rec.keep_stage("g-all", False)
    for f in ("cache.json", "cache.log", "gates.txt", "metrics.txt"):
        rec.keep(f"g-all.d/{f}", run.dfile("g-all", f))
    return rec


def g_needle(run: Run) -> Rec:
    rec = Rec("needle", "needle", run, ["g-all"], ["g-start"])
    rec.min_tests = 2
    stage_ok(rec, ["g-all"])
    rec.commands = ["./inkling2mini.sh bench all   (gate needle: bench/tools/needle.py --tokens 262144 --max-model-len 262144, depth 0.1 and 0.5)"]
    found = 0
    for dep in ("0.1", "0.5"):
        d = jload(run.dfile("g-all", f"needle-{dep}.json"))
        rec.need(d, f"g-all kept no needle-{dep}.json")
        if not d:
            continue
        lane = d["lanes"][0]
        ok = bool(lane.get("found")) and d.get("pass") is True
        found += ok
        rec.test("bench/tools/needle.py", f"262K window, depth {dep}: the code {lane.get('code')} found in the reply", 1 if ok else 0, 0 if ok else 1)
        key = dep.replace(".", "_")
        rec.metric(f"prefill_s_depth_{key}", lane.get("prefill_s"))
        rec.metric(f"prompt_tokens_depth_{key}", lane.get("prompt_tokens"))
        rec.metric(f"wall_s_depth_{key}", d.get("wall_s"))
        rec.keep(f"needle-{dep}.json", run.dfile("g-all", f"needle-{dep}.json"))
    rec.metric("needles_found", found)
    rec.metric("needles_total", 2)
    rec.metric("prompt_tokens", rec.metrics.get("prompt_tokens_depth_0_1"))
    rec.need(run.gates("g-all").get("needle") is True, "gate needle did not pass in bench all")
    rec.notes.append("Window of profile bootstrap (262,144 tokens), the shipping boot g; max_tokens 1024 (the pack reasons before it answers). The 1,048,576-token needle of profile max is "
                     "not part of the final run (docs/FIT.md, record needle-window).")
    rec.keep("g-all.d/needle.log", run.dfile("g-all", "needle.log"))
    rec.keep("g-all.d/gates.txt", run.dfile("g-all", "gates.txt"))
    return rec


def g_capacity(run: Run) -> Rec:
    rec = Rec("capacity", "capacity", run, ["g-all"], ["g-start"])
    rec.min_tests = 1
    stage_ok(rec, ["g-all"])
    rec.commands = ["./inkling2mini.sh bench all   (gate capacity: the engine's capacity line against the profile)"]
    m = run.metrics("g-all")
    for k in ("lanes_at_max_context", "kv_tokens", "kv_gib", "free_after_gib", "prefix_gib"):
        rec.metric(k, m.get(k))
    plan = run.dfile("g-all", "capacity.txt")
    rec.metric("spare_gib", log_field(plan, "spare_gib"))
    rec.metric("tower_gib", log_field(plan, "tower_gib"))
    rec.metric("slots", log_field(plan, "slots", int))
    rec.metric("kv_bytes_per_token", log_field(plan, "kv_bytes_per_token", int))
    rec.need(run.gates("g-all").get("capacity") is True and "kv_tokens" in rec.metrics, "gate capacity did not pass")
    rec.need(rec.metrics.get("lanes_at_max_context", 0) >= 4 and rec.metrics.get("kv_tokens", 0) >= 4 * 262144, "capacity below 4 lanes x 262,144 tokens")
    rec.test("inkling2mini.sh", "bench check capacity, profile bootstrap: lanes_at_max_context >= 4 and kv_tokens >= 4 x 262,144", 1 if not rec.problems else 0, 1 if rec.problems else 0)
    rec.notes.append("Profile bootstrap (4 slots x 262,144), drafter dspark, prefix cache on, vision on: the plan line of the boot g log (raw/capacity.txt).")
    rec.keep("capacity.txt", plan)
    return rec


def g_vision_smoke(run: Run) -> Rec:
    rec = Rec("vision-smoke", "vision-smoke", run, ["p-smoke", "g-all"], ["p-start", "g-start"])
    rec.min_tests = 8
    stage_ok(rec, ["p-smoke", "g-all"])
    rec.commands = ["./inkling2mini.sh smoke   (boot p: the image check)", "./inkling2mini.sh bench all   (boot g: gates.py vision against bench/reference-vision.json)"]
    v = jload(run.dfile("g-all", "vision.json"))
    rec.need(v, "g-all kept no vision.json")
    rec.need(run.gates("g-all").get("vision") is True and run.gates("g-all").get("vision-record") is True, "gates vision-record / vision did not pass in bench all")
    if v:
        for i in v["images"]:
            ok = bool(i["names_colour"]) and i["finish_reason"] in ("stop", "length")
            rec.test("bench/tools/gates.py vision", f"image {i['name']}: the reply names the colour(s)", 1 if ok else 0, 0 if ok else 1)
        rec.metric(f"image_prompt_tokens_min", min(i["usage"]["prompt_tokens"] for i in v["images"]))
        rec.metric(f"image_prompt_tokens_max", max(i["usage"]["prompt_tokens"] for i in v["images"]))
        rec.metric("image_replies_wall_s_max", max(i["seconds"] for i in v["images"]))
    log = run.dfile("g-all", "vision.log")
    same = re.findall(r"^(text|image)/(\S+): identical", log, re.M)
    rec.test("bench/tools/gates.py diff", "image replies equal the pinned reply-vision.json replies (ids and finish reason)", sum(1 for k, _ in same if k == "image"),
             3 - sum(1 for k, _ in same if k == "image"))
    rec.test("bench/tools/gates.py diff", "text replies equal the pinned reference-vision.json replies", sum(1 for k, _ in same if k == "text"), 4 - sum(1 for k, _ in same if k == "text"))
    rec.need(re.search(r"^ok +image ", run.out("p-smoke"), re.M), "smoke's image check did not pass")
    rec.test("bench/tools/smoke.py", "image: a red square on white is named red (boot p)", 1 if re.search(r"^ok +image ", run.out("p-smoke"), re.M) else 0,
             0 if re.search(r"^ok +image ", run.out("p-smoke"), re.M) else 1)
    rec.metric("vision_replies_identical", len(same))
    plan = run.dfile("g-all", "capacity.txt")
    rec.metric("planned_tower_gib", log_field(plan, "tower_gib"))
    rec.metric("planned_vision_reserve_gib", log_field(plan, "vision_reserve_gib"))
    mm = re.search(r"processes: \d+, \S+, (\d+) MiB", run.extra("p-status-ready.out"))
    rec.metric("gpu_process_mib_vision_on", int(mm.group(1)) if mm else None)
    rec.notes.append("VISION=on at the shipping defaults (drafter and prefix cache on as well). The reference-vision.json replies were pinned by boot p; boot g reproduced them byte for byte "
                     "(token ids and finish reasons of three synthetic images and four text prompts).")
    rec.keep("g-all.d/vision.json", run.dfile("g-all", "vision.json")); rec.keep("g-all.d/vision.log", log)
    rec.keep("g-all.d/gates.txt", run.dfile("g-all", "gates.txt"))
    rec.keep_stage("p-smoke", False)
    return rec


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def g_pin(run: Run, root: Path) -> Rec:
    rec = Rec("pin", "pin", run, ["p-pin", "g-all"], ["p-start", "g-start"])
    rec.min_tests = 3
    stage_ok(rec, ["p-pin", "g-all"])
    rec.commands = ["./inkling2mini.sh bench pin   (boot p: writes or reproduces bench/reference.json and bench/reference-vision.json)",
                    "./inkling2mini.sh bench all   (boot g, a fresh boot: every reply compared with the pinned references)"]
    out = run.out("p-pin")
    ref, vref = root / "bench/reference.json", root / "bench/reference-vision.json"
    rec.need(ref.exists() and vref.exists(), "bench/reference.json or reference-vision.json is missing after the pin run")
    if ref.exists() and vref.exists():
        r, v = json.loads(ref.read_text()), json.loads(vref.read_text())
        rec.extra_meta["references"] = {"bench/reference.json": {"sha256": sha256(ref), "replies": len(r["replies"]), "engine": r["meta"].get("engine")},
                                        "bench/reference-vision.json": {"sha256": sha256(vref), "images": len(v["images"]), "text": len(v["text"])}}
        rec.facts += [f"bench/reference.json sha256 {sha256(ref)}: {len(r['replies'])} replies ({', '.join(sorted(r['replies']))})",
                      f"bench/reference-vision.json sha256 {sha256(vref)}: {len(v['images'])} image replies, {len(v['text'])} text replies"]
        rec.metric("reference_replies", len(r["replies"]))
        rec.metric("reference_vision_replies", len(v["images"]) + len(v["text"]))
        rec.need(r["meta"].get("engine") == run.pin, "bench/reference.json was pinned on another engine than the pin")
        rec.keep("reference.json", ref); rec.keep("reference-vision.json", vref)
    m = re.search(r": (\d+) replies pinned, added (\[.*?\]|none|\S+); (\d+) already pinned ones reproduced", out)
    if rec.need(m, "bench pin printed no 'N replies pinned' line"):
        n, added, rep = int(m.group(1)), m.group(2), int(m.group(3))
        rec.metric("pin_replies", n); rec.metric("pin_replies_reproduced_at_pin", rep)
        rec.test("bench/tools/bench.py --pin", f"decode x4 and cold 32K/128K/224K replies pinned or reproduced ({n}; {rep} reproduced byte for byte, added {added})", n, 0)
        rec.facts.append("pin run: " + m.group(0).strip(" :"))
    rec.need("PINNED:" in out, "bench pin did not end with its PINNED line")
    rec.need(run.gates("p-pin").get("vision-record") is True, "vision-record did not pass at the pin")
    rec.test("bench/tools/bench.py", "pin run: every concurrent (c=4, c=8) reply equal to its solo reply", 1 if run.rc("p-pin") == 0 else 0, 0 if run.rc("p-pin") == 0 else 1)
    g = run.gates("g-all")
    rec.need(g.get("pin") is True, "boot g: bench all did not print PASS pin (replies differ from the pinned ones)")
    rec.test("bench/tools/bench.py --ref", "boot g (fresh boot): decode x4, c=4/c=8/rolling and cold 32K/128K/224K replies equal the pinned replies: PASS pin x2 (decode run and prefill run)",
             sum(1 for l in run.dfile("g-all", "gates.txt").splitlines() if l.strip() == "PASS pin"), sum(1 for l in run.dfile("g-all", "gates.txt").splitlines() if l.strip() == "FAIL pin"))
    rec.test("bench/tools/gates.py diff", "boot g: vision replies equal bench/reference-vision.json", 1 if g.get("vision") else 0, 0 if g.get("vision") else 1)
    rec.notes.append("Boot to boot: references written (or reproduced) by boot p of this run, then every pinned reply reproduced by boot g, a fresh boot of the same build and "
                     "configuration. The references name the engine they were pinned on (meta.engine). bench/reference*.json are committed by the lead with this record.")
    rec.keep_stage("p-pin", True)
    rec.keep("g-all.d/gates.txt", run.dfile("g-all", "gates.txt"))
    rec.keep("g-all.d/bench-decode.out", run.dfile("g-all", "bench-decode.out")); rec.keep("g-all.d/bench-prefill.out", run.dfile("g-all", "bench-prefill.out"))
    return rec


def g_cache_images(run: Run) -> Rec:
    rec = Rec("cache-images", "cache-images", run, ["g-cache-images"], ["g-start"])
    rec.min_tests = 8
    stage_ok(rec, ["g-cache-images"])
    rec.commands = ["./inkling2mini.sh bench cache-images   (bench/tools/cache_images.py: cached == uncached for prompts that carry images; prefix cache, vision and drafter on)"]
    log = run.dfile("g-cache-images", "cache-images.log")
    d = jload(run.dfile("g-cache-images", "cache-images.json"))
    rec.need(d, "g-cache-images kept no cache-images.json")
    rec.need(run.gates("g-cache-images").get("cache-images") is True, "gate cache-images did not pass")
    scen = {"after-text": "filler, image, question: cold, warm (text before the image served)", "other-image": "another image: its own reply, the image not served",
            "text-first": "text cached first, then the same text with an image", "image-first": "image first: cold, warm, another question",
            "two-images": "two images: cold, warm, only the second changed", "concurrent": "two image prompts at once"}
    for n, what in scen.items():
        m = re.search(rf"^{n}: .*$", log, re.M)
        line = m.group(0) if m else ""
        ok = bool(m) and "False" not in line and "equal" in line
        rec.test("bench/tools/cache_images.py", f"{n}: {what}; every cached reply equals its uncached reference", 1 if ok else 0, 0 if ok else 1)
    mt = run.metrics("g-cache-images")
    hits, lookups = mt.get("cache_images_hits_total", 0), mt.get("cache_images_lookups_total", 0)
    rec.test("bench/tools/cache_images.py", f"the cache served {int(hits)} of {int(lookups)} lookups and saved {int(mt.get('cache_images_tokens_saved_total', 0))} tokens", 1 if hits > 0 else 0, 0 if hits > 0 else 1)
    rec.test("bench/tools/cache_images.py", "no request error", 1 if d and not d["errors"] else 0, 0 if d and not d["errors"] else 1)
    rec.metric("cache_images_scenarios", len(scen))
    rec.metric("cache_images_hits", int(hits)); rec.metric("cache_images_lookups", int(lookups))
    rec.metric("cache_images_tokens_saved", int(mt.get("cache_images_tokens_saved_total", 0)))
    rec.metric("cache_images_evictions", int(mt.get("cache_images_evictions_total", 0)))
    for key, pat in (("cache_images_after_text_warm_cached", r"^after-text: .*warm cached (\d+)"), ("cache_images_image_first_warm_cached", r"^image-first: cold \d+ warm (\d+)"),
                     ("cache_images_two_images_warm_cached", r"^two-images: cold \d+ warm (\d+)")):
        m = re.search(pat, log, re.M)
        rec.metric(key, int(m.group(1)) if m else None)
    rec.notes.append("Boot g, shipping defaults (SPEC=dspark PREFIX_CACHE=on VISION=on): the drafter serves the image prompts and its state rides in the cache snapshots. Every cached reply "
                     "equal to its uncached reference, id for id (48 greedy tokens).")
    rec.keep_stage("g-cache-images")
    return rec


def g_api_test(run: Run) -> Rec:
    rec = Rec("api-test", "api-test", run, ["g-probe", "g-api"], ["g-start"])
    rec.min_tests = 15
    stage_ok(rec, ["g-probe", "g-api"])
    rec.commands = ["./inkling2mini.sh bench probe   (bench/probe.py: the OpenAI surface, the drafter on, one image)", "./inkling2mini.sh bench api   (tests/api_conformance.py)"]
    for stage, log, bin_ in (("g-api", "api-test.log", "tests/api_conformance.py"), ("g-probe", "probe.log", "bench/probe.py")):
        text = run.dfile(stage, log)
        rows = re.findall(r"^(PASS|FAIL) ([^:\n]+)", text, re.M)
        for st, n in rows:
            rec.test(bin_, n.strip(), 1 if st == "PASS" else 0, 0 if st == "PASS" else 1)
        rec.need(rows, f"{stage}: no PASS/FAIL line in {log}")
    api = run.dfile("g-api", "api-test.log")
    m = re.search(r"api_conformance: (\d+) PASS, (\d+) FAIL", api)
    if rec.need(m, "api_conformance printed no summary"):
        rec.metric("api_checks_passed", int(m.group(1))); rec.metric("api_checks_failed", int(m.group(2)))
    rec.need(re.search(r"^probe: OK", run.dfile("g-probe", "probe.log"), re.M), "probe did not end 'probe: OK'")
    pm = run.log_metrics("g-probe", "probe.log")
    rec.metric("probe_checks_passed", sum(1 for t in rec.tests if t["bin"] == "bench/probe.py" and t["passed"]))
    for k, v in pm.items():
        rec.metric(k, v)
    boot = re.search(r"^boot_s=(\d+)$", run.out("g-start"), re.M)
    rec.metric("boot_s", int(boot.group(1)) if boot else None)
    rec.notes.append("Boot g, shipping defaults: the drafter is on, so the probe's spec and serial checks run in their drafter-on form (docs: probe.py --spec auto reads the state from /metrics).")
    rec.keep_stage("g-probe"); rec.keep_stage("g-api")
    return rec


ROOT = HERE.parent.parent


def loop_summ(j):
    import importlib.util
    spec = importlib.util.spec_from_file_location("loop_probe", ROOT / "scripts/loop_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("loop_probe", mod)
    spec.loader.exec_module(mod)
    return mod.summarize(j)


def loop_metrics_and_tests(rec: Rec, run: Run, lr: dict, report_text: str):
    """Tests and metrics of the three loop-probe modes (text: a baseline count; reasoning: no loop; agentic: no destructive call)."""
    if not all(lr.values()):
        return
    t, r, a = (loop_summ(lr[m]) for m in ("text", "reasoning", "agentic"))
    rec.test("scripts/loop_probe.py", f"text, greedy 512 tokens: {t['prompts']} prompts answered (loop flags are a baseline, counted not judged: {t['loops']} flagged)", t["prompts"], 0)
    rec.test("scripts/loop_probe.py", "reasoning: no loop; finished or cut by the budget", r["prompts"] - r["loops"], r["loops"])
    rec.test("scripts/loop_probe.py", "agentic: no destructive tool call", a["prompts"] - a["prompts_with_destructive_call"], a["prompts_with_destructive_call"])
    rec.need(r["loops"] == 0, f"reasoning mode: {r['loops']} loops")
    rec.need(a["destructive_calls"] == 0, f"agentic mode: {a['destructive_calls']} destructive calls")
    for k, v in (("loop_text_prompts", t["prompts"]), ("loop_text_loops", t["loops"]), ("loop_text_mean_distinct4", t["mean_distinct4"]), ("loop_reasoning_prompts", r["prompts"]),
                 ("loop_reasoning_loops", r["loops"]), ("loop_reasoning_unfinished", r["unfinished"]), ("loop_reasoning_tokens_mean", r["reasoning_tokens_mean"]),
                 ("loop_reasoning_tokens_max", r["reasoning_tokens_max"]), ("loop_agentic_prompts", a["prompts"]), ("loop_agentic_destructive_calls", a["destructive_calls"])):
        rec.metric(k, v)
    rec.keep("loop-report.txt", report_text)


def g_loop_probe(run: Run) -> Rec:
    rec = Rec("loop-probe", "loop-probe", run, ["g-loop"], ["g-start"])
    rec.min_tests = 3
    stage_ok(rec, ["g-loop"])
    rec.commands = ["./inkling2mini.sh bench loop   (scripts/loop_probe.py run / reasoning / agentic)", "python3 scripts/loop_probe.py compare <earlier loop-greedy.json> loop-text.json"]
    lr = {m: jload(run.dfile("g-loop", f"loop-{m}.json")) for m in ("text", "reasoning", "agentic")}
    for m, j in lr.items():
        rec.need(j, f"g-loop kept no loop-{m}.json")
    rec.need(run.gates("g-loop").get("loop-probe") is True, "gate loop-probe did not pass")
    loop_metrics_and_tests(rec, run, lr, run.dfile("g-loop", "loop-probe.log"))
    earlier = ROOT / "bench/records/2026-11-23-loop-probe/loop-greedy.json"
    if lr["text"] and earlier.exists():
        new = run.path / "g-loop.d" / "loop-text.json"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/loop_probe.py"), "compare", str(earlier), str(new)], capture_output=True, text=True)
        rec.keep("compare-earlier-record.txt", r.stdout + r.stderr)
        m = re.search(r"(\d+) prompts: identical replies (\d+)", r.stdout)
        if rec.need(m, "loop_probe compare printed no summary"):
            n, same = int(m.group(1)), int(m.group(2))
            rec.metric("greedy_identical_to_earlier_record", same); rec.metric("greedy_prompts_compared", n)
            rec.test("scripts/loop_probe.py", f"text greedy: {n} replies compared with the replies of record 2026-11-23-loop-probe (engine 918e685), byte for byte", same, n - same)
    else:
        rec.notes.append("The earlier record's loop-greedy.json was not found: no comparison with it.")
    rec.metric("loops_greedy", rec.metrics.get("loop_text_loops")); rec.metric("prompts", rec.metrics.get("loop_text_prompts"))
    rec.metric("mean_distinct4_greedy", rec.metrics.get("loop_text_mean_distinct4"))
    rec.notes.append("Boot g, shipping defaults, drafter on. Text mode is a baseline (the tool records loop flags and judges nothing); the sampled temperature-1.0 run of the earlier record is not "
                     "part of the final run. Reasoning mode: 12 chat prompts at effort high, 4,096-token budget; agentic mode: 6 prompts with tools offered, a destructive one among them.")
    rec.keep_stage("g-loop")
    return rec


def g_quality_tools(run: Run) -> Rec:
    rec = Rec("quality-tools", "quality-tools", run, ["g-quality", "g-tools", "g-loop"], ["g-start"])
    rec.min_tests = 1
    rec.commands = ["./inkling2mini.sh bench quality   (bench/quality_gate.py: 60 exact-answer questions, effort none)",
                    "./inkling2mini.sh bench tools   (bench/toolgate.py: schema cases and the destructive-action probes)",
                    "./inkling2mini.sh bench loop   (scripts/loop_probe.py: text, reasoning and agentic modes)"]
    stage_ok(rec, ["g-quality"])
    for s in ("g-tools", "g-loop"):
        if not run.ran(s):
            raise NoRecord(f"phase {s} never ran: nothing to record for quality-tools")
    qm, tm = run.log_metrics("g-quality", "quality.log"), run.log_metrics("g-tools", "tools.log")
    qc, qt = int(qm.get("quality_correct", 0)), int(qm.get("quality_total", 0))
    rec.test("bench/quality_gate.py", "60 exact-answer questions, effort none (the baseline tolerance of the tool applies)", qc, qt - qc)
    rec.need(run.gates("g-quality").get("quality") is True and qt == 60, f"quality gate: {qc}/{qt}, gate {'passed' if run.gates('g-quality').get('quality') else 'FAILED'}")
    tp, tt = int(tm.get("toolgate_pass", 0)), int(tm.get("toolgate_total", 0))
    tlog = run.dfile("g-tools", "tools.log")
    schema = re.findall(r"^(PASS|FAIL) schema:(\S+)", tlog, re.M)
    sp = sum(1 for s, _ in schema if s == "PASS")
    rec.test("bench/toolgate.py", f"schema cases ({len(schema)}; misses listed in raw/g-tools.d/tools.log)", sp, len(schema) - sp)
    dest = re.findall(r"^(PASS|FAIL) destructive:(\S+)", tlog, re.M)
    dp = sum(1 for s, _ in dest if s == "PASS")
    rec.test("bench/toolgate.py", "destructive probes: safe requests with no destructive call", dp, len(dest) - dp)
    misses = [n for s, n in schema if s == "FAIL"]
    if misses:
        rec.problems.append(f"toolgate: {len(misses)} of {len(schema)} schema cases fail ({', '.join(n.rstrip(':') for n in misses)}); the tool's own verdict is FAIL (--min-pass {tt})")
    rec.need(run.gates("g-tools").get("tools") is True or bool(misses), "bench tools failed for a reason other than schema cases")
    rec.need(not tm.get("toolgate_unrequested_destructive") and tm.get("toolgate_invalid_calls") == 0, "toolgate: an unrequested destructive call or an invalid call")
    # loop probe: baselines (the tool states it records them and judges nothing), except what it does judge
    lm = run.dfile("g-loop", "metrics.txt")
    lr = {}
    for mode in ("text", "reasoning", "agentic"):
        j = jload(run.dfile("g-loop", f"loop-{mode}.json"))
        rec.need(j, f"g-loop kept no loop-{mode}.json")
        lr[mode] = j
    rep = run.dfile("g-loop", "loop-probe.log")
    rec.keep_stage("g-quality"); rec.keep_stage("g-tools"); rec.keep_stage("g-loop")
    loop_metrics_and_tests(rec, run, lr, rep)
    rec.need(run.gates("g-loop").get("loop-probe") is True, "gate loop-probe did not pass")
    rec.metric("quality_correct", qc); rec.metric("quality_total", qt)
    rec.metric("quality_reasoning_tokens_mean", qm.get("quality_reasoning_tokens_mean"))
    for k in ("toolgate_pass", "toolgate_total", "toolgate_invalid_calls", "toolgate_unrequested_destructive", "toolgate_explicit_called", "toolgate_reasoning_tokens_mean"):
        rec.metric(k, tm.get(k))
    rec.notes.append("Boot g, shipping defaults, drafter on. Quality: 60 exact-answer questions at reasoning effort none. Tools: 28 schema cases (the tool's verdict needs all 28) and "
                     "16 destructive-action probes. Loop probe: text mode (greedy 512 tokens) is a baseline whose loop flags are counted, not judged (the tool says so); reasoning mode must "
                     "finish or hit its budget without a loop; agentic mode must make no destructive call. The earlier record quality-tools counted the text-mode loop flags as failing tests; "
                     "here they are a metric (loop_text_loops) and not a test failure.")
    return rec


def g_speed_baseline(run: Run) -> Rec:
    ds = decode_stages(run)
    rec = Rec("speed-baseline", "speed-baseline", run, ["g-all", "p-pin"] + ds[1:], ["g-start", "p-start"])
    rec.need_metrics = ["decode_c1_tok_s", "prefill_32k_tok_s"]
    stage_ok(rec, ["g-all", "p-pin"])
    rec.commands = ["./inkling2mini.sh bench pin   (boot p: decode, c=4/c=8, cold 32K/128K/224K)", "./inkling2mini.sh bench all   (boot g: decode, c=4/c=8, rolling, cold 32K/128K/224K)"] + \
                   [f"./inkling2mini.sh bench {'decode' if n.endswith('decode') else 'rolling'}   ({n})" for n in ds[1:]]
    dec = [s for s in ["p-pin"] + ds if "decode_c1_tok_s" in run.metrics(s)]
    pre = [s for s in ["p-pin", "g-all"] if "prefill_32k_tok_s" in run.metrics(s)]
    good, bad = bench_runs(run, ["p-pin", "g-all"] + ds[1:])
    # the pin run prints no PASS pin line (it pins); its comparison is the concurrent == solo requirement and its exit code
    pin_ok = run.rc("p-pin") == 0
    for s in bad:
        if s != "p-pin":
            rec.problems.append(f"{s}: the run or its comparison failed")
    if not pin_ok:
        rec.problems.append("p-pin exited non-zero")
    nd = len([s for s in dec if (s in good) or (s == "p-pin" and pin_ok)])
    rec.test("bench/tools/bench.py", f"decode x{len(dec)} runs: every reply produced and equal to the pinned/solo reply, concurrent == solo", nd, len(dec) - nd)
    npf = len([s for s in pre if (s in good) or (s == "p-pin" and pin_ok)])
    rec.test("bench/tools/bench.py", f"prefill x{len(pre)} runs: cold 32K, 128K, 224K replies equal to the pinned replies", npf, len(pre) - npf)
    med = lambda k, ss, nd_=3: (round(median([v for v in pooled(run, ss, k)]), nd_) if pooled(run, ss, k) else None)
    for k in ("decode_c1_tok_s", "decode_c4_tok_s", "decode_c8_tok_s", "decode_tokens_per_round", "decode_acceptance"):
        rec.metric(k, med(k, dec))
    for k in ("prefill_32k_tok_s", "prefill_128k_tok_s", "prefill_224k_tok_s", "prefill_32k_s", "prefill_128k_s", "prefill_224k_s", "decode_ctx32k_tok_s", "decode_ctx128k_tok_s", "decode_ctx224k_tok_s"):
        rec.metric(k, med(k, pre, 4))
    rec.metric("gpu_clock_mhz", clock_mhz(run, ["g-clock-start.out", "g-clock-end.out", "p-clock-start.out", "p-clock-end.out"]))
    rec.metric("passes_decode", len(dec)); rec.metric("passes_prefill", len(pre))
    for k in ("decode_c1_tok_s", "prefill_32k_tok_s"):
        rec.need(k in rec.metrics, f"no {k} measured")
    for s in set(dec + pre):
        rec.keep_stage(s, False)
        for f in ("metrics.txt", "gates.txt"):
            rec.keep(f"{s}.d/{f}", run.dfile(s, f))
    rec.notes.append("Shipping defaults (SPEC=dspark PREFIX_CACHE=on VISION=on), profile bootstrap, GPU capped at 2,200 MHz. Medians over every run that measured the metric: decode over "
                     f"{', '.join(dec)}; prefill over {', '.join(pre)} (each run starts with its own discarded warm-up). The numbers are single-boot figures per boot; the spread is in the per-run metrics.txt.")
    return rec


def g_memory(run: Run) -> Rec:
    rec = Rec("memory", "memory", run, ["g-all", "g-profile", "g-memwatch"], ["g-start"])
    rec.min_tests = 2
    stage_ok(rec, ["g-all", "g-profile", "g-memwatch"])
    rec.commands = ["./inkling2mini.sh memwatch begin; ./inkling2mini.sh bench all   (gate memory: the guard's minimum against the engine floor, memcheck.py)",
                    "./inkling2mini.sh bench profile   (capacity line; GPU MiB and MemAvailable at ready; one cold 262,144-token prompt; 4 concurrent ones; a decode burst; node-side sampler)",
                    "./inkling2mini.sh memwatch report   (after every gate of the boot: the guard's minimum since it armed, loading included)"]
    floor = 8192
    final = run.out("g-memwatch")
    m = {k: v for k, v in (kv.split("=", 1) for kv in re.findall(r"\b\w+=\S+", final))}
    rec.need(m.get("guard") == "running", f"the guard was '{m.get('guard')}' at the final report (a stopped guard watched nothing)")
    mn = int(m["min_mib"]) if m.get("min_mib", "").isdigit() else None
    rec.need(mn is not None, "the final memwatch report has no min_mib")
    if mn is not None:
        rec.need(mn >= floor, f"guard minimum {mn} MiB is under the engine floor {floor} MiB")
        rec.test("inkling2mini.sh", f"memwatch report after the whole boot: guard still running, minimum MemAvailable {mn} MiB >= the {floor} MiB floor", 1 if (mn >= floor and m.get("guard") == "running") else 0,
                 0 if (mn >= floor and m.get("guard") == "running") else 1)
        rec.metric("guard_min_mib", mn); rec.metric("mem_available_min_mib", mn); rec.metric("floor_mib", floor); rec.metric("headroom_mib", mn - floor)
    for k in ("loading_min_mib", "ready_min_mib"):
        if m.get(k, "").isdigit():
            rec.metric(f"guard_{k}", int(m[k]))
    ga, gp = run.gates("g-all"), run.gates("g-profile")
    rec.test("bench/tools/memcheck.py", "bench all: guard minimum over the floor, guard running", 1 if ga.get("memory") is True else 0, 0 if ga.get("memory") is True else 1)
    rec.need(ga.get("memory") is True, "gate memory did not pass in bench all")
    rec.test("bench/tools/memcheck.py", "bench profile: guard minimum over the floor after one and four concurrent cold 262,144-token prompts and a decode burst", 1 if gp.get("profile") is True else 0,
             0 if gp.get("profile") is True else 1)
    rec.need(gp.get("profile") is True, "gate profile did not pass")
    solo = jload(run.dfile("g-profile", "window-solo.json")); lanes = jload(run.dfile("g-profile", "window-lanes.json"))
    if solo:
        f = sum(1 for l in solo["lanes"] if l.get("found"))
        rec.test("bench/tools/needle.py", "profile: one cold prompt of the window (262,144 tokens), the code found", f, len(solo["lanes"]) - f)
        rec.metric("window_solo_prefill_s", solo["lanes"][0].get("prefill_s")); rec.metric("window_solo_prompt_tokens", solo["lanes"][0].get("prompt_tokens"))
    else:
        rec.problems.append("no window-solo.json: the solo window prompt did not run")
    if lanes:
        f = sum(1 for l in lanes["lanes"] if l.get("found"))
        rec.test("bench/tools/needle.py", f"profile: {len(lanes['lanes'])} concurrent cold prompts of the window, every code found", f, len(lanes["lanes"]) - f)
        rec.metric("window_lanes_found", f); rec.metric("window_lanes_prefill_s_max", max(l.get("prefill_s") or 0 for l in lanes["lanes"]))
    else:
        rec.problems.append("no window-lanes.json: the concurrent window prompts did not run")
    ready = run.dfile("g-profile", "ready.txt")
    for key, pat in (("gpu_ready_mib", r"gpu_mib=(\d+)"), ("avail_ready_mib", r"avail_mib=(\d+)")):
        mm = re.search(pat, ready)
        rec.metric(key, int(mm.group(1)) if mm else None)
    pm = run.metrics("g-profile")
    for k in ("sampled_gpu_peak_mib", "sampled_avail_min_mib", "kv_tokens", "lanes_at_max_context", "kv_gib", "free_after_gib", "prefix_gib", "mem_min_gib"):
        rec.metric(k, pm.get(k))
    rec.metric("bench_all_mem_min_gib", run.metrics("g-all").get("mem_min_gib"))
    if rec.metrics.get("gpu_ready_mib") and rec.metrics.get("sampled_gpu_peak_mib"):
        rec.metric("gpu_growth_mib", rec.metrics["sampled_gpu_peak_mib"] - rec.metrics["gpu_ready_mib"])
    rec.need("sampled_gpu_peak_mib" in rec.metrics, "the profile's node-side sampler reported no GPU peak")
    rec.notes.append("Boot g, shipping defaults (drafter, prefix cache 16 entries, vision tower), GPU capped at 2,200 MHz, the memory guard armed on the server pid for the whole boot. "
                     "guard_min_mib is the guard's lowest MemAvailable sample since it armed (loading included) over everything the boot ran: the gates of bench all, the sweeps, the quality, "
                     "tool and loop probes and the profile bracket; the engine's floor is 8 GiB (8,192 MiB). sampled_* are the node-side 5 s sampler over the profile only.")
    for f_ in ("g-memwatch.out",):
        rec.keep(f_, run.path / f_)
    for f_ in ("g-memwatch-ready.out", "g-clock-start.out", "g-clock-end.out"):
        if (run.path / f_).exists():
            rec.keep(f_, run.path / f_)
    rec.keep_stage("g-profile"); rec.keep_stage("g-all", False)
    for f_ in ("memwatch.txt", "memory.log", "metrics.txt", "gates.txt", "memwatch-begin.txt"):
        rec.keep(f"g-all.d/{f_}", run.dfile("g-all", f_))
    return rec


GATES = {"engine-build": g_engine_build, "boot": g_boot, "smoke": g_smoke, "exact-basic": g_exact_basic, "spec": g_spec, "cache": g_cache,
         "cache-images": g_cache_images, "vision-smoke": g_vision_smoke, "needle": g_needle, "api-test": g_api_test, "speed-decode": g_speed_decode,
         "speed-prefill": g_speed_prefill, "speed-c4": g_speed_c4, "capacity": g_capacity, "quality-tools": g_quality_tools,
         "pin": g_pin, "speed-baseline": g_speed_baseline, "loop-probe": g_loop_probe, "memory": g_memory}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gate")
    ap.add_argument("--run", required=True)
    ap.add_argument("--node-name", required=True)
    ap.add_argument("--root", default=str(HERE.parent.parent))
    ap.add_argument("--engine", default=os.path.expanduser("~/projects/rexl3-inkling2-loop"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"node-[1-4]", args.node_name):
        print("final_records: --node-name must be node-1..node-4", file=sys.stderr)
        return 2
    root = Path(args.root).resolve()
    try:
        run = Run(Path(args.run).resolve())
        if args.gate not in GATES:
            raise NoRecord(f"unknown gate {args.gate} (known: {', '.join(sorted(GATES))})")
        fn = GATES[args.gate]
        rec = fn(run, root) if fn.__code__.co_argcount == 2 else fn(run)
        return write(rec, args, root)
    except NoRecord as e:
        print(f"final_records: {args.gate}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
