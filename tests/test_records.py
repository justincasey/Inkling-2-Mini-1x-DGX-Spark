#!/usr/bin/env python3
"""Unit check of the ladder's evidence rule: scripts/check_record.py decides a node rung from the newest record of its gate.

What the ladder relies on, each against throwaway git repositories (no node, no engine build):
  - two records of one day are ordered by when they were committed, not by their directory names (a speed record made after
    an older one of the same day must win even when its name sorts first);
  - a newer failing record outranks an older passing one;
  - a record counts only when its engine commit is in the engine head and its tree was clean;
  - a metric floor holds, and a lab address anywhere in the record directory (raw outputs included) fails it;
  - `--same-engine`: a speed record counts only beside a passing exactness record of the very same engine commit.

    python3 tests/test_records.py

Ends with the line `test_records: N passed, 0 failed` (exit 1 and `F failed` when a test fails).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "scripts", "check_record.py")
IDENT = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]


def git(repo: str, *args: str, when: str | None = None) -> str:
    env = dict(os.environ)
    if when:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when
    r = subprocess.run(["git", "-C", repo, *IDENT, *args], capture_output=True, text=True, env=env, check=True)
    return r.stdout.strip()


def new_repo(base: str, name: str) -> str:
    path = os.path.join(base, name)
    os.makedirs(path)
    git(path, "init", "-q", "-b", "main")
    return path


def engine_repo(base: str) -> tuple[str, str, str]:
    """An engine repo whose HEAD holds c1 and c2; c3 sits on a side branch (not in HEAD)."""
    e = new_repo(base, "engine")
    with open(os.path.join(e, "a"), "w") as f:
        f.write("1")
    git(e, "add", "-A")
    git(e, "commit", "-q", "-m", "c1")
    c1 = git(e, "rev-parse", "HEAD")
    git(e, "checkout", "-q", "-b", "side")
    with open(os.path.join(e, "a"), "w") as f:
        f.write("side")
    git(e, "commit", "-q", "-am", "c3")
    c3 = git(e, "rev-parse", "HEAD")
    git(e, "checkout", "-q", "main")
    with open(os.path.join(e, "a"), "w") as f:
        f.write("2")
    git(e, "commit", "-q", "-am", "c2")
    return e, c1, c3


def record(repo: str, dirname: str, gate: str, when: str, *, date: str = "2026-11-23", result: str = "pass", commit: str | None = None,
           dirty: bool = False, metrics: dict | None = None, tests: list | None = None, raw: str = "ok") -> None:
    d = os.path.join(repo, "bench", "records", dirname)
    os.makedirs(d, exist_ok=True)
    meta = {"name": dirname, "gate": gate, "date": date, "result": result, "node": "node-1",
            "tests": tests if tests is not None else [{"bin": "t", "filter": "", "passed": 3, "failed": 0}],
            "metrics": metrics or {}}
    if commit is not None:
        meta["engine"] = {"commit": commit, "dirty": dirty, "version": "x"}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("# r\n")
    with open(os.path.join(d, "raw.txt"), "w") as f:
        f.write(raw)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", dirname, when=when)


def run(repo: str, engine: str, gate: str, *args: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, CHECK, gate, "--root", repo, "--engine", engine, *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def test_same_day_records_are_ordered_by_commit_time_not_by_name() -> None:
    with tempfile.TemporaryDirectory() as base:
        repo = new_repo(base, "recipe")
        record(repo, "2026-11-23-zzz-first", "speed", "2026-11-23T10:00:00Z", metrics={"tok_s": 100})
        record(repo, "2026-11-23-aaa-second", "speed", "2026-11-23T11:00:00Z", metrics={"tok_s": 200})
        rc, out = run(repo, "", "speed", "--show-metric", "tok_s")
        assert (rc, out) == (0, "200"), (rc, out)
        # and the other way round: the name sorts last AND it is the newest
        record(repo, "2026-11-23-zzz-third", "speed", "2026-11-23T12:00:00Z", metrics={"tok_s": 300})
        assert run(repo, "", "speed", "--show-metric", "tok_s") == (0, "300")
        # a later date outranks any commit time of an earlier date
        record(repo, "2026-11-22-old", "speed", "2026-11-25T00:00:00Z", date="2026-11-22", metrics={"tok_s": 1})
        assert run(repo, "", "speed", "--show-metric", "tok_s") == (0, "300")


def test_the_record_is_ordered_by_the_commit_that_added_it() -> None:
    with tempfile.TemporaryDirectory() as base:
        repo = new_repo(base, "recipe")
        record(repo, "2026-11-23-old", "speed", "2026-11-23T10:00:00Z", metrics={"tok_s": 100})
        record(repo, "2026-11-23-new", "speed", "2026-11-23T11:00:00Z", metrics={"tok_s": 200})
        # the OLDER record is edited in a later commit: it must not become the newest
        path = os.path.join(repo, "bench", "records", "2026-11-23-old", "meta.json")
        meta = json.load(open(path))
        meta["superseded"] = "by 2026-11-23-new"
        json.dump(meta, open(path, "w"))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "note", when="2026-11-23T15:00:00Z")
        assert run(repo, "", "speed", "--show-metric", "tok_s") == (0, "200")
        index = os.path.join(ROOT, "scripts", "records_index.py")
        out = subprocess.run([sys.executable, index, "--root", repo], capture_output=True, text=True, check=True).stdout
        assert "`2026-11-23-new`" in out and "`2026-11-23-old`" not in out, out
        # an untracked record falls back to its mtime (now), so it sorts after the committed ones
        d = os.path.join(repo, "bench", "records", "2026-11-23-untracked")
        os.makedirs(d)
        json.dump({"gate": "speed", "date": "2026-11-23", "result": "pass", "metrics": {"tok_s": 300}}, open(os.path.join(d, "meta.json"), "w"))
        assert run(repo, "", "speed", "--show-metric", "tok_s") == (0, "300")


def audit(repo: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "records_index.py"), "--root", repo, "--cap-audit"],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def audit_record(repo: str, name: str, meta: dict, raw: str | None = None) -> None:
    d = os.path.join(repo, "bench", "records", name)
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    if raw is not None:
        open(os.path.join(d, "raw", "cap-proof.txt"), "w").write(raw)


def audit_repo(base: str) -> str:
    repo = os.path.join(base, "recipe")
    os.makedirs(os.path.join(repo, "config"))
    open(os.path.join(repo, "config", "pins.sh"), "w").write("GPU_CLOCK_CAP_MHZ=2200\n")
    return repo


def test_cap_audit_classifies_locked_sampled_and_none() -> None:
    with tempfile.TemporaryDirectory() as base:
        repo = audit_repo(base)
        proof = 'GPU clocks set to "(gpuClkMin 2200, gpuClkMax 2200)" for GPU 0000:01:00.0\n'
        audit_record(repo, "a-locked", {"node": "node-1", "gpu_clock_sm_max": "2900 MHz, 3003 MHz"}, raw=proof)
        audit_record(repo, "b-sampled", {"node": "node-2", "gpu_clock_sm_max": "2197 MHz, 3003 MHz"})
        audit_record(repo, "c-none", {"node": "node-3", "sm_clock_mhz_samples_under_load": "2418 2400", "note": "applications graphics 2418"})
        audit_record(repo, "e-ops", {"node": "node-4"}, raw="GPU SM clock capped at 2200 MHz on node-4\n")   # the gpu_cap.sh line alone
        audit_record(repo, "d-nonode", {"gpu_clock_sm_max": "9999 MHz, 3003 MHz"})   # not a GPU-node record: ignored
        rc, out = audit(repo)
        assert rc == 0, out
        assert out.splitlines()[0] == "cap-audit: 2 locked, 1 sampled (highest current clock 2197 MHz), 1 none; cap 2200 MHz", out
        assert "none: c-none" in out and "a-locked" not in out and "d-nonode" not in out, out


def test_cap_audit_fails_on_a_sample_above_the_cap_but_not_on_the_hardware_maximum() -> None:
    with tempfile.TemporaryDirectory() as base:
        repo = audit_repo(base)
        audit_record(repo, "ok", {"node": "node-1", "gpu_clock_sm_max": "2197 MHz, 3003 MHz"})   # 3003 is clocks.max.sm, not a sample
        assert audit(repo)[0] == 0
        audit_record(repo, "hot", {"node": "node-4", "gpu_clock_sm_max": "2201 MHz, 3003 MHz"})
        rc, out = audit(repo)
        assert rc == 1 and "OVER CAP: hot sampled 2201 MHz" in out and "ok" not in out.split("OVER CAP")[1], out


def test_cap_audit_over_the_real_tree_passes() -> None:
    # Holds today because every GPU-node record was written under the 2,200 MHz cap: the sampled current clocks are 2184-2197 MHz
    # and the 3003 beside them is the hardware maximum, which the audit does not read as a sample.
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "records_index.py"), "--cap-audit"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_newer_failing_record_outranks_an_older_passing_one() -> None:
    with tempfile.TemporaryDirectory() as base:
        repo = new_repo(base, "recipe")
        record(repo, "2026-11-23-good", "gate", "2026-11-23T10:00:00Z")
        assert run(repo, "", "gate")[0] == 0
        record(repo, "2026-11-23-bad", "gate", "2026-11-23T11:00:00Z", result="fail")
        rc, out = run(repo, "", "gate")
        assert rc == 1 and "result is 'fail'" in out, out


def test_the_engine_commit_must_be_clean_and_in_the_engine_head() -> None:
    with tempfile.TemporaryDirectory() as base:
        engine, c1, c3 = engine_repo(base)
        repo = new_repo(base, "recipe")
        record(repo, "2026-11-23-ok", "g", "2026-11-23T10:00:00Z", commit=c1)
        assert run(repo, engine, "g", "--need-engine")[0] == 0
        record(repo, "2026-11-23-side", "g", "2026-11-23T11:00:00Z", commit=c3)
        rc, out = run(repo, engine, "g", "--need-engine")
        assert rc == 1 and "not in the engine head" in out, out
        record(repo, "2026-11-23-dirty", "g", "2026-11-23T12:00:00Z", commit=c1, dirty=True)
        rc, out = run(repo, engine, "g", "--need-engine")
        assert rc == 1 and "dirty" in out, out
        record(repo, "2026-11-23-short", "g", "2026-11-23T13:00:00Z", commit=c1[:7])
        rc, out = run(repo, engine, "g", "--need-engine")
        assert rc == 1 and "40-hex" in out, out


def test_a_metric_floor_and_a_lab_address_anywhere_fail_the_record() -> None:
    with tempfile.TemporaryDirectory() as base:
        repo = new_repo(base, "recipe")
        record(repo, "2026-11-23-slow", "speed", "2026-11-23T10:00:00Z", metrics={"tok_s": 2999.9})
        rc, out = run(repo, "", "speed", "--metric", "tok_s", "--min-metric", "tok_s=3000")
        assert rc == 1 and "need at least 3000" in out, out
        record(repo, "2026-11-23-fast", "speed", "2026-11-23T11:00:00Z", metrics={"tok_s": 3000})
        assert run(repo, "", "speed", "--metric", "tok_s", "--min-metric", "tok_s=3000")[0] == 0
        record(repo, "2026-11-23-leak", "speed", "2026-11-23T12:00:00Z", metrics={"tok_s": 3100}, raw="curl http://" + "192.168." + "9.9:8000/v1")
        rc, out = run(repo, "", "speed", "--min-metric", "tok_s=3000")
        assert rc == 1 and "lab address" in out, out
        assert run(repo, "", "speed", "--show-metric", "missing") [0] == 1


def test_same_engine_pairs_a_speed_record_with_its_exactness_record() -> None:
    with tempfile.TemporaryDirectory() as base:
        engine, c1, _ = engine_repo(base)
        c2 = git(engine, "rev-parse", "HEAD")
        repo = new_repo(base, "recipe")
        args = ("--need-engine", "--same-engine", "exact")
        record(repo, "2026-11-23-speed", "speed", "2026-11-23T10:00:00Z", commit=c2, metrics={"tok_s": 1})
        rc, out = run(repo, engine, "speed", *args)
        assert rc == 1 and "no 'exact' record" in out, out
        record(repo, "2026-11-23-exact-old", "exact", "2026-11-23T10:30:00Z", commit=c1)
        rc, out = run(repo, engine, "speed", *args)
        assert rc == 1 and "differs" in out, out
        record(repo, "2026-11-23-exact-new", "exact", "2026-11-23T11:00:00Z", commit=c2)
        assert run(repo, engine, "speed", *args)[0] == 0
        record(repo, "2026-11-23-exact-failed", "exact", "2026-11-23T12:00:00Z", commit=c2, result="fail")
        rc, out = run(repo, engine, "speed", *args)
        assert rc == 1 and "not 'pass'" in out, out


def test_the_records_index_shows_what_the_checker_judges() -> None:
    index = os.path.join(ROOT, "scripts", "records_index.py")
    with tempfile.TemporaryDirectory() as base:
        repo = new_repo(base, "recipe")
        record(repo, "2026-11-23-zzz-old", "speed", "2026-11-23T10:00:00Z", metrics={"tok_s": 100, "digest": "f" * 64})
        record(repo, "2026-11-23-aaa-new", "speed", "2026-11-23T11:00:00Z", metrics={"tok_s": 200, "digest": "e" * 64}, tests=[])
        record(repo, "2026-11-23-other", "exact", "2026-11-23T12:00:00Z", result="fail")
        os.makedirs(os.path.join(repo, "bench", "records", "2026-11-22-probe"))
        with open(os.path.join(repo, "bench", "records", "2026-11-22-probe", "meta.json"), "w") as f:
            json.dump({"name": "probe"}, f)   # no gate: left out
        out = subprocess.run([sys.executable, index, "--root", repo], capture_output=True, text=True, check=True).stdout
        rows = [l for l in out.splitlines() if l.startswith("| `")]
        assert len(rows) == 2, out
        speed = next(l for l in rows if "`speed`" in l)
        assert "`2026-11-23-aaa-new`" in speed and "tok_s = 200" in speed and "tok_s = 100" not in speed, speed   # the record the checker judges
        assert "digest" not in speed, speed   # a hash-sized value is not shown
        assert "| fail |" in next(l for l in rows if "`exact`" in l)
        # --update replaces what sits between the markers and nothing else; a file without them is refused
        doc = os.path.join(base, "doc.md")
        with open(doc, "w") as f:
            f.write("before\n<!-- records-index:start -->\nstale\n<!-- records-index:end -->\nafter\n")
        subprocess.run([sys.executable, index, "--root", repo, "--update", doc], capture_output=True, text=True, check=True)
        text = open(doc).read()
        assert text.startswith("before\n<!-- records-index:start -->\n| gate |") and text.endswith("<!-- records-index:end -->\nafter\n") and "stale" not in text
        with open(doc, "w") as f:
            f.write("no markers\n")
        assert subprocess.run([sys.executable, index, "--root", repo, "--update", doc], capture_output=True, text=True).returncode == 1


def test_the_headline_table_quotes_only_passing_records_and_fails_closed() -> None:
    index = os.path.join(ROOT, "scripts", "records_index.py")
    with tempfile.TemporaryDirectory() as base:
        repo = new_repo(base, "recipe")
        record(repo, "2026-11-23-old", "speed", "2026-11-23T10:00:00Z", metrics={"tok_s": 1000.5, "n": 3})
        record(repo, "2026-11-23-new", "speed", "2026-11-23T11:00:00Z", metrics={"tok_s": 2357.4, "n": 3}, commit=None)
        spec = os.path.join(base, "headline.tsv")

        def run_headline(text: str, *extra: str) -> subprocess.CompletedProcess:
            with open(spec, "w") as f:
                f.write(text)
            return subprocess.run([sys.executable, index, "--root", repo, "--headline", spec, *extra], capture_output=True, text=True)

        r = run_headline("# a comment\nCold prefill\tspeed\ttok_s\ttok/s\n\nRuns\tspeed\tn\t\n")
        assert r.returncode == 0, r.stderr
        assert "| Cold prefill | 2,357.4 tok/s | `2026-11-23-new` |" in r.stdout and "| Runs | 3 |" in r.stdout and "1,000.5" not in r.stdout, r.stdout   # the newest record, formatted
        for bad, why in (("x\tnogate\ttok_s\t\n", "has no record"), ("x\tspeed\tmissing\t\n", "has no metric"), ("only two\tcolumns\n", "a row is")):
            r = run_headline(bad)
            assert r.returncode != 0 and why in r.stderr and not r.stdout, (bad, r.stdout, r.stderr)
        record(repo, "2026-11-23-failed", "speed", "2026-11-23T12:00:00Z", result="fail", metrics={"tok_s": 9999})
        r = run_headline("x\tspeed\ttok_s\t\n")
        assert r.returncode != 0 and "not 'pass'" in r.stderr and not r.stdout, (r.stdout, r.stderr)   # a failed newest record is never quoted
        # --headline --update fills the headline markers and leaves the other markers alone
        record(repo, "2026-11-23-fixed", "speed", "2026-11-23T13:00:00Z", metrics={"tok_s": 2400})
        doc = os.path.join(base, "readme.md")
        with open(doc, "w") as f:
            f.write("a\n<!-- headline:start -->\nstale\n<!-- headline:end -->\nb\n<!-- records-index:start -->\nkeep\n<!-- records-index:end -->\n")
        r = run_headline("Cold prefill\tspeed\ttok_s\ttok/s\n", "--update", doc)
        text = open(doc).read()
        assert r.returncode == 0 and "2,400 tok/s" in text and "stale" not in text and "\nkeep\n" in text, (r.stderr, text)
        # `@DIRECTORY` names one record: an older record of the gate is quoted although a newer one exists, a missing or failed one stops the run
        r = run_headline("Older figure\t@2026-11-23-old\ttok_s\ttok/s\n")
        assert r.returncode == 0 and "| Older figure | 1,000.5 tok/s | `2026-11-23-old` |" in r.stdout, (r.stdout, r.stderr)
        r = run_headline("x\t@2026-11-23-absent\ttok_s\t\n")
        assert r.returncode != 0 and "does not exist" in r.stderr and not r.stdout, (r.stdout, r.stderr)
        r = run_headline("x\t@2026-11-23-failed\ttok_s\t\n")
        assert r.returncode != 0 and "not 'pass'" in r.stderr and not r.stdout, (r.stdout, r.stderr)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}", file=sys.stderr)
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"test_records: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
