"""finalize_fixtures_record.py on a synthetic copy of the node's fixtures tree (`--local-root`): every verification passes on a
good tree and each one fails, by name, when its condition is broken; addresses are scrubbed; the record is complete."""

import hashlib
import json
import re

import pytest

import finalize_fixtures_record as fin

NAMES = ("chat-capital-effort0", "chat-sky-effort0.9", "raw-counting", "raw-long-window")
ADDR = re.compile(r"192\.168\.\d+\.\d+|10\.100\.\d+\.\d+")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(root, *, rc=(0, 0, 0), done=True, bad_sha=None, unverified=None, engine_version="2", engine_names=NAMES,
          dspark_names=NAMES, empty_dspark=None, log_text="job started"):
    """A node base directory: fixtures/{spec,engine,dspark}, progress.txt, logs."""
    fx = root / "fixtures"
    for d in ("spec", "engine", "dspark"):
        (fx / d).mkdir(parents=True)
    spec, eng = {"meta": {"numerics_version": "2", "emulate": "none"}, "fixtures": {}}, \
        {"meta": {"numerics_version": engine_version, "emulate": "engine"}, "fixtures": {}}
    for i, n in enumerate(NAMES):
        body = f"spec {n}".encode() * (i + 3)
        (fx / "spec" / f"{n}.safetensors").write_bytes(body)
        spec["fixtures"][n] = {"sha256": "0" * 64 if n == bad_sha else sha(body), "size": len(body), "prompt_len": 10 + i,
                               "seconds": 100.0 + i, "verified_recompute": n != unverified,
                               "meta": {"numerics_version": "2"}}
    for n in engine_names:
        body = f"engine {n}".encode() * 4
        (fx / "engine" / f"{n}.safetensors").write_bytes(body)
        eng["fixtures"][n] = {"sha256": sha(body), "prompt_len": 10, "seconds": 120.0, "verified_recompute": False}
    (fx / "spec" / "manifest.json").write_text(json.dumps(spec))
    (fx / "engine" / "manifest.json").write_text(json.dumps(eng))
    for n in dspark_names:
        case = f"real_{n}"
        (fx / "dspark" / f"{case}_meta.json").write_text(json.dumps({"anchors": [1]}))
        for p in fin.DSPARK_PARTS:
            (fx / "dspark" / f"{case}_{p}.npy").write_bytes(b"" if case == empty_dspark and p == "hd" else b"x" * 64)
    lines = [f"{s} rc={r} seconds={100 * (i + 1)} 2026-11-23T10:0{i}:00Z" for i, (s, r) in enumerate(zip(fin.STAGES, rc))]
    (fx / "progress.txt").write_text("\n".join(lines + (["done"] if done else [])) + "\n")
    (fx / "started.txt").write_text("2026-11-23T08:00:00Z\n")
    (fx / "run-fixtures.sh").write_text("#!/usr/bin/env bash\necho job\n")
    (fx / "run.out").write_text("")
    (fx / "engine.log").write_text(log_text + "\n")
    (fx / "dspark.log").write_text("ok\n")


def run(tmp_path, **kw):
    root, out = tmp_path / "node", tmp_path / "rec"
    build(root, **kw)
    rc = fin.main(["--local-root", str(root), "--out", str(out), "--date", "2026-11-23", "--recipe-commit", "a" * 40])
    return rc, out, json.loads((out / "meta.json").read_text())


def failed(meta):
    return {t["filter"] for t in meta["tests"] if t["failed"]}


def test_good_tree_passes_and_the_record_is_complete(tmp_path):
    rc, out, meta = run(tmp_path)
    assert rc == 0 and meta["result"] == "pass" and failed(meta) == set()
    assert [t["filter"] for t in meta["tests"]] == ["progress", "spec", "engine", "dspark", "no-address"]
    m = meta["metrics"]
    assert (m["fixtures"], m["engine_fixtures"], m["dspark_fixtures"]) == (4, 4, 4)
    assert (m["spec_seconds"], m["engine_seconds"], m["dspark_seconds"]) == (100, 200, 300)
    assert m["spec_bytes"] == sum(len(f"spec {n}") * (i + 3) for i, n in enumerate(NAMES))
    assert m["engine_bytes"] == sum(len(f"engine {n}") * 4 for n in NAMES) and m["dspark_bytes"] > 0
    assert meta["gate"] == "oracle-fixtures" and meta["node"] == "node-1" and meta["recipe"]["commit"] == "a" * 40
    for f in ("README.md", "meta.json", "progress.txt", "run-fixtures.sh", "engine.log", "spec/manifest.json",
              "engine/manifest.json", "dspark/real_raw-counting_meta.json"):
        assert (out / f).is_file(), f
    readme = (out / "README.md").read_text()
    assert "oracle-fixtures" in readme and "raw-long-window" in readme and "## The three stages" in readme


def test_record_passes_the_ladders_record_checker(tmp_path):
    import subprocess
    import sys

    from pathlib import Path

    rc, out, _ = run(tmp_path)
    root = tmp_path / "repo"
    (root / "bench" / "records").mkdir(parents=True)
    out.rename(root / "bench" / "records" / "2026-11-23-oracle-fixtures")
    checker = Path(fin.REPO) / "scripts" / "check_record.py"
    r = subprocess.run([sys.executable, str(checker), "oracle-fixtures", "--root", str(root), "--metric", "fixtures",
                        "--min-tests", "4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("kw,name", [
    (dict(rc=(0, 1, 0)), "progress"),
    (dict(rc=(0, 0, 3)), "progress"),
    (dict(done=False), "progress"),
    (dict(bad_sha="raw-counting"), "spec"),
    (dict(unverified="raw-long-window"), "spec"),
    (dict(engine_version="1"), "engine"),
    (dict(engine_names=NAMES[:3]), "engine"),
    (dict(dspark_names=NAMES[:3]), "dspark"),
    (dict(empty_dspark="real_raw-counting"), "dspark"),
])
def test_each_verification_fails_by_name(tmp_path, kw, name):
    rc, _, meta = run(tmp_path, **kw)
    assert rc == 1 and meta["result"] == "fail" and failed(meta) == {name}, (kw, failed(meta))
    assert name in meta["notes"]


def test_a_missing_fixture_file_fails_spec_and_engine(tmp_path):
    root, out = tmp_path / "node", tmp_path / "rec"
    build(root)
    (root / "fixtures" / "spec" / "raw-counting.safetensors").unlink()
    (root / "fixtures" / "engine" / "raw-counting.safetensors").unlink()
    assert fin.main(["--local-root", str(root), "--out", str(out), "--recipe-commit", "b" * 40]) == 1
    assert failed(json.loads((out / "meta.json").read_text())) == {"spec", "engine"}


def test_a_size_that_differs_from_the_manifest_fails_spec(tmp_path):
    root, out = tmp_path / "node", tmp_path / "rec"
    build(root)
    p = root / "fixtures" / "spec" / "raw-counting.safetensors"
    man = json.loads((root / "fixtures" / "spec" / "manifest.json").read_text())
    man["fixtures"]["raw-counting"]["size"] += 1
    (root / "fixtures" / "spec" / "manifest.json").write_text(json.dumps(man))
    assert fin.main(["--local-root", str(root), "--out", str(out), "--recipe-commit", "b" * 40]) == 1
    assert failed(json.loads((out / "meta.json").read_text())) == {"spec"} and p.exists()


def test_node_addresses_are_scrubbed_everywhere(tmp_path):
    log = "ssh to " + "192.168." + "9.1 and 10.100." + "9.2 and 192.168." + "9.3 done"   # address-shaped samples, built so the source holds no literal address
    rc, out, meta = run(tmp_path, log_text=log)
    assert rc == 0 and meta["result"] == "pass"
    for f in out.rglob("*"):
        if f.is_file():
            assert not ADDR.search(f.read_text()), f
    assert "<node>" in (out / "engine.log").read_text()


def test_the_oracle_the_record_belongs_to_emulates_dense_fp16():
    from inkling2_ref import Emulation

    assert Emulation.engine().dense_fp16 and "dense_fp16" in Emulation.NAMES


def test_progress_parser_reads_the_nodeops_style_lines():
    stages, done = fin.parse_progress("spec rc=0 seconds=4886 2026-11-23T08:25:04Z (gathered)\nengine rc=1 seconds=5 2026-11-23T09:00:00Z\ndone\n")
    assert stages == {"spec": {"rc": 0, "seconds": 4886, "ended": "2026-11-23T08:25:04Z"},
                      "engine": {"rc": 1, "seconds": 5, "ended": "2026-11-23T09:00:00Z"}} and done
    assert fin.parse_progress("spec rc=0 seconds=1 t\n")[1] is False


def test_exactly_one_source_is_required():
    with pytest.raises(ValueError):
        fin.Node("/x")
    with pytest.raises(ValueError):
        fin.Node("/x", ssh="u@h", local_root="/y")
