#!/usr/bin/env python3
"""Turn the finished real-pack fixtures job of node-1 into the record `bench/records/<UTC date>-oracle-fixtures/` (gate
`oracle-fixtures`). Run on the Mac once `progress.txt` on the node ends with `done`:

    python3 oracle/finalize_fixtures_record.py --ssh spark@NODE [--base /home/spark/inkling-2-mini-1x-dgx-spark]
                                               [--out DIR] [--date YYYY-MM-DD] [--recipe-commit SHA]

The node address is an argument, never stored: nothing written into the record may contain one (addresses of the form
192.168.x.x / 10.100.x.x in fetched text are replaced by `<node>`). `--local-root DIR` reads a copy of the node's base
directory instead of ssh (the unit tests and dry runs use it). The job's layout under BASE/fixtures:

  spec/    four emulate-none fixtures + manifest.json (sha256, size, verified_recompute per fixture)
  engine/  the same four prompts with `--emulate engine` (numerics_version 2, dense_fp16 included) + manifest.json
  dspark/  drafter fixtures: per case C the files C_meta.json and C_{feats,tokens,anchors,drafts,hd,frnn,state,latent,logits,ctxkv}.npy
  progress.txt  one line per stage `<stage> rc=<n> seconds=<s> <UTC>`, then `done`;  run-fixtures.sh, *.log, started.txt

Only small files are copied into the record (manifests, progress, logs, the run script, the drafter `_meta.json`s); the
fixtures stay on the node, and their sha256 and size are checked there with one remote command. Verifications (each one
becomes a `tests` entry of meta.json; `result` is `pass` only if all pass):

  progress      the three stages spec, engine, dspark each ended `rc=0`, and the last line is `done`
  spec          4 fixtures in the manifest; every file exists with the manifest's sha256 and size; verified_recompute true
  engine        manifest meta numerics_version 2 and emulate engine; the emulation includes dense_fp16; the same prompts as
                spec; every file exists with the manifest's sha256 (the engine manifest holds no size)
  dspark        a `real_<prompt>` case with all 11 files for every spec fixture; every file is non-empty
  no-address    no lab address in any record file
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_BASE = "/home/spark/inkling-2-mini-1x-dgx-spark"
LAB_ADDRESS = re.compile(r"192\.168\.\d+\.\d+|10\.100\.\d+\.\d+")
STAGES = ("spec", "engine", "dspark")
DSPARK_PARTS = ("feats", "tokens", "anchors", "drafts", "hd", "frnn", "state", "latent", "logits", "ctxkv")
SMALL_FILES = ("progress.txt", "started.txt", "run-fixtures.sh", "run.out", "engine.log", "dspark.log",
               "spec/manifest.json", "spec/manifest-chat-capital-node2.json", "engine/manifest.json")
PROGRESS = re.compile(r"^(\w+) rc=(\d+) seconds=(\d+) (\S+)")

# the remote half: run with python3 on the node (or locally on a copy): sizes of every file under the three directories
# and the sha256 of the requested ones. Input on stdin: {"base": ..., "sha": [relative paths]}.
REMOTE = r"""
import hashlib, json, os, sys
req = json.load(sys.stdin); base = req["base"]
sizes = {}
for d in ("spec", "engine", "dspark"):
    for root, _, files in os.walk(os.path.join(base, d)):
        for f in files:
            p = os.path.join(root, f); sizes[os.path.relpath(p, base)] = os.path.getsize(p)
sha = {}
for rel in req["sha"]:
    p = os.path.join(base, rel)
    if not os.path.isfile(p):
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    sha[rel] = h.hexdigest()
print(json.dumps({"sizes": sizes, "sha": sha}))
"""


class Node:
    """Where the fixtures are: ssh target + base directory, or a local copy of that directory."""

    def __init__(self, base: str, ssh: str | None = None, local_root: str | None = None):
        if bool(ssh) == bool(local_root):
            raise ValueError("give exactly one of --ssh and --local-root")
        self.ssh, self.local = ssh, local_root
        self.base = local_root if local_root else base

    def inspect(self, sha_rel: list[str]) -> dict:
        req = json.dumps({"base": os.path.join(self.base, "fixtures"), "sha": sha_rel})
        cmd = [sys.executable, "-c", REMOTE] if self.local else ["ssh", "-o", "BatchMode=yes", self.ssh, "python3", "-c", "'" + REMOTE.replace("'", "'\"'\"'") + "'"]
        r = subprocess.run(cmd, input=req, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"remote inspection failed: {r.stderr.strip()[-400:]}")
        return json.loads(r.stdout)

    def fetch(self, rels: list[str], dest: Path) -> list[str]:
        """Copy the files that exist (relative to BASE/fixtures) into `dest`, same relative paths; returns the ones copied."""
        got = []
        for rel in rels:
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if self.local:
                src = Path(self.base) / "fixtures" / rel
                if src.is_file():
                    shutil.copy(src, out); got.append(rel)
            else:
                r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{self.ssh}:{self.base}/fixtures/{rel}", str(out)],
                                   capture_output=True)
                if r.returncode == 0:
                    got.append(rel)
        return got


def scrub(text: str) -> str:
    return LAB_ADDRESS.sub("<node>", text)


def parse_progress(text: str) -> tuple[dict[str, dict], bool]:
    stages, done = {}, False
    for line in text.splitlines():
        m = PROGRESS.match(line)
        if m:
            stages[m.group(1)] = {"rc": int(m.group(2)), "seconds": int(m.group(3)), "ended": m.group(4)}
        done = line.strip() == "done"
    return stages, done


def human(n: int) -> str:
    return f"{n / 2 ** 30:.2f} GiB" if n >= 2 ** 30 else f"{n / 2 ** 20:.1f} MiB"


def dspark_cases(sizes: dict[str, int]) -> dict[str, set[str]]:
    """case -> the set of its files found under dspark/ (`C_meta.json`, `C_<part>.npy`)."""
    cases: dict[str, set[str]] = {}
    for rel in sizes:
        m = re.match(r"dspark/(.+)_(meta\.json|(?:" + "|".join(DSPARK_PARTS) + r")\.npy)$", rel)
        if m:
            cases.setdefault(m.group(1), set()).add(m.group(2))
    return cases


def verify(local: Path, info: dict) -> tuple[list[dict], dict]:
    """The verifications over the fetched small files (`local`) and the remote inspection `info`; returns (tests, facts)."""
    tests: list[dict] = []
    facts: dict = {"messages": []}

    def check(name: str, ok: bool, why: str = "") -> bool:
        tests.append({"bin": "finalize_fixtures_record.py", "filter": name, "passed": int(ok), "failed": int(not ok)})
        if not ok:
            facts["messages"].append(f"{name}: {why}")
        return ok

    def load(rel: str):
        try:
            return json.loads((local / rel).read_text())
        except (OSError, ValueError):
            return None

    # progress
    try:
        stages, done = parse_progress((local / "progress.txt").read_text())
    except OSError:
        stages, done = {}, False
    facts["stages"] = stages
    bad = [s for s in STAGES if stages.get(s, {}).get("rc") != 0]
    check("progress", not bad and done, f"stages without rc=0: {bad or 'none'}; done line {'present' if done else 'missing'}")

    sizes, sha = info["sizes"], info["sha"]
    spec, eng = load("spec/manifest.json"), load("engine/manifest.json")

    def files_ok(man, name, need_size):
        probs = []
        for n, e in (man or {}).get("fixtures", {}).items():
            rel = f"{name}/{n}.safetensors"
            if sha.get(rel) != e.get("sha256"):
                probs.append(f"{rel}: sha256 {'missing on the node' if rel not in sha else 'differs from the manifest'}")
            if need_size and sizes.get(rel) != e.get("size"):
                probs.append(f"{rel}: size {sizes.get(rel)} != manifest {e.get('size')}")
        return probs

    sp = files_ok(spec, "spec", True)
    nspec = len((spec or {}).get("fixtures", {}))
    unver = [n for n, e in (spec or {}).get("fixtures", {}).items() if e.get("verified_recompute") is not True]
    check("spec", spec is not None and nspec == 4 and not sp and not unver,
          f"{nspec} fixtures (4 expected); {'; '.join(sp)}; not verified_recompute: {unver}")

    names = set((spec or {}).get("fixtures", {}))
    meta = (eng or {}).get("meta", {})
    sys.path.insert(0, str(HERE))
    from inkling2_ref import Emulation  # the oracle this record belongs to

    dense = Emulation.engine().dense_fp16
    ep = files_ok(eng, "engine", False)
    check("engine", eng is not None and str(meta.get("numerics_version")) == "2" and meta.get("emulate") == "engine" and dense
          and set(eng.get("fixtures", {})) == names and not ep,
          f"meta {meta}; dense_fp16 in Emulation.engine(): {dense}; prompts {sorted((eng or {}).get('fixtures', {}))} vs spec "
          f"{sorted(names)}; {'; '.join(ep)}")

    cases = dspark_cases(sizes)
    want = {f"real_{n}" for n in names}
    full = {c for c, s in cases.items() if s == {"meta.json"} | {f"{p}.npy" for p in DSPARK_PARTS}}
    empty = [r for r, s in sizes.items() if r.startswith("dspark/") and s == 0]
    check("dspark", bool(names) and want <= full and not empty,
          f"cases with all files: {sorted(full)}; missing for {sorted(want - full)}; empty files {empty}")
    facts.update(spec=spec, engine=eng, dspark_cases=sorted(full))
    return tests, facts


def render(node_label: str, facts: dict, info: dict, result: str, commit: str, date: str, local: Path) -> tuple[dict, str]:
    sizes = info["sizes"]
    by = lambda d: sum(s for r, s in sizes.items() if r.startswith(d + "/") and not r.endswith("manifest.json") and "manifest-" not in r)  # noqa: E731
    stages = facts["stages"]
    sec = lambda s: stages.get(s, {}).get("seconds", 0)  # noqa: E731
    spec, eng = facts["spec"] or {"fixtures": {}}, facts["engine"] or {"fixtures": {}}
    metrics = {
        "fixtures": len(spec["fixtures"]), "engine_fixtures": len(eng["fixtures"]), "dspark_fixtures": len(facts["dspark_cases"]),
        "spec_bytes": by("spec"), "engine_bytes": by("engine"), "dspark_bytes": by("dspark"),
        "spec_seconds": sec("spec"), "engine_seconds": sec("engine"), "dspark_seconds": sec("dspark"),
    }
    n_ok = sum(1 for _ in facts["tests"] if _["failed"] == 0)
    meta = {
        "name": "oracle-fixtures", "gate": "oracle-fixtures", "date": date, "result": result, "node": node_label,
        "recipe": {"commit": commit}, "pack": {"repo": "vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw",
                                               "revision": "44c78c9d229b43dd45e41506071b9f5f939bed44"},
        "tests": facts["tests"],
        "commands": [
            "oracle/gen_fixtures.py --pack PACK --out fixtures/spec --workers 3 --threads 4 --weight-cache-gib 12 --verify-recompute (per prompt, two runs)",
            "oracle/gen_fixtures.py --pack PACK --out fixtures/engine --emulate engine --workers 3 --threads 4 --weight-cache-gib 12",
            "oracle/gen_dspark_fixtures.py --pack PACK --fixtures fixtures/spec --out fixtures/dspark --threads 16 --weight-cache-gib 10",
            "fixtures/run-fixtures.sh (engine and dspark stages, nohup setsid, nice 19)",
            "oracle/finalize_fixtures_record.py --ssh <node> (this record)"],
        "metrics": metrics,
        "notes": "; ".join(facts["messages"]) or f"all {n_ok} verifications passed. The fixtures stay on the node (fixtures/spec, engine, dspark under the base directory); "
                                                  "the record holds the manifests, progress, logs and the drafter case metas.",
    }
    rows = "\n".join(
        f"| `{n}` | {e['prompt_len']} | {e['size']:,} B | {e['seconds']:.0f} | `{e['sha256'][:12]}` | {e.get('meta', {}).get('numerics_version', spec['meta'].get('numerics_version'))} |"
        for n, e in spec["fixtures"].items())
    erows = "\n".join(f"| `{n}` | {sizes.get(f'engine/{n}.safetensors', 0):,} B | {e['seconds']:.0f} | `{e['sha256'][:12]}` |"
                      for n, e in eng["fixtures"].items())
    readme = f"""# oracle-fixtures: the real-pack fixtures of the oracle (gate `oracle-fixtures`)

Result **{result}** (`meta.json`, date {date}, node {node_label}, recipe commit `{commit[:12]}`). Each verification is a `tests`
entry of `meta.json`; what failed, if anything, is in `notes`. Pack `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` at the pinned
revision. The fixtures are CPU float64 oracle runs (no GPU); they stay on the node; this record holds their manifests, the job's
progress file, logs and run script, and the drafter case metas.

## The three stages

| Stage | What it is | Count | Size | Seconds |
|---|---|---|---|---|
| `spec` | `gen_fixtures.py` with no emulation: the oracle's own numerics, each prompt prefilled and greedy-decoded from its cache, checked against a full recompute (`--verify-recompute`: bf16 planes and routes bit for bit, logits within 2 fp32 ulp) | {metrics['fixtures']} | {human(metrics['spec_bytes'])} | {metrics['spec_seconds']} (sum of the manifests' generation and verification seconds) |
| `engine` | the same prompts with `--emulate engine`: the lossy choices of the REXL3 engine (int8 KV on the global passes, bf16 P in attention, fp16 expert GEMM inputs, fp16 rotated inputs of the dense EXL3 linears); numerics_version {facts['engine']['meta'].get('numerics_version') if facts['engine'] else '?'} | {metrics['engine_fixtures']} | {human(metrics['engine_bytes'])} | {metrics['engine_seconds']} |
| `dspark` | `gen_dspark_fixtures.py`: the block drafter on the features of the spec fixtures (`stack(hidden[5], hidden[23], hidden[41], h_final)`), 8 anchors per case, plus one synthetic 1,100-row case (`long`): drafts, hd, f_rnn, Markov state/latent, logits, the context K/V | {metrics['dspark_fixtures']} cases | {human(metrics['dspark_bytes'])} | {metrics['dspark_seconds']} |

What a fixture holds (`oracle/README.md`, "Fixtures on the real pack"): `tokens`, `logits` [N-1, 200058], `hidden` [49, N-1, 2304]
(P entering each of the 48 layer passes, then after the last), `h_final`, `core_out` [2, N-1, 2304] (the message bank partials),
`router_idx` / `router_w` / `router_margin` for every pass; metadata `prompt_len`, `pack_config_sha256`, `oracle_source_sha256`,
`numerics_version`, `emulate`. The drafter files are named in `dspark/*_meta.json`.

### spec (`spec-manifest.json`)

| Prompt | Prompt tokens | Size | Seconds | sha256 | numerics_version |
|---|---|---|---|---|---|
{rows}

`chat-capital-effort0` was written by the earlier job on another node with numerics_version 1; for `emulate none` the oracle is
bit-identical to version 2 (`bench/records/2026-11-23-oracle-cacheddiff/logs/old-vs-new-oracle.txt`). The other three were written
on node-1 by the verified generator of that record. The spec manifest is the merge of those runs.

### engine (`engine-manifest.json`)

| Prompt | Size | Seconds | sha256 |
|---|---|---|---|
{erows}

## Where, how to regenerate

On {node_label}: `<base>/fixtures/{{spec,engine,dspark}}/`, base `/home/spark/inkling-2-mini-1x-dgx-spark`; the job is `fixtures/run-fixtures.sh`
(`run-fixtures.sh` here), progress in `progress.txt` (`{json.dumps(stages)}`). Regenerate a stage by running its command from
`run-fixtures.sh` with the pack staged and the oracle (`oracle/` of this repository) copied to `<base>/oracle`; every command is
niced and takes about the seconds above. `spec` needs `--verify-recompute` and one run per prompt group (see `oracle/README.md`).
`python3 oracle/finalize_fixtures_record.py --ssh <node>` rebuilds this directory.
"""
    return meta, readme


def finalize(node: Node, out: Path, date: str, commit: str) -> int:
    label = "node-1"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        node.fetch(list(SMALL_FILES), tmp)
        first = node.inspect([])  # sizes only, to know which drafter metas to copy
        metas = [r for r in first["sizes"] if r.startswith("dspark/") and r.endswith("_meta.json")]
        node.fetch(metas, tmp)
        spec = json.loads((tmp / "spec/manifest.json").read_text()) if (tmp / "spec/manifest.json").exists() else {"fixtures": {}}
        eng = json.loads((tmp / "engine/manifest.json").read_text()) if (tmp / "engine/manifest.json").exists() else {"fixtures": {}}
        want = [f"spec/{n}.safetensors" for n in spec["fixtures"]] + [f"engine/{n}.safetensors" for n in eng["fixtures"]]
        want += [r for r in first["sizes"] if r.startswith("dspark/")]
        info = node.inspect(want)
        tests, facts = verify(tmp, info)
        no_addr_ok = True
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                dst = out / p.relative_to(tmp)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(scrub(p.read_text(errors="replace")))
        facts["tests"] = tests
        meta, readme = render(label, facts, info, "pass", commit, date, out)
        (out / "README.md").write_text(scrub(readme))
        (out / "meta.json").write_text(scrub(json.dumps(meta, indent=1)))
        left = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and LAB_ADDRESS.search(p.read_text(errors="ignore"))]
        no_addr_ok = not left
        tests.append({"bin": "finalize_fixtures_record.py", "filter": "no-address", "passed": int(no_addr_ok), "failed": int(not no_addr_ok)})
        if not no_addr_ok:
            facts["messages"].append(f"no-address: lab address left in {left}")
        result = "pass" if all(t["failed"] == 0 for t in tests) else "fail"
        meta, readme = render(label, facts, info, result, commit, date, out)
        (out / "README.md").write_text(scrub(readme))
        (out / "meta.json").write_text(scrub(json.dumps(meta, indent=1)))
    print(f"record {out}: {result}; " + (" | ".join(facts["messages"]) if facts["messages"] else "all verifications passed"))
    return 0 if result == "pass" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ssh", help="user@host of the node (never stored)")
    ap.add_argument("--local-root", help="a local copy of the node's base directory instead of ssh")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", help="default bench/records/<UTC date>-oracle-fixtures under the repository")
    ap.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--recipe-commit", default=None)
    a = ap.parse_args(argv)
    commit = a.recipe_commit or subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path(a.out) if a.out else REPO / "bench" / "records" / f"{a.date}-oracle-fixtures"
    return finalize(Node(a.base, a.ssh, a.local_root), out, a.date, commit)


if __name__ == "__main__":
    sys.exit(main())
