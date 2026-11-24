#!/usr/bin/env python3
"""Complete a record directory that `./inkling2mini.sh record NAME` started.

  finalize_record.py DIR SPEC.json [--engine-worktree PATH] [--node node-N]

SPEC.json carries the parts the controller cannot know: gate, result (pass|fail), tests [{bin, filter, passed, failed}],
metrics {name: number}, commands [str], notes, and optionally extra top-level keys. They are merged into DIR/meta.json
together with node (--node, default node-4) and engine.dirty (false when the recorded engine commit is a commit of the engine repository:
the build stages it with `git archive`). Every text file of DIR then has its lab addresses (192.168.x.x, 10.100.x.x) replaced by the node name, because
scripts/check_record.py refuses a record that carries one. README.md must exist in DIR.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

LAB = re.compile(r"192\.168\.\d+\.\d+|10\.100\.\d+\.\d+")


def finalize(d, spec, eng, node="node-4"):
    """Merge SPEC into DIR/meta.json (node, engine.dirty), scrub the lab addresses of every text file of DIR; returns the meta."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    for k in ("gate", "result", "tests", "metrics", "commands", "notes"):
        if k not in spec:
            raise SystemExit(f"finalize_record: SPEC lacks '{k}'")
    if spec["result"] not in ("pass", "fail"):
        raise SystemExit("finalize_record: result is pass or fail")
    if not (d / "README.md").is_file():
        raise SystemExit(f"finalize_record: {d}/README.md is missing")
    # `build REV` stages the tree with `git archive REV`, so the binary is a committed tree whenever REV is a commit of the
    # engine repository; the worktree's own HEAD and status move on with other lanes and say nothing about the binary
    known = subprocess.run(["git", "-C", eng, "cat-file", "-e", meta["engine"]["commit"] + "^{commit}"], capture_output=True)
    meta["engine"]["dirty"] = known.returncode != 0
    meta["node"] = node
    meta.update(spec)
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    for f in d.rglob("*"):
        if f.is_file():
            try:
                t = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if LAB.search(t):
                f.write_text(LAB.sub(node, t))
    return meta


def main(argv):
    eng = str(Path.home() / "projects/rexl3-inkling2-loop")
    node = "node-4"
    args = list(argv)
    for opt in ("--engine-worktree", "--node"):
        if opt in args:
            i = args.index(opt)
            if opt == "--node":
                node = args[i + 1]
            else:
                eng = args[i + 1]
            del args[i:i + 2]
    if len(args) != 2:
        print(__doc__)
        return 2
    d, spec = Path(args[0]), json.loads(Path(args[1]).read_text())
    meta = finalize(d, spec, eng, node)
    print(f"finalized {d}: gate {meta['gate']}, result {meta['result']}, engine dirty={meta['engine']['dirty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
