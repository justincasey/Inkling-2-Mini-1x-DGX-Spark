#!/usr/bin/env bash
# The engine-emulating and drafter fixtures of oracle/README.md "Fixtures on the real pack" on node-1: CPU only, nice 19.
# spec/ was gathered beforehand (four emulate-none fixtures, merged manifest.json). progress.txt: one line per stage
# `<stage> rc=<n> seconds=<s> <UTC>`, then `done`. Restart one stage by hand: run its command from this script alone.
BASE=/home/spark/inkling-2-mini-1x-dgx-spark
cd $BASE
PACK=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw
PY=$BASE/lab/rexl3-inkling2-loop-draft/venv/bin/python
date -u +%FT%TZ > fixtures/started.txt
stage() {  # stage NAME COMMAND...: run, then one progress line
  local name=$1; shift; local t0=$SECONDS
  "$@"; local rc=$?
  echo "$name rc=$rc seconds=$((SECONDS - t0)) $(date -u +%FT%TZ)" >> fixtures/progress.txt
}
engine_check() {  # every prompt in the engine manifest: numerics_version 2, emulate engine (with dense_fp16), 4 prompts
  $PY - <<'PYEOF'
import json
d = json.load(open("fixtures/engine/manifest.json"))
assert d["meta"]["numerics_version"] == "2" and d["meta"]["emulate"] == "engine", d["meta"]
assert len(d["fixtures"]) == 4, sorted(d["fixtures"])
PYEOF
}
engine() {
  nice -n 19 $PY oracle/gen_fixtures.py --pack "$PACK" --out fixtures/engine --emulate engine --workers 3 --threads 4 \
      --weight-cache-gib 12 > fixtures/engine.log 2>&1 && engine_check >> fixtures/engine.log 2>&1
}
dspark() {
  nice -n 19 $PY oracle/gen_dspark_fixtures.py --pack "$PACK" --fixtures fixtures/spec --out fixtures/dspark --threads 16 \
      --weight-cache-gib 10 > fixtures/dspark.log 2>&1
}
stage engine engine
stage dspark dspark
echo done >> fixtures/progress.txt
