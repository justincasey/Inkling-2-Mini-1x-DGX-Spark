#!/bin/bash
# loop_probe.sh 'label|KEY=VAL ...' ...   boots the HEAD of CANDIDATE_SRC (default the engine worktree) once per variant on the node, runs scripts/loop_probe.py
# against it (text mode: 32 prompts, 512 greedy tokens each; MODES="text reasoning agentic" picks the modes, default text) and compares every variant
# with the first one: first differing token, repetition loops, speed; reasoning and agentic runs are reported (loops, unfinished reasoning, destructive calls). Results in
# .work/loop/STAMP/. It builds, boots and stops the engine: run it under `scripts/safeloop.sh cmd TIMEOUT`.
set -u
cd "$(dirname "$0")/.." || exit 2
export CANDIDATE_SRC=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-loop}
export REXL3_SRC=$CANDIDATE_SRC
OUT=${OUT:-.work/loop/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
SHA=$(git -C "$CANDIDATE_SRC" rev-parse --short=12 HEAD)
./inkling2mini.sh stop >/dev/null 2>&1
./inkling2mini.sh build "$SHA" 2>&1 | tail -1
names=()
for v in "$@"; do
  label=${v%%|*}; envs=${v#*|}; [[ $v == *'|'* ]] || envs=""
  names+=("$label")
  echo "== $label  env: ${envs:-(none)}"
  ./inkling2mini.sh stop >/dev/null 2>&1
  EXTRA_ENV="$envs" ./inkling2mini.sh start 2>&1 | grep -E 'boot_s=|FAIL|fatal'
  for mode in ${MODES:-text}; do
    case $mode in
      text) python3 scripts/loop_probe.py run "$OUT/$label.json" | tail -3 ;;
      reasoning|agentic) python3 scripts/loop_probe.py "$mode" "$OUT/$label-$mode.json" | tail -3
        python3 scripts/loop_probe.py report "$OUT/$label-$mode.json" ;;
      *) echo "loop_probe.sh: MODES holds text, reasoning or agentic (got $mode)" >&2; exit 2 ;;
    esac
  done
  if [[ -n ${NEEDLE:-} ]]; then   # 200K needle at three depths with a 512-token answer budget (the gate's is 128): is the code found, and in how many reasoning tokens
    for d in 0.1 0.5 0.9; do
      python3 bench/tools/needle.py --tokens 200000 --max-model-len "${MAX_MODEL_LEN:-262144}" --depth "$d" --max-tokens 512 --out "$OUT/needle-$label-$d.json" >/dev/null 2>&1
      python3 - "$OUT/needle-$label-$d.json" "$label" "$d" <<'PY'
import json, sys
l = json.load(open(sys.argv[1]))["lanes"][0]
u = l["usage"]
print(f"needle {sys.argv[2]:6} depth {sys.argv[3]}: found={l['found']} answer={l['answer']!r} completion={u['completion_tokens']} reasoning={u['completion_tokens_details']['reasoning_tokens']} finish={l['finish_reason']}")
PY
    done
  fi
done
./inkling2mini.sh stop 2>&1 | tail -1
for l in "${names[@]:1}"; do
  [[ -f $OUT/${names[0]}.json && -f $OUT/$l.json ]] || continue
  echo "== ${names[0]} -> $l"
  python3 scripts/loop_probe.py compare "$OUT/${names[0]}.json" "$OUT/$l.json" | tee "$OUT/compare-${names[0]}-$l.txt"
done
echo "results: $OUT"
