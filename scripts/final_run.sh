#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# final_run.sh: the final procedure, one command, from this machine: pin check, build, engine tests, three boots, every node gate,
# one record per gate. Re-run it at any pin on any node.
#
#   scripts/final_run.sh [--continue] [--resume RUN_DIR]
#
# What it needs (refuses with a reason otherwise, before anything touches the node):
#   - NODES (one address) and NODE_NAME (node-1..node-4, the name the records carry) in .env or the environment; the address
#     never enters a tracked file, the records say NODE_NAME;
#   - REXL3_COMMIT pinned in config/pins.sh; the engine installed on the node after the build must be exactly that commit
#     (`doctor` is read before every boot), a lab build is refused;
#   - a clean recipe tree (tracked files): a record names the recipe commit it ran at.
# What it does, in order (each phase's wall seconds are printed and kept in RUN_DIR/stages.tsv):
#   0 doctor            read-only: node idle, MemAvailable, pack at full size
#   1 build             ./inkling2mini.sh build (the pin), engine-test host, engine-test goldens          -> gate engine-build
#   2 boot p (pin)      shipping defaults (.env.example: SPEC=dspark PREFIX_CACHE=on VISION=on, bootstrap, 4 x 262,144):
#                       start, exactness record, smoke, bench pin (writes bench/reference*.json when absent, else reproduces them)
#                                                                                                          -> boot, smoke, pin (with g)
#   3 boot g (gates)    fresh boot, same configuration: exactness record, bench all (check smoke decode rolling prefill vision spec
#                       cache needle memory), cache-images, prefill sweep (8K 32K 128K 224K), decode at 4K, FINAL_PASSES-1 more passes
#                       of decode and rolling, probe, api, quality, tools, loop, profile (the memory bracket), memwatch
#                                                                                                          -> spec cache cache-images
#                       vision-smoke needle api-test speed-baseline speed-decode speed-prefill speed-c4 capacity memory quality-tools
#                       loop-probe pin
#   4 boot c (chunk)    PREFILL_CHUNK=1000: start, exactness record; the chunk-width diff against boot p                   -> exact-basic
# Every record is written under its gate's existing name: bench/records/<UTC date>-final-<gate>/ (meta.json, README.md, raw/).
# It stops at the first gate that fails unless --continue is given (the remaining phases then still run). It ends by stopping the
# server (always, also after an error or ^C) and printing one line per gate: `PASS gate`, `FAIL gate` or `NOT RUN gate`, and a total.
# Exit 0 only when every gate is PASS. --resume RUN_DIR continues a run that stopped (bench/runs/final-<stamp>): finished phases are
# kept, an unfinished boot is restarted; a fixed tool needs nothing more than this. A phase that failed is run again.
#
# The GPU SM clock cap (2,200 MHz, config/pins.sh) is hardware protection: every node action goes through ./inkling2mini.sh, which
# applies it and requires the nvidia-smi success line before anything runs (build, engine-test, start). The lines are in the raw output
# of every record. This script never changes, releases or skips the cap, and has no flag that does.
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 )); then echo "final_run.sh needs bash >= 4 (macOS: brew install bash)" >&2; exit 1; fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1
CTL="$ROOT/inkling2mini.sh"
FINAL_PASSES=${FINAL_PASSES:-3}
ENGINE_REPO=${REXL3_SRC:-$HOME/projects/rexl3-inkling2-loop}

CONTINUE=0 RESUME=
while (( $# )); do
  case $1 in
    --continue) CONTINUE=1 ;;
    --resume) shift; RESUME=${1:-}; [[ -n $RESUME ]] || { echo "final_run: --resume needs a run directory" >&2; exit 2; } ;;
    -h | --help) sed -n '3,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "usage: scripts/final_run.sh [--continue] [--resume RUN_DIR]" >&2; exit 2 ;;
  esac
  shift
done

die() { echo "final_run: $*" >&2; exit 1; }
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

# ── the settings: .env (NODES, NODE_NAME) under the environment; the pins from config/pins.sh ──
env_value() {   # env_value KEY: the environment, else the last KEY=value line of .env (comments and quotes stripped)
  local k=$1 v
  if [[ -n ${!k:-} ]]; then printf '%s' "${!k}"; return; fi
  [[ -f $ROOT/.env ]] || return 0
  v=$(sed -n "s/^[[:space:]]*$k=//p" "$ROOT/.env" | tail -n 1)
  v=${v%%[[:space:]]#*}; v=${v#\"}; v=${v%\"}; v=${v#\'}; v=${v%\'}
  v=${v#"${v%%[![:space:]]*}"}; v=${v%"${v##*[![:space:]]}"}
  printf '%s' "$v"
}
NODES=$(env_value NODES)
NODE_NAME=$(env_value NODE_NAME)
[[ -n $NODES ]] || die "NODES is empty (set it in .env: the one address of the node)"
read -r -a _n <<<"$NODES"; (( ${#_n[@]} == 1 )) || die "NODES takes one address (got ${#_n[@]})"
[[ $NODE_NAME =~ ^node-[1-4]$ ]] || die "NODE_NAME must be node-1..node-4 (set it in .env; the records carry this name, never the address; got '$NODE_NAME')"
REXL3_COMMIT=$(bash -c 'source "$1/config/pins.sh" && printf "%s" "$REXL3_COMMIT"' _ "$ROOT")
[[ $REXL3_COMMIT =~ ^[0-9a-f]{40}$ ]] || die "REXL3_COMMIT in config/pins.sh is not pinned (a full commit sha): the final run is made on the pinned engine"
GPU_CAP=$(bash -c 'source "$1/config/pins.sh" && printf "%s" "$GPU_CLOCK_CAP_MHZ"' _ "$ROOT")
[[ $GPU_CAP =~ ^[0-9]+$ ]] || die "GPU_CLOCK_CAP_MHZ is not set in config/pins.sh"
if ! git -C "$ENGINE_REPO" cat-file -e "$REXL3_COMMIT^{commit}" 2>/dev/null; then die "the pinned engine commit ${REXL3_COMMIT:0:12} is not in $ENGINE_REPO (REXL3_SRC)"; fi
if [[ -n $(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null) ]]; then
  die "the recipe tree has uncommitted changes: commit them first (every record names the recipe commit it ran at)"
fi
RECIPE_HEAD=$(git -C "$ROOT" rev-parse HEAD)

# the shipping configuration, explicitly: .env.example's serving defaults whatever .env says, so every final run is comparable
export SPEC=dspark PREFIX_CACHE=on VISION=on PROFILE=bootstrap PREFILL_CHUNK=4096
export MAX_MODEL_LEN= SLOTS= KV_TOKENS= EXTRA_FLAGS= EXTRA_ENV= MEMORY_FLOOR_GIB=

if [[ -n $RESUME ]]; then
  RUN=$RESUME; [[ $RUN == /* ]] || RUN=$ROOT/$RUN
  [[ -d $RUN/stages ]] || die "$RUN is not a run directory of this script"
  [[ $(cat "$RUN/recipe_head" 2>/dev/null) == "$RECIPE_HEAD" ]] || die "$RUN was started at recipe commit $(cut -c1-12 "$RUN/recipe_head" 2>/dev/null): resume at that commit"
  [[ $(cat "$RUN/pin" 2>/dev/null) == "$REXL3_COMMIT" ]] || die "$RUN was started at another engine pin"
else
  RUN=$ROOT/bench/runs/final-$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$RUN/stages" || die "cannot create $RUN"
  printf '%s\n' "$RECIPE_HEAD" > "$RUN/recipe_head"; printf '%s\n' "$REXL3_COMMIT" > "$RUN/pin"
  printf '%s\n' "$NODE_NAME" > "$RUN/node_name"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/started"
fi
[[ $(cat "$RUN/node_name") == "$NODE_NAME" ]] || die "$RUN was run on $(cat "$RUN/node_name"), not $NODE_NAME"
say "final run at recipe ${RECIPE_HEAD:0:12}, engine pin ${REXL3_COMMIT:0:12}, $NODE_NAME, GPU cap $GPU_CAP MHz (the controller applies it); results in ${RUN#"$ROOT"/}"

# ── the gates, their order, and what has been decided ──
GATES=(engine-build boot smoke exact-basic spec cache cache-images vision-smoke needle api-test speed-baseline speed-decode speed-prefill speed-c4
  capacity memory quality-tools loop-probe pin)
declare -A VERDICT=()
STOPPING=0 FAILED=0

records() {   # records GATE...: write the records of the gates whose stages have finished; PASS/FAIL line each
  local g out rc
  for g in "$@"; do
    [[ -z ${VERDICT[$g]:-} ]] || continue
    if grep -qx "$g" "$RUN/records.done" 2>/dev/null; then VERDICT[$g]=$(sed -n "s/^$g //p" "$RUN/verdicts" | tail -n 1); continue; fi
    out=$(python3 "$ROOT/bench/tools/final_records.py" --run "$RUN" --node-name "$NODE_NAME" --root "$ROOT" --engine "$ENGINE_REPO" "$g" 2>&1); rc=$?
    printf '%s\n' "$out" | tee -a "$RUN/records.log"
    # PASS needs the builder's exit 0 and its own `GATE: pass` line: a builder that printed nothing proves nothing
    if (( rc == 0 )) && grep -q "^$g: pass " <<<"$out"; then VERDICT[$g]=PASS; else VERDICT[$g]=FAIL; fi
    echo "$g" >> "$RUN/records.done"; echo "$g ${VERDICT[$g]}" >> "$RUN/verdicts"
    if [[ ${VERDICT[$g]} == FAIL ]]; then
      FAILED=1
      if (( ! CONTINUE )); then say "gate $g FAILED: stopping (--continue runs the rest)"; STOPPING=1; fi
    fi
  done
  return 0
}

summary() {
  local g pass=0 fail=0 nr=0
  echo
  for g in "${GATES[@]}"; do
    case ${VERDICT[$g]:-} in
      PASS) echo "PASS $g"; pass=$((pass + 1)) ;;
      FAIL) echo "FAIL $g"; fail=$((fail + 1)) ;;
      *) echo "NOT RUN $g"; nr=$((nr + 1)) ;;
    esac
  done
  echo "final_run: $pass PASS, $fail FAIL, $nr NOT RUN of ${#GATES[@]} gates (engine ${REXL3_COMMIT:0:12}, recipe ${RECIPE_HEAD:0:12}, $NODE_NAME; ${RUN#"$ROOT"/})"
  echo "phases (seconds):"; awk -F'\t' '{printf "  %-22s rc=%s %6ss\n", $1, $2, $3}' "$RUN/stages.tsv" 2>/dev/null
  (( pass == ${#GATES[@]} ))
}

finish() {   # always: the server is stopped, the clock is read; then the summary and the exit code
  local rc=$?
  trap - EXIT INT TERM
  say "stopping the server"
  "$CTL" stop > "$RUN/final-stop.out" 2>&1 || echo "WARNING: stop failed, see $RUN/final-stop.out"
  "$CTL" gpu-clocks status > "$RUN/final-clock.out" 2>&1 || true
  echo "gpu clock at the end: $(tr '\n' ' ' < "$RUN/final-clock.out" | cut -c1-120)"
  if summary; then exit 0; fi
  exit 1
}
trap finish EXIT
trap 'echo "interrupted"; exit 130' INT TERM

# ── stages ──
done_ok() { [[ $(cat "$RUN/stages/$1.rc" 2>/dev/null) == 0 ]]; }
stage() {   # stage NAME CMD...: run CMD (output to RUN/NAME.out and the screen), keep rc and seconds, copy the bench run directory it names
  local name=$1 t0 rc=0 d; shift
  say "── $name: $*"
  t0=$(date +%s)
  "$@" 2>&1 | tee "$RUN/$name.out" | sed 's/^/    /'
  rc=${PIPESTATUS[0]}
  printf '%s\n' "$rc" > "$RUN/stages/$name.rc"
  printf '%s\n' "$(( $(date +%s) - t0 ))" > "$RUN/stages/$name.sec"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/stages/$name.end"
  d=$(sed -n 's/.*run=\(bench\/runs\/[^ ]*\)$/\1/p' "$RUN/$name.out" | tail -n 1)
  if [[ -n $d && -d $ROOT/$d ]]; then rm -rf "$RUN/$name.d"; cp -R "$ROOT/$d" "$RUN/$name.d"; fi
  { grep -v "^$name	" "$RUN/stages.tsv" 2>/dev/null || true; printf '%s\t%s\t%s\n' "$name" "$rc" "$(cat "$RUN/stages/$name.sec")"; } > "$RUN/stages.tmp"; mv "$RUN/stages.tmp" "$RUN/stages.tsv"
  say "── $name: rc=$rc, $(cat "$RUN/stages/$name.sec") s"
  return "$rc"
}
pending() {   # pending NAME...: true when any of the stages has not finished with rc 0
  local s; for s in "$@"; do done_ok "$s" || return 0; done; return 1
}
check_pin() {   # the installed engine is the pin: read from the node's doctor, before every boot and after the build
  local out got
  out=$("$CTL" doctor 2>&1) || { printf '%s\n' "$out" >&2; return 1; }
  got=$(awk '$1 == "engine" {print $2}' <<<"$out")
  if [[ $got != "$REXL3_COMMIT" ]]; then echo "the installed engine is '${got:-none}', not the pin $REXL3_COMMIT: refusing to run" >&2; return 1; fi
  echo "installed engine is the pin ${got:0:12}"
}
clock() { "$CTL" gpu-clocks status; }

# a boot group: GROUP (p|g|c), its stage names; the server is started once per invocation that has work to do
boot() {   # boot GROUP: (re)start the server for the group, always through the controller (cap, guard, doctor inside)
  local g=$1 name=$1-start n=0
  check_pin || { FATAL="$name: the engine is not the pin"; return 1; }
  if done_ok "$name"; then n=1; name=$g-restart-$(date +%H%M%S); fi   # a resumed run: the first start of this group is kept as it was
  stage "$name" "$CTL" start || { FATAL="$name: the server did not start"; return 1; }
  if (( n == 0 )); then
    "$CTL" logs 500 > "$RUN/$g-serve-boot.out" 2>&1 || true
    "$CTL" status > "$RUN/$g-status-ready.out" 2>&1 || true
    "$CTL" memwatch report > "$RUN/$g-memwatch-ready.out" 2>&1 || true
    clock > "$RUN/$g-clock-start.out" 2>&1 || true
  fi
}
unboot() { stage "$1-stop" "$CTL" stop || true; }
FATAL=

bench_stage() { stage "$1" "$CTL" bench "${@:2}"; }

phase_build() {
  stage doctor "$CTL" doctor || { FATAL="the node is not ready (doctor)"; return 1; }
  if pending build; then stage build "$CTL" build || { FATAL="build failed"; return 1; }; fi
  check_pin || { FATAL="the build did not leave the pin installed"; return 1; }
  pending engine-host && stage engine-host "$CTL" engine-test host
  pending engine-goldens && stage engine-goldens "$CTL" engine-test goldens
  records engine-build
}

phase_pin_boot() {
  pending p-start p-exact p-smoke p-pin || { records boot smoke; return 0; }
  boot p || return 1
  records boot
  (( STOPPING )) && return 0
  pending p-exact && bench_stage p-exact exactness
  pending p-smoke && stage p-smoke "$CTL" smoke
  records smoke
  (( STOPPING )) && return 0
  pending p-pin && bench_stage p-pin pin
  clock > "$RUN/p-clock-end.out" 2>&1 || true
  unboot p
}

phase_gates_boot() {
  local k extra=()
  for (( k = 2; k <= FINAL_PASSES; k++ )); do extra+=("g-pass$k-decode" "g-pass$k-rolling"); done
  if ! pending g-exact g-all g-cache-images g-prefill-sweep g-decode-4k ${extra[@]+"${extra[@]}"} g-probe g-api g-quality g-tools g-loop g-profile g-memwatch; then
    records spec cache vision-smoke needle capacity pin cache-images speed-prefill speed-baseline speed-decode speed-c4 api-test quality-tools loop-probe memory
    return 0
  fi
  boot g || return 1
  pending g-exact && bench_stage g-exact exactness
  pending g-all && bench_stage g-all all
  records spec cache vision-smoke needle capacity pin
  (( STOPPING )) && return 0
  pending g-cache-images && bench_stage g-cache-images cache-images
  records cache-images
  (( STOPPING )) && return 0
  pending g-prefill-sweep && bench_stage g-prefill-sweep prefill-sweep
  records speed-prefill
  (( STOPPING )) && return 0
  pending g-decode-4k && bench_stage g-decode-4k decode-4k
  for (( k = 2; k <= FINAL_PASSES; k++ )); do
    pending "g-pass$k-decode" && bench_stage "g-pass$k-decode" decode
    pending "g-pass$k-rolling" && bench_stage "g-pass$k-rolling" rolling
  done
  records speed-baseline speed-decode speed-c4
  (( STOPPING )) && return 0
  pending g-probe && bench_stage g-probe probe
  pending g-api && bench_stage g-api api
  records api-test
  (( STOPPING )) && return 0
  pending g-quality && bench_stage g-quality quality
  pending g-tools && bench_stage g-tools tools
  pending g-loop && bench_stage g-loop loop
  records quality-tools loop-probe
  (( STOPPING )) && return 0
  pending g-profile && bench_stage g-profile profile
  pending g-memwatch && stage g-memwatch "$CTL" memwatch report
  clock > "$RUN/g-clock-end.out" 2>&1 || true
  records memory
  unboot g
}

phase_chunk_boot() {
  pending c-start c-exact || { records exact-basic; return 0; }
  PREFILL_CHUNK=1000 boot c || return 1
  PREFILL_CHUNK=1000 bench_stage c-exact exactness
  clock > "$RUN/c-clock-end.out" 2>&1 || true
  unboot c
  records exact-basic
}

main() {
  phase_build || { say "FATAL: $FATAL"; return 1; }
  (( STOPPING )) && return 1
  phase_pin_boot || { say "FATAL: $FATAL"; return 1; }
  (( STOPPING )) && return 1
  phase_gates_boot || { say "FATAL: $FATAL"; return 1; }
  (( STOPPING )) && return 1
  phase_chunk_boot || { say "FATAL: $FATAL"; return 1; }
  return 0
}
main
exit $?
