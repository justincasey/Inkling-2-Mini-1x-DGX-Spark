#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ops_dev.sh: the engine's GPU tests on a node, from a lane's engine worktree. One lane per node, one lab tree per lane.
#
#   NODE=<address> LAB_SRC=<engine worktree> scripts/ops_dev.sh test  TESTBIN FILTER...   build, then run every FILTER
#   NODE=<address> LAB_SRC=<engine worktree> scripts/ops_dev.sh build TESTBIN             build only (cargo test --no-run)
#   NODE=<address> LAB_SRC=<engine worktree> scripts/ops_dev.sh run   TESTBIN FILTER...   run the last build
#   NODE=<address> LANE=<name> scripts/ops_dev.sh status                                  lab size, locks, MemAvailable, GPU
#   NODE=<address> LANE=<name> scripts/ops_dev.sh clean                                   remove the lane's lab tree
#
# TESTBIN is a `cargo test -p gb-cuda --test TESTBIN` target, or `--lib` (then PKG names the package: PKG=gb-model).
# FILTER is a test-name substring, run with --test-threads=1 --nocapture under the GPU lock /tmp/rexl3-gpu.lock (the lock
# the Inkling-2-Small ranks and the Mini server hold, so no two GPU jobs overlap). A filter that matches nothing, or any
# failing test, fails the script. Nothing here starts, stops or touches any server.
#
# Lab layout on the node: $LAB_BASE/$LANE/src (a copy of LAB_SRC without .git and target/) and $LAB_BASE/$LANE/target.
# Every build runs under the lane's build lock, with CARGO_BUILD_JOBS (default 8) and a timeout (BUILD_TIMEOUT_S, 3000):
# an unsupervised compile once took two lab nodes down (CYBERFROST outage, 2026-11-10). Nothing compiles on a node except
# through this script or `inkling2mini.sh build`. Both refuse to start under MIN_AVAIL_GIB of MemAvailable (default 30).
#
# Environment: NODE (required), SSH_USER (spark), LAB_BASE (/home/spark/inkling-2-mini-1x-dgx-spark/lab), LANE (basename of
# LAB_SRC), PKG (gb-cuda), RUST_TOOLCHAIN (config/pins.sh), TEST_ENV ("K=V K2=V2" for the test binary; REXL3_REQUIRE_GPU=1 is always set), OPS_GREP (extra regex
# of output lines to show), OPS_LINES (30), OPS_DRY=1 (print the remote scripts and stop).
#
# The GPU SM clock cap (GPU_CLOCK_CAP_MHZ in config/pins.sh: 2200 MHz) is hardware protection and non-negotiable (owner, 2026-11-23): `build`
# and `run` apply it on the node through scripts/gpu_cap.sh before any work and stop unless its success line came back. Nothing here
# releases, raises or changes it (OPS_DRY=1 prints the gpu_cap.sh call instead of making it).

set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/config/pins.sh"

die() { echo "ops_dev: $*" >&2; exit 1; }
NODE=${NODE:-}
[ -n "$NODE" ] || die "NODE is not set (the address of the node this lane owns)"
SSH_USER=${SSH_USER:-spark}
LAB_BASE=${LAB_BASE:-/home/spark/inkling-2-mini-1x-dgx-spark/lab}
LAB_SRC=${LAB_SRC:-$PWD}
LANE=${LANE:-$(basename "$LAB_SRC")}
PKG=${PKG:-gb-cuda}
TEST_ENV="REXL3_REQUIRE_GPU=1 ${TEST_ENV:-}"   # a GPU test that cannot find its GPU, pack or ids file fails instead of passing silently
OPS_GREP=${OPS_GREP:-}
OPS_LINES=${OPS_LINES:-30}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-30}
BUILD_TIMEOUT_S=${BUILD_TIMEOUT_S:-3000}
MODEL_DIR=${MODEL_DIR:-/home/spark/models/$MODEL_NAME}
case $LANE in *[!A-Za-z0-9._-]* | "") die "LANE '$LANE' must be letters, digits, . _ -" ;; esac
LAB=$LAB_BASE/$LANE

# remote SCRIPT [VAR=value...]: the script runs under bash -s on the node with the variables exported by name
remote() {
  local script=$1 prelude="set -euo pipefail"$'\n' kv; shift
  for kv in "$@"; do prelude+="export $(printf '%q' "${kv%%=*}")=$(printf '%q' "${kv#*=}")"$'\n'; done
  if [ "${OPS_DRY:-}" = 1 ]; then printf '%s%s\n' "$prelude" "$script"; return 0; fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_USER@$NODE" bash -s <<< "$prelude$script"
}

# cap: the GPU SM clock cap, applied on the node (scripts/gpu_cap.sh requires the success line) before any work runs there
cap() {
  if [ "${OPS_DRY:-}" = 1 ]; then echo "scripts/gpu_cap.sh $NODE"; return 0; fi
  "$ROOT/scripts/gpu_cap.sh" "$NODE" || die "no work on $NODE without the GPU cap"
}

guard() {
  remote '
    avail=$(awk "/^MemAvailable:/ {printf \"%d\", \$2 / 1048576}" /proc/meminfo)
    [ "$avail" -ge "$MIN_AVAIL" ] || { echo "ops_dev: MemAvailable ${avail} GiB < ${MIN_AVAIL} GiB on this node" >&2; exit 1; }' \
    MIN_AVAIL="$MIN_AVAIL_GIB"
}

sync_src() {
  [ -f "$LAB_SRC/Cargo.toml" ] && [ -d "$LAB_SRC/crates/cuda" ] || die "LAB_SRC=$LAB_SRC is not an engine worktree"
  remote 'mkdir -p "$LAB/src"' LAB="$LAB"
  [ "${OPS_DRY:-}" = 1 ] && { echo "rsync $LAB_SRC/ -> $NODE:$LAB/src/"; return 0; }
  # Content decides, the clock does not: `-c` sends only files whose bytes differ, `--no-times` stamps what is sent with the node's
  # clock, so cargo's mtime fingerprint always sees a changed file as newer than the artifact it built (an edit saved while a build
  # was running used to leave a stale test binary: the source mtime was older than the build's stamp).
  rsync -rlpgoD -c --no-times --delete --exclude '/target/' --exclude '/.git' --exclude '/.git/' "$LAB_SRC/" "$SSH_USER@$NODE:$LAB/src/" || die "rsync failed"
}

# build TESTBIN: leaves the test executable's path in $LAB/bin.path
do_build() {
  local bin=$1 targ
  if [ "$bin" = "--lib" ]; then targ="--lib"; else targ="--test $bin"; fi
  [ -f "$LAB_SRC/Cargo.toml" ] && [ -d "$LAB_SRC/crates/cuda" ] || die "LAB_SRC=$LAB_SRC is not an engine worktree"
  cap
  guard || exit 1
  sync_src
  remote '
    exec 8>"$LAB/build.lock"
    flock -x -n 8 || { echo "ops_dev: another build of lane $LANE holds $LAB/build.lock" >&2; exit 1; }
    cd "$LAB/src"
    export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda REXL3_CUDA_ARCH=sm_121 REXL3_COMMIT="lab-$LANE" \
      CARGO_TARGET_DIR="$LAB/target" CARGO_BUILD_JOBS="$JOBS"
    rm -f "$LAB/bin.path"
    set +e
    timeout "$TMO" cargo "+$TOOLCHAIN" test --release --locked -p "$PKG" $TARG --no-run --message-format=json-render-diagnostics \
      > "$LAB/build.json" 2> "$LAB/build.err"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "ops_dev: the build of $PKG $TARG failed (exit $rc; 124 = timeout)" >&2
      grep -E "^error|nvcc|error:" -A8 "$LAB/build.err" | head -60 >&2 || true
      exit 1
    fi
    # the test executable: a package with a bin target (gb-server) also builds that bin as a dependency of its integration tests, and it is
    # not the last artifact by chance; the test profile marks the artifact that holds the tests
    grep "\"profile\":{[^}]*\"test\":true" "$LAB/build.json" | grep -o "\"executable\":\"[^\"]*\"" | tail -n 1 | cut -d"\"" -f4 > "$LAB/bin.path"
    [ -s "$LAB/bin.path" ] || { echo "ops_dev: cargo built no test executable for $PKG $TARG" >&2; exit 1; }
    echo "built $(cat "$LAB/bin.path")"' \
    LAB="$LAB" LANE="$LANE" PKG="$PKG" TARG="$targ" TOOLCHAIN="$RUST_TOOLCHAIN" JOBS="${CARGO_BUILD_JOBS:-8}" TMO="$BUILD_TIMEOUT_S"
}

# run FILTER...: every filter must run at least one test, and none may fail
do_run() {
  local f out rc=0
  cap
  for f in "$@"; do
    echo "== $PKG $f"
    out=$(remote '
      exec 9>>/tmp/rexl3-gpu.lock
      flock -x -n 9 || { echo "ops_dev: /tmp/rexl3-gpu.lock is held by another GPU job" >&2; exit 1; }
      cd "$LAB/src"
      BIN=$(cat "$LAB/bin.path")
      set +e
      env $TENV REXL3_INKLING2_LOOP_DIR="$MODEL" "$BIN" "$FILTER" --test-threads=1 --nocapture 2>&1 \
        | grep -E "^test |test result|panicked|FAILED|failed${EXTRA:+|$EXTRA}" | head -"$LINES"
      exit "${PIPESTATUS[0]}"' \
      LAB="$LAB" FILTER="$f" MODEL="$MODEL_DIR" TENV="$TEST_ENV" EXTRA="$OPS_GREP" LINES="$OPS_LINES") || true
    echo "$out"
    { grep -q 'test result: ok\. [1-9]' <<< "$out" && ! grep -qE 'FAILED|panicked|test result: FAILED' <<< "$out"; } \
      || { echo "ops_dev: '$f' did not pass"; rc=1; }
  done
  return $rc
}

cmd=${1:-}; [ $# -gt 0 ] && shift
case $cmd in
  build) [ $# -eq 1 ] || die "usage: build TESTBIN"; do_build "$1" ;;
  run) [ $# -ge 2 ] || die "usage: run TESTBIN FILTER..."; shift; do_run "$@" ;;
  test) [ $# -ge 2 ] || die "usage: test TESTBIN FILTER..."; b=$1; shift; do_build "$b" && do_run "$@" ;;
  status)
    remote '
      echo "lab: $(du -sh "$LAB" 2>/dev/null | cut -f1) $LAB"
      echo "MemAvailable: $(awk "/^MemAvailable:/ {printf \"%.1f GiB\", \$2 / 1048576}" /proc/meminfo)"
      if [ -e /tmp/rexl3-gpu.lock ] && ! flock -n -x /tmp/rexl3-gpu.lock true 2>/dev/null; then echo "GPU lock: held"; else echo "GPU lock: free"; fi
      nvidia-smi --query-gpu=utilization.gpu,clocks.sm,memory.used --format=csv,noheader 2>/dev/null || true
      pgrep -fa "[c]argo|[r]ustc|[n]vcc" | cut -c1-120 | head -5 || true' LAB="$LAB"
    ;;
  clean)
    case $LAB in "$LAB_BASE"/?*) ;; *) die "refusing to clean $LAB" ;; esac
    remote 'rm -rf "$LAB"; echo "removed $LAB"' LAB="$LAB"
    ;;
  *) die "usage: ops_dev.sh test|build|run|status|clean (see the header)" ;;
esac
