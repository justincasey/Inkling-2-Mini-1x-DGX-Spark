#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# inkling2mini.sh: Inkling-2-Mini (vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw, EXL3 mixed K) on ONE DGX Spark, TP1 on the
# REXL3 engine family inkling2_loop. Run from any control machine with key-based ssh to the node.
# Adapted from the Inkling-2-Small recipe's controller (four ranks) and CYBERFROST's one-node setup. TP1 is one process:
# no ranks, no fabric, no collectives.
#
# Usage: ./inkling2mini.sh <command> [args]
#   doctor         read-only: the pins (pack manifest), then on the node: GB10 GPU idle, /tmp/rexl3-gpu.lock free, no
#                  rexl3-serve, MemAvailable >= MIN_AVAIL_GIB, no foreign container (ALLOWED_CONTAINERS), every pack file at
#                  its manifest size, installed engine --version equal to bin/rexl3-serve.commit. Says so and stops when
#                  NODES is empty or the node cannot be reached
#   setup          first time, idempotent: preflight (architecture, GB10, memory, tools, disk), install (Rust toolchain,
#                  hf CLI, directories) and fetch. It does not build the engine: `build` does
#   fetch          download the 13 pack files at the pinned revision (hf, resumable) into MODELS_DIR, then verify
#   verify         sha256 of every pack file on the node against intake/manifest.tsv (no download)
#   build [REV]    stage the engine source with `git archive` from the local repository (REXL3_SRC) and build rexl3-serve +
#                  rexl3-cluster on the node (cargo +RUST_TOOLCHAIN --release --locked, sm_121) under the node build lock.
#                  Without REV it builds REXL3_COMMIT (config/pins.sh) and refuses while that pin is empty (plan step 4.8
#                  sets it). With REV it is a lab build (recorded as engine.lab_build=true). The commit must descend from
#                  REXL3_UPSTREAM_BASE; the binary must report the commit and sm_121; bin/rexl3-serve.commit is written
#   print-serve    print the exact command line (environment included) the server runs; `start` runs exactly this. An
#                  invalid configuration prints nothing and exits non-zero
#   start          doctor (plus an installed engine), then the memory guard (scripts/memguard.sh), then rexl3-serve detached
#                  under flock -x -n on /tmp/rexl3-gpu.lock (held for the server's whole life, so no other GPU job of the
#                  lab overlaps); the server is killed unless the guard armed on its pid. Waits up to READY_TIMEOUT_S for
#                  /health (200: engine ready) while watching for `level=fatal` or death (then stops and prints the log
#                  tails); prints the capacity lines and boot_s. PROFILE, MAX_MODEL_LEN, SLOTS, KV_TOKENS, PREFILL_CHUNK,
#                  SPEC (--spec), PREFIX_CACHE, VISION (off: --no-vision), MEMORY_FLOOR_GIB, EXTRA_FLAGS and EXTRA_ENV shape it
#   stop           SIGTERM the server, SIGKILL after STOP_GRACE_S, then the guard; checks the GPU holds nothing of ours
#   status         pid, uptime, guard stage and floor, MemAvailable now and the guard's minimum, GPU, last log line, /v1/models
#   logs [N]       last N (default 40) lines of the server log and of the memory guard's log
#   smoke          against the server (bench/tools/smoke.py): a chat, a raw completion, a tool call, reasoning separation and,
#                  unless VISION=off, an image
#   bench PHASE    the bench tools against the running server, from this machine; PHASE is decode (warm, 1 stream, 4 and 8
#                  at once: the c=4 figure is the span to the last reply, the c=8 figure is 8 requests on 4 slots, two waves),
#                  rolling (sustained c=4: four workers in a closed loop, 32 requests; decode_c4_rolling_tok_s and
#                  decode_c4_steady_tok_s, each reply equal to its solo reply; the record of gate speed-c4), prefill (cold 32K, 128K and 224K prompts, those that fit MAX_MODEL_LEN), prefill-sweep (cold 8K, 32K, 128K, 224K, a unique prompt every pass, median of PREFILL_PASSES-1 after a discarded warm-up; PREFILL_SIZES, PREFILL_DECODE_AFTER), prefill-1m (one cold 1M
#                  prompt; needs a 1,048,576 window), needle (a code at 10% and 50% depth at the window), spec (drafted ==
#                  serial, concurrent == solo), cache (cached == uncached), vision (text and image replies against
#                  bench/reference-vision.json), pin (write or extend bench/reference.json and reference-vision.json: only on
#                  the installed engine equal to REXL3_COMMIT), check (the capacity line against the profile, the memory
#                  floor against the guard's record) or all (check, smoke, decode, rolling, prefill, vision, spec, cache, needle,
#                  memory). Phases of the final procedure (scripts/final_run.sh): exactness (record the fixed prompt set at the booted
#                  PREFILL_CHUNK, then concurrent == solo: exactness.py), cache-images (cached == uncached with images) and decode-4k
#                  (c = 1 on a cold 4,096-token prompt, three runs). The records of plan Phase 6.7-6.10 have phases of their own: probe and api (the OpenAI surface:
#                  bench/probe.py, tests/api_conformance.py: 0 FAIL), quality and tools (bench/quality_gate.py, the
#                  tool-call gate with its destructive-action probes), loop (scripts/loop_probe.py: text, reasoning and
#                  agentic modes), sweep (decode and prefill over the context grid, bench/rexl3/), longctx and sixcat (the
#                  pinned SixCat, run from this machine) and profile (the profile bracket: capacity line, GPU MiB and MemAvailable at
#                  ready, then the memory peak: one cold prompt of the window and PROFILE_LANES (default SLOTS; 1 skips) concurrent ones (PROFILE_SOLO=0 skips the solo one), with a
#                  node-side sampler of GPU MiB and MemAvailable and the guard's minimum; rerun it per profile whenever the cache or
#                  the drafter lands); none is part of `all`. Each gate prints `PASS id` or `FAIL id`; METRIC lines, logs and JSON go to bench/runs/<UTC
#                  stamp>-PHASE/ (git-ignored). Without bench/reference.json the replies are measured, not compared (--raw)
#   memwatch begin|report
#                  begin: open a measurement window on the guard; report: MemAvailable minimum in the window, while loading,
#                  once ready, overall (the guard's records)
#   record NAME    create bench/records/<UTC date>-NAME/meta.json: recipe commit and dirty files, pack repo, revision and
#                  directory, engine (commit, binary sha256, pinned commit, lab build, upstream base, --version fields), the
#                  configuration keys that shape the run, the exact serve command, the GPU clock, results[]
#   gpu-clocks lock|status
#                  lock: apply the GPU SM clock cap (GPU_CLOCK_CAP_MHZ in config/pins.sh: 2200 MHz) with `sudo -n nvidia-smi -lgc`
#                  and require its success line; status: show the current and maximum clock. The cap is hardware protection (owner,
#                  2026-11-23): applied before every `start` and every GPU test, never released, raised or changed (there is no
#                  reset and no MHz argument)
#   engine-test host|goldens
#                  the engine's tests on the node from the installed source, engine stopped: host = the tests that need no
#                  GPU; goldens = the GPU golden tests (kernel hashes), under /tmp/rexl3-gpu.lock
#   stop-other     run STOP_OTHER: take down the stack that normally owns the GPU (by default the Inkling-2-Small TP4 group)
#   restore-other  stop this server, then run RESTORE_OTHER
#   help
#
# Configuration: environment > .env > profile (config/profiles/PROFILE.env) > defaults below. Every key is documented in
# .env.example. config/pins.sh is read last and cannot be overridden.
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 )); then echo "inkling2mini.sh needs bash >= 4 (macOS: brew install bash)" >&2; exit 1; fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ── configuration: caller env > .env > profile > defaults ─────────────────────────────────────────
declare -A _CALLER=()
while IFS= read -r _k; do _CALLER[$_k]=1; done < <(compgen -e)
if [[ -f $ROOT/.env ]]; then
  while IFS= read -r _line || [[ -n $_line ]]; do
    [[ $_line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _k=${BASH_REMATCH[1]} _v=${BASH_REMATCH[2]}
    [[ -n ${_CALLER[$_k]:-} ]] && continue
    _v=${_v#"${_v%%[![:space:]]*}"}
    if [[ $_v =~ ^\"(.*)\"[[:space:]]*(#.*)?$ || $_v =~ ^\'(.*)\'[[:space:]]*(#.*)?$ ]]; then _v=${BASH_REMATCH[1]}
    else _v=${_v%%[[:space:]]#*}; [[ $_v == \#* ]] && _v=; _v=${_v%"${_v##*[![:space:]]}"}; fi
    printf -v "$_k" '%s' "$_v"; export "${_k?}"
  done < "$ROOT/.env"
fi
unset _CALLER _k _v _line
# shellcheck source=config/pins.sh
source "$ROOT/config/pins.sh"

: "${NODES:=}"                                                                   # one management address or host name; empty: no node
: "${SSH_USER:=spark}"
: "${MODELS_DIR:=/home/spark/models}"
: "${HF_BIN:=/home/spark/venvs/hf/bin/hf}"
: "${BASE_DIR:=/home/spark/inkling-2-mini-1x-dgx-spark}"                         # engine build, binaries, logs on the node
: "${CACHE_DIR:=/home/spark/.cache/inkling-2-mini-1x-dgx-spark}"                 # HF_HOME of the download
: "${CONTAINER:=inkling-2-mini-1x-dgx-spark}"                                    # this recipe's container name (allowed beside the server)
: "${REXL3_SRC:=$HOME/projects/rexl3-inkling2-loop}"                             # local engine git repository (this machine)
: "${SMALL_MODEL_DIR:=/home/spark/models/Inkling-2-Small-EXL3-4bpw}"             # the Small's pack on the node: engine-test goldens reads its mtp.safetensors (read-only)
: "${ALLOWED_CONTAINERS:=other-service-1 other-service-2}"                                     # containers that may run beside the server
: "${STOP_OTHER:=}"                                                              # shell command: stop the GPU's usual owner
: "${RESTORE_OTHER:=}"                                                           # shell command: bring it back
: "${SERVED_NAME:=inkling-2-mini-1x-dgx-spark}"
: "${PORT:=8000}"
: "${API_KEY:=}"                                                                 # the engine has no API key: refused when set
: "${PROFILE:=bootstrap}"
: "${KV_TOKENS:=}"
: "${PREFILL_CHUNK:=4096}"
: "${EXTRA_FLAGS:=}"                                                             # more rexl3-serve flags
: "${EXTRA_ENV:=}"                                                               # more KEY=VALUE words
: "${READY_TIMEOUT_S:=1800}"                                                     # start: wait this long for /health
# The shipping defaults: every feature has its gate record (docs/DECISIONS.md): the block drafter (4.6), the prefix cache (4.5) and the
# vision tower (4.7) work together (the drafter and the cache serve image prompts; the cache keys an image by its content).
: "${SPEC:=dspark}"                                                              # --spec: dspark|none|copy|auto
: "${PREFIX_CACHE:=on}"                                                          # --prefix-cache on|off, auto (no flag) or MiB
: "${VISION:=on}"                                                                # off: --no-vision
: "${MEMORY_FLOOR_GIB:=}"                                                        # REXL3_MEMORY_FLOOR_GIB (empty: the engine's 8)
: "${STOP_GRACE_S:=60}"                                                          # stop: SIGTERM, then SIGKILL after this
: "${MIN_AVAIL_GIB:=100}"                                                        # doctor/start/fetch: MemAvailable before a boot
: "${BUILD_TIMEOUT_S:=3600}"                                                     # build: cargo is killed after this
: "${GUARD_START_MIB:=2048}"                                                     # guard floor while the pack loads
: "${GUARD_MIB:=1024}"                                                           # guard floor once ready
: "${GUARD_SWAP_MAX_MIB:=64}"                                                    # guard: swap growth that kills the server...
: "${GUARD_SWAP_AVAIL_MIB:=8192}"                                                # ...while MemAvailable is under this (the engine floor)
: "${DROP_CACHES_EVERY_S:=20}"                                                   # guard: page-cache drop while loading
GPU_LOCK=/tmp/rexl3-gpu.lock

# the profile supplies MAX_MODEL_LEN, SLOTS and PREFILL_CHUNK where neither the environment nor .env set them
[[ $PROFILE =~ ^[a-z][a-z0-9-]*$ && -f $ROOT/config/profiles/$PROFILE.env ]] \
  || { echo "PROFILE '$PROFILE' has no file config/profiles/$PROFILE.env" >&2; exit 1; }
PROFILE_MEASURED=
while IFS= read -r _line || [[ -n $_line ]]; do
  [[ -z $_line || $_line == \#* ]] && continue
  [[ $_line =~ ^(PROFILE_MEASURED|MAX_MODEL_LEN|SLOTS)=([0-9]+)$ ]] \
    || { echo "config/profiles/$PROFILE.env: unexpected line '$_line'" >&2; exit 1; }
  _k=${BASH_REMATCH[1]} _v=${BASH_REMATCH[2]}
  case $_k in
    PROFILE_MEASURED) PROFILE_MEASURED=$_v ;;
    *) [[ -n ${!_k:-} ]] || printf -v "$_k" '%s' "$_v" ;;
  esac
done < "$ROOT/config/profiles/$PROFILE.env"
unset _line _k _v
[[ -n $PROFILE_MEASURED ]] || { echo "config/profiles/$PROFILE.env lacks PROFILE_MEASURED" >&2; exit 1; }
: "${MAX_MODEL_LEN:?config/profiles/$PROFILE.env sets no MAX_MODEL_LEN}"
: "${SLOTS:?config/profiles/$PROFILE.env sets no SLOTS}"
: "${KV_TOKENS:=}"
if [[ $MAX_MODEL_LEN =~ ^[0-9]+$ && $SLOTS =~ ^[0-9]+$ && -z $KV_TOKENS ]]; then KV_TOKENS=$((SLOTS * MAX_MODEL_LEN)); fi   # every lane full

# ── validation: everything is checked before any output, so a bad value leaves no partial result ──
_bad() { echo "$*" >&2; exit 1; }
: "${GPU_CLOCK_CAP_MHZ:?config/pins.sh sets no GPU_CLOCK_CAP_MHZ}"
for _k in PORT MAX_MODEL_LEN SLOTS KV_TOKENS PREFILL_CHUNK READY_TIMEOUT_S STOP_GRACE_S MIN_AVAIL_GIB BUILD_TIMEOUT_S GPU_CLOCK_CAP_MHZ \
  GUARD_START_MIB GUARD_MIB GUARD_SWAP_MAX_MIB GUARD_SWAP_AVAIL_MIB DROP_CACHES_EVERY_S; do
  [[ ${!_k} =~ ^[0-9]+$ ]] || _bad "$_k must be a whole number (got '${!_k}')"
done
for _k in PORT MAX_MODEL_LEN SLOTS KV_TOKENS PREFILL_CHUNK GPU_CLOCK_CAP_MHZ; do
  (( ${!_k} > 0 )) || _bad "$_k must be above 0 (got '${!_k}')"
done
# The GPU SM clock cap is hardware protection (owner, 2026-11-23): not a setting, never released, raised or changed. The old key GPU_CLOCKS
# is refused unless it equals the pin; it stays defined as the pin so a record's configuration block still names the clock.
[[ -z ${GPU_CLOCKS:-} || $GPU_CLOCKS == "$GPU_CLOCK_CAP_MHZ" ]] \
  || _bad "GPU_CLOCKS is not a setting: the GPU SM clock cap is $GPU_CLOCK_CAP_MHZ MHz, pinned in config/pins.sh (hardware protection: never released, raised or changed)"
GPU_CLOCKS=$GPU_CLOCK_CAP_MHZ
(( PORT <= 65535 )) || _bad "PORT must be at most 65535 (got '$PORT')"
read -r -a _ew <<<"$EXTRA_ENV"
for _w in "${_ew[@]}"; do
  [[ $_w =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || _bad "EXTRA_ENV takes KEY=VALUE words (got '$_w')"
done
for _k in MODELS_DIR HF_BIN BASE_DIR CACHE_DIR SMALL_MODEL_DIR; do
  [[ ${!_k} =~ ^/[A-Za-z0-9._/-]+$ ]] || _bad "$_k must be an absolute path of letters, digits and ._/- (got '${!_k}')"
done
for _k in SSH_USER SERVED_NAME CONTAINER; do
  [[ ${!_k} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || _bad "$_k must be letters, digits and ._- (got '${!_k}')"
done
[[ -z $API_KEY ]] || _bad "REXL3 serves without an API key: unset API_KEY and keep the port on a private network"
[[ -z $REXL3_COMMIT || $REXL3_COMMIT =~ ^[0-9a-f]{40}$ ]] || _bad "REXL3_COMMIT must be empty or a full commit sha (config/pins.sh)"
[[ $REXL3_UPSTREAM_BASE =~ ^[0-9a-f]{40}$ ]] || _bad "REXL3_UPSTREAM_BASE must be a full commit sha (config/pins.sh)"
[[ $SPEC =~ ^(none|copy|dspark|auto)$ ]] || _bad "SPEC is none, copy, dspark or auto (got '$SPEC')"
[[ $PREFIX_CACHE =~ ^(on|off|auto|[0-9]+)$ ]] || _bad "PREFIX_CACHE is on, off, auto or a whole number of MiB (got '$PREFIX_CACHE')"
[[ $VISION =~ ^(on|off)$ ]] || _bad "VISION is on or off (got '$VISION')"
# `cache_runs`: whether the configured server runs a prefix cache (the cache gate needs one). The cache serves text and images alike.
cache_runs() { [[ $PREFIX_CACHE != off && $PREFIX_CACHE != 0 ]]; }
if [[ -n $MEMORY_FLOOR_GIB ]] && ! { [[ $MEMORY_FLOOR_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v v="$MEMORY_FLOOR_GIB" 'BEGIN {exit !(v >= 0.25)}'; }; then
  _bad "MEMORY_FLOOR_GIB is empty (engine default 8) or a number of GiB >= 0.25 (got '$MEMORY_FLOOR_GIB')"
fi
read -r -a NODE <<<"$NODES"
(( ${#NODE[@]} <= 1 )) || _bad "NODES takes one address or host name (got ${#NODE[@]})"
HOST=${NODE[0]:-}
[[ -z $HOST || $HOST =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || _bad "NODES must be an address or host name (got '$HOST')"
unset _k _w
MODEL_DIR=$MODELS_DIR/$MODEL_NAME
BIN=$BASE_DIR/bin/rexl3-serve
API=http://${HOST:-no-node}:$PORT
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto -o "ControlPath=/tmp/inkling2mini-ssh-%C" -o ControlPersist=60)

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'FAIL %s\n' "$*" >&2; exit 1; }
need_node() { [[ -n $HOST ]] || die "no node: NODES is empty (set it in .env, see .env.example)"; }

# remote SCRIPT [VAR=value ...]: run SCRIPT under `bash -s` on the node with the variables exported (by name: the script never
# reads $1). The script travels on stdin as one brace group with stdin from /dev/null, so no child that reads stdin can eat the
# rest of it.
remote() {
  local script=$1 kv prelude="set -euo pipefail"$'\n'; shift
  need_node
  for kv in "$@"; do
    [[ ${kv%%=*} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "remote: '${kv%%=*}' is not a variable name"
    prelude+="export ${kv%%=*}=$(printf '%q' "${kv#*=}")"$'\n'
  done
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" bash -s <<<"$prelude{"$'\n'"$script"$'\n'"} </dev/null"
}

# ensure_gpu_cap: the GPU SM clock cap (GPU_CLOCK_CAP_MHZ, config/pins.sh) is hardware protection and non-negotiable (owner, 2026-11-23):
# it is in place whenever any work runs on a node. It is applied (the same value again is harmless) before every server start and every
# GPU test, and its success line is required: the lock does not survive a reboot or a driver reload, and an idle clock reading near the
# cap does not prove it. Without sudo, or without the success line, there is no GPU work. Nothing here releases, raises or changes it.
ensure_gpu_cap() {
  local out
  need_node
  out=$(remote 'sudo -n nvidia-smi -lgc "$CAP,$CAP" 2>&1' CAP="$GPU_CLOCK_CAP_MHZ") \
    && grep -qF "gpuClkMin $GPU_CLOCK_CAP_MHZ, gpuClkMax $GPU_CLOCK_CAP_MHZ" <<<"$out" \
    || die "the GPU cap of $GPU_CLOCK_CAP_MHZ MHz could not be applied on $HOST (sudo -n nvidia-smi -lgc $GPU_CLOCK_CAP_MHZ,$GPU_CLOCK_CAP_MHZ): no GPU work without it${out:+: $out}"
  log "GPU SM clock capped at $GPU_CLOCK_CAP_MHZ MHz on $HOST: $(grep -F "gpuClkMin $GPU_CLOCK_CAP_MHZ, gpuClkMax $GPU_CLOCK_CAP_MHZ" <<<"$out" | head -n 1)"   # the proof line stays in the raw output of the run
}

# check_manifest: the pack manifest must be the one the pins name (the byte authority for every size and sha256 below).
check_manifest() {
  [[ $MODEL_REV =~ ^[0-9a-f]{40}$ ]] || die "MODEL_REV must be a full commit sha (config/pins.sh)"
  local have; have=$( (shasum -a 256 "$ROOT/$MANIFEST_FILE" 2>/dev/null || sha256sum "$ROOT/$MANIFEST_FILE") | cut -d' ' -f1)
  [[ $have == "$MANIFEST_SHA256" ]] || die "$MANIFEST_FILE sha256 $have != MANIFEST_SHA256 in config/pins.sh"
}
manifest_files() { awk -F'\t' '{print $3}' "$ROOT/$MANIFEST_FILE"; }                 # paths, one a line
manifest_sizes() { awk -F'\t' '{print $2, $3}' "$ROOT/$MANIFEST_FILE"; }              # `size path` lines
manifest_sums() { awk -F'\t' '{print $1 "  " $3}' "$ROOT/$MANIFEST_FILE"; }           # sha256sum -c format

# ── setup ──────────────────────────────────────────────────────────────────────────────────────
cmd_preflight() {
  need_node
  check_manifest
  local pack_gib; pack_gib=$(manifest_sizes | awk '{s += $1} END {printf "%d", s / 1073741824 + 1}')
  log "preflight on $HOST"
  remote '
    [[ $(uname -m) == aarch64 ]] || { echo "FAIL architecture $(uname -m), need aarch64"; exit 1; }
    echo "ok   arch: aarch64"
    gpus=$(nvidia-smi -L 2>/dev/null || true)
    grep -q GB10 <<<"$gpus" || { echo "FAIL nvidia-smi -L shows no NVIDIA GB10: ${gpus:-no GPU}"; exit 1; }
    echo "ok   gpu: $(head -n 1 <<<"$gpus")"
    mem=$(awk "/MemTotal/ {printf \"%d\", \$2/1048576}" /proc/meminfo)
    (( mem >= 110 )) || { echo "FAIL memory: ${mem} GiB MemTotal, need 110"; exit 1; }
    echo "ok   memory: ${mem} GiB"
    for t in git curl rsync flock tar python3 sha256sum; do command -v "$t" >/dev/null || { echo "FAIL $t missing"; exit 1; }; done
    echo "ok   tools: git curl rsync flock tar python3 sha256sum"
    mkdir -p "$MODELS_DIR" "$BASE_DIR"
    have=$(du -s -B1G "$MODEL_DIR" 2>/dev/null | cut -f1 || true); have=${have:-0}
    free=$(df -B1G --output=avail "$MODELS_DIR" | tail -n 1 | tr -d " ")
    (( free + have >= PACK_GIB )) || { echo "FAIL disk: ${free} GiB free at $MODELS_DIR (+ ${have} GiB of the pack already there), need ${PACK_GIB}"; exit 1; }
    echo "ok   disk: ${free} GiB free at $MODELS_DIR (${have} GiB of the pack already there; the pack needs ${PACK_GIB})"
    free=$(df -B1G --output=avail "$BASE_DIR" | tail -n 1 | tr -d " ")
    (( free >= 20 )) || { echo "FAIL disk: ${free} GiB free at $BASE_DIR, need 20 for the engine build"; exit 1; }
    echo "ok   disk: ${free} GiB free at $BASE_DIR"
    busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
    [[ -z $busy ]] || { echo "FAIL a process holds the GPU: $busy"; exit 1; }
    echo "ok   gpu: idle"
    if sudo -n true 2>/dev/null; then echo "ok   sudo -n works (the GPU cap, the page-cache drop)"
    else echo "FAIL sudo -n does not work: the GPU cap cannot be applied, and no GPU work runs without it"; exit 1; fi' \
    MODELS_DIR="$MODELS_DIR" MODEL_DIR="$MODEL_DIR" BASE_DIR="$BASE_DIR" PACK_GIB="$pack_gib" </dev/null || die "preflight failed on $HOST"
  log "preflight passed"
}

# install: the Rust toolchain of the pin, the hf CLI and the directories. The engine is built by `build`, never here.
cmd_install() {
  need_node
  log "installing on $HOST: Rust $RUST_TOOLCHAIN, hf CLI, directories"
  remote '
    export PATH="$HOME/.cargo/bin:$PATH"
    [[ -x /usr/local/cuda/bin/nvcc ]] || { echo "FAIL /usr/local/cuda/bin/nvcc missing (DGX OS ships CUDA 13)"; exit 1; }
    echo "nvcc: $(/usr/local/cuda/bin/nvcc --version | sed -n "s/.*release \([0-9.]*\).*/\1/p")"
    if ! command -v rustup >/dev/null; then
      echo "installing rustup (minimal profile, no default toolchain)"
      curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none >/dev/null
    fi
    rustup toolchain list | grep -q "^$RUST_TOOLCHAIN" || rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
    echo "rust: $(rustup run "$RUST_TOOLCHAIN" rustc --version)"
    mkdir -p "$BASE_DIR/bin" "$BASE_DIR/run" "$BASE_DIR/logs" "$MODELS_DIR" "$CACHE_DIR"
    if [[ ! -x $HF_BIN ]]; then
      venv=${HF_BIN%/bin/hf}
      python3 -m venv "$venv"
      "$venv/bin/pip" install -q huggingface_hub
    fi
    [[ -x $HF_BIN ]] || { echo "FAIL $HF_BIN is not executable after the install"; exit 1; }
    echo "hf: $("$HF_BIN" version 2>/dev/null | head -n 1 || echo installed)"' \
    BASE_DIR="$BASE_DIR" MODELS_DIR="$MODELS_DIR" CACHE_DIR="$CACHE_DIR" HF_BIN="$HF_BIN" RUST_TOOLCHAIN="$RUST_TOOLCHAIN" </dev/null \
    || die "install failed on $HOST"
}

cmd_setup() { cmd_preflight; cmd_install; cmd_fetch; log "setup complete: ./inkling2mini.sh build REV (or build, once REXL3_COMMIT is pinned), then start"; }

# ── pack ────────────────────────────────────────────────────────────────────────────────────────
cmd_verify() {
  need_node
  check_manifest
  remote '
    cd "$MODEL_DIR" 2>/dev/null || { echo "FAIL $MODEL_DIR missing" >&2; exit 1; }
    if printf "%s\n" "$MANIFEST" | sha256sum -c --quiet - >/dev/null 2>&1
    then echo "verified $(printf "%s\n" "$MANIFEST" | wc -l) files under $MODEL_DIR"
    else echo "FAIL sha256 mismatch or missing file under $MODEL_DIR" >&2; exit 1; fi' \
    MODEL_DIR="$MODEL_DIR" MANIFEST="$(manifest_sums)" || die "checksum verification failed"
}

cmd_fetch() {
  need_node
  check_manifest
  log "$MODEL_REPO@$MODEL_REV -> $MODEL_DIR on $HOST"
  local -a tok=(); [[ -z ${HF_TOKEN:-} ]] || tok=(HF_TOKEN="$HF_TOKEN")
  remote '
    # a 14 GiB write goes through the page cache: not next to another stack that holds the memory
    avail=$(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo)
    (( avail >= MIN_AVAIL )) || { echo "FAIL MemAvailable ${avail} GiB < $MIN_AVAIL: another stack holds this node" >&2; exit 1; }
    [[ -x $HF_BIN ]] || { echo "FAIL $HF_BIN missing: run install first" >&2; exit 1; }
    mkdir -p "$MODEL_DIR" "$CACHE_DIR"
    HF_HOME="$CACHE_DIR" HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
      "$HF_BIN" download "$REPO" --revision "$REV" --local-dir "$MODEL_DIR" $FILES >/dev/null
    echo "downloaded $(wc -w <<<"$FILES") files"' \
    MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" HF_BIN="$HF_BIN" REPO="$MODEL_REPO" REV="$MODEL_REV" MIN_AVAIL="$MIN_AVAIL_GIB" \
    FILES="$(manifest_files | tr '\n' ' ')" ${tok[@]+"${tok[@]}"} || die "download failed"
  cmd_verify
  log "fetch: the pack is on $HOST and verified"
}

# ── doctor ──────────────────────────────────────────────────────────────────────────────────────
# cmd_doctor [start]: read-only. `start` (the start preflight) also requires an installed engine. A working sudo -n is required in every
# mode: the GPU cap (ensure_gpu_cap) and the guard's page-cache drop need it.
cmd_doctor() {
  local mode=${1:-} out bad=0 commit
  check_manifest
  log "pins: manifest $MANIFEST_FILE matches MANIFEST_SHA256; pack $MODEL_REPO@${MODEL_REV:0:12}"
  need_node
  out=$(remote '
      [[ $(uname -m) == aarch64 ]] && echo "ok   arch aarch64" || echo "FAIL architecture $(uname -m), need aarch64"
      gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)
      [[ $gpu == *GB10* ]] && echo "ok   gpu ${gpu%%, *}, driver ${gpu##*, }" || echo "FAIL no NVIDIA GB10 (nvidia-smi: ${gpu:-none})"
      busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
      if [[ -z $busy ]]; then echo "ok   gpu idle"
      else
        echo "FAIL GPU in use:"
        while IFS=", " read -r p _; do
          echo "     $(ps -o pid=,user=,etime=,args= -p "$p" 2>/dev/null | cut -c1-160 || echo "$p (gone)")"
        done <<<"$busy"
      fi
      # the lab GPU lock (the Inkling-2-Small ranks and the engine tests hold it): free, or name its holders
      if [[ -e $GPU_LOCK ]] && ! flock -n -x "$GPU_LOCK" true 2>/dev/null; then
        echo "FAIL $GPU_LOCK is held:"
        for d in /proc/[0-9]*; do
          if ls -l "$d/fd" 2>/dev/null | grep -q " $GPU_LOCK\$"; then echo "     pid ${d#/proc/}: $(tr "\0" " " < "$d/cmdline" 2>/dev/null | cut -c1-140)"; fi
        done
      else echo "ok   $GPU_LOCK free"; fi
      ours=$(pgrep -f "^$BASE_DIR/bin/[r]exl3-serve( |\$)" | paste -sd" " - || true)
      [[ -z $ours ]] && echo "ok   no rexl3-serve of this recipe running" || echo "FAIL rexl3-serve already running here (pid $ours): ./inkling2mini.sh stop"
      avail=$(awk "/MemAvailable/ {printf \"%d\", \$2/1048576}" /proc/meminfo)
      (( avail >= MIN_AVAIL )) && echo "ok   MemAvailable ${avail} GiB" || echo "FAIL MemAvailable ${avail} GiB < $MIN_AVAIL"
      for c in $(docker ps --format "{{.Names}}" 2>/dev/null); do
        [[ " $ALLOWED $CONTAINER " == *" $c "* ]] || echo "FAIL foreign container running: $c"
      done
      missing=0
      while read -r size path; do
        [[ $(stat -c %s "$MODEL_DIR/$path" 2>/dev/null) == "$size" ]] || missing=$((missing + 1))
      done <<<"$FILES"
      (( missing == 0 )) && echo "ok   pack: $(wc -l <<<"$FILES") files at full size under $MODEL_DIR (verify checks bytes)" \
        || echo "FAIL pack: $missing files missing or short under $MODEL_DIR (./inkling2mini.sh fetch)"
      command -v flock >/dev/null && echo "ok   flock present" || echo "FAIL flock missing"
      if sudo -n true 2>/dev/null; then echo "ok   sudo -n works (the GPU cap, the page-cache drop)"
      else echo "FAIL sudo -n does not work: the GPU cap cannot be applied (no GPU work runs without it) and the guard cannot drop the page cache"; fi
      if [[ -x $BASE_DIR/bin/rexl3-serve ]]; then
        commit=$("$BASE_DIR/bin/rexl3-serve" --version 2>/dev/null | awk "\$1 == \"commit\" {print \$2}")
        [[ $commit == "$(cat "$BASE_DIR/bin/rexl3-serve.commit" 2>/dev/null)" ]] \
          || echo "FAIL engine: --version reports commit ${commit:-none}, rexl3-serve.commit says otherwise"
        echo "engine ${commit:-unknown} $(sha256sum "$BASE_DIR/bin/rexl3-serve" | cut -d" " -f1)"
      else echo "engine none"; fi' \
      MODEL_DIR="$MODEL_DIR" BASE_DIR="$BASE_DIR" ALLOWED="$ALLOWED_CONTAINERS" CONTAINER="$CONTAINER" MIN_AVAIL="$MIN_AVAIL_GIB" \
      GPU_LOCK="$GPU_LOCK" FILES="$(manifest_sizes)") \
    || { [[ -z $out ]] || echo "$out"; die "cannot reach $HOST over ssh, or the doctor script failed there"; }
  echo "$out"
  grep -q '^FAIL' <<<"$out" && bad=1
  if [[ $mode == start ]] && grep -q '^engine none' <<<"$out"; then echo "FAIL no engine installed (./inkling2mini.sh build)"; bad=1; fi
  commit=$(awk '$1 == "engine" {print $2}' <<<"$out")
  if [[ -z $REXL3_COMMIT ]]; then echo "note REXL3_COMMIT is not pinned yet (plan step 4.8): any installed engine is a lab build"
  elif [[ $commit != "$REXL3_COMMIT" ]]; then echo "note installed engine is not REXL3_COMMIT ${REXL3_COMMIT:0:12} (lab build or none)"; fi
  (( bad == 0 )) && log "doctor: the node is ready" || die "doctor found problems"
}

# ── build ───────────────────────────────────────────────────────────────────────────────────────
cmd_build() {
  local rev=${1:-} sha archive archive_sum stage got
  if [[ -z $rev ]]; then
    [[ -n $REXL3_COMMIT ]] \
      || die "REXL3_COMMIT in config/pins.sh is empty: the engine pin is set in plan step 4.8 (golden and real-pack tests against the oracle); until then build a lab revision: ./inkling2mini.sh build REV"
    rev=$REXL3_COMMIT
  fi
  need_node
  sha=$(git -C "$REXL3_SRC" rev-parse --verify "$rev^{commit}" 2>/dev/null) || die "$rev is not a commit in $REXL3_SRC"
  git -C "$REXL3_SRC" merge-base --is-ancestor "$REXL3_UPSTREAM_BASE" "$sha" \
    || die "$sha does not descend from the upstream base ${REXL3_UPSTREAM_BASE:0:12}"
  if [[ $sha == "$REXL3_COMMIT" ]]; then log "build the pinned engine ${sha:0:12} on $HOST"
  else log "build a LAB engine ${sha:0:12} on $HOST (not REXL3_COMMIT: engine.lab_build=true in records)"; fi
  ensure_gpu_cap                                    # a build is work on the node too: the cap first
  archive=$(mktemp)
  git -C "$REXL3_SRC" archive --format=tar "$sha" > "$archive"
  archive_sum=$( (shasum -a 256 "$archive" 2>/dev/null || sha256sum "$archive") | cut -d' ' -f1)
  stage=$BASE_DIR/run/src-$sha-$$-$RANDOM.tar
  # the copy is verified on the destination
  got=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" "mkdir -p '$BASE_DIR/run' && cat > '$stage' && sha256sum '$stage' | cut -d' ' -f1" < "$archive") \
    || { rm -f "$archive"; die "could not copy the source to $HOST"; }
  rm -f "$archive"
  [[ $got == "$archive_sum" ]] || { ssh -n "${SSH_OPTS[@]}" "$SSH_USER@$HOST" "rm -f '$stage'" || true; die "the source archive arrived damaged ($got != $archive_sum)"; }
  # git archive stamps every file with the commit's time; cargo trusts file times, so a build after a newer build of another tree
  # would link that tree's crates under this commit's name. The touch makes every source newer than any earlier artifact.
  remote '
    mkdir -p "$BASE_DIR/run" "$BASE_DIR/logs" "$BASE_DIR/bin"
    exec 8>"$BASE_DIR/run/node.lock"
    flock -x -n 8 || { echo "FAIL another build or run holds $BASE_DIR/run/node.lock" >&2; rm -f "$STAGE"; exit 1; }
    pgrep -f "[r]exl3-serve" >/dev/null && { echo "FAIL a rexl3-serve is running on this node: stop it first" >&2; rm -f "$STAGE"; exit 1; }
    [[ $(sha256sum "$STAGE" | cut -d" " -f1) == "$TAR_SUM" ]] || { echo "FAIL the staged archive changed" >&2; exit 1; }
    rm -rf "$BASE_DIR/src.new" "$BASE_DIR/src"; mkdir -p "$BASE_DIR/src.new"
    tar -x -C "$BASE_DIR/src.new" -f "$STAGE"; rm -f "$STAGE"
    find "$BASE_DIR/src.new" -type f -exec touch {} +
    mv "$BASE_DIR/src.new" "$BASE_DIR/src"
    cd "$BASE_DIR/src"
    export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda REXL3_CUDA_ARCH=sm_121 \
      REXL3_COMMIT="$SHA" CARGO_TARGET_DIR="$BASE_DIR/target" CARGO_BUILD_JOBS=8
    timeout "$BUILD_TIMEOUT_S" cargo +"$TOOLCHAIN" build --release --locked -p gb-server --bin rexl3-serve -p gb-cluster --bin rexl3-cluster \
      > "$BASE_DIR/logs/build.log" 2>&1 \
      || { grep -E "^error" -A6 "$BASE_DIR/logs/build.log" | head -n 60; echo "FAIL build (timeout ${BUILD_TIMEOUT_S}s; log: $BASE_DIR/logs/build.log)" >&2; exit 1; }
    v=$("$BASE_DIR/target/release/rexl3-serve" --version 2>/dev/null)
    grep -q "^commit *$SHA\$" <<<"$v" || { echo "FAIL the built binary reports another commit:" >&2; echo "$v" >&2; exit 1; }
    grep -Eq "^cuda_arch *sm_121\$" <<<"$v" || { echo "FAIL the built binary is not sm_121:" >&2; echo "$v" >&2; exit 1; }
    for b in rexl3-serve rexl3-cluster; do cp "$BASE_DIR/target/release/$b" "$BASE_DIR/bin/$b.new" && mv -f "$BASE_DIR/bin/$b.new" "$BASE_DIR/bin/$b"; done
    echo "$SHA" > "$BASE_DIR/bin/rexl3-serve.commit"
    [[ $(cat "$BASE_DIR/bin/rexl3-serve.commit") == "$SHA" ]] || { echo "FAIL rexl3-serve.commit was not written" >&2; exit 1; }
    echo "built $SHA: rexl3-serve sha256 $(sha256sum "$BASE_DIR/bin/rexl3-serve" | cut -d" " -f1)"' \
    BASE_DIR="$BASE_DIR" SHA="$sha" TOOLCHAIN="$RUST_TOOLCHAIN" STAGE="$stage" TAR_SUM="$archive_sum" BUILD_TIMEOUT_S="$BUILD_TIMEOUT_S" \
    || die "build failed on $HOST"
  log "build: engine ${sha:0:12} installed under $BASE_DIR/bin"
}

# ── serving ─────────────────────────────────────────────────────────────────────────────────────
# serve_cmd: the one command line the server runs, environment included. `env -i`: nothing from the ssh session reaches the
# server. print-serve prints this string and start hands this very string to the node. TP1: one process, no --tp-* flags.
serve_cmd() {
  local w
  local -a extra_env extra_flags env args wrap
  read -r -a extra_env <<<"$EXTRA_ENV"
  read -r -a extra_flags <<<"$EXTRA_FLAGS"
  read -r -a wrap <<<"${SERVE_WRAP:-}"   # lab only (profiling): words placed before the server binary, e.g. an `nsys profile ...` command line
  env=(HOME="/home/$SSH_USER" USER="$SSH_USER" LOGNAME="$SSH_USER"
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin RUST_BACKTRACE=1)
  [[ -z $MEMORY_FLOOR_GIB ]] || env+=(REXL3_MEMORY_FLOOR_GIB="$MEMORY_FLOOR_GIB")
  env+=("${extra_env[@]}")
  args=(--model "$MODEL_DIR" --max-model-len "$MAX_MODEL_LEN" --slots "$SLOTS" --kv-tokens "$KV_TOKENS"
    --prefill-chunk "$PREFILL_CHUNK" --kv-dtype int8 --spec "$SPEC")
  case $PREFIX_CACHE in
    on|off) args+=(--prefix-cache "$PREFIX_CACHE") ;;
    auto) ;;   # the engine's default: on when the KV plan leaves memory spare
    *) args+=(--prefix-cache-mib "$PREFIX_CACHE") ;;
  esac
  [[ $VISION == on ]] || args+=(--no-vision)
  args+=(--allow-unvalidated-pack --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" "${extra_flags[@]}")
  printf 'exec env -i'
  for w in "${env[@]}" "${wrap[@]}" "$BIN" "${args[@]}"; do printf ' %q' "$w"; done
  printf '\n'
}

cmd_print_serve() {
  local out label=${HOST:+$SSH_USER@$HOST}
  out=$(printf '# profile %s (PROFILE_MEASURED=%s), %s, HTTP :%s\n' "$PROFILE" "$PROFILE_MEASURED" "${label:-no node}" "$PORT"
    serve_cmd)
  printf '%s\n' "$out"
}

api_up() { curl -fsS -m 5 -o /dev/null "$API/health" 2>/dev/null; }

# launch: the guard first (it validates itself, then waits for the pid file), then the server, detached (setsid nohup; the
# previous log becomes .prev); fails, with the server killed, unless the guard armed on the server's pid.
launch() {
  remote '
    mkdir -p "$BASE_DIR/logs" "$BASE_DIR/run"
    log=$BASE_DIR/logs/serve.log glog=$BASE_DIR/logs/memguard.log gmin=$BASE_DIR/logs/memguard.min
    pidf=$BASE_DIR/run/serve.pid gpidf=$BASE_DIR/run/memguard.pid stage=$BASE_DIR/run/stage
    if [[ -s $pidf ]] && kill -0 "$(<"$pidf")" 2>/dev/null; then echo "FAIL already running (pid $(<"$pidf"))" >&2; exit 1; fi
    for f in "$log" "$glog" "$gmin"; do if [[ -f $f ]]; then mv -f "$f" "$f.prev"; fi; done
    rm -f "$pidf" "$gpidf"
    printf "%s\n" "$GUARD_SRC" > "$BASE_DIR/run/memguard.sh"
    echo loading > "$stage"
    GUARD_PIDFILE=$pidf GUARD_LOG=$glog GUARD_MIN_FILE=$gmin GUARD_STAGE_FILE=$stage \
    GUARD_WINDOW_FILE=$BASE_DIR/run/memwatch.window GUARD_SERVE_LOG=$log GUARD_START_MIB=$G_START GUARD_MIB=$G_MIB \
    GUARD_SWAP_MAX_MIB=$G_SWAP GUARD_SWAP_AVAIL_MIB=$G_SWAP_AVAIL DROP_CACHES_EVERY_S=$G_DROP \
      setsid nohup bash "$BASE_DIR/run/memguard.sh" >> "$glog" 2>&1 < /dev/null &
    gpid=$!
    echo "$gpid" > "$gpidf"
    for _ in $(seq 120); do   # 30 s: the first page-cache drop of the guard includes a sync
      if grep -q " waiting for " "$glog" 2>/dev/null || ! kill -0 "$gpid" 2>/dev/null; then break; fi
      sleep 0.25
    done
    if ! kill -0 "$gpid" 2>/dev/null || ! grep -q " waiting for " "$glog"; then
      echo "FAIL the memory guard did not start, so the server is not started:" >&2; cat "$glog" >&2; exit 1
    fi
    cd "$BASE_DIR"
    # the lab GPU lock, held exclusively for the server life: fd 9 survives the exec into rexl3-serve and closes with it
    setsid nohup bash -c "exec 9>>/tmp/rexl3-gpu.lock; flock -x -n 9 || { echo \"FAIL /tmp/rexl3-gpu.lock is held by another GPU job\"; exit 75; }; $CMD" \
      > "$log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$pidf"
    for _ in $(seq 40); do
      if grep -q " armed pid=$pid " "$glog" || ! kill -0 "$gpid" 2>/dev/null; then break; fi
      sleep 0.25
    done
    if ! grep -q " armed pid=$pid " "$glog" || ! kill -0 "$gpid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true; kill "$gpid" 2>/dev/null || true
      echo "FAIL the memory guard did not arm on pid $pid; the server is killed:" >&2; tail -n 5 "$glog" >&2; exit 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "FAIL rexl3-serve exited at once:" >&2; tail -n 20 "$log" >&2; exit 1; fi
    echo "pid $pid, guard $gpid, log $log"' \
    BASE_DIR="$BASE_DIR" CMD="$(serve_cmd)" GUARD_SRC="$(<"$ROOT/scripts/memguard.sh")" \
    G_START="$GUARD_START_MIB" G_MIB="$GUARD_MIB" G_SWAP="$GUARD_SWAP_MAX_MIB" G_SWAP_AVAIL="$GUARD_SWAP_AVAIL_MIB" G_DROP="$DROP_CACHES_EVERY_S"
}

# Boot watch: the server and its guard alive, no `level=fatal` line; prints the last event seen.
PROBE='
  log=$BASE_DIR/logs/serve.log rc=0
  pid=$(cat "$BASE_DIR/run/serve.pid" 2>/dev/null || true)
  gpid=$(cat "$BASE_DIR/run/memguard.pid" 2>/dev/null || true)
  if [[ -z $pid ]] || ! kill -0 "$pid" 2>/dev/null; then echo "DEAD rexl3-serve (pid ${pid:-none})"; rc=1; fi
  if [[ -z $gpid ]] || ! kill -0 "$gpid" 2>/dev/null; then echo "DEAD memory guard (pid ${gpid:-none})"; rc=1; fi
  if grep -q "level=fatal" "$log" 2>/dev/null; then echo "FATAL $(grep -m1 "level=fatal" "$log" | cut -c1-300)"; rc=1; fi
  echo "last $(grep -o "event=[A-Za-z0-9_]*" "$log" 2>/dev/null | tail -n 1 || true)"
  exit $rc'

print_tails() {
  remote 'echo "== $BASE_DIR/logs/serve.log"; tail -n 40 "$BASE_DIR/logs/serve.log" 2>/dev/null || echo "(no log)"
    echo "== memguard"; tail -n 6 "$BASE_DIR/logs/memguard.log" 2>/dev/null || echo "(no guard log)"' BASE_DIR="$BASE_DIR" || true
}

cmd_start() {

  cmd_doctor start
  ensure_gpu_cap                                    # before anything is launched: no server runs on a node without the cap
  local out t0 boot_s last=0
  t0=$(date +%s)
  log "launch on $HOST"
  if ! launch; then
    log "the server did not launch: stopping"; print_tails; ( cmd_stop ) || true; die "start failed"
  fi
  log "waiting up to ${READY_TIMEOUT_S}s for $API/health"
  until api_up; do
    if ! out=$(remote "$PROBE" BASE_DIR="$BASE_DIR"); then
      echo "$out"; print_tails; log "the server failed during boot: stopping"; ( cmd_stop ) || true
      die "boot failed after $(( $(date +%s) - t0 ))s"
    fi
    if (( $(date +%s) - t0 >= READY_TIMEOUT_S )); then
      echo "$out"; print_tails; log "not ready after ${READY_TIMEOUT_S}s: stopping"; ( cmd_stop ) || true
      die "boot timed out (READY_TIMEOUT_S=$READY_TIMEOUT_S)"
    fi
    if (( $(date +%s) - last >= 30 )); then
      log "booting $(( $(date +%s) - t0 ))s: $(grep '^last ' <<<"$out" | sed 's/^last //')"
      last=$(date +%s)
    fi
    sleep 5
  done
  boot_s=$(( $(date +%s) - t0 ))
  remote 'echo ready > "$BASE_DIR/run/stage"' BASE_DIR="$BASE_DIR" >/dev/null \
    || die "ready, but the guard could not be told (it keeps the loading floor GUARD_START_MIB)"
  remote 'grep -h -E "event=[A-Za-z0-9_]*(plan|capacity|ready)" "$BASE_DIR/logs/serve.log" | tail -n 3 | cut -c1-400' BASE_DIR="$BASE_DIR" || true
  log "start: ready at $API/v1 ($SERVED_NAME)"
  echo "boot_s=$boot_s"
}

# stop: SIGTERM the server (pid file, and any rexl3-serve of this BASE_DIR), SIGKILL after STOP_GRACE_S, then its guard (after the
# server: a guard exits by itself once its server is gone).
cmd_stop() {
  need_node
  remote '
    live=""
    for p in $(cat "$BASE_DIR/run/serve.pid" 2>/dev/null || true) $(pgrep -f "^$BASE_DIR/bin/[r]exl3-serve( |\$)" || true); do
      if kill -0 "$p" 2>/dev/null && [[ " $live " != *" $p "* ]]; then live+=" $p"; fi
    done
    how=none
    if [[ -n $live ]]; then
      kill -TERM $live 2>/dev/null || true; how=SIGTERM
      for (( i = 0; i < GRACE; i++ )); do
        still=""; for p in $live; do if kill -0 "$p" 2>/dev/null; then still+=" $p"; fi; done
        if [[ -z $still ]]; then break; fi
        sleep 1
      done
      still=""; for p in $live; do if kill -0 "$p" 2>/dev/null; then still+=" $p"; fi; done
      if [[ -n $still ]]; then kill -KILL $still 2>/dev/null || true; how="SIGKILL after ${GRACE}s"; sleep 1; fi
    fi
    for g in $(cat "$BASE_DIR/run/memguard.pid" 2>/dev/null || true) $(pgrep -f "^bash $BASE_DIR/run/[m]emguard.sh" || true); do
      kill "$g" 2>/dev/null || true
    done
    rm -f "$BASE_DIR/run/serve.pid" "$BASE_DIR/run/memguard.pid" "$BASE_DIR/run/stage"
    echo "stopped: ${how}${live:+ (pid$live)}"
    # the driver frees a dead process memory within seconds: none of our pids may still hold the GPU
    for _ in $(seq 10); do
      held=""
      for p in $live; do
        if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qx "$p"; then held+=" $p"; fi
      done
      [[ -z $held ]] && break
      sleep 2
    done
    [[ -z ${held:-} ]] || { echo "FAIL pid$held still holds the GPU after stop" >&2; exit 1; }
    echo "gpu: nothing of ours"' BASE_DIR="$BASE_DIR" GRACE="$STOP_GRACE_S" || die "could not stop the server"
  log "stop: the server and its guard are gone"
}

cmd_status() {
  need_node
  remote '
    log=$BASE_DIR/logs/serve.log glog=$BASE_DIR/logs/memguard.log gmin=$BASE_DIR/logs/memguard.min
    pid=$(cat "$BASE_DIR/run/serve.pid" 2>/dev/null || true)
    gpid=$(cat "$BASE_DIR/run/memguard.pid" 2>/dev/null || true)
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
      up=$(ps -o etimes= -p "$pid" | tr -d " ")
      printf "serve   pid %s, up %dh%02dm%02ds\n" "$pid" $((up / 3600)) $((up % 3600 / 60)) $((up % 60))
    else echo "serve   down${pid:+ (stale pid file: $pid)}"; fi
    if [[ -n $gpid ]] && kill -0 "$gpid" 2>/dev/null; then echo "guard   pid $gpid, $(cat "$BASE_DIR/run/stage" 2>/dev/null || echo "?") floor"
    elif grep -q " KILL " "$glog" 2>/dev/null; then echo "guard   FIRED: $(grep " KILL " "$glog" | tail -n 1)"
    else echo "guard   off"; fi
    set +u; declare -A g; if [[ -f $gmin ]]; then while IFS== read -r k v; do g[$k]=$v; done < "$gmin"; fi
    avail=$(awk "/^MemAvailable:/ {printf \"%d\", \$2 / 1024}" /proc/meminfo)
    if [[ -f $gmin ]]; then
      echo "memory  MemAvailable $avail MiB now; guard min ${g[min_mib]} MiB at ${g[min_at]} (loading ${g[loading_min_mib]}, ready ${g[ready_min_mib]}, window ${g[window_min_mib]}); swap ${g[swap_now_mib]} MiB (base ${g[swap_base_mib]})"
    else echo "memory  MemAvailable $avail MiB now; no guard record"; fi
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed")
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | paste -sd";" - || true)
    echo "gpu     util/mem/temp/power/sm clock: $gpu; processes: ${apps:-none}"
    echo "log     $(tail -n 1 "$log" 2>/dev/null | cut -c1-200)"' BASE_DIR="$BASE_DIR" || true
  local models
  if models=$(curl -fsS -m 5 "$API/v1/models" 2>&1); then echo "$API/v1/models: $models"
  else echo "$API/v1/models: down ($models)"; return 1; fi
}

cmd_logs() {
  local n=${1:-40}
  [[ $n =~ ^[0-9]+$ ]] || die "usage: logs [N]"
  need_node
  echo "== $HOST $BASE_DIR/logs/serve.log"
  remote 'tail -n "$N" "$BASE_DIR/logs/serve.log" 2>/dev/null || echo "(no log)"
    if grep -q " KILL " "$BASE_DIR/logs/memguard.log" 2>/dev/null; then
      echo "== the memory guard killed the server:"; grep -A 10 " KILL " "$BASE_DIR/logs/memguard.log" | tail -n 11
    fi' BASE_DIR="$BASE_DIR" N="$n" || echo "(ssh failed)"
}

# smoke: VISION=off boots skip the image check (the server refuses image parts without a tower)
cmd_smoke() {
  local -a skip=(); [[ $VISION == on ]] || skip=(--skip-image)
  python3 "$ROOT/bench/tools/smoke.py" --url "$API" --model "$SERVED_NAME" "${skip[@]}"
}

# ── bench: the bench tools against the running server ───────────────────────────────────────────
# bench_gate ID CMD...: run CMD (output in $BENCH_OUT/ID.log) and print and record PASS ID or FAIL ID
bench_gate() {
  local id=$1; shift
  log "gate $id: $*"
  if "$@" > "$BENCH_OUT/$id.log" 2>&1; then echo "PASS $id" | tee -a "$BENCH_OUT/gates.txt"
  else echo "FAIL $id" | tee -a "$BENCH_OUT/gates.txt"; tail -n 25 "$BENCH_OUT/$id.log" >&2; BENCH_FAIL=1; fi
}

# bench_run LABEL PHASES [bench.py args]: one bench.py run; its METRIC lines and its PASS/FAIL pin line are kept
bench_run() {
  local label=$1 phases=$2 rc=0; shift 2
  local -a mode=(--raw)
  if [[ ${1:-} == --pin ]]; then mode=()
  elif [[ -f $ROOT/bench/reference.json ]]; then mode=(--ref "$ROOT/bench/reference.json")
  else log "no bench/reference.json: replies are measured, not compared (bench pin writes it on the pinned engine)"; fi
  log "bench.py $phases"
  python3 "$ROOT/bench/tools/bench.py" --url "$API" --model "$SERVED_NAME" --max-model-len "$MAX_MODEL_LEN" --phases "$phases" \
    --out "$BENCH_OUT/bench-$label.json" "${mode[@]}" "$@" > "$BENCH_OUT/bench-$label.out" 2> "$BENCH_OUT/bench-$label.err" || rc=$?
  { grep '^METRIC ' "$BENCH_OUT/bench-$label.out" || true; } >> "$BENCH_OUT/metrics.txt"
  { grep -E '^(PASS|FAIL) pin$' "$BENCH_OUT/bench-$label.out" || true; } | tee -a "$BENCH_OUT/gates.txt"
  tail -n 40 "$BENCH_OUT/bench-$label.err" >&2
  (( rc == 0 )) || BENCH_FAIL=1
  return 0
}

# The Mini reasons before it answers (even at effort none: ~100-500 reasoning tokens on a 260K prompt), so the tool's default
# 128 new tokens end the reply inside the reasoning (record 2026-11-23-needle): 1024 leaves room for the answer.
: "${NEEDLE_MAX_TOKENS:=1024}"
bench_needle() {
  local d rc=0
  for d in 0.1 0.5; do
    python3 "$ROOT/bench/tools/needle.py" --url "$API" --model "$SERVED_NAME" --tokens "$MAX_MODEL_LEN" --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$NEEDLE_MAX_TOKENS" --depth "$d" --out "$BENCH_OUT/needle-$d.json" || rc=1
  done
  return "$rc"
}

# the vision replies against bench/reference-vision.json; with no reference they are recorded (a gate named vision-record)
bench_vision_record() {
  local -a v=(); [[ $VISION == on ]] || v=(--no-images)
  python3 "$ROOT/bench/tools/gates.py" vision "$BENCH_OUT/vision.json" --url "$API" --model "$SERVED_NAME" --label "VISION=$VISION" "${v[@]}"
}
bench_vision_diff() {
  python3 "$ROOT/bench/tools/gates.py" diff "$ROOT/bench/reference-vision.json" "$BENCH_OUT/vision.json"
}
bench_vision() {
  if [[ -f $ROOT/bench/reference-vision.json ]]; then bench_gate vision-record bench_vision_record; bench_gate vision bench_vision_diff
  else log "no bench/reference-vision.json: the vision replies are recorded, not compared (bench pin writes it)"; bench_gate vision-record bench_vision_record; fi
}

# capacity: the engine's capacity line (an event=..._plan line of the server log) against the profile
bench_capacity() {
  local plan lanes kv want_lanes
  plan=$(remote 'grep -h -E "event=[A-Za-z0-9_]*_plan( |$)" "$BASE_DIR/logs/serve.log" | head -n 1 | cut -c1-600 || true' BASE_DIR="$BASE_DIR")
  [[ -n $plan ]] || { echo "no capacity line (event=..._plan) in the server log"; return 1; }
  echo "$plan" | tee "$BENCH_OUT/capacity.txt"
  field() { grep -o " $1=[^ ]*" <<<"$plan" | head -n 1 | cut -d= -f2; }
  lanes=$(field lanes_at_max_context); kv=$(field kv_tokens)
  local f; for f in lanes_at_max_context kv_tokens kv_gib free_after_gib prefix_gib; do [[ -z $(field $f) ]] || echo "METRIC $f=$(field $f)" >> "$BENCH_OUT/metrics.txt"; done
  [[ $lanes =~ ^[0-9]+$ && $kv =~ ^[0-9]+$ ]] || { echo "capacity line without lanes_at_max_context and kv_tokens"; return 1; }
  want_lanes=$(( KV_TOKENS / MAX_MODEL_LEN )); (( want_lanes <= SLOTS )) || want_lanes=$SLOTS
  (( lanes >= want_lanes && kv >= KV_TOKENS )) \
    || { echo "capacity: lanes_at_max_context=$lanes kv_tokens=$kv, want at least $want_lanes and $KV_TOKENS (profile $PROFILE)"; return 1; }
}

# memory: the guard's minimum MemAvailable since it armed against the engine floor (bench/tools/memcheck.py)
bench_memory() {
  local floor
  floor=$(awk -v g="${MEMORY_FLOOR_GIB:-8}" 'BEGIN {printf "%d", g * 1024}')
  cmd_memwatch report > "$BENCH_OUT/memwatch.txt" 2>&1 || { cat "$BENCH_OUT/memwatch.txt"; return 1; }
  cat "$BENCH_OUT/memwatch.txt"
  python3 "$ROOT/bench/tools/memcheck.py" --report "$BENCH_OUT/memwatch.txt" --floor-mib "$floor" | tee -a "$BENCH_OUT/metrics.txt"
}

# bench_pin [PHASES]: pin or extend bench/reference.json (and reference-vision.json) on the unmodified pinned engine
bench_pin() {
  local phases=${1:-warm,decode,conc,prefill} installed
  [[ -n $REXL3_COMMIT ]] || die "REXL3_COMMIT is not pinned (config/pins.sh, plan step 4.8): replies are pinned on the pinned engine only"
  installed=$(remote 'cat "$BASE_DIR/bin/rexl3-serve.commit"' BASE_DIR="$BASE_DIR") || die "no installed engine"
  [[ $installed == "$REXL3_COMMIT" ]] || die "the installed engine ${installed:0:12} is not REXL3_COMMIT ${REXL3_COMMIT:0:12}: ./inkling2mini.sh build"
  bench_run pin "$phases" --pin "$ROOT/bench/reference.json" --engine "$REXL3_COMMIT"
  bench_gate vision-record bench_vision_record
  if [[ -f $ROOT/bench/reference-vision.json ]]; then bench_gate vision bench_vision_diff
  elif [[ -f $BENCH_OUT/vision.json ]] && grep -q '^PASS vision-record$' "$BENCH_OUT/gates.txt"; then
    # the node address is not part of a reference (the tracked tree carries no address): meta.url is dropped, the replies are untouched
    python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); d["meta"].pop("url", None); json.dump(d, open(sys.argv[2], "w"), indent=1); open(sys.argv[2], "a").write("\n")' \
      "$BENCH_OUT/vision.json" "$ROOT/bench/reference-vision.json" && echo "pinned bench/reference-vision.json"
  fi
  echo "PINNED: bench/reference.json and bench/reference-vision.json written or extended (engine ${installed:0:12}); review the diff and commit them"
}

# the records of Phase 6.7-6.10: one gate each, results in $BENCH_OUT (the window and the pool come from the profile)
bench_sweep() {
  python3 "$ROOT/bench/rexl3/sweep_decode.py" --url "$API" --model "$SERVED_NAME" --window "$MAX_MODEL_LEN" --max-total "$KV_TOKENS" \
    --out "$BENCH_OUT/sweep-decode.json" || return 1
  python3 "$ROOT/bench/rexl3/sweep_prefill.py" --url "$API" --model "$SERVED_NAME" --window "$MAX_MODEL_LEN" --max-total "$KV_TOKENS" \
    --out "$BENCH_OUT/sweep-prefill.json" || return 1
  python3 "$ROOT/bench/rexl3/profile_report.py" "$BENCH_OUT/sweep-decode.json" "$BENCH_OUT/sweep-prefill.json" --md "$BENCH_OUT/profile.md"
}
bench_loop() {
  local m
  python3 "$ROOT/scripts/loop_probe.py" --url "$API" run "$BENCH_OUT/loop-text.json" || return 1
  for m in reasoning agentic; do python3 "$ROOT/scripts/loop_probe.py" --url "$API" "$m" "$BENCH_OUT/loop-$m.json" || return 1; done
  for m in text reasoning agentic; do python3 "$ROOT/scripts/loop_probe.py" report "$BENCH_OUT/loop-$m.json"; done
}
# profile: the memory bracket of the booted profile (plan 6.1, docs/FIT.md "Measured"): the window-length cold prompt, SLOTS of them at
# once, then a decode burst (decode graphs and overlap lanes allocate on first use). PROFILE_SAMPLE_S is the node-side sampler's period.
: "${PROFILE_SAMPLE_S:=5}"
bench_profile() {
  local rc=0 sampler=$BENCH_OUT/samples.txt ssh_pid
  cmd_memwatch begin > "$BENCH_OUT/memwatch-begin.txt" 2>&1 || { echo "could not open the memwatch window"; return 1; }
  bench_capacity || rc=1
  remote 'echo "ready gpu_mib=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}") avail_mib=$(awk "/^MemAvailable:/ {printf \"%d\", \$2/1024}" /proc/meminfo)"' > "$BENCH_OUT/ready.txt" || rc=1
  cat "$BENCH_OUT/ready.txt"
  # the sampler runs on the node for at most PROFILE_SAMPLE_MAX_S (4 h) and ends when its ssh is killed (its next write fails)
  local script
  script=$(cat <<'SAMPLER'
for ((i = 0; i < N; i++)); do
  echo "$(date -u +%H:%M:%S) gpu_mib=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}') avail_mib=$(awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo)"
  sleep "$P"
done
SAMPLER
)
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST" "P=$PROFILE_SAMPLE_S N=$(( ${PROFILE_SAMPLE_MAX_S:-14400} / PROFILE_SAMPLE_S )) bash -s" <<<"$script" > "$sampler" 2>&1 &
  ssh_pid=$!
  if [[ ${PROFILE_SOLO:-1} == 1 ]]; then
  log "profile: one cold prompt of the window ($MAX_MODEL_LEN tokens)"
  python3 "$ROOT/bench/tools/needle.py" --url "$API" --model "$SERVED_NAME" --tokens "$MAX_MODEL_LEN" --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$NEEDLE_MAX_TOKENS" --depth 0.5 --seed 11 --out "$BENCH_OUT/window-solo.json" > "$BENCH_OUT/window-solo.out" 2> "$BENCH_OUT/window-solo.err" || echo "note: the solo needle was not found or failed (window-solo.json)"
  fi
  if (( ${PROFILE_LANES:-$SLOTS} > 1 )); then
    log "profile: ${PROFILE_LANES:-$SLOTS} concurrent cold prompts of the window"
    python3 "$ROOT/bench/tools/needle.py" --url "$API" --model "$SERVED_NAME" --tokens "$MAX_MODEL_LEN" --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$NEEDLE_MAX_TOKENS" --lanes "${PROFILE_LANES:-$SLOTS}" --seed 12 --out "$BENCH_OUT/window-lanes.json" > "$BENCH_OUT/window-lanes.out" 2> "$BENCH_OUT/window-lanes.err" || echo "note: a lane did not find its needle or failed (window-lanes.json)"
  fi
  log "profile: a decode burst (1, 4 and 8 streams: the CUDA graphs and overlap lanes allocate on first use)"
  bench_run decode warm,decode,conc
  kill "$ssh_pid" 2>/dev/null || true; wait "$ssh_pid" 2>/dev/null || true
  awk '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="gpu_mib" && a[2]+0>g) g=a[2]+0; if(a[1]=="avail_mib" && (m==""||a[2]+0<m)) m=a[2]+0}} END {print "METRIC sampled_gpu_peak_mib=" g; print "METRIC sampled_avail_min_mib=" m}' "$sampler" | tee -a "$BENCH_OUT/metrics.txt"
  bench_memory || rc=1
  return "$rc"
}
bench_records() {
  local -a vis=() base=(); [[ $VISION != on ]] || vis=(--vision)
  [[ ! -f $ROOT/bench/records/quality.json ]] || base=(--baseline "$ROOT/bench/records/quality.json")
  case $1 in
    probe) python3 "$ROOT/bench/probe.py" --url "$API" --model "$SERVED_NAME" --window "$MAX_MODEL_LEN" "${vis[@]}" --out "$BENCH_OUT/probe.json" ;;
    api) python3 "$ROOT/tests/api_conformance.py" --url "$API" --model "$SERVED_NAME" --window "$MAX_MODEL_LEN" ;;
    quality) python3 "$ROOT/bench/quality_gate.py" --url "$API" --model "$SERVED_NAME" "${base[@]}" --out "$BENCH_OUT/quality.json" ;;
    tools) python3 "$ROOT/bench/toolgate.py" --url "$API" --model "$SERVED_NAME" --out "$BENCH_OUT/toolgate.json" ;;
    longctx) python3 "$ROOT/bench/longctx.py" --url "$API" --model "$SERVED_NAME" --window "$MAX_MODEL_LEN" --out "$BENCH_OUT/longctx.json" ;;
    sixcat) "$ROOT/bench/sixcat.sh" --url "$API/v1" --model "$SERVED_NAME" --out "$BENCH_OUT/sixcat.json" ;;
  esac
}

cmd_bench() {
  local what=${1:-} stamp
  case $what in
    decode|decode-4k|rolling|prefill|prefill-sweep|prefill-1m|needle|spec|cache|cache-images|exactness|vision|pin|check|all|probe|api|quality|tools|loop|sweep|longctx|sixcat|profile) ;;
    *) die "usage: bench decode|decode-4k|rolling|prefill|prefill-sweep|prefill-1m|needle|spec|cache|cache-images|exactness|vision|pin|check|all|probe|api|quality|tools|loop|sweep|longctx|sixcat|profile" ;;
  esac
  need_node
  shift
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  BENCH_OUT=$ROOT/bench/runs/$stamp-$what BENCH_FAIL=0
  mkdir -p "$BENCH_OUT"; : > "$BENCH_OUT/gates.txt"; : > "$BENCH_OUT/metrics.txt"
  log "bench $what against $API ($SERVED_NAME, window $MAX_MODEL_LEN); results in bench/runs/$stamp-$what"
  local -a spec=(); [[ $SPEC != none ]] || spec=(--serial-boot)
  case $what in
    decode) bench_run decode warm,decode,conc ;;
    decode-4k)
      # c = 1 on a cold 4,096-token prompt (256 new tokens), the drafter as booted: the like-for-like 4K figure of docs/ROOFLINE.md
      bench_gate decode-4k python3 "$ROOT/bench/rexl3/sweep_decode.py" --url "$API" --model "$SERVED_NAME" --window "$MAX_MODEL_LEN" \
        --max-total "$KV_TOKENS" --phase depth --ctxs 4096 --conc 1 --modes default --depth-modes default --depth-reps 3 \
        --out "$BENCH_OUT/decode-4k.json"
      { grep '^METRIC ' "$BENCH_OUT/decode-4k.log" || true; } >> "$BENCH_OUT/metrics.txt" ;;
    exactness)
      # the chunk-width and boot-to-boot evidence: the fixed prompt set solo (labelled with the booted PREFILL_CHUNK), then concurrent == solo
      bench_gate exact-record python3 "$ROOT/bench/tools/exactness.py" record "$BENCH_OUT/exact-chunk$PREFILL_CHUNK.json" --url "$API" --model "$SERVED_NAME" \
        --label "PREFILL_CHUNK=$PREFILL_CHUNK SPEC=$SPEC PREFIX_CACHE=$PREFIX_CACHE VISION=$VISION" --max-tokens 128
      bench_gate exact-concurrent python3 "$ROOT/bench/tools/exactness.py" concurrent --url "$API" --model "$SERVED_NAME" --max-tokens 128 ;;
    rolling) bench_run rolling warm,decode,rolling ;;   # decode first: every rolling reply is held to the solo reply of its kind
    prefill) bench_run prefill warm,prefill ;;
    prefill-1m) bench_run prefill-1m warm,prefill-1m ;;
    prefill-sweep)
      # cold prefill at 8K, 32K, 128K and 224K (those that fit MAX_MODEL_LEN), a unique prompt every pass, the first pass of each size discarded
      bench_gate prefill-sweep python3 "$ROOT/bench/tools/prefill_sweep.py" --url "$API" --model "$SERVED_NAME" --passes "${PREFILL_PASSES:-4}" \
        --sizes "${PREFILL_SIZES:-8192,32768,131072,229376}" --decode-after "${PREFILL_DECODE_AFTER:-128}" --out "$BENCH_OUT/prefill-sweep.json" \
        ${PREFILL_CACHE_DEFAULT:+--cache}
      { grep '^METRIC ' "$BENCH_OUT/prefill-sweep.log" || true; } >> "$BENCH_OUT/metrics.txt" ;;
    needle) bench_gate needle bench_needle ;;
    spec) bench_gate spec python3 "$ROOT/bench/tools/gates.py" spec "$BENCH_OUT/spec.json" --url "$API" --model "$SERVED_NAME" "${spec[@]}" ;;
    cache)
      cache_runs || die "bench cache needs a prefix cache: boot with PREFIX_CACHE=on and VISION=off (the cache is text only)"
      bench_gate cache python3 "$ROOT/bench/tools/gates.py" cache "$BENCH_OUT/cache.json" --url "$API" --model "$SERVED_NAME" ;;
    cache-images)
      cache_runs || die "bench cache-images needs a prefix cache: boot with PREFIX_CACHE=on"
      [[ $VISION == on ]] || die "bench cache-images needs the vision tower: boot with VISION=on"
      bench_gate cache-images python3 "$ROOT/bench/tools/cache_images.py" --out "$BENCH_OUT/cache-images.json" --url "$API" --model "$SERVED_NAME"
      { grep '^METRIC ' "$BENCH_OUT/cache-images.log" || true; } >> "$BENCH_OUT/metrics.txt" ;;
    vision) bench_vision ;;
    pin) bench_pin "$@" ;;
    check) bench_gate capacity bench_capacity; bench_gate memory bench_memory ;;
    probe|quality|tools|longctx|sixcat) bench_gate "$what" bench_records "$what" ;;
    api) bench_gate api-test bench_records api ;;
    loop) bench_gate loop-probe bench_loop ;;
    sweep) bench_gate sweep bench_sweep ;;
    profile) bench_gate profile bench_profile ;;
    all)
      cmd_memwatch begin > "$BENCH_OUT/memwatch-begin.txt" 2>&1 || die "could not open the memwatch window"
      bench_gate capacity bench_capacity
      bench_gate smoke cmd_smoke
      bench_run decode warm,decode,conc,rolling
      bench_run prefill prefill
      bench_vision
      bench_gate spec python3 "$ROOT/bench/tools/gates.py" spec "$BENCH_OUT/spec.json" --url "$API" --model "$SERVED_NAME" "${spec[@]}"
      if cache_runs; then
        bench_gate cache python3 "$ROOT/bench/tools/gates.py" cache "$BENCH_OUT/cache.json" --url "$API" --model "$SERVED_NAME"
      else log "no prefix cache (PREFIX_CACHE=$PREFIX_CACHE, VISION=$VISION): the cache gate is skipped"; fi
      bench_gate needle bench_needle
      bench_gate memory bench_memory ;;
  esac
  cat "$BENCH_OUT/metrics.txt"
  local failed verdict=pass; failed=$(grep '^FAIL ' "$BENCH_OUT/gates.txt" | cut -d' ' -f2 | sort -u | paste -sd, - || true)
  [[ -z $failed ]] || verdict="FAIL ($failed)"
  echo "bench $what: $verdict run=bench/runs/$stamp-$what"
  return "$BENCH_FAIL"
}

# memwatch begin: a new window token (the guard restarts its window minimum when it sees it);
# memwatch report: the guard's record. The guard runs only while the server runs; the record stays after.
cmd_memwatch() {
  case ${1:-} in
    begin)
      local token; token="$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM"
      remote 'mkdir -p "$BASE_DIR/run" && echo "$TOKEN" > "$BASE_DIR/run/memwatch.window"' BASE_DIR="$BASE_DIR" TOKEN="$token" \
        || die "could not open the window"
      echo "memwatch window=$token (memwatch report reads the minimum since now)" ;;
    report)
      remote '
        f=$BASE_DIR/logs/memguard.min
        [[ -f $f ]] || { echo "no guard record ($f): the server has not run under the guard"; exit 1; }
        set +u; declare -A g; while IFS== read -r k v; do g[$k]=$v; done < "$f"
        cur=$(cat "$BASE_DIR/run/memwatch.window" 2>/dev/null || true)
        w=${g[window_min_mib]} wat=${g[window_min_at]}
        if [[ -z $cur || ${g[window]} != "$cur" ]]; then w=n/a wat=n/a; fi
        gpid=$(cat "$BASE_DIR/run/memguard.pid" 2>/dev/null || true)
        if [[ -n $gpid ]] && kill -0 "$gpid" 2>/dev/null; then gstate=running; else gstate=stopped; fi
        echo "memwatch guard=$gstate window=${cur:-none} window_min_mib=$w window_min_at=$wat loading_min_mib=${g[loading_min_mib]} ready_min_mib=${g[ready_min_mib]} min_mib=${g[min_mib]} min_at=${g[min_at]} now_mib=${g[now_mib]} updated=${g[updated]}"' \
        BASE_DIR="$BASE_DIR" ;;
    *) die "usage: memwatch begin|report" ;;
  esac
}

cmd_record() {
  local name=${1:-} day dir vers serve
  [[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "usage: record NAME (letters, digits, . _ -)"
  day=$(date -u +%Y-%m-%d) dir=$ROOT/bench/records/$day-$name
  [[ ! -e $dir ]] || die "$dir exists"
  vers=$(remote '[[ -x $BIN ]] || { echo "no engine at $BIN"; exit 1; }
    "$BIN" --version 2>/dev/null; echo "sha256 $(sha256sum "$BIN" | cut -d" " -f1)"
    echo "gpu_clock $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null || echo unknown)"' BIN="$BIN") \
    || { echo "$vers"; die "could not read the installed engine"; }
  serve=$(serve_cmd)
  REC_DIR="$dir" REC_NAME="$name" REC_DAY="$day" REC_VERSIONS="$vers" REC_SERVE="$serve" REC_SSH_USER="$SSH_USER" \
  REC_GIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo none)" \
  REC_DIRTY="$(git -C "$ROOT" status --porcelain 2>/dev/null | sed 's/^...//' || true)" \
  REC_MODEL_REPO="$MODEL_REPO" REC_MODEL_REV="$MODEL_REV" REC_MODEL_DIR="$MODEL_DIR" REC_PINNED="$REXL3_COMMIT" \
  REC_UPSTREAM_BASE="$REXL3_UPSTREAM_BASE" \
  REC_CONFIG="$(for k in PROFILE PROFILE_MEASURED SERVED_NAME PORT MAX_MODEL_LEN SLOTS KV_TOKENS PREFILL_CHUNK SPEC PREFIX_CACHE VISION \
    MEMORY_FLOOR_GIB EXTRA_FLAGS EXTRA_ENV GPU_CLOCKS \
    GUARD_START_MIB GUARD_MIB GUARD_SWAP_MAX_MIB GUARD_SWAP_AVAIL_MIB DROP_CACHES_EVERY_S; do printf '%s=%s\n' "$k" "${!k:-}"; done)" \
  python3 - <<'PY' || die "could not write the record"
import json, os, re
e = os.environ
v = {}
for line in e["REC_VERSIONS"].splitlines():
    m = re.match(r"(\S+)\s+(.*)$", line)
    if m:
        v[m.group(1)] = m.group(2).strip()
for key in ("commit", "sha256"):
    if not v.get(key):
        raise SystemExit(f"record: the engine reported no {key}")
meta = {
    "name": e["REC_NAME"], "date": e["REC_DAY"],
    "recipe": {"commit": e["REC_GIT"], "dirty_files": [l for l in e["REC_DIRTY"].splitlines() if l]},
    "pack": {"repo": e["REC_MODEL_REPO"], "revision": e["REC_MODEL_REV"], "dir": e["REC_MODEL_DIR"]},
    "engine": {"commit": v["commit"], "sha256": v["sha256"], "pinned_commit": e["REC_PINNED"],
               "lab_build": v["commit"] != e["REC_PINNED"], "upstream_base": e["REC_UPSTREAM_BASE"],
               "version": {k: x for k, x in v.items() if k not in ("sha256", "gpu_clock")}},
    "gpu_clock_sm_max": v.get("gpu_clock", "unknown"),
    "config": dict(l.split("=", 1) for l in e["REC_CONFIG"].splitlines()),
    "serve": {"ssh_user": e["REC_SSH_USER"], "command": e["REC_SERVE"]},
    "results": [],
}
os.makedirs(e["REC_DIR"])
with open(os.path.join(e["REC_DIR"], "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
    f.write("\n")
print(os.path.join(e["REC_DIR"], "meta.json"))
PY
}

# gpu-clocks lock|status: `lock` applies the pinned GPU SM clock cap and requires its success line (ensure_gpu_cap); `status` shows the
# current and maximum SM clock. There is no reset and no other value: the cap is hardware protection (owner, 2026-11-23).
cmd_gpu_clocks() {
  case ${1:-} in
    lock) ensure_gpu_cap ;;
    status) remote 'nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader' ;;
    *) die "usage: gpu-clocks lock|status. The GPU SM clock cap ($GPU_CLOCK_CAP_MHZ MHz, config/pins.sh) is hardware protection: it is never released, raised or changed, so there is no reset and no MHz argument" ;;
  esac
}

# engine-test host|goldens: the engine's own tests on the node, from the source `build` installed ($BASE_DIR/src, the commit in
# bin/rexl3-serve.commit), with every rexl3-serve stopped. Each suite must run at least one test; a golden that skips for want
# of a device or the pack ("not run") fails. The two gitignored nemotron fixtures that gb-model's test build includes are
# created empty if absent and removed again (the Inkling tests do not read them).
cmd_engine_test() {
  local which=${1:-}
  [[ $which == host || $which == goldens ]] || die "usage: engine-test host|goldens"
  ensure_gpu_cap                                    # the engine tests build and run on the node (the goldens on the GPU): the cap first
  remote '
    pgrep -f "[r]exl3-serve" >/dev/null && { echo "FAIL a rexl3-serve is running on this node: ./inkling2mini.sh stop" >&2; exit 1; }
    sha=$(cat "$BASE_DIR/bin/rexl3-serve.commit" 2>/dev/null) || { echo "FAIL no engine installed (./inkling2mini.sh build)" >&2; exit 1; }
    [[ -d $BASE_DIR/src ]] || { echo "FAIL no engine source at $BASE_DIR/src (./inkling2mini.sh build)" >&2; exit 1; }
    cd "$BASE_DIR/src"
    export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda REXL3_CUDA_ARCH=sm_121 \
      REXL3_COMMIT="$sha" CARGO_TARGET_DIR="$BASE_DIR/target"
    made=()
    cleanup() { rm -f "${made[@]}"; }
    trap cleanup EXIT
    for f in layer1_fixture layer1_fixture_repaired; do
      p=testdata/nemotron_omni/moe/$f.safetensors
      [[ -e $p ]] || { mkdir -p "${p%/*}"; : > "$p"; made+=("$p"); }
    done
    # suite: runs `cargo test` with the array args under the name in label; ok only when it ran at least one test and nothing
    # failed or skipped
    suite() {
      out=$(mktemp)
      if cargo +"$TOOLCHAIN" test --release --locked "${args[@]}" >"$out" 2>&1 \
        && grep -Eq "^test result: ok\. [1-9][0-9]* passed" "$out" && ! grep -qi "not run" "$out"; then
        echo "ok   $label: $(grep -E "^test result: ok" "$out" | head -n 1)"
        rm -f "$out"
      else
        echo "FAIL $label" >&2; tail -n 40 "$out" >&2; rm -f "$out"; exit 1
      fi
    }
    echo "engine $sha"
    if [[ $WHICH == host ]]; then
      label="gb-model families::inkling2 + inkling2_loop"; args=(-p gb-model --lib -- families::inkling2:: families::inkling2_loop::); suite
      label="gb-cuda inkling2"; args=(-p gb-cuda --lib inkling2); suite
      label="gb-server formats::inkling2"; args=(-p gb-server --lib formats::inkling2); suite
      label="gb-server inkling2_format"; args=(-p gb-server --test inkling2_format); suite
    else
      (
        exec 9>>/tmp/rexl3-gpu.lock
        flock -x -n 9 || { echo "FAIL /tmp/rexl3-gpu.lock is held by another GPU job" >&2; exit 1; }
        label="golden attn"; args=(-p gb-cuda --test inkling2_attn golden_inkling2_attn_hashes -- --test-threads=1 --nocapture); suite
        label="golden ops"; args=(-p gb-cuda --test inkling2_ops golden_inkling2_ops_hashes -- --test-threads=1 --nocapture); suite
        label="golden moe"; args=(-p gb-cuda --test inkling2_moe golden_inkling2_moe_hashes -- --test-threads=1 --nocapture); suite
        label="golden vision"; args=(-p gb-model --test inkling2_vision golden_inkling2_vision_hashes -- --test-threads=1 --nocapture); suite
        # the drafter golden of the Small reads the pack of the Small (its mtp.safetensors is bf16, the one of the Mini is exl3), read-only; absent: it fails, not skips
        [[ -f $SMALL_DIR/mtp.safetensors ]] || { echo "FAIL golden dspark needs the Small pack for its drafter weights: $SMALL_DIR/mtp.safetensors is missing" >&2; exit 1; }
        label="golden dspark"; args=(-p gb-cuda --test inkling2_dspark golden_inkling2_dspark_hashes -- --test-threads=1 --nocapture)
        REXL3_REQUIRE_GPU=1 REXL3_INKLING2_DIR="$SMALL_DIR" suite
      )
    fi
    echo "engine-test $WHICH: pass"' \
    BASE_DIR="$BASE_DIR" WHICH="$which" TOOLCHAIN="$RUST_TOOLCHAIN" MODEL_DIR="$MODEL_DIR" SMALL_DIR="$SMALL_MODEL_DIR"
}

# stop-other and restore-other run only when someone types them: no other command, no test and no doctor calls them.
# The other stack's controller reads its own settings from the environment first: BASE_DIR, PORT, NODES and the rest of this
# recipe's exported .env would make it look for its processes under the Mini's directories and stop nothing. It runs with
# a scrubbed environment (HOME, PATH, the ssh agent, locale and terminal only).
other_env() { OTHER_ENV=(env -i "HOME=$HOME" "PATH=$PATH" "USER=${USER:-}" "TERM=${TERM:-dumb}" "LANG=${LANG:-C}"); [[ -z ${SSH_AUTH_SOCK:-} ]] || OTHER_ENV+=("SSH_AUTH_SOCK=$SSH_AUTH_SOCK"); }
cmd_stop_other() { [[ -n $STOP_OTHER ]] || die "STOP_OTHER is not set (.env.example)"; log "STOP_OTHER"; other_env; "${OTHER_ENV[@]}" bash -c "$STOP_OTHER"; }
cmd_restore_other() {
  [[ -n $RESTORE_OTHER ]] || die "RESTORE_OTHER is not set (.env.example)"
  cmd_stop
  log "RESTORE_OTHER"; other_env; "${OTHER_ENV[@]}" bash -c "$RESTORE_OTHER"
}

cmd_help() { sed -n '3,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

# The dispatch is a function and its call is the last line of this file: bash reads a script as it runs, so a controller that
# is edited while it runs must not find new text after the call.
main() {
  case ${1:-help} in
    doctor) cmd_doctor ;;
    setup) cmd_setup ;;
    fetch) cmd_fetch ;;
    verify) cmd_verify ;;
    build) shift; cmd_build "$@" ;;
    print-serve) cmd_print_serve ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    logs) shift; cmd_logs "$@" ;;
    smoke) cmd_smoke ;;
    bench) shift; cmd_bench "$@" ;;
    memwatch) shift; cmd_memwatch "$@" ;;
    record) shift; cmd_record "$@" ;;
    gpu-clocks) shift; cmd_gpu_clocks "$@" ;;
    engine-test) shift; cmd_engine_test "$@" ;;
    stop-other) cmd_stop_other ;;
    restore-other) cmd_restore_other ;;
    help|-h|--help) cmd_help ;;
    *) cmd_help >&2; return 2 ;;
  esac
}
main "$@"; exit $?
