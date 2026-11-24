#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# autoresearch.sh: the bring-up ladder of Inkling-2-Mini-1x-DGX-Spark. Segment 8: control machine plus recorded node gates.
#
# What it measures
#   How many rungs of the bring-up ladder stand. A rung is one real command whose own output decides it (a test count, a
#   checker's exit status, a record's fields); none is decided by the existence of a file alone. The harness never touches a
#   Spark. Segment 1 (rungs 1-15) covers what runs on the Mac. Segment 2 adds the Mac rungs of wave 2 (chat template,
#   bench tools) and the node gates, which are decided by their records: `scripts/check_record.py` takes the newest
#   bench/records/*/meta.json of the gate, which must say pass, sit on a committed engine tree that is an ancestor of the
#   engine head (so a lane counts once merged), and carry its test counts and metrics. Segment 3 adds the assembly and the
#   optional features: forward-oracle, boot, smoke, kernels-draft, kernels-vision. Segment 4 adds the first Phase 6 gates:
#   vision-smoke, api-test, exact-basic, needle, greedy-oracle, speed-baseline. Segment 5 adds the serving features: cache,
#   cache-images, exact-graph, speed-graph, oracle-cacheddiff. Segment 6 adds the oracle's whole-forward gap, the memory plan,
#   the expert arena, the prefill speed, and the block drafter (drafted-exact, drafted-images, drafted-cache, speed-drafted,
#   markov-variants). Segment 7 adds the integrity of the evidence rule itself (record-checker) and the owner's two stretch
#   targets, each only standing beside an exactness proof of the same engine commit: cold prefill at 32K over 3,000 tok/s
#   (speed-prefill-3k) and the mean decode speed over bench.py's four kinds over 100 tok/s (speed-decode-100), byte-exact.
#   Segment 8 adds the owner's third stretch target, c=4 aggregate decode over 200 tok/s (speed-c4-200, on the sustained
#   `decode_c4_rolling_tok_s` that `bench rolling` measures: bench.py's span number has a tail artifact) and its measured
#   baseline (c4-baseline). The remaining node rungs (pin, long context, quality baselines) and the speed score `target_score`
#   open the next segments (plan Phases 6 and 7).
#
# Usage    ./autoresearch.sh                        the whole ladder
#          RECIPE_ROOT=<worktree> ENGINE_SRC=<engine worktree> ./autoresearch.sh    a lane's tree, before it is merged
#          ORACLE_PY=<python>                       the oracle's interpreter (default oracle/.venv/bin/python)
# Output   `PASS <id>` or `FAIL <id>` per rung, an `INFO <id>: <detail>` line each, then METRIC lines:
#            bringup_gates=<n>        rungs standing: the primary metric, higher is better
#            bringup_gates_total, rung_<id>=0|1, recipe_tests, oracle_tests, engine_small_tests, engine_loop_tests, harness_s
# Exit     0 when the ladder ran, whatever stands (a failing rung is the result, not an error); 2 when the harness itself
#          cannot run (not a recipe root, a tool missing). Per-rung logs: bench/runs/ladder-<pid>/ (git-ignored).
#
# Rungs (id: what decides it)
#   pins                 config/pins.sh parses; the manifest sha256 matches its pin; the upstream base is an ancestor of the
#                        engine head; RUST_TOOLCHAIN equals the engine's rust-toolchain.toml; REXL3_COMMIT, when set, too
#   intake               13 manifest rows summing to 15,070,275,721 B; census and intake parse; roofline resident 14.000 GiB
#   recipe-tests         tests/test_recipe.sh: exit 0 and a last line `test_recipe: N passed, 0 failed`, N >= 15
#   probe-record         bench/records/2026-11-22-node-probe/meta.json parses, lists 4 nodes, holds no lab address
#   spec                 scripts/check_spec.py passes (headings, pinned sources, every UNKNOWN resolved or open)
#   oracle-unit          the oracle's pytest suite: >= 20 passed, nothing failed, errored or skipped
#   oracle-exl3-real     oracle/records/exl3-real-verify.json: every entry bit-exact; K=4, 5 and 6 each covered
#   engine-small-intact  the engine head contains the Small family (inkling2-a) and its 31 host tests pass
#   loop-suite           families::inkling2_loop:: host tests: >= 12 pass, none fails
#   loop-config loop-schedule loop-taps loop-dims loop-bind loop-memory
#                        a passing families::inkling2_loop:: test whose name contains the tag, and none failing
#   chat-template        gb-server `inkling2_loop_chat` tests: >= 10 pass, names starting render_ and parse_ both present
#   bench-tools          tests/test_bench.py and tests/test_gates.py: each ends `<name>: N passed, 0 failed`, N >= 15
#   record-checker       tests/test_records.py: ends `test_records: N passed, 0 failed`, N >= 5 (the rule every node rung relies on)
#   roofline             record, gate roofline: measured read_gbs_mean3 and bf16_tflops_8192_mean3 (no engine)
#   engine-build         record, gate engine-build: the integration head built and host-tested on a node, >= 60 tests
#   oracle-fixtures      record, gate oracle-fixtures: fixtures of the real pack, fixtures >= 4 (no engine)
#   kernels-moe          record, gate kernels-moe: >= 6 GPU tests pass, on an engine commit in the head
#   kernels-attn         record, gate kernels-attn: >= 8 GPU tests pass, on an engine commit in the head
#   kernels-dense        record, gate kernels-dense: >= 10 GPU tests pass, on an engine commit in the head
#   forward-oracle       record, gate forward-oracle: the engine's 48-pass forward teacher-forced against the oracle, >= 4 tests,
#                        passes_compared >= 48
#   boot                 record, gate boot: the Mini server booted on a node from a committed engine tree, with boot_s
#   smoke                record, gate smoke: chat, completion, tool call and reasoning answered, >= 4 checks
#   kernels-draft        record, gate kernels-draft: >= 6 GPU tests pass (the block drafter's kernels)
#   kernels-vision       record, gate kernels-vision: >= 6 GPU tests pass (the vision tower)
#   vision-smoke         record, gate vision-smoke: images served through the engine, >= 8 checks
#   api-test             record, gate api-test: probe and the 17 api_conformance checks pass, >= 15
#   exact-basic          record, gate exact-basic: concurrent == solo, chunk width, boot to boot, >= 4 checks
#   needle               record, gate needle: the code found at 10% and 50% depth at the profile's window, >= 2
#   greedy-oracle        record, gate greedy-oracle: the engine's greedy tokens equal the oracle's on a real-pack fixture, 16 tokens
#   speed-baseline       record, gate speed-baseline: raw decode and prefill numbers of the unmodified engine exist (not a score)
#   cache                record, gate cache: cached == uncached replies, 0 cached tokens on uncached and cold, shared prefix served from the cache
#   cache-images         record, gate cache-images: the same with images in the prompt (cuts never inside an image), graphs on
#   exact-graph          record, gate exact-graph: graph replay and every later speed change leave the output bytes of the start engine
#   speed-graph          record, gate speed-graph: serial decode with CUDA graphs, c=1 at least 45 tok/s (the first boot measured 37.2)
#   oracle-cacheddiff    record, gate oracle-cacheddiff: the oracle's cached decode equals its full recompute within the fp32 ulp bound
#   oracle-gap           record, gate oracle-gap: the engine against the real-pack fixtures (planes, argmax, 64 greedy tokens), >= 8 checks
#   row-probe            record, gate row-probe: per-row teacher forcing and the unembed alone explain the 3 argmax differences, >= 4
#   memory-plan          record, gate memory-plan: the plan predicts MemAvailable at ready within 1 GiB at bootstrap, long, max, >= 4
#   expert-arena         record, gate expert-arena: experts carved from an arena, no byte moved, about 6 GiB returned, >= 4
#   speed-prefill        record, gate speed-prefill: cold prefill at 32K, at least 2,120 tok/s (the proposed target)
#   exact-prefill        record, gate exact-prefill: the prefill change leaves every output byte, >= 8 checks
#   drafted-exact        record, gate spec: drafted == serial at greedy (gates.py spec, 65 edge cases, taps equal the oracle's), >= 8
#   drafted-images       record, gate spec-images: the same with images and the prefix cache on, >= 8
#   drafted-cache        record, gate spec-cache: the cache carries the drafter's state, restored == uncached, >= 4
#   speed-drafted        record, gate spec-final: the shipped drafter default on the merged engine, code at least 1.5x serial
#   markov-variants      record, gate markov-variants: UNKNOWN-17 decided by measurement (fold kept), >= 4
#   speed-prefill-3k     record, gate speed-prefill: prefill_32k_tok_s >= 3000 (cold, 32K, shipping defaults) AND the newest
#                        exact-prefill record passes on the same engine commit (byte-exact A lane)
#   speed-decode-100     record, gate speed-decode: decode_mean4_tok_s >= 100 (bench.py decode, mean of its four kinds, c=1,
#                        shipping defaults with the drafter) AND the newest spec record (drafted == serial) passes on the same
#                        engine commit
#   c4-baseline          record, gate speed-c4-baseline: the sustained c=4 aggregate (drafted and serial) measured on the merged engine,
#                        with its metric decode_c4_rolling_tok_s (131.5 drafted at the first measurement), >= 20 checks
#   speed-c4-200         record, gate speed-c4: decode_c4_rolling_tok_s >= 200 (4 workers closed loop, shipping defaults) AND the newest
#                        spec record passes on the same engine commit (byte-exact A lane); C4Plan: not reachable on the bench's mix
#                        byte-exactly (byte roofline 196 at today's acceptance), the open rung is the owner's target
# Engine host tests build with REXL3_NO_CUDA=1 (the stub build: kernels skip) and need two empty Nemotron fixture
# placeholders, which this script creates (git-excluded; docs/CLEANUP.md).

set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE=${RECIPE_ROOT:-$ROOT}
ENGINE=${ENGINE_SRC:-$HOME/projects/rexl3-inkling2-loop}
ORACLE_PY=${ORACLE_PY:-$RECIPE/oracle/.venv/bin/python}
SMALL_FAMILY_BASE=a73ff7a3fe1c25902b9d6c7d9d0870b9a75bc915   # inkling2-a: the Small family's last byte-exact commit
MANIFEST_ROWS=13
MANIFEST_BYTES=15070275721
MIN_RECIPE_TESTS=15
MIN_ORACLE_TESTS=20
MIN_SMALL_TESTS=31
MIN_LOOP_TESTS=12
MIN_CHAT_TESTS=10
MIN_BENCH_TESTS=15
LOOP_TAGS="config schedule taps dims bind memory"
PROBE_RECORD=bench/records/2026-11-22-node-probe/meta.json
OUT=
passed=0
total=0
failed=""

info() { printf 'INFO %s: %s\n' "$1" "$2"; }
count() { printf '%s %s\n' "$1" "$2" >> "$OUT/counts.txt"; }

# record ID RC DETAIL
record() {
  total=$((total + 1))
  if [ "$2" -eq 0 ]; then passed=$((passed + 1)); printf 'PASS %s\n' "$1"; printf '%s 1\n' "$1" >> "$OUT/rungs.txt"
  else failed="$failed $1"; printf 'FAIL %s\n' "$1"; printf '%s 0\n' "$1" >> "$OUT/rungs.txt"; fi
  info "$1" "$3"
}

# run_rung ID FUNCTION [ARG]: the function prints a one-line detail and returns 0 (stands) or 1 (does not)
run_rung() {
  local id=$1 fn=$2 detail rc
  detail=$("$fn" "${3:-}" 2> "$OUT/$id.err") && rc=0 || rc=$?
  if [ -z "$detail" ] && [ -s "$OUT/$id.err" ]; then detail=$(head -n 1 "$OUT/$id.err" | cut -c1-200); fi
  record "$id" "$rc" "$detail"
}

engine_ok() { git -C "$ENGINE" rev-parse --git-dir > /dev/null 2>&1; }

ensure_fixtures() {
  local d=$ENGINE/testdata/nemotron_omni/moe f
  mkdir -p "$d"
  for f in layer1_fixture.safetensors layer1_fixture_repaired.safetensors; do [ -e "$d/$f" ] || : > "$d/$f"; done
}

# cargo_tests FILTER: gb-model host tests under the stub build, offline, one filter; output on stdout
cargo_tests() {
  ( cd "$ENGINE" && REXL3_NO_CUDA=1 CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}" \
      cargo test -p gb-model --lib --locked "$1" -- --test-threads=4 2>&1 )
}

# the `N` of the last `test result: ok. N passed; 0 failed` line of cargo test output on stdin; empty when there is none
passed_count() { sed -n 's/^test result: ok\. \([0-9][0-9]*\) passed; 0 failed.*/\1/p' | tail -n 1; }

first_problem() {
  printf '%s\n' "$1" | grep -m1 -E '^error|FAILED|panicked|could not compile|command not found|no such' | cut -c1-160
}

r_pins() (
  cd "$RECIPE" || exit 1
  # shellcheck disable=SC1091
  . config/pins.sh
  [[ ${MODEL_REV:-} =~ ^[0-9a-f]{40}$ ]] || { echo "MODEL_REV is not 40-hex"; exit 1; }
  [ -n "${MODEL_REPO:-}" ] || { echo "MODEL_REPO is empty"; exit 1; }
  got=$(shasum -a 256 "$MANIFEST_FILE" | cut -d' ' -f1)
  [ "$got" = "${MANIFEST_SHA256:-}" ] || { echo "manifest sha256 $got differs from MANIFEST_SHA256"; exit 1; }
  [[ ${REXL3_UPSTREAM_BASE:-} =~ ^[0-9a-f]{40}$ ]] || { echo "REXL3_UPSTREAM_BASE is not 40-hex"; exit 1; }
  engine_ok || { echo "ENGINE_SRC $ENGINE is not a git worktree"; exit 1; }
  git -C "$ENGINE" merge-base --is-ancestor "$REXL3_UPSTREAM_BASE" HEAD || { echo "REXL3_UPSTREAM_BASE is not an ancestor of the engine head"; exit 1; }
  want=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$ENGINE/rust-toolchain.toml")
  [ "${RUST_TOOLCHAIN:-}" = "$want" ] || { echo "RUST_TOOLCHAIN ${RUST_TOOLCHAIN:-} differs from the engine's $want"; exit 1; }
  if [ -n "${REXL3_COMMIT:-}" ]; then
    [[ $REXL3_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "REXL3_COMMIT is not 40-hex"; exit 1; }
    git -C "$ENGINE" merge-base --is-ancestor "$REXL3_COMMIT" HEAD || { echo "REXL3_COMMIT is not an ancestor of the engine head"; exit 1; }
  fi
  echo "model pinned, manifest sha256 matches, engine base is an ancestor, toolchain $RUST_TOOLCHAIN, engine pin ${REXL3_COMMIT:-unset}"
)

r_intake() (
  cd "$RECIPE" || exit 1
  n=$(grep -c . intake/manifest.tsv)
  sum=$(awk -F'\t' '{s += $2} END {printf "%d", s}' intake/manifest.tsv)
  [ "$n" = "$MANIFEST_ROWS" ] && [ "$sum" = "$MANIFEST_BYTES" ] || { echo "manifest rows=$n bytes=$sum (want $MANIFEST_ROWS, $MANIFEST_BYTES)"; exit 1; }
  python3 -c 'import json; json.load(open("intake/census.json")); json.load(open("intake/intake.json"))' || { echo "census or intake JSON does not parse"; exit 1; }
  out=$(python3 scripts/roofline.py) || { echo "scripts/roofline.py failed"; exit 1; }
  printf '%s\n' "$out" | grep -qF '| **resident total** | **14.000** |' || { echo "roofline resident total is not 14.000 GiB"; exit 1; }
  echo "$n files, $sum B, census parses, roofline resident total 14.000 GiB"
)

r_recipe_tests() (
  cd "$RECIPE" || exit 1
  [ -f tests/test_recipe.sh ] || { echo "tests/test_recipe.sh does not exist"; exit 1; }
  out=$(bash tests/test_recipe.sh 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" > "$OUT/recipe-tests.out"
  last=$(printf '%s\n' "$out" | tail -n 1)
  [ "$rc" -eq 0 ] || { echo "tests/test_recipe.sh exited $rc: $last"; exit 1; }
  [[ $last =~ ^test_recipe:\ ([0-9]+)\ passed,\ 0\ failed$ ]] || { echo "last line is not 'test_recipe: N passed, 0 failed': $last"; exit 1; }
  n=${BASH_REMATCH[1]}
  count recipe_tests "$n"
  [ "$n" -ge "$MIN_RECIPE_TESTS" ] || { echo "only $n static tests (need $MIN_RECIPE_TESTS)"; exit 1; }
  echo "$n static tests pass"
)

r_probe_record() (
  cd "$RECIPE" || exit 1
  [ -f "$PROBE_RECORD" ] || { echo "$PROBE_RECORD does not exist"; exit 1; }
  python3 - "$PROBE_RECORD" <<'EOF'
import json, re, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"{p} does not parse: {e}"); sys.exit(1)
nodes = d.get("nodes") if isinstance(d, dict) else None
n = len(nodes) if isinstance(nodes, (list, dict)) else 0
if n != 4:
    print(f"nodes: {n} (need 4)"); sys.exit(1)
if re.search(r"192\.168\.\d+\.\d+|10\.100\.\d+\.\d+", open(p).read()):
    print("a lab address is in the record"); sys.exit(1)
print("4 nodes, no lab address")
EOF
)

r_spec() (
  [ -f "$RECIPE/scripts/check_spec.py" ] || { echo "scripts/check_spec.py does not exist"; exit 1; }
  out=$(python3 "$RECIPE/scripts/check_spec.py" --root "$RECIPE" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | head -n 1
  exit "$rc"
)

r_oracle_unit() (
  [ -x "$ORACLE_PY" ] || { echo "$ORACLE_PY does not exist (oracle/README.md: Setup)"; exit 1; }
  [ -d "$RECIPE/oracle/tests" ] || { echo "oracle/tests does not exist"; exit 1; }
  cd "$RECIPE" || exit 1
  out=$("$ORACLE_PY" -m pytest -q oracle/tests -p no:cacheprovider 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" > "$OUT/oracle-unit.out"
  last=$(printf '%s\n' "$out" | tail -n 1)
  [[ $last =~ ^([0-9]+)\ passed(,\ [0-9]+\ warnings?)?\ in\  ]] || { echo "pytest summary: $last"; exit 1; }
  n=${BASH_REMATCH[1]}
  count oracle_tests "$n"
  [ "$rc" -eq 0 ] || { echo "pytest exited $rc: $last"; exit 1; }
  [ "$n" -ge "$MIN_ORACLE_TESTS" ] || { echo "only $n oracle tests (need $MIN_ORACLE_TESTS)"; exit 1; }
  echo "$n oracle tests pass"
)

r_oracle_exl3_real() (
  p=$RECIPE/oracle/records/exl3-real-verify.json
  [ -f "$p" ] || { echo "oracle/records/exl3-real-verify.json does not exist"; exit 1; }
  python3 - "$p" <<'EOF'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"does not parse: {e}"); sys.exit(1)
entries = []
def walk(x):
    if isinstance(x, dict):
        if any(k in x for k in ("k", "K", "bits")) and any(k in x for k in ("match", "bit_exact", "matches")):
            entries.append(x)
        for v in x.values():
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
walk(d)
if not entries:
    print("no entry has both a K and a match field"); sys.exit(1)
bad = [e for e in entries if e.get("match", e.get("bit_exact", e.get("matches"))) is not True]
if bad:
    print(f"{len(bad)} of {len(entries)} entries are not bit-exact"); sys.exit(1)
ks = {}
for e in entries:
    k = int(e.get("k", e.get("K", e.get("bits"))))
    ks[k] = ks.get(k, 0) + 1
short = {k: [ks.get(k, 0), n] for k, n in {4: 3, 5: 2, 6: 2}.items() if ks.get(k, 0) < n}
if short:
    print(f"too few tensors per K, [have, need]: {short}"); sys.exit(1)
print(f"{len(entries)} tensors bit-exact; per K {dict(sorted(ks.items()))}")
EOF
)

r_engine_small_intact() (
  engine_ok || { echo "ENGINE_SRC $ENGINE is not a git worktree"; exit 1; }
  git -C "$ENGINE" merge-base --is-ancestor "$SMALL_FAMILY_BASE" HEAD || { echo "the engine head does not contain inkling2-a ($SMALL_FAMILY_BASE)"; exit 1; }
  ensure_fixtures
  out=$(cargo_tests 'families::inkling2::')
  printf '%s\n' "$out" > "$OUT/engine-small.out"
  n=$(printf '%s\n' "$out" | passed_count)
  [ -n "$n" ] || { echo "Small-family host tests do not build or fail: $(first_problem "$out")"; exit 1; }
  count engine_small_tests "$n"
  [ "$n" -ge "$MIN_SMALL_TESTS" ] || { echo "only $n Small-family tests pass (need $MIN_SMALL_TESTS)"; exit 1; }
  echo "the head contains inkling2-a; $n Small-family host tests pass"
)

r_loop_suite() (
  out=$(cat "$OUT/loop.out" 2> /dev/null)
  n=$(printf '%s\n' "$out" | passed_count)
  [ -n "$n" ] || { echo "inkling2_loop host tests do not build or fail: $(first_problem "$out")"; exit 1; }
  count engine_loop_tests "$n"
  [ "$n" -ge "$MIN_LOOP_TESTS" ] || { echo "only $n inkling2_loop tests pass (need $MIN_LOOP_TESTS)"; exit 1; }
  echo "$n inkling2_loop host tests pass"
)

r_chat_template() (
  engine_ok || { echo "ENGINE_SRC $ENGINE is not a git worktree"; exit 1; }
  [ -f "$ENGINE/crates/server/tests/inkling2_loop_chat.rs" ] || { echo "crates/server/tests/inkling2_loop_chat.rs does not exist"; exit 1; }
  ensure_fixtures
  out=$( cd "$ENGINE" && REXL3_NO_CUDA=1 CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}" \
    cargo test -p gb-server --locked --test inkling2_loop_chat -- --test-threads=4 2>&1 )
  printf '%s\n' "$out" > "$OUT/chat.out"
  n=$(printf '%s\n' "$out" | passed_count)
  [ -n "$n" ] || { echo "the chat tests do not build or fail: $(first_problem "$out")"; exit 1; }
  count chat_tests "$n"
  [ "$n" -ge "$MIN_CHAT_TESTS" ] || { echo "only $n chat tests pass (need $MIN_CHAT_TESTS)"; exit 1; }
  printf '%s\n' "$out" | grep -q '^test render_.* \.\.\. ok$' || { echo "no passing render_ test"; exit 1; }
  printf '%s\n' "$out" | grep -q '^test parse_.* \.\.\. ok$' || { echo "no passing parse_ test"; exit 1; }
  echo "$n chat tests pass (render_ and parse_ present)"
)

r_bench_tools() (
  cd "$RECIPE" || exit 1
  total=0
  for f in test_bench test_gates; do
    [ -f "tests/$f.py" ] || { echo "tests/$f.py does not exist"; exit 1; }
    out=$(python3 "tests/$f.py" 2>&1) && rc=0 || rc=$?
    printf '%s\n' "$out" > "$OUT/$f.out"
    last=$(printf '%s\n' "$out" | tail -n 1)
    [ "$rc" -eq 0 ] || { echo "tests/$f.py exited $rc: $last"; exit 1; }
    [[ $last =~ ^$f:\ ([0-9]+)\ passed,\ 0\ failed$ ]] || { echo "tests/$f.py last line is not '$f: N passed, 0 failed': $last"; exit 1; }
    n=${BASH_REMATCH[1]}
    [ "$n" -ge "$MIN_BENCH_TESTS" ] || { echo "only $n $f tests (need $MIN_BENCH_TESTS)"; exit 1; }
    total=$((total + n))
  done
  count bench_tests "$total"
  echo "$total bench and gate tool tests pass"
)

# r_record GATE ARGS...: the newest record of the gate, checked by scripts/check_record.py
r_record() (
  gate=$1; shift
  [ -f "$RECIPE/scripts/check_record.py" ] || { echo "scripts/check_record.py does not exist"; exit 1; }
  out=$(python3 "$RECIPE/scripts/check_record.py" "$gate" --root "$RECIPE" --engine "$ENGINE" "$@" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | head -n 1
  exit "$rc"
)
r_roofline() { r_record roofline --metric read_gbs_mean3 --metric bf16_tflops_8192_mean3; }
r_engine_build() { r_record engine-build --need-engine --min-tests 60; }
r_oracle_fixtures() { r_record oracle-fixtures --metric fixtures; }
r_kernels_moe() { r_record kernels-moe --need-engine --min-tests 6; }
r_kernels_attn() { r_record kernels-attn --need-engine --min-tests 8; }
r_kernels_dense() { r_record kernels-dense --need-engine --min-tests 10; }
r_forward_oracle() { r_record forward-oracle --need-engine --min-tests 4 --min-metric passes_compared=48; }
r_boot() { r_record boot --need-engine --metric boot_s; }
r_smoke() { r_record smoke --need-engine --min-tests 4; }
r_kernels_draft() { r_record kernels-draft --need-engine --min-tests 6; }
r_kernels_vision() { r_record kernels-vision --need-engine --min-tests 6; }
r_vision_smoke() { r_record vision-smoke --need-engine --min-tests 8; }
r_api_test() { r_record api-test --need-engine --min-tests 15; }
r_exact_basic() { r_record exact-basic --need-engine --min-tests 4; }
r_needle() { r_record needle --need-engine --min-tests 2; }
r_greedy_oracle() { r_record greedy-oracle --need-engine --min-tests 8 --min-metric tokens_equal=16; }
r_speed_baseline() { r_record speed-baseline --need-engine --metric decode_c1_tok_s --metric prefill_32k_tok_s; }
r_cache() { r_record cache --need-engine --min-tests 8; }
r_cache_images() { r_record cache-images --need-engine --min-tests 8; }
r_exact_graph() { r_record exact-graph --need-engine --min-tests 10; }
r_speed_graph() { r_record speed-graph --need-engine --metric decode_c1_short_tok_s --min-metric decode_c1_short_tok_s=45; }
r_oracle_cacheddiff() { r_record oracle-cacheddiff --min-tests 2; }
r_oracle_gap() { r_record oracle-gap --need-engine --min-tests 8; }
r_row_probe() { r_record row-probe --min-tests 4; }
r_memory_plan() { r_record memory-plan --need-engine --min-tests 4; }
r_expert_arena() { r_record expert-arena --need-engine --min-tests 4; }
r_speed_prefill() { r_record speed-prefill --need-engine --metric prefill_32k_tok_s --min-metric prefill_32k_tok_s=2120; }
r_exact_prefill() { r_record exact-prefill --need-engine --min-tests 8; }
r_drafted_exact() { r_record spec --need-engine --min-tests 8; }
r_drafted_images() { r_record spec-images --need-engine --min-tests 8; }
r_drafted_cache() { r_record spec-cache --need-engine --min-tests 4; }
r_speed_drafted() { r_record spec-final --need-engine --metric speedup_code --min-metric speedup_code=1.5; }
r_markov_variants() { r_record markov-variants --need-engine --min-tests 4; }
r_speed_prefill_3k() { r_record speed-prefill --need-engine --min-tests 1 --metric prefill_32k_tok_s --min-metric prefill_32k_tok_s=3000 --same-engine exact-prefill; }
r_speed_decode_100() { r_record speed-decode --need-engine --min-tests 1 --metric decode_mean4_tok_s --min-metric decode_mean4_tok_s=100 --same-engine spec; }
r_c4_baseline() { r_record speed-c4-baseline --need-engine --min-tests 20 --metric decode_c4_rolling_tok_s; }
r_speed_c4_200() { r_record speed-c4 --need-engine --min-tests 1 --metric decode_c4_rolling_tok_s --min-metric decode_c4_rolling_tok_s=200 --same-engine spec; }

r_record_checker() (
  cd "$RECIPE" || exit 1
  [ -f tests/test_records.py ] || { echo "tests/test_records.py does not exist"; exit 1; }
  out=$(python3 tests/test_records.py 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" > "$OUT/test_records.out"
  last=$(printf '%s\n' "$out" | tail -n 1)
  [ "$rc" -eq 0 ] || { echo "tests/test_records.py exited $rc: $last"; exit 1; }
  [[ $last =~ ^test_records:\ ([0-9]+)\ passed,\ 0\ failed$ ]] || { echo "last line is not 'test_records: N passed, 0 failed': $last"; exit 1; }
  [ "${BASH_REMATCH[1]}" -ge 5 ] || { echo "only ${BASH_REMATCH[1]} record-checker tests (need 5)"; exit 1; }
  echo "${BASH_REMATCH[1]} record-checker tests pass"
)

r_loop_tag() (
  tag=$1
  grep -q 'FAILED' "$OUT/loop.out" 2> /dev/null && { echo "a families::inkling2_loop:: test fails"; exit 1; }
  hits=$(sed -n 's/^test \(families::inkling2_loop::[^ ]*\) \.\.\. ok$/\1/p' "$OUT/loop.out" 2> /dev/null | grep -c -- "$tag")
  [ "${hits:-0}" -ge 1 ] || { echo "no passing families::inkling2_loop:: test has '$tag' in its name"; exit 1; }
  echo "$hits passing test(s) named *$tag*"
)

emit_metrics() {
  local id v k
  printf 'METRIC bringup_gates=%s\n' "$passed"
  printf 'METRIC bringup_gates_total=%s\n' "$total"
  while read -r id v; do printf 'METRIC rung_%s=%s\n' "${id//-/_}" "$v"; done < "$OUT/rungs.txt"
  for k in recipe_tests oracle_tests engine_small_tests engine_loop_tests chat_tests bench_tests; do
    v=$(awk -v k="$k" '$1 == k {v = $2} END {print v + 0}' "$OUT/counts.txt")
    printf 'METRIC %s=%s\n' "$k" "$v"
  done
  # the best numbers on record, for watching the stretch targets move (they are not the rungs: the rungs also need the exactness pairing)
  for k in "speed-prefill prefill_32k_tok_s" "speed-decode decode_mean4_tok_s" "speed-graph decode_c1_4k_tok_s" "speed-c4-baseline decode_c4_rolling_tok_s"; do
    set -- $k
    v=$(python3 "$RECIPE/scripts/check_record.py" "$1" --root "$RECIPE" --show-metric "$2" 2> /dev/null) && printf 'METRIC %s=%s\n' "$2" "$v"
  done
  printf 'METRIC harness_s=%s\n' "$SECONDS"
  printf 'ASI recipe_head=%s engine_head=%s failed="%s"\n' \
    "$(git -C "$RECIPE" rev-parse --short HEAD 2> /dev/null || echo none)" \
    "$(git -C "$ENGINE" rev-parse --short HEAD 2> /dev/null || echo none)" "${failed# }"
}

prune_logs() {
  local d n=0
  while read -r d; do
    n=$((n + 1))
    if [ "$n" -gt 10 ]; then rm -rf "$d"; fi
  done < <(ls -1dt "$RECIPE"/bench/runs/ladder-* 2> /dev/null)
  return 0
}

run_all() {
  local tag
  [ -f "$RECIPE/config/pins.sh" ] || { echo "autoresearch: $RECIPE is not the recipe root (no config/pins.sh)" >&2; exit 2; }
  for tag in git python3 shasum; do
    command -v "$tag" > /dev/null 2>&1 || { echo "autoresearch: $tag is not installed" >&2; exit 2; }
  done
  OUT=$RECIPE/bench/runs/ladder-$$
  mkdir -p "$OUT" && : > "$OUT/rungs.txt" && : > "$OUT/counts.txt" || { echo "autoresearch: cannot write $OUT" >&2; exit 2; }

  run_rung pins r_pins
  run_rung intake r_intake
  run_rung recipe-tests r_recipe_tests
  run_rung probe-record r_probe_record
  run_rung spec r_spec
  run_rung oracle-unit r_oracle_unit
  run_rung oracle-exl3-real r_oracle_exl3_real
  run_rung engine-small-intact r_engine_small_intact
  if engine_ok; then ensure_fixtures; cargo_tests 'families::inkling2_loop::' > "$OUT/loop.out"; else : > "$OUT/loop.out"; fi
  run_rung loop-suite r_loop_suite
  for tag in $LOOP_TAGS; do run_rung "loop-$tag" r_loop_tag "$tag"; done
  run_rung chat-template r_chat_template
  run_rung bench-tools r_bench_tools
  run_rung record-checker r_record_checker
  run_rung roofline r_roofline
  run_rung engine-build r_engine_build
  run_rung oracle-fixtures r_oracle_fixtures
  run_rung kernels-moe r_kernels_moe
  run_rung kernels-attn r_kernels_attn
  run_rung kernels-dense r_kernels_dense
  run_rung forward-oracle r_forward_oracle
  run_rung boot r_boot
  run_rung smoke r_smoke
  run_rung kernels-draft r_kernels_draft
  run_rung kernels-vision r_kernels_vision
  run_rung vision-smoke r_vision_smoke
  run_rung api-test r_api_test
  run_rung exact-basic r_exact_basic
  run_rung needle r_needle
  run_rung greedy-oracle r_greedy_oracle
  run_rung speed-baseline r_speed_baseline
  run_rung cache r_cache
  run_rung cache-images r_cache_images
  run_rung exact-graph r_exact_graph
  run_rung speed-graph r_speed_graph
  run_rung oracle-cacheddiff r_oracle_cacheddiff
  run_rung oracle-gap r_oracle_gap
  run_rung row-probe r_row_probe
  run_rung memory-plan r_memory_plan
  run_rung expert-arena r_expert_arena
  run_rung speed-prefill r_speed_prefill
  run_rung exact-prefill r_exact_prefill
  run_rung drafted-exact r_drafted_exact
  run_rung drafted-images r_drafted_images
  run_rung drafted-cache r_drafted_cache
  run_rung speed-drafted r_speed_drafted
  run_rung markov-variants r_markov_variants
  run_rung speed-prefill-3k r_speed_prefill_3k
  run_rung speed-decode-100 r_speed_decode_100
  run_rung c4-baseline r_c4_baseline
  run_rung speed-c4-200 r_speed_c4_200

  emit_metrics
  prune_logs
  return 0
}

run_all "$@"; exit $?
