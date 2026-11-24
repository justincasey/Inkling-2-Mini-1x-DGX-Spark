#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Static checks of the recipe: no GPU, no real ssh, nothing on a node. Run before any hardware run (AGENTS.md).
#   - bash -n on every shell script; the dispatch is a function and `main "$@"; exit $?` is the last line
#   - the usage header documents exactly the dispatched commands
#   - print-serve for every profile (the contract flags and environment), the env-to-flag mapping, determinism, and refusal
#     of bad values with no partial output; .env.example parses and gives the default command
#   - start runs exactly what print-serve prints (a stub ssh on PATH records what would run on the node) under the exclusive
#     GPU lock and the memory guard; a failing doctor launches nothing; build copies, verifies and builds under the node lock
#   - the pins: MODEL_REV, MANIFEST_SHA256, REXL3_UPSTREAM_BASE; an empty REXL3_COMMIT makes `build` refuse (step 4.8)
#   - remote scripts take variables by name (no positional $1) and a nested ssh uses -n
#   - the GPU SM clock cap (hardware protection, config/pins.sh): pinned and not a setting; `gpu-clocks` has no reset and no MHz; applied, with
#     its success line required, before start, build, engine-test and every ops_dev.sh build or run; every script that runs GPU work applies it
#   - the memory guard on a fake /proc/meminfo: kill on a low floor, kill on swap growth, minimum and window records,
#     refusal of a 0 MiB floor
#   - stop-other / restore-other are never reached from any other command
# The controller runs in a copy of the tree under a clean environment (no .env, no inherited variables).
# The last line is `test_recipe: N passed, 0 failed`.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1
BASH_BIN=$(command -v bash)
passed=0 fails=0
ok() { printf 'ok   %s\n' "$*"; passed=$((passed + 1)); }
bad() { printf 'FAIL %s\n' "$*"; fails=$((fails + 1)); }

work=$(mktemp -d)
web_pid=
trap '[[ -n $web_pid ]] && kill "$web_pid" 2>/dev/null; kill $(jobs -p) 2>/dev/null; [[ -n ${rpid:-} ]] && kill "$rpid" 2>/dev/null; rm -rf "$work"' EXIT

# ── shell syntax ──
for f in inkling2mini.sh config/pins.sh scripts/*.sh tests/*.sh; do
  if "$BASH_BIN" -n "$f"; then ok "bash -n $f"; else bad "bash -n $f"; fi
done
if [[ $(tail -n 1 inkling2mini.sh) == 'main "$@"; exit $?' ]] && [[ $(grep -c '^main "\$@"; exit \$?$' inkling2mini.sh) == 1 ]] \
  && grep -q '^main() {$' inkling2mini.sh; then ok "the dispatch is main() and its call is the last line"
else bad "inkling2mini.sh: main() { case ...; esac; } and a last line 'main \"\$@\"; exit \$?' are required"; fi
if python3 -m py_compile scripts/*.py; then ok "py_compile scripts"; else bad "py_compile scripts"; fi

# ── a clean copy of the tree, a stub ssh, a clean environment ──
tree=$work/tree stubbin=$work/bin stub=$work/stub
mkdir -p "$tree/scripts" "$tree/intake" "$stubbin" "$stub"
cp inkling2mini.sh "$tree/"; cp -R config "$tree/config"; cp scripts/memguard.sh "$tree/scripts/"; cp intake/manifest.tsv "$tree/intake/"
cat > "$stubbin/ssh" <<'EOF'
#!/usr/bin/env bash
# stub ssh: records stdin and arguments of every call; answers like a ready node
n=$(find "$STUB_DIR" -name 'call.*' | wc -l | tr -d ' ')
cat > "$STUB_DIR/call.$n"
printf '%s\n' "$*" > "$STUB_DIR/args.$n"
if grep -q 'engine none' "$STUB_DIR/call.$n"; then   # the doctor script
  echo "ok   stub doctor"
  [[ -z ${STUB_FAIL:-} ]] || echo "FAIL $STUB_FAIL"
  echo "engine 0123456789abcdef0123456789abcdef01234567 $(printf '%064d' 0)"
fi
if grep -q -- '-lgc' "$STUB_DIR/call.$n" "$STUB_DIR/args.$n"; then   # the GPU cap: sudo -n nvidia-smi -lgc CAP,CAP (in the stdin script or an ssh argument)
  [[ -z ${STUB_NOCAP:-} ]] || { echo "sudo: a password is required"; exit 1; }
  cap=$(sed -n 's/^export CAP=//p' "$STUB_DIR/call.$n"); [[ -n $cap ]] || cap=$(grep -o -- '-lgc [0-9]*' "$STUB_DIR/args.$n" | cut -d' ' -f2)
  cap=${STUB_BADCAP:-$cap}
  echo "GPU clocks set to \"(gpuClkMin $cap, gpuClkMax $cap)\" for GPU 0000000F:01:00.0"; echo "All done."
fi
if grep -q 'cat >' "$STUB_DIR/args.$n"; then shasum -a 256 "$STUB_DIR/call.$n" | cut -d' ' -f1; fi   # the source copy: report its sha256
exit 0
EOF
chmod +x "$stubbin/ssh"
# ctl VAR=value ... -- ARGS: the controller of the copied tree, clean environment, stub ssh first on PATH
ctl() {
  local -a e=(); while [[ $1 != -- ]]; do e+=("$1"); shift; done; shift
  ( cd "$tree" && env -i HOME="$HOME" PATH="$stubbin:$PATH" STUB_DIR="$stub" "${e[@]}" "$BASH_BIN" ./inkling2mini.sh "$@" )
}
stub_calls() { find "$stub" -name 'call.*' | wc -l | tr -d ' '; }
stub_reset() { rm -f "$stub"/call.* "$stub"/args.*; }

# ── usage header lists exactly the dispatched commands ──
helptext=$(ctl -- help)
dispatched=$(awk '/^main\(\) \{/ {on = 1; next} on && /^}/ {exit} on && /^    [a-z-]+(\|[a-z-]+)*\)/ {sub(/\).*/, ""); sub(/^    /, ""); print}' inkling2mini.sh \
  | sed 's/^help|.*/help/' | sort | paste -sd' ' -)
documented=$(sed -n 's/^  \([a-z][a-z-]*\)\( .*\)\{0,1\}$/\1/p' <<<"$helptext" | sort | paste -sd' ' -)
[[ -n $dispatched ]] || bad "no dispatcher found in inkling2mini.sh"
if [[ $dispatched == "$documented" ]]; then ok "usage documents exactly the dispatched commands ($dispatched)"
else bad "usage [$documented] differs from the dispatch [$dispatched]"; fi

# ── print-serve: the contract, per profile ──
# check_serve LABEL JSON TEXT: JSON holds the expected values
check_serve() {
  T=$3 python3 - "$1" "$2" <<'PY'
import json, os, shlex, sys
label, want = sys.argv[1], json.loads(sys.argv[2])
text = os.environ["T"]
lines = [l for l in text.splitlines() if l and not l.startswith("#")]
heads = [l for l in text.splitlines() if l.startswith("# profile ")]
errs = []
if len(lines) != 1 or len(heads) != 1:
    errs.append(f"expected one header and one command, got {len(heads)} and {len(lines)}")
    lines = lines[:1] or [""]
w = shlex.split(lines[0])
if w[:3] != ["exec", "env", "-i"]:
    errs.append("does not start with `exec env -i`")
b = next((i for i, x in enumerate(w) if x.endswith("/bin/rexl3-serve")), None)
if b is None:
    errs.append("no .../bin/rexl3-serve")
    b = len(w)
env = dict(x.split("=", 1) for x in w[3:b])
args = w[b + 1:]
val = lambda f: args[args.index(f) + 1] if f in args else None
exp_env = {"HOME", "USER", "LOGNAME", "PATH", "RUST_BACKTRACE"}
if want.get("floor"):
    exp_env.add("REXL3_MEMORY_FLOOR_GIB")
exp_env |= set(want.get("extra_env", {}))
if set(env) != exp_env:
    errs.append(f"environment keys {sorted(env)} != {sorted(exp_env)}")
if env.get("REXL3_MEMORY_FLOOR_GIB") != (want.get("floor") or None):
    errs.append(f"REXL3_MEMORY_FLOOR_GIB={env.get('REXL3_MEMORY_FLOOR_GIB')!r}, want {want.get('floor')!r}")
for k, v in want.get("extra_env", {}).items():
    if env.get(k) != v:
        errs.append(f"{k}={env.get(k)!r}, want {v!r}")
if any(k.startswith("NCCL_") for k in env) or any(a.startswith("--tp-") for a in args):
    errs.append("TP1 has no NCCL environment and no --tp-* flags")
for f, v in [("--max-model-len", want["len"]), ("--slots", want["slots"]), ("--kv-tokens", want["kv"]),
             ("--prefill-chunk", want.get("chunk", "4096")), ("--kv-dtype", "int8"), ("--spec", want.get("spec", "dspark")),
             ("--host", "0.0.0.0"), ("--port", want.get("port", "8000")),
             ("--served-model-name", want.get("name", "inkling-2-mini-1x-dgx" "-spark"))]:
    if args.count(f) != 1 or val(f) != v:
        errs.append(f"{f} {val(f)!r} (x{args.count(f)}), contract {v!r}")
if not (val("--model") or "").endswith("/Inkling-2-Mini-SAGE-REXL3-4.23bpw"):
    errs.append(f"--model {val('--model')}")
if args.count("--allow-unvalidated-pack") != 1:
    errs.append("--allow-unvalidated-pack is not present exactly once")
p = want.get("prefix", "on")
got = (val("--prefix-cache"), val("--prefix-cache-mib"))
exp = {"on": ("on", None), "off": ("off", None), "auto": (None, None)}.get(p, (None, p))
if got != exp:
    errs.append(f"PREFIX_CACHE={p}: (--prefix-cache, --prefix-cache-mib) {got}, want {exp}")
if ("--no-vision" in args) != (want.get("vision", "on") == "off"):
    errs.append(f"VISION={want.get('vision', 'on')}: --no-vision {'present' if '--no-vision' in args else 'absent'}")
tail = want.get("extra_flags", "").split()
if tail and args[-len(tail):] != tail:
    errs.append(f"extra flags {tail} are not last")
if "measured" in want and f"(PROFILE_MEASURED={want['measured']})" not in " ".join(heads):
    errs.append(f"header lacks PROFILE_MEASURED={want['measured']}")
for e in errs:
    print(f"FAIL print-serve ({label}): {e}")
if not errs:
    print(f"ok   print-serve ({label}): contract flags, exact environment keys, no TP flags")
sys.exit(1 if errs else 0)
PY
}
# the contract's own profile values come from docs/FIT.md: bootstrap 262,144 x 4, long 1,048,576 x 2, max 1,048,576 x 4
for pv in bootstrap:262144:4 long:1048576:2 max:1048576:4; do
  IFS=: read -r prof len slots <<<"$pv"
  out=$(ctl PROFILE=$prof -- print-serve 2>&1) || bad "print-serve PROFILE=$prof exited non-zero: $out"
  res=$(check_serve "profile $prof" "{\"len\":\"$len\",\"slots\":\"$slots\",\"kv\":\"$((len * slots))\",\"measured\":1}" "$out") \
    && { echo "$res"; passed=$((passed + 1)); } || { echo "$res"; fails=$((fails + 1)); }
  grep -q "^# profile $prof (PROFILE_MEASURED=1)" <<<"$out" && ok "profile $prof is marked measured" || bad "profile $prof header: $(head -n 1 <<<"$out")"
  [[ $out == "$(ctl PROFILE=$prof -- print-serve 2>&1)" ]] && ok "print-serve $prof is deterministic" || bad "print-serve $prof differs between two runs"
done
for f in config/profiles/*.env; do
  prof=$(basename "$f" .env)
  if grep -qx 'PROFILE_MEASURED=1' "$f"; then
    python3 - "$prof" <<'PY' || bad "$f says PROFILE_MEASURED=1 but no passing bench/records/*-profile-$prof record (gate memory) exists"
import glob, json, sys
ok = [p for p in glob.glob(f"bench/records/*-profile-{sys.argv[1]}/meta.json")
      if json.load(open(p)).get("gate") == "memory" and json.load(open(p)).get("result") == "pass"]
sys.exit(0 if ok else 1)
PY
  else grep -qx 'PROFILE_MEASURED=0' "$f" || bad "$f must say PROFILE_MEASURED=0 or 1"; fi
done
ok "every profile's PROFILE_MEASURED agrees with its records"

# ── env to flag mapping ──
run_map() {   # run_map LABEL JSON VAR=value ...
  local label=$1 json=$2 out res; shift 2
  out=$(ctl "$@" -- print-serve 2>&1) || { bad "print-serve ($label) exited non-zero: $out"; return; }
  if res=$(check_serve "$label" "$json" "$out"); then echo "$res"; passed=$((passed + 1)); else echo "$res"; fails=$((fails + 1)); fi
}
run_map "overrides" '{"len":"4096","slots":"3","kv":"9999","chunk":"2048","port":"8123","name":"mini-x","spec":"copy","prefix":"off","vision":"off","floor":"6.5","extra_env":{"REXL3_TEST_A":"1","REXL3_TEST_B":"x"},"extra_flags":"--foo 1 --bar"}' \
  MAX_MODEL_LEN=4096 SLOTS=3 KV_TOKENS=9999 PREFILL_CHUNK=2048 PORT=8123 SERVED_NAME=mini-x SPEC=copy PREFIX_CACHE=off VISION=off \
  MEMORY_FLOOR_GIB=6.5 EXTRA_ENV="REXL3_TEST_A=1 REXL3_TEST_B=x" EXTRA_FLAGS="--foo 1 --bar"
run_map "spec none, cache MiB" '{"len":"262144","slots":"4","kv":"1048576","spec":"none","prefix":"2048"}' SPEC=none PREFIX_CACHE=2048
run_map "spec auto, cache auto" '{"len":"262144","slots":"4","kv":"1048576","spec":"auto","prefix":"auto"}' SPEC=auto PREFIX_CACHE=auto
run_map "features on (steps 4.6, 4.5, 4.7)" '{"len":"262144","slots":"4","kv":"1048576","spec":"dspark","prefix":"on","vision":"on"}' SPEC=dspark PREFIX_CACHE=on VISION=on
run_map "kv default follows the override" '{"len":"8192","slots":"4","kv":"32768"}' MAX_MODEL_LEN=8192
out=$(ctl NODES=203.0.113.7 -- print-serve 2>&1); grep -q '^# profile bootstrap .*spark@203.0.113.7' <<<"$out" \
  && ok "print-serve names the node when NODES is set" || bad "print-serve header with a node: $(head -n 1 <<<"$out")"

# ── bad values are refused, with no partial output ──
badvals=(SPEC=mtp SPEC=DSpark PREFIX_CACHE=pool PREFIX_CACHE=1.5 VISION=yes MEMORY_FLOOR_GIB=0.1 MEMORY_FLOOR_GIB=abc MEMORY_FLOOR_GIB=-2
  MEMORY_FLOOR_GIB=8GiB PORT=0 PORT=70000 PORT=abc MAX_MODEL_LEN=0 MAX_MODEL_LEN=1e6 SLOTS=0 SLOTS=x KV_TOKENS=0 PREFILL_CHUNK=0
  GPU_CLOCKS=fast GPU_CLOCKS=2600 GPU_CLOCKS=1500 GPU_CLOCKS=0 EXTRA_ENV=notakeyvalue PROFILE=nope PROFILE=../pins API_KEY=secret "NODES=bad;host" BASE_DIR=relative/dir "SERVED_NAME=a b")
refused=0
for kv in "${badvals[@]}"; do
  out=$(ctl "$kv" -- print-serve 2>"$work/err"); rc=$?
  if (( rc != 0 )) && [[ -z $out && -s $work/err ]]; then refused=$((refused + 1)); else bad "$kv: rc=$rc, stdout '${out:0:60}', stderr '$(head -c 80 "$work/err")'"; fi
done
(( refused == ${#badvals[@]} )) && ok "$refused bad values refused with empty stdout, a message and a non-zero exit"
out=$(ctl NODES="a b" -- print-serve 2>&1); [[ $out == *"one address"* ]] && ok "two nodes are refused (TP1)" || bad "NODES with two entries: $out"
# a broken profile file is refused before any output
cp -R "$tree" "$work/tree2"; printf 'PROFILE_MEASURED=0\nMAX_MODEL_LEN=1024\nSLOTS=1\nFOO=1\n' > "$work/tree2/config/profiles/bootstrap.env"
out=$(cd "$work/tree2" && env -i HOME="$HOME" PATH="$PATH" "$BASH_BIN" ./inkling2mini.sh print-serve 2>/dev/null); rc=$?
(( rc != 0 )) && [[ -z $out ]] && ok "a profile file with an unknown key is refused with no output" || bad "profile with an unknown key: rc=$rc out '$out'"
rm -rf "$work/tree2"

# ── .env.example: parses, gives the default command, documents every key the controller defaults ──
cp -R "$tree" "$work/tree3"; cp .env.example "$work/tree3/.env"
out=$(cd "$work/tree3" && env -i HOME="$HOME" PATH="$PATH" "$BASH_BIN" ./inkling2mini.sh print-serve 2>&1); rc=$?
(( rc == 0 )) && [[ $out == "$(ctl -- print-serve 2>&1)" ]] && ok ".env.example parses and gives the default print-serve" || bad ".env.example as .env: rc=$rc"
rm -rf "$work/tree3"
for k in $(sed -n 's/^: "${\([A-Z_0-9]*\):=.*/\1/p' inkling2mini.sh); do
  grep -Eq "^#? ?$k=" .env.example && ok ".env.example documents $k" || bad ".env.example lacks $k"
done
cap_pin=$( . config/pins.sh; echo "$GPU_CLOCK_CAP_MHZ" )
grep -qx 'STOP_OTHER="cd ~/recipes/Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark && ./inkling2.sh stop"' .env.example \
  && grep -qx "RESTORE_OTHER=\"cd ~/recipes/Inkling-2-Small-EXL3-4bpw-4x-DGX-Spark && ./inkling2.sh gpu-clocks $cap_pin && ./inkling2.sh start\"" .env.example \
  && ok ".env.example stops the Inkling-2-Small group, and restores it with its own controller after locking every rank at the pinned cap" || bad ".env.example STOP_OTHER / RESTORE_OTHER (the restore must lock the ranks at the pinned $cap_pin MHz first)"
if grep -Eq '^GPU_CLOCKS=' .env.example; then bad ".env.example sets GPU_CLOCKS: the cap is pinned, not a setting"
else ok ".env.example has no GPU_CLOCKS: the cap is pinned in config/pins.sh"; fi

# ── pins ──
( set -eu; . config/pins.sh
  [[ $MODEL_REV =~ ^[0-9a-f]{40}$ ]] || { echo "MODEL_REV is not a 40-hex commit sha"; exit 1; }
  [[ $REXL3_UPSTREAM_BASE =~ ^[0-9a-f]{40}$ ]] || { echo "REXL3_UPSTREAM_BASE is not a 40-hex commit sha"; exit 1; }
  [[ -z $REXL3_COMMIT || $REXL3_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "REXL3_COMMIT is neither empty nor a 40-hex sha"; exit 1; }
  [[ $(shasum -a 256 "$MANIFEST_FILE" | cut -d' ' -f1) == "$MANIFEST_SHA256" ]] || { echo "MANIFEST_SHA256 != shasum -a 256 $MANIFEST_FILE"; exit 1; }
  [[ $(wc -l < "$MANIFEST_FILE" | tr -d ' ') == 13 ]] || { echo "manifest does not list 13 files"; exit 1; }
) && ok "pins: MODEL_REV and REXL3_UPSTREAM_BASE are full shas, MANIFEST_SHA256 equals shasum of the manifest, 13 files" || bad "pins"
sed -i.bak 's/^REXL3_COMMIT=.*/REXL3_COMMIT=/' "$tree/config/pins.sh"; rm -f "$tree/config/pins.sh.bak"   # the copy holds the real pin; this test is about the empty one
stub_reset
out=$(ctl -- build 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"step 4.8"* ]] && (( $(stub_calls) == 0 )) \
  && ok "build refuses while REXL3_COMMIT is empty, names step 4.8, touches no node" || bad "build with an empty pin: rc=$rc $out"
out=$(ctl NODES=203.0.113.7 -- build 2>&1); (( $? != 0 )) && [[ $out == *"step 4.8"* ]] && ok "build refuses an empty pin even with a node" || bad "build with node: $out"

# ── a throwaway engine repository: build refusals and the copy/verify/lock sequence ──
eng=$work/engine; git init -q "$eng"; export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.invalid GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.invalid
echo a > "$eng/a"; git -C "$eng" add a; git -C "$eng" commit -q -m base; base=$(git -C "$eng" rev-parse HEAD)
echo b > "$eng/b"; git -C "$eng" add b; git -C "$eng" commit -q -m child; child=$(git -C "$eng" rev-parse HEAD)
git -C "$eng" checkout -q --orphan other; git -C "$eng" rm -q -rf .; echo c > "$eng/c"; git -C "$eng" add c; git -C "$eng" commit -q -m unrelated
stranger=$(git -C "$eng" rev-parse HEAD)
sed -i.bak "s/^REXL3_UPSTREAM_BASE=.*/REXL3_UPSTREAM_BASE=$base/; s/^REXL3_COMMIT=.*/REXL3_COMMIT=$child/" "$tree/config/pins.sh"; rm -f "$tree/config/pins.sh.bak"
stub_reset
out=$(ctl NODES=203.0.113.7 REXL3_SRC="$eng" -- build "$stranger" 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"does not descend"* ]] && (( $(stub_calls) == 0 )) && ok "build refuses a commit that does not descend from the upstream base" || bad "stranger commit: rc=$rc $out"
out=$(ctl NODES=203.0.113.7 REXL3_SRC="$eng" -- build 0000000000000000000000000000000000000001 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"not a commit"* ]] && ok "build refuses an unknown revision" || bad "unknown revision: rc=$rc $out"
stub_reset
out=$(ctl NODES=203.0.113.7 REXL3_SRC="$eng" -- build 2>&1); rc=$?
script=$(cat "$(grep -l 'export SHA=' "$stub"/call.* 2>/dev/null | head -n 1)" 2>/dev/null)
if (( rc == 0 )) && [[ $out == *"pinned engine"* ]] && grep -q "export SHA=$child" <<<"$script" && grep -q 'flock -x -n 8' <<<"$script" \
  && grep -q 'timeout "$BUILD_TIMEOUT_S" cargo +"$TOOLCHAIN" build --release --locked' <<<"$script" && grep -q 'CARGO_BUILD_JOBS=8' <<<"$script" \
  && grep -q 'REXL3_CUDA_ARCH=sm_121' <<<"$script" && grep -q 'REXL3_COMMIT="$SHA"' <<<"$script" && grep -q 'export TOOLCHAIN=1.89.0' <<<"$script" \
  && grep -q 'rexl3-serve.commit' <<<"$script" && grep -q 'cuda_arch' <<<"$script" && grep -q 'cat >' "$stub"/args.*; then
  ok "build: the copy is verified by sha256, then built under the node lock with the pinned toolchain, sm_121, the commit baked in"
else bad "build sequence: rc=$rc out=$out"; fi
capn=$(grep -l -e '-lgc' "$stub"/call.* 2>/dev/null | head -n 1 | sed 's/.*call\.//'); copyn=$(grep -l 'cat >' "$stub"/args.* 2>/dev/null | head -n 1 | sed 's/.*args\.//')
[[ -n $capn && -n $copyn ]] && (( capn < copyn )) && ok "build applies the GPU cap before it copies or compiles anything" || bad "build: cap call '$capn', source copy call '$copyn'"
stub_reset
out=$(ctl NODES=203.0.113.7 REXL3_SRC="$eng" -- build "$base" 2>&1); rc=$?
(( rc == 0 )) && [[ $out == *"LAB engine"* ]] && ok "build REV other than the pin is announced as a lab build" || bad "lab build: rc=$rc $out"

# ── bench ──
stub_reset
out=$(ctl NODES=203.0.113.7 -- bench bogus 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"usage: bench"* ]] && (( $(stub_calls) == 0 )) && ok "bench refuses an unknown phase and touches no node" || bad "bench bogus: rc=$rc $out"
out=$(ctl -- bench rolling 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"no node"* && $out != *"usage: bench"* ]] && ok "bench rolling is a phase (refused for want of a node, not as an unknown phase)" || bad "bench rolling without a node: rc=$rc $out"
[[ $helptext == *"rolling (sustained c=4"* && $helptext == *"decode, rolling, prefill"* ]] && ok "the usage names the rolling phase and puts it in all" || bad "usage lacks the rolling phase"
out=$(ctl -- bench decode 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"no node"* ]] && ok "bench with no node says so and fails" || bad "bench without a node: rc=$rc $out"
out=$(ctl -- bench profile 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"no node"* ]] && ok "bench profile (the memory bracket) is a known phase and with no node says so" || bad "bench profile without a node: rc=$rc $out"
out=$(ctl NODES=203.0.113.7 -- bench pin 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"is not REXL3_COMMIT"* ]] && ok "bench pin refuses an installed engine that is not REXL3_COMMIT" || bad "bench pin on another engine: rc=$rc $out"

# ── doctor ──
out=$(ctl -- doctor 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"no node"* ]] && ok "doctor with no node says so and fails" || bad "doctor without a node: rc=$rc $out"
stub_reset
out=$(ctl NODES=203.0.113.7 -- doctor 2>&1); rc=$?
(( rc == 0 )) && [[ $out == *"doctor: the node is ready"* ]] && ok "doctor passes on a stub node that reports ok" || bad "doctor on the stub node: rc=$rc $out"
out=$(ctl NODES=203.0.113.7 STUB_FAIL="GPU in use:" -- doctor 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"FAIL GPU in use:"* ]] && ok "doctor fails when the node reports a FAIL (GPU in use)" || bad "doctor with a busy GPU: rc=$rc $out"
dbody=$(awk '/^cmd_doctor\(\)/ {on = 1} on {print} on && /^}/ {exit}' inkling2mini.sh)
for what in 'GPU in use' 'GPU_LOCK is held' 'MemAvailable' 'foreign container' 'files missing or short' 'no NVIDIA GB10' 'rexl3-serve.commit says otherwise'; do
  grep -qF "$what" <<<"$dbody" && ok "doctor checks: $what" || bad "doctor lacks the check '$what'"
done

# ── start: exactly what print-serve prints, under the GPU lock and the guard; a failing doctor launches nothing ──
port=$("$BASH_BIN" -c 'python3 -c "import socket; s = socket.socket(); s.bind((\"127.0.0.1\", 0)); print(s.getsockname()[1])"')
mkdir -p "$work/web"; echo ok > "$work/web/health"
python3 -m http.server "$port" --bind 127.0.0.1 --directory "$work/web" >/dev/null 2>&1 & web_pid=$!
for _ in $(seq 50); do curl -fsS -m 1 -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null && break; sleep 0.1; done
stub_reset
out=$(ctl NODES=127.0.0.1 PORT="$port" VISION=off STUB_FAIL="GPU in use:" -- start 2>&1); rc=$?
launched=$(grep -l 'export CMD=' "$stub"/call.* 2>/dev/null | wc -l | tr -d ' ')
(( rc != 0 && launched == 0 )) && ok "a failing doctor stops start before anything is launched" || bad "start after a failing doctor: rc=$rc launched=$launched"
# the engine's prefix cache serves images too: `start` refuses no cache/vision combination (the stub doctor's failure is all that stops it)
stub_reset
for kv in PREFIX_CACHE=on PREFIX_CACHE=2048 PREFIX_CACHE=auto; do
  out=$(ctl NODES=127.0.0.1 PORT="$port" VISION=on "$kv" STUB_FAIL="GPU in use:" -- start 2>&1)
  [[ $out != *"not built for images"* ]] && (( $(stub_calls) > 0 )) \
    && ok "start passes $kv with images on to the node's doctor" || bad "start with $kv and VISION=on was stopped before the node: $out"
  stub_reset
done
out=$(ctl NODES=127.0.0.1 PORT="$port" VISION=off STOP_OTHER="touch $work/stop-other-ran" RESTORE_OTHER="touch $work/restore-other-ran" -- start 2>&1); rc=$?
(( rc == 0 )) && [[ $out == *"boot_s="* ]] && ok "start completes against a stub node and a stub /health" || bad "start on the stub node: rc=$rc $out"
launch=$(grep -l 'export CMD=' "$stub"/call.* 2>/dev/null | head -n 1)
if [[ -n $launch ]]; then
  want=$(ctl NODES=127.0.0.1 PORT="$port" VISION=off -- print-serve | grep -v '^#')
  got=$("$BASH_BIN" -c "$(grep '^export CMD=' "$launch"); printf '%s' \"\$CMD\"")
  [[ $got == "$want" ]] && ok "start hands the node exactly the string print-serve prints" || bad "start's command differs from print-serve: [$got] vs [$want]"
  grep -q 'exec 9>>/tmp/rexl3-gpu.lock; flock -x -n 9' "$launch" && ok "the server runs under flock -x -n on /tmp/rexl3-gpu.lock (fd 9 held for its life)" || bad "no exclusive GPU lock in the launch script"
  grep -q ' armed pid=\$pid ' "$launch" && grep -q 'the memory guard did not arm' "$launch" && grep -q 'kill -9 "$pid"' "$launch" \
    && ok "the server is killed unless the guard armed on its pid" || bad "the guard gate is missing from the launch script"
  guard_line=$(grep -n 'memguard.sh" >> "$glog"' "$launch" | head -n 1 | cut -d: -f1); serve_line=$(grep -n 'setsid nohup bash -c "exec 9' "$launch" | head -n 1 | cut -d: -f1)
  [[ -n $guard_line && -n $serve_line ]] && (( guard_line < serve_line )) && ok "the guard starts before the server" || bad "guard/server order in the launch script"
  grep -q '^export GUARD_SRC=' "$launch" && grep -q '^export G_START=2048$' "$launch" && grep -q '^export G_MIB=1024$' "$launch" \
    && ok "the guard script and its floors (2048 MiB loading, 1024 MiB ready) ship with the launch" || bad "guard floors missing from the launch"
else bad "start never launched"; fi
# stop-other / restore-other are typed by a person only
[[ ! -e $work/stop-other-ran && ! -e $work/restore-other-ran ]] && ok "start, doctor and the tests never ran STOP_OTHER or RESTORE_OTHER" || bad "STOP_OTHER or RESTORE_OTHER ran"
users=$(awk '/^[a-z_]+\(\)/ {fn = $1; sub(/\(\).*/, "", fn); one = ($0 ~ /\}$/)} !/^#/ && /STOP_OTHER|RESTORE_OTHER/ {print fn} one {fn = ""; one = 0} /^}$/ {fn = ""}' inkling2mini.sh \
  | sort -u | grep -v '^$' | paste -sd' ' -)
[[ $(grep -c 'bash -c "\$STOP_OTHER"' inkling2mini.sh) == 1 && $(grep -c 'bash -c "\$RESTORE_OTHER"' inkling2mini.sh) == 1 && $users == "cmd_restore_other cmd_stop_other" ]] \
  && ok "STOP_OTHER and RESTORE_OTHER appear only in their two commands" || bad "STOP_OTHER or RESTORE_OTHER used outside cmd_stop_other / cmd_restore_other ($users)"
# the other stack's controller must not inherit this recipe's settings (its BASE_DIR, PORT, NODES decide where it looks)
ctl BASE_DIR=/mini/dir PORT=9 NODES=127.0.0.1 STOP_OTHER="printf '%s|%s|%s' \"\${BASE_DIR-unset}\" \"\${PORT-unset}\" \"\${NODES-unset}\" > $work/other-env" -- stop-other >/dev/null 2>&1
[[ $(cat "$work/other-env" 2>/dev/null) == "unset|unset|unset" ]] && ok "stop-other runs STOP_OTHER with a scrubbed environment" || bad "stop-other leaks BASE_DIR/PORT/NODES into STOP_OTHER ($(cat "$work/other-env" 2>/dev/null))"
restore=$(awk '/^cmd_restore_other\(\)/ {on = 1} on {print} on && /^}/ {exit}' inkling2mini.sh)
[[ $(grep -n 'cmd_stop$' <<<"$restore" | cut -d: -f1) -lt $(grep -n 'bash -c "\$RESTORE_OTHER"' <<<"$restore" | cut -d: -f1) ]] \
  && ok "restore-other stops this server before it runs RESTORE_OTHER" || bad "restore-other does not stop this server first"

# ── the GPU SM clock cap: hardware protection (owner, 2026-11-23): pinned, applied before any work, never released or changed ──
( . config/pins.sh; [[ ${GPU_CLOCK_CAP_MHZ:-} == 2200 ]] ) && ok "pins: GPU_CLOCK_CAP_MHZ is 2200" || bad "pins: GPU_CLOCK_CAP_MHZ is not 2200"
out=$(ctl GPU_CLOCKS=2200 -- print-serve 2>&1); rc=$?
(( rc == 0 )) && ok "an old GPU_CLOCKS line equal to the pin is tolerated (any other value is refused: see the bad values above)" || bad "GPU_CLOCKS=2200: rc=$rc $out"
# gpu-clocks takes lock|status only: no release, no other value; a refusal touches no node
stub_reset; n_refused=0
for a in reset 2600 0 -rgc 2200,2200 ""; do
  out=$(ctl NODES=203.0.113.7 -- gpu-clocks $a 2>&1); rc=$?
  if (( rc != 0 )) && [[ $out == *"never released, raised or changed"* ]]; then n_refused=$((n_refused + 1)); else bad "gpu-clocks '$a': rc=$rc $out"; fi
done
(( n_refused == 6 )) && (( $(stub_calls) == 0 )) && ok "gpu-clocks refuses reset, MHz values and no argument, and touches no node" || bad "gpu-clocks refusals: $n_refused of 6, $(stub_calls) node calls"
# `lock` applies exactly the pinned cap and needs the node's success line for that value
stub_reset
out=$(ctl NODES=203.0.113.7 -- gpu-clocks lock 2>&1); rc=$?
capcall=$(grep -l -e '-lgc' "$stub"/call.* 2>/dev/null | head -n 1)
if (( rc == 0 )) && [[ $out == *"capped at 2200 MHz"* && -n $capcall ]] && (( $(stub_calls) == 1 )) && grep -q '^export CAP=2200$' "$capcall" \
  && grep -q 'sudo -n nvidia-smi -lgc "\$CAP,\$CAP"' "$capcall"; then ok "gpu-clocks lock sends sudo -n nvidia-smi -lgc 2200,2200, the pinned value, and nothing else"
else bad "gpu-clocks lock: rc=$rc calls=$(stub_calls) $out"; fi
for f in STUB_NOCAP=1 STUB_BADCAP=2600; do
  out=$(ctl NODES=203.0.113.7 "$f" -- gpu-clocks lock 2>&1); rc=$?
  (( rc != 0 )) && [[ $out == *"no GPU work without it"* ]] && ok "gpu-clocks lock fails without the node's success line for the pin ($f)" || bad "gpu-clocks lock with $f: rc=$rc $out"
done
# start and engine-test apply the cap first; without it nothing is launched or run
first_call() { grep -l -e "$1" "$stub"/call.* 2>/dev/null | head -n 1 | sed 's/.*call\.//'; }
stub_reset
out=$(ctl NODES=127.0.0.1 PORT="$port" VISION=off -- start 2>&1); rc=$?
capn=$(first_call '-lgc'); launchn=$(first_call 'export CMD=')
(( rc == 0 )) && [[ -n $capn && -n $launchn ]] && (( capn < launchn )) && ok "start applies the GPU cap before it launches the server" || bad "start: cap call '$capn', launch call '$launchn', rc=$rc"
for f in STUB_NOCAP=1 STUB_BADCAP=2600; do
  stub_reset
  out=$(ctl NODES=127.0.0.1 PORT="$port" VISION=off "$f" -- start 2>&1); rc=$?
  (( rc != 0 )) && [[ -z $(first_call 'export CMD=') ]] && ok "start without the cap ($f) launches nothing" || bad "start with $f: rc=$rc"
done
for w in host goldens; do
  stub_reset
  out=$(ctl NODES=127.0.0.1 -- engine-test "$w" 2>&1); rc=$?
  capn=$(first_call '-lgc'); runn=$(first_call 'engine-test \$WHICH')
  (( rc == 0 )) && [[ -n $capn && -n $runn ]] && (( capn < runn )) && ok "engine-test $w applies the GPU cap before it runs anything" || bad "engine-test $w: cap call '$capn', run call '$runn', rc=$rc"
  stub_reset
  out=$(ctl NODES=127.0.0.1 STUB_NOCAP=1 -- engine-test "$w" 2>&1); rc=$?
  (( rc != 0 )) && [[ -z $(first_call 'engine-test \$WHICH') ]] && ok "engine-test $w without the cap runs nothing" || bad "engine-test $w with STUB_NOCAP: rc=$rc"
done
# scripts/gpu_cap.sh and scripts/ops_dev.sh: the cap before any work on the node
stub_reset
out=$(PATH="$stubbin:$PATH" STUB_DIR="$stub" "$BASH_BIN" scripts/gpu_cap.sh 203.0.113.7 2>&1); rc=$?
(( rc == 0 )) && [[ $out == *"capped at 2200 MHz on 203.0.113.7"* ]] && grep -q -e '-lgc 2200,2200' "$stub"/args.0 \
  && ok "gpu_cap.sh asks the node for -lgc 2200,2200 and reports the success line" || bad "gpu_cap.sh: rc=$rc $out"
for f in STUB_NOCAP=1 STUB_BADCAP=2600; do
  out=$(env PATH="$stubbin:$PATH" STUB_DIR="$stub" "$f" "$BASH_BIN" scripts/gpu_cap.sh 203.0.113.7 2>&1); rc=$?
  (( rc != 0 )) && [[ $out == *"no GPU work without it"* ]] && ok "gpu_cap.sh fails without the success line for the pin ($f)" || bad "gpu_cap.sh with $f: rc=$rc $out"
done
mkdir -p "$work/engine/crates/cuda"; : > "$work/engine/Cargo.toml"
for sub in "build --lib" "run --lib some_test"; do
  out=$(OPS_DRY=1 NODE=203.0.113.7 LAB_SRC="$work/engine" "$BASH_BIN" scripts/ops_dev.sh $sub 2>&1) || true
  [[ $(head -n 1 <<<"$out") == "scripts/gpu_cap.sh 203.0.113.7" ]] && ok "ops_dev.sh $sub applies the GPU cap before any work on the node" || bad "ops_dev.sh $sub: first line '$(head -n 1 <<<"$out")'"
done
stub_reset
out=$(PATH="$stubbin:$PATH" STUB_DIR="$stub" STUB_NOCAP=1 NODE=203.0.113.7 LAB_SRC="$work/engine" "$BASH_BIN" scripts/ops_dev.sh run --lib some_test 2>&1); rc=$?
(( rc != 0 )) && (( $(stub_calls) == 1 )) && [[ $out == *"no work on 203.0.113.7 without the GPU cap"* ]] && ok "ops_dev.sh stops at once when the cap cannot be applied" || bad "ops_dev.sh without the cap: rc=$rc calls=$(stub_calls) $out"
# static: the only clock command in any script is the pinned lock; every script that runs GPU work applies the cap or runs through one that does
clockcmds=$(grep -rnE 'nvidia-smi[^|;&]*(-rgc|-rmc|-rac|-ac |-lgc|-lmc|--(reset|lock)-(gpu|memory)-clocks|--applications-clocks)' --include='*.sh' --include='*.py' inkling2mini.sh autoresearch.sh scripts bench 2>/dev/null \
  | grep -vE ':[0-9]+:[[:space:]]*#' | grep -vE -e '-lgc "\$CAP,\$CAP"' -e '-lgc \$GPU_CLOCK_CAP_MHZ,\$GPU_CLOCK_CAP_MHZ' || true)
[[ -z $clockcmds ]] && ok "the only clock command in any script is the pinned lock (-lgc CAP,CAP): no release, no other clock" || bad "clock commands other than the pinned lock: $clockcmds"
nocap=
for f in $(grep -rlE -e 'flock[^|]*rexl3-gpu\.lock' -e 'nsys profile' -e 'rexl3-serve --model' --include='*.sh' autoresearch.sh bench scripts); do
  case $f in scripts/ops_dev.sh | scripts/gpu_cap.sh) continue ;; esac
  grep -vE '^[[:space:]]*#' "$f" | grep -qE 'gpu_cap\.sh|ops_dev\.sh|inkling2mini\.sh' || nocap="$nocap $f"   # a comment that names the cap does not apply it
done
[[ -z $nocap ]] && grep -q 'scripts/gpu_cap.sh' scripts/ops_dev.sh && ok "every script that runs GPU work applies the GPU cap or runs through ops_dev.sh / the controller" || bad "scripts that run GPU work without the GPU cap:$nocap"

# ── no lab address in any tracked file (node-1..node-4 are the names; the addresses live only in the untracked .env) ──
if git rev-parse --git-dir >/dev/null 2>&1; then
  leaked=$(git ls-files -z | xargs -0 grep -InE '(192\.168|10\.100)\.[0-9]+\.[0-9]+' 2>/dev/null | cut -c1-160 || true)
  [[ -z $leaked ]] && ok "no lab address (192.168.x.x, 10.100.x.x) in any tracked file" || bad "lab addresses in tracked files: $leaked"
else bad "tests/test_recipe.sh runs in a git checkout (the lab-address scan reads the tracked files)"; fi

# ── names the owner excluded from the recipe (2026-11-23): none may appear in any tracked file. The patterns are assembled from fragments so that this file does not contain them. ──
if git rev-parse --git-dir >/dev/null 2>&1; then
  excluded=$(printf '%s%s|%s%s' kna pcio speed bench)
  found=$(git ls-files -z | xargs -0 grep -IinE "$excluded" 2>/dev/null | cut -c1-160 || true)
  [[ -z $found ]] && ok "no excluded name in any tracked file" || bad "excluded names in tracked files: $found"
else bad "tests/test_recipe.sh runs in a git checkout (the name scan reads the tracked files)"; fi

# ── remote scripts: variables by name, nested ssh with -n ──
python3 - inkling2mini.sh <<'PY' && ok "remote bodies: no positional \$1, every nested ssh has -n or reads a pipe; remote() closes stdin" || bad "remote bodies"
import re, sys
src = open(sys.argv[1]).read()
bodies = re.findall(r"\bremote '([^']*)'", src) + re.findall(r"^[A-Z_]+='([^']*)'$", src, re.M)
errs = []
if not bodies:
    errs.append("no remote bodies found")
for b in bodies:
    first = b.strip().splitlines()[0][:50] if b.strip() else ""
    for m in re.finditer(r"(?<!\\)\$(?:[0-9]|@|\*|#)", b):
        errs.append(f"positional parameter {m.group(0)} in the body starting '{first}'")
    for line in b.splitlines():
        if re.search(r"\bssh\b", line) and not re.search(r"\bssh +(-\S+ +)*-n\b|\|\s*ssh\b", line):
            errs.append(f"nested ssh without -n: {line.strip()[:80]}")
if "} </dev/null" not in src or "bash -s <<<" not in src:
    errs.append("remote() does not run the script as a brace group with stdin from /dev/null")
for line in src.splitlines():
    if re.search(r"\bssh +(-n +)?\"\$\{SSH_OPTS", line) and not line.lstrip().startswith("#"):
        if "-n" not in line and "<<<" not in line and '< "' not in line:
            errs.append(f"local ssh with an unclear stdin: {line.strip()[:80]}")
print("\n".join(errs))
sys.exit(1 if errs else 0)
PY

# ── records: valid JSON, no lab addresses ──
recs=0
for f in bench/records/*/meta.json; do
  [[ -e $f ]] || continue
  recs=$((recs + 1))
  if python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$f" && ! grep -Eq '(^|[^0-9])(192\.168|10\.100)\.[0-9]+\.[0-9]+' "$f"; then ok "$f is valid JSON without lab addresses"
  else bad "$f is not valid JSON or holds a lab address"; fi
done
(( recs > 0 )) || bad "no bench/records/*/meta.json (the node probe record is expected)"

# ── scripts/final_run.sh: the preflight refusals, the pin refusal, one failing gate (a copy of the tree, a stub node, a fake engine repository) ──
ft=$work/final-tree fstub=$work/final-bin fdir=$work/final-stub feng=$work/final-engine
mkdir -p "$ft/scripts" "$ft/bench/tools" "$ft/bench/records" "$ft/intake" "$fstub" "$fdir" "$feng"
cp inkling2mini.sh .env.example "$ft/"; cp -R config "$ft/config"; cp intake/manifest.tsv "$ft/intake/"
cp scripts/final_run.sh scripts/memguard.sh scripts/check_record.py "$ft/scripts/"
cp bench/tools/final_records.py bench/tools/finalize_record.py bench/tools/exactness.py bench/tools/inkling_api.py "$ft/bench/tools/"
git -C "$feng" init -q && git -C "$feng" -c user.name=t -c user.email=t@t commit -q --allow-empty -m engine
fsha=$(git -C "$feng" rev-parse HEAD)
sed -i.bak -e "s/^REXL3_COMMIT=.*/REXL3_COMMIT=$fsha/" -e "s/^REXL3_UPSTREAM_BASE=.*/REXL3_UPSTREAM_BASE=$fsha/" "$ft/config/pins.sh"; rm -f "$ft/config/pins.sh.bak"
git -C "$ft" init -q && git -C "$ft" add -A && git -C "$ft" -c user.name=t -c user.email=t@t commit -q -m tree
cat > "$fstub/ssh" <<'EOF'
#!/usr/bin/env bash
# stub node for final_run.sh: a ready node whose installed engine is STUB_ENGINE; answers the cap, the doctor, the record's engine read
n=$(find "$STUB_DIR" -name 'call.*' | wc -l | tr -d ' ')
cat > "$STUB_DIR/call.$n"
printf '%s\n' "$*" > "$STUB_DIR/args.$n"
call=$STUB_DIR/call.$n
if grep -q 'engine none' "$call"; then echo "ok   stub doctor"; echo "engine ${STUB_ENGINE:-none} $(printf '%064d' 0)"; fi
if grep -q -- '-lgc' "$call"; then
  cap=$(sed -n 's/^export CAP=//p' "$call")
  echo "GPU clocks set to \"(gpuClkMin $cap, gpuClkMax $cap)\" for GPU 0000000F:01:00.0"; echo "All done."
fi
if grep -q 'no engine at' "$call"; then echo "commit ${STUB_ENGINE:-none}"; echo "version 0.1.1"; echo "cuda_arch sm_121"; echo "sha256 $(printf 'a%.0s' $(seq 64))"; echo "gpu_clock 2197 MHz, 3003 MHz"; fi
if grep -q 'cat >' "$STUB_DIR/args.$n"; then shasum -a 256 "$call" | cut -d' ' -f1; fi
exit 0
EOF
chmod +x "$fstub/ssh"
# frun VAR=value ... -- ARGS: final_run.sh of the copied tree, clean environment, stub node first on PATH
frun() {
  local -a e=(); while [[ $1 != -- ]]; do e+=("$1"); shift; done; shift
  ( cd "$ft" && env -i HOME="$HOME" PATH="$fstub:$PATH" STUB_DIR="$fdir" REXL3_SRC="$feng" "${e[@]}" "$BASH_BIN" scripts/final_run.sh "$@" )
}
fcalls() { find "$fdir" -name 'call.*' | wc -l | tr -d ' '; }
freset() { rm -f "$fdir"/call.* "$fdir"/args.*; rm -rf "$ft/bench/runs"; }
freset
out=$(frun NODE_NAME=node-3 -- 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"NODES is empty"* ]] && (( $(fcalls) == 0 )) && ok "final_run.sh refuses with no node and touches nothing" || bad "final_run without NODES: rc=$rc calls=$(fcalls) $out"
out=$(frun NODES=203.0.113.7 NODE_NAME=203.0.113.7 -- 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"NODE_NAME must be node-1..node-4"* ]] && (( $(fcalls) == 0 )) && ok "final_run.sh refuses a node address as the record's node name" || bad "final_run with an address as NODE_NAME: rc=$rc $out"
out=$(frun NODES=203.0.113.7 -- 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"NODE_NAME must be node-1..node-4"* ]] && ok "final_run.sh refuses a missing NODE_NAME" || bad "final_run without NODE_NAME: rc=$rc $out"
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 -- --bogus 2>&1); rc=$?
(( rc == 2 )) && [[ $out == *"usage: scripts/final_run.sh"* ]] && (( $(fcalls) == 0 )) && ok "final_run.sh refuses an unknown option" || bad "final_run --bogus: rc=$rc $out"
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 -- --resume "$work/none" 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"is not a run directory"* ]] && (( $(fcalls) == 0 )) && ok "final_run.sh --resume refuses a directory that is no run" || bad "final_run --resume of nothing: rc=$rc $out"
sed -i.bak "s/^REXL3_COMMIT=.*/REXL3_COMMIT=/" "$ft/config/pins.sh"
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 -- 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"not pinned"* ]] && (( $(fcalls) == 0 )) && ok "final_run.sh refuses an empty REXL3_COMMIT" || bad "final_run with no pin: rc=$rc $out"
mv "$ft/config/pins.sh.bak" "$ft/config/pins.sh"
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 REXL3_SRC="$work/final-tree" -- 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"is not in $work/final-tree"* ]] && (( $(fcalls) == 0 )) && ok "final_run.sh refuses a pin that is not in the engine repository" || bad "final_run with a foreign pin: rc=$rc $out"
echo "# edit" >> "$ft/scripts/memguard.sh"
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 -- 2>&1); rc=$?
(( rc != 0 )) && [[ $out == *"uncommitted changes"* ]] && (( $(fcalls) == 0 )) && ok "final_run.sh refuses a dirty recipe tree" || bad "final_run on a dirty tree: rc=$rc $out"
git -C "$ft" checkout -q -- scripts/memguard.sh
# the pin: an installed engine that is not REXL3_COMMIT is refused after the build, before any test or boot
freset
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 STUB_ENGINE=0123456789abcdef0123456789abcdef01234567 -- 2>&1); rc=$?
if (( rc != 0 )) && [[ $out == *"not the pin $fsha: refusing to run"* && $out == *"FATAL"* && $out == *"NOT RUN engine-build"* ]] \
  && ! grep -l 'engine-test \$WHICH' "$fdir"/call.* >/dev/null 2>&1 && ! grep -l 'export CMD=' "$fdir"/call.* >/dev/null 2>&1; then
  ok "final_run.sh refuses an installed engine that is not the pin: no engine test, no boot"
else bad "final_run with a foreign installed engine: rc=$rc $out"; fi
grep -l 'kill -TERM' "$fdir"/call.* >/dev/null 2>&1 && ok "final_run.sh stops the server on its way out (also after a refusal)" || bad "final_run did not run the controller's stop on exit"
# one failing gate: the pin is installed, the stub node runs no tests, so engine-build FAILs; the run stops there and says so
freset
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 STUB_ENGINE="$fsha" READY_TIMEOUT_S=3 -- 2>&1); rc=$?
recdir=$(ls -d "$ft"/bench/records/*-final-engine-tests 2>/dev/null | head -n 1)
if (( rc == 1 )) && [[ $out == *"FAIL engine-build"* && $out == *"NOT RUN boot"* && $out == *"NOT RUN pin"* && $out == *"stopping (--continue runs the rest)"* && $out == *"0 PASS, 1 FAIL, 18 NOT RUN of 19 gates"* ]] \
  && [[ -n $recdir ]] && python3 - "$recdir/meta.json" "$fsha" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["gate"] == "engine-build" and d["result"] == "fail" and d["node"] == "node-3", d
assert d["engine"]["commit"] == sys.argv[2] and d["engine"]["lab_build"] is False and d["engine"]["dirty"] is False
assert "0 suites reported" in d["notes"] or "suites reported" in d["notes"], d["notes"]
assert "date" in d and d["date"].endswith("Z") and len(d["date"]) == 20
PY
then ok "final_run.sh: a failing gate is recorded result=fail with its cause, the run stops at it, the rest is NOT RUN, exit 1"
else bad "final_run with a failing gate: rc=$rc recdir=$recdir $out"; fi
[[ -f $recdir/raw/cap-proof.txt ]] && grep -q 'gpuClkMin 2200, gpuClkMax 2200' "$recdir/raw/cap-proof.txt" && grep -q '^engine-build FAIL$' "$ft"/bench/runs/final-*/verdicts \
  && ok "the failing record keeps the GPU cap proof line of the node and the run keeps its verdict" || bad "no cap proof or verdict kept: $(ls "$recdir/raw" 2>/dev/null | tr '\n' ' ')"
# a record builder that exits 0 and says nothing proves nothing: the gate is FAIL
freset
printf 'import sys\nsys.exit(0)\n' > "$ft/bench/tools/final_records.py"; git -C "$ft" -c user.name=t -c user.email=t@t commit -q -am silent
out=$(frun NODES=203.0.113.7 NODE_NAME=node-3 STUB_ENGINE="$fsha" READY_TIMEOUT_S=3 -- 2>&1); rc=$?
git -C "$ft" reset -q --hard HEAD~1
(( rc == 1 )) && [[ $out == *"FAIL engine-build"* ]] && ok "final_run.sh counts a silent record builder as FAIL (PASS needs the builder's own 'gate: pass' line)" || bad "final_run with a silent builder: rc=$rc $out"
! grep -rIE '(192\.168|10\.100)\.[0-9]+\.[0-9]+' "$recdir" >/dev/null 2>&1 && ok "the record carries no lab address" || bad "a lab address in $recdir"
# the script itself: node work only through the controller; no clock command of its own; the stop is a trap on EXIT
if ! grep -nE '^[^#]*\b(ssh|scp|rsync|nvidia-smi)\b' scripts/final_run.sh | grep -v '^[0-9]*:\s*#' >/dev/null && grep -q '^trap finish EXIT$' scripts/final_run.sh \
  && grep -q '"\$CTL" stop' scripts/final_run.sh && grep -q 'check_pin' scripts/final_run.sh; then
  ok "final_run.sh touches the node only through the controller, stops the server in an EXIT trap, checks the pin"
else bad "final_run.sh: a node command of its own, no EXIT trap or no pin check"; fi
# the controller phases the final run uses, and the Small's drafter golden reading the Small's pack
for ph in exactness cache-images decode-4k; do
  out=$(ctl -- bench "$ph" 2>&1); rc=$?
  (( rc != 0 )) && [[ $out == *"no node"* && $out != *"usage: bench"* ]] && ok "bench $ph is a phase (refused for want of a node, not as unknown)" || bad "bench $ph: rc=$rc $out"
done
grep -q 'REXL3_INKLING2_DIR="\$SMALL_DIR" suite' inkling2mini.sh && ! grep -q 'REXL3_INKLING2_DIR="\$MODEL_DIR"' inkling2mini.sh \
  && grep -q 'golden dspark needs the Small pack' inkling2mini.sh && ok "engine-test goldens runs the Small's drafter hash test on the Small's pack (fails when it is missing)" || bad "engine-test goldens: the dspark golden's pack"
grep -q 'd\["meta"\].pop("url", None)' inkling2mini.sh && ok "bench pin drops the node address from reference-vision.json" || bad "bench pin keeps the url in reference-vision.json"

# ── memory guard logic (fake meminfo; local processes) ──
meminfo() { printf 'MemTotal: 125000000 kB\nMemAvailable: %d kB\nSwapTotal: 16777216 kB\nSwapFree: %d kB\n' \
  "$(( $1 * 1024 ))" "$(( 16777216 - $2 * 1024 ))" > "$work/meminfo.new"; mv "$work/meminfo.new" "$work/meminfo"; }
# shellcheck disable=SC2034  # gpid and rpid are read by the wait_for conditions (eval)
guard() {   # guard NAME: start a guard over a fresh `sleep` server; sets gpid, rpid
  local n=$1
  : > "$work/$n.serve.log"; echo loading > "$work/$n.stage"
  MEMINFO=$work/meminfo GUARD_PIDFILE=$work/$n.pid GUARD_LOG=$work/$n.guard.log GUARD_MIN_FILE=$work/$n.min \
    GUARD_STAGE_FILE=$work/$n.stage GUARD_WINDOW_FILE=$work/$n.window GUARD_SERVE_LOG=$work/$n.serve.log \
    GUARD_START_MIB=2048 GUARD_MIB=1024 GUARD_SWAP_MAX_MIB=64 DROP_CACHES_EVERY_S=0 GUARD_SAMPLE_S=0.05 \
    "$BASH_BIN" scripts/memguard.sh & gpid=$!
  rpid=$( (sleep 300 >/dev/null 2>&1 & echo $!) )   # not a job of this shell: no "Killed" chatter
  echo "$rpid" > "$work/$n.pid"
}
wait_for() { local i; for (( i = 0; i < 100; i++ )); do eval "$1" && return 0; sleep 0.05; done; return 1; }
get() { sed -n "s/^$2=//p" "$work/$1.min"; }

meminfo 60000 100
guard low
wait_for 'grep -q " armed pid=$rpid " "$work/low.guard.log" 2>/dev/null' && ok "guard arms on the server's pid file" || bad "guard did not arm"
meminfo 1500 100   # above the ready floor, below the loading floor: loading kills
wait_for '! kill -0 $rpid 2>/dev/null' && ok "guard SIGKILLs the server below the loading floor" || bad "guard did not kill below the floor"
wait_for '! kill -0 $gpid 2>/dev/null' && ok "guard exits after its server" || bad "guard outlived its server"
grep -q 'level=fatal event=memguard_kill' "$work/low.serve.log" && ok "the kill reason is a level=fatal line in the server log" || bad "no level=fatal line in the server log"
[[ $(get low min_mib) == 1500 && $(get low loading_min_mib) == 1500 ]] && ok "guard records the minimum (1500 MiB)" || bad "guard minimum: $(get low min_mib)"

meminfo 60000 100
guard ready
echo ready > "$work/ready.stage"
wait_for 'grep -q " armed " "$work/ready.guard.log" 2>/dev/null' || bad "guard did not arm (ready)"
meminfo 1500 100   # ready floor 1024: survives
sleep 0.5
kill -0 "$rpid" 2>/dev/null && ok "once ready, 1500 MiB is above the floor" || bad "guard killed above the ready floor"
meminfo 3000 100; sleep 0.3   # the guard has sampled 3000 before the window opens (a sample between two writes would keep 1500)
echo w1 > "$work/ready.window"
wait_for '[[ $(get ready window) == w1 && $(get ready window_min_mib) == 3000 ]]' \
  && ok "memwatch window restarts its minimum (3000 MiB, overall $(get ready min_mib))" || bad "window minimum: $(get ready window_min_mib)"
[[ $(get ready ready_min_mib) == 1500 ]] && ok "ready minimum recorded" || bad "ready minimum: $(get ready ready_min_mib)"
meminfo 3000 200   # swap +100 MiB > 64
wait_for '! kill -0 $rpid 2>/dev/null' && ok "guard SIGKILLs the server on swap growth" || bad "guard ignored swap growth"
grep -q 'swap grew 100 MiB' "$work/ready.guard.log" && ok "the guard log says why" || bad "no swap reason in the guard log"
wait_for '! kill -0 $gpid 2>/dev/null' || bad "guard outlived its server (swap)"

if GUARD_PIDFILE=$work/x GUARD_LOG=$work/x.log GUARD_MIN_FILE=$work/x.min GUARD_STAGE_FILE=$work/x.s \
  GUARD_WINDOW_FILE=$work/x.w GUARD_SERVE_LOG=$work/x.serve GUARD_START_MIB=2048 GUARD_MIB=0 GUARD_SWAP_MAX_MIB=64 \
  DROP_CACHES_EVERY_S=0 MEMINFO=$work/meminfo "$BASH_BIN" scripts/memguard.sh 2>/dev/null; then bad "guard accepted a 0 MiB floor"
else ok "guard refuses a 0 MiB floor"; fi
if GUARD_PIDFILE=$work/x GUARD_LOG=$work/x.log GUARD_MIN_FILE=$work/x.min GUARD_STAGE_FILE=$work/x.s \
  GUARD_WINDOW_FILE=$work/x.w GUARD_SERVE_LOG=$work/x.serve GUARD_START_MIB=2048 GUARD_MIB=1024 GUARD_SWAP_MAX_MIB=64 \
  DROP_CACHES_EVERY_S=20 GUARD_DROP_CMD=false MEMINFO=$work/meminfo "$BASH_BIN" scripts/memguard.sh 2>/dev/null; then
  bad "guard started without page-cache drop permission"
else ok "guard refuses to start when it cannot drop the page cache"; fi

echo
echo "test_recipe: $passed passed, $fails failed"
(( fails == 0 ))
