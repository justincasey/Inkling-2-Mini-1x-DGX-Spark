#!/bin/bash
# logits_check.sh: the byte-for-byte logits check of a candidate engine against the pinned one (one node, TP1). Three cold boots, one
# cold N-id prefill each (N=9292, prefix cache off): the pinned engine (REXL3_COMMIT of config/pins.sh), the HEAD of CANDIDATE_SRC with
# the full dump, and the HEAD with REXL3_INKLING2_DUMP_LOGITS_ONLY=1. Every dumped logits file and chunk record of the candidate boots is
# compared with `cmp` against the pinned engine's, by relative path: all must be identical, and the pinned boot must have dumped at
# least one logits file. The dump variables (REXL3_INKLING2_DUMP, REXL3_INKLING2_DUMP_LOGITS_ONLY) are the inkling2 family's, passed
# through EXTRA_ENV; confirm they are unchanged in the inkling2_loop family before relying on a verdict ("no logits file" fails the check).
# It builds, boots and stops the engine: run it under `scripts/safeloop.sh cmd`. The node is NODES of the environment or .env.
set -u
cd "$(dirname "$0")/.." || exit 2
export CANDIDATE_SRC=${CANDIDATE_SRC:-$HOME/projects/rexl3-inkling2-loop}
export REXL3_SRC=$CANDIDATE_SRC
N=${N:-9292}
setting() {   # KEY from the environment, else .env, else .env.example
  local k=$1 f line v
  if [[ -n ${!k:-} ]]; then printf '%s' "${!k}"; return 0; fi
  for f in .env .env.example; do
    [[ -f $f ]] || continue
    line=$(grep -E "^$k=" "$f" | head -n 1 || true)
    [[ -n $line ]] || continue
    v=${line#*=}; v=${v%%#*}
    printf '%s' "$(echo "$v" | xargs)"; return 0
  done
}
HOST=$(setting NODES | awk '{print $1}')
SSH_USER=$(setting SSH_USER); SSH_USER=${SSH_USER:-spark}
[[ -n $HOST ]] || { echo "logits_check: no node: set NODES (environment or .env)" >&2; exit 2; }
PINNED=$(sed -n 's/^REXL3_COMMIT=\([0-9a-f]\{40\}\).*/\1/p' config/pins.sh)
[[ -n $PINNED ]] || { echo "logits_check: REXL3_COMMIT is not pinned in config/pins.sh" >&2; exit 2; }
BASE=$(setting BASE_DIR)   # the controller's default install directory on the node when .env sets none
[[ -n $BASE ]] || BASE=$(ssh -o BatchMode=yes "$SSH_USER@$HOST" "ls -d /home/$SSH_USER/inkling-2-mini-*/ | head -n 1") || exit 2
BASE=${BASE%/}
D=/home/$SSH_USER/plp
node() { ssh -o BatchMode=yes "$SSH_USER@$HOST" "$@"; }
node "rm -rf $D-A $D-Bfull $D-Blanes"
FIN=$(git -C "$CANDIDATE_SRC" rev-parse HEAD)
run() {  # name sha extra_env
  local name=$1 sha=$2 xenv=$3
  ./inkling2mini.sh stop >/dev/null 2>&1
  ./inkling2mini.sh build "$sha" 2>&1 | tail -1
  echo "  $name: crates compiled by this build: $(node "grep -c 'Compiling gb-' $BASE/logs/build.log") (gb-cuda: $(node "grep -c 'Compiling gb-cuda' $BASE/logs/build.log"))"
  EXTRA_ENV="REXL3_INKLING2_DUMP=$D-$name $xenv" ./inkling2mini.sh start 2>&1 | grep -E 'boot_s=|FAIL|fatal'
  python3 scripts/plp_req.py "$N" 2>&1 | tail -1
}
run A "$PINNED" ""
run Bfull "$FIN" ""
run Blanes "$FIN" "REXL3_INKLING2_DUMP_LOGITS_ONLY=1"
./inkling2mini.sh stop 2>&1 | tail -1
out=$(node 'cd '"$D"'-A && find . -type f \( -name "*.logits.f32" -o -name "*.json" \) | sort | while read -r f; do for v in Bfull Blanes; do if cmp -s "$f" ../plp-$v/"$f"; then echo "same $v $f"; else echo "DIFF $v $f"; fi; done; done')
logits=$(node "cd $D-A && find . -type f -name '*.logits.f32' | wc -l")
echo "== $(echo "$out" | grep -c '^same') same, $(echo "$out" | grep -c '^DIFF') differ; the pinned boot dumped $logits logits file(s)"
echo "$out" | grep '^DIFF' | head -3
node "rm -rf $D-A $D-Bfull $D-Blanes"
[[ $logits -gt 0 ]] && ! echo "$out" | grep -q '^DIFF' && [[ -n $out ]]
