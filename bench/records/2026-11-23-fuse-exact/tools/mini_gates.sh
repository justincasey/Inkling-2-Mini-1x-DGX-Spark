#!/usr/bin/env bash
# mini_gates.sh OUTDIR: the engine's byte-identity gates of the Mini on node-2 (lab tests through scripts/ops_dev.sh), one log a gate.
# NODE (address) and LAB_SRC (the engine worktree) come from the environment; run it from the recipe worktree. Prints one line per gate:
# `gate NAME done`, then gate_summary.py's lines `GATE name passed failed` and the hashes the rank's gates print.
OUT=${1:?outdir}; mkdir -p "$OUT"
export OPS_GREP="LOOP_GRAPH_equal|sha256"
D=/home/spark/inkling-2-mini-1x-dgx-spark/lab/fuse-ids
IDS="LOOP_LONG_IDS_FILE=$D/ids_long.json"
run() { # name pkg bin env filters...
  local name=$1 pkg=$2 bin=$3 env=$4; shift 4
  PKG=$pkg TEST_ENV="$env" OPS_LINES=400 scripts/ops_dev.sh test "$bin" "$@" > "$OUT/$name.log" 2>&1
  echo "gate $name done"
}
run rank-graphs-on gb-model inkling2_loop_rank "$IDS" forward_oracle_gate long_context_gate
run rank-graphs-off gb-model inkling2_loop_rank "$IDS REXL3_INKLING2_LOOP_GRAPHS=0" forward_oracle_gate long_context_gate
run graphs gb-model inkling2_loop_graphs "$IDS" graph_replay_equals capture_refusal
run spec gb-model inkling2_loop_spec "$IDS" _
run prefix gb-model inkling2_loop_prefix "$IDS" _
run dspark-mini gb-cuda inkling2_dspark "" mini_
run attn gb-cuda inkling2_attn "" mini:: small::
run ops gb-cuda inkling2_ops "" split_topk sharded_argmax unembed_softcap golden
run moe gb-cuda inkling2_moe "" golden graph
run moe-mixed gb-cuda inkling2_moe_mixed "" graph
python3 "$(dirname "$0")/gate_summary.py" "$OUT"
