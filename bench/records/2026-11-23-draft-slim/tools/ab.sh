#!/usr/bin/env bash
# ab.sh BASE_REV SLIM_REV: the A/B of the DraftSlim record on node-1, every step under the 2,200 MHz cap (session.sh applies and checks it).
# base = the integration head; slim = the lane's final commit; slimoff = the same binary with REXL3_INKLING2_LOOP_DRAFT_SLIM=0.
# The engine's byte-identity gates (tools/mini_gates.sh) run first, on the final commit, before any server.
BASE=$1; SLIM=$2
export NODES=node-1 REXL3_SRC=$HOME/projects/rexl3-inkling2-loop-slim NODE=node-1 LAB_SRC=$HOME/projects/rexl3-inkling2-loop-slim
cd $HOME/projects/inkling2-mini-k-slim
T=bench/records/2026-11-23-draft-slim/tools
scripts/gpu_cap.sh $NODES > /tmp/slim/gpu-cap-gates.txt 2>&1 || { echo "the GPU cap did not apply"; exit 1; }
mkdir -p /tmp/slim/runs
OPS_GREP="LOOP_GRAPH_equal|sha256|SLIM_AGREE real cases" $T/mini_gates.sh /tmp/slim/gates-final > /tmp/slim/gates-final.out 2>&1
./inkling2mini.sh build $BASE > /tmp/slim/runs/build-base.log 2>&1 || { echo build base failed; exit 1; }
$T/session.sh base base
./inkling2mini.sh build $SLIM > /tmp/slim/runs/build-slim.log 2>&1 || { echo build slim failed; exit 1; }
$T/session.sh slim full
$T/session.sh slimoff base REXL3_INKLING2_LOOP_DRAFT_SLIM=0
echo ab done
