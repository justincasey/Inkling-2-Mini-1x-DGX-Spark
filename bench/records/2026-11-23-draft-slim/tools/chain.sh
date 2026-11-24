#!/usr/bin/env bash
# chain.sh REV: the confirm session of the final engine commit REV (capped): a lab build of it, then one `full` session (tag confirm), then the
# engine's byte-identity gates on it (tools/mini_gates.sh, capped). The speed and exactness records are made from these, so both name REV.
REV=$1
export NODES=node-1 NODE=node-1 REXL3_SRC=$HOME/projects/rexl3-inkling2-loop-slim LAB_SRC=$HOME/projects/rexl3-inkling2-loop-slim
cd $HOME/projects/inkling2-mini-k-slim
T=bench/records/2026-11-23-draft-slim/tools
while ssh -o BatchMode=yes spark@$NODES 'flock -n /tmp/rexl3-gpu.lock true' 2>/dev/null; [ $? -ne 0 ]; do sleep 20; done
./inkling2mini.sh build $REV > /tmp/slim/runs/build-confirm.log 2>&1 || { echo build confirm failed; exit 1; }
$T/session.sh confirm full
scripts/gpu_cap.sh $NODES > /tmp/slim/gpu-cap-gates-final.txt 2>&1 || { echo "the GPU cap did not apply"; exit 1; }
OPS_GREP="LOOP_GRAPH_equal|sha256|SLIM_AGREE real cases" $T/mini_gates.sh /tmp/slim/gates-final2 > /tmp/slim/gates-final2.out 2>&1
echo chain done
