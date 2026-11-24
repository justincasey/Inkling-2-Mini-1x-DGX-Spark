#!/usr/bin/env bash
# lab_final.sh: after the A/B (the GPU idle), under the 2,200 MHz cap: the proposal's lab timings at 4,096 and 32,768 positions with DraftSlim off and on
# (1, 2, 4 requests), the agreement of DraftSlim with the full head on the real-feature cases, and an nsys timeline of one proposal.
export NODE=node-1 LAB_SRC=$HOME/projects/rexl3-inkling2-loop-slim NODES=node-1
cd $HOME/projects/inkling2-mini-k-slim
O=/tmp/slim/final-lab; mkdir -p $O
scripts/gpu_cap.sh $NODES > $O/gpu-cap.txt 2>&1 || { echo "the GPU cap did not apply"; exit 1; }
ORA=REXL3_INKLING2_LOOP_DSPARK_ORACLE=/home/spark/inkling-2-mini-1x-dgx-spark/fixtures/dspark
for ctx in 4096 32768; do
  OPS_GREP="SLIM_TIMING" OPS_LINES=20 TEST_ENV="SLIM_CTX=$ctx" scripts/ops_dev.sh test inkling2_dspark slim_proposal_timing > $O/timing-$ctx.log 2>&1
done
OPS_GREP="SLIM_AGREE" OPS_LINES=20 TEST_ENV="$ORA SLIM_STRIDE=1" scripts/ops_dev.sh run inkling2_dspark slim_picks_the_full_heads_winner > $O/agree.log 2>&1
bash bench/records/2026-11-23-draft-slim/tools/nsys_slim.sh final SLIM_CTX=4096 > $O/nsys.log 2>&1
scp -q spark@$NODES:/home/spark/inkling-2-mini-1x-dgx-spark/lab/slim-prof/final.sqlite /tmp/slim/final.sqlite
python3 bench/records/2026-11-23-draft-slim/tools/tl.py /tmp/slim/final.sqlite 28 > $O/timeline-proposal-final.txt 2>&1
echo done
