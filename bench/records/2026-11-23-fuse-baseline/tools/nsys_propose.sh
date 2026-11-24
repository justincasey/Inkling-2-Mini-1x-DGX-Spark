#!/usr/bin/env bash
# nsys_propose.sh NAME [ENV...]: profile the drafter's proposals (inkling2_loop_spec_timing, LOOP_TIMING_ONLY=propose, 4,096 positions) of the lane's
# engine worktree on node-2 under the GPU lock; leaves NAME.sqlite in the lane's ids dir. NODE and LANE as in nsys_run.sh.
NAME=$1; shift
LANE=${LANE:-rexl3-inkling2-loop-fuse}
"$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/gpu_cap.sh" "$NODE" || exit 1   # the GPU cap first: hardware protection, no GPU work without it
ssh -o BatchMode=yes spark@"$NODE" "set -e; B=/home/spark/inkling-2-mini-1x-dgx-spark/lab; LAB=\$B/$LANE; D=\$B/fuse-ids; cd \$LAB/src; BIN=\$(cat \$LAB/bin.path); env REXL3_INKLING2_LOOP_DIR=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw LOOP_LONG_IDS_FILE=\$D/ids_long.json LOOP_TIMING=1 LOOP_TIMING_ONLY=propose LOOP_TIMING_CTX=4096 $* flock -x -n /tmp/rexl3-gpu.lock nsys profile -t cuda --cuda-graph-trace=node -o \$D/$NAME --force-overwrite true \$BIN round_cost_table --test-threads=1 --nocapture 2>&1 | grep -E 'LOOP_TIMING|rror'; nsys export --type sqlite -o \$D/$NAME.sqlite --force-overwrite true \$D/$NAME.nsys-rep >/dev/null 2>&1; ls -la \$D/$NAME.sqlite"
