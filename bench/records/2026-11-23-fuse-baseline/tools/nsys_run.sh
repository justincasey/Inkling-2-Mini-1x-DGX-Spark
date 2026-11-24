#!/usr/bin/env bash
# nsys_run.sh NAME ROWS [ENV...]: profile 20 replayed decode windows of ROWS rows (1..8) at ~3.8K of context with the lab test binary
# of the lane's engine worktree (node-2, the GPU lock is taken by the command), export the sqlite and leave it in the lane's ids dir.
# NODE (the node's address) and LANE (the lab tree, default rexl3-inkling2-loop-fuse) come from the environment.
NAME=$1; ROWS=$2; shift; shift
LANE=${LANE:-rexl3-inkling2-loop-fuse}
"$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/gpu_cap.sh" "$NODE" || exit 1   # the GPU cap first: hardware protection, no GPU work without it
ssh -o BatchMode=yes spark@"$NODE" "set -e; B=/home/spark/inkling-2-mini-1x-dgx-spark/lab; LAB=\$B/$LANE; D=\$B/fuse-ids; cd \$LAB/src; BIN=\$(cat \$LAB/bin.path); env REXL3_INKLING2_LOOP_DIR=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw LOOP_LONG_IDS_FILE=\$D/ids_long.json LOOP_PROFILE_STEPS=20 LOOP_PROFILE_ROWS=$ROWS LOOP_PROFILE_MODE=graph $* flock -x -n /tmp/rexl3-gpu.lock nsys profile -t cuda --cuda-graph-trace=node -o \$D/$NAME --force-overwrite true \$BIN decode_step_timing --test-threads=1 --nocapture 2>&1 | grep -E 'LOOP_GRAPH|rror'; nsys export --type sqlite -o \$D/$NAME.sqlite --force-overwrite true \$D/$NAME.nsys-rep >/dev/null 2>&1; ls -la \$D/$NAME.sqlite"
