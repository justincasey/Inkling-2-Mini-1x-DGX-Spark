#!/usr/bin/env bash
# nsys_pf.sh NAME [ENV...]: profile the prefill_curve test of the lab build on node-3 under `nsys profile -t cuda` and export the sqlite.
# Env words (K=V) go to the test binary: PF_IDS_FILE (default the lab's pf_ids.json), PF_TOKENS, PF_CHUNK. The GPU lock is taken by the command.
NAME=$1; shift
NODE_ADDR=${NODE_ADDR:-$(sed -n "s/^NODES=//p" "$(dirname "$0")/../../../../.env")}
"$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/gpu_cap.sh" "$NODE_ADDR" || exit 1   # the GPU cap first: hardware protection, no GPU work without it
ssh -o BatchMode=yes spark@$NODE_ADDR "set -e; B=/home/spark/inkling-2-mini-1x-dgx-spark/lab; LAB=\$B/rexl3-inkling2-loop-prefill; mkdir -p \$LAB/prof; cd \$LAB/src; BIN=\$(cat \$LAB/bin.path); env REXL3_REQUIRE_GPU=1 REXL3_INKLING2_LOOP_DIR=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw PF_IDS_FILE=\$LAB/pf_ids.json $* flock -x -n /tmp/rexl3-gpu.lock nsys profile -t cuda -o \$LAB/prof/$NAME --force-overwrite true \$BIN prefill_curve --test-threads=1 --nocapture 2>&1 | grep -E '^PF_|rror|panicked'; nsys export --type sqlite -o \$LAB/prof/$NAME.sqlite --force-overwrite true \$LAB/prof/$NAME.nsys-rep >/dev/null 2>&1; ls -la \$LAB/prof/$NAME.sqlite"
