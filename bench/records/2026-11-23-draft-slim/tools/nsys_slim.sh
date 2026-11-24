#!/usr/bin/env bash
# nsys_slim.sh NAME [ENV=V...]: profile the slim_proposal_timing test of the last ops_dev build (lane rexl3-inkling2-loop-slim) under the GPU lock
NAME=$1; shift
scripts/gpu_cap.sh node-1 > /dev/null || exit 1
ssh -o BatchMode=yes spark@node-1 "set -e; B=/home/spark/inkling-2-mini-1x-dgx-spark/lab; LAB=\$B/rexl3-inkling2-loop-slim; mkdir -p \$B/slim-prof; cd \$LAB/src; BIN=\$(cat \$LAB/bin.path); env REXL3_REQUIRE_GPU=1 REXL3_INKLING2_LOOP_DIR=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw $* flock -x -n /tmp/rexl3-gpu.lock nsys profile -t cuda -o \$B/slim-prof/$NAME --force-overwrite true \$BIN ${TESTNAME:-slim_proposal_timing} --test-threads=1 --nocapture 2>&1 | grep -E 'SLIM|rror' ; nsys export --type sqlite -o \$B/slim-prof/$NAME.sqlite --force-overwrite true \$B/slim-prof/$NAME.nsys-rep >/dev/null 2>&1; ls -la \$B/slim-prof/$NAME.sqlite"
