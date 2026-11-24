#!/usr/bin/env bash
# Runs every GPU test binary of the Mini family at the pinned engine commit on node-3; one output file per binary.
set -u
cd "$(dirname "$0")/../../.."
R=$(dirname "$0")/raw
SMALL="REXL3_INKLING2_DIR=/home/spark/models/Inkling-2-Small-EXL3-4bpw"
export NODE=${NODE:?address of node-3} LAB_SRC=$HOME/projects/rexl3-inkling2-loop LANE=rexl3-inkling2-loop-pintests OPS_GREP=. OPS_LINES=5000
for b in inkling2_attn inkling2_attn_oracle inkling2_ops inkling2_dspark inkling2_moe inkling2_moe_mixed inkling2_loop_vision; do
  TEST_ENV="$SMALL" PKG=gb-cuda scripts/ops_dev.sh test "$b" "" > "$R/gb-cuda-$b.txt" 2>&1; echo "gb-cuda $b rc=$?"
done
for b in inkling2_vision inkling2_loop_vision; do
  TEST_ENV="$SMALL" PKG=gb-model scripts/ops_dev.sh test "$b" "" > "$R/gb-model-$b.txt" 2>&1; echo "gb-model $b rc=$?"
done
