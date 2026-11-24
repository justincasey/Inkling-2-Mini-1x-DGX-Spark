#!/usr/bin/env bash
# Runs the tests that printed "not run" in the record 2026-11-24-engine-gpu-tests-pin, with their real fixtures, at the pinned engine commit.
# Needs NODE (the address of node-3). Fixtures live under $FIX (copied to the lab tree) or in place on node-3.
set -u
cd "$(dirname "$0")/../../.."
R=$(dirname "$0")/raw
LABN=/home/spark/inkling-2-mini-1x-dgx-spark/lab
FIX=$LABN/rexl3-inkling2-loop-pintests/fix
SMALL=REXL3_INKLING2_DIR=/home/spark/models/Inkling-2-Small-EXL3-4bpw
export NODE=${NODE:?address of node-3} LAB_SRC=$HOME/projects/rexl3-inkling2-loop LANE=rexl3-inkling2-loop-pintests OPS_GREP=. OPS_LINES=5000
t() { out=$1; env=$2; pkg=$3; bin=$4; shift 4; TEST_ENV="$env" PKG=$pkg scripts/ops_dev.sh test "$bin" "$@" > "$R/$out" 2>&1; echo "$out rc=$?"; }
t attn_oracle-real.txt "I2L_ATTN_REAL=$LABN/rexl3-inkling2-loop-attn/real/attn_real.bin" gb-cuda inkling2_attn_oracle real::
t ops-oracle.txt "$SMALL REXL3_INKLING2_ORACLE=$FIX/ops-small" gb-cuda inkling2_ops ops_equal_the_oracle_on_the_real_weights
t dspark-small-oracle.txt "$SMALL REXL3_INKLING2_DSPARK_ORACLE=/home/spark/inkling2-dspark-fixtures" gb-cuda inkling2_dspark matches_the_oracle_on_one_rank matches_the_oracle_at_tp4
t dspark-mini-oracle.txt "REXL3_INKLING2_LOOP_DSPARK_ORACLE=$FIX/dspark-mini" gb-cuda inkling2_dspark mini_matches_the_oracle slim_picks_the_full_heads_winner
t moe-mixed-bench.txt "REXL3_MOE_BENCH=1" gb-cuda inkling2_moe_mixed bench_decode_cost_of_bands bench_prefill_layers
