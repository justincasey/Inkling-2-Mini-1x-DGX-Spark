#!/usr/bin/env bash
# gates.sh VARIANT OUTFILE: the byte-exactness gates of the prefill work on node-3, from the engine worktree (ENGINE) through scripts/ops_dev.sh.
# VARIANT new  = the engine tree as it is; base = the same tree with csrc/moe_prefill.cu of the integration head 7da8c61 (the only kernel file the
# work changed), restored afterwards. Every line the gates print that starts LOOP_ or PF_ or BENCH, and every `test result` line, goes to OUTFILE.
set -u
VARIANT=$1; OUT=$2
ENGINE=${ENGINE:-$HOME/projects/rexl3-inkling2-loop-prefill}
RECIPE=${RECIPE:-$HOME/projects/inkling2-mini-k-prefill}
NODE_ADDR=$(sed -n 's/^NODES=//p' "$RECIPE/.env")
LAB=/home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-prefill
export NODE=$NODE_ADDR LAB_SRC=$ENGINE
cd "$RECIPE" || exit 1
if [ "$VARIANT" = base ]; then
  cp "$ENGINE/crates/cuda/csrc/moe_prefill.cu" /tmp/moe_prefill.head
  git -C "$ENGINE" show 7da8c61:crates/cuda/csrc/moe_prefill.cu > "$ENGINE/crates/cuda/csrc/moe_prefill.cu"
fi
restore() { [ "$VARIANT" = base ] && cp /tmp/moe_prefill.head "$ENGINE/crates/cuda/csrc/moe_prefill.cu"; }
trap restore EXIT
run() { # PKG TESTBIN ENV GREP FILTER...
  local pkg=$1 bin=$2 tenv=$3 grep=$4; shift 4
  echo "### $VARIANT $pkg $bin $*" >> "$OUT"
  PKG=$pkg TEST_ENV="$tenv" OPS_GREP="$grep" OPS_LINES=400 scripts/ops_dev.sh test "$bin" "$@" 2>&1 | grep -E "^==|^test |test result|LOOP_|PF_|BENCH|did not|error" >> "$OUT"
}
: > "$OUT"
IDS="LOOP_LONG_IDS_FILE=$LAB/ids_long.json"
run gb-model inkling2_loop_rank "$IDS" 'LOOP_' forward_oracle_gate long_context_gate
run gb-model inkling2_loop_graphs "$IDS" 'LOOP_GRAPH' graph_replay_equals_eager capture_refusal_runs_eager_and_counts
run gb-model inkling2_loop_prefix "" 'LOOP_PFX' prefix_exact_at_cuts prefix_logits_planes_equal prefix_restore_after_eviction prefix_long_prompt
run gb-cuda inkling2_moe "" 'BENCH' partial_equals prefill_equals_decode mul1_prefill_equals prefill_with_tight four_rank_banks graph_replay_equals golden_inkling2
run gb-cuda inkling2_moe_mixed "" 'BENCH' mul1_bands_decode_equals_prefill band_split_equals every_pair_row graph_replay_equals band_entries_on bands_entries_refuse real_mini_layers
run gb-cuda inkling2_attn "" 'BENCH' "mini::" "small::"
run gb-cuda inkling2_ops "" 'BENCH' split_topk sharded_argmax unembed_softcap golden
run gb-cuda exl3_big "" 'BENCH' big_linear_equals_the_decode_path_on big_linear_equals_the_decode_path_for big_linear_rows the_big_plan
run gb-model inkling2_loop_prefill_perf "PF_IDS_FILE=$LAB/pf_ids.json PF_TOKENS=135168 PF_EVERY=1" 'PF_' prefill_curve
run gb-model inkling2_loop_prefill_perf "PF_IDS_FILE=$LAB/pf_ids.json PF_TOKENS=135168 PF_CHUNK=8192" 'PF_' prefill_curve
run gb-cuda inkling2_dspark "" 'BENCH' ""
echo "### done $VARIANT" >> "$OUT"
