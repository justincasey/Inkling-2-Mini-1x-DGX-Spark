#!/usr/bin/env bash
# nsys_moe.sh NAME [ENV...]: the MoE layer bench (inkling2_moe_mixed::bench_prefill_layers, the last lab build of that test binary) under nsys on node-3;
# prints the kernel table (prof_pf.py's reader, every kernel launch of the run in one "chunk").
NAME=$1; shift
NODE_ADDR=${NODE_ADDR:-$(sed -n "s/^NODES=//p" "$(dirname "$0")/../../../../.env")}
"$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/gpu_cap.sh" "$NODE_ADDR" || exit 1   # the GPU cap first: hardware protection, no GPU work without it
ssh -o BatchMode=yes spark@$NODE_ADDR "set -e; B=/home/spark/inkling-2-mini-1x-dgx-spark/lab; LAB=\$B/rexl3-inkling2-loop-prefill; mkdir -p \$LAB/prof; cd \$LAB/src; BIN=\$(cat \$LAB/bin.path); env REXL3_REQUIRE_GPU=1 REXL3_INKLING2_LOOP_DIR=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw REXL3_MOE_BENCH=1 $* flock -x -n /tmp/rexl3-gpu.lock nsys profile -t cuda -o \$LAB/prof/$NAME --force-overwrite true \$BIN bench_prefill_layers --test-threads=1 --nocapture 2>&1 | grep -E 'BENCH_PF|rror|panicked' || true; nsys export --type sqlite -o \$LAB/prof/$NAME.sqlite --force-overwrite true \$LAB/prof/$NAME.nsys-rep >/dev/null 2>&1; python3 - <<'PY'
import sqlite3,collections,sys
c=sqlite3.connect('/home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-prefill/prof/$NAME.sqlite')
rows=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start').fetchall()
agg=collections.defaultdict(lambda:[0,0])
for r in rows:
    n=r[2].split('(const')[0].replace('(anonymous namespace)::','')[:100]
    agg[n][0]+=1; agg[n][1]+=r[1]-r[0]
for n,(k,d) in sorted(agg.items(),key=lambda x:-x[1][1])[:14]: print('%5d %9.3f ms  mean %8.1f us  %s'%(k,d/1e6,d/1e3/k,n))
PY"
