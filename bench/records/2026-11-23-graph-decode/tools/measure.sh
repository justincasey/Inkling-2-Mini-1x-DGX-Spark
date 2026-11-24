#!/usr/bin/env bash
# measure.sh LABEL "K=V K2=V2" [sweep]: boot the installed engine with EXTRA_ENV, warm up, sweep decode (3 reps), save metrics, stop.
set -u
LABEL=$1; XENV=$2; MODE=${3:-sweep}
cd $HOME/projects/inkling2-mini-k-graph
OUT=/tmp/gl/$LABEL; mkdir -p $OUT
export EXTRA_ENV="REXL3_ALLOW_UNGATED_COMPONENTS=1 $XENV"
./inkling2mini.sh start > $OUT/start.log 2>&1 || { tail -20 $OUT/start.log; exit 1; }
URL=http://node-2:8000
python3 bench/tools/bench.py --help >/dev/null 2>&1
./inkling2mini.sh bench decode > $OUT/bench-decode.log 2>&1   # warm (discarded) + decode + conc
python3 bench/rexl3/sweep_decode.py --url $URL --model inkling-2-mini-1x-dgx-spark --window 262144 --max-total 1048576 --phase both --ctxs 4096 --conc 1,2,4 --modes serial --depth-modes serial --reps 3 --depth-reps 3 --out $OUT/sweep.json > $OUT/sweep.log 2>&1
curl -s $URL/metrics | grep -E "^rexl3_graph" > $OUT/graph-metrics.txt
./inkling2mini.sh logs 400 2>&1 | grep -E "inkling2_loop_(window_timing|graph)" > $OUT/serve-events.txt
./inkling2mini.sh stop > $OUT/stop.log 2>&1
python3 - <<PY
import json,statistics
r=json.load(open("$OUT/sweep.json"))
for ph in ("shallow","depth"):
    cells={}
    for x in r.get(ph,[]):
        if x.get("ok"): cells.setdefault(x["c"],[]).append(x["tok_s"])
    for c,v in sorted(cells.items()): print("$LABEL",ph,"c",c,"median %.3f"%statistics.median(v),["%.3f"%t for t in v])
PY
grep -E "METRIC decode_c" $OUT/bench-decode.log
