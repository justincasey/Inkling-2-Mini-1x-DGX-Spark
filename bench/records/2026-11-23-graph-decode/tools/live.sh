#!/usr/bin/env bash
# live.sh LABEL MODE "ENV..." : start the installed engine, run the byte-exactness tools, stop.  MODE pin|ref
set -u
LABEL=$1; MODE=$2; XENV=$3
cd $HOME/projects/inkling2-mini-k-graph
OUT=/tmp/gl/live-$LABEL; mkdir -p $OUT
export EXTRA_ENV="REXL3_ALLOW_UNGATED_COMPONENTS=1 $XENV"
export INKLING2MINI_URL=http://node-2:8000
NAME=inkling-2-mini-1x-dgx-spark
./inkling2mini.sh start > $OUT/start.log 2>&1 || { tail -20 $OUT/start.log; exit 1; }
if [ "$MODE" = pin ]; then
  python3 bench/tools/bench.py --url $INKLING2MINI_URL --model $NAME --max-model-len 262144 --phases warm,decode,conc,prefill-32k --pin /tmp/gl/ref.json --engine "$LABEL" --out $OUT/bench.json > $OUT/bench.out 2> $OUT/bench.err
  echo "bench pin rc=$?"
else
  python3 bench/tools/bench.py --url $INKLING2MINI_URL --model $NAME --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref /tmp/gl/ref.json --out $OUT/bench.json > $OUT/bench.out 2> $OUT/bench.err
  echo "bench ref rc=$?"; grep -E "^(PASS|FAIL)" $OUT/bench.out
fi
tail -3 $OUT/bench.err
python3 bench/tools/exactness.py record $OUT/exact.json --max-tokens 128 --label "$LABEL" > $OUT/exact-record.out 2>&1; echo "exactness record rc=$?"
if [ "$MODE" = pin ]; then cp $OUT/exact.json /tmp/gl/exact-base.json; else
  python3 bench/tools/exactness.py diff /tmp/gl/exact-base.json $OUT/exact.json; echo "exactness diff rc=$?"
  python3 bench/tools/exactness.py concurrent --max-tokens 64 > $OUT/concurrent.out 2>&1; echo "exactness concurrent rc=$?"; tail -3 $OUT/concurrent.out
fi
curl -s $INKLING2MINI_URL/metrics | grep -E "^rexl3_graph" > $OUT/graph-metrics.txt
./inkling2mini.sh logs 300 2>&1 | grep -E "inkling2_loop_(window_timing|graph|plan)|level=(error|fatal)" | tail -12 > $OUT/events.txt
./inkling2mini.sh stop > $OUT/stop.log 2>&1
grep -c . $OUT/events.txt
