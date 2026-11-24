#!/usr/bin/env bash
# session.sh TAG: shipping-defaults server session: start, exactness record, bench decode (warm-up + measured), specbench, stop
TAG=$1; O=/tmp/fuse/$TAG; mkdir -p $O; cd $HOME/projects/inkling2-mini-k-fuse
./inkling2mini.sh start > $O/start.log 2>&1 || { tail -5 $O/start.log; exit 1; }
python3 bench/tools/exactness.py record $O/exactness.json --max-tokens 128 --label "$TAG" > $O/exactness.out 2>&1
./inkling2mini.sh bench decode > $O/bench-decode-warm.log 2>&1
./inkling2mini.sh bench decode > $O/bench-decode.log 2>&1
./inkling2mini.sh bench decode > $O/bench-decode-2.log 2>&1
./inkling2mini.sh bench decode > $O/bench-decode-3.log 2>&1
python3 bench/tools/specbench.py --out $O/specbench.json --label "$TAG" --concurrency 4 > $O/specbench.out 2> $O/specbench.err
curl -s http://node-2:8000/metrics | grep -E "^rexl3_(graph|spec)" > $O/metrics.txt 2>&1
./inkling2mini.sh stop > $O/stop.log 2>&1
grep -h "METRIC decode_c1_tok_s\|METRIC decode_.*tok_s" $O/bench-decode*.log | head -20
