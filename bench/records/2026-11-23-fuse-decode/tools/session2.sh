#!/usr/bin/env bash
# shipping-defaults session of the final engine: start, exactness record, bench decode x4, specbench, specedge, gates spec/cache, bench.py --ref, stop
TAG=$1; O=/tmp/fuse/$TAG; mkdir -p $O; cd $HOME/projects/inkling2-mini-k-fuse
./inkling2mini.sh start > $O/start.log 2>&1 || { tail -5 $O/start.log; exit 1; }
python3 bench/tools/exactness.py record $O/exactness.json --max-tokens 128 --label "$TAG" > $O/exactness.out 2>&1
python3 bench/tools/exactness.py diff /tmp/fuse/base2/exactness.json $O/exactness.json > $O/exactness-diff.out 2>&1
python3 bench/tools/exactness.py concurrent --max-tokens 64 > $O/exactness-concurrent.out 2>&1
./inkling2mini.sh bench decode > $O/bench-decode-warm.log 2>&1
for i in 1 2 3; do ./inkling2mini.sh bench decode > $O/bench-decode-$i.log 2>&1; done
python3 bench/tools/specbench.py --out $O/specbench.json --label "$TAG" --concurrency 4 > $O/specbench.out 2> $O/specbench.err
python3 bench/tools/specedge.py --out $O/specedge.json --label "$TAG" > $O/specedge.out 2>&1
python3 bench/tools/gates.py spec > $O/gates-spec.out 2>&1
python3 bench/tools/gates.py cache > $O/gates-cache.out 2>&1
python3 bench/tools/bench.py --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref bench/records/2026-11-23-graph-exact/raw/pinned-replies-engine-51f2993.json > $O/bench-ref.out 2>&1
curl -s http://node-2:8000/metrics | grep -E "^rexl3_(graph|spec)" > $O/metrics.txt 2>&1
./inkling2mini.sh stop > $O/stop.log 2>&1
echo done
