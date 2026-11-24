#!/usr/bin/env bash
# session.sh TAG MODE [ENV=V ...]: one fresh server on node-1 with the shipping defaults (SPEC=dspark PREFIX_CACHE=on VISION=on), the GPU SM clock
# cap of 2,200 MHz re-applied (and its success line required) before the server starts, then
#   MODE=base   exactness record, bench decode (warm + 3), bench rolling (warm + 3), specbench --concurrency 4, stop
#   MODE=full   the same, then exactness diff against base, exactness concurrent, specedge, gates spec / cache, specmix, bench.py --ref, metrics, stop
# Output in /tmp/slim/runs/TAG (the record's raw/ is a copy). Needs the recipe worktree as cwd.
TAG=$1; MODE=$2; shift 2
O=/tmp/slim/runs/$TAG; mkdir -p $O
export NODES=node-1
[ $# -gt 0 ] && export EXTRA_ENV="$*"
scripts/gpu_cap.sh $NODES > $O/gpu-cap.txt 2>&1 || { echo "the GPU cap did not apply"; exit 1; }
date -u +%FT%TZ > $O/started.txt
./inkling2mini.sh start > $O/start.log 2>&1 || { tail -5 $O/start.log; exit 1; }
python3 bench/tools/exactness.py record $O/exactness.json --max-tokens 128 --label "$TAG" > $O/exactness.out 2>&1
./inkling2mini.sh bench decode > $O/bench-decode-warm.log 2>&1
for i in 1 2 3; do ./inkling2mini.sh bench decode > $O/bench-decode-$i.log 2>&1; done
./inkling2mini.sh bench rolling > $O/bench-rolling-warm.log 2>&1
for i in 1 2 3; do ./inkling2mini.sh bench rolling > $O/bench-rolling-$i.log 2>&1; done
python3 bench/tools/specbench.py --out $O/specbench.json --label "$TAG" --concurrency 4 > $O/specbench.out 2> $O/specbench.err
if [ "$MODE" = full ]; then
  python3 bench/tools/exactness.py diff /tmp/slim/runs/base/exactness.json $O/exactness.json > $O/exactness-diff.out 2>&1
  python3 bench/tools/exactness.py concurrent --max-tokens 64 > $O/exactness-concurrent.out 2>&1
  python3 bench/tools/specedge.py --out $O/specedge.json --label "$TAG" > $O/specedge.out 2>&1
  python3 bench/tools/gates.py spec $O/gates-spec.json --label "$TAG" > $O/gates-spec.out 2>&1
  python3 bench/tools/gates.py cache $O/gates-cache.json --label "$TAG" > $O/gates-cache.out 2>&1
  python3 bench/tools/specmix.py --out $O/specmix.json --label "$TAG" > $O/specmix.out 2>&1
  python3 bench/tools/bench.py --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref bench/records/2026-11-23-graph-exact/raw/pinned-replies-engine-51f2993.json > $O/bench-ref.out 2>&1
  curl -s http://$NODES:8000/metrics | grep -E "^rexl3_(graph|spec)" > $O/metrics.txt 2>&1
fi
ssh -o BatchMode=yes spark@$NODES 'nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader' > $O/gpu-clock-after.txt 2>&1
./inkling2mini.sh status > $O/status.txt 2>&1
./inkling2mini.sh stop > $O/stop.log 2>&1
date -u +%FT%TZ > $O/finished.txt
echo done
