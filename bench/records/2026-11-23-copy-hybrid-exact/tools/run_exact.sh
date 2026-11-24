#!/usr/bin/env bash
# the exactness battery on one fresh server of the engine under test (shipping defaults; the copy hybrid at the engine's default)
# usage: run_exact.sh OUTDIR     (node-4's address in NODES; the engine is built beforehand)
set -u
O=$1; mkdir -p "$O"
cd "$(dirname "$0")/../../../.."
bash scripts/gpu_cap.sh "$NODES"
U=http://$NODES:8000
G=bench/records/2026-11-23-graph-exact
./inkling2mini.sh start 2>&1 | tail -1
python3 bench/tools/exactness.py record "$O/exact.json" --max-tokens 128 --label copy-hybrid > "$O/exact-record.out" 2>&1; tail -1 "$O/exact-record.out"
python3 bench/tools/exactness.py diff $G/raw/graph-9fa93ea/exact.json "$O/exact.json" 2>&1 | tail -2
python3 -c "
import json,hashlib
d=json.load(open('$O/exact.json'))['replies']
print('replies sha256',hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest())" | tee "$O/exact-sha.txt"
python3 bench/tools/exactness.py concurrent --max-tokens 64 > "$O/concurrent.out" 2>&1; tail -2 "$O/concurrent.out"
python3 bench/tools/bench.py --url "$U" --model inkling-2-mini-1x-dgx-spark --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref $G/raw/pinned-replies-engine-51f2993.json > "$O/bench-ref.out" 2> "$O/bench-ref.err"; grep -E "PASS|FAIL" "$O/bench-ref.out" "$O/bench-ref.err" | tail -3
python3 bench/tools/specedge.py --url "$U" --out "$O/specedge.json" > "$O/specedge.out" 2>&1; tail -3 "$O/specedge.out"
python3 bench/tools/gates.py spec "$O/gates-spec.json" --max-tokens 128 > "$O/gates-spec.out" 2>&1; tail -3 "$O/gates-spec.out"
python3 bench/tools/gates.py cache "$O/gates-cache.json" --max-tokens 64 > "$O/gates-cache.out" 2>&1; tail -3 "$O/gates-cache.out"
curl -s "$U/metrics" | grep -E "^rexl3_spec_copy" > "$O/copy-metrics.txt"
./inkling2mini.sh stop 2>&1 | tail -1
echo ALLDONE
