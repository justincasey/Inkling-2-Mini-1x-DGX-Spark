#!/usr/bin/env bash
# the final measurement + exactness battery on one fresh server of the engine under test
set -u
cd $HOME/projects/inkling2-mini-k-round
U=http://node-4:8000
O=/tmp/rnd/final; mkdir -p $O
./inkling2mini.sh start 2>&1 | tail -1
# 1. a fresh server: prose, code, prose (specmix)
python3 bench/tools/specmix.py --url $U --out $O/specmix.json --mix chat,code,chat --per-phase 8 --max-tokens 256 > $O/specmix.out 2>&1; tail -4 $O/specmix.out
# 2. bench decode x3 (the number)
for i in 1 2 3; do ./inkling2mini.sh bench decode > $O/decode-$i.txt 2>&1; grep -E "METRIC decode_c1|^decode (prose|code|structured|json)" $O/decode-$i.txt | cut -c1-120; done
# 3. specbench: all kinds + c=4 (the mixed run)
python3 bench/tools/specbench.py --url $U --out $O/specbench.json --label final --concurrency 4 > $O/specbench.md 2> $O/specbench.err; grep -E "^(chat|code|json|longctx|tool|reasoning):|speedup|PASS|FAIL" $O/specbench.err $O/specbench.md | tail -12
# 4. homogeneous c=4
python3 bench/tools/homo_c4.py --url $U --out $O/homo.json 2>&1 | tail -6
# 5. exactness
python3 bench/tools/exactness.py record $O/exact.json --max-tokens 128 --label final > $O/exact-record.out 2>&1; tail -1 $O/exact-record.out
python3 bench/tools/exactness.py diff $HOME/projects/inkling2-mini-k-round/bench/records/2026-11-23-graph-exact/raw/graph-9fa93ea/exact.json $O/exact.json 2>&1 | tail -2
python3 -c "
import json,hashlib
d=json.load(open('$O/exact.json'))['replies']
print('replies sha256',hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest())"
python3 bench/tools/exactness.py concurrent --max-tokens 64 > $O/concurrent.out 2>&1; tail -2 $O/concurrent.out
python3 bench/tools/bench.py --url $U --model inkling-2-mini-1x-dgx-spark --max-model-len 262144 --phases warm,decode,conc,prefill-32k --ref $HOME/projects/inkling2-mini-k-round/bench/records/2026-11-23-graph-exact/raw/pinned-replies-engine-51f2993.json > $O/bench-ref.out 2> $O/bench-ref.err; grep -E "PASS|FAIL" $O/bench-ref.out $O/bench-ref.err | tail -3
python3 bench/tools/specedge.py --url $U --out $O/specedge.json > $O/specedge.out 2>&1; tail -3 $O/specedge.out
python3 bench/tools/gates.py spec $O/gates-spec.json --max-tokens 128 > $O/gates-spec.out 2>&1; tail -3 $O/gates-spec.out
python3 bench/tools/gates.py cache $O/gates-cache.json --max-tokens 64 > $O/gates-cache.out 2>&1; tail -3 $O/gates-cache.out
./inkling2mini.sh stop 2>&1 | tail -1
echo ALLDONE
