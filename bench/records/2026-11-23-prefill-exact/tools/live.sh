#!/usr/bin/env bash
# live.sh: the live byte-exactness checks and the speed sweep of the final engine on the running server (node-3), from the recipe worktree.
# Writes raw/new-<engine>/ under the exact record and raw/sweep-final under the speed record.
set -u
R=$HOME/projects/inkling2-mini-k-prefill
cd "$R" || exit 1
URL=http://$(sed -n 's/^NODES=//p' .env):8000
E=bench/records/2026-11-23-prefill-exact
S=bench/records/2026-11-23-prefill-speed
mkdir -p "$E/raw/new" "$S/raw"
./inkling2mini.sh memwatch begin > "$E/raw/new/memwatch-begin.txt" 2>&1
python3 bench/tools/exactness.py record "$E/raw/new/exactness.json" --max-tokens 128 --label new --url "$URL" > "$E/raw/new/exactness-record.out" 2>&1
python3 bench/tools/exactness.py diff "$E/raw/base-7da8c61/exactness.json" "$E/raw/new/exactness.json" > "$E/raw/new/exactness-diff.out" 2>&1; echo "exactness diff exit $?" >> "$E/raw/new/exactness-diff.out"
python3 bench/tools/bench.py --url "$URL" --model inkling-2-mini-1x-dgx-spark --max-model-len 262144 --phases warm,decode,conc,prefill-32k \
  --ref "$E/raw/pinned-replies-engine-51f2993.json" --out "$E/raw/new/bench-ref.json" > "$E/raw/new/bench.out" 2> "$E/raw/new/bench.err"; echo "bench ref exit $?" >> "$E/raw/new/bench.out"
python3 bench/tools/exactness.py concurrent --max-tokens 64 --url "$URL" > "$E/raw/new/concurrent.out" 2>&1; echo "concurrent exit $?" >> "$E/raw/new/concurrent.out"
./inkling2mini.sh bench prefill-sweep > "$S/raw/sweep-final.out" 2>&1
cp -r "$(ls -d bench/runs/*-prefill-sweep | tail -1)" "$S/raw/sweep-final"
./inkling2mini.sh memwatch report > "$S/raw/memwatch-report.txt" 2>&1
echo done > "$S/raw/live-done.txt"
