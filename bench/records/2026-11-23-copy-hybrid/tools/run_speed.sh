#!/usr/bin/env bash
# one fresh server of the engine under test per configuration (REXL3_INKLING2_LOOP_SPEC_COPY=off|on), then, in this order:
# specmix (prose, code, structured, json, prose on a fresh server), bench decode x3, bench rolling x3, specbench with c=4.
# usage: run_speed.sh CONFIG OUTDIR      (node-4's address in NODES; the engine is built beforehand: ./inkling2mini.sh build REV)
set -u
CFG=$1; O=$2; mkdir -p "$O"
U=http://$NODES:8000
cd "$(dirname "$0")/../../../.."
bash scripts/gpu_cap.sh "$NODES"
EXTRA_ENV="REXL3_INKLING2_LOOP_SPEC_COPY=$CFG" ./inkling2mini.sh start 2>&1 | tail -2
python3 bench/tools/specmix.py --url "$U" --out "$O/specmix.json" --mix chat,code,reasoning,json,chat --per-phase 8 --max-tokens 256 > "$O/specmix.out" 2>&1; tail -3 "$O/specmix.out"
for i in 1 2 3; do ./inkling2mini.sh bench decode > "$O/decode-$i.txt" 2>&1; grep -E "METRIC decode_c1|METRIC decode_c4|^decode (prose|code|structured|json)" "$O/decode-$i.txt" | cut -c1-140; done
for i in 1 2 3; do ./inkling2mini.sh bench rolling > "$O/rolling-$i.txt" 2>&1; grep -E "METRIC decode_c4_(rolling|steady)" "$O/rolling-$i.txt"; done
python3 bench/tools/specbench.py --url "$U" --out "$O/specbench.json" --label "copy=$CFG" --concurrency 4 > "$O/specbench.md" 2> "$O/specbench.err"; tail -12 "$O/specbench.err"
curl -s "$U/metrics" | grep -E "^rexl3_spec_copy" > "$O/copy-metrics.txt"
./inkling2mini.sh stop 2>&1 | tail -1
echo ALLDONE
