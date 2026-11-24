#!/usr/bin/env bash
# one fresh server per configuration (SPEC=none|dspark, REXL3_INKLING2_LOOP_AHEAD=0|1), then bench decode x3 and bench rolling x3
# usage: run_serial.sh SPEC AHEAD OUTDIR [TRACE]     (node-4's address in NODES; the engine is built beforehand)
set -u
SPECV=$1; AH=$2; O=$3; TR=${4:-0}; mkdir -p "$O"
cd "$(dirname "$0")/../../../.."
bash scripts/gpu_cap.sh "$NODES"
EXTRA="REXL3_INKLING2_LOOP_AHEAD=$AH"
[ "$TR" = 1 ] && EXTRA="$EXTRA REXL3_ROUND_TRACE=1"
SPEC=$SPECV EXTRA_ENV="$EXTRA" ./inkling2mini.sh start 2>&1 | tail -2
for i in 1 2 3; do ./inkling2mini.sh bench decode > "$O/decode-$i.txt" 2>&1; grep -E "METRIC decode_(c1|c4|prose|code|structured|json)_tok" "$O/decode-$i.txt" | tr '\n' ' '; echo; done
for i in 1 2 3; do ./inkling2mini.sh bench rolling > "$O/rolling-$i.txt" 2>&1; grep -E "METRIC decode_c4_(rolling|steady)_tok" "$O/rolling-$i.txt" | tr '\n' ' '; echo; done
[ "$TR" = 1 ] && ssh -o BatchMode=yes spark@$NODES 'grep event=round_trace /home/spark/inkling-2-mini-1x-dgx-spark/logs/serve.log' > "$O/round-trace.txt"
./inkling2mini.sh stop 2>&1 | tail -1
echo ALLDONE
