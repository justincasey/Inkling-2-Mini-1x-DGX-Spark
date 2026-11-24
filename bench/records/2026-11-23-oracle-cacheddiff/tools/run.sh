#!/usr/bin/env bash
cd /home/spark/inkling-2-mini-1x-dgx-spark/lab/oracle-cacheddiff
P=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw
date -u +%FT%TZ > started.txt
nice -n 19 /home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-draft/venv/bin/python oracle/gen_fixtures.py --pack $P --out fixtures/spec --only raw-long-window --workers 3 --threads 4 --weight-cache-gib 12 --verify-recompute > long.log 2>&1; echo "long rc=$? $(date -u +%FT%TZ)" >> progress.txt
nice -n 19 /home/spark/inkling-2-mini-1x-dgx-spark/lab/rexl3-inkling2-loop-draft/venv/bin/python oracle/gen_fixtures.py --pack $P --out fixtures/spec --only chat-sky-effort0.9 raw-counting --workers 3 --threads 4 --weight-cache-gib 12 --verify-recompute > short.log 2>&1; echo "short rc=$? $(date -u +%FT%TZ)" >> progress.txt
echo done >> progress.txt
