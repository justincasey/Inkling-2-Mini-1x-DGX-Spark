#!/bin/bash
# one comparison pass of the oracle-gap record: per prompt, the engine dump against the engine- and spec-fixtures, and the two fixtures against each other
B=/home/spark/inkling-2-mini-1x-dgx-spark
L=$B/lab/fixgate
PY=$B/lab/rexl3-inkling2-loop-draft/venv/bin/python
PACK=/home/spark/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw
mkdir -p $L/out
cd $L/oracle
for n in chat-capital-effort0 chat-sky-effort0.9 raw-counting raw-long-window; do
  for kind in engine spec; do
    echo "== $n engine-dump vs $kind fixture"
    nice -n 10 $PY engine_gap.py --oracle $B/fixtures/$kind/$n.safetensors --engine $L/dump --pack $PACK --threads 8 --stock-gates --json $L/out/dump-vs-$kind-$n.json || echo "FAILED rc=$?"
  done
  echo "== $n engine fixture vs spec fixture"
  nice -n 10 $PY engine_gap.py --oracle $B/fixtures/spec/$n.safetensors --engine $B/fixtures/engine/$n.safetensors --pack $PACK --threads 8 --stock-gates --json $L/out/fixture-engine-vs-spec-$n.json || echo "FAILED rc=$?"
done
echo done
