#!/usr/bin/env python3
"""gate_summary.py OUTDIR: one `GATE name passed failed` line a log of mini_gates.sh, and the hashes the rank's gates print."""
import glob, os, re, sys
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, '*.log'))):
    t = open(f, errors='replace').read()
    res = re.findall(r'test result: (ok|FAILED)\. (\d+) passed; (\d+) failed', t)
    p = sum(int(r[1]) for r in res); fl = sum(int(r[2]) for r in res)
    print('GATE %-18s passed=%d failed=%d (%d binaries)%s' % (os.path.basename(f)[:-4], p, fl, len(res), '' if res else '  NO RESULT'))
seen = set()
for f in sorted(glob.glob(os.path.join(d, '*.log'))):
    for l in open(f, errors='replace'):
        if re.search(r'sha256|c25f9876|a105f245|6d0a555b|top1', l) and l.strip() not in seen:
            seen.add(l.strip()); print(os.path.basename(f)[:-4], '|', l.strip()[:200])
