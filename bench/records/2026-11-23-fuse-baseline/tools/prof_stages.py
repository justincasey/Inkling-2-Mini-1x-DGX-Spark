#!/usr/bin/env python3
"""prof_stages.py DB.sqlite [STEPS]: device time a step by stage of the replayed decode windows of an nsys_run.sh profile (microseconds a step,
summed over every stream, so the side streams' overlap makes the sum exceed the step's span; the span is printed too)."""
import sqlite3, sys, collections, statistics, re
db = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
c = sqlite3.connect(db)
rows = c.execute("select k.start,k.end,s.value,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
marks = [i for i, r in enumerate(rows) if 'linear_kernel' in r[2] and r[3] >= 1000]
m = marks[-(steps + 1):]
sel = rows[m[0]:m[-1]]
span = statistics.median([rows[m[i + 1]][0] - rows[m[i]][0] for i in range(len(m) - 1)]) / 1e3
def stage(n, g):
    if re.search(r'moe_gemv_kernel|gateup_epilogue|down_combine|combine_f32|group_kernel|own_remap', n): return 'MoE (gate/up, down, epilogues, combine, group)'
    if 'rot_in_kernel' in n and g in (8, 16, 24, 32, 48, 64) and False: return 'MoE'
    if re.search(r'attn_kernel|merge_kernel<|prologue_kernel|bias_kernel|ring_write', n) and 'split' not in n: return 'attention (prologue, bias, attn, merge, ring)'
    if re.search(r'split_local|split_merge|epilogue_kernel|commit|plan_kernel|embed_vec|rmsnorm_k', n): return 'tail (top-k, commits, embed, norms)'
    if 'big_kernel' in n or 'lin_wide2' in n: return 'prefill kernels (should be 0)'
    if re.search(r'linear_kernel<\(int\)12, \(int\)2, \(int\)8', n) and g >= 1000: return 'unembed'
    if re.search(r'linear_kernel|rot_in_kernel|rot_out_kernel|rot_in_frag', n): return 'dense EXL3 (q/k/v/r/gate/wo: rotations + GEMV)'
    return 'glue (mix, conv, controllers, router, select)'
agg = collections.defaultdict(float)
for r in sel:
    agg[stage(r[2], r[3])] += (r[1] - r[0]) / 1e3 / steps
print("span %.0f us a step" % span)
for k, v in sorted(agg.items(), key=lambda x: -x[1]): print("%-50s %8.0f us" % (k, v))
