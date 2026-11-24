import sqlite3,sys,collections
db=sys.argv[1]; steps=int(sys.argv[2]); top=int(sys.argv[3]) if len(sys.argv)>3 else 45
c=sqlite3.connect(db)
rows=c.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
marks=[i for i,r in enumerate(rows) if r[2].startswith("void <unnamed>::linear_kernel<(int)12, (int)2, (int)8, (int)1, (int)1>")]
print("kernels total",len(rows),"unembed launches",len(marks))
m=marks[-(steps+1):]
sel=rows[m[0]:m[-1]]
print("kernels/step %.1f"%(len(sel)/steps))
t0=sel[0][0]; t1=sel[-1][1]
busy=sum(r[1]-r[0] for r in sel)
print("span_ms/step %.3f"%((t1-t0)/1e6/steps),"busy_ms/step %.3f"%(busy/1e6/steps))
gaps=0; last_end=sel[0][1]; ovl=0
for r in sel[1:]:
    if r[0]>last_end: gaps+=r[0]-last_end
    else: ovl+=min(last_end,r[1])-r[0]
    last_end=max(last_end,r[1])
print("idle gap ms/step %.3f overlap ms/step %.3f"%(gaps/1e6/steps,ovl/1e6/steps))
agg=collections.defaultdict(lambda:[0,0])
for r in sel:
    k=r[2][:105]; a=agg[k]; a[0]+=1; a[1]+=r[1]-r[0]
tot=sum(a[1] for a in agg.values())
print("name | launches/step | us/step | mean us | share")
for k,(cnt,d) in sorted(agg.items(),key=lambda x:-x[1][1])[:top]:
    print("%-105s %6.1f %9.1f %7.2f %5.1f%%"%(k,cnt/steps,d/1e3/steps,d/1e3/cnt,100*d/tot))
