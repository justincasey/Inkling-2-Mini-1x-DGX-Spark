import sqlite3,sys
c=sqlite3.connect(sys.argv[1])
rows=c.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
marks=[i for i,r in enumerate(rows) if r[2].startswith("void <unnamed>::linear_kernel<(int)12, (int)2, (int)8, (int)1, (int)1>")]
a,b=marks[-3],marks[-2]
t0=rows[a][0]
lo=int(sys.argv[2]); hi=int(sys.argv[3])
sid={}
for i in range(a+lo,a+hi):
    r=rows[i]; s=sid.setdefault(r[7],len(sid))
    nm=r[2].replace("void <unnamed>::","").replace("<unnamed>::","")[:46]
    print("%9.1f %7.1f L%d g(%d,%d,%d) %s"%((r[0]-t0)/1e3,(r[1]-r[0])/1e3,s,r[3],r[4],r[5],nm))
