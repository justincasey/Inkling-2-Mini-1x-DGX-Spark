import sqlite3,sys
db=sys.argv[1]; which=int(sys.argv[2]) if len(sys.argv)>2 else 28
c=sqlite3.connect(db)
rows=c.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
idx=[i for i,r in enumerate(rows) if r[2].startswith('block_input_kernel')]
print(len(idx),'proposals')
a=idx[which]; b=idx[which+1] if which+1<len(idx) else len(rows)
t0=rows[a][0]
for r in rows[a:b]:
    print(f"{(r[0]-t0)/1000:8.1f} {(r[1]-r[0])/1000:7.1f} g({r[3]},{r[4]},{r[5]})x{r[6]} {r[2]}")
print('span',(rows[b-1][1]-t0)/1000)
