import sqlite3,sys,statistics as S
c=sqlite3.connect(sys.argv[1])
rows=c.execute("select s.value,(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value like 'slim_%'").fetchall()
for n in ('slim_scan_kernel','slim_pick_kernel'):
    d=sorted(r[1]/1000 for r in rows if r[0]==n)
    print(n,len(d),'median',d[len(d)//2],'p90',d[len(d)*9//10],'p99',d[len(d)*99//100],'max',d[-1])
