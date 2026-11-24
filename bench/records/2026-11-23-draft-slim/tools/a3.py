from load import *
G=fw.astype(np.float64).T@fw.astype(np.float64)
ev,evec=np.linalg.eigh(G); idx=np.argsort(ev)[::-1]; evec=evec[:,idx]
rows=real[:]
cache=[]
for i in rows:
    for s in range(7):
        b,dot,x=xtrue(i,s)
        am=int(np.argmax(x[:VV])); cache.append((i,s,b+fb,am,x[am]))
print(len(cache),'cases')
Ds=(1,2,3,4,6,8,12)
for r in (8,12,16,24,32):
    Vr=evec[:,:r].astype(np.float32); P=fw@Vr
    miss={d:0 for d in Ds}; cnt={d:[] for d in Ds}; ranks=[]
    for (i,s,bfb,am,xm) in cache:
        lp=Vr.T@lat[i,s]
        xt=(bfb+P@lp)[:VV]
        mx=xt.max()
        for d in Ds:
            c=int((xt>=mx-d).sum()); cnt[d].append(c)
            if xt[am]<mx-d: miss[d]+=1
        ranks.append(int((xt>xt[am]).sum()))
    ranks=np.array(ranks)
    print(f'r={r}: argmax rank in proxy: max {ranks.max()} p99 {np.percentile(ranks,99)} p90 {np.percentile(ranks,90)} frac>=64 {np.mean(ranks>=64):.4f}')
    for d in Ds:
        a=np.array(cnt[d]); print(f'   delta {d}: miss {miss[d]}/{len(cache)}  cand med {np.median(a):.0f} p99 {np.percentile(a,99):.0f} max {a.max()}')
