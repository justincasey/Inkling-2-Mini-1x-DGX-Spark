from load import *
G=fw.astype(np.float64).T@fw.astype(np.float64)
ev,evec=np.linalg.eigh(G); idx=np.argsort(ev)[::-1]; evec=evec[:,idx]
cache=[]
for i in real:
    for s in range(7):
        b,dot,x=xtrue(i,s)
        am=int(np.argmax(x[:VV])); cache.append((i,s,b+fb,am))
BL=1024; NB=-(-V//BL)
for r in (8,16,24):
    Vr=evec[:,:r].astype(np.float32); P=(fw@Vr).astype(np.float32)
    P=rb(P)  # approx bf16 storage via round (value-level)
    res={k:[0,[]] for k in (32,64,96,128)}
    for (i,s,bfb,am) in cache:
        lp=Vr.T@lat[i,s]
        sc=(bfb+P@lp); sc[VV:]=-np.inf
        pad=np.full(NB*BL,-np.inf,np.float32); pad[:V]=sc
        bm=pad.reshape(NB,BL).max(1)
        sb=np.sort(bm)[::-1]
        for k in res:
            T=sb[k-1]
            c=int((sc>=T).sum()); res[k][1].append(c)
            if sc[am]<T: res[k][0]+=1
    for k in res:
        a=np.array(res[k][1]); print(f'r={r} k={k}: miss {res[k][0]}/{len(cache)} cand med {np.median(a):.0f} p99 {np.percentile(a,99):.0f} max {a.max()}')
