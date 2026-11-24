from load import *
G=fw.astype(np.float64).T@fw.astype(np.float64)
ev,evec=np.linalg.eigh(G); idx=np.argsort(ev)[::-1]; ev=ev[idx]; evec=evec[:,idx]
cum=np.cumsum(ev)/ev.sum()
print('energy top r:',{r:round(float(cum[r-1]),4) for r in (4,8,16,32,48,64,96,128,192)})
rows=real[::2]
for r in (16,32,64,96):
    Vr=evec[:,:r].astype(np.float32)
    P=fw@Vr            # [V,r] proxy table
    res=np.sqrt(np.maximum((fw.astype(np.float64)**2).sum(1)-(P.astype(np.float64)**2).sum(1),0)).astype(np.float32)  # residual norm per row
    errs=[];rec={16:0,64:0,256:0};tot=0;prov=[];provm=[]
    for i in rows:
        for s in range(7):
            b,dot,x=xtrue(i,s)
            l=lat[i,s]; lp=Vr.T@l
            dt=P@lp                       # proxy dot
            e=dot-dt; errs.append(np.abs(e[:VV]))
            xt=b+fb+dt
            xt[VV:]=-1e9
            am=int(np.argmax(x[:VV]))
            order=np.argpartition(-xt,256)[:256]
            rank=int((xt[:VV]>xt[am]).sum())
            tot+=1
            for N in rec:
                if rank<N: rec[N]+=1
            # provable: eps_v = res_v * ||l_perp||
            lperp=np.sqrt(max(float(l@l)-float(lp@lp),0))
            eps=res*lperp*1.0001
            # threshold = max over v of (xt - eps) ; candidates = xt+eps >= thr
            thr=(xt-eps)[:VV].max()
            prov.append(int(((xt+eps)[:VV]>=thr).sum()))
    ea=np.concatenate(errs)
    print(f'r={r}: |err| med {np.median(ea):.4f} p99 {np.percentile(ea,99):.3f} p99.99 {np.percentile(ea,99.99):.3f} max {ea.max():.3f}; recall top16 {rec[16]/tot:.4f} top64 {rec[64]/tot:.4f} top256 {rec[256]/tot:.4f}; provable cand med {np.median(prov)} p90 {np.percentile(prov,90)} max {max(prov)}')
