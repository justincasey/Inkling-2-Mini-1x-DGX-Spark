from load import *
import sys
print(len(real),'real positions')
ds=[0,0.0625,0.125,0.25,0.5,1,2,4]
cnt={d:[] for d in ds}
bdot=[];
for i in real[::2]:
    for s in range(7):
        b,dot,x=xtrue(i,s)
        x=x[:VV]; M=x.max()
        for d in ds: cnt[d].append(int((x>=M-d).sum()))
        bdot.append(np.abs(dot[:VV]).max())
for d in ds:
    a=np.array(cnt[d]); print('delta',d,'count med',np.median(a),'p90',np.percentile(a,90),'p99',np.percentile(a,99),'max',a.max())
print('max|dot| mean',np.mean(bdot))
