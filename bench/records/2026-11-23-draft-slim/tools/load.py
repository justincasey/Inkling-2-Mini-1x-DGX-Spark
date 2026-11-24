import numpy as np
D='/tmp/slim/'
def bf(u): return (u.astype(np.uint32)<<16).view(np.float32)
def rb(x):  # round f32 -> bf16 -> f32 (RNE)
    u=x.astype(np.float32).view(np.uint32)
    r=((u>>16)&1)+0x7fff
    return ((u+r)&0xffff0000).view(np.float32)
V=201088; VV=200058; R=256
fw=bf(np.fromfile(D+'fold_w.bin',dtype=np.uint16)).reshape(V,R)
fb=bf(np.fromfile(D+'fold_b.bin',dtype=np.uint16))
names=[l.split()[0] for l in open(D+'anchors.txt')]
real=[i for i,n in enumerate(names) if n.startswith('real_')]
base=np.memmap(D+'base.bin',dtype=np.uint16,mode='r').reshape(-1,7,V)
lat=bf(np.fromfile(D+'latent.bin',dtype=np.uint16)).reshape(-1,7,R)
def xtrue(i,s):
    b=bf(np.array(base[i,s]))
    dot=(fw.astype(np.float64)@lat[i,s].astype(np.float64)).astype(np.float32)
    return b, dot, rb(b+rb(dot+fb))
