"""Runs ON node-1 (read-only: header-range reads of the router planes of the four spec fixtures; writes only to stdout).
    ssh -o BatchMode=yes spark@<node-1> "nice -n 19 python3 -" < copy_planes_remote.py > planes.bin
planes.bin is a stream of [u32 header_len][u32 data_len][json header][data] frames; gen_fixture_stats.py reads it."""
import json,struct,sys,os
base='/home/spark/inkling-2-mini-1x-dgx-spark/fixtures/spec/'
out=sys.stdout.buffer
for fn in sorted(os.listdir(base)):
    if not fn.endswith('.safetensors'): continue
    with open(base+fn,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        h=json.loads(f.read(n))
        meta=h.get('__metadata__',{})
        for name,t in h.items():
            if name=='__metadata__': continue
            if not (name.startswith('router_') or name=='tokens'): continue
            a,b=t['data_offsets']
            f.seek(8+n+a); data=f.read(b-a)
            hdr=json.dumps({'file':fn,'name':name,'dtype':t['dtype'],'shape':t['shape'],'meta':meta}).encode()
            out.write(struct.pack('<II',len(hdr),len(data))); out.write(hdr); out.write(data)
