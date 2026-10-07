import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

out=Path('/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/assets')
url='https://hf-mirror.com/spaces/rolpotamias/WiLoR/resolve/99fe3d7acff8104ecca1055df7467709506c2fa6/pretrained_models/wilor_final.ckpt'
size=2564989533
chunk=32<<20
def fetch(i):
    start=i*chunk;end=min(size,start+chunk)-1;p=out/f'checkpoint.part{i:03d}'
    if p.exists() and p.stat().st_size==end-start+1:return p
    for attempt in range(3):
        try:
            with requests.get(url+f'?download=true&part={i}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{size}',r.headers.get('Content-Range')
                with p.open('wb') as f:
                    for data in r.iter_content(1<<20):f.write(data)
            assert p.stat().st_size==end-start+1
            return p
        except Exception:
            if attempt==2:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    parts=list(pool.map(fetch,range((size+chunk-1)//chunk)))
dest=out/'wilor_final.mirror.ckpt'
h=hashlib.sha256()
with dest.open('wb') as f:
    for p in parts:
        with p.open('rb') as r:
            for b in iter(lambda:r.read(8<<20),b''):f.write(b);h.update(b)
print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,sha256=h.hexdigest())),flush=True)
