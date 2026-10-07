from pathlib import Path
import json,time,requests
from urllib.parse import urlsplit,parse_qs
R=Path('/mnt/why/HOT3D');m=json.loads((R/'subset_manifest.json').read_text());clip=m['sequences'][0]['clips'][0];url=f"{m['mirror']}/datasets/{m['official_repo']}/resolve/{m['revision']}/train_aria/clip-{clip:06d}.tar"
s=requests.Session();rows=[];cdn=None
for label,start,end,direct in [('initial',0,2047,False),('reuse',100000,102047,True),('larger',0,4194303,False),('reuse_large',4194304,8388607,True)]:
    begin=time.time();r=s.get(cdn if direct else url,headers={'Range':f'bytes={start}-{end}'},timeout=30);cdn=r.url
    rows.append(dict(test=label,status=r.status_code,bytes=len(r.content),range=r.headers.get('Content-Range'),seconds=time.time()-begin,redirects=[x.status_code for x in r.history],query_keys=list(parse_qs(urlsplit(r.url).query))))
(R/'provenance/cdn_range_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
