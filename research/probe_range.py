from pathlib import Path
import json,requests,time,tarfile,io
from urllib.parse import urlsplit
R=Path('/mnt/why/HOT3D');m=json.loads((R/'subset_manifest.json').read_text());clip=m['sequences'][0]['clips'][0];name=f'train_aria/clip-{clip:06d}.tar'
url=f"{m['mirror']}/datasets/{m['official_repo']}/resolve/{m['revision']}/{name}"
started=time.time();r=requests.get(url,headers={'Range':'bytes=0-1048575'},timeout=(10,40),stream=True)
info=dict(clip=clip,status=r.status_code,content_range=r.headers.get('Content-Range'),content_length=r.headers.get('Content-Length'),final_host=urlsplit(r.url).hostname,seconds=time.time()-started)
if r.status_code==206:
    payload=r.content;(R/'provenance/range_prefix.bin').write_bytes(payload);members=[];offset=0
    while offset+512<=len(payload):
        block=payload[offset:offset+512]
        if not block.strip(b'\0'):break
        try:t=tarfile.TarInfo.frombuf(block,'utf-8','surrogateescape')
        except Exception as e:info['parse_error']=str(e);break
        members.append(dict(name=t.name,size=t.size,offset=offset));offset+=512+((t.size+511)//512)*512
    info.update(bytes=len(payload),members=members)
r.close();(R/'provenance/range_probe.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
