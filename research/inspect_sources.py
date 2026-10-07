from pathlib import Path
import json,requests
from collections import Counter
ROOT=Path('/mnt/why/HOT3D');P=ROOT/'provenance'
def read(n):return json.loads((P/(n+'.response')).read_text())
d=read('ms_projectaria_meta')['Data'];print('MODELSCOPE',json.dumps(d,ensure_ascii=False)[:11500],flush=True)
h=read('hf_hugg_meta');ss=h['siblings'];print('HUGG',json.dumps(dict(sha=h['sha'],gated=h['gated'],usedStorage=h.get('usedStorage'),entries=len(ss),sample=[s['rfilename'] for s in ss if s['rfilename'].startswith('P0003_c701bd11/')]),ensure_ascii=False),flush=True)
c=read('clip_definitions');splits=read('clip_splits');print('CLIPS',json.dumps(dict(first={k:(v[:4] if isinstance(v,list) else v) for k,v in c['0'].items()},split_counts={k:len(v) for k,v in splits.items()})),flush=True)
aria={k:v for k,v in c.items() if v.get('device')=='Aria'};print('ARIA_KEYS',json.dumps(list(c['0'])),flush=True)
for name,path in [('ms_tree','https://modelscope.cn/api/v1/datasets/projectaria/hot3d/repo/tree?Revision=master&Root=&Recursive=false'),('hugg_seq_tree','https://hf-mirror.com/api/datasets/LIDAR-GT/HUGG_ARIA/tree/main/P0003_c701bd11?recursive=true')]:
    try:
        r=requests.get(path,timeout=20);(P/f'{name}.response').write_bytes(r.content);print(name,r.status_code,r.text[:3500],flush=True)
    except Exception as e:print(name,str(e))
