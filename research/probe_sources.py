from pathlib import Path
import requests,json,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/mnt/why/HOT3D');ROOT.mkdir(parents=True,exist_ok=True);P=ROOT/'provenance';P.mkdir(exist_ok=True)
tasks={
'hf_official_meta':'https://hf-mirror.com/api/datasets/bop-benchmark/hot3d',
'hf_official_tree':'https://hf-mirror.com/api/datasets/bop-benchmark/hot3d/tree/main?recursive=false',
'hf_full_tree':'https://hf-mirror.com/api/datasets/projectaria/hot3d/tree/main?recursive=true',
'hf_hugg_meta':'https://hf-mirror.com/api/datasets/LIDAR-GT/HUGG_ARIA',
'hf_hugg_tree':'https://hf-mirror.com/api/datasets/LIDAR-GT/HUGG_ARIA/tree/main?recursive=false',
'ms_bop_meta':'https://modelscope.cn/api/v1/datasets/bop-benchmark/hot3d',
'ms_projectaria_meta':'https://modelscope.cn/api/v1/datasets/projectaria/hot3d',
'ms_search':'https://modelscope.cn/api/v1/datasets?page_number=1&page_size=20&search=hot3d',
'aria_explorer':'https://explorer.projectaria.com/hot3d-aria/P0001_a9d6c83d',
'clip_definitions':'https://hf-mirror.com/datasets/bop-benchmark/hot3d/resolve/main/clip_definitions.json',
'clip_splits':'https://hf-mirror.com/datasets/bop-benchmark/hot3d/resolve/main/clip_splits.json',
}
def get(item):
    name,url=item;started=time.time()
    try:
        r=requests.get(url,timeout=(8,25));(P/f'{name}.response').write_bytes(r.content)
        out=dict(name=name,status=r.status_code,bytes=len(r.content),seconds=time.time()-started,content_type=r.headers.get('Content-Type'))
        try:
            d=r.json()
            if isinstance(d,list):out['entries']=[dict(path=v.get('path'),size=v.get('size'),type=v.get('type')) for v in d[:30]]
            elif isinstance(d,dict):out['keys']=list(d)[:20];out['sha']=d.get('sha');out['error']=d.get('error');out['code']=d.get('Code');out['message']=d.get('Message')
        except ValueError:pass
        return out
    except Exception as e:return dict(name=name,error=str(e)[:180],seconds=time.time()-started)
with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(get,tasks.items()))
(P/'source_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
