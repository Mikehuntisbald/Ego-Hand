from pathlib import Path
import json,requests
from collections import defaultdict,Counter
R=Path('/mnt/why/HOT3D');P=R/'provenance'
d=json.loads((P/'clip_definitions.response').read_text());s=json.loads((P/'clip_splits.response').read_text())
out=dict(split_keys={k:list(v) for k,v in s.items()},splits={})
for split,devices in s.items():
    out['splits'][split]={}
    for dev,ids in devices.items():
        seq=Counter(d[str(i)]['sequence_id'] for i in ids);subjects=Counter()
        for q,n in seq.items():subjects[q.split('_')[0]]+=n
        out['splits'][split][dev]=dict(clips=len(ids),sequences=len(seq),subjects=dict(subjects),sequence_counts=dict(seq))
(R/'catalog.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:{dv:{kk:vv for kk,vv in v.items() if kk!='sequence_counts'} for dv,v in dd.items()} for k,dd in out['splits'].items()}),flush=True)
urls={
'ms_hugg_meta':'https://modelscope.cn/api/v1/datasets/LIDAR-GT/HUGG_ARIA',
'ms_hugg_tree':'https://modelscope.cn/api/v1/datasets/LIDAR-GT/HUGG_ARIA/repo/tree?Revision=master&Root=&Recursive=false',
'pypi_ultralytics':'https://pypi.org/pypi/ultralytics/json',
'pypi_projectaria':'https://pypi.org/pypi/projectaria-tools/json',
}
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=20);(P/f'{name}.response').write_bytes(r.content);data=r.json();summary=dict(name=name,status=r.status_code,code=data.get('Code'),message=data.get('Message'))
        if 'info' in data:summary.update(version=data['info']['version'],python=data['info']['requires_python'])
        if name=='ms_hugg_meta' and isinstance(data.get('Data'),dict):summary['storage']=data['Data'].get('StorageSize')
        if name=='ms_hugg_tree' and isinstance(data.get('Data'),dict):summary['files']=[x['Path'] for x in data['Data'].get('Files',[])][:25]
        print(json.dumps(summary),flush=True)
    except Exception as e:print(name,str(e),flush=True)
