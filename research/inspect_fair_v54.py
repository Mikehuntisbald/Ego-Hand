import json
from pathlib import Path
import torch

E = Path('/mnt/why/HOT3D/experiments')
def describe(v):
    if torch.is_tensor(v): return {'shape': list(v.shape), 'dtype': str(v.dtype)}
    if isinstance(v, dict): return {k: describe(x) for k,x in v.items()}
    if isinstance(v, list): return {'length':len(v), 'first':describe(v[0]) if v else None}
    return v
for folder,filename in [('online_rgb_3d_v47','inputs.pt'),('online_rgb_3d_v47','targets.pt'),('full_model_gloves_multihand_v51_20261007','mask_conditions.pt')]:
    v=torch.load(E/folder/filename,mmap=True,weights_only=False,map_location='cpu')
    print(folder,filename,json.dumps(describe(v),default=str)[:14000],flush=True)
for folder,filename in [('online_rgb_3d_v47','records.json'),('rfdetr_multidata_v53B_20261007','glove_rgb.json'),('online_rgb_3d_v47/diagnostic_v46','rows.json')]:
    p=E/folder/filename
    v=json.loads(p.read_text()); first=v[0] if isinstance(v,list) else v['frames'][0]
    print(str(p),json.dumps(first)[:5000],flush=True)
