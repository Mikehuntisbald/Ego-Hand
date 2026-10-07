import json
from pathlib import Path
run=Path('/mnt/why/HOT3D/experiments/dit_wilor_v3')
out={}
for folder in sorted(run.iterdir()):
    p=folder/'development_results.json'
    if not p.is_file():continue
    r=json.loads(p.read_text());g={}
    for name,group in r['groups'].items():
        key='axial_relative' if name=='multiple_ray_aligned_fingers' else 'relative'
        g[name]=dict(joints=group['joints'],metric=key,**group.get(key,{}))
    out[folder.name]=dict(accepted=r['development_acceptance_passed'],operating=r['operating'],groups=g)
for name in ['locked_data_status.json','locked_data_ready.json']:
    p=run/name
    if p.exists():out[name]=json.loads(p.read_text())
print(json.dumps(out,indent=2))
