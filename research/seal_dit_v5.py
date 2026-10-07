"""Seal a conservative accepted operating point before new-sequence inference."""
import hashlib
import json
import shutil
import os
import argparse
from pathlib import Path
import torch

RUN=Path(os.environ.get('HOT3D_DIT_RUN','/mnt/why/HOT3D/experiments/dit_wilor_v5'))
parser=argparse.ArgumentParser()
parser.add_argument('--dit-arm',nargs='+',default=['rollout_dit','coherent_dit'])
parser.add_argument('--regression-arm',nargs='+',default=['rollout_regression','coherent_regression'])
args=parser.parse_args()
CODE=Path('/mnt/why/hot3d_hand_residual')
out=RUN/'sealed';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
dest=out/'selection.json'
if dest.exists():
    print(dest.read_text());raise SystemExit(0)
selected={}
for kind in ['dit','regression']:
    candidates=[]
    for name in getattr(args,f'{kind}_arm'):
        folder=RUN/name
        scan=json.loads((folder/'operating_acceptance_scan.json').read_text())
        candidates.extend(dict(r,_folder=str(folder)) for r in scan['candidates'] if r['accepted'])
    assert candidates
    def margin(r):
        op=r['operating']
        gains=[g['axial_relative' if k=='multiple_ray_aligned_fingers' else 'relative']['improvement_pct'] for k,g in r['groups'].items()]
        return min([(v-5)/5 for v in gains]+[(.05-op['harm_camera'])/.05,(.05-op['harm_relative'])/.05])
    chosen=max(candidates,key=lambda r:(margin(r),-r['operating']['score']))
    folder=Path(chosen['_folder'])
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    ck['operating']=chosen['operating'];ck['sampling']=ck.get('sampling',dict(policy='independent',samples=2,steps=10))
    target=out/f'{kind}.pt';torch.save(ck,target)
    selected[kind]=dict(source=str(folder),checkpoint=str(target),sha256=sha(target),operating=chosen['operating'],
                         sampling=ck['sampling'],coherent_gate=ck.get('coherent_gate',False),development_groups=chosen['groups'],normalized_acceptance_margin=margin(chosen))
for name in sorted(p.name for p in CODE.glob('*.py')):
    shutil.copy2(CODE/('dit_v5_inference.py' if name=='dit_v3_inference.py' else name),out/name)
record=dict(models=selected,selection='Maximize worst normalized margin above 5% hard-group improvement and below 5% correct-point harm, among CI-passing development settings; then lowest development error',
            locked_data_scored=False,locked_manifest_sha256=sha(RUN/'locked_manifest.json'),protocol_sha256=sha(RUN/'protocol.json'),
            code={p.name:sha(p) for p in out.glob('*.py')})
dest.write_text(json.dumps(record,indent=2))
(RUN/'status.json').write_text(json.dumps(dict(stage='sealed_for_locked_evaluation',complete=False,acceptance_passed=False),indent=2))
print(json.dumps({k:dict(operating=v['operating'],margin=v['normalized_acceptance_margin']) for k,v in selected.items()},indent=2))

