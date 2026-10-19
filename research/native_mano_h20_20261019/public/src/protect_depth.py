"""Reject mono anchors contradicted by nearby measured stereo; retain raw evidence."""
import json,hashlib
from pathlib import Path
import numpy as np
import os
ROOT=Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))
source=json.loads((ROOT/'outputs/depth_fused_clip2102/depth.json').read_text())
folder=ROOT/'outputs/depth_protected_clip2102';folder.mkdir(exist_ok=False)
stereo=[x for x in source['observations'] if x['valid'] and 'LightESS' in x['source']]
checks=[]
for row in source['observations']:
    if not row['valid'] or 'UniDepth' not in row['source']:continue
    nearby=[x for x in stereo if x['track_id']==row['track_id'] and abs(x['timestamp_ns']-row['timestamp_ns'])<=200_000_000]
    if not nearby:continue
    z=np.median([x['wrist_depth_m'] for x in nearby]);threshold=max(.05,.25*z)
    rejected=abs(row['wrist_depth_m']-z)>threshold
    row['neighbour_stereo_check']=dict(reference_m=float(z),max_disagreement_m=float(threshold),rejected=bool(rejected),nearby_samples=len(nearby))
    if rejected:
        row['rejected_mono_depth_m']=row['wrist_depth_m'];row['valid']=False;row['wrist_depth_m']=None
    checks.append(dict(observation=row['observation'],**row['neighbour_stereo_check']))
source.update(valid=sum(x['valid'] for x in source['observations']),mono_temporal_protection=checks,
              selection_uses_GT=False,scope='development repair after first diagnostic evaluation; not new held-out result')
path=folder/'depth.json';path.write_text(json.dumps(source,indent=2));(folder/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),GT_used=False)))
print(json.dumps(checks),flush=True)
