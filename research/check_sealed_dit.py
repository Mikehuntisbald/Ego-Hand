"""Check the delivered model and API using real observations without GT inputs."""
import os
import sys
import json
from pathlib import Path
RUN=Path(os.environ.get('HOT3D_DIT_RUN','/mnt/why/HOT3D/experiments/dit_wilor_v4'))
sys.path.insert(0,str(RUN/'sealed'))
import wilor_eval_common
import torch
from dit_v3_inference import Refiner

torch.set_num_threads(4)
obs=torch.load(RUN/'observations.pt',map_location='cpu',weights_only=False,mmap=True)
chunk=torch.load(RUN/'feature_chunks/000000.pt',map_location='cpu',weights_only=False)
keys=['coarse','confidence','wilor','transform','geometry','wilor_2d','shape','global','local']
all_data={**obs,**chunk};b={k:all_data[k][:4].to('cuda:0') for k in keys}
checks={}
for kind in ['dit','regression']:
    refiner=Refiner(RUN/'sealed'/f'{kind}.pt')
    a=refiner(b)
    polluted=dict(b,gt=torch.full_like(b['coarse'],float('nan')),hard_ray=torch.ones(4,20,device='cuda:0'),visibility_label=torch.zeros(4,device='cuda:0'),side_label=torch.ones(4,device='cuda:0'))
    c=refiner(polluted)
    assert torch.equal(a['xyz'],c['xyz']),'GT or label affected inference'
    assert torch.isfinite(a['xyz']).all()
    closed=a['gates'].eq(0)
    if closed.ndim==3:closed=closed.all(-1)
    assert torch.equal(a['xyz'][closed],b['coarse'][closed])
    if refiner.checkpoint['sampling']['policy']=='zero':
        d=refiner(b,seed=22)
        assert torch.equal(a['xyz'],d['xyz']),'Zero-latent inference depends on random seed'
    checks[kind]=dict(finite=True,no_gt_required=True,extra_gt_ignored=True,closed_point_exact_identity=True,
                     deterministic_zero=refiner.checkpoint['sampling']['policy']=='zero')
(RUN/'sealed_api_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
