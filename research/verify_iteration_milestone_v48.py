"""Read-only CPU inspection of a completed checkpoint during long training."""
import argparse,json,time
from pathlib import Path
import torch
ROOT=Path('/mnt/why/HOT3D/experiments')
RUN=ROOT/'online_rgb_iterative_v48';SOURCE=ROOT/'online_rgb_3d_v47'
ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['frozen','joint','protected'],default='protected');a=ap.parse_args()
torch.set_num_threads(4)
resume=torch.load(RUN/a.arm/'resume.pt',weights_only=False,map_location='cpu',mmap=True)
ck=resume['checkpoint'];source_arm='dit_frozen' if a.arm=='frozen' else 'dit_joint'
warm=torch.load(SOURCE/source_arm/'last.pt',weights_only=False,map_location='cpu',mmap=True)
assert ck['step']>=100 and ck['cumulative_step']==600+ck['step']
changes={}
if a.arm=='frozen':
    assert ck['visual_tail'] is None
    first=json.loads((RUN/a.arm/'first_update.json').read_text());assert first['last_block_max_weight_change']==0
else:
    for prefix in ['blocks.28.','blocks.29.','blocks.30.','blocks.31.','last_norm.']:
        changes[prefix]=max(float((v-warm['visual_tail'][k]).abs().max()) for k,v in ck['visual_tail'].items() if k.startswith(prefix))
    assert all(x>0 for x in changes.values())
head=float((ck['model']['semantic_head.weight']-warm['model']['semantic_head.weight']).abs().max());assert head>0
for key,v in warm['model'].items():
    if key.startswith('codec.'):assert torch.equal(v,ck['model'][key]),key
pre=json.loads((RUN/a.arm/'preflight.json').read_text());rv=json.loads((RUN/a.arm/'resume_verification.json').read_text())
assert pre['passed'] and rv['passed']
if a.arm=='frozen':assert all(x==0 for x in pre['visual_block_gradient_norms'].values())
else:
    assert pre['pure_fk_visual_gradient']>0
    assert all(x>0 for x in pre['visual_block_gradient_norms'].values())
result=dict(passed=True,arm=a.arm,step=ck['step'],cumulative_step=ck['cumulative_step'],exported_at=time.time(),
    actual_visual_max_updates_from_warm600=changes,semantic_head_max_update=head,FK_geometry_buffers_exact=True,
    GPU_training_preflight_passed=True,visual_trainable=a.arm!='frozen',frozen_visual_no_updates=a.arm=='frozen',
    resume_verification_passed=True,last_validation=resume['history'][-1],
    scope='Interim engineering verification and locked development metrics; full controls and sealed replay still pending',default_changed=False)
path=RUN/f"milestone_{a.arm}_{ck['step']}.json";path.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
