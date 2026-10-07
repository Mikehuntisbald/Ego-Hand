"""Keep v3 failure intact, promote its opened set to development, freeze new holdout."""
import json
import os
import random
from pathlib import Path
from collections import defaultdict
import torch

ROOT=Path('/mnt/why/HOT3D');OLD=ROOT/'experiments/dit_wilor_v3';RUN=ROOT/'experiments/dit_wilor_v4'
RUN.mkdir(exist_ok=True)
if (RUN/'condition_cache_done.json').exists():print('v4 setup already complete');raise SystemExit(0)
original=json.loads((ROOT/'subset_manifest.json').read_text())
v2=json.loads((ROOT/'experiments/dit_subject_oof_v2/protocol.json').read_text())
v3=json.loads((OLD/'protocol.json').read_text())
used={r['sequence'] for r in original['sequences']}|{r['sequence'] for r in v2['fresh_sequences']}|{r['sequence'] for r in v3['locked_sequences']}
defs=json.loads((ROOT/'provenance/clip_definitions.response').read_text());splits=json.loads((ROOT/'provenance/clip_splits.response').read_text())
available=defaultdict(list)
for clip in splits['train']['Aria']:
    seq=defs[str(clip)]['sequence_id']
    if seq not in used:available[seq].append(int(clip))
rng=random.Random(202610033);fresh=[];availability={}
for subject in ['P0003','P0010','P0015']:
    seqs=sorted(q for q,clips in available.items() if q.startswith(subject+'_') and len(clips)>=1)
    rng.shuffle(seqs);availability[subject]=len(seqs)
    for seq in seqs[:6]:
        clips=available[seq].copy();rng.shuffle(clips)
        fresh.append(dict(split='dit_v4_locked',subject=subject,sequence=seq,clips=sorted(clips[:6])))
assert len(fresh)>=7,availability
protocol={**v3,'study':'dit_wilor_v4','locked_sequences':fresh,
    'development':'Original v3 development plus opened v3 locked sequences; v3 failed ray-group significance and is retained unchanged',
    'change':'Reduce stochastic sampling variance; refit gates on original gate-fit sequences; no new test labels in training',
    'test_scope':'Fixed new sequence sample selected from metadata before viewing labels, including shorter sequences to improve sequence coverage; all v3 test results remain a recorded failed stage',
    'status':'sampling_and_gate_validation'}
(RUN/'protocol.json').write_text(json.dumps(protocol,indent=2))
(RUN/'locked_manifest.json').write_text(json.dumps({**{k:original[k] for k in ['official_repo','revision','mirror','stream']},'sequences':fresh},indent=2))
rows=json.loads((OLD/'rows.json').read_text());locked=json.loads((OLD/'locked_rows.json').read_text())
indices=[i for i,r in enumerate(locked) if r['matched']];start=len(rows)
for i in indices:
    r=locked[i]
    rows.append(dict(index=len(rows),role='development',subject=r['subject'],sequence=r['sequence'],clip=r['clip'],frame=r['frame'],
        image=r['image'],roi=None,side_label=r['side_label'],visibility_label=r['visibility_label'],projection_valid=r['projection_valid'],source='opened_v3_locked'))
(RUN/'rows.json').write_text(json.dumps(rows,indent=2))
obs=torch.load(OLD/'observations.pt',map_location='cpu',weights_only=False)
pieces={};previous=0
for path in sorted((OLD/'locked_observations').glob('*.pt')):
    p=torch.load(path,map_location='cpu',weights_only=False);assert p.pop('start')==previous;previous=p.pop('end')
    for k,v in p.items():pieces.setdefault(k,[]).append(v)
added={k:torch.cat(v)[indices] for k,v in pieces.items()};del pieces
gt=torch.tensor([locked[i]['gt'] for i in indices],dtype=torch.float32)
for key,value in [('gt',gt),('coarse',added['coarse']),('final_coarse',added['coarse']),('confidence',added['confidence']),('final_confidence',added['confidence'])]:obs[key]=torch.cat([obs[key],value])
torch.save(obs,RUN/'observations.pt')
folder=RUN/'feature_chunks';folder.mkdir(exist_ok=True)
for source in (OLD/'feature_chunks').glob('*.pt'):
    dest=folder/source.name
    if not dest.exists():os.link(source,dest)
features={k:v for k,v in added.items() if k not in ['coarse','confidence']}
for offset in range(0,len(indices),512):
    end=min(offset+512,len(indices));p={k:v[offset:end] for k,v in features.items()}
    p.update(start=start+offset,end=start+end);torch.save(p,folder/f'{start+offset:06d}.pt')
for kind in ['dit','regression']:
    source=OLD/f'ray_focus_{kind}';folder=RUN/f'base_{kind}';folder.mkdir(exist_ok=True)
    if not (folder/'proposal_best.pt').exists():os.link(source/'proposal_best.pt',folder/'proposal_best.pt')
    (folder/'proposal_done.json').write_text(json.dumps(dict(complete=True,imported_from=str(source),new_training_performed=False),indent=2))
(RUN/'condition_cache_done.json').write_text(json.dumps(dict(complete=True,samples=len(rows),development_extension=len(indices),training_rows_unchanged=True),indent=2))
(RUN/'status.json').write_text(json.dumps(dict(stage='sampling_and_gate_validation',complete=False,acceptance_passed=False),indent=2))
print(json.dumps(dict(run=str(RUN),samples=len(rows),new_locked_sequences=len(fresh),new_locked_clips=sum(len(s['clips']) for s in fresh),available_unused_sequences=availability),indent=2))
