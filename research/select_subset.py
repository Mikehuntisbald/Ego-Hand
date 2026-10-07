from pathlib import Path
from collections import defaultdict,Counter
import json,random,requests
R=Path('/mnt/why/HOT3D');P=R/'provenance';d=json.loads((P/'clip_definitions.response').read_text());s=json.loads((P/'clip_splits.response').read_text());train_ids=s['train']['Aria']
by=defaultdict(list)
for i in train_ids:by[d[str(i)]['sequence_id']].append(int(i))
subjects=defaultdict(list)
for q,ids in by.items():subjects[q.split('_')[0]].append(q)
validation_subjects=['P0003','P0010','P0015'];train_subjects=sorted(set(subjects)-set(validation_subjects));rng=random.Random(20261002)
selected=[]
for j,subject in enumerate(train_subjects):
    candidates=sorted(subjects[subject]);rng.shuffle(candidates)
    # Prefer sequences with at least 10 curated clips; retain random choice
    # among those, so the subset is not simply the longest recordings.
    candidates=sorted(candidates,key=lambda q:len(by[q])<10)
    for q in candidates[:5 if j==0 else 4]:selected.append(dict(split='train',subject=subject,sequence=q,clips=sorted(by[q])))
for subject in validation_subjects:
    candidates=sorted(subjects[subject]);rng.shuffle(candidates);candidates=sorted(candidates,key=lambda q:len(by[q])<10)
    for q in candidates[:2]:selected.append(dict(split='val',subject=subject,sequence=q,clips=sorted(by[q])))
assert sum(r['split']=='train' for r in selected)==25
assert not {r['subject'] for r in selected if r['split']=='train'}&{r['subject'] for r in selected if r['split']=='val'}
manifest=dict(format='HOT3D-Clips grouped by original source sequence; curated, non-continuous segments, not complete VRS recordings',official_repo='bop-benchmark/hot3d',revision=json.loads((P/'hf_official_meta.response').read_text())['sha'],mirror='https://hf-mirror.com',stream='214-1',seed=20261002,train_subjects=train_subjects,validation_subjects=validation_subjects,sequences=selected,totals={k:dict(sequences=sum(r['split']==k for r in selected),clips=sum(len(r['clips']) for r in selected if r['split']==k),frames=150*sum(len(r['clips']) for r in selected if r['split']==k)) for k in ['train','val']},retained_members=['*.image_214-1.jpg','*.cameras.json','*.hands.json','__hand_shapes.json__','*.info.json'],excluded=['monochrome image streams','object poses','object models','depth','SLAM point clouds'])
(R/'subset_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='sequences'}),flush=True)
# Fetch tar listing, including immutable file size and LFS sha for auditing.
url=f"https://hf-mirror.com/api/datasets/bop-benchmark/hot3d/tree/{manifest['revision']}/train_aria?recursive=false&limit=1000"
entries=[]
while url:
    response=requests.get(url,timeout=30);response.raise_for_status();entries+=response.json();url=response.links.get('next',{}).get('url')
    if url:url=url.replace('https://huggingface.co','https://hf-mirror.com')
(P/'train_aria_tree.json').write_text(json.dumps(entries,indent=2));print(json.dumps(dict(tar_count=len(entries),sample=entries[:1])),flush=True)
