import json,re
from collections import defaultdict
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D');RUN=ROOT/'experiments/offline_hand3d_v8'
used=defaultdict(set);evidence=[]
# Only explicit observation/frame manifests; no metric values or labels inspected.
for folder in sorted((ROOT/'experiments').iterdir()):
    if not folder.is_dir() or folder.name.startswith('archive_') or folder.name=='offline_hand3d_v8':continue
    for name in ['locked_frames.json','locked_rows.json','rows.json','rgb_records.json','predicted_manifest.jsonl','fresh_predicted_manifest.jsonl','manifest.jsonl','fresh_manifest.json','locked_manifest.json','manifest.json']:
        path=folder/name
        if not path.exists():continue
        try:
            raw=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.suffix=='.jsonl' else json.loads(path.read_text())
        except Exception:continue
        def collect(value):
            if isinstance(value,dict):
                if isinstance(value.get('sequence'),str):
                    seq=value['sequence'];clips=value.get('clips',[value.get('clip')])
                    for c in clips:
                        if isinstance(c,int):used[seq].add(c)
                for v in value.values():
                    if isinstance(v,(list,dict)):collect(v)
            elif isinstance(value,list):
                for v in value:collect(v)
        collect(raw);evidence.append(str(path))
tree=json.loads((ROOT/'provenance/train_aria_tree.json').read_text());available=defaultdict(set)
definitions=json.loads((ROOT/'provenance/clip_definitions.response').read_text())
splits=json.loads((ROOT/'provenance/clip_splits.response').read_text())
available_ids={int(re.search(r'clip-(\d+)\.tar',r['path']).group(1)) for r in tree if re.search(r'clip-(\d+)\.tar',r['path'])}
for cid in splits['train']['Aria']:
    if int(cid) in available_ids:available[definitions[str(cid)]['sequence_id']].add(int(cid))
# Reject a source sequence mentioned anywhere in experiment metadata, even when
# its per-frame observation manifests are not among the standard names above.
mentioned=set(used)
for path in (ROOT/'experiments').glob('*/*.json'):
    if path.parent.name=='offline_hand3d_v8' or path.stat().st_size>50_000_000:continue
    mentioned.update(re.findall(r'P\d{4}_[0-9a-f]{8}',path.read_text(errors='ignore')))
targets=[]
for seq in sorted(available):
    if seq.startswith(('P0010_','P0015_')):
        targets.append(dict(sequence=seq,mentioned_before=seq in mentioned,used=sorted(used[seq]),available=sorted(available[seq]),unused=sorted(available[seq]-used[seq])))
out=dict(evidence=evidence,targets=targets,scope='Unused source clips, reused subjects; not subject held-out pretraining evidence')
(RUN/'fresh_inventory.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(files=len(evidence),targets=targets)),flush=True)
