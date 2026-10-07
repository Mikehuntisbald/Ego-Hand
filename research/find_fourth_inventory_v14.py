import json,re,collections
from pathlib import Path
from hand3d_v8_common import V7,save
ROOT=V7.parent.parent;RUN=V7.parent/'fourth_dense_v14'

def main():
    definitions=json.loads((ROOT/'provenance/clip_definitions.response').read_text());splits=json.loads((ROOT/'provenance/clip_splits.response').read_text());tree=json.loads((ROOT/'provenance/train_aria_tree.json').read_text());ids={int(re.search(r'clip-(\d+)\.tar',r['path']).group(1)) for r in tree if re.search(r'clip-(\d+)\.tar',r['path'])}
    exported=collections.defaultdict(set)
    for path in (ROOT/'export/annotations').glob('*/*/clip-*.jsonl'):exported[path.parent.name].add(int(path.stem.split('-')[1]))
    names=['locked_frames.json','locked_rows.json','rows.json','rgb_records.json','predicted_manifest.jsonl','fresh_predicted_manifest.jsonl','manifest.jsonl','fresh_manifest.json','locked_manifest.json','manifest.json']
    def collect(value):
        if isinstance(value,dict):
            if isinstance(value.get('sequence'),str):
                for cid in value.get('clips',[value.get('clip')]):
                    if isinstance(cid,int):exported[value['sequence']].add(cid)
            for v in value.values():
                if isinstance(v,(dict,list)):collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for folder in (ROOT/'experiments').iterdir():
        if not folder.is_dir() or folder.name.startswith('archive_') or folder.name=='fourth_dense_v14':continue
        for name in names:
            path=folder/name
            if not path.exists():continue
            try:raw=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.suffix=='.jsonl' else json.loads(path.read_text())
            except Exception:continue
            collect(raw)
    for r in json.loads((V7.parent/'offline_hand3d_v8/fresh_inventory.json').read_text())['targets']:exported[r['sequence']].update(r['used'])
    available=collections.defaultdict(list)
    for cid in splits['train']['Aria']:
        if int(cid) in ids:available[definitions[str(cid)]['sequence_id']].append(int(cid))
    rows=[dict(sequence=s,subject=s.split('_')[0],available=sorted(v),exported=sorted(exported[s]),unused=sorted(set(v)-exported[s])) for s,v in sorted(available.items())]
    untouched=[r for r in rows if not r['exported'] and len(r['unused'])>=2];known_holdout=[r for r in rows if r['subject'] in ['P0010','P0015'] and len(r['unused'])>=2]
    save(RUN/'expanded_inventory.json',dict(rows=rows,scope='Official clip definitions/tree and existing exported filenames only; no RGB/pose/error inspected'))
    print(json.dumps(dict(tree_entries=len(tree),official_train_aria_clips=len(splits['train']['Aria']),available_tree_clips=len(ids),subjects=dict(collections.Counter(r['subject'] for r in rows)),untouched_sequences=[{k:v for k,v in r.items() if k in ['sequence','subject','unused']} for r in untouched],known_holdout_remaining=[{k:v for k,v in r.items() if k in ['sequence','subject','unused']} for r in known_holdout]),indent=2),flush=True)

if __name__=='__main__':main()
