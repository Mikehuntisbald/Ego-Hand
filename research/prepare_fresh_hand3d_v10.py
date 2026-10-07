import json,time,hashlib
from hand3d_v8_common import V7,save
import prepare_fresh_hand3d_v8 as worker
RUN=V7.parent/'offline_hand3d_v10_native'
def main():
    path=RUN/'fresh_manifest.json'
    if not path.exists():
        previous=json.loads((V7.parent/'offline_hand3d_v8/fresh_manifest.json').read_text());next_manifest=json.loads((V7.parent/'offline_hand3d_v9/fresh_manifest.json').read_text());inventory=json.loads((V7.parent/'offline_hand3d_v8/fresh_inventory.json').read_text());sequences=[]
        for row in previous['sequences']:
            entry=next(x for x in inventory['targets'] if x['sequence']==row['sequence']);already=set(row['clips'])|set(next(x for x in next_manifest['sequences'] if x['sequence']==row['sequence'])['clips']);unused=[x for x in entry['unused'] if x not in already];assert len(unused)>=2
            clips=[unused[0],unused[-1]];sequences.append(dict(split='hand3d_v10_fresh',subject=row['subject'],sequence=row['sequence'],clips=clips))
        manifest={k:previous[k] for k in ['official_repo','revision','mirror','stream']};manifest.update(sequences=sequences,created_unix=time.time(),scope='Further12unused clips; excludes all v8/v9 clips; same reused subjects/sequences; no new-subject claim',selection='Clip IDs frozen before final model/policy selection, metrics unopened');save(path,manifest);save(RUN/'fresh_manifest_receipt.json',dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),clips=12,sequences=6))
    worker.RUN=RUN;worker.prepare()
if __name__=='__main__':main()
