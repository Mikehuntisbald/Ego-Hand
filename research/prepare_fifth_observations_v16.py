"""Fifth unused-clip inventory/observations, before reading any model metrics."""
import hashlib,json
from pathlib import Path
from hand3d_v8_common import V7,save
import find_fourth_inventory_v14 as inventory
RUN=V7.parent/'fifth_dense_v16';CODE=Path(__file__).resolve().parent

def main():
    RUN.mkdir(exist_ok=True);inventory.RUN=RUN
    if not (RUN/'expanded_inventory.json').exists():inventory.main()
    original=(CODE/'prepare_fourth_dense_v14.py').read_text()
    before="['offline_hand3d_v8','offline_hand3d_v9','offline_hand3d_v10_native']"
    after="['offline_hand3d_v8','offline_hand3d_v9','offline_hand3d_v10_native','fourth_dense_v14']"
    assert original.count(before)==1;generated=original.replace(before,after).replace('Fourth12','Fifth12').replace('fourth12','fifth12').replace('all previous3batches','all previous4batches').replace('hand3d_fourth_dense_v14','hand3d_fifth_dense_v16')
    snapshot=RUN/'worker_snapshot.py';snapshot.write_text(generated)
    save(RUN/'source_provenance.json',dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),prior_batches=4,metrics_unopened=True))
    ns=dict(__name__='fifth_observation_worker_v16',__file__=str(snapshot));exec(compile(generated,str(snapshot),'exec'),ns);ns['RUN']=RUN;ns['freeze']()
    manifest=json.loads((RUN/'fresh_manifest.json').read_text());available={r['sequence']:set(r['unused']) for r in json.loads((RUN/'expanded_inventory.json').read_text())['rows']}
    bad=[dict(sequence=r['sequence'],clip=cid) for r in manifest['sequences'] for cid in r['clips'] if cid not in available[r['sequence']]]
    save(RUN/'unseen_check.json',dict(passed=not bad,previously_seen=bad,scope='All exported clips, explicit previous manifests and four previous independent batches excluded; no metric values inspected'))
    assert not bad,bad
    ns['main']()

if __name__=='__main__':main()
