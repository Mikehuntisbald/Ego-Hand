import json,time,hashlib,subprocess,sys
from pathlib import Path
from hand3d_trajectory_data_v9 import RUN
from hand3d_v8_common import save
code=Path(__file__).resolve().parent;v8=RUN.parent/'offline_hand3d_v8';inventory=json.loads((v8/'fresh_inventory.json').read_text());previous=json.loads((v8/'fresh_manifest.json').read_text());picked=[]
for item in previous['sequences']:
    entry=next(r for r in inventory['targets'] if r['sequence']==item['sequence']);unused=[x for x in entry['unused'] if x not in item['clips']];assert len(unused)>=2;clips=[unused[len(unused)//3],unused[2*len(unused)//3]];assert len(set(clips))==2;picked.append(dict(split='hand3d_v9_fresh',subject=item['subject'],sequence=item['sequence'],clips=clips))
manifest={k:previous[k] for k in ['official_repo','revision','mirror','stream']};manifest.update(sequences=picked,created_unix=time.time(),scope='Further unused source clips; v8 fresh clips excluded; reused subjects/sequences')
save(RUN/'fresh_manifest.json',manifest)
save(RUN/'protocol.json',dict(task='Restore naturally missing finger information through an entire3D trajectory',prediction='17x20x3 aligned camera trajectory; center20x3 output',reason='v8 full-history ablation only0.07mm relative change; v8 bounded severe8 remains79.35mm',training_subjects='Original v7 train subjects only; frozen dev roles',arms=['rgb_dit','rgb_regression'],steps=4000,batch=8,seed=202610091,selection='Bounded dev_select feasible first; dev_calibrate policy only; subsequent fresh12clips locked by this manifest',pretraining_overlap='WiLoR unknown; original supervised spatial stem already frozen, not full-pipeline OOF',target_supervision='All available annotated frames in original predicted tracks; labels never conditioning',fresh_manifest_sha256=hashlib.sha256((RUN/'fresh_manifest.json').read_bytes()).hexdigest(),created_unix=time.time(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [code/'hand3d_trajectory_v9.py',code/'hand3d_trajectory_data_v9.py',code/'train_hand3d_trajectory_v9.py']}))
jobs=[]
for kind,gpu in [('dit',0),('regression',1)]:
    assert json.loads((RUN/('preflight_'+kind+'.json')).read_text())['passed'];log=open(RUN/('rgb_'+kind+'.log'),'w');proc=subprocess.Popen([sys.executable,str(code/'train_hand3d_trajectory_v9.py'),'--kind',kind,'--device',f'cuda:{gpu}'],cwd=code,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);jobs.append(dict(kind=kind,gpu=gpu,pid=proc.pid));log.close()
save(RUN/'processes.json',jobs);print(json.dumps(jobs),flush=True)
