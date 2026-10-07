import json,hashlib,subprocess,sys,time,shutil
from pathlib import Path
BASE=Path('/mnt/why/HOT3D/experiments/offline_hand3d_v7');ROOT=BASE.parent/'offline_hand3d_v7_protected';ROOT.mkdir(exist_ok=True);CODE=Path('/mnt/why/hot3d_hand_residual')
protocol=json.loads((BASE/'protocol.json').read_text());protocol.update(parent_stage=str(BASE),steps=4000,lr=.00002,protection='Learned proposal trust + direct correct-camera/relative-point loss; targets only in training labels',selection='Same dev_select camera +0.5 relative criterion; includes step0',test_policy='No test labels used for repair, checkpoint or gate selection; existing inspected test remains diagnostic',created_unix=time.time())
protocol['code_sha256']={n:hashlib.sha256((CODE/n).read_bytes()).hexdigest() for n in ['hand3d_temporal_v7.py','hand3d_protected_v7.py','hand3d_data_v7.py','train_hand3d_protected_v7.py']}
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
for name in ['risk_all.pt','risk_calibration.json','preflight.json','data_checks.json']:shutil.copy2(BASE/name,ROOT/name)
jobs=[]
for gpu,(kind,tracks) in enumerate([('regression',False),('dit',False),('regression',True),('dit',True)]):
    name=('tracks_' if tracks else 'rgb_')+kind;f=open(ROOT/f'{name}.log','w');args=[sys.executable,str(CODE/'train_hand3d_protected_v7.py'),'--kind',kind,'--device',f'cuda:{gpu}']
    if tracks:args+=['--tracks-only']
    p=subprocess.Popen(args,cwd=CODE,stdout=f,stderr=subprocess.STDOUT);jobs.append((name,p,f))
status={}
for name,p,f in jobs:status[name]=p.wait();f.close()
(ROOT/'training_status.json').write_text(json.dumps(status))
if any(status.values()):raise RuntimeError(status)
