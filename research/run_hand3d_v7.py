import json,hashlib,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/offline_hand3d_v7');CODE=Path('/mnt/why/hot3d_hand_residual')
assert json.loads((ROOT/'preflight.json').read_text())['passed'] and (ROOT/'risk_done.json').exists()
protocol=dict(task='Natural offline 3D hand joint completion for annotation',output='20x3 XYZ in current camera coordinates, meters',root_pose='wrist translation and wrist-relative pose predicted separately',inputs='RGB spatial grids, aligned predicted WiLoR XYZ, timestamps, camera poses, predicted 3D error risks',
    forbidden_inference_inputs=['GT joints','GT side','GT hand shape','visibility labels'],window=17,offsets_s=[-4,-8/3,-5/3,-1,-2/3,-.5,-1/3,-1/6,0,1/6,1/3,.5,2/3,1,5/3,8/3,4],
    encoder='Frozen WiLoR 32-block ViT, 16x12x1280; validated spatial stem ->16x12x128; all cells retained',heads='width192 depth4 heads6; XYZ residual tokens include separate wrist root',
    arms=['rgb_regression','rgb_dit','tracks_regression','tracks_dit'],steps=6000,batch=32,lr=.0002,seed=202610073,selection='dev_select camera MPJPE19 +0.5 wrist-relative MPJPE19, includes step0',
    gate='dev_calibrate only; <=1% initially <=10mm joints harmed beyond20mm in both camera and relative error; require >=0.2mm camera improvement and >=5% reduction of camera bad-point error; relative may not worsen >0.1mm',
    scope='P0010/P0015 previously inspected subjects/sequences; diagnostic validation, not untouched or broad generalization evidence',archive='/mnt/why/HOT3D/experiments/archive_2d_before_3d_v7_20261003',created_unix=time.time(),
    code_sha256={n:hashlib.sha256((CODE/n).read_bytes()).hexdigest() for n in ['hand3d_temporal_v7.py','hand3d_data_v7.py','hand3d_risk_v7.py','train_hand3d_v7.py']})
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
jobs=[]
for gpu,(kind,tracks) in enumerate([('regression',False),('dit',False),('regression',True),('dit',True)]):
    name=('tracks_' if tracks else 'rgb_')+kind;f=open(ROOT/f'{name}.log','w');args=[sys.executable,str(CODE/'train_hand3d_v7.py'),'--kind',kind,'--device',f'cuda:{gpu}']
    if tracks:args+=['--tracks-only']
    p=subprocess.Popen(args,cwd=CODE,stdout=f,stderr=subprocess.STDOUT);jobs.append((name,p,f))
status={}
for name,p,f in jobs:status[name]=p.wait();f.close()
(ROOT/'training_status.json').write_text(json.dumps(status))
if any(status.values()):raise RuntimeError(status)
