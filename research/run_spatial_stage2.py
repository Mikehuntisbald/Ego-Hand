import subprocess,sys,json
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/offline_keypoint_spatial_v3');CODE=Path(__file__).resolve().parent
subprocess.run([sys.executable,'export_spatial_rgb.py'],cwd=CODE,check=True)
jobs=[('rgb_dit','cuda:0','dit',False),('rgb_regression','cuda:2','regression',False),('tracks_dit','cuda:3','dit',True)]
processes=[]
for arm,device,kind,tracks in jobs:
    log=(ROOT/f'train_{arm}.log').open('w')
    args=[sys.executable,'train_spatial_temporal.py','--device',device,'--kind',kind]+(['--tracks-only'] if tracks else [])
    p=subprocess.Popen(args,cwd=CODE,stdout=log,stderr=subprocess.STDOUT);processes.append((arm,p,log))
arm,p,log=processes.pop();code=p.wait();log.close();assert code==0,(arm,code)
log=(ROOT/'train_tracks_regression.log').open('w')
p=subprocess.Popen([sys.executable,'train_spatial_temporal.py','--device','cuda:3','--kind','regression','--tracks-only'],cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
processes.append(('tracks_regression',p,log))
failures=[]
for arm,p,log in processes:
    code=p.wait();log.close()
    if code:failures.append((arm,code))
assert not failures,failures
print(json.dumps(dict(stage2_complete=True)),flush=True)
