import subprocess,json
from hand3d_v8_common import V7
ROOT=V7.parent/'point_critic_native_v11';ROOT.mkdir(exist_ok=True)
jobs=[]
for arm,device in [('native','cuda:0'),('geometry','cuda:1')]:
    log=(ROOT/f'{arm}.log').open('w')
    p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','train_point_critic_native_v11.py','--arm',arm,'--device',device],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    jobs.append(dict(arm=arm,device=device,pid=p.pid))
(ROOT/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
