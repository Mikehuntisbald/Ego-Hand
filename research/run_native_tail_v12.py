import json,subprocess
from hand3d_v8_common import V7
RUN=V7.parent/'native_tail_v12';RUN.mkdir(exist_ok=True);jobs=[]
for mode,device in [('ft','cuda:0'),('frozen','cuda:1')]:
    log=(RUN/f'{mode}.log').open('w');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','train_native_tail_v12.py','--mode',mode,'--device',device],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);jobs.append(dict(mode=mode,device=device,pid=p.pid))
(RUN/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs))
