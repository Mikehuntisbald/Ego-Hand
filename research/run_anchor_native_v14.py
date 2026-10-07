import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'anchor_native_v14';RUN.mkdir(exist_ok=True)
if (RUN/'jobs.json').exists():
    for previous in json.loads((RUN/'jobs.json').read_text()):
        path=Path(f"/proc/{previous['pid']}/cmdline")
        if path.exists() and b'train_anchor_native_v14.py' in path.read_bytes():print('Existing anchor trial still running');raise SystemExit(0)
jobs=[]
for blend,device in [(0.,'cuda:0'),(.5,'cuda:1'),(1.,'cuda:2')]:
    log=(RUN/f'blend{blend:g}.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','train_anchor_native_v14.py','--blend',str(blend),'--device',device],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);jobs.append(dict(blend=blend,device=device,pid=p.pid))
save(RUN/'jobs.json',jobs);print(json.dumps(jobs))
