import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'side_data_v16';CODE=Path(__file__).resolve().parent
PY='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
def child(script,args,name):
    log=(RUN/f'{name}.log').open('a');return subprocess.Popen([PY,str(CODE/script),*args],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
assert json.loads((RUN/'pipeline_status.json').read_text())['complete']
p=child('evaluate_side_upstream_v16.py',[],'upstream');assert p.wait()==0
assert json.loads((RUN/'upstream_results.json').read_text())['approved_for_3d_training'],'Upstreamfailed; inspectupstreamresults'
jobs=[]
for variant,device in [('control','cuda:0'),('consensus','cuda:1')]:
    if (V7.parent/'side_native_v16'/variant/'uniform_adaptive/done.json').exists():continue
    p=child('train_side_model_v16.py',['--variant',variant,'--device',device],f'train_{variant}');jobs.append((variant,p))
save(RUN/'training_status.json',dict(stage='matched3dcontrols',jobs=[dict(variant=v,pid=p.pid) for v,p in jobs]))
for variant,p in jobs:assert p.wait()==0,f'Training{variant}failed'
save(RUN/'training_status.json',dict(stage='completed',complete=True,scope='Train/devonly; development comparison and new independent evaluation pending'))
