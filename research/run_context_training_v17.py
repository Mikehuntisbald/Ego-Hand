import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'context_data_v17';CODE=Path(__file__).resolve().parent;PY='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
def child(script,args,name):
    log=(RUN/f'{name}.log').open('a');return subprocess.Popen([PY,str(CODE/script),*args],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
assert json.loads((RUN/'ready.json').read_text())['complete'];jobs=[]
for v,d in [('control','cuda:0'),('bridge','cuda:1')]:
    if (RUN/v/'risk_dense/done.json').exists():continue
    p=child('train_context_risk_v17.py',['--variant',v,'--device',d],f'risk_{v}');jobs.append((v,p))
save(RUN/'training_status.json',dict(stage='retrainOOFrisks',jobs=[dict(variant=v,pid=p.pid) for v,p in jobs]))
for v,p in jobs:assert p.wait()==0,f'Risk{v}failed'
jobs=[]
for v,d in [('control','cuda:0'),('bridge','cuda:1')]:
    if (V7.parent/'context_native_v17'/v/'uniform_adaptive/done.json').exists():continue
    p=child('train_context_model_v17.py',['--variant',v,'--device',d],f'train_{v}');jobs.append((v,p))
save(RUN/'training_status.json',dict(stage='matched3Dcontrols',jobs=[dict(variant=v,pid=p.pid) for v,p in jobs]))
for v,p in jobs:assert p.wait()==0,f'Model{v}failed'
save(RUN/'training_status.json',dict(complete=True,stage='completed',scope='Train/devonly; effect not claimed untildevelopment and another unused independentbatch'))
