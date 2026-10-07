import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'side_data_v16';RUN.mkdir(exist_ok=True);CODE=Path(__file__).resolve().parent
PY='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
def child(script,args,name):
    log=(RUN/f'{name}.log').open('a');return subprocess.Popen([PY,str(CODE/script),*args],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
if not (RUN/'ready.json').exists():
    p=child('prepare_side_data_v16.py',[],'prepare');save(RUN/'pipeline_status.json',dict(stage='recompute_prediction_side_XYZ',pid=p.pid));assert p.wait()==0,'Preparation failed; inspectprepare.log'
jobs=[]
for variant,device in [('control','cuda:0'),('consensus','cuda:1')]:
    if (RUN/variant/'risk_dense/done.json').exists():continue
    p=child('train_side_risk_v16.py',['--variant',variant,'--device',device],f'risk_{variant}');jobs.append((variant,p))
save(RUN/'pipeline_status.json',dict(stage='refit_risk_heads',jobs=[dict(variant=v,pid=p.pid) for v,p in jobs]))
for variant,p in jobs:assert p.wait()==0,f'Risk{variant}failed'
save(RUN/'pipeline_status.json',dict(stage='prepared',complete=True,training_started=False,scope='Originaltrain/dev only; risks re-fit for both upstream controls;3Dtraining pending'))
