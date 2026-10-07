"""One supervised task pipeline; persisted stages avoid duplicate work."""
import subprocess,json,os,time
from pathlib import Path
from hand3d_v8_common import V7,save
DATA=V7.parent/'aligned_density_v13';DATA.mkdir(exist_ok=True);CODE=Path(__file__).resolve().parent
PY='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'

def child(script,args,name):
    log=(DATA/f'{name}.log').open('a');p=subprocess.Popen([PY,str(CODE/script),*args],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);return p

def main():
    if not (DATA/'ready.json').exists():
        p=child('build_aligned_density_v13.py',[],'build');save(DATA/'pipeline_status.json',dict(stage='align_centers',pid=p.pid));assert p.wait()==0,'Alignment failed; inspect build.log'
    p=child('create_density_prior_v13.py',[],'prior');assert p.wait()==0
    if not (DATA/'model_preflight.json').exists():
        p=child('check_density_model_v13.py',[],'preflight');save(DATA/'pipeline_status.json',dict(stage='preflight3D',pid=p.pid));assert p.wait()==0,'3D preflight failed; inspect preflight.log'
    jobs=[]
    for arm,device in [('sparse','cuda:0'),('dense','cuda:1')]:
        if (DATA/f'risk_{arm}/done.json').exists():continue
        p=child('train_density_risk_v13.py',['--arm',arm,'--device',device],f'risk_{arm}');jobs.append((arm,p))
    save(DATA/'pipeline_status.json',dict(stage='retrain_risk',jobs=[dict(arm=a,pid=p.pid) for a,p in jobs]))
    for arm,p in jobs:assert p.wait()==0,f'Risk{arm} failed; inspect corresponding log'
    jobs=[];RUN=V7.parent/'native_density_v13';RUN.mkdir(exist_ok=True)
    for kind,density,device in [('dit','sparse','cuda:0'),('dit','dense','cuda:1'),('regression','dense','cuda:2')]:
        if (RUN/f'{kind}_{density}/done.json').exists():continue
        p=child('train_aligned_density_v13.py',['--kind',kind,'--density',density,'--device',device],f'train_{kind}_{density}');jobs.append((kind,density,p))
    save(DATA/'pipeline_status.json',dict(stage='train3D',jobs=[dict(kind=k,density=d,pid=p.pid) for k,d,p in jobs]))
    for kind,density,p in jobs:assert p.wait()==0,f'{kind}_{density} failed; inspect log'
    save(DATA/'pipeline_status.json',dict(stage='completed',complete=True,metrics_opened='Development only; independent evaluation remains pending'))

if __name__=='__main__':main()
