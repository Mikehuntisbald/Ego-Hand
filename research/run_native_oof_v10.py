import json,subprocess,sys
from pathlib import Path
from train_native_oof_v10 import RUN
from hand3d_v8_common import save
root=Path(__file__).resolve().parent;jobs=[]
for fold in range(3):
    log=open(RUN/f'fold{fold}.log','w');proc=subprocess.Popen([sys.executable,str(root/'train_native_oof_v10.py'),'--fold',str(fold),'--device',f'cuda:{fold}'],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close();jobs.append(dict(fold=fold,gpu=fold,pid=proc.pid))
save(RUN/'processes.json',jobs);print(json.dumps(jobs),flush=True)
