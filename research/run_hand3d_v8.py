import json,subprocess,sys,time,hashlib
from pathlib import Path
from hand3d_v8_common import RUN,save
root=Path(__file__).resolve().parent
protocol=dict(task='Natural RGB temporal 3D hand completion for offline annotation',output='20x3 current-camera XYZ meters',acceptance=dict(relative_improvement_fraction=.05,camera_max_regression_mm=.1,camera_good_harm_rate=.01,relative_good_harm_rate=.01,source_paired_ci_upper_delta_mm=0,difficult_group='Actual bad-joint recovery and lower error required'),data='Frozen v7 train/dev roles, existing inspected test diagnostic only; fresh manifest not used in selection',initialization='v7 raw RGB checkpoint, same initial weights for DiT arms',arms=['dit_rollout','dit_one_step','regression_rollout'],visual='Frozen native spatial RGB first; 3D visual fine tuning as separately recorded followup if needed',steps=2400,batch=16,seed=202610081,created_unix=time.time(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'hand3d_rollout_v8.py',root/'hand3d_v8_common.py',root/'train_hand3d_v8.py']})
save(RUN/'protocol.json',protocol)
processes=[]
for gpu,(kind,mode) in enumerate([('dit','rollout'),('dit','one_step'),('regression','rollout')]):
    arm=kind+'_'+mode;log=open(RUN/(arm+'.log'),'w')
    proc=subprocess.Popen([sys.executable,str(root/'train_hand3d_v8.py'),'--kind',kind,'--mode',mode,'--device',f'cuda:{gpu}'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    processes.append(dict(arm=arm,pid=proc.pid,gpu=gpu));log.close()
save(RUN/'processes.json',processes);print(json.dumps(processes),flush=True)
