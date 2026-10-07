import sys,json,time,subprocess,hashlib
from pathlib import Path
from hand3d_v8_common import save
from train_hand3d_native_v10 import RUN
root=Path(__file__).resolve().parent
assert all(r['passed'] for r in json.loads((RUN/'preflight.json').read_text()).values())
protocol=dict(task='Test whether frozen2D spatial compression discards3D recovery evidence',arms=['dit_c1280','dit_c128','regression_c1280','regression_c128'],steps=4000,batch=8,seed=202610101,inputs='Same unmasked RGB/native XYZ/camera/time; only semantic condition channels differ',GT='Trajectory supervised on original train6subjects; no GT/visibility in inference',fixes='Noisy target residuals exclude unlabeled placeholders; absent frames masked from self/cross attention. Both condition arms use the same fixes.',parameter_caveat='Native condition projection has more parameters; this is a practical input ablation, not capacity-matched causal attribution',next_data='Evaluate successful architecture on another unused-clip manifest; all v8/v9 inspected clips remain diagnostic',created_unix=time.time(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'hand3d_native_v10.py',root/'train_hand3d_native_v10.py',root/'hand3d_trajectory_v9.py',root/'hand3d_trajectory_data_v9.py']});save(RUN/'protocol.json',protocol);jobs=[]
for gpu,(kind,ch) in enumerate([('dit',1280),('dit',128),('regression',1280),('regression',128)]):
    arm=f'{kind}_c{ch}';log=open(RUN/(arm+'.log'),'w');proc=subprocess.Popen([sys.executable,str(root/'train_hand3d_native_v10.py'),'--kind',kind,'--channels',str(ch),'--device',f'cuda:{gpu}'],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close();jobs.append(dict(arm=arm,pid=proc.pid,gpu=gpu))
save(RUN/'processes.json',jobs);print(json.dumps(jobs),flush=True)
