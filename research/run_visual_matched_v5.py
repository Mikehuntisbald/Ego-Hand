import subprocess,sys,json,hashlib,time
from pathlib import Path
root=Path('/mnt/why/HOT3D/experiments/natural_visual_ft_v5')
code=Path('/mnt/why/hot3d_hand_residual/finetune_visual_matched_v5.py')
(root/'matched_control_protocol.json').write_text(json.dumps(dict(reason='Eliminate cached FP16 stem versus live BF16 numerical confound; added before reading test comparisons',selection='unchanged dev_select checkpoint rule',steps=600,seed=202610053,code_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),created=time.time()),indent=2))
jobs=[]
for kind,gpu in [('regression',2),('dit',3)]:
    f=open(root/f'{kind}_matched.log','w');p=subprocess.Popen([sys.executable,str(code),'train','--kind',kind,'--mode','matched','--device',f'cuda:{gpu}'],stdout=f,stderr=subprocess.STDOUT,cwd=code.parent);jobs.append((kind,p,f))
status={}
for kind,p,f in jobs:status[kind]=p.wait();f.close()
(root/'matched_status.json').write_text(json.dumps(status))
