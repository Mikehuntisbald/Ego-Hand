"""Real 3+3 versus uninterrupted 6 updates, including the extra sampler RNG."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_mano_rollout_v4_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')

def same(x,y):
    if torch.is_tensor(x):return torch.equal(x.cpu(),y.cpu())
    if isinstance(x,np.ndarray):return np.array_equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
    return x==y

def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(V2/'src'))
    commands=[('continuous',[R/'src/train_resume_probe.py','--arm','protected','--steps','6','--rollout','--output',R/'runs/resume_continuous']),
        ('interrupt',[R/'src/train_resume_probe.py','--arm','protected','--steps','6','--rollout','--stop-after','3','--output',R/'runs/resume_interrupted']),
        ('resume',[R/'src/train_native.py','--arm','protected','--steps','6','--rollout','--resume','--output',R/'runs/resume_interrupted'])]
    for name,cmd in commands:
        with (R/'logs'/f'resume_probe_{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,cmd)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if name=='interrupt':
            ck=torch.load(R/'runs/resume_interrupted/last.pt',weights_only=False,map_location='cpu');assert ck['step']==3 and not (R/'runs/resume_interrupted/done.json').exists()
    a=torch.load(R/'runs/resume_continuous/last.pt',weights_only=False,map_location='cpu');b=torch.load(R/'runs/resume_interrupted/last.pt',weights_only=False,map_location='cpu')
    fields=['model','optimizer','rng_cpu','rng_cuda','noise_rng','rollout_rng','python_rng','numpy_rng','batch_plan_sha','best_score','selection','step']
    equality={key:same(a[key],b[key]) for key in fields};assert all(equality.values()),equality
    best0=torch.load(R/'runs/resume_continuous/best_development.pt',weights_only=False,map_location='cpu');best1=torch.load(R/'runs/resume_interrupted/best_development.pt',weights_only=False,map_location='cpu');assert same(best0['model'],best1['model'])
    result=dict(complete=True,actual_interruption=True,continuous_updates=6,interrupted_at=3,resumed_updates=3,exact_fields=equality,selected_best_weights_exact=True,
        note='both probes checkpoint/select at 3 and 6; resume uses unchanged production experiment train_native.py, not a handwritten restore; small integration check, not long-run quality evidence',
        trainer_sha256=hashlib.sha256((R/'src/train_native.py').read_bytes()).hexdigest())
    (R/'receipts/actual_resume_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
