import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_candidate_risk_v11_20261019');ARMS=['pose_base','RGB_base','RGB_preserve']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def eq(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b
def main():
    assert not (R/'controller_status.json').exists(),'Controller already exists';assert json.loads((R/'data/candidates_ready.json').read_text())['complete'];assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0
    (R/'runs').mkdir(exist_ok=True);sources={p.name:sha(p) for p in (R/'src').glob('*.py')};(R/'receipts/training_source_seal.json').write_text(json.dumps(sources,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(name,args,stage):
        for n,h in sources.items():assert sha(R/'src'/n)==h
        (R/'controller_status.json').write_text(json.dumps(dict(active=True,pid=os.getpid(),stage=stage,unix=time.time()),indent=2))
        with (R/'logs'/f'{stage}.log').open('w') as log:subprocess.run([sys.executable,str(R/'src'/name),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=stage,complete=True)),flush=True)
    for name in ['prepare_geometry.py','prepare_targets.py','audit_official_contract.py','preflight.py']:run(name,[],name[:-3])
    run('train_quality.py',['--arm','RGB_preserve','--output',R/'runs/probe_continuous','--steps',6,'--save-every',3],'probe_continuous')
    run('train_quality.py',['--arm','RGB_preserve','--output',R/'runs/probe_resumed','--steps',6,'--stop-after',3,'--save-every',3],'probe_first3')
    run('train_quality.py',['--arm','RGB_preserve','--output',R/'runs/probe_resumed','--steps',6,'--save-every',3,'--resume'],'probe_resume3')
    a=torch.load(R/'runs/probe_continuous/last.pt',map_location='cpu',weights_only=False);b=torch.load(R/'runs/probe_resumed/last.pt',map_location='cpu',weights_only=False);fields=['model','optimizer','rng_cpu','rng_cuda','python_rng','numpy_rng','batch_rng','step','arm','total_steps','best_score','best_model','batch_plan'];checks={k:bool(eq(a[k],b[k])) for k in fields};assert all(checks.values()),'Actual resume state mismatch';(R/'receipts/actual_resume.json').write_text(json.dumps(dict(complete=True,actual_interruption=True,continuous_steps=6,interrupted_at=3,resumed_steps=3,exact_fields=checks),indent=2))
    for arm in ARMS:run('train_quality.py',['--arm',arm,'--output',R/'runs'/arm],'train_'+arm)
    done={a:json.loads((R/'runs'/a/'done.json').read_text()) for a in ARMS};assert len({d['batch_plan_sha256'] for d in done.values()})==1
    for arm,d in done.items():
        for key in ['point.0.weight','delta.weight','harm.weight','coarse.weight']:assert d['gradient_sums'][key]>0 and d['actual_updates'][key]>0
        if arm!='pose_base':assert d['gradient_sums']['rgb.1.weight']>0 and d['actual_updates']['rgb.1.weight']>0
    (R/'controller_status.json').write_text(json.dumps(dict(active=False,complete=True,same_batch_plan=True,all_three_arms=True,production_admitted=False),indent=2))
if __name__=='__main__':main()
