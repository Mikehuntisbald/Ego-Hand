"""Single GPU3 controller. Freeze sources, verify short resume, train two arms."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_rgb_observation_v9_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def eq(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b
def main():
    (R/'logs').mkdir(exist_ok=True);(R/'receipts').mkdir(exist_ok=True);(R/'runs').mkdir(exist_ok=True)
    assert not (R/'controller_status.json').exists(),'Do not duplicate controller'
    mem=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert mem==0,f'GPU3 occupied {mem}'
    sources={p.name:sha(p) for p in (R/'src').glob('*.py')};(R/'receipts/source_seal.json').write_text(json.dumps(sources,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(script,args,name):
        for n,h in sources.items():assert sha(R/'src'/n)==h
        (R/'controller_status.json').write_text(json.dumps(dict(active=True,pid=os.getpid(),stage=name,unix=time.time()),indent=2))
        with (R/'logs'/f'{name}.log').open('w') as log:subprocess.run([sys.executable,str(R/'src'/script),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    run('prepare_targets.py',[],'prepare_targets')
    run('preflight.py',[],'preflight')
    run('train_head.py',['--mode','spatial','--output',R/'runs/probe_continuous','--steps',6,'--save-every',3],'probe_continuous')
    run('train_head.py',['--mode','spatial','--output',R/'runs/probe_resumed','--steps',6,'--stop-after',3,'--save-every',3],'probe_first3')
    run('train_head.py',['--mode','spatial','--output',R/'runs/probe_resumed','--steps',6,'--save-every',3,'--resume'],'probe_resume3')
    a=torch.load(R/'runs/probe_continuous/last.pt',map_location='cpu',weights_only=False);b=torch.load(R/'runs/probe_resumed/last.pt',map_location='cpu',weights_only=False)
    fields=['model','optimizer','rng_cpu','rng_cuda','python_rng','numpy_rng','batch_rng','step','mode','total_steps','best_score','best_model','batch_plan'];checks={k:bool(eq(a[k],b[k])) for k in fields};assert all(checks.values()),checks
    (R/'receipts/actual_resume.json').write_text(json.dumps(dict(complete=True,actual_interruption=True,continuous=6,interrupted_at=3,resumed_updates=3,exact_fields=checks,scope='short integration check, not cross-device or long-run bit-parity'),indent=2))
    for mode in ['spatial','pooled']:run('train_head.py',['--mode',mode,'--output',R/'runs'/mode],'train_'+mode)
    done={m:json.loads((R/'runs'/m/'done.json').read_text()) for m in ['spatial','pooled']};assert done['spatial']['batch_plan_sha256']==done['pooled']['batch_plan_sha256']
    for m,d in done.items():assert all(v>0 for v in d['actual_weight_updates'].values()),(m,d['actual_weight_updates']);assert all(v>0 for v in d['gradient_sums'].values()),(m,d['gradient_sums'])
    (R/'controller_status.json').write_text(json.dumps(dict(complete=True,active=False,both_arms=True,same_batch_plan=True,production_admitted=False),indent=2))
if __name__=='__main__':main()
