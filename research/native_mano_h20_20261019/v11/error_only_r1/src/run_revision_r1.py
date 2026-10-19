import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
BASE=Path('/mnt/why/native_candidate_risk_v11_20261019');R=BASE/'error_only_r1';ARMS=['pose_base','RGB_base','RGB_preserve']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def eq(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b
def main():
    assert not (R/'controller_status.json').exists();assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0
    sources={p.name:sha(p) for p in (R/'src').glob('*.py')};(R/'receipts/source_seal.json').write_text(json.dumps(sources,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=os.pathsep.join([str(R/'src'),str(BASE/'src')]))
    def run(script,args,name):
        for n,h in sources.items():assert sha(R/'src'/n)==h
        (R/'controller_status.json').write_text(json.dumps(dict(active=True,pid=os.getpid(),stage=name,unix=time.time()),indent=2))
        with (R/'logs'/f'{name}.log').open('w') as log:subprocess.run([sys.executable,str(R/'src'/script),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    run('preflight_r1.py',[],'preflight')
    run('train_quality_r1.py',['--arm','RGB_preserve','--output',R/'runs/probe_continuous','--steps',6,'--save-every',3],'probe_continuous')
    run('train_quality_r1.py',['--arm','RGB_preserve','--output',R/'runs/probe_resumed','--steps',6,'--stop-after',3,'--save-every',3],'probe_first3')
    run('train_quality_r1.py',['--arm','RGB_preserve','--output',R/'runs/probe_resumed','--steps',6,'--save-every',3,'--resume'],'probe_resume3')
    a=torch.load(R/'runs/probe_continuous/last.pt',map_location='cpu',weights_only=False);b=torch.load(R/'runs/probe_resumed/last.pt',map_location='cpu',weights_only=False);fields=['model','optimizer','rng_cpu','rng_cuda','python_rng','numpy_rng','batch_rng','step','arm','total_steps','best_score','best_model','batch_plan'];check={k:bool(eq(a[k],b[k])) for k in fields};assert all(check.values());(R/'receipts/actual_resume.json').write_text(json.dumps(dict(complete=True,actual_interruption=True,exact_fields=check),indent=2))
    for arm in ARMS:run('train_quality_r1.py',['--arm',arm,'--output',R/'runs'/arm],'train_'+arm)
    done=[json.loads((R/'runs'/a/'done.json').read_text()) for a in ARMS];assert len({x['batch_plan_sha256'] for x in done})==1;(R/'controller_status.json').write_text(json.dumps(dict(active=False,complete=True,same_batch_plan=True,all_three_arms=True,production_admitted=False),indent=2))
if __name__=='__main__':main()
