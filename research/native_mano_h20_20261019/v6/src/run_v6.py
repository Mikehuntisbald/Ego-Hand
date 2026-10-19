"""GPU3 matched shape/pose supervision study. No extra data or gate changes."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_mano_shape_v6_20261019');V5=Path('/mnt/why/native_mano_supervision_v5_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(x,y):
    if torch.is_tensor(x):return torch.equal(x,y)
    if isinstance(x,np.ndarray):return np.array_equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
    return x==y
def main():
    lock=R/'controller.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        memory=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert memory<100,'GPU3 occupied'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=f'{R}/src:/mnt/why/native_mano_supervision_v5_20261019/src:/mnt/why/native_mano_dit_v2_20261019/src:/mnt/why/native_mano_rollout_v4_20261019/src')
        sources={p.name:sha(p) for p in (R/'src').glob('*.py')};(R/'receipts/training_source_seal.json').write_text(json.dumps(sources,indent=2))
        protocol=dict(complete=True,hypothesis='Separate unreachable fixed-shape error from pose FK/velocity training; actual native GT remains selection, evaluation and accurate-point protection',
            arms=['control_native','compatible','compatible_protected'],steps=600,train_right_windows=155,select_right_windows=25,actual_sampler='4draw10step mean',
            locked_v5_targets_sha256=sha(V5/'data/native_targets.pt'),no_GT_inference=True,no_shape_head=True,no_new_data=True,visual_backbone_frozen=True,
            all_loss_coefficients_unchanged=True,no_protection_gate_relaxation=True,selection='actual native20 relative mm +20*original-good-harm fraction, each200',
            evaluation='same historically exposed six recordings; not a new blind test',visualization_candidate='compatible_guard',
            expansion='First run matched bounded pilot; enlarge data only after held development evidence supports the hypothesis')
        (R/'protocol.json').write_text(json.dumps(protocol,indent=2))
        jobs=[('pilot','dit','compatible',6),('control_native','dit','native',600),('compatible','dit','compatible',600),('compatible_protected','protected','compatible',600)]
        for name,arm,supervision,steps in jobs:
            folder=R/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(R/'src/train_native_v6.py'),'--arm',arm,'--supervision',supervision,'--steps',str(steps),'--rollout','--output',str(folder)]
            if (folder/'last.pt').exists():command+=['--resume']
            (R/'controller_status.json').write_text(json.dumps(dict(pid=os.getpid(),stage=name,command=command,started=time.time()),indent=2))
            with (folder/'console.log').open('a') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            done=json.loads((folder/'done.json').read_text());assert done['complete'] and done['FK_buffers_unchanged']
            assert all(x>0 for x in done['gradient_max'].values()) and all(x>0 for x in done['weight_update_max'].values())
            assert sources=={p.name:sha(p) for p in (R/'src').glob('*.py')},'Frozen source changed'
            if name=='control_native':
                current=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);prior=torch.load(V5/'runs/native_dit/last.pt',map_location='cpu',weights_only=False)
                fields=['model','optimizer','rng_cpu','rng_cuda','noise_rng','rollout_rng','python_rng','numpy_rng','batch_plan_sha','selection','best_score','step']
                equality={k:same(current[k],prior[k]) for k in fields};assert all(equality.values()),equality
                (R/'receipts/v5_control_exact.json').write_text(json.dumps(dict(complete=True,exact_fields=equality,matched_v5_last_sha256=sha(V5/'runs/native_dit/last.pt')),indent=2))
        plans={json.loads((R/'runs'/name/'done.json').read_text())['batch_plan_sha'] for name,_,_,steps in jobs if steps==600};assert len(plans)==1
        (R/'controller_status.json').write_text(json.dumps(dict(complete=True,stages=[j[0] for j in jobs],batch_plan_sha=plans.pop(),sources_frozen=True,production_admitted=False),indent=2))
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
