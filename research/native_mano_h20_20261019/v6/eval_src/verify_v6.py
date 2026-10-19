"""Nonzero sampler gradients, GT isolation, and real interrupted resume."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_mano_shape_v6_20261019')
sys.path.insert(0,str(R/'src'));sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src');sys.path.insert(0,'/mnt/why/native_mano_rollout_v4_20261019/src')
from native_data_v6 import Data
from native_rollout import NativeRolloutDiT
from native_mano import NativeMANO

def same(x,y):
    if torch.is_tensor(x):return torch.equal(x.cpu(),y.cpu())
    if isinstance(x,np.ndarray):return np.array_equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
    return x==y
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert json.loads((R/'controller_status.json').read_text())['complete'];torch.set_num_threads(4)
    data=Data(supervision='compatible');codec=NativeMANO();b=data.batch(data.train[:3]);target=data.target(data.train[:3]);ck=torch.load(R/'runs/compatible/best_development.pt',weights_only=False,map_location='cuda');model=NativeRolloutDiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model'])
    with torch.autocast('cuda',dtype=torch.bfloat16):reference=model.sample(b,draws=4,steps=10,seed=190019)
    # Poison only loss labels; verify separately constructed inference inputs and samples.
    saved={k:v.clone() for k,v in data.labels.items() if torch.is_tensor(v) and v.is_floating_point()}
    for k in saved:data.labels[k].fill_(float('nan'))
    b2=data.batch(data.train[:3]);assert same(b,b2)
    with torch.autocast('cuda',dtype=torch.bfloat16):poisoned=model.sample(b2,draws=4,steps=10,seed=190019)
    assert torch.equal(reference,poisoned)
    for k,v in saved.items():data.labels[k].copy_(v)
    g=torch.Generator(device='cuda').manual_seed(190019)
    with torch.autocast('cuda',dtype=torch.bfloat16):actual=model.sample_train(b,g,draws=4,steps=10)
    assert torch.equal(reference,actual)
    result=codec.apply_residual(b['state'],b['betas'],b['right'],actual.mean(0));xyz=result['joints'];gt=data.pose_target(b,target,codec);loss=((xyz-xyz[:,:1])-(gt-gt[:,:1])).square()[:,1:].mean();loss.backward()
    gradients={prefix:max(float(p.grad.abs().max()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) for prefix in ['rgb_stem','spatial','temporal','head']};assert all(x>0 for x in gradients.values())
    parity=float((codec.decode_exported(codec.export_native(result))-result['joints']).abs().max());assert parity<1e-6
    (R/'receipts/GT_free_sampler_and_gradient.json').write_text(json.dumps(dict(complete=True,GT_label_poison_unchanged=True,deployment_gradient_sampler_torch_equal=True,sampler_only_gradient_max=gradients,native_export_redecode_max_m=parity),indent=2))
    del data,codec,model,b,b2,reference,actual,result,xyz,gt,target,ck;torch.cuda.empty_cache()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=f'{R}/src:/mnt/why/native_mano_dit_v2_20261019/src:/mnt/why/native_mano_rollout_v4_20261019/src')
    commands=[('continuous',[R/'eval_src/train_resume_probe_v6.py','--arm','protected','--supervision','compatible','--steps','6','--rollout','--output',R/'runs/resume_continuous']),
        ('interrupt',[R/'eval_src/train_resume_probe_v6.py','--arm','protected','--supervision','compatible','--steps','6','--rollout','--stop-after','3','--output',R/'runs/resume_interrupted']),
        ('resume',[R/'src/train_native_v6.py','--arm','protected','--supervision','compatible','--steps','6','--rollout','--resume','--output',R/'runs/resume_interrupted'])]
    for name,args in commands:
        with (R/'receipts'/f'resume_{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if name=='interrupt':assert torch.load(R/'runs/resume_interrupted/last.pt',weights_only=False,map_location='cpu')['step']==3
    a=torch.load(R/'runs/resume_continuous/last.pt',weights_only=False,map_location='cpu');b=torch.load(R/'runs/resume_interrupted/last.pt',weights_only=False,map_location='cpu')
    fields=['model','optimizer','rng_cpu','rng_cuda','noise_rng','rollout_rng','python_rng','numpy_rng','batch_plan_sha','selection','best_score','step','supervision']
    equality={k:same(a[k],b[k]) for k in fields};assert all(equality.values()),equality
    best0=torch.load(R/'runs/resume_continuous/best_development.pt',weights_only=False,map_location='cpu');best1=torch.load(R/'runs/resume_interrupted/best_development.pt',weights_only=False,map_location='cpu');assert same(best0['model'],best1['model'])
    report=dict(complete=True,actual_interruption=True,continuous_updates=6,interrupted_at=3,resumed_updates=3,exact_fields=equality,selected_best_weights_exact=True,
        trainer_sha256=sha(R/'src/train_native_v6.py'),note='deterministic native-target trainer; probes differ only checkpoint frequency/stop; resume uses exact frozen main trainer. Small integration check, not long-run quality or bit-reproduction across GPUs.')
    (R/'receipts/actual_resume_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
