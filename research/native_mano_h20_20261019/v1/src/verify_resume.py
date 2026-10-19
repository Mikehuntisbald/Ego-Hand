"""Exact saved model/Adam/RNG roundtrip, not a resumed-quality claim."""
import hashlib,json,random
import numpy as np,torch
from native_data import ROOT
from native_dit import NativeMANODiT

def digest(value):
    if torch.is_tensor(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if isinstance(value,dict):return {str(k):digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [digest(x) for x in value]
    return value

reports={}
for arm in ['regression','dit','protected']:
    ck=torch.load(ROOT/'runs'/arm/'last.pt',weights_only=False,map_location='cuda')
    model=NativeMANODiT(**ck['config']).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
    assert digest(model.state_dict())==digest(ck['model']);assert digest(optimizer.state_dict())==digest(ck['optimizer'])
    noise=torch.Generator(device='cuda')
    def restore():
        torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu());noise.set_state(ck['noise_rng'].cpu());random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
    restore();first=(torch.rand(32),torch.rand(32,device='cuda'),torch.randn(32,device='cuda',generator=noise),random.random(),np.random.rand(32))
    restore();second=(torch.rand(32),torch.rand(32,device='cuda'),torch.randn(32,device='cuda',generator=noise),random.random(),np.random.rand(32))
    assert all(torch.equal(a,b) for a,b in zip(first[:3],second[:3])) and first[3]==second[3] and np.array_equal(first[4],second[4])
    reports[arm]=dict(step=ck['step'],model_exact=True,Adam_exact=True,torch_CPU_CUDA_noise_RNG_exact=True,Python_NumPy_RNG_exact=True,
                      scope='saved-state roundtrip verified; no claim of a mid-training interruption/recovery test')
(ROOT/'receipts/resume_roundtrip.json').write_text(json.dumps(dict(complete=True,arms=reports),indent=2));print(json.dumps(reports))
