"""Actual observation smoke: gradients, coordinate inverse, GT isolation, local gate."""
import json
import wilor_eval_common
import torch
from pathlib import Path
from dit_v3_model import VisualResidual

run=Path('/mnt/why/HOT3D/experiments/dit_wilor_v3')
torch.set_num_threads(4);torch.manual_seed(202610031)
obs=torch.load(run/'observations.pt',map_location='cpu',weights_only=False,mmap=True)
chunk=torch.load(run/'feature_chunks/000000.pt',map_location='cpu',weights_only=False)
b={k:v[:4].to('cuda:0') for k,v in {**obs,**chunk}.items() if torch.is_tensor(v)}
delta=b['gt']-b['coarse'];roundtrip=(delta@b['transform'])@b['transform'].transpose(-1,-2)
assert torch.allclose(delta,roundtrip,atol=1e-6)
results=[]
for kind in ['dit','regression']:
    model=VisualResidual(kind).to('cuda:0')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.prediction_loss(b)
    assert torch.isfinite(loss),parts
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    model.eval()
    with torch.no_grad():
        p=model.propose(b,generator=torch.Generator(device='cuda:0').manual_seed(7))
        changed=dict(b);changed['gt']=torch.full_like(b['gt'],float('nan'))
        q=model.propose(changed,generator=torch.Generator(device='cuda:0').manual_seed(7))
        assert torch.equal(p,q),'Inference changed when GT was replaced'
        zero=model.apply(p,b,torch.zeros(4,20,device='cuda:0'))
        assert torch.equal(zero,b['coarse'])
        gates=torch.ones(4,20,device='cuda:0');gates[:,7]=0
        applied=model.apply(p,b,gates)
        assert torch.equal(applied[:,7],b['coarse'][:,7]),'Other joint/root updates bypassed closed point gate'
        assert p.shape==(4,20,3) and torch.isfinite(p).all()
    results.append(dict(kind=kind,real_data_backward=True,gt_inference_isolation=True,zero_gate_identity=True,local_protection=True,loss=float(loss.detach())))
    del model
(run/'model_checks.json').write_text(json.dumps(dict(coordinate_roundtrip=True,models=results),indent=2))
print(json.dumps(results,indent=2))
