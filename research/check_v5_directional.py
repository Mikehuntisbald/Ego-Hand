import wilor_eval_common
import torch
from dit_v5_model import VisualResidual
from gate_dit_v5 import gate_loss
torch.set_num_threads(4)
model=VisualResidual('dit').cuda()
n=4;proposal=torch.randn(n,20,3,device='cuda')
q,_=torch.linalg.qr(torch.randn(n,3,3,device='cuda'))
b=dict(coarse=torch.randn(n,20,3,device='cuda'),transform=q)
zero=model.apply(proposal,b,torch.zeros_like(proposal))
assert torch.equal(zero,b['coarse'])
gates=torch.zeros_like(proposal);gates[:,:,2]=.5
delta=model.apply(proposal,b,gates)-b['coarse'];local=delta@q
assert local[:,:,:2].abs().max()<1e-6
assert torch.allclose(local[:,:,2],proposal[:,:,2]*.025,atol=1e-6)
features=torch.randn(n,20,266,device='cuda');gt=b['coarse']+.01*torch.randn_like(proposal)
loss=gate_loss(model,features,proposal*.05,q,b['coarse'],gt,torch.ones(n,20,device='cuda',dtype=torch.bool))
assert torch.isfinite(loss);loss.backward()
assert model.gate[-1].weight.grad is not None and torch.isfinite(model.gate[-1].weight.grad).all()
print('PASS: viewing-axis gate, zero identity, gate gradients')
