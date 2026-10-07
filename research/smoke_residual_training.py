import json,torch
from residual_models import ResidualModel
torch.manual_seed(20261003);device='cuda:3';torch.set_num_threads(4)
c=torch.randn(4,20,3,device=device)*.03;c[:,:,2]+=.5
gt=c+torch.randn_like(c)*.005;confidence=torch.full((4,21),.7,device=device);rgb=torch.randn(4,64,128,device=device)
for kind in ['dit','regression']:
    model=ResidualModel(kind=kind).to(device)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.prediction_loss(gt,c,confidence,rgb)
    loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):proposal=model.propose(c,confidence,rgb,steps=10,samples=4)
    proposal=proposal.clone()
    with torch.autocast('cuda',dtype=torch.bfloat16):gate_loss,parts=model.gate_loss(proposal,gt,c,confidence,rgb)
    gate_loss.backward();assert torch.isfinite(gate_loss)
    assert proposal.abs().max()<10,'Untrained v sampling should not produce exploding residuals'
    assert proposal[:,6].abs().max()==0
    print(json.dumps(dict(kind=kind,finite_backward=True,proposal_max_normalized=float(proposal.abs().max()),gate_loss=float(gate_loss.detach()))),flush=True)
