import json,torch
from temporal_residual import TemporalResidualModel
torch.set_num_threads(4);torch.manual_seed(20261003);d='cuda:3';b=4
c=torch.randn(b,20,3,device=d)*.04;c[:,:,2]+=.5;conf=torch.full((b,21),.6,device=d);rgb=torch.randn(b,64,128,device=d)
context=dict(pose=c[:,None].repeat(1,4,1,1),confidence=conf[:,None].repeat(1,4,1),rgb=rgb[:,None].repeat(1,4,1,1),
             dt=torch.tensor([0,-.167,-.333,-.667],device=d)[None].repeat(b,1),valid=torch.tensor([True,True,False,True],device=d)[None].repeat(b,1))
for kind in ['dit','regression']:
    model=TemporalResidualModel(kind=kind).to(d);model.set_context(context)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=model.prediction_loss(c+.003*torch.randn_like(c),c,conf,rgb)
    loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    bad=dict(context);bad['dt']=context['dt']+.8
    try:model.set_context(bad)
    except AssertionError:pass
    else:raise RuntimeError('Future context accepted')
    print(json.dumps(dict(kind=kind,finite_backward=True,future_context_rejected=True)),flush=True)
