import json
import wilor_eval_common
import torch
from offline_kp_data import RUN,save
from offline_kp_model import KeypointCompleter,condition,artificial_gap
torch.set_num_threads(4)
data=torch.load(RUN/'windows.pt',weights_only=False);xy=data['xy'][:4].cuda();obs=data['observed'][:4].cuda();dt=data['dt'][:4].cuda()
fingers=torch.zeros(4,5,device='cuda',dtype=torch.bool);fingers[:,1]=True
b,selected=artificial_gap(xy,obs,dt,torch.full((4,),6,device='cuda'),fingers)
poisoned=xy.clone();poisoned[~b['observed']]=float('nan')
c=condition(poisoned,b['observed'],dt)
assert all(torch.equal(b[k],c[k]) for k in b),'Hidden coordinates leaked into interpolation or conditioning'
assert not b['observed'][:,6:12,[8,9,10,1]].any()
checks={}
for kind in ['dit','regression']:
    model=KeypointCompleter(kind).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][:4].cuda(),data['valid'][:4].cuda())
    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(model.head.weight.grad).all()
    model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):a=model.predict(b);d=model.predict(c)
    assert torch.equal(a['xy'],d['xy'])
    known=~b['missing'];assert torch.equal(a['xy'][known],xy[:,8][known])
    checks[kind]=dict(real_data_gradient=True,hidden_coordinate_nan_isolation=True,whole_gap_masked=True,known_coordinates_exactly_preserved=True)
save(RUN/'implementation_checks.json',checks);print(json.dumps(checks,indent=2))
