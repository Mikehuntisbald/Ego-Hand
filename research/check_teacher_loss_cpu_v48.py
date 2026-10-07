"""CPU head/FK sanity check while the full live-RGB preflight awaits GPU3."""
import json,time
import torch
from hand3d_v8_common import V7,save
from online_parameter_model_v47 import RUN as SOURCE,load_head
from parameter_codec_v31 import OUT,observation_batch
from protected_parameter_model_v48 import ProtectedParameterHand
RUN=V7.parent/'online_rgb_iterative_v48'
torch.set_num_threads(4);torch.manual_seed(2026100517);start=time.time()
data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True)
target=torch.load(SOURCE/'targets.pt',weights_only=False,mmap=True)
target['teacher_xyz']=torch.load(RUN/'teacher_centers.pt',weights_only=False)
train=torch.tensor([i for i,r in enumerate(data['roles']) if r=='train'])
gt=target['gt'][:,8];teacher=target['teacher_xyz'];valid=target['valid'][:,8].clone();valid[:,5]=False
rel=lambda x:x-x[:,5:6]
good=valid&(((teacher-gt).norm(dim=-1)<=.01)|((rel(teacher)-rel(gt)).norm(dim=-1)<=.01))
ix=train[good[train].sum(-1).argsort(descending=True)[:2]]
coarse=torch.load(V7.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True)
right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False)
risk=torch.load(SOURCE/'risk.pt',weights_only=False)['train_oof']
b=observation_batch(data,ix,risk,coarse,right,preserve_fitted=True)
native=torch.load(V7.parent/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True)
b['rgb_native']=native[data['feature_ids'][ix]].float()
plain,_,_=load_head('dit','cpu');ck=torch.load(SOURCE/'dit_joint/last.pt',weights_only=False,map_location='cpu');plain.load_state_dict(ck['model']);plain.eval()
protected=ProtectedParameterHand('dit','cpu',teacher_weight=4.);protected.load_state_dict(plain.state_dict());protected.eval()
tt={k:v[ix] for k,v in target.items()};seed=2026100991
with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
    a=plain.predict_parameters(b,seed);p=protected.predict_parameters(b,seed)
    dirty=protected.predict_parameters(dict(b,gt=torch.full_like(tt['gt'],float('nan')),teacher_xyz=torch.full_like(tt['teacher_xyz'],float('nan'))),seed)
assert torch.equal(a['state'],p['state']) and torch.equal(p['state'],dirty['state'])
state=torch.get_rng_state()
with torch.autocast('cpu',dtype=torch.bfloat16):loss,parts=protected.objective(b,tt,seed)
torch.set_rng_state(state)
with torch.autocast('cpu',dtype=torch.bfloat16):base,baseparts=plain.objective(b,tt,seed)
assert torch.allclose(loss.detach(),base.detach()+4*parts['teacher_guard'],atol=1e-5,rtol=1e-6)
with torch.autocast('cpu',dtype=torch.bfloat16):
    generated=protected.generate(b,seed);guard,_=protected.teacher_penalty(generated['xyz'][:,8],tt)
    probe,_=protected.teacher_penalty(generated['xyz'][:,8]+torch.tensor([.03,-.02,.02]),tt)
natural=torch.autograd.grad(guard,protected.semantic_head.weight,retain_graph=True)[0]
activated=torch.autograd.grad(probe,protected.semantic_head.weight,retain_graph=True)[0]
assert torch.isfinite(natural).all() and torch.isfinite(activated).all() and activated.norm()>0
loss.backward();assert torch.isfinite(protected.semantic_head.weight.grad).all()
result=dict(passed=True,scope='CPU head/FK with original cached natural features only; full live visual gradient preflight is separate',
    selected_train_indices=ix.tolist(),teacher_good_points_per_window=good[ix].sum(-1).tolist(),
    inference_state_identity_exact=True,teacher_and_GT_poison_exact=True,objective_equals_original_plus_4guard=True,
    natural_teacher_guard_gradient=float(natural.norm()),activated_guard_probe_gradient=float(activated.norm()),
    probe_only_FK_output_offset_m=[.03,-.02,.02],loss=float(loss.detach()),parts=parts,seconds=time.time()-start)
save(RUN/'cpu_loss_check.json',result);print(json.dumps(result),flush=True)
