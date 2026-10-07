"""Diagnose 2D supervision limits; oracle fitting is explicitly diagnostic."""
import json,collections
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from instance_parameter_model_v51_r2 import InstanceParameterHand
from online_parameter_model_v47 import OnlineVisual
from complete_instance_v51_r1 import pinhole_project
from cache_instance_conditions_v51 import RUN

def main():
    torch.set_num_threads(4);device='cuda:0';folder=RUN/'surgical_pose';saved=torch.load(folder/'inputs.pt',weights_only=False,mmap=True);meta=saved['metadata'];ids=[i for i,r in enumerate(meta) if r['split']=='dev'];pixels=np.load(folder/'pixels.npy',mmap_mode='r');inputs=saved['inputs']
    ck=torch.load(RUN/'paired_protocol/core_r1/dit_joint/best.pt',weights_only=False,map_location='cpu');model=InstanceParameterHand('dit',device).to(device);model.load_state_dict(ck['model']);model.eval()
    visual=OnlineVisual(device,True);visual.load_tail(ck['visual_tail']);states=[];sides=[];locations=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for begin in range(0,len(ids),4):
            ix=ids[begin:begin+4];b={k:v[ix].to(device) for k,v in inputs.items()};b['rgb_native']=torch.zeros(len(ix),17,192,1280,device=device);b['rgb_native'][:,8]=visual.encode_pixels([pixels[i] for i in ix],device)
            p=model.generate(b,2026100753+begin);states.append(p['state'][:,8].float().cpu());sides.append(p['right'].cpu());locations.append(p['encoded'][3]['xy'][:,8].float().cpu())
    state=torch.cat(states);side=torch.cat(sides);location=torch.cat(locations);torch.save(dict(indices=ids,state=state,right=side,location=location),folder/'sealed_diagnostic_states.pt')
    target=torch.load(folder/'targets.pt',weights_only=False);gt=target['xy'][ids].to(device);valid=target['valid'][ids].to(device);truthside=target['right'][ids].to(device);scale=(inputs['roi'][ids,8,2:]-inputs['roi'][ids,8,:2]).mean(-1).to(device).clamp_min(.01);params=inputs['camera_params'][ids].to(device)
    flat=state.flatten(1);report=dict(instances=len(ids),side_accuracy=float((side==truthside.cpu()).float().mean()),angle_saturation_fraction=float((flat[:,9:29].abs()>=.9999).float().mean()),localizer_PCK10=float((((location.to(device)-gt).norm(dim=-1)/scale[:,None]<=.1)&valid).sum()/valid.sum()),oracle_is_GT_assisted_diagnostic_only=True)
    for name,which in [('predicted_side',side.to(device)),('GT_side',truthside)]:
        x=flat.to(device);raw=nn.Parameter(torch.logit(((x[:,9:29].clamp(-.999,.999)+1)/2)));rotation=nn.Parameter(x[:,3:9].clone());root=nn.Parameter(x[:,:3].clone()*.1);beta=nn.Parameter(x[:,29:34].clone().clamp(-2,2));group=torch.arange(len(ids),device=device)
        optimizer=torch.optim.Adam([dict(params=[raw],lr=.04),dict(params=[rotation],lr=.01),dict(params=[root],lr=.001),dict(params=[beta],lr=.01)])
        for step in range(200):
            xyz=model.codec(raw,rotation,root,beta.clamp(-2,2),group,1-2*which.float());uv=pinhole_project(xyz,params)/1408
            error=F.smooth_l1_loss((uv-gt)/scale[:,None,None]*10,torch.zeros_like(uv),reduction='none',beta=.5).sum(-1);loss=(error*valid).sum()/valid.sum()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        e=(uv-gt).norm(dim=-1)/scale[:,None];report['oracle_'+name]=dict(PCK10=float(((e<=.1)&valid).sum()/valid.sum()),mean_error=float(e[valid].mean()))
    # Actual pure projected-FK loss must reach the live final visual block.
    ix=ids[:2];b={k:v[ix].to(device) for k,v in inputs.items()};b['rgb_native']=torch.zeros(2,17,192,1280,device=device);b['rgb_native'][:,8]=visual.encode_pixels([pixels[i] for i in ix],device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        p=model.generate(b,2026100754);uv=pinhole_project(p['xyz'][:,8],b['camera_params'])/1408;g=target['xy'][ix].to(device);v=target['valid'][ix].to(device);loss=(((uv-g).norm(dim=-1))*v).sum()/v.sum()
    grad=torch.autograd.grad(loss,visual.backbone.blocks[31].attn.qkv.weight)[0];report['pure_projected_FK_visual_gradient']=float(grad.norm());assert report['pure_projected_FK_visual_gradient']>0
    (folder/'mechanism_diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
