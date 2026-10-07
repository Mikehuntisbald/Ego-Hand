"""GT-only inverse-kinematics label refinement; never an inference model."""
import json,time
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_v8_common import V7,save
from parameter_codec_v31 import ParameterCodec,OUT
from joint_mano_model_v29 import six_to_rotation,rotation_to_six

def main():
    torch.set_num_threads(4);device='cuda:0'
    source=torch.load(OUT/'targets.pt',weights_only=False,mmap=True);data=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    codec=ParameterCodec(device);state=source['state'].clone();mask=source['mask'];window,slot=torch.nonzero(mask,as_tuple=True)
    initial=state[window,slot].flatten(1);R=data['rotation'][data['feature_ids'][window,8]];T=data['translation'][data['feature_ids'][window,8]]
    globalworld=R@six_to_rotation(initial[:,3:9]);root=torch.einsum('nj,nkj->nk',initial[:,:3]*.1,R)+T
    gtworld=torch.einsum('njc,nkc->njk',source['gt'][window,slot],R)+T[:,None]
    right=source['right'][window];pieces=[];start=time.time()
    for begin in range(0,len(window),4096):
        end=min(begin+4096,len(window));base=initial[begin:end].to(device);n=len(base)
        raw=nn.Parameter(torch.logit(((base[:,9:29]+1)/2).clamp(.001,.999)))
        rot0=rotation_to_six(globalworld[begin:end].to(device));rot=nn.Parameter(rot0.clone());beta=base[:,29:34]
        root0=root[begin:end].to(device);target=gtworld[begin:end].to(device);sign=1-2*right[begin:end].to(device).float();group=torch.arange(n,device=device)
        optimizer=torch.optim.Adam([dict(params=[raw],lr=.04),dict(params=[rot],lr=.005)])
        for step in range(100):
            pred=codec(raw,rot,root0,beta,group,sign)
            fit=F.smooth_l1_loss((pred-target)/.01,torch.zeros_like(pred),beta=.5)
            prior=.0005*((codec.angles(raw)[:,:20]-codec.angles(torch.logit(((base[:,9:29]+1)/2).clamp(.001,.999)))[:,:20])/.5).square().mean()
            loss=fit+prior+.0005*(six_to_rotation(rot)-six_to_rotation(rot0)).square().mean()
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.no_grad():
            result=base.clone();result[:,3:9]=rotation_to_six(six_to_rotation(rot))
            lo,hi=codec.joint_limits[:20].unbind(-1);result[:,9:29]=2*(codec.angles(raw)[:,:20]-lo)/(hi-lo)-1
            pieces.append(result.cpu())
        print(json.dumps(dict(done=end,total=len(window),seconds=time.time()-start)),flush=True)
    refined=torch.cat(pieces);worldrot=six_to_rotation(refined[:,3:9]);eyemat=R.transpose(-1,-2)@worldrot
    refined[:,3:9]=rotation_to_six(eyemat)
    # Root and shape targets remain identical; only pose is adapted to the
    # generic training-model coordinate/axis convention.
    state[window,slot].flatten(1) # advanced indexing returns a copy
    for begin in range(0,len(window),4096):
        w=window[begin:begin+4096];s=slot[begin:begin+4096]
        block=state[w,s].flatten(1);block[:,3:29]=refined[begin:begin+4096,3:29];state[w,s]=block.reshape(-1,21,3)
    assert torch.equal(state.flatten(2)[:,:,:3],source['state'].flatten(2)[:,:,:3])
    assert torch.equal(state.flatten(2)[:,:,29:],source['state'].flatten(2)[:,:,29:])
    torch.save(dict(source,state=state),OUT/'refined_targets.pt');save(OUT/'target_refinement.json',dict(complete=True,frames=len(window),steps=100,
        scope='Oracle teacherlabel IK only. GT pose/shape/XYZ never enter inference; originalXYZ evaluation labels unchanged. Root and staticshape targets unchanged.',seconds=time.time()-start))

if __name__=='__main__':main()
