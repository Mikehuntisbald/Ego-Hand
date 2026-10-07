"""GT-free WiLoR XYZ -> fitted kinematic seed; labels only posthoc diagnostic."""
import json,time,torch
from torch import nn
from torch.nn import functional as F
from hand3d_v8_common import V7,save,metrics
from parameter_codec_v31 import OUT,ParameterCodec
from joint_mano_model_v29 import six_to_rotation,rotation_to_six


def main():
    torch.set_num_threads(4);device='cuda:0';run=V7.parent/'observation_ik_v38';run.mkdir(exist_ok=True)
    assert not (run/'done.json').exists()
    data=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    xyz=data['xyz_camera_bank'];right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False)
    initial=torch.load(OUT/'coarse_state_bank.pt',weights_only=False,mmap=True);result=initial.clone()
    codec=ParameterCodec(device);start=time.time();errors=[]
    save(run/'protocol.json',dict(scope='Inverse FK fit to original predicted WiLoR XYZ only. No GT poses/calibration/identity enters fitting.',
                                 steps=200,shape='Train-only5PCA',root='Jointly fit root, rotation, angles and shape to source observations',
                                 artificial_occlusion=False,default_changed=False))
    for begin in range(1,len(xyz),4096):
        end=min(begin+4096,len(xyz));z=initial[begin:end].to(device).flatten(1);n=len(z)
        raw=nn.Parameter(torch.logit(((z[:,9:29]+1)/2).clamp(.001,.999)))
        R=nn.Parameter(z[:,3:9].clone());root=nn.Parameter(z[:,:3].clone()*.1)
        sh=nn.Parameter(torch.zeros(n,5,device=device));target=xyz[begin:end].to(device)
        group=torch.arange(n,device=device);sign=1-2*right[begin:end].to(device).float()
        optimizer=torch.optim.Adam([dict(params=[raw],lr=.04),dict(params=[R],lr=.01),dict(params=[root],lr=.001),dict(params=[sh],lr=.01)])
        for step in range(200):
            beta=4*sh.tanh();pred=codec(raw,R,root,beta,group,sign)
            # A common robust fit for every observed point; no semantic GT
            # wrist correction or participant hand model is supplied.
            loss=F.smooth_l1_loss((pred-target)/.01,torch.zeros_like(pred),beta=.5)+.0001*beta.square().mean()
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.no_grad():
            pred=codec(raw,R,root,4*sh.tanh(),group,sign);errors.append((pred-target).norm(dim=-1).cpu()*1000)
            lo,hi=codec.joint_limits[:20].unbind(-1);z[:,:3]=root/.1;z[:,3:9]=rotation_to_six(six_to_rotation(R))
            z[:,9:29]=2*(codec.angles(raw)[:,:20]-lo)/(hi-lo)-1;z[:,29:34]=4*sh.tanh()
            result[begin:end]=z.reshape(n,21,3).cpu()
        print(json.dumps(dict(done=end,total=len(xyz),seconds=time.time()-start)),flush=True)
    torch.save(result,run/'coarse_state_bank.pt')
    error=torch.cat(errors);save(run/'done.json',dict(complete=True,rows=len(result)-1,mean_source_fit_mm=float(error.mean()),p95_source_fit_mm=float(error.quantile(.95)),seconds=time.time()-start,default_changed=False))
    # Ground truth is accessed only after the observation-only bank freezes.
    ids=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select']);fid=data['feature_ids'][ids,8]
    with torch.no_grad():
        state=result[fid].to(device)[:,None].expand(-1,17,-1,-1)
        p=codec.decode(state,right[fid].to(device))[:,8].cpu()
    m=metrics(p,data['original_base_for_evaluation'][ids],data['gt'][ids],data['valid'][ids])
    save(run/'posthoc_development_diagnostic.json',dict(scope='Observation fitting is GT-free. This posthoc metric is dev_select, not independent evidence or automatic adoption.',metrics=m))
    print(json.dumps(dict(fit=json.loads((run/'done.json').read_text()),diagnostic=m)),flush=True)


if __name__=='__main__':main()
