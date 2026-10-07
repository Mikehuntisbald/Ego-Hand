"""Isolate real 2D RGB/mask supervision from scale-ambiguous 3D parameters."""
import json,hashlib,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from spatial_rgb_model import SpatialHead
from online_parameter_model_v47 import OnlineVisual,RUN as SOURCE
from cache_instance_conditions_v51 import RUN
from hand3d_rollout_v8 import project_fisheye624

class MaskLocalizer(SpatialHead):
    def __init__(self):
        super().__init__()
        self.instance=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.GELU(),nn.Conv2d(32,128,1))
        nn.init.zeros_(self.instance[-1].weight);nn.init.zeros_(self.instance[-1].bias)
    def forward(self,rgb,positions,roi,own,other):
        x=self.features(rgb)+self.instance(torch.stack([own,other],1).reshape(-1,2,16,12).float())
        return self.decode(x,positions,roi)

def main():
    torch.set_num_threads(4);torch.manual_seed(2026100756);device='cuda:0'
    folder=RUN/'paired_protocol/localizer_isolated_r2';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    checkpoint=RUN/'paired_protocol/core_r1/dit_joint/best.pt';ck=torch.load(checkpoint,weights_only=False,map_location='cpu')
    head=MaskLocalizer().to(device);missing,unexpected=head.load_state_dict(dict({k[13:]:v for k,v in ck['model'].items() if k.startswith('localization.')},**{k[12:]:v for k,v in ck['model'].items() if k.startswith('visual_head.')}),strict=False)
    assert not unexpected and all(k.startswith('instance.') for k in missing),(missing,unexpected)
    visual=OnlineVisual(device,False);visual.load_tail(ck['visual_tail']);visual.configure()
    surgery=torch.load(RUN/'surgical_pose/inputs.pt',mmap=True,weights_only=False);sinputs=surgery['inputs'];spixels=np.load(RUN/'surgical_pose/pixels.npy',mmap_mode='r')
    native=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True);ntarget=torch.load(SOURCE/'targets.pt',weights_only=False,mmap=True)
    masks=torch.load(RUN/'mask_conditions.pt',weights_only=False);indices=masks['indices'];npixels=np.load(SOURCE/'pixels.npy',mmap_mode='r')
    cached=torch.load(RUN/'paired_protocol/localizer_isolated_r1/features.pt',weights_only=False,mmap=True)
    nids=list(indices['train'])+list(indices['dev_select']);fids=native['feature_ids'][nids,8]
    assert cached['native_indices']==nids and cached['checkpoint']==str(checkpoint)
    sf=cached['surgical'];nf=cached['native']
    del visual;torch.cuda.empty_cache()
    # Coordinates/flags loaded as loss targets only; feature extraction is sealed first.
    torch.save(dict(surgical=sf,native=nf,native_indices=nids,checkpoint=str(checkpoint)),folder/'features.pt')
    starget=torch.load(RUN/'surgical_pose/targets.pt',weights_only=False,mmap=True)
    camera=torch.load(SOURCE.parent/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True)
    nxy=project_fisheye624(ntarget['gt'][nids,8],camera[fids])/1408
    nv=ntarget['valid'][nids,8]&torch.isfinite(nxy).all(-1)&(nxy>=0).all(-1)&(nxy<1).all(-1)
    nxy=torch.nan_to_num(nxy)
    def sb(ids):
        return [x.to(device) for x in [sf[ids],sinputs['positions'][ids,8],sinputs['roi'][ids,8],sinputs['instance_own'][ids,8],sinputs['instance_other'][ids,8]]]
    def nb(ids):
        ix=torch.tensor(nids)[ids];fid=fids[ids]
        return [x.to(device) for x in [nf[ids],native['positions_bank'][fid],native['roi'][fid],masks['own'][fid],masks['other'][fid]]]
    strain=[i for i,r in enumerate(surgery['metadata']) if r['split']=='train'];sdev=[i for i,r in enumerate(surgery['metadata']) if r['split']=='dev']
    ntrain=list(range(len(indices['train'])));ndev=list(range(len(ntrain),len(nids)))
    @torch.no_grad()
    def evaluate():
        head.eval();outputs={}
        for name,ids,fun,xy,valid in [('surgery',sdev,sb,starget['xy'],starget['valid']),('native',ndev,nb,nxy,nv)]:
            pieces=[]
            for begin in range(0,len(ids),32):
                with torch.autocast('cuda',dtype=torch.bfloat16):pieces.append(head(*fun(ids[begin:begin+32]))['xy'].float().cpu())
            pred=torch.cat(pieces);roi=fun(ids)[2].cpu();scale=(roi[:,2:]-roi[:,:2]).mean(-1)
            error=(pred-xy[ids]).norm(dim=-1)/scale[:,None];v=valid[ids]
            outputs[name]=dict(pred=pred,error=error,valid=v,PCK10=float((error[v]<=.1).float().mean()),points=int(v.sum()))
        # Seal predictions before the downstream metrics/report are exposed.
        return outputs
    baseline=evaluate();torch.save(baseline,folder/'baseline_predictions.pt')
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0002,weight_decay=.01);rng=np.random.default_rng(2026100756)
    history=[];selected=0;best_gain=-1.;start=time.time()
    for step in range(1,401):
        head.train();si=rng.choice(strain,16).tolist();ni=rng.choice(ntrain,8).tolist();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            so=head(*sb(si));no=head(*nb(ni));loss=head.loss(so,starget['xy'][si].to(device),starget['valid'][si].to(device),sb(si)[1],sb(si)[2])
            loss+=head.loss(no,nxy[ni].to(device),nv[ni].to(device),nb(ni)[1],nb(ni)[2])
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step()
        if step%50==0:
            output=evaluate();v=output['native']['valid'];good=v&(baseline['native']['error']<=.05)
            harmed=int((good&(output['native']['error']>.1)).sum());gain=output['surgery']['PCK10']-baseline['surgery']['PCK10']
            admitted=output['native']['PCK10']>=baseline['native']['PCK10'] and harmed==0 and gain>=.03
            entry=dict(step=step,surgery_PCK10=output['surgery']['PCK10'],native_PCK10=output['native']['PCK10'],native_correct_points=int(good.sum()),native_correct_harmed=harmed,admitted=admitted,seconds=time.time()-start)
            history.append(entry);print(json.dumps(entry),flush=True);(folder/'history.json').write_text(json.dumps(history,indent=2))
            torch.save(dict(model=head.state_dict(),optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state,step=step),folder/'last.pt')
            if admitted and gain>best_gain:
                selected=step;best_gain=gain;torch.save(dict(model=head.state_dict(),step=step,baseline_checkpoint=str(checkpoint)),folder/'best.pt');torch.save(output,folder/'selected_predictions.pt')
    (folder/'done.json').write_text(json.dumps(dict(complete=True,steps=400,selected_step=selected,admitted=selected>0,baseline={k:{x:v[x] for x in ['PCK10','points']} for k,v in baseline.items()},scope='RGB/mask spatial head only; temporal/FK/3D/visual backbone unchanged. Final 3D pipeline validation still required.',synthetic_occlusion=False,default_changed=False),indent=2))

if __name__=='__main__':main()
