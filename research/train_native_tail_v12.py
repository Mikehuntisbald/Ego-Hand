"""Matched 3D training of native WiLoR final-four-block RGB conditions.

Unlike the earlier128-channel trial, this feeds full1280 native tokens directly
to the3D trajectory transformer. All17 time slots get live tail features.
"""
import argparse,json,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from hand3d_v8_common import V7,load,save,metrics,score
from hand3d_data_v7 import batch
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_trajectory_data_v9 import RUN as V9,targets
from hand3d_temporal_v7 import pack,unpack,WRIST
from finetune_visual_v5 import load_visual,prefix_bank
from native_projection_policy_v11 import apply
RUN=V7.parent/'native_tail_v12';PROPOSAL=V7.parent/'offline_hand3d_v10_rollout/rgb_dit'
POLICY=dict(strength=1.,cap_m=.0095,root_cap_m=.0095,relative_weight=4.)

class NativeTail(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.blocks=nn.ModuleList(list(backbone.blocks[28:]));self.norm=backbone.last_norm
    def forward(self,x):
        for block in self.blocks:x=block(x)
        return self.norm(x)[:,-192:]
    def replace(self,b,data,ids,prefix):
        f=data['feature_ids'][ids];unique,inverse=torch.unique(f,return_inverse=True);keep=unique>0
        z=torch.zeros(len(unique),192,1280,device=f.device,dtype=torch.float16)
        # Avoid executing absent-frame prefixes. Fixed16 padding gives reproducible
        # suffix arithmetic independent of the number of unique observations.
        selected=prefix[unique[keep]];parts=[]
        for start in range(0,len(selected),16):
            x=selected[start:start+16];n=len(x)
            if n<16:x=torch.cat([x,x[-1:].expand(16-n,-1,-1)])
            parts.append(self(x)[:n])
        z[keep]=torch.cat(parts).to(z.dtype);b['rgb_native']=z[inverse.flatten()].reshape(len(ids),17,192,1280)
        return b

def loss(model,b,gt,valid,gt_uv,uv_valid,seed):
    encoded=model.encode(b);delta=model.sample_trajectory(b,encoded,10,4,seed);raw=unpack(pack(b['xyz'])[None]+delta).mean(0);pred=apply(raw[:,8],b['base'],POLICY,b['confirmed'])
    canonical=valid.clone();canonical[:,:,WRIST]=False;m=canonical.float();cm=m[:,8]
    def errors(x,g):return (x-g).norm(dim=-1),((x-x[...,WRIST:WRIST+1,:])-(g-g[...,WRIST:WRIST+1,:])).norm(dim=-1)
    ce,re=errors(raw,gt);pe,pr=errors(pred,gt[:,8]);be,br=errors(b['base'],gt[:,8])
    fit=((ce/.03+.75*re/.03)*m).sum()/m.sum().clamp_min(1);final=((pe/.03+.75*pr/.03)*cm).sum()/cm.sum().clamp_min(1)
    goodc=canonical[:,8]&(be<=.01);goodr=canonical[:,8]&(br<=.01)
    protect=((pe-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((pr-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
    rawprotect=((ce[:,8]-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((re[:,8]-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
    dt=(b['dt'][:,1:]-b['dt'][:,:-1]).clamp_min(.05);diff=(raw[:,1:]-raw[:,:-1])-(gt[:,1:]-gt[:,:-1]);vm=(valid[:,1:]&valid[:,:-1]).float()
    velocity=(F.smooth_l1_loss(diff/(dt[:,:,None,None]*.1),torch.zeros_like(diff),reduction='none',beta=.5).sum(-1)*vm).sum()/vm.sum().clamp_min(1)
    return .4*fit+final+2*protect+.1*velocity+.1*model.loss(b,gt,valid,gt_uv,uv_valid)+.3*rawprotect

@torch.no_grad()
def predict(model,tail,prefix,data,prob,ids):
    model.eval();tail.eval();out=[]
    for start in range(0,len(ids),8):
        ix=ids[start:start+8]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            b=tail.replace(batch(data,ix,prob),data,ix,prefix);p=model.predict_rollout(b,seed=202610113+start)
        out.append(p['xyz_camera_m'].float())
    return torch.cat(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['ft','frozen'],required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=900);args=ap.parse_args()
    torch.set_num_threads(4);seed=202610113;torch.manual_seed(seed);np.random.seed(seed);run=RUN/args.mode;run.mkdir(parents=True,exist_ok=True);assert not (run/'done.json').exists()
    data=load(args.device);extra={k:v.to(args.device) for k,v in torch.load(V9/'trajectory_targets.pt',weights_only=False).items()};roles=np.asarray(data['roles']);train=torch.tensor(np.where(roles=='train')[0],device=args.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=args.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=args.device)
    risk=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risk['train_oof'].to(args.device);ep=risk['joint'].to(args.device)
    backbone,stem=load_visual(args.device);tail=NativeTail(backbone).to(args.device).eval();del backbone,stem;tail.requires_grad_(args.mode=='ft');prefix=prefix_bank(args.device,len(data['world']))
    ck=torch.load(PROPOSAL/'best.pt',weights_only=False,map_location=args.device);model=NativeTrajectoryHand3D('dit',True).to(args.device);model.load_state_dict(ck['model']);model.eval()
    # Native suffix/reference parity uses the same16-padded arithmetic.
    from hand3d_visual_v8 import dense_bank
    bank=dense_bank(args.device,len(data['world']));ix=dev[:8]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        b=batch(data,ix,ep);b['rgb_native']=bank[data['feature_ids'][ix]];old=model.predict_rollout(b,seed=seed)
        live=tail.replace(batch(data,ix,ep),data,ix,prefix);new=model.predict_rollout(live,seed=seed)
        parity=dict(feature_max_abs=float((b['rgb_native'].float()-live['rgb_native'].float()).abs().max()),prediction_mean_mm=float((old['xyz_camera_m']-new['xyz_camera_m']).norm(dim=-1).mean()*1000),prediction_max_mm=float((old['xyz_camera_m']-new['xyz_camera_m']).norm(dim=-1).max()*1000))
    save(run/'initial_parity.json',parity);assert parity['prediction_max_mm']<.5,parity;del bank;torch.cuda.empty_cache()
    groups=[dict(params=list(model.parameters()),lr=5e-6,initial_lr=5e-6)]
    if args.mode=='ft':groups.append(dict(params=list(tail.parameters()),lr=1e-6,initial_lr=1e-6))
    opt=torch.optim.AdamW(groups,weight_decay=.04);trainable=[p for g in groups for p in g['params']];history=[];started=time.time()
    config=dict(**vars(args),seed=seed,batch=4,initial_step=ck['step'],trainable_backbone_blocks=[28,29,30,31] if args.mode=='ft' else [],frozen_prefix_blocks=list(range(28)),native_channels=1280,native_cells=192,lr_head=5e-6,lr_tail=1e-6,policy=POLICY,selection='Dev-select only, feasible first and finger-relative error first; old and all three fresh batches diagnostic from now on',natural_only=True)
    save(run/'config.json',config)
    def validate(step):
        raw=predict(model,tail,prefix,data,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];pred=apply(raw,base,POLICY);m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);_,ok=score(m);key=(not ok,m['relative_mm'],m['camera_mm'])
        history.append(dict(step=step,seconds=time.time()-started,feasible=ok,**m,raw=metrics(raw,base,data['gt'][dev],data['valid'][dev])));save(run/'history.json',history);print(json.dumps(history[-1]),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),tail=tail.state_dict(),step=step,config=config),run/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,args.steps+1):
        model.train();model.localization.eval();tail.eval();g=torch.Generator(device=args.device).manual_seed(seed+step);ids=train[torch.randint(len(train),(4,),device=args.device,generator=g)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):b=tail.replace(batch(data,ids,prob),data,ids,prefix);value=loss(model,b,*targets(data,extra,ids),seed+step)
        assert torch.isfinite(value);opt.zero_grad(set_to_none=True);value.backward()
        if step==1 and args.mode=='ft':
            norms={str(i+28):sum(float(p.grad.float().square().sum()) for p in block.parameters() if p.grad is not None)**.5 for i,block in enumerate(tail.blocks)}
            assert all(np.isfinite(v) and v>0 for v in norms.values());tracked=tail.blocks[-1].attn.qkv.weight;before=tracked.detach().clone();save(run/'gradient_check.json',norms)
        torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step()
        if step==1 and args.mode=='ft':
            change=float((tracked-before).abs().max());assert change>0;save(run/'weight_update_check.json',dict(last_block_max_weight_change=change))
        for group in opt.param_groups:group['lr']=group['initial_lr']*(.1+.9*.5*(1+np.cos(np.pi*step/args.steps)))
        if step%100==0:print(json.dumps(dict(mode=args.mode,step=step,loss=float(value.detach()),seconds=time.time()-started)),flush=True)
        if step%150==0 or step==args.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(run/'best.pt',weights_only=False,map_location=args.device);model.load_state_dict(ck['model']);tail.load_state_dict(ck['tail']);raw=predict(model,tail,prefix,data,ep,cal);torch.save(dict(indices=cal.cpu(),proposal=raw.cpu()),run/'calibration.pt');save(run/'done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started,scope='Training complete does not establish independent severe-case recovery'))

if __name__=='__main__':main()
