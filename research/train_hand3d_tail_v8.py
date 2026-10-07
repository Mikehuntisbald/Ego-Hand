import argparse,json,time
import numpy as np,torch
from hand3d_v8_common import RUN,V7,load,batch,save,predict,metrics,score,camera_bank
from hand3d_rollout_v8 import RolloutHand3D
from hand3d_visual_v8 import NativeSpatialStem,dense_bank
from finetune_visual_v5 import load_visual,Tail,prefix_bank

class LiveTail(Tail):
    def replace_batch(self,b,data,ids):
        f=data['feature_ids'][ids];unique,inverse=torch.unique(f,return_inverse=True);keep=unique>0
        features=torch.zeros(len(unique),192,128,device=f.device,dtype=torch.float16)
        features[keep]=self(self.prefix[unique[keep]]).half();b['rgb']=features[inverse.flatten()].reshape(len(ids),17,192,128);return b

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['frozen','tail'],default='tail');p.add_argument('--device',default='cuda:2');p.add_argument('--steps',type=int,default=600);a=p.parse_args();torch.set_num_threads(4);seed=202610085;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);params=camera_bank().to(a.device);roles=np.asarray(data['roles']);risks=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risks['train_oof'].to(a.device);ep=risks['joint'].to(a.device)
    train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device)
    folder=RUN/('dit_3d_'+a.mode);folder.mkdir(exist_ok=True);assert not (folder/'training_done.json').exists()
    initial=torch.load(RUN/'dit_visual_stem/best.pt',weights_only=False,map_location=a.device);model=RolloutHand3D('dit',True,cap_m=.0095).to(a.device);model.load_state_dict(initial['model'])
    reference=NativeSpatialStem(a.device,dense_bank(a.device,len(data['world']))).to(a.device).eval();reference.load_state_dict(initial['visual']);reference.requires_grad_(False)
    if a.mode=='tail':
        backbone,stem=load_visual(a.device);visual=LiveTail(backbone,stem).to(a.device).eval();del backbone,stem
        missing=visual.load_state_dict(initial['visual'],strict=False);assert all(k.startswith(('blocks.','norm.')) for k in missing.missing_keys) and not missing.unexpected_keys
        visual.prefix=prefix_bank(a.device,len(data['world']));visual.requires_grad_(True)
    else:visual=reference
    ix=dev[:4];model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        rb=reference.replace_batch(batch(data,ix,ep),data,ix);vb=visual.replace_batch(batch(data,ix,ep),data,ix)
        rp=model.predict(rb);vp=model.predict(vb);parity=dict(feature_mean_abs=float((rb['rgb'].float()-vb['rgb'].float()).abs().mean()),prediction_max_mm=float((rp['xyz_camera_m']-vp['xyz_camera_m']).norm(dim=-1).max()*1000))
    save(folder/'initial_parity.json',parity);assert parity['prediction_max_mm']<.5,parity
    if a.mode=='tail':del reference;torch.cuda.empty_cache()
    groups=[dict(params=model.parameters(),lr=5e-6,initial_lr=5e-6)]
    if a.mode=='tail':
        groups.extend([dict(params=list(visual.blocks.parameters())+list(visual.norm.parameters()),lr=1e-6,initial_lr=1e-6),dict(params=list(visual.project.parameters())+list(visual.spatial.parameters()),lr=5e-6,initial_lr=5e-6)])
    opt=torch.optim.AdamW(groups,weight_decay=.04);trainable=[p for g in groups for p in g['params']];history=[];started=time.time()
    config=dict(**vars(a),batch=4,seed=seed,initial_visual_stem_step=initial['step'],cap_m=.0095,lr_head=5e-6,lr_backbone=1e-6,lr_stem=5e-6,trainable_blocks=[28,29,30,31] if a.mode=='tail' else [],prefix='Native WiLoR blocks0..27 frozen; original unmasked RGB cached tokens',loss='Differentiable deployed10step4draw sampler, final bounded camera and relative errors',selection='Dev_select only; matched frozen visual control; fresh unopened')
    save(folder/'config.json',config)
    def validate(step):
        output=predict(model,data,ep,dev,size=4,visual=visual);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];m=metrics(output['xyz_camera_m'],base,data['gt'][dev],data['valid'][dev]);key,feasible=score(m);row=dict(step=step,seconds=time.time()-started,feasible=feasible,**m);history.append(row);save(folder/'history.json',history);print(json.dumps(dict(arm=folder.name,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),visual=visual.state_dict(),kind='dit',step=step,config=config),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,a.steps+1):
        model.train();visual.eval();gen=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(4,),device=a.device,generator=gen)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            b=visual.replace_batch(batch(data,ix,prob),data,ix);loss,parts=model.loss(b,data['gt'][ix],data['valid'][ix],data['gt_uv'][ix],data['uv_valid'][ix],params[data['feature_ids'][ix,8]],seed+step)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        if step==1 and a.mode=='tail':
            gradients={str(j+28):sum(float(p.grad.square().sum()) for p in block.parameters() if p.grad is not None)**.5 for j,block in enumerate(visual.blocks)};assert all(np.isfinite(v) and v>0 for v in gradients.values());save(folder/'gradient_check.json',gradients);before=visual.blocks[-1].attn.qkv.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(trainable,1);opt.step()
        if step==1 and a.mode=='tail':
            delta=float((visual.blocks[-1].attn.qkv.weight-before).abs().max());assert delta>0;save(folder/'weight_update_check.json',dict(last_block_max_weight_change=delta))
        for g in opt.param_groups:g['lr']=g['initial_lr']*min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%20==0:print(json.dumps(dict(arm=folder.name,step=step,seconds=time.time()-started,loss=float(loss),**parts)),flush=True)
        if step%100==0 or step==a.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);visual.load_state_dict(ck['visual']);output=predict(model,data,ep,cal,size=4,visual=visual);torch.save(dict(indices=cal,predictions=output),folder/'calibration.pt');save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started))

if __name__=='__main__':main()
