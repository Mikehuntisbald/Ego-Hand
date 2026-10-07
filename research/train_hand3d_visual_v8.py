import argparse,json,time
import numpy as np,torch
from hand3d_v8_common import RUN,V7,load,batch,save,predict,metrics,score,camera_bank
from hand3d_rollout_v8 import RolloutHand3D
from hand3d_visual_v8 import NativeSpatialStem,dense_bank

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['frozen','stem'],default='stem');p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=1200);a=p.parse_args();torch.set_num_threads(4);seed=202610083;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);params=camera_bank().to(a.device);roles=np.asarray(data['roles']);risks=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risks['train_oof'].to(a.device);ep=risks['joint'].to(a.device)
    train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device)
    folder=RUN/('dit_visual_'+a.mode);folder.mkdir(exist_ok=True);assert not (folder/'training_done.json').exists()
    initial=torch.load(RUN/'dit_rollout/best.pt',weights_only=False,map_location=a.device);model=RolloutHand3D('dit',True).to(a.device);model.load_state_dict(initial['model'])
    visual=NativeSpatialStem(a.device,dense_bank(a.device,len(data['world']))).to(a.device).eval();visual.requires_grad_(a.mode=='stem')
    model.eval();ix=dev[:8];b=batch(data,ix,ep)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        old=model.predict(b);oldrgb=b['rgb'].clone();live=visual.replace_batch(b,data,ix);new=model.predict(live)
        parity=dict(feature_max_abs=float((oldrgb.float()-live['rgb'].float()).abs().max()),feature_mean_abs=float((oldrgb.float()-live['rgb'].float()).abs().mean()),prediction_max_mm=float((old['xyz_camera_m']-new['xyz_camera_m']).norm(dim=-1).max()*1000),prediction_mean_mm=float((old['xyz_camera_m']-new['xyz_camera_m']).norm(dim=-1).mean()*1000))
    save(folder/'initial_parity.json',parity);assert parity['prediction_max_mm']<.5,parity
    groups=[dict(params=model.parameters(),lr=5e-6,initial_lr=5e-6)]
    if a.mode=='stem':groups.append(dict(params=visual.parameters(),lr=1e-5,initial_lr=1e-5))
    opt=torch.optim.AdamW(groups,weight_decay=.04);trainable=[p for g in groups for p in g['params']];history=[];started=time.time()
    config=dict(**vars(a),batch=8,seed=seed,initial_step=initial['step'],lr_head=5e-6,lr_stem=1e-5,backbone='32 native blocks frozen; full1280 features cached',trainable_rgb='LayerNorm1280 -> Linear128 + spatial convs, 192 cells retained' if a.mode=='stem' else 'same live spatial stem frozen',loss='Real deployed sampler and camera/relative preservation, no artificial occlusion',selection='Dev_select feasible first; fresh remains unopened')
    save(folder/'config.json',config)
    def validate(step):
        output=predict(model,data,ep,dev,visual=visual);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];m=metrics(output['xyz_camera_m'],base,data['gt'][dev],data['valid'][dev]);key,feasible=score(m)
        row=dict(step=step,seconds=time.time()-started,feasible=feasible,trust_mean=float(output['trust'].mean()),**m);history.append(row);save(folder/'history.json',history);print(json.dumps(dict(arm=folder.name,**row)),flush=True)
        return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),visual=visual.state_dict(),kind='dit',step=step,config=config),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,a.steps+1):
        model.train();visual.eval();gen=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(8,),device=a.device,generator=gen)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            b=visual.replace_batch(batch(data,ix,prob),data,ix);loss,parts=model.loss(b,data['gt'][ix],data['valid'][ix],data['gt_uv'][ix],data['uv_valid'][ix],params[data['feature_ids'][ix,8]],seed+step)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        if step==1 and a.mode=='stem':
            gradients={k:float(p.grad.norm()) for k,p in visual.named_parameters() if p.grad is not None};assert all(np.isfinite(v) and v>0 for v in gradients.values());save(folder/'gradient_check.json',gradients);before=visual.project[1].weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(trainable,1);opt.step()
        if step==1 and a.mode=='stem':
            delta=float((visual.project[1].weight-before).abs().max());assert delta>0;save(folder/'weight_update_check.json',dict(max_stem_weight_change=delta))
        for g in opt.param_groups:g['lr']=g['initial_lr']*min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(arm=folder.name,step=step,seconds=time.time()-started,loss=float(loss),**parts)),flush=True)
        if step%200==0 or step==a.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);visual.load_state_dict(ck['visual']);output=predict(model,data,ep,cal,visual=visual);torch.save(dict(indices=cal,predictions=output),folder/'calibration.pt');save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started))

if __name__=='__main__':main()
