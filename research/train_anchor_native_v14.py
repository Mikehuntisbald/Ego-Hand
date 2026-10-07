import argparse,json,time
import numpy as np,torch
from torch import nn
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from hand3d_trajectory_data_v9 import targets
from train_aligned_density_v13 import DATA,make
from anchor_reference_v14 import replace_with_anchor
from anchor_model_v14 import AnchoredTrajectoryHand3D
from density_model_v13 import POLICY
from native_projection_policy_v11 import apply
RUN=V7.parent/'anchor_native_v14_closed';INITIAL=V7.parent/'native_density_v13/dit_dense/best.pt'

def anchored_batch(data,sparse,bank,ids,prob,blend):
    b=make(data,bank,ids,prob);reference=batch(sparse,ids)
    # Only observed XYZ, camera-aligned time and availability form the prior.
    return replace_with_anchor(b,reference,blend)

@torch.no_grad()
def predict(model,data,sparse,bank,prob,ids,blend):
    model.eval();out=[];spread=[]
    for start in range(0,len(ids),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(anchored_batch(data,sparse,bank,ids[start:start+8],prob,blend),seed=202610115+start)
        out.append(p['xyz_camera_m'].float());spread.append(p['std_m'].float())
    return torch.cat(out),torch.cat(spread)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--blend',type=float,choices=[0.,.5,1.],required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args();torch.set_num_threads(4);seed=202610115;torch.manual_seed(seed);np.random.seed(seed)
    run=RUN/f'blend{args.blend:g}';run.mkdir(parents=True,exist_ok=True);assert not (run/'done.json').exists();raw=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in raw.items()};old=torch.load(DATA/'sparse_data.pt',weights_only=False,mmap=True)
    sparse={**data,**{k:old[k].to(args.device) for k in ['feature_ids','dt','xy','observed_2d']}};bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(args.device);extra={k:v.to(args.device) for k,v in torch.load(DATA/'trajectory_targets.pt',weights_only=False,mmap=True).items()};roles=np.array(data['roles']);train=torch.tensor(np.where(roles=='train')[0],device=args.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=args.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=args.device);risk=torch.load(DATA/'risk_dense/risk_probabilities.pt',weights_only=False);prob=risk['train_oof'].to(args.device);ep=risk['joint'].to(args.device)
    ck=torch.load(INITIAL,weights_only=False,map_location=args.device);model=AnchoredTrajectoryHand3D('dit',True).to(args.device);model.load_state_dict(ck['model']);nn.init.zeros_(model.head.weight);nn.init.zeros_(model.head.bias);opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.04);history=[];started=time.time()
    config=dict(**vars(args),seed=seed,steps=2400,denoise_steps=1200,real_sampling_steps=1200,batch=8,real_sampling_batch=4,initial='Same selected native-density model; head reset for all three controls',anchor='Whole-hand robust0.5s fit of sparse XYZ excluding original current frame; evaluated at actual dense query times',rgb='All17dense frames,192native1280-dimensional cells per frame; localization and semantic projection trainable;32-layer backbone frozen',selection='Dev-select finger-relative feasible first; dev-calibrate policy only; old/fresh withheld until weights fixed',scope='Matched0/0.5/1anchor coefficients. No GT in anchor/encoder. Complete3D trajectory targets; actual10step4draw deployed loss in second stage',policy=POLICY)
    save(run/'config.json',config)
    def validate(step):
        raw,std=predict(model,data,sparse,bank,ep,dev,args.blend);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];pred=apply(raw,base,POLICY);m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);_,ok=score(m);key=(not ok,m['relative_mm'],m['camera_mm']);row=dict(step=step,seconds=time.time()-started,feasible=ok,**m,raw=metrics(raw,base,data['gt'][dev],data['valid'][dev]),sampling_std_mean_mm=float(std.mean()*1000));history.append(row);save(run/'history.json',history);print(json.dumps(dict(blend=args.blend,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),step=step,blend=args.blend,config=config),run/'best.pt')
    validate(0);best=None
    for step in range(1,2401):
        model.train();model.localization.eval();g=torch.Generator(device=args.device).manual_seed(seed+step);size=8 if step<=1200 else 4;ids=train[torch.randint(len(train),(size,),device=args.device,generator=g)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            b=anchored_batch(data,sparse,bank,ids,prob,args.blend);gt,valid,uv,uv_valid=targets(data,extra,ids)
            loss=model.loss_without_velocity_floor(b,gt,valid,uv,uv_valid) if step<=1200 else model.rollout_loss(b,gt,valid,uv,uv_valid,seed+step)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        if step==2:
            norms=dict(native_projection=float(model.rgb_project.weight.grad.norm()),localization=float(model.localization.project[1].weight.grad.norm()));assert all(v>0 and np.isfinite(v) for v in norms.values());save(run/'gradient_check.json',norms)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        stage_step=step if step<=1200 else step-1200;initial_lr=5e-5 if step<=1200 else 1e-5
        for group in opt.param_groups:group['lr']=initial_lr*min(1.,.1+.9*stage_step/100)*(.1+.9*.5*(1+np.cos(np.pi*stage_step/1200)))
        if step%100==0:print(json.dumps(dict(blend=args.blend,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%300==0:
            key=validate(step)
            # Only actual complete-sampling trained weights are eligible.
            # The denoising-only stage remains diagnostic, including step0.
            if step>1200 and (best is None or key<best):best=key;checkpoint(step)
    ck=torch.load(run/'best.pt',weights_only=False,map_location=args.device);assert ck['step']>1200;model.load_state_dict(ck['model']);raw,std=predict(model,data,sparse,bank,ep,cal,args.blend);torch.save(dict(indices=cal.cpu(),proposal=raw.cpu(),std=std.cpu(),source_window_indices=data['source_window_indices'][cal].cpu()),run/'calibration.pt');save(run/'done.json',dict(complete=True,selected_step=ck['step'],real_sampling_trained_selected=True,seconds=time.time()-started))

if __name__=='__main__':main()
