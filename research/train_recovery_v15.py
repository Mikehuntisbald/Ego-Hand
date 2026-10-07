"""Controlled natural-hard sampling and deployment-policy training.

Train GT defines sampling difficulty only. It never enters inference conditions.
Fourth-batch and retained-failure data are absent from this training dataset.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_trajectory_data_v9 import targets
from train_aligned_density_v13 import make,predict
from recovery_model_v15 import RecoveryHand3D
from adaptive_projection_v14 import apply
from native_projection_policy_v11 import apply as conservative
from density_model_v13 import POLICY

RUN=V7.parent/'natural_recovery_v15';DATA=V7.parent/'aligned_density_v13';INITIAL=V7.parent/'native_density_v13/dit_dense/best.pt'

def difficulty(data,ids):
    base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];mask=data['valid'][ids].clone();mask[:,5]=False
    error=(((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    return error>40

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['uniform_adaptive','hard_adaptive','hard_conservative'],required=True);ap.add_argument('--device',required=True);ap.add_argument('--steps',type=int,default=900);args=ap.parse_args()
    torch.set_num_threads(4);seed=202610117;torch.manual_seed(seed);np.random.seed(seed);run=RUN/args.arm;run.mkdir(parents=True,exist_ok=True);assert not (run/'done.json').exists()
    policy=json.loads((V7.parent/'adaptive_projection_v14/dit_dense/fourth_seal.json').read_text())['policy']
    raw=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in raw.items()};bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(args.device);extra={k:v.to(args.device) for k,v in torch.load(DATA/'trajectory_targets.pt',weights_only=False,mmap=True).items()}
    roles=np.asarray(data['roles']);train=torch.tensor(np.where(roles=='train')[0],device=args.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=args.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=args.device)
    hard=train[difficulty(data,train)];assert len(hard)>0
    risk=torch.load(DATA/'risk_dense/risk_probabilities.pt',weights_only=False);prob=risk['train_oof'].to(args.device);ep=risk['joint'].to(args.device)
    ck=torch.load(INITIAL,weights_only=False,map_location=args.device);model=RecoveryHand3D(policy,args.arm!='hard_conservative').to(args.device);model.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=.04);started=time.time();history=[]
    config=dict(**vars(args),seed=seed,batch=4,initial_step=ck['step'],initial_sha256=hashlib.sha256(INITIAL.read_bytes()).hexdigest(),policy=policy,policy_not_retuned=True,training_hard_windows=len(hard),training_windows=len(train),hard_definition='Train GT baseline nonwrist mean relative error>40mm; sampling only, never inference',natural_only=True,hard_batch_fraction=.5 if args.arm!='uniform_adaptive' else None,selection='Dev_select feasibility first, then bad relative-point recovery on fixed difficult group, then hard relative mean and overall relative mean',evaluation_policy='Same sealed adaptive v14 policy for every arm; conservative arm changes only its training projection',scope='Existing train/dev only; fourth batch and old failures already read and excluded from training/selection; new independent batch required before adoption')
    save(run/'config.json',config)
    def validate(step):
        raw,std=predict(model,data,bank,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];b=make(data,bank,dev,ep);pred=apply(raw,b,policy)
        m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);_,ok=score(m);keep=difficulty(data,dev);hm=metrics(pred[keep],base[keep],data['gt'][dev][keep],data['valid'][dev][keep])
        key=(not ok,-hm['relative_bad_recovered20'],hm['relative_mm'],m['relative_mm'],m['camera_mm'])
        row=dict(step=step,seconds=time.time()-started,feasible=ok,**m,hard_windows=int(keep.sum()),hard=hm,raw=metrics(raw,base,data['gt'][dev],data['valid'][dev]));history.append(row);save(run/'history.json',history);print(json.dumps(dict(arm=args.arm,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),step=step,kind='dit',config=config),run/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,args.steps+1):
        model.train();model.localization.eval();generator=torch.Generator(device=args.device).manual_seed(seed+step)
        if args.arm=='uniform_adaptive':ids=train[torch.randint(len(train),(4,),device=args.device,generator=generator)]
        else:
            ids=torch.cat([train[torch.randint(len(train),(2,),device=args.device,generator=generator)],hard[torch.randint(len(hard),(2,),device=args.device,generator=generator)]])
        torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(make(data,bank,ids,prob),*targets(data,extra,ids),seed+step)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        if step==1:
            norms=dict(native_projection=float(model.rgb_project.weight.grad.norm()),localization_projection=float(model.localization.project[1].weight.grad.norm()));assert all(np.isfinite(v) and v>0 for v in norms.values());save(run/'gradient_check.json',dict(passed=True,**norms))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        for group in opt.param_groups:group['lr']=5e-6*min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/args.steps)))
        if step%100==0:print(json.dumps(dict(arm=args.arm,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%300==0 or step==args.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(run/'best.pt',weights_only=False,map_location=args.device);model.load_state_dict(ck['model']);proposal,std=predict(model,data,bank,ep,cal)
    torch.save(dict(indices=cal.cpu(),proposal=proposal.cpu(),std=std.cpu()),run/'calibration.pt')
    save(run/'done.json',dict(complete=True,selected_step=ck['step'],steps=args.steps,seconds=time.time()-started))

if __name__=='__main__':main()
