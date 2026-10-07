import argparse,json,time
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from hand3d_trajectory_data_v9 import targets
from density_model_v13 import DensityTrajectoryHand3D,POLICY
from native_projection_policy_v11 import apply
DATA=V7.parent/'aligned_density_v13';RUN=V7.parent/'native_density_v13';INITIAL=V7.parent/'offline_hand3d_v10_rollout'

def make(data,bank,ids,prob):
    b=batch(data,ids,prob);b['rgb_native']=bank[data['feature_ids'][ids]];return b

@torch.no_grad()
def predict(model,data,bank,prob,ids):
    model.eval();out=[];spread=[]
    for start in range(0,len(ids),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(make(data,bank,ids[start:start+8],prob),seed=202610114+start)
        out.append(p['xyz_camera_m'].float());spread.append(p['std_m'].float())
    return torch.cat(out),torch.cat(spread)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['dit','regression'],required=True);ap.add_argument('--density',choices=['sparse','dense'],required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=1800);args=ap.parse_args();torch.set_num_threads(4)
    assert json.loads((DATA/'ready.json').read_text())['complete'];assert json.loads((DATA/f'risk_{args.density}/done.json').read_text())['complete'];run=RUN/f'{args.kind}_{args.density}';run.mkdir(parents=True,exist_ok=True);assert not (run/'done.json').exists()
    seed=202610114;torch.manual_seed(seed);np.random.seed(seed);raw=torch.load(DATA/f'{args.density}_data.pt',weights_only=False,mmap=True);data={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in raw.items()};bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(args.device);extra={k:v.to(args.device) for k,v in torch.load(DATA/'trajectory_targets.pt',weights_only=False,mmap=True).items()};roles=np.array(data['roles']);train=torch.tensor(np.where(roles=='train')[0],device=args.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=args.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=args.device);risk=torch.load(DATA/f'risk_{args.density}/risk_probabilities.pt',weights_only=False);prob=risk['train_oof'].to(args.device);ep=risk['joint'].to(args.device)
    ck=torch.load(INITIAL/f'rgb_{args.kind}/best.pt',weights_only=False,map_location=args.device);model=DensityTrajectoryHand3D(args.kind,True).to(args.device);model.load_state_dict(ck['model']);opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.04);history=[];started=time.time()
    config=dict(**vars(args),seed=seed,batch=4,initial_step=ck['step'],native_semantic_channels=1280,spatial_cells=192,policy=POLICY,velocity='Actual timestamp intervals,1ms numerical guard; no50ms truncation',same_original_centers=True,inputs='Original centerXYZ/RGB/ROI/camera/XY, new predicted tracks for all context, sparse/dense schedules',risk='Re-fit on corresponding schedule; train own-subject excluded risk head',selection='Dev_select feasible first and finger-relative first; no old/test/fresh rows in this dataset; dev_calibrate operating decision only',scope='Matched DiT sparse/dense fine tuning. Regression same center/steps but different pretrained initializer. Shared visual pretraining and2D training overlap not excluded',natural_only=True)
    save(run/'config.json',config)
    def validate(step):
        proposal,std=predict(model,data,bank,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];pred=apply(proposal,base,POLICY);m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);_,ok=score(m);key=(not ok,m['relative_mm'],m['camera_mm']);row=dict(step=step,seconds=time.time()-started,feasible=ok,**m,raw=metrics(proposal,base,data['gt'][dev],data['valid'][dev]),sampling_std_mean_mm=float(std.mean()*1000));history.append(row);save(run/'history.json',history);print(json.dumps(dict(kind=args.kind,density=args.density,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),step=step,kind=args.kind,density=args.density,config=config),run/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,args.steps+1):
        model.train();model.localization.eval();generator=torch.Generator(device=args.device).manual_seed(seed+step);ids=train[torch.randint(len(train),(4,),device=args.device,generator=generator)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(make(data,bank,ids,prob),*targets(data,extra,ids),seed+step)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        if step==1:
            norms=dict(native_projection=float(model.rgb_project.weight.grad.norm()),localization_projection=float(model.localization.project[1].weight.grad.norm()));assert all(np.isfinite(v) and v>0 for v in norms.values());save(run/'gradient_check.json',norms)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        for group in opt.param_groups:group['lr']=1e-5*min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/args.steps)))
        if step%100==0:print(json.dumps(dict(kind=args.kind,density=args.density,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%300==0 or step==args.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(run/'best.pt',weights_only=False,map_location=args.device);model.load_state_dict(ck['model']);proposal,std=predict(model,data,bank,ep,cal);torch.save(dict(indices=cal.cpu(),proposal=proposal.cpu(),std=std.cpu(),source_window_indices=data['source_window_indices'][cal].cpu()),run/'calibration.pt');save(run/'done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started))

if __name__=='__main__':main()
