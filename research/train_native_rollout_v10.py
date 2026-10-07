import argparse,json,time
import numpy as np,torch
from hand3d_native_v10 import NativeTrajectoryHand3D
from train_hand3d_native_v10 import RUN as INITIAL,make_batch
from hand3d_trajectory_data_v9 import RUN as V9,targets
from hand3d_v8_common import V7,load,batch,save,metrics,score
from hand3d_visual_v8 import dense_bank
from bounded_policy_v8 import apply
RUN=V7.parent/'offline_hand3d_v10_rollout';RUN.mkdir(exist_ok=True)

@torch.no_grad()
def predict(model,data,bank,prob,ids):
    model.eval();out=[]
    for start in range(0,len(ids),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(make_batch(data,bank,ids[start:start+8],prob),seed=202610103+start)
        out.append(p['xyz_camera_m'].float())
    return torch.cat(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dit','regression'],default='dit');p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=1800);a=p.parse_args();torch.set_num_threads(4);seed=202610103;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);bank=dense_bank(a.device,len(data['world']));extra={k:v.to(a.device) for k,v in torch.load(V9/'trajectory_targets.pt',weights_only=False).items()};risks=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risks['train_oof'].to(a.device);ep=risks['joint'].to(a.device);roles=np.asarray(data['roles']);train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device)
    initial=torch.load(INITIAL/(a.kind+'_c1280')/'best.pt',weights_only=False,map_location=a.device);model=NativeTrajectoryHand3D(a.kind,True).to(a.device);model.load_state_dict(initial['model']);folder=RUN/('rgb_'+a.kind);folder.mkdir(exist_ok=True);assert not (folder/'training_done.json').exists();history=[];started=time.time();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.04)
    config=dict(**vars(a),batch=4,seed=seed,initial_step=initial['step'],native_channels=1280,spatial_cells=192,raw_protection_weight=.3,supervision='Entire3D trajectory on actual10step4draw samples, final bounded recovery and both raw/final good-point protection',input='Natural RGB/XYZ; target placeholders erased; missing-frame self/cross attention masks; no GT inference inputs',selection='Development only; all previously inspected clips diagnostic');save(folder/'config.json',config)
    def validate(step):
        raw=predict(model,data,bank,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];pred=apply(raw,base,dict(cap_m=.0095,strength=.5));m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);key,ok=score(m);row=dict(step=step,seconds=time.time()-started,feasible=ok,**m,raw=metrics(raw,base,data['gt'][dev],data['valid'][dev]));history.append(row);save(folder/'history.json',history);print(json.dumps(dict(kind=a.kind,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),kind=a.kind,native=True,step=step,config=config),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,a.steps+1):
        model.train();model.localization.eval();g=torch.Generator(device=a.device).manual_seed(seed+step);ids=train[torch.randint(len(train),(4,),device=a.device,generator=g)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(make_batch(data,bank,ids,prob),*targets(data,extra,ids),seed+step)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        if step==1:
            norm=float(model.rgb_project.weight.grad.norm());assert norm>0;save(folder/'gradient_check.json',dict(native_projection_gradient_norm=norm))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for group in opt.param_groups:group['lr']=2e-5*min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(kind=a.kind,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%300==0 or step==a.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);raw=predict(model,data,bank,ep,cal);torch.save(dict(indices=cal,raw=raw),folder/'calibration.pt');save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started))

if __name__=='__main__':main()
