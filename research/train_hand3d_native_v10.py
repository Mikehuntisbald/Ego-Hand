import argparse,json,time
import numpy as np,torch
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_trajectory_data_v9 import RUN as V9,targets
from hand3d_v8_common import V7,load,batch,save,metrics,score
from hand3d_visual_v8 import dense_bank
from train_hand3d_v7 import initialize
from bounded_policy_v8 import apply
RUN=V7.parent/'offline_hand3d_v10_native';RUN.mkdir(exist_ok=True)

def make_batch(data,bank,ids,prob):
    b=batch(data,ids,prob);b['rgb_native']=bank[data['feature_ids'][ids]];return b

@torch.no_grad()
def predict(model,data,bank,prob,ids):
    model.eval();out=[]
    for start in range(0,len(ids),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model.predict(make_batch(data,bank,ids[start:start+8],prob),seed=202610101+start)
        out.append(value['xyz_camera_m'].float())
    return torch.cat(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dit','regression'],default='dit');p.add_argument('--channels',type=int,choices=[128,1280],default=1280);p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=4000);a=p.parse_args();torch.set_num_threads(4);seed=202610101;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);bank=dense_bank(a.device,len(data['world']));extra={k:v.to(a.device) for k,v in torch.load(V9/'trajectory_targets.pt',weights_only=False).items()};risks=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risks['train_oof'].to(a.device);ep=risks['joint'].to(a.device);roles=np.asarray(data['roles']);train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device)
    arm=f'{a.kind}_c{a.channels}';folder=RUN/arm;folder.mkdir(exist_ok=True);assert not (folder/'training_done.json').exists();model=NativeTrajectoryHand3D(a.kind,a.channels==1280).to(a.device);initialize(model,a.device);model.initialize_visual(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.04);history=[];started=time.time();config=dict(**vars(a),batch=8,seed=seed,backbone='Native WiLoR32 blocks frozen',spatial='192 cells per frame retained; localization stem trained with3D trajectory losses',difference='Native1280 raw channels directly reach semantic attention vs2D-compressed128 channels; same localization branch and supervised trajectory',scope='Development test of a suspected2D bottleneck; v8/v9 fresh clips already inspected, not fresh final evidence',selection='Bounded dev_select feasible first; dev_calibrate policy only')
    save(folder/'config.json',config)
    def validate(step):
        raw=predict(model,data,bank,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];pred=apply(raw,base,dict(cap_m=.0095,strength=.5));m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);key,ok=score(m);row=dict(step=step,seconds=time.time()-started,feasible=ok,**m,raw=metrics(raw,base,data['gt'][dev],data['valid'][dev]));history.append(row);save(folder/'history.json',history);print(json.dumps(dict(arm=arm,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),kind=a.kind,native=a.channels==1280,step=step,config=config),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,a.steps+1):
        model.train();model.localization.eval();gen=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(8,),device=a.device,generator=gen)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(make_batch(data,bank,ix,prob),*targets(data,extra,ix))
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        if step==2:
            gradients=dict(native_projection=float(model.rgb_project.weight.grad.norm()),localization_projection=float(model.localization.project[1].weight.grad.norm()));assert all(np.isfinite(v) and v>0 for v in gradients.values());save(folder/'gradient_check.json',gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=1e-4*min(1.,.1+.9*step/200)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(arm=arm,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);raw=predict(model,data,bank,ep,cal);torch.save(dict(indices=cal,raw=raw),folder/'calibration.pt');save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started))

if __name__=='__main__':main()
