import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from hand3d_trajectory_data_v9 import RUN as BASE,targets
from hand3d_trajectory_v9 import TrajectoryHand3D
from hand3d_v8_common import V7,load,batch,save,metrics,score
from bounded_policy_v8 import apply
RUN=BASE.parent/'offline_hand3d_v9_rollout';RUN.mkdir(exist_ok=True)

@torch.no_grad()
def predict(model,data,prob,ids):
    model.eval();parts=[]
    for start in range(0,len(ids),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict_rollout(batch(data,ids[start:start+8],prob),seed=202610093+start)
        parts.append(out['xyz_camera_m'].float())
    return torch.cat(parts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dit','regression'],default='dit');p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=1600);p.add_argument('--preflight',action='store_true');a=p.parse_args();torch.set_num_threads(4);seed=202610093;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);extra={k:v.to(a.device) for k,v in torch.load(BASE/'trajectory_targets.pt',weights_only=False).items()};roles=np.asarray(data['roles']);risks=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risks['train_oof'].to(a.device);ep=risks['joint'].to(a.device);train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device)
    initial=torch.load(BASE/('rgb_'+a.kind)/'best.pt',weights_only=False,map_location=a.device);model=TrajectoryHand3D(a.kind).to(a.device);model.load_state_dict(initial['model']);folder=RUN/('rgb_'+a.kind);folder.mkdir(exist_ok=True)
    if a.preflight:
        ix=train[:4];b=batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(b,*targets(data,extra,ix),seed)
        loss.backward();gradient=float(model.head.weight.grad.norm());assert torch.isfinite(loss) and gradient>0
        save(folder/'rollout_preflight.json',dict(passed=True,loss=float(loss.detach()),gradient=gradient,max_cuda_gb=torch.cuda.max_memory_allocated(torch.device(a.device))/1e9));print((folder/'rollout_preflight.json').read_text(),flush=True);return
    assert json.loads((folder/'rollout_preflight.json').read_text())['passed'];opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.04);history=[];started=time.time()
    config=dict(**vars(a),batch=4,seed=seed,initial_step=initial['step'],sampler='10steps4draws differentiated; same vectorized sampler evaluated',input='Natural RGB/XYZ17frames; no manufactured visibility gaps',targets='Whole17frame XYZ trajectory, velocities, bounded central output and good-point protection',selection='Development only; v9 fresh clips unopened',code_sha256=hashlib.sha256(Path(__file__).with_name('hand3d_trajectory_v9.py').read_bytes()).hexdigest());save(folder/'config.json',config)
    def validate(step):
        raw=predict(model,data,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];pred=apply(raw,base,dict(cap_m=.0095,strength=.5));m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);key,ok=score(m);row=dict(step=step,seconds=time.time()-started,feasible=ok,**m,raw=metrics(raw,base,data['gt'][dev],data['valid'][dev]));history.append(row);save(folder/'history.json',history);print(json.dumps(dict(arm=folder.name,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),kind=a.kind,step=step,config=config),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,a.steps+1):
        model.train();gen=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(4,),device=a.device,generator=gen)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(batch(data,ix,prob),*targets(data,extra,ix),seed+step)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=2e-5*min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(arm=folder.name,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%200==0 or step==a.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);raw=predict(model,data,ep,cal);torch.save(dict(indices=cal,raw=raw),folder/'calibration.pt');save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started))

if __name__=='__main__':main()
