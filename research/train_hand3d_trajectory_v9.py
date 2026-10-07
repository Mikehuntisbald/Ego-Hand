import argparse,json,time
import numpy as np,torch
from hand3d_trajectory_data_v9 import RUN,targets
from hand3d_trajectory_v9 import TrajectoryHand3D
from hand3d_v8_common import V7,load,batch,save,metrics,score
from bounded_policy_v8 import apply
from train_hand3d_v7 import initialize

@torch.no_grad()
def predict(model,data,prob,ids):
    model.eval();parts=[]
    for start in range(0,len(ids),8):
        ix=ids[start:start+8]
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model.predict(batch(data,ix,prob),seed=202610091+start)
        parts.append(output['xyz_camera_m'].float())
    return torch.cat(parts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dit','regression'],default='dit');p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=4000);p.add_argument('--preflight',action='store_true');a=p.parse_args();torch.set_num_threads(4);seed=202610091;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);extra={k:v.to(a.device) for k,v in torch.load(RUN/'trajectory_targets.pt',weights_only=False).items()};roles=np.asarray(data['roles']);risks=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risks['train_oof'].to(a.device);ep=risks['joint'].to(a.device)
    train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device);model=TrajectoryHand3D(a.kind).to(a.device);initialize(model,a.device)
    if a.preflight:
        ix=train[:4];b=batch(data,ix,prob);target=targets(data,extra,ix);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        for step in range(2):
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,*target)
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        norms={k:float(p.grad.norm()) for k,p in model.named_parameters() if k in ['head.weight','rgb_project.weight','observation.0.weight']};assert all(np.isfinite(v) and v>0 for v in norms.values())
        model.eval();b['confirmed'][:,[0,5,9]]=True
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.predict(b);poison={**b,'gt':torch.ones_like(target[0])*999,'visibility':torch.zeros(4,17,20,device=a.device)};other=model.predict(poison)
        assert torch.equal(output['xyz_camera_m'],other['xyz_camera_m']);assert torch.equal(output['xyz_camera_m'][b['confirmed']],b['base'][b['confirmed']]);assert output['trajectory_xyz_camera_m'].shape==(4,17,20,3)
        save(RUN/('preflight_'+a.kind+'.json'),dict(passed=True,gt_poison_exact_invariance=True,locks_exact=True,trajectory_shape=[4,17,20,3],output_center_shape=[4,20,3],gradient_norms=norms,max_cuda_gb=torch.cuda.max_memory_allocated(torch.device(a.device))/1e9));print((RUN/('preflight_'+a.kind+'.json')).read_text(),flush=True);return
    folder=RUN/('rgb_'+a.kind);folder.mkdir(exist_ok=True);assert not (folder/'training_done.json').exists();history=[];started=time.time();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.04)
    config=dict(**vars(a),batch=8,width=192,depth=4,heads=6,seed=seed,lr=1e-4,output='Whole17frame20joint current-camera XYZ trajectory; center20x3 for annotation',tokens='17x21 root/relative tokens; raw spatial192cells/frame retained in decoder memory',supervision='All naturally labeled frames, 3D coordinates/velocities/bone lengths plus native2D heatmaps; no pixel masking',split='Frozen v7 train/dev; v8 inspected fresh is diagnostic only; next untouched clip manifest frozen before final',selection='Bounded dev_select feasible first; camera+.75relative; cal policies separately')
    save(folder/'config.json',config)
    def validate(step):
        raw=predict(model,data,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];pred=apply(raw,base,dict(cap_m=.0095,strength=.5));m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);key,feasible=score(m);row=dict(step=step,seconds=time.time()-started,feasible=feasible,**m,raw=metrics(raw,base,data['gt'][dev],data['valid'][dev]));history.append(row);save(folder/'history.json',history);print(json.dumps(dict(arm=folder.name,**row)),flush=True);return key
    def checkpoint(step):torch.save(dict(model=model.state_dict(),kind=a.kind,step=step,config=config),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,a.steps+1):
        model.train();gen=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(8,),device=a.device,generator=gen)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(batch(data,ix,prob),*targets(data,extra,ix))
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=1e-4*min(1.,.1+.9*step/200)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(arm=folder.name,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            key=validate(step)
            if key<best:best=key;checkpoint(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);raw=predict(model,data,ep,cal);torch.save(dict(indices=cal,raw=raw),folder/'calibration.pt');save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],seconds=time.time()-started))

if __name__=='__main__':main()
