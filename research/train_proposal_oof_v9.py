"""Out-of-fold proposal heads for a critic trained on actual generalization errors.
The frozen2D spatial encoder was already supervised on the original train set;
this is proposal-head/risk OOF, not an entirely subject-excluded pipeline.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from hand3d_trajectory_data_v9 import RUN as V9,targets
from hand3d_trajectory_v9 import TrajectoryHand3D
from hand3d_v8_common import V7,load,batch,save,metrics
from hand3d_risk_v7 import Risk3D
from hand3d_data_v7 import risk_features
from train_hand3d_trajectory_v9 import predict
from train_hand3d_v7 import initialize
RUN=V9.parent/'proposal_critic_v9';RUN.mkdir(exist_ok=True)

@torch.no_grad()
def probability(data,fold,device):
    ck=torch.load(V7/f'risk_fold{fold}.pt',weights_only=False,map_location=device);model=Risk3D(ck['dim']).to(device).eval();model.load_state_dict(ck['model']);parts=[]
    temperature=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device)
    for start in range(0,len(data['roles']),96):
        ids=torch.arange(start,min(start+96,len(data['roles'])),device=device);parts.append((model(risk_features(batch(data,ids)))/temperature).sigmoid())
    return torch.cat(parts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=2000);a=p.parse_args();torch.set_num_threads(4);seed=202610095;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);extra={k:v.to(a.device) for k,v in torch.load(V9/'trajectory_targets.pt',weights_only=False).items()};roles=np.asarray(data['roles']);subjects=np.asarray(data['subjects']);mapping=json.loads((V7/'risk_calibration.json').read_text())['folds'];train=torch.tensor([i for i in np.where(roles=='train')[0] if mapping[subjects[i]]!=a.fold],device=a.device);hold=torch.tensor([i for i in np.where(roles=='train')[0] if mapping[subjects[i]]==a.fold],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);prob=probability(data,a.fold,a.device)
    assert len(train) and len(hold);assert not set(subjects[train.cpu()])&set(subjects[hold.cpu()]);model=TrajectoryHand3D('dit').to(a.device);initialize(model,a.device);folder=RUN/f'fold{a.fold}';folder.mkdir(exist_ok=True);assert not (folder/'done.json').exists()
    config=dict(**vars(a),batch=8,seed=seed,excluded_subjects=sorted(set(subjects[hold.cpu()])),train_subjects=sorted(set(subjects[train.cpu()])),initialization='Random proposal head; frozen shared native/spatial visual features; pretrained2D heatmap',risk='Risk fold excludes same held subjects; joint temperature from P0003 dev',scope='OOF proposal heads and risk only; shared supervised2D encoder/pretraining overlap remains',selection='P0003 dev_select raw camera+.75relative; held training subjects excluded from all proposal fitting and selection',code_sha256=hashlib.sha256(Path(__file__).with_name('hand3d_trajectory_v9.py').read_bytes()).hexdigest());save(folder/'config.json',config)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.04);history=[];started=time.time();best=float('inf')
    for step in range(1,a.steps+1):
        model.train();gen=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(8,),device=a.device,generator=gen)];torch.manual_seed(seed+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(batch(data,ix,prob),*targets(data,extra,ix))
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=1e-4*min(1.,.1+.9*step/200)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(fold=a.fold,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            pred=predict(model,data,prob,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);score=m['camera_mm']+.75*m['relative_mm'];history.append(dict(step=step,**m));save(folder/'history.json',history)
            if score<best:best=score;torch.save(dict(model=model.state_dict(),kind='dit',step=step,config=config),folder/'best.pt')
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);model.eval();out=[];std=[]
    with torch.no_grad():
        for start in range(0,len(hold),8):
            ix=hold[start:start+8]
            with torch.autocast('cuda',dtype=torch.bfloat16):value=model.predict(batch(data,ix,prob),seed=202610091+start)
            out.append(value['xyz_camera_m'].float());std.append(value['std_m'].float())
    prediction=torch.cat(out);torch.save(dict(indices=hold.cpu(),prediction=prediction.cpu(),std=torch.cat(std).cpu(),probability=prob[hold].cpu(),excluded_subjects=config['excluded_subjects'],proposal_subjects=config['train_subjects'],selected_step=ck['step']),folder/'held_proposals.pt')
    base=data['xyz_camera_bank'][data['feature_ids'][hold,8]];save(folder/'done.json',dict(complete=True,selected_step=ck['step'],held_windows=len(hold),seconds=time.time()-started,metrics=metrics(prediction,base,data['gt'][hold],data['valid'][hold]),scope=config['scope']));print((folder/'done.json').read_text(),flush=True)

if __name__=='__main__':main()
