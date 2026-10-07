"""OOF native1280 trajectory heads for the new proposal distribution."""
import argparse,json,time
import numpy as np,torch
from hand3d_v8_common import V7,load,batch,save,metrics
from hand3d_trajectory_data_v9 import RUN as V9,targets
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_visual_v8 import dense_bank
from train_hand3d_native_v10 import make_batch
from train_hand3d_v7 import initialize
from train_proposal_oof_v9 import probability
RUN=V7.parent/'native_oof_v10';RUN.mkdir(exist_ok=True)

@torch.no_grad()
def predict(model,data,bank,prob,ids,rollout=False):
    model.eval();outputs=[];stds=[]
    for start in range(0,len(ids),8):
        b=make_batch(data,bank,ids[start:start+8],prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610109+start) if rollout else model.predict(b,seed=202610109+start)
        outputs.append(p['xyz_camera_m'].float());stds.append(p['std_m'].float())
    return torch.cat(outputs),torch.cat(stds)

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=2000);p.add_argument('--rollout-steps',type=int,default=600);a=p.parse_args();torch.set_num_threads(4);seed=202610109;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);bank=dense_bank(a.device,len(data['world']));extra={k:v.to(a.device) for k,v in torch.load(V9/'trajectory_targets.pt',weights_only=False).items()};roles=np.asarray(data['roles']);subjects=np.asarray(data['subjects']);mapping=json.loads((V7/'risk_calibration.json').read_text())['folds'];train=torch.tensor([i for i in np.where(roles=='train')[0] if mapping[subjects[i]]!=a.fold],device=a.device);hold=torch.tensor([i for i in np.where(roles=='train')[0] if mapping[subjects[i]]==a.fold],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);prob=probability(data,a.fold,a.device)
    assert not set(subjects[train.cpu()])&set(subjects[hold.cpu()]);folder=RUN/f'fold{a.fold}';folder.mkdir(exist_ok=True);assert not (folder/'done.json').exists();model=NativeTrajectoryHand3D('dit',True).to(a.device);initialize(model,a.device);model.initialize_visual(a.device)
    config=dict(**vars(a),excluded_subjects=sorted(set(subjects[hold.cpu()])),train_subjects=sorted(set(subjects[train.cpu()])),native_channels=1280,spatial_cells=192,seed=seed,shared_encoder='Frozen WiLoR plus shared supervised2D initialization; not full-pipeline OOF',risk='Same excluded subject fold',supervision='Targets mask genuinely unlabeled slots; self/cross masks use observable frame availability only',selection='P0003 dev_select bounded camera+.75relative; held subjects never selected/fitted')
    save(folder/'config.json',config);started=time.time();history=[]
    def validate(stage,step,rollout):
        raw,_=predict(model,data,bank,prob,dev,rollout);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]]
        from bounded_policy_v8 import apply
        pred=apply(raw,base,dict(cap_m=.0095,strength=.5));m=metrics(pred,base,data['gt'][dev],data['valid'][dev]);row=dict(stage=stage,step=step,seconds=time.time()-started,**m);history.append(row);save(folder/'history.json',history);return m['camera_mm']+.75*m['relative_mm']
    for stage,steps,lr,size in [('denoise',a.steps,1e-4,8),('rollout',a.rollout_steps,2e-5,4)]:
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.04);best=validate(stage,0,stage=='rollout');torch.save(dict(model=model.state_dict(),step=0,stage=stage,config=config),folder/f'{stage}_best.pt')
        for step in range(1,steps+1):
            model.train();model.localization.eval();g=torch.Generator(device=a.device).manual_seed(seed+step);ids=train[torch.randint(len(train),(size,),device=a.device,generator=g)];torch.manual_seed(seed+step)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                b=make_batch(data,bank,ids,prob);target=targets(data,extra,ids);loss=model.rollout_loss(b,*target,seed+step) if stage=='rollout' else model.loss(b,*target)
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            for group in optimizer.param_groups:group['lr']=lr*min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
            if step%100==0:print(json.dumps(dict(fold=a.fold,stage=stage,step=step,seconds=time.time()-started,loss=float(loss.detach()))),flush=True)
            if step%500==0 or step==steps:
                score=validate(stage,step,stage=='rollout')
                if score<best:best=score;torch.save(dict(model=model.state_dict(),step=step,stage=stage,config=config),folder/f'{stage}_best.pt')
        ck=torch.load(folder/f'{stage}_best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model'])
    pred,std=predict(model,data,bank,prob,hold,True);torch.save(dict(indices=hold.cpu(),prediction=pred.cpu(),std=std.cpu(),probability=prob[hold].cpu(),excluded_subjects=config['excluded_subjects'],proposal_subjects=config['train_subjects'],selected_step=ck['step']),folder/'held_proposals.pt');torch.save(ck,folder/'best.pt');base=data['xyz_camera_bank'][data['feature_ids'][hold,8]];save(folder/'done.json',dict(complete=True,held_windows=len(hold),seconds=time.time()-started,metrics=metrics(pred,base,data['gt'][hold],data['valid'][hold])));print((folder/'done.json').read_text(),flush=True)

if __name__=='__main__':main()
