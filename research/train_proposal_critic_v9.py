import json,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from hand3d_v8_common import V7,load,batch,save,camera_bank
from hand3d_trajectory_v9 import TrajectoryHand3D
from train_trajectory_rollout_v9 import RUN as ROLLOUT
from proposal_critic_v9 import ProposalCritic,features,labels
RUN=V7.parent/'proposal_critic_v9'

@torch.no_grad()
def make(data,ids,prob,proposal,std,params):
    x=[];y=[]
    for alpha in [.25,.5,.75,1.]:
        for start in range(0,len(ids),64):
            ix=ids[start:start+64];b=batch(data,ix,prob);p=proposal[start:start+64];s=std[start:start+64];x.append(features(b,p,s,params[data['feature_ids'][ix,8]],alpha).half().cpu());y.append(labels(b['base'],b['base']+alpha*(p-b['base']),data['gt'][ix],data['valid'][ix]).cpu())
    return torch.cat(x),torch.cat(y)

@torch.no_grad()
def development(model,data,prob,ids,params):
    model.eval();proposal=[];std=[]
    for start in range(0,len(ids),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(batch(data,ids[start:start+8],prob),seed=202610093+start)
        proposal.append(p['xyz_camera_m'].float());std.append(p['std_m'].float())
    proposal=torch.cat(proposal);std=torch.cat(std);x,y=make(data,ids,prob,proposal,std,params)
    return dict(x=x,y=y,indices=ids.cpu(),proposal=proposal.cpu(),std=std.cpu(),strengths=[.25,.5,.75,1.])

def main():
    torch.set_num_threads(4);device='cuda:2';data=load(device);params=camera_bank().to(device);started=time.time();seed=202610097;torch.manual_seed(seed);np.random.seed(seed)
    target=torch.tensor(np.where(np.asarray(data['roles'])=='train')[0],device=device);train_x=[];train_y=[];seen=[];lineage=[]
    for fold in range(3):
        folder=RUN/f'fold{fold}';assert json.loads((folder/'done.json').read_text())['complete'];cache=torch.load(folder/'held_proposals.pt',weights_only=False);ids=cache['indices'].to(device);seen.extend(ids.cpu().tolist());prob=torch.zeros(len(data['roles']),20,2,device=device);prob[ids]=cache['probability'].to(device);x,y=make(data,ids,prob,cache['prediction'].to(device),cache['std'].to(device),params);train_x.append(x);train_y.append(y);lineage.append(dict(fold=fold,excluded_subjects=cache['excluded_subjects'],proposal_subjects=cache['proposal_subjects'],windows=len(ids)))
    assert len(seen)==len(set(seen)) and set(seen)==set(target.cpu().tolist());x=torch.cat(train_x).to(device).float();y=torch.cat(train_y).to(device)
    ck=torch.load(ROLLOUT/'rgb_dit/best.pt',weights_only=False,map_location=device);assert (ROLLOUT/'rgb_dit/training_done.json').exists();proposal_model=TrajectoryHand3D('dit').to(device).eval();proposal_model.load_state_dict(ck['model']);risks=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device);roles=np.asarray(data['roles']);dev=torch.tensor(np.where(roles=='dev_select')[0],device=device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=device)
    d=development(proposal_model,data,risks,dev,params);c=development(proposal_model,data,risks,cal,params);torch.save(d,RUN/'critic_development.pt');torch.save(c,RUN/'critic_calibration.pt');del proposal_model;torch.cuda.empty_cache();dx=d['x'].to(device).float();dy=d['y'].to(device)
    model=ProposalCritic(x.shape[-1]).to(device);model.mean.copy_(x.mean((0,1)));model.scale.copy_(x.std((0,1)).clamp_min(.02));opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.04);best=float('inf');history=[]
    def loss(pred,target):return F.smooth_l1_loss(pred[:,:2],target[:,:2],beta=.2)+F.binary_cross_entropy_with_logits(pred[:,2:],target[:,2:],pos_weight=torch.tensor([2.,2.,1.],device=device))
    config=dict(seed=seed,steps=2500,batch=128,lineage=lineage,training='Only actual OOF proposal heads, never in-sample joint proposals',scope='Shared2D spatial encoder and native pretraining are not subject-excluded; no full-pipeline OOF claim',outputs=['camera_gain/30mm','relative_gain/30mm','any_correct_camera_point_harmed','any_correct_relative_point_harmed','safe_useful_hand_proposal'],development='P0003 dev_select critic checkpoint; dev_calibrate threshold; subsequent v9fresh untouched',proposal_selected_step=ck['step'])
    save(RUN/'critic_config.json',config)
    for step in range(1,2501):
        model.train();ix=torch.randint(len(x),(128,),device=device);pred=model(x[ix]);value=loss(pred,y[ix]);assert torch.isfinite(value);opt.zero_grad(set_to_none=True);value.backward();opt.step()
        if step%200==0 or step==2500:
            model.eval()
            with torch.no_grad():score=float(loss(model(dx),dy));history.append(dict(step=step,dev_loss=score,seconds=time.time()-started));save(RUN/'critic_history.json',history)
            if score<best:best=score;torch.save(dict(model=model.state_dict(),dim=x.shape[-1],step=step,config=config),RUN/'critic.pt')
            print(json.dumps(history[-1]),flush=True)
    save(RUN/'critic_done.json',dict(complete=True,seconds=time.time()-started,best_dev_loss=best,actual_oof_windows=len(target),training_variants=len(x)))

if __name__=='__main__':main()
