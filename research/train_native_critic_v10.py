"""Fit the critic to actual native-generator OOF proposals, not v9 distribution."""
import json,time
import numpy as np,torch
from torch.nn import functional as F
from hand3d_v8_common import V7,load,save,camera_bank
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_visual_v8 import dense_bank
from train_hand3d_native_v10 import make_batch
from train_native_rollout_v10 import RUN as PROPOSAL
from train_native_oof_v10 import RUN as OOF
from proposal_critic_v9 import ProposalCritic
from train_proposal_critic_v9 import make
RUN=V7.parent/'native_critic_v10'

@torch.no_grad()
def development(model,data,bank,prob,ids,params):
    model.eval();pred=[];std=[]
    for start in range(0,len(ids),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(make_batch(data,bank,ids[start:start+8],prob),seed=202610103+start)
        pred.append(p['xyz_camera_m'].float());std.append(p['std_m'].float())
    pred=torch.cat(pred);std=torch.cat(std);x,y=make(data,ids,prob,pred,std,params);return dict(x=x,y=y,indices=ids.cpu(),proposal=pred.cpu(),std=std.cpu())

def main():
    torch.set_num_threads(4);device='cuda:3';data=load(device);params=camera_bank().to(device);seed=202610111;torch.manual_seed(seed);np.random.seed(seed);started=time.time();roles=np.asarray(data['roles']);expected=set(np.where(roles=='train')[0]);xs=[];ys=[];seen=[];lineage=[]
    for fold in range(3):
        folder=OOF/f'fold{fold}';assert json.loads((folder/'done.json').read_text())['complete'];c=torch.load(folder/'held_proposals.pt',weights_only=False);ids=c['indices'].to(device);seen.extend(ids.cpu().tolist());prob=torch.zeros(len(roles),20,2,device=device);prob[ids]=c['probability'].to(device);x,y=make(data,ids,prob,c['prediction'].to(device),c['std'].to(device),params);xs.append(x);ys.append(y);lineage.append(dict(fold=fold,excluded=c['excluded_subjects'],trained=c['proposal_subjects'],windows=len(ids)))
    assert len(seen)==len(set(seen)) and set(seen)==expected;x=torch.cat(xs).to(device).float();y=torch.cat(ys).to(device);bank=dense_bank(device,len(data['world']));prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device);ck=torch.load(PROPOSAL/'rgb_dit/best.pt',weights_only=False,map_location=device);model=NativeTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model'])
    dev=torch.tensor(np.where(roles=='dev_select')[0],device=device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=device);d=development(model,data,bank,prob,dev,params);c=development(model,data,bank,prob,cal,params);torch.save(d,RUN/'critic_development.pt');torch.save(c,RUN/'critic_calibration.pt');del bank,model;torch.cuda.empty_cache();dx=d['x'].to(device).float();dy=d['y'].to(device)
    critic=ProposalCritic(x.shape[-1]).to(device);critic.mean.copy_(x.mean((0,1)));critic.scale.copy_(x.std((0,1)).clamp_min(.02));opt=torch.optim.AdamW(critic.parameters(),lr=.0005,weight_decay=.04);history=[];best=float('inf')
    def loss(pred,target):return F.smooth_l1_loss(pred[:,:2],target[:,:2],beta=.2)+F.binary_cross_entropy_with_logits(pred[:,2:],target[:,2:],pos_weight=torch.tensor([2.,2.,1.],device=device))
    config=dict(seed=seed,steps=2500,batch=128,lineage=lineage,native_condition_channels=1280,training='Actual held-subject native proposals including complete-sampling adaptation',scope='Proposal/risk heads OOF; shared supervised2D visual initialization and native pretraining are not fully excluded',selection='P0003 dev_select loss; dev_calibrate operating policy; fresh12clip metrics unopened',proposal_step=ck['step']);save(RUN/'critic_config.json',config)
    for step in range(1,2501):
        critic.train();ids=torch.randint(len(x),(128,),device=device);value=loss(critic(x[ids]),y[ids]);assert torch.isfinite(value);opt.zero_grad(set_to_none=True);value.backward();opt.step()
        if step%200==0 or step==2500:
            critic.eval()
            with torch.no_grad():score=float(loss(critic(dx),dy))
            history.append(dict(step=step,dev_loss=score,seconds=time.time()-started));save(RUN/'critic_history.json',history)
            if score<best:best=score;torch.save(dict(model=critic.state_dict(),dim=x.shape[-1],step=step,config=config),RUN/'critic.pt')
            print(json.dumps(history[-1]),flush=True)
    save(RUN/'critic_done.json',dict(complete=True,best_dev_loss=best,seconds=time.time()-started,actual_oof_windows=len(expected)))

if __name__=='__main__':main()
