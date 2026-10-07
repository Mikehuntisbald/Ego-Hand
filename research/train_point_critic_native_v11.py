import argparse,json,time
import numpy as np,torch
from torch.nn import functional as F
from hand3d_v8_common import V7,load,save,camera_bank
from hand3d_data_v7 import batch
from hand3d_visual_v8 import dense_bank
from train_native_oof_v10 import RUN as OOF
from train_proposal_critic_v9 import make
from point_critic_native_v11 import JointProposalCritic,visual_queries,point_labels
ROOT=V7.parent/'point_critic_native_v11'
SOURCE=V7.parent/'native_critic_v10'

@torch.no_grad()
def build(data,bank,ids,prob,raw,std,params,native):
    x,y=make(data,ids,prob,raw,std,params);points=[];masks=[];visual=[]
    for alpha in [.25,.5,.75,1.]:
        for start in range(0,len(ids),32):
            ix=ids[start:start+32];b=batch(data,ix,prob);p=raw[start:start+32]
            z,mask=point_labels(b['base'],b['base']+alpha*(p-b['base']),data['gt'][ix],data['valid'][ix]);points.append(z);masks.append(mask)
            if native:
                b['rgb_native']=bank[data['feature_ids'][ix]]
                visual.append(visual_queries(b,p,params[data['feature_ids'][ix,8]],alpha).half())
    return dict(x=x.to(ids.device).float(),y=y.to(ids.device),points=torch.cat(points),mask=torch.cat(masks),visual=torch.cat(visual) if native else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['native','geometry'],required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    torch.set_num_threads(4);device=args.device;native=args.arm=='native';run=ROOT/args.arm;run.mkdir(parents=True,exist_ok=True)
    seed=202610112;torch.manual_seed(seed);np.random.seed(seed);started=time.time();data=load(device);params=camera_bank().to(device);bank=dense_bank(device,len(data['world'])) if native else None
    pieces=[];seen=[];lineage=[];roles=np.asarray(data['roles'])
    for fold in range(3):
        cache=torch.load(OOF/f'fold{fold}/held_proposals.pt',weights_only=False);ids=cache['indices'].to(device);prob=torch.zeros(len(roles),20,2,device=device);prob[ids]=cache['probability'].to(device)
        assert not set(cache['excluded_subjects'])&set(cache['proposal_subjects'])
        seen+=ids.cpu().tolist();pieces.append(build(data,bank,ids,prob,cache['prediction'].to(device),cache['std'].to(device),params,native))
        lineage.append(dict(fold=fold,excluded=cache['excluded_subjects'],trained=cache['proposal_subjects'],windows=len(ids)))
    assert len(seen)==len(set(seen)) and set(seen)==set(np.where(roles=='train')[0])
    train={k:torch.cat([p[k] for p in pieces]) if pieces[0][k] is not None else None for k in pieces[0]};del pieces
    probability=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device)
    c=torch.load(SOURCE/'critic_development.pt',weights_only=False);ids=c['indices'].to(device)
    dev=build(data,bank,ids,probability,c['proposal'].to(device),c['std'].to(device),params,native)
    del bank,data;torch.cuda.empty_cache()
    model=JointProposalCritic(train['x'].shape[-1],native).to(device);model.mean.copy_(train['x'].mean((0,1)));model.scale.copy_(train['x'].std((0,1)).clamp_min(.02));opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.08)
    pos=train['points'][:,:,4:][train['mask']].mean(0);positive_weight=((1-pos)/pos.clamp_min(.001)).clamp(1,30);best=float('inf');history=[]
    def loss(hand,point,y,target,mask):
        global_loss=F.smooth_l1_loss(hand[:,:2],y[:,:2],beta=.2)+F.binary_cross_entropy_with_logits(hand[:,2:],y[:,2:],pos_weight=torch.tensor([2.,2.,1.],device=device))
        point_loss=F.smooth_l1_loss(point[:,:,:4][mask],target[:,:,:4][mask],beta=.2)+F.binary_cross_entropy_with_logits(point[:,:,4:][mask],target[:,:,4:][mask],pos_weight=positive_weight)
        return global_loss+.5*point_loss
    config=dict(arm=args.arm,seed=seed,steps=2200,batch=64,loss='Whole-hand decisions plus explicit point-error and point-harm supervision',native_queries=['current_observed','current_proposal','near_past_observed','near_future_observed','current_global','temporal_global'] if native else [],native_channels=1280 if native else 0,lineage=lineage,scope='Proposal/risk heads excluded from own held subjects. Shared visual pretraining and prior2D training not fully subject-excluded.',selection='dev_select loss, dev_calibrate operating policy; third12fresh unopened')
    save(run/'config.json',config)
    for step in range(1,2201):
        model.train();ix=torch.randint(len(train['x']),(64,),device=device);hand,point=model(train['x'][ix],train['visual'][ix] if native else None,True);value=loss(hand,point,train['y'][ix],train['points'][ix],train['mask'][ix]);assert torch.isfinite(value)
        opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if step%100==0 or step==2200:
            model.eval();total=0.;count=0
            with torch.no_grad():
                for start in range(0,len(dev['x']),64):
                    end=start+64;h,p=model(dev['x'][start:end],dev['visual'][start:end] if native else None,True);n=len(h);total+=float(loss(h,p,dev['y'][start:end],dev['points'][start:end],dev['mask'][start:end]))*n;count+=n
            val=total/count;history.append(dict(step=step,loss=val,seconds=time.time()-started));save(run/'history.json',history)
            if val<best:best=val;torch.save(dict(model=model.state_dict(),dim=train['x'].shape[-1],native=native,step=step,config=config),run/'critic.pt')
            print(json.dumps(history[-1]),flush=True)
    save(run/'done.json',dict(complete=True,best_dev_loss=best,seconds=time.time()-started,max_gpu_gb=torch.cuda.max_memory_allocated()/1e9,positive_weight=positive_weight.cpu().tolist()))

if __name__=='__main__':main()
