import argparse,json,time
import numpy as np,torch
import spatial_rgb_common as s
from natural_reliability import RUN
from train_spatial_temporal import load
from natural_corrector import NaturalCorrector,natural_batch

@torch.inference_mode()
def validate(model,data,prob,ids):
    model.eval();errors=[];base=[];valid=[]
    for start in range(0,len(ids),48):
        ix=ids[start:start+48];b=natural_batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict(b)['xy']
        errors.append((out-data['gt'][ix]).norm(dim=-1)*1408);base.append((b['linear']-data['gt'][ix]).norm(dim=-1)*1408)
        m=data['valid'][ix].clone();m[:,5]=False;valid.append(m)
    e=torch.cat(errors);be=torch.cat(base);m=torch.cat(valid);bad=m&(be>20);good=m&(be<=10)
    return dict(all_px=float(e[m].mean()),bad_px=float(e[bad].mean()),base_bad_px=float(be[bad].mean()),base_all_px=float(be[m].mean()),
        harm_good_to_over20=float((e[good]>20).float().mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--steps',type=int,default=3000);p.add_argument('--force',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610043);np.random.seed(202610043)
    rows,data=load(a.device);r=torch.load(RUN/'features.pt',weights_only=False,mmap=True);roles=np.array(r['roles'])
    prob=torch.load(RUN/'probabilities.pt',weights_only=False)['train_oof'].to(a.device)
    train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device)
    folder=RUN/a.kind;folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists() and not a.force:return
    model=NaturalCorrector(a.kind).to(a.device)
    ck=torch.load(s.RUN/'sealed'/f'rgb_{a.kind}.pt',weights_only=False,map_location=a.device)
    mismatch=model.load_state_dict(ck['model'],strict=False);assert all(k.startswith('reliability.') for k in mismatch.missing_keys) and not mismatch.unexpected_keys
    opt=torch.optim.AdamW(model.parameters(),lr=6e-5,weight_decay=.04);history=[];best=float('inf');started=time.time()
    s.save(folder/'config.json',dict(kind=a.kind,steps=a.steps,seed=202610043,batch=48,lr=6e-5,
        selection='dev_select natural all-finger mean error; no hardcase/test selection',
        risk_training_inputs='3-fold subject-held-out predictions for original-coordinate reliability',
        inputs='Natural original RGB and all original coordinate predictions retained as soft conditions',
        target='GT residual correction, emphasis on naturally wrong points; no artificial pixel/coordinate masking',
        confirmed='Only explicit confirmed coordinates are immutable',encoder='Same full frozen WiLoR and validated spatial stem as v3'))
    for step in range(1,a.steps+1):
        model.train();ix=train[torch.randint(len(train),(48,),device=a.device)];b=natural_batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ix],data['valid'][ix])
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=6e-5*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%300==0 or step==a.steps:
            report=validate(model,data,prob,dev);history.append(dict(step=step,**report));s.save(folder/'history.json',history)
            print(json.dumps(history[-1]),flush=True)
            if report['all_px']<best:
                best=report['all_px'];torch.save(dict(model=model.state_dict(),kind=a.kind,step=step,selection=report),folder/'best.pt')
    s.save(folder/'done.json',dict(complete=True,best_development_all_px=best))

if __name__=='__main__':main()
