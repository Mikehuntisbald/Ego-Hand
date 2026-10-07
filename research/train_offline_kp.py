import argparse,json,time,shutil
from pathlib import Path
import wilor_eval_common
import numpy as np
import torch
from offline_kp_data import RUN,save
from offline_kp_model import KeypointCompleter,artificial_gap

def load(device):
    rows=json.loads((RUN/'rows.json').read_text())
    data={k:v.to(device) for k,v in torch.load(RUN/'windows.pt',weights_only=False).items()}
    return rows,data

def augment(xy,gt):
    n=len(xy);device=xy.device
    angle=(torch.rand(n,device=device)-.5)*np.pi/3;scale=.8+.4*torch.rand(n,device=device)
    cosine=angle.cos()*scale;sine=angle.sin()*scale
    mat=torch.stack([cosine,-sine,sine,cosine],-1).reshape(n,2,2)
    mat[:,0]*=torch.where(torch.rand(n,device=device)>.5,1.,-1.)[:,None]
    offset=(torch.rand(n,1,1,2,device=device)-.5)*.1
    return torch.einsum('btjc,bcd->btjd',xy-.5,mat)+.5+offset,torch.einsum('bjc,bcd->bjd',gt-.5,mat)+.5+offset[:,0]

def case(data,ids,gap,all_points=False):
    n=len(ids);device=ids.device;fingers=torch.zeros(n,5,device=device,dtype=torch.bool)
    fingers[torch.arange(n,device=device),ids%5]=True
    if all_points:fingers[:]=True
    b,selected=artificial_gap(data['xy'][ids],data['observed'][ids],data['dt'][ids],torch.full((n,),gap,device=device),fingers)
    mask=b['missing']&data['valid'][ids]&selected;mask[:,5]=False
    return b,mask

@torch.inference_mode()
def validate(model,data,ids):
    model.eval();total=0.;base=0.;count=0
    for gap in [1,3,6,9]:
        for start in range(0,len(ids),128):
            ix=ids[start:start+128];b,mask=case(data,ix,gap)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.predict(b,samples=4)['xy']
            total+=float(((pred-data['gt'][ix]).norm(dim=-1)*mask).sum())
            base+=float(((b['linear']-data['gt'][ix]).norm(dim=-1)*mask).sum());count+=int(mask.sum())
    return dict(hidden_error_px=total/max(count,1)*1408,linear_error_px=base/max(count,1)*1408,hidden_joints=count)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True,choices=['dit','regression']);ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=5000);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610034);np.random.seed(202610034)
    folder=RUN/a.kind;folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows,data=load(a.device)
    train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train'],device=a.device)
    development=np.array([i for i,r in enumerate(rows) if r['role']=='development'])
    rng=np.random.default_rng(202610034);chosen=np.sort(rng.choice(development,min(384,len(development)),replace=False))
    dev=torch.tensor(chosen,device=a.device)
    model=KeypointCompleter(a.kind).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.03)
    save(folder/'config.json',dict(kind=a.kind,steps=a.steps,train_samples=len(train),development_indices=chosen.tolist(),seed=202610034,
        model=dict(width=128,depth=3),batch_size=128,lr=1e-4,selection='hidden-joint pixel error across four gaps on development only',test_used=False))
    for name in ['offline_kp_model.py','train_offline_kp.py']:shutil.copy2(Path(__file__).parent/name,folder/name)
    history=[];best=float('inf');started=time.time()
    for step in range(1,a.steps+1):
        model.train();ix=train[torch.randint(len(train),(128,),device=a.device)]
        xy,gt=augment(data['xy'][ix],data['gt'][ix]);obs=data['observed'][ix]
        lengths=torch.tensor([1,3,6,9],device=a.device)[torch.randint(4,(len(ix),),device=a.device)]
        ranks=torch.rand(len(ix),5,device=a.device).argsort(-1).argsort(-1)
        number=torch.randint(1,4,(len(ix),),device=a.device);fingers=ranks<number[:,None]
        fingers[torch.rand(len(ix),device=a.device)<.2]=True
        b,_=artificial_gap(xy,obs,data['dt'][ix],lengths,fingers)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,gt,data['valid'][ix])
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        for group in opt.param_groups:group['lr']=1e-4*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(kind=a.kind,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            metrics=validate(model,data,dev);entry=dict(step=step,**metrics);history.append(entry)
            if metrics['hidden_error_px']<best:
                best=metrics['hidden_error_px'];torch.save(dict(model=model.state_dict(),kind=a.kind,step=step,selection=metrics),folder/'best.pt')
            save(folder/'history.json',history);print(json.dumps(entry),flush=True)
    save(folder/'done.json',dict(complete=True,best_development_error_px=best,test_evaluated=False))

if __name__=='__main__':main()
