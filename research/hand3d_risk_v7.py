import json,time
import numpy as np,torch
from torch import nn
from hand3d_data_v7 import RUN,load,batch,risk_features,save
from hand3d_temporal_v7 import WRIST

class Risk3D(nn.Module):
    def __init__(self,dim):
        super().__init__();self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.net=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Dropout(.2),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
    def forward(self,x):return self.net(((x-self.mean)/self.scale.clamp_min(.01)).clamp(-10,10))

def main():
    torch.set_num_threads(4);data=load('cuda:0');N=len(data['roles']);parts=[]
    with torch.no_grad():
        for start in range(0,N,96):
            ix=torch.arange(start,min(start+96,N),device='cuda:0');parts.append(risk_features(batch(data,ix)))
    x=torch.cat(parts);base=data['xyz_camera_bank'][data['feature_ids'][:,8]];gt=data['gt'];valid=data['valid'];roles=np.array(data['roles']);subjects=np.array(data['subjects'])
    camera=(base-gt).norm(dim=-1);relative=((base-base[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)
    y=torch.stack([camera>.02,relative>.02],-1).float();mask=valid[...,None].expand(-1,-1,2)
    train=np.where(roles=='train')[0];dev=torch.tensor(np.where(roles=='dev_select')[0],device=x.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=x.device)
    fold={s:i%3 for i,s in enumerate(sorted(set(subjects[train])))};oof=torch.zeros(N,20,2,device=x.device);all_logits=None;started=time.time()
    for excluded in [None,0,1,2]:
        name='all' if excluded is None else f'fold{excluded}';ix=torch.tensor([j for j in train if excluded is None or fold[subjects[j]]!=excluded],device=x.device)
        torch.manual_seed(202610072);model=Risk3D(x.shape[-1]).to(x.device);model.mean.copy_(x[ix].mean((0,1)));model.scale.copy_(x[ix].std((0,1)).clamp_min(.02))
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03);best=float('inf')
        for step in range(1,1601):
            model.train();ids=ix[torch.randint(len(ix),(128,),device=x.device)];logit=model(x[ids]);loss=nn.functional.binary_cross_entropy_with_logits(logit[mask[ids]],y[ids][mask[ids]])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%200==0:
                model.eval()
                with torch.no_grad():score=float(nn.functional.binary_cross_entropy_with_logits(model(x[dev])[mask[dev]],y[dev][mask[dev]]))
                if score<best:best=score;torch.save(dict(model=model.state_dict(),dim=x.shape[-1],step=step,folds=fold),RUN/f'risk_{name}.pt')
        ck=torch.load(RUN/f'risk_{name}.pt',weights_only=False,map_location=x.device);model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():logits=model(x)
        if excluded is None:all_logits=logits
        else:
            hold=torch.tensor([j for j in train if fold[subjects[j]]==excluded],device=x.device);oof[hold]=logits[hold]
        print(json.dumps(dict(risk=name,selected_step=ck['step'],dev_bce=best,seconds=time.time()-started)),flush=True)
    temps=[]
    for channel in range(2):
        best=(float('inf'),1.)
        for temp in np.linspace(.5,3,51):
            value=float(nn.functional.binary_cross_entropy_with_logits(all_logits[cal,:,channel][valid[cal]]/temp,y[cal,:,channel][valid[cal]]))
            if value<best[0]:best=(value,float(temp))
        temps.append(best[1])
    temperature=torch.tensor(temps,device=x.device);prob=(all_logits/temperature).sigmoid();train_prob=prob.clone();train_prob[train]=(oof[train]/temperature).sigmoid()
    torch.save(dict(joint=prob.cpu(),train_oof=train_prob.cpu()),RUN/'risk_probabilities.pt')
    save(RUN/'risk_calibration.json',dict(temperature=temps,outputs=['P(camera error >20mm)','P(wrist-relative error >20mm)'],folds=fold,inputs='3D temporal consistency, original RGB projection and YOLO score; no GT/visibility/GT side in features',selection='dev_select checkpoint, dev_calibrate temperatures; train OOF excludes subject fold'))
    save(RUN/'risk_done.json',dict(complete=True,seconds=time.time()-started))

if __name__=='__main__':main()
