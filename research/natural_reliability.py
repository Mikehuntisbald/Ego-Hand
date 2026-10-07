"""Reliability features use original predictions, original RGB, and metadata only."""
import json,hashlib
from pathlib import Path
import spatial_rgb_common as s
import torch,numpy as np
from torch import nn
from offline_kp_model import condition

RUN=s.common.ROOT/'experiments/natural_reliability_v4'

class RiskHead(nn.Module):
    def __init__(self,dim,box_only=False):
        super().__init__();self.box_only=box_only
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.net=nn.Linear(1,1) if box_only else nn.Sequential(nn.Linear(dim,96),nn.SiLU(),nn.Dropout(.2),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,1))
    def forward(self,x):
        z=((x-self.mean)/self.scale.clamp_min(.01)).clamp(-8,8)
        return self.net(z[...,:1] if self.box_only else z)[...,0]

def risk_features(xy,available,dt,roi,scores,rgb,positions):
    """rgb: fixed unsupervised projection of frozen ViT tokens, Bx17x192x32."""
    xy=torch.where(available[...,None],xy,torch.zeros_like(xy))
    base=xy[:,8];B=xy.shape[0];size=(roi[:,8,2:]-roi[:,8,:2]).mean(-1).clamp_min(.01)
    parts=[scores[:,8,None,None].expand(-1,20,1),available[:,8,:,None].float(),
        ((base-roi[:,8,None,:2])/size[:,None,None]).clamp(-3,3),
        size[:,None,None].expand(-1,20,1),torch.eye(20,device=xy.device)[None].expand(B,-1,-1)]
    for radius in [0,1,3]:
        obs=available.clone();obs[:,8-radius:9+radius]=False
        estimate=condition(xy,obs,dt)
        parts.extend([((base-estimate['linear'])/size[:,None,None]).clamp(-4,4),estimate['both_sides'][...,None].float(),estimate['has_context'][...,None].float()])
    local=[]
    for t in [5,8,11]:
        distance=((positions[:,t,None]-xy[:,t,:,None])/size[:,None,None,None]).square().sum(-1)
        weight=(-distance/(2*(1/12)**2)).softmax(-1)
        local.append(torch.einsum('bjs,bsc->bjc',weight,rgb[:,t].float()))
    parts.extend(local);parts.append(rgb[:,8].float().mean(1)[:,None].expand(-1,20,-1))
    features=torch.cat(parts,-1)
    assert torch.isfinite(features).all()
    return features

def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'features.pt').exists():
        previous=torch.load(RUN/'features.pt',weights_only=False,mmap=True)
        if previous.get('schema_version')==2:return
    rows=json.loads((s.OLD/'rows.json').read_text());records,index=s.records_and_index();w=torch.load(s.OLD/'windows.pt',weights_only=False)
    n=len(records)+1;g=torch.Generator().manual_seed(202610041)
    projection=torch.randn(1280,32,generator=g)/np.sqrt(1280)
    bank=torch.zeros(n,192,32,dtype=torch.float16);positions=torch.zeros(n,192,2);last=0
    for path in sorted((s.RUN/'dense_chunks').glob('*.pt')):
        c=torch.load(path,weights_only=False,mmap=True);assert c['start']==last;last=c['end'];lo,hi=c['start']+1,c['end']+1
        bank[lo:hi]=(c['features'][:,0].float()@projection).half();positions[lo:hi]=c['positions']
    assert last==len(records)
    scores=torch.tensor([0.]+[r['score'] for r in records]);out=[]
    for start in range(0,len(rows),96):
        sl=slice(start,start+96);f=index['feature_ids'][sl]
        out.append(risk_features(w['xy'][sl],w['observed'][sl],w['dt'][sl],index['roi'][f],scores[f],bank[f],positions[f]))
    x=torch.cat(out);err=(w['xy'][:,8]-w['gt']).norm(dim=-1)*1408
    valid=w['valid'].clone()&w['observed'][:,8];valid[:,5]=False
    subjects=sorted({r['subject'] for r in rows if r['role']=='train'});fold={s:i%3 for i,s in enumerate(subjects)}
    roles=[]
    for r in rows:
        role=r['role']
        if role=='development':role='dev_select' if int(hashlib.sha256(f"{r['sequence']}/{r['clip']}".encode()).hexdigest()[:8],16)%2==0 else 'dev_calibrate'
        roles.append(role)
    torch.save(dict(schema_version=2,x=x,error=err,valid=valid,roles=roles,subjects=[r['subject'] for r in rows],folds=fold,
        projection=projection,frame_bank=bank,positions=positions,scores=scores),RUN/'features.pt')
    s.save(RUN/'protocol.json',dict(task='Natural prediction reliability and selective correction',threshold_error_px=20,
        inputs='Original RGB and original WiLoR coordinates; no artificial occlusion or coordinate dropout',
        risk_rgb='32-channel fixed random projection of full frozen WiLoR ViT; no task-supervised visual head used for reliability inputs',
        splits={r:roles.count(r) for r in sorted(set(roles))},train_subject_folds=fold,
        calibration='P0003 clips split deterministically into model selection and calibration; no clip overlap',
        test='Previously inspected P0010/P0015 and 47 mined windows; reused audit only, not untouched evidence',
        trust='available is presence; confirmed alone is immutable; predicted reliability is separate',
        upstream_overlap='WiLoR pretraining overlap with HOT3D remains unknown',gt_in_inference=False))

def metrics(prob,error,valid):
    p=prob[valid].detach().cpu().numpy();y=(error[valid]>20).cpu().numpy();order=np.argsort(-p,kind='stable');yy=y[order]
    tp=np.cumsum(yy);precision=tp/np.arange(1,len(y)+1)
    ends=np.r_[np.flatnonzero(np.diff(p[order])!=0),len(p)-1]
    ap=(precision[ends]*np.diff(np.r_[0,tp[ends]])).sum()/max(1,yy.sum())
    return dict(points=len(y),bad_fraction=float(y.mean()),average_precision=float(ap),
        brier=float(((p-y)**2).mean()))

def train():
    prepare();d=torch.load(RUN/'features.pt',weights_only=False);device='cuda:0';x=d['x'].to(device);error=d['error'].to(device);valid=d['valid'].to(device)
    roles=np.array(d['roles']);subjects=np.array(d['subjects']);train_ix=np.where(roles=='train')[0];dev=np.where(roles=='dev_select')[0];cal=np.where(roles=='dev_calibrate')[0]
    assert len(dev)>50 and len(cal)>50
    all_logits={};oof=torch.zeros(len(x),20,device=device)
    for label,excluded,box in [('box',None,True),('joint',None,False)]+[(f'fold{i}',i,False) for i in range(3)]:
        torch.manual_seed(202610042);model=RiskHead(x.shape[-1],box).to(device)
        train=train_ix if excluded is None else np.array([i for i in train_ix if d['folds'][subjects[i]]!=excluded])
        ix=torch.tensor(train,device=device);sel=torch.tensor(dev,device=device)
        model.mean.copy_(x[ix].mean((0,1)));model.scale.copy_(x[ix].std((0,1)).clamp_min(.02))
        opt=torch.optim.AdamW(model.parameters(),lr=.002 if box else .0005,weight_decay=.03);best=float('inf')
        for step in range(1,1601):
            model.train();b=ix[torch.randint(len(ix),(128,),device=device)];logit=model(x[b]);m=valid[b]
            loss=nn.functional.binary_cross_entropy_with_logits(logit[m],(error[b][m]>20).float())
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%200==0:
                model.eval()
                with torch.no_grad():v=nn.functional.binary_cross_entropy_with_logits(model(x[sel])[valid[sel]],(error[sel][valid[sel]]>20).float()).item()
                if v<best:best=v;torch.save(dict(model=model.state_dict(),dim=x.shape[-1],box_only=box,step=step),RUN/f'risk_{label}.pt')
        ck=torch.load(RUN/f'risk_{label}.pt',weights_only=False,map_location=device);model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():logits=model(x)
        if excluded is not None:
            hold=torch.tensor([i for i in train_ix if d['folds'][subjects[i]]==excluded],device=device);oof[hold]=logits[hold]
        else:all_logits[label]=logits
        print(json.dumps(dict(risk=label,step=ck['step'],development=metrics(logits[sel].sigmoid(),error[sel],valid[sel]))),flush=True)
    calibration={};probs={}
    ci=torch.tensor(cal,device=device)
    for label,logits in all_logits.items():
        best=(float('inf'),1.)
        for temp in np.linspace(.5,3,51):
            loss=nn.functional.binary_cross_entropy_with_logits(logits[ci][valid[ci]]/temp,(error[ci][valid[ci]]>20).float()).item()
            if loss<best[0]:best=(loss,float(temp))
        temp=best[1];p=(logits/temp).sigmoid();probs[label]=p.cpu()
        calibration[label]=dict(temperature=temp,selection=metrics(p[dev],error[dev],valid[dev]),calibration=metrics(p[cal],error[cal],valid[cal]))
    # Training corrections receive subject-held-out reliability predictions.
    train_p=probs['joint'].clone();train_p[train_ix]=(oof[train_ix]/calibration['joint']['temperature']).sigmoid().cpu()
    torch.save(dict(**probs,train_oof=train_p),RUN/'probabilities.pt');s.save(RUN/'risk_calibration.json',calibration)

if __name__=='__main__':
    torch.set_num_threads(4);train()
