import argparse,json,time,shutil
from pathlib import Path
import wilor_eval_common
import numpy as np,torch
from offline_rgb_data import RUN,save
from offline_kp_model import condition
from offline_rgb_model import RGBKeypointCompleter

def load(device):
    while not (RUN/'rgb_done.json').exists():time.sleep(10)
    rows=json.loads((RUN/'rows.json').read_text());data=torch.load(RUN/'windows.pt',weights_only=False)
    index=torch.load(RUN/'rgb_index.pt',weights_only=False);data.update(index)
    parts=[];last=0
    for path in sorted((RUN/'rgb_chunks').glob('*.pt')):
        chunk=torch.load(path,weights_only=False);assert chunk['start']==last;last=chunk['end'];parts.append(chunk['features'])
    bank=torch.cat(parts);data['bank']=torch.cat([torch.zeros_like(bank[:1]),bank]);del bank,parts
    return rows,{k:v.to(device) for k,v in data.items()}

def batch(data,ids,lengths,variants):
    xy=data['xy'][ids];obs=data['observed'][ids];dt=data['dt'][ids]
    starts=8-(lengths-1)//2;t=torch.arange(17,device=ids.device)[None]
    affected=(t>=starts[:,None])&(t<(starts+lengths)[:,None])
    # All old keypoints in the image-corrupted span are removed, including guesses
    # for visible points, to exclude indirect pre-occlusion-prediction leakage.
    b=condition(xy,obs&~affected[:,:,None],dt)
    fids=data['feature_ids'][ids];vids=torch.where(affected,variants[:,None],torch.zeros_like(fids))
    b.update(rgb=data['bank'][fids,vids],roi=data['roi'][fids],rgb_valid=fids>0)
    rect=data['rectangles'][fids[:,8],variants].float()
    roi=b['roi'][:,8];uv=(data['gt'][ids]-roi[:,None,:2])/(roi[:,None,2:]-roi[:,None,:2]).clamp_min(1e-5)*256
    hidden=(uv[...,0]>=rect[:,None,0]+2)&(uv[...,0]<rect[:,None,2]-2)&(uv[...,1]>=rect[:,None,1]+2)&(uv[...,1]<rect[:,None,3]-2)
    hidden=torch.where((variants==0)[:,None],torch.ones_like(hidden),hidden)&data['valid'][ids]
    hidden[:,5]=False
    return b,hidden

@torch.inference_mode()
def validate(model,data,ids):
    model.eval();total=0.;base=0.;count=0
    for gap in [1,3,6,9]:
        for start in range(0,len(ids),96):
            ix=ids[start:start+96];n=len(ix)
            b,mask=batch(data,ix,torch.full((n,),gap,device=ix.device),1+ix%4)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(b,samples=4)['xy']
            total+=float(((p-data['gt'][ix]).norm(dim=-1)*mask).sum());base+=float(((b['linear']-data['gt'][ix]).norm(dim=-1)*mask).sum());count+=int(mask.sum())
    return dict(hidden_px=total/max(count,1)*1408,linear_px=base/max(count,1)*1408,joints=count)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True,choices=['dit','regression']);ap.add_argument('--modality',required=True,choices=['rgb','tracks']);ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=6000);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610035);np.random.seed(202610035)
    folder=RUN/f'{a.modality}_{a.kind}';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows,data=load(a.device);train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train'],device=a.device)
    dev_all=np.array([i for i,r in enumerate(rows) if r['role']=='development']);rng=np.random.default_rng(202610035)
    dev=torch.tensor(np.sort(rng.choice(dev_all,min(320,len(dev_all)),replace=False)),device=a.device)
    model=RGBKeypointCompleter(a.kind,a.modality=='rgb').to(a.device)
    init=torch.load(wilor_eval_common.ROOT/f'experiments/offline_keypoint_diffusion_v1/{a.kind}/best.pt',map_location=a.device,weights_only=False)
    mismatch=model.load_state_dict(init['model'],strict=False);assert not mismatch.unexpected_keys
    opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=.03)
    save(folder/'config.json',dict(kind=a.kind,modality=a.modality,steps=a.steps,seed=202610035,batch=96,width=128,depth=3,
        train_windows=len(train),development_indices=dev.tolist(),initialization='v1 counterpart, same for RGB and tracks ablation',
        selection='Artificially occluded GT joints inside metadata-only rectangles; development only',test_used=False))
    history=[];best=float('inf');started=time.time()
    for step in range(1,a.steps+1):
        model.train();ix=train[torch.randint(len(train),(96,),device=a.device)]
        lengths=torch.tensor([1,3,6,9],device=a.device)[torch.randint(4,(len(ix),),device=a.device)]
        variants=torch.randint(1,6,(len(ix),),device=a.device)
        b,_=batch(data,ix,lengths,variants)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ix],data['valid'][ix])
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        for g in opt.param_groups:g['lr']=8e-5*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            metrics=validate(model,data,dev);entry=dict(step=step,**metrics);history.append(entry)
            if metrics['hidden_px']<best:
                best=metrics['hidden_px'];torch.save(dict(model=model.state_dict(),kind=a.kind,use_rgb=a.modality=='rgb',step=step,selection=metrics),folder/'best.pt')
            save(folder/'history.json',history);print(json.dumps(entry),flush=True)
    save(folder/'done.json',dict(complete=True,best_development_px=best,test_evaluated=False))

if __name__=='__main__':main()
