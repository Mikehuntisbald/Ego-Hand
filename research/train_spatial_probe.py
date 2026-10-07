import argparse,json,time
import spatial_rgb_common as s
import numpy as np,torch
from spatial_rgb_model import SpatialHead

def load(device,raw=True):
    records,index=s.records_and_index();rows=json.loads((s.OLD/'rows.json').read_text())
    n=len(records)+1
    data={k:v.to(device) for k,v in index.items()}
    if raw:data['bank']=torch.zeros((n,6,192,1280),device=device,dtype=torch.float16)
    data['positions']=torch.zeros(n,192,2,device=device);last=0
    for path in sorted((s.RUN/'dense_chunks').glob('*.pt')):
        c=torch.load(path,weights_only=False,mmap=True);assert c['start']==last
        lo,hi=c['start']+1,c['end']+1;last=c['end']
        if raw:data['bank'][lo:hi]=c['features'].to(device)
        data['positions'][lo:hi]=c['positions'].to(device)
    assert last==len(records)
    gt,valid=s.targets(records,index,None);data['gt_frame']=gt.to(device);data['valid_frame']=valid.to(device)
    windows=torch.load(s.OLD/'windows.pt',weights_only=False)
    center=data['feature_ids'][:,8];error=(data['gt_frame'][center].cpu()-windows['gt']).abs()
    assert error[windows['valid']].max()<1e-5
    data.update({k:v.to(device) for k,v in windows.items()})
    return rows,data

@torch.inference_mode()
def validate(model,data,ids,diagnostics=False):
    model.eval();results={}
    for mode in (['clean','partial','shuffle_rgb','shuffle_space','zero_rgb'] if diagnostics else ['clean','partial']):
        errors=[];masks=[];visibility=[];spatial_dist=[]
        for start in range(0,len(ids),96):
            ix=ids[start:start+96];v=torch.zeros_like(ix) if mode=='clean' else 1+ix%4
            features=data['bank'][ix,v]
            if mode=='shuffle_rgb':features=data['bank'][ids[(torch.arange(start,start+len(ix),device=ids.device)+len(ids)//2)%len(ids)],v]
            if mode=='shuffle_space':features=features[:,torch.arange(191,-1,-1,device=ix.device)]
            if mode=='zero_rgb':features=torch.zeros_like(features)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(features,data['positions'][ix],data['roi'][ix])
            err=(out['xy']-data['gt_frame'][ix]).norm(dim=-1)*1408
            valid=data['valid_frame'][ix].clone();valid[:,5]=False
            roi=data['roi'][ix];uv=(data['gt_frame'][ix]-roi[:,None,:2])/(roi[:,None,2:]-roi[:,None,:2])*256
            rect=data['rectangles'][ix,v]
            hidden=(uv[...,0]>=rect[:,None,0]+2)&(uv[...,0]<rect[:,None,2]-2)&(uv[...,1]>=rect[:,None,1]+2)&(uv[...,1]<rect[:,None,3]-2)
            errors.append(err);masks.append(valid);visibility.append(hidden)
            spatial_dist.append(torch.cdist(data['gt_frame'][ix].float(),data['positions'][ix].float()).amin(-1)*1408)
        e=torch.cat(errors);m=torch.cat(masks);h=torch.cat(visibility);near=torch.cat(spatial_dist)
        results[mode]=dict(all_px=float(e[m].mean()),pck20=float((e[m]<=20).float().mean()),
            hidden_px=float(e[m&h].mean()) if (m&h).any() else None,
            uncovered_px=float(e[m&~h].mean()),nearest_cell_px=float(near[m].mean()))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--geometry-only',action='store_true');p.add_argument('--steps',type=int,default=4000);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610036);np.random.seed(202610036)
    folder=s.RUN/('probe_geometry' if a.geometry_only else 'probe_rgb');folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    started=time.time();rows,data=load(a.device)
    train=data['feature_ids'][[i for i,r in enumerate(rows) if r['role']=='train'],8].unique()
    dev=data['feature_ids'][[i for i,r in enumerate(rows) if r['role']=='development'],8].unique()
    model=SpatialHead(not a.geometry_only).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    s.save(folder/'config.json',dict(steps=a.steps,batch=96,train_frames=len(train),dev_frames=len(dev),use_rgb=not a.geometry_only,
        encoder='Full 32-block frozen WiLoR ViT; native 16x12 grid, 1280 channels; no handedness input',
        label_scope='GT for heatmap/coordinate losses only',test_used=False,selection='mean(clean all-point error, partial all-point error) on P0003',
        purpose='Verify independent RGB localization before training temporal fusion'))
    history=[];best=float('inf')
    for step in range(1,a.steps+1):
        model.train();ix=train[torch.randint(len(train),(96,),device=a.device)]
        roll=torch.rand(96,device=a.device);v=torch.randint(1,5,(96,),device=a.device)
        v=torch.where(roll<.4,0,v);v=torch.where(roll>.9,5,v)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(data['bank'][ix,v],data['positions'][ix],data['roi'][ix])
            loss=model.loss(out,data['gt_frame'][ix],data['valid_frame'][ix],data['positions'][ix],data['roi'][ix])
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=3e-4*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            report=validate(model,data,dev);score=(report['clean']['all_px']+report['partial']['all_px'])/2
            history.append(dict(step=step,score=score,**report));s.save(folder/'history.json',history);print(json.dumps(history[-1]),flush=True)
            if score<best:
                best=score;torch.save(dict(model=model.state_dict(),step=step,score=score,use_rgb=not a.geometry_only),folder/'best.pt')
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model'])
    report=validate(model,data,dev,diagnostics=True);s.save(folder/'done.json',dict(complete=True,step=ck['step'],development=report,test_evaluated=False))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
