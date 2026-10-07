import argparse,json,time
import spatial_rgb_common as s
import numpy as np,torch
from offline_kp_model import condition
from spatial_temporal_model import SpatialTemporalCompleter

def load(device):
    rows=json.loads((s.OLD/'rows.json').read_text());data=torch.load(s.OLD/'windows.pt',weights_only=False)
    records,index=s.records_and_index();data.update(index);n=len(records)+1
    data={k:v.to(device) for k,v in data.items()}
    data['bank']=torch.zeros(n,6,192,128,device=device,dtype=torch.float16)
    data['positions_bank']=torch.zeros(n,192,2,device=device);last=0
    for path in sorted((s.RUN/'spatial_chunks').glob('*.pt')):
        c=torch.load(path,weights_only=False,mmap=True);assert c['start']==last;last=c['end']
        lo,hi=c['start']+1,c['end']+1
        data['bank'][lo:hi]=c['features'].to(device);data['positions_bank'][lo:hi]=c['positions'].to(device)
    assert last==len(records)
    return rows,data

def batch(data,ids,lengths,variants,drop_tracks=False):
    xy=data['xy'][ids];obs=data['observed'][ids];dt=data['dt'][ids]
    starts=8-(lengths-1)//2;t=torch.arange(17,device=ids.device)[None]
    affected=(t>=starts[:,None])&(t<(starts+lengths)[:,None])
    obs=obs&~affected[:,:,None]
    if drop_tracks:
        # Same augmentation in the RGB and tracks control. Twenty percent of
        # windows must learn from RGB/box geometry without coordinate shortcuts.
        drop_window=torch.rand(len(ids),device=ids.device)<.2
        drop_joint=torch.rand(len(ids),20,device=ids.device)<.15
        obs=obs&~drop_window[:,None,None]&~drop_joint[:,None]
    b=condition(xy,obs,dt);fids=data['feature_ids'][ids]
    vids=torch.where(affected,variants[:,None],torch.zeros_like(fids))
    b.update(rgb=data['bank'][fids,vids],positions=data['positions_bank'][fids],roi=data['roi'][fids],rgb_valid=fids>0)
    rect=data['rectangles'][fids[:,8],variants].float();roi=b['roi'][:,8]
    uv=(data['gt'][ids]-roi[:,None,:2])/(roi[:,None,2:]-roi[:,None,:2]).clamp_min(1e-5)*256
    hidden=(uv[...,0]>=rect[:,None,0]+2)&(uv[...,0]<rect[:,None,2]-2)&(uv[...,1]>=rect[:,None,1]+2)&(uv[...,1]<rect[:,None,3]-2)
    hidden=torch.where((variants==0)[:,None],torch.ones_like(hidden),hidden)&data['valid'][ids]
    hidden[:,5]=False
    return b,hidden

@torch.inference_mode()
def validate(model,data,ids):
    model.eval();total=base=count=0
    for gap in [1,3,6,9]:
        for start in range(0,len(ids),48):
            ix=ids[start:start+48];b,m=batch(data,ix,torch.full_like(ix,gap),1+ix%4)
            with torch.autocast('cuda',dtype=torch.bfloat16):o=model.predict(b,samples=4)
            total+=float(((o['xy']-data['gt'][ix]).norm(dim=-1)*m).sum());base+=float(((b['linear']-data['gt'][ix]).norm(dim=-1)*m).sum());count+=int(m.sum())
    return dict(hidden_px=total/max(count,1)*1408,linear_px=base/max(count,1)*1408,joints=count)

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--kind',choices=['dit','regression'],required=True);p.add_argument('--tracks-only',action='store_true');p.add_argument('--steps',type=int,default=4000);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610037);np.random.seed(202610037)
    arm=('tracks_' if a.tracks_only else 'rgb_')+a.kind;folder=s.RUN/arm;folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows,data=load(a.device);train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train'],device=a.device)
    dev_all=np.array([i for i,r in enumerate(rows) if r['role']=='development'])
    dev=torch.tensor(np.sort(np.random.default_rng(202610035).choice(dev_all,min(320,len(dev_all)),replace=False)),device=a.device)
    model=SpatialTemporalCompleter(a.kind,not a.tracks_only).to(a.device)
    init=torch.load(s.OLD/f'sealed/tracks_{a.kind}.pt',map_location=a.device,weights_only=False)
    # Identical learned trajectory starting point; obsolete 4x4 visual modules
    # are not loaded into the new visual branch.
    state={k:v for k,v in init['model'].items() if not k.startswith(('visual','local_visual','grid'))}
    mismatch=model.load_state_dict(state,strict=False);assert not mismatch.unexpected_keys
    probe=torch.load(s.RUN/'probe_rgb/best.pt',map_location=a.device,weights_only=False)
    decoder={k:v for k,v in probe['model'].items() if k.startswith(('heatmap.','offset.','prior'))}
    mismatch=model.visual_head.load_state_dict(decoder,strict=False);assert not mismatch.missing_keys and not mismatch.unexpected_keys
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.04)
    s.save(folder/'config.json',dict(kind=a.kind,use_rgb=not a.tracks_only,steps=a.steps,batch=48,
        train_windows=len(train),development_indices=dev.tolist(),trajectory_window_dropout=.2,joint_dropout=.15,
        auxiliary_rgb_heatmap_loss=.15,visual_gain_constraint=None,all_fusion_and_decoder_parameters_trainable=True,
        encoder_and_visual_stem='Frozen after independent RGB localization passes; all 192 spatial cells retained',
        selection='P0003 hidden-point error, same mask/gap protocol as v2',test_used=False))
    history=[];best=float('inf');started=time.time()
    for step in range(1,a.steps+1):
        model.train();ix=train[torch.randint(len(train),(48,),device=a.device)]
        lengths=torch.tensor([1,3,6,9],device=a.device)[torch.randint(4,(len(ix),),device=a.device)]
        variants=torch.randint(1,6,(len(ix),),device=a.device)
        b,_=batch(data,ix,lengths,variants,drop_tracks=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ix],data['valid'][ix])
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=1e-4*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            report=validate(model,data,dev);history.append(dict(step=step,**report));s.save(folder/'history.json',history);print(json.dumps(history[-1]),flush=True)
            if report['hidden_px']<best:
                best=report['hidden_px'];torch.save(dict(model=model.state_dict(),step=step,selection=report,kind=a.kind,use_rgb=not a.tracks_only),folder/'best.pt')
    s.save(folder/'done.json',dict(complete=True,best_development_px=best,test_evaluated=False))

if __name__=='__main__':main()
