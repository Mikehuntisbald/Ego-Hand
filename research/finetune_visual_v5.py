"""Controlled natural-RGB trial: train final four WiLoR blocks with temporal head.
No GT/visibility enters image preparation or conditioning. Frozen prefix is cached.
"""
import argparse,json,time,copy,hashlib,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch,cv2
from torch import nn
import spatial_rgb_common as s
from natural_reliability import RUN as V4
from natural_corrector import NaturalCorrector,natural_batch
from spatial_rgb_model import SpatialHead
from train_spatial_temporal import load
from evaluate_natural_reliability import measure
from natural_policy import apply_policy
from evaluate_offline_kp import bootstrap
RUN=s.common.ROOT/'experiments/natural_visual_ft_v5'
RUN.mkdir(exist_ok=True)

def load_visual(device):
    full,_=s.common.load_model(device);backbone=full.backbone;del full
    backbone.eval();assert len(backbone.blocks)==32 and not backbone.skip_blocks
    stem=SpatialHead().to(device).eval()
    stem.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'])
    return backbone,stem

def cache(a):
    records,index=s.records_and_index();folder=RUN/'prefix';folder.mkdir(exist_ok=True)
    backbone,stem=load_visual(a.device);captured={}
    hook=backbone.blocks[28].register_forward_pre_hook(lambda module,args:captured.update(x=args[0].detach()))
    def prep(i):
        r=records[i];whitelist={k:r[k] for k in ['image','camera','clip']}
        return s.prepare(whitelist,index['roi'][i+1].numpy()*1408,np.zeros((1,4),np.float32))[0][0]
    started=time.time();checks=[]
    with ThreadPoolExecutor(max_workers=6) as pool,torch.inference_mode():
        for start in range(a.shard*16,len(records),a.shards*16):
            end=min(start+16,len(records));dest=folder/f'{start:06d}.pt'
            if dest.exists():continue
            ims=list(pool.map(prep,range(start,end)));n=len(ims);ims+=ims[-1:]*(16-n)
            x=s.common.input_tensor(ims,[1]*16,rotation=1).to(a.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                native=backbone(x[:,:,:,32:-32])[-1]
                prefix=captured['x'].clone();z=prefix
                for blk in backbone.blocks[28:]:z=blk(z)
                z=backbone.last_norm(z)[:,-192:]
                err=float((z-native.flatten(2).transpose(1,2)).abs().max())
            assert err==0.,err
            torch.save(dict(start=start,end=end,tokens=prefix[:n].cpu()),dest)
            checks.append(err)
            if len(checks)%10==0:print(json.dumps(dict(stage='cache',shard=a.shard,end=end,seconds=time.time()-started)),flush=True)
    hook.remove();s.save(RUN/f'cache_{a.shard}.json',dict(complete=True,max_split_forward_error=max(checks,default=0),chunks=len(checks),seconds=time.time()-started))

class Tail(nn.Module):
    def __init__(self,backbone,stem):
        super().__init__();self.blocks=nn.ModuleList(list(backbone.blocks[28:]));self.norm=backbone.last_norm
        self.project=stem.project;self.spatial=stem.spatial
    def forward(self,x):
        for block in self.blocks:x=block(x)
        x=self.norm(x)[:,-192:];x=self.project(x.float()).transpose(1,2).reshape(-1,128,16,12)
        return (x+self.spatial(x)).flatten(2).transpose(1,2)

def prefix_bank(device,n):
    parts=sorted((RUN/'prefix').glob('*.pt'));assert parts
    first=torch.load(parts[0],weights_only=False);bank=torch.zeros(n,*first['tokens'].shape[1:],dtype=first['tokens'].dtype,device=device);last=0
    for path in parts:
        c=torch.load(path,weights_only=False);assert c['start']==last;last=c['end'];bank[c['start']+1:c['end']+1]=c['tokens'].to(device)
    assert last==n-1
    return bank

def visual_batch(tail,prefix,data,ix,prob):
    b=natural_batch(data,ix,prob)
    if tail is not None:
        f=data['feature_ids'][ix];unique,inverse=torch.unique(f,return_inverse=True);keep=unique>0
        features=torch.zeros(len(unique),192,128,device=ix.device,dtype=torch.bfloat16)
        # Prefix features have no trainable layers upstream. Tail gradients remain live.
        features[keep]=tail(prefix[unique[keep]]).to(features.dtype)
        b['rgb']=features[inverse.reshape(-1)].reshape(len(ix),17,192,128)
    return b

@torch.no_grad()
def predict(model,tail,prefix,data,prob,ids):
    model.eval()
    if tail is not None:tail.eval()
    out=[]
    for start in range(0,len(ids),8):
        ix=ids[start:start+8]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            b=visual_batch(tail,prefix,data,ix,prob);out.append(model.predict(b)['xy'].float())
    return torch.cat(out)

def train(a):
    seed=202610053;torch.manual_seed(seed);np.random.seed(seed)
    folder=RUN/f'{a.kind}_{a.mode}';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows,data=load(a.device);roles=np.array(torch.load(V4/'features.pt',weights_only=False,mmap=True)['roles'])
    probabilities=torch.load(V4/'probabilities.pt',weights_only=False)
    prob=probabilities['train_oof'].to(a.device);evalprob=probabilities['joint'].to(a.device)
    ids=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device)
    model=NaturalCorrector(a.kind).to(a.device);model.load_state_dict(torch.load(V4/'sealed'/f'{a.kind}.pt',map_location=a.device,weights_only=False)['model'])
    tail=prefix=None;parity={}
    if a.mode=='ft':
        backbone,stem=load_visual(a.device);tail=Tail(backbone,stem).to(a.device).eval();del backbone,stem
        for p in tail.parameters():p.requires_grad_(True)
        prefix=prefix_bank(a.device,len(data['bank']))
        ix=dev[:8]
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            frozen=natural_batch(data,ix,evalprob);live=visual_batch(tail,prefix,data,ix,evalprob)
            delta=(frozen['rgb'].float()-live['rgb'].float()).abs()
            y0=model.predict(frozen)['xy'];y1=model.predict(live)['xy']
            parity=dict(feature_mean_abs=float(delta.mean()),feature_max_abs=float(delta.max()),prediction_mean_px=float((y0-y1).norm(dim=-1).mean()*1408),prediction_max_px=float((y0-y1).norm(dim=-1).max()*1408))
        s.save(folder/'initial_parity.json',parity)
        assert parity['prediction_mean_px']<.3 and parity['prediction_max_px']<2,parity
    # Both arms get same head LR, optimizer steps, examples and seed schedule.
    groups=[dict(params=model.parameters(),lr=2e-5,initial_lr=2e-5)]
    if tail is not None:
        groups.extend([dict(params=list(tail.blocks.parameters())+list(tail.norm.parameters()),lr=1e-6,initial_lr=1e-6),dict(params=list(tail.project.parameters())+list(tail.spatial.parameters()),lr=1e-5,initial_lr=1e-5)])
    opt=torch.optim.AdamW(groups,weight_decay=.04)
    config=dict(kind=a.kind,mode=a.mode,steps=a.steps,batch=8,accumulation=2,seed=seed,trainable_backbone_blocks=[28,29,30,31] if tail else [],prefix_frozen_blocks=list(range(28)),lr_head=2e-5,lr_backbone=1e-6,lr_stem=1e-5,selection='minimum dev_select raw mean finger error; step0 included',gate='fixed v4 regression policy for both heads; no test calibration',scope='Existing inspected test set is diagnostic, not fresh generalization; natural inputs only; tracking unchanged')
    s.save(folder/'config.json',config);history=[];started=time.time()
    trainable=list(model.parameters())+(list(tail.parameters()) if tail else [])
    def validate(step):
        pred=predict(model,tail,prefix,data,evalprob,dev);b=natural_batch(data,dev,evalprob);m=data['valid'][dev].clone();m[:,5]=False;m=m&data['observed'][dev,8]
        score=measure(pred,b['linear'],data['gt'][dev],m);history.append(dict(step=step,seconds=time.time()-started,**score));s.save(folder/'history.json',history)
        print(json.dumps(dict(stage='validate',arm=folder.name,**history[-1])),flush=True)
        return score['mean_px']
    best=validate(0)
    def save_best(step):
        torch.save(dict(model=model.state_dict(),tail=tail.state_dict() if tail else None,step=step,config=config),folder/'best.pt')
    save_best(0);gradient_checked=False
    for step in range(1,a.steps+1):
        model.train()
        # Keep pretrained backbone and stem dropout disabled, matching cached control.
        if tail is not None:tail.eval()
        opt.zero_grad(set_to_none=True);total=0.
        for micro in range(2):
            gen=torch.Generator(device=a.device).manual_seed(seed+step*2+micro)
            ix=ids[torch.randint(len(ids),(8,),device=a.device,generator=gen)]
            torch.manual_seed(seed+step*2+micro)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                b=visual_batch(tail,prefix,data,ix,prob);loss=model.loss(b,data['gt'][ix],data['valid'][ix])/2
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        if tail is not None and not gradient_checked:
            gradients={str(i+28):sum(float(p.grad.float().square().sum()) for p in block.parameters() if p.grad is not None)**.5 for i,block in enumerate(tail.blocks)}
            assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
            tracked=tail.blocks[-1].attn.qkv.weight;before=tracked.detach().clone();s.save(folder/'gradient_check.json',dict(block_gradient_norms=gradients))
        torch.nn.utils.clip_grad_norm_(trainable,1);opt.step()
        if tail is not None and not gradient_checked:
            delta=float((tracked-before).abs().max());assert delta>0
            s.save(folder/'weight_update_check.json',dict(last_block_max_weight_change=delta));gradient_checked=True
        for group in opt.param_groups:group['lr']=group['initial_lr']*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        if step%20==0:print(json.dumps(dict(stage='train',arm=folder.name,step=step,loss=total,seconds=time.time()-started)),flush=True)
        if step%100==0 or step==a.steps:
            score=validate(step)
            if score<best:best=score;save_best(step)
    s.save(folder/'done.json',dict(complete=True,best_dev_px=best,seconds=time.time()-started))
    # All selection ends here; evaluation follows the predeclared fixed policy.
    ck=torch.load(folder/'best.pt',map_location=a.device,weights_only=False);model.load_state_dict(ck['model'])
    if tail is not None:tail.load_state_dict(ck['tail'])
    test=torch.tensor(np.where(roles=='test')[0],device=a.device);pred=predict(model,tail,prefix,data,evalprob,test)
    b=natural_batch(data,test,evalprob);base=b['linear'];available=data['observed'][test,8];m=data['valid'][test].clone();m[:,5]=False;m=m&available
    policy=json.loads((V4/'sealed/policy.json').read_text())['policy'];assert policy['arm']=='regression' and policy['agreement_roi'] is None
    final,selected=apply_policy(base,{'regression':pred},evalprob[test],available,torch.zeros_like(available),b['roi'][:,8],policy)
    audit=json.loads((V4/'delivery/hardcase_review_queue.json').read_text());lookup={v:i for i,v in enumerate(test.tolist())};hard=torch.tensor([lookup[q['window_index']] for q in audit],device=a.device)
    report=dict(arm=folder.name,selected_step=ck['step'],raw=measure(pred,base,data['gt'][test],m),gated=measure(final,base,data['gt'][test],m,selected),hardcases=measure(final[hard],base[hard],data['gt'][test][hard],m[hard],selected[hard]),cases=[])
    for q in audit:
        j=lookup[q['window_index']];report['cases'].append(dict(id=q['id'],before_px=q['before_px'],v4_px=q['after_px'],v5_px=float(((final[j]-data['gt'][test[j]]).norm(dim=-1)*1408)[m[j]].mean())))
    s.save(folder/'results.json',report)
    np.savez_compressed(folder/'predictions.npz',window_indices=test.cpu().numpy(),raw=pred.cpu().numpy(),prediction=final.cpu().numpy(),base=base.cpu().numpy(),gt=data['gt'][test].cpu().numpy(),mask=m.cpu().numpy())
    s.save(folder/'evaluation_done.json',dict(complete=True,seconds=time.time()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cache','train']);p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);p.add_argument('--kind',default='regression');p.add_argument('--mode',choices=['ft','frozen'],default='ft');p.add_argument('--steps',type=int,default=600);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    cache(a) if a.stage=='cache' else train(a)
