"""Proposal training on frozen, real OOF observations; development selection only."""
import argparse
import hashlib
import json
import random
import os
import shutil
import time
from pathlib import Path
import wilor_eval_common  # Register the pinned SDK and existing experiment packages.
import torch
import numpy as np
from dit_v5_model import VisualResidual
from metrics_3d import EVAL_INDICES
from evaluate_proof import ray_masks

ROOT=Path('/mnt/why/HOT3D');RUN=Path(os.environ.get('HOT3D_DIT_RUN',str(ROOT/'experiments/dit_wilor_v5')))


def save(path,obj):
    p=path.with_suffix(path.suffix+'.partial');p.write_text(json.dumps(obj,indent=2));p.replace(path)


def load_data(device):
    while not (RUN/'condition_cache_done.json').exists():time.sleep(15)
    rows=json.loads((RUN/'rows.json').read_text())
    data=torch.load(RUN/'observations.pt',map_location='cpu',weights_only=False,mmap=True)
    pieces={}
    previous=0
    for path in sorted((RUN/'feature_chunks').glob('*.pt')):
        part=torch.load(path,map_location='cpu',weights_only=False)
        assert part.pop('start')==previous;previous=part.pop('end')
        for k,v in part.items():pieces.setdefault(k,[]).append(v)
    assert previous==len(rows)
    data.update({k:torch.cat(v) for k,v in pieces.items()});del pieces
    data={k:v.to(device) for k,v in data.items()}
    return rows,data


def batch(data,ids,augment=False):
    b={k:v[ids] for k,v in data.items() if k not in ['final_coarse','final_confidence']}
    if augment:
        mix=torch.rand(len(ids),1,1,device=ids.device)<.25
        b['coarse']=torch.where(mix,data['final_coarse'][ids],b['coarse'])
        b['confidence']=torch.where(mix[:,:,0],data['final_confidence'][ids],b['confidence'])
        b['coarse']=b['coarse']+torch.randn_like(b['coarse'])*.001
    return b


@torch.inference_mode()
def evaluate(model,data,ids):
    model.eval();pred=[];proposals=[];generator=torch.Generator(device=ids.device).manual_seed(914003)
    for start in range(0,len(ids),64):
        b=batch(data,ids[start:start+64])
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.rollout(b)
        proposals.append(p.float());pred.append(b['coarse']+model.native_delta(p,b))
    p=torch.cat(pred);gt=data['gt'][ids];c=data['coarse'][ids]
    camera=(p-gt).norm(dim=-1)[:,EVAL_INDICES]*1000
    relative=((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)[:,EVAL_INDICES]*1000
    before=(c-gt).norm(dim=-1)[:,EVAL_INDICES]*1000
    rb=((c-c[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)[:,EVAL_INDICES]*1000
    result=dict(camera_mm=float(camera.mean()),relative_mm=float(relative.mean()),
                baseline_camera_mm=float(before.mean()),baseline_relative_mm=float(rb.mean()),
                harm_camera=float(((camera>before+1)&(before<=10)).sum()/(before<=10).sum().clamp_min(1)),
                harm_relative=float(((relative>rb+1)&(rb<=10)).sum()/(rb<=10).sum().clamp_min(1)))
    if 'hard_ray' in data:
        ray=gt/gt.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        error=(p-p[:,5:6])-(gt-gt[:,5:6]);mask=data['hard_ray'][ids]
        result['ray_axial_mm']=float((error*ray).sum(-1).abs()[mask].mean()*1000)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['dit','regression'],required=True);ap.add_argument('--device',default='cuda:3');ap.add_argument('--steps',type=int,default=8000);ap.add_argument('--name',default='initial');ap.add_argument('--init-source');ap.add_argument('--hard-focus',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610031);np.random.seed(202610031);random.seed(202610031)
    folder=RUN/f'{a.name}_{a.kind}';folder.mkdir(exist_ok=True)
    if (folder/'proposal_done.json').exists():print('Already complete',flush=True);return
    rows,data=load_data(a.device)
    train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='denoise'],device=a.device)
    if a.hard_focus:
        valid=np.array([r['projection_valid'] for r in rows],bool)
        ray,count=ray_masks(data['gt'].cpu().numpy(),valid)
        ray=ray&((count>=2)&(valid.sum(1)>=18))[:,None]
        data['hard_ray']=torch.tensor(ray,device=a.device)
        ray_train=train[data['hard_ray'][train].any(-1)]
        occluded=train[torch.tensor([rows[i]['visibility_label']<.5 for i in train.tolist()],device=a.device)]
        assert len(ray_train)>0 and len(occluded)>0
    # Fixed balanced selection subset; all other previously-used evaluation rows remain development data.
    rng=np.random.default_rng(202610031);selected=[]
    for subject in ['P0003','P0010','P0015']:
        ix=[i for i,r in enumerate(rows) if r['role']=='development' and r['subject']==subject]
        selected.extend(rng.choice(ix,min(256,len(ix)),replace=False).tolist())
    hard=[i for i,r in enumerate(rows) if r['role']=='development' and r['visibility_label']<.5]
    selected=sorted(set(selected+hard))
    if a.hard_focus:selected=sorted(set(selected+[i for i,r in enumerate(rows) if r['role']=='development' and ray[i].any()]))
    dev=torch.tensor(selected,device=a.device)
    save(folder/'training_config.json',dict(kind=a.kind,steps=a.steps,width=256,depth=6,batch=128,lr=.00003,
         train_samples=len(train),selection_indices=selected,final_model_train_mix=.25,
         trained_subjects=sorted({rows[i]['subject'] for i in train.tolist()}),selection_metric='camera + wrist-relative error on balanced development subset',
         testing='New locked source sequences not loaded'))
    if a.hard_focus:save(folder/'hard_focus.json',dict(ray_train_samples=len(ray_train),occlusion_train_samples=len(occluded),sampling_fractions=dict(uniform=.5,ray=.3,occlusion=.2),extra_axial_supervision=True,init_source=a.init_source))
    model=VisualResidual(a.kind).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.00003,weight_decay=.01)
    start=1;best=float('inf');history=[]
    path=folder/'proposal_last.pt'
    if path.exists():
        ck=torch.load(path,map_location=a.device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step']+1;best=ck['best'];history=ck['history'];torch.set_rng_state(ck['cpu_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),a.device)
    else:
        if a.init_source:
            init=torch.load(RUN/f'{a.init_source}_{a.kind}'/'proposal_best.pt',map_location=a.device,weights_only=False)
            model.load_state_dict({k:v for k,v in init['model'].items() if not k.startswith('gate.')},strict=False)
        for source in ['dit_v5_model.py','train_dit_v5.py']:
            shutil.copy2(Path(__file__).parent/source,folder/source)
        b=batch(data,train[:8]);loss,parts=model.rollout_loss(b)
        assert torch.isfinite(loss);loss.backward();opt.zero_grad(set_to_none=True)
        zero=model.apply(torch.randn(8,20,3,device=a.device),b,torch.zeros(8,20,device=a.device))
        assert torch.equal(zero,b['coarse'])
        save(folder/'smoke.json',dict(real_data_forward_backward=True,zero_gate_exact_identity=True,loss=float(loss.detach())))
    started=time.time()
    for step in range(start,a.steps+1):
        model.train()
        if a.hard_focus:
            ix=torch.cat([train[torch.randint(len(train),(64,),device=a.device)],ray_train[torch.randint(len(ray_train),(38,),device=a.device)],occluded[torch.randint(len(occluded),(26,),device=a.device)]])
        else:ix=train[torch.randint(len(train),(128,),device=a.device)]
        b=batch(data,ix,augment=True)
        lr=.00003*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.rollout_loss(b)
        assert torch.isfinite(loss),parts
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:
            status=dict(stage='proposal_training',kind=a.kind,step=step,steps=a.steps,loss=float(loss.detach()),parts=parts,seconds=time.time()-started,complete=False,acceptance_passed=False)
            save(folder/'status.json',status);print(json.dumps(status),flush=True)
        if step%500==0 or step==a.steps:
            metrics=evaluate(model,data,dev);score=metrics['camera_mm']+metrics['relative_mm']+metrics.get('ray_axial_mm',0)
            row=dict(step=step,selection=metrics,seconds=time.time()-started);history.append(row)
            if score<best:
                best=score;torch.save(dict(model=model.state_dict(),step=step,kind=a.kind,selection=metrics),folder/'proposal_best.pt')
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,history=history,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(a.device))
            tmp=folder/'proposal_last.partial';torch.save(ck,tmp);tmp.replace(path)
            save(folder/'history.json',history);print(json.dumps(row),flush=True)
    save(folder/'proposal_done.json',dict(complete=True,best_score=best,checkpoint=str(folder/'proposal_best.pt'),acceptance_passed=False,requires='Gate training, full development acceptance, locked evaluation'))
    print('PROPOSAL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()

