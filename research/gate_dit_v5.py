"""Train continuous local gates from real samples, then audit full development."""
import argparse
import json
import time
from pathlib import Path
import wilor_eval_common
import numpy as np
import torch
from torch.nn import functional as F
from dit_v5_model import VisualResidual
from dit_v3_sampling import propose as sample_proposal
from train_dit_v5 import RUN,load_data,batch,save
from metrics_3d import EVAL_INDICES
from evaluate_proof import ray_masks,group_report


@torch.inference_mode()
def collect(model,data,ids,seed,policy='independent',samples=2,steps=10):
    store={};features=[];deltas=[];generator=torch.Generator(device=ids.device).manual_seed(seed)
    handle=model.gate.register_forward_pre_hook(lambda module,inputs:store.update(inputs=inputs[0].detach()))
    for start in range(0,len(ids),64):
        b=batch(data,ids[start:start+64])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            p=sample_proposal(model,b,policy=policy,steps=steps,samples=samples,generator=generator)
            model.gate_values(p,b)
        features.append(store['inputs'].half());deltas.append(p.float()*.05)
    handle.remove()
    return dict(inputs=torch.cat(features),delta=torch.cat(deltas),transform=data['transform'][ids])


def masks(rows,data,ids):
    chosen=[rows[i] for i in ids.cpu().tolist()]
    gt=data['gt'][ids].cpu().numpy();valid=np.array([r['projection_valid'] for r in chosen],bool)
    vis=np.array([r['visibility_label'] for r in chosen]);ray,counts=ray_masks(gt,valid)
    canonical=torch.zeros(len(ids),20,device=ids.device,dtype=torch.bool);canonical[:,EVAL_INDICES]=True
    # Original frozen threshold, not a new quantile selected to flatter this model.
    low=data['confidence'][ids,1:]<.69746333360672
    return dict(all=canonical,low_pose_confidence=canonical&low,
                high_occlusion_in_view=canonical&torch.tensor(((vis<.5)&(valid.sum(1)>=18))[:,None],device=ids.device),
                multiple_ray_aligned_fingers=canonical&torch.tensor(ray&((counts>=2)&(valid.sum(1)>=18))[:,None],device=ids.device)),np.array([r['sequence'] for r in chosen])


def error_arrays(p,gt):
    difference=p-gt;relative=(p-p[:,5:6])-(gt-gt[:,5:6])
    ray=gt/gt.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    return dict(camera=difference.norm(dim=-1)*1000,relative=relative.norm(dim=-1)*1000,
                axial_relative=(relative*ray).sum(-1).abs()*1000)


@torch.inference_mode()
def calibration(model,bank,coarse,gt,groups):
    with torch.autocast('cuda',dtype=torch.bfloat16):raw=model.activate_gate(bank['inputs'].float())
    before=error_arrays(coarse,gt);canonical=groups['all']
    precise=(before['camera']<=10)&canonical;precise_rel=(before['relative']<=10)&canonical
    baseline=float(before['camera'][canonical].mean())
    best=None;best_pred=None;grid=[]
    for threshold in [0,.1,.25,.5,.75]:
        selected=raw*(raw>=threshold)
        for strength in [0,.025,.05,.1,.2,.35,.5,.75,1.]:
            pred=coarse+(bank['delta']*(selected*strength))@bank['transform'].transpose(-1,-2)
            after=error_arrays(pred,gt)
            harm=float(((after['camera']>before['camera']+1)&precise).sum()/precise.sum().clamp_min(1))
            harm_rel=float(((after['relative']>before['relative']+1)&precise_rel).sum()/precise_rel.sum().clamp_min(1))
            camera=float(after['camera'][canonical].mean());relative=float(after['relative'][canonical].mean())
            group_errors={name:float(after['axial_relative' if name=='multiple_ray_aligned_fingers' else 'relative'][mask].mean()) for name,mask in groups.items() if name!='all' and mask.any()}
            admissible=harm<=.05 and harm_rel<=.05 and camera<=baseline+.1
            score=camera+relative+.5*sum(group_errors.values())
            row=dict(threshold=threshold,strength=strength,camera_mm=camera,relative_mm=relative,harm_camera=harm,harm_relative=harm_rel,
                     group_errors=group_errors,admissible=admissible,score=score)
            grid.append(row)
            if admissible and (best is None or score<best['score']):best=row;best_pred=pred.clone()
    assert best is not None
    return best,best_pred,grid


def gate_loss(model,features,delta,transform,c,gt,hard_ray=None):
    with torch.autocast('cuda',dtype=torch.bfloat16):g=model.activate_gate(features.float())
    desired=(gt-c)@transform;optimal=(desired*delta/delta.square().clamp_min(1e-8)).clamp(0,1)
    p=c+(g*delta)@transform.transpose(-1,-2)
    before=(c-gt).norm(dim=-1);after=(p-gt).norm(dim=-1)
    rb=((c-c[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
    ra=((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
    pc=(before<=.01).float();pr=(rb<=.01).float();pr[:,5]=0
    protect=((after-before-.0005).clamp_min(0)*pc).sum()/pc.sum().clamp_min(1)/.003
    protect_rel=((ra-rb-.0005).clamp_min(0)*pr).sum()/pr.sum().clamp_min(1)/.003
    loss=after.mean()/.03+.7*ra.mean()/.03+.2*F.mse_loss(g,optimal)+3*protect+3*protect_rel
    if hard_ray is not None:
        ray=gt/gt.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        rel=(p-p[:,5:6])-(gt-gt[:,5:6])
        axial=(rel*ray).sum(-1).abs()
        loss=loss+2*(axial*hard_ray).sum()/hard_ray.sum().clamp_min(1)/.03
    return loss


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['dit','regression'],required=True);ap.add_argument('--device',default='cuda:3');ap.add_argument('--name',default='initial');ap.add_argument('--steps',type=int,default=4000)
    ap.add_argument('--proposal-source');ap.add_argument('--sampling',choices=['independent','antithetic','zero'],default='independent');ap.add_argument('--samples',type=int,default=2);ap.add_argument('--denoise-steps',type=int,default=10);ap.add_argument('--hard-focus',action='store_true');ap.add_argument('--coherent',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610032)
    folder=RUN/f'{a.name}_{a.kind}'
    source=RUN/f'{a.proposal_source}_{a.kind}' if a.proposal_source else folder
    while not (source/'proposal_done.json').exists():time.sleep(15)
    folder.mkdir(exist_ok=True)
    if (folder/'gate_done.json').exists():print('Already complete',flush=True);return
    rows,data=load_data(a.device)
    train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='gate_fit'],device=a.device)
    dev=torch.tensor([i for i,r in enumerate(rows) if r['role']=='development'],device=a.device)
    hard_ray=None
    if a.hard_focus:
        valid=np.array([r['projection_valid'] for r in rows],bool)
        ray,count=ray_masks(data['gt'].cpu().numpy(),valid)
        hard_ray=torch.tensor(ray&((count>=2)&(valid.sum(1)>=18))[:,None],device=a.device)
        ray_positions=torch.flatnonzero(hard_ray[train].any(-1)) if hasattr(torch,'flatnonzero') else torch.nonzero(hard_ray[train].any(-1)).flatten()
        assert len(ray_positions)>0
    model=VisualResidual(a.kind).to(a.device).eval()
    model.coherent_gate=a.coherent
    checkpoint=torch.load(source/'proposal_best.pt',map_location=a.device,weights_only=False);model.load_state_dict(checkpoint['model'])
    sampling=dict(policy=a.sampling,samples=a.samples,steps=a.denoise_steps)
    save(folder/'gate_config.json',dict(proposal_source=str(source),sampling=sampling,steps=a.steps,hard_focus=a.hard_focus,directional_gate=True))
    for p in model.parameters():p.requires_grad_(False)
    cache=folder/'gate_banks.pt'
    if cache.exists():
        banks=torch.load(cache,map_location=a.device,weights_only=False)
        assert banks['proposal_step']==checkpoint['step']
        assert banks.get('sampling',dict(policy='independent',samples=2,steps=10))==sampling
    else:
        print('Generating actual inference proposals for independent gate-fit sequences',flush=True)
        banks=dict(train=[collect(model,data,train,seed,**sampling) for seed in [915001,915002]],dev=collect(model,data,dev,915003,**sampling),proposal_step=checkpoint['step'],sampling=sampling)
        torch.save(banks,cache)
    for p in model.gate.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(model.gate.parameters(),lr=.0005,weight_decay=.01)
    groups,clusters=masks(rows,data,dev);best=float('inf');history=[];started=time.time()
    for step in range(1,a.steps+1):
        pos=torch.randint(len(train),(256,),device=a.device)
        if a.hard_focus:pos[:77]=ray_positions[torch.randint(len(ray_positions),(77,),device=a.device)]
        ix=train[pos];bank=banks['train'][step%2]
        opt.zero_grad(set_to_none=True)
        loss=gate_loss(model,bank['inputs'][pos],bank['delta'][pos],bank['transform'][pos],data['coarse'][ix],data['gt'][ix],hard_ray[ix] if hard_ray is not None else None)
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.gate.parameters(),1.);opt.step()
        if step%500==0 or step==a.steps:
            chosen,pred,grid=calibration(model,banks['dev'],data['coarse'][dev],data['gt'][dev],groups)
            row=dict(step=step,loss=float(loss.detach()),operating=chosen,seconds=time.time()-started);history.append(row)
            if chosen['score']<best:
                best=chosen['score']
                torch.save(dict(model=model.state_dict(),kind=a.kind,proposal_step=checkpoint['step'],gate_step=step,operating=chosen,sampling=sampling,coherent_gate=a.coherent),folder/'best.pt')
                save(folder/'calibration_grid.json',grid)
            save(folder/'gate_history.json',history);save(folder/'status.json',dict(stage='gate_training',step=step,steps=a.steps,complete=False,acceptance_passed=False))
            print(json.dumps(row),flush=True)
    selected=torch.load(folder/'best.pt',map_location=a.device,weights_only=False);model.load_state_dict(selected['model'])
    chosen,pred,grid=calibration(model,banks['dev'],data['coarse'][dev],data['gt'][dev],groups)
    c=data['coarse'][dev].cpu().numpy();g=data['gt'][dev].cpu().numpy();p=pred.cpu().numpy()
    reports={name:group_report(p,g,c,mask.cpu().numpy(),clusters) for name,mask in groups.items()}
    for report in reports.values():
        for value in report.values():
            if isinstance(value,dict) and 'unit' in value:value['unit']='source-sequence bootstrap; three development subjects'
    success={}
    for name in ['low_pose_confidence','high_occlusion_in_view','multiple_ray_aligned_fingers']:
        r=reports[name];v=r.get('axial_relative' if name=='multiple_ray_aligned_fingers' else 'relative',{})
        success[name]=bool(r['joints']>=100 and v.get('improvement_pct',0)>=5 and v.get('ci95') is not None and v['ci95'][1]<0)
    accepted=all(success.values()) and chosen['admissible']
    evidence=dict(development_only=True,samples=len(dev),operating=chosen,groups=reports,hard_group_passed=success,development_acceptance_passed=accepted,
                  locked_test_evaluated=False,checkpoint=str(folder/'best.pt'),gate_step=selected['gate_step'],proposal_step=selected['proposal_step'],sampling=sampling)
    save(folder/'development_results.json',evidence)
    save(folder/'gate_done.json',dict(complete=True,development_acceptance_passed=accepted,final_acceptance_passed=False))
    save(folder/'status.json',dict(stage='development_evaluated',complete=False,acceptance_passed=False,development_acceptance_passed=accepted))
    print(json.dumps(evidence,indent=2),flush=True)


if __name__=='__main__':main()

