"""Matched native regression/DiT/protected DiT pilots with complete resume state."""
import argparse,collections,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from native_dit import NativeMANODiT
from native_mano import NativeMANO,common
from native_data import Data,ROOT

def errors(x,gt):return ((x-x[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)

def predict(model,codec,b,seed=190019,zero_rgb=False):
    samples=model.sample(b,draws=4,steps=10,seed=seed,zero_rgb=zero_rgb)
    # A Lie-residual mean followed by exp produces valid MANO rotations.
    # This is GT-free posterior averaging, not GT best-of-K selection.
    delta=samples.mean(0)
    result=codec.apply_residual(b['state'],b['betas'],b['right'],delta)
    return result,delta,samples

def evaluate(model,codec,data,ids):
    values=[];model.eval()
    with torch.no_grad():
        for begin in range(0,len(ids),8):
            subset=ids[begin:begin+8];b=data.batch(subset);t=data.target(subset)
            with torch.autocast('cuda',dtype=torch.bfloat16):result,_,_=predict(model,codec,b)
            xyz=result['joints'][:,common.MAPPING];base=codec.from_state(b['state'],b['betas'],b['right'])[:,common.MAPPING]
            e=errors(xyz,t['gt']);e0=errors(base,t['gt']);valid=t['valid'];good=valid&(e0<=.01);good[:,5]=False
            values.append(dict(error=e[valid].cpu(),harm=((e-e0>.001)&good).cpu(),good=good.cpu()))
    e=torch.cat([v['error'] for v in values]);good=sum(int(v['good'].sum()) for v in values);harm=sum(int(v['harm'].sum()) for v in values)
    return dict(relative_mm=float(e.mean()*1000),harm_over1mm=harm,good10=good,harm_fraction=harm/max(1,good),score=float(e.mean()*1000)+20*harm/max(1,good))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['regression','dit','protected'],required=True);ap.add_argument('--steps',type=int,default=600);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(190019);np.random.seed(190019);random.seed(190019);torch.cuda.manual_seed_all(190019)
    output=ROOT/'runs'/a.arm;output.mkdir(parents=True,exist_ok=True)
    assert a.resume or not (output/'last.pt').exists(),'Explicit resume required'
    if a.resume and (output/'done.json').exists() and json.loads((output/'done.json').read_text())['steps']>=a.steps:
        print(json.dumps(dict(complete=True,already_complete=True,arm=a.arm,checkpoint=str(output/'last.pt'))));return
    data=Data();codec=NativeMANO();model=NativeMANODiT('regression' if a.arm=='regression' else 'dit').cuda()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    generator=torch.Generator(device='cuda').manual_seed(910019)
    groups=collections.defaultdict(list)
    for i in data.train:
        r=data.rows[i];groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    choices=[]
    for key,ids in sorted(groups.items()):
        ids.sort(key=lambda i:data.rows[i]['timestamp_ns'])
        for pos in range(len(ids)):choices.append(ids[max(0,pos-1):pos+2])
    rng=np.random.default_rng(910019);plan=[]
    for step in range(a.steps):
        batch=[]
        for chosen in rng.choice(len(choices),3,replace=False):batch.extend(choices[chosen])
        plan.append(batch)
    plan_sha=hashlib.sha256(json.dumps(plan).encode()).hexdigest();history=[];start_step=0;best=math.inf
    if a.resume:
        ck=torch.load(output/'last.pt',weights_only=False,map_location='cuda')
        old_budget=ck.get('planned_steps',ck['step'])
        old_plan_sha=hashlib.sha256(json.dumps(plan[:old_budget]).encode()).hexdigest()
        assert ck['batch_plan_sha'] in [plan_sha,old_plan_sha],'Resume would change earlier batch choices'
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu());generator.set_state(ck['noise_rng'].cpu());random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        start_step=ck['step'];history=ck['history'];best=ck['best_score']
    preflight=json.loads((ROOT/'receipts/native_preflight.json').read_text());assert preflight['complete']
    frozen_hash={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in codec.state_dict().items()}
    max_gradient=collections.defaultdict(float);start=time.time()
    model.train()
    for step in range(start_step,a.steps):
        ids=plan[step];b=data.batch(ids);t=data.target(ids);optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=3e-4*(.1+.9*(1+math.cos(math.pi*step/a.steps))/2)
        with torch.autocast('cuda',dtype=torch.bfloat16):clean,vloss=model.training_prediction(b,t['delta'],generator)
        result=codec.apply_residual(b['state'],b['betas'],b['right'],clean.float());xyz=result['joints'][:,common.MAPPING]
        gt=t['gt'];mask=t['valid'];relative=(xyz-xyz[:,5:6])-(gt-gt[:,5:6])
        task=(F.smooth_l1_loss(relative/.02,torch.zeros_like(relative),reduction='none').sum(-1)*mask).sum()/mask.sum().clamp_min(1)
        with torch.no_grad():base=codec.from_state(b['state'],b['betas'],b['right'])[:,common.MAPPING];base_error=errors(base,gt);good=mask&(base_error<=.01);good[:,5]=False
        student_error=errors(xyz,gt);guard=((student_error-base_error-.001).clamp_min(0)/.005*good).sum()/good.sum().clamp_min(1)
        parameter=F.smooth_l1_loss(clean.float(),t['delta'])
        velocity=xyz.sum()*0;count=0
        # Match true-time motion over adjacent centers selected from a track.
        for j in range(len(ids)-1):
            ra,rb=data.rows[ids[j]],data.rows[ids[j+1]];dt=(rb['timestamp_ns']-ra['timestamp_ns'])/1e9
            if (ra['sequence'],ra['clip'],ra['track_id'])!=(rb['sequence'],rb['clip'],rb['track_id']) or not 0<dt<.2:continue
            p0=xyz[j]-xyz[j,5:6];p1=xyz[j+1]-xyz[j+1,5:6];g0=gt[j]-gt[j,5:6];g1=gt[j+1]-gt[j+1,5:6]
            R0,R1=b['rotation'][j],b['rotation'][j+1]
            vp=(p1@R1.T-p0@R0.T)/(60*dt);vg=(g1@R1.T-g0@R0.T)/(60*dt);valid=mask[j]&mask[j+1]
            velocity+=(F.smooth_l1_loss(vp/.01,vg/.01,reduction='none').sum(-1)*valid).sum()/valid.sum().clamp_min(1);count+=1
        velocity=velocity/max(1,count)
        loss=vloss+task+.2*parameter+.05*velocity+(guard if a.arm=='protected' else 0)
        assert torch.isfinite(loss);loss.backward()
        for key,prefix in [('rgb_stem','rgb_stem'),('spatial','spatial'),('temporal','temporal'),('dit_head','head')]:
            grads=[p.grad.abs().max().item() for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
            if grads:max_gradient[key]=max(max_gradient[key],max(grads))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%50==0:
            row=dict(step=step+1,loss=float(loss.detach()),v=float(vloss.detach()),FK=float(task.detach()),protection=float(guard.detach()),velocity=float(velocity.detach()),seconds=time.time()-start)
            history.append(row);(output/'progress.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
        if (step+1)%200==0 or step+1==a.steps:
            score=evaluate(model,codec,data,data.select);model.train();improved=score['score']<best;best=min(best,score['score'])
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,history=history,best_score=best,selection=score,
                rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),noise_rng=generator.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                batch_plan_sha=plan_sha,planned_steps=a.steps,config=dict(kind=model.kind,width=128,blocks=4,heads=4),arm=a.arm,GT_inference=False)
            temporary=output/'last.partial.pt';torch.save(ck,temporary);temporary.replace(output/'last.pt')
            if improved:torch.save(ck,output/'best_development.pt')
            (output/'selection.json').write_text(json.dumps(dict(step=step+1,**score),indent=2))
    assert all(v>0 for v in max_gradient.values()),max_gradient
    updates={prefix:max(float((v.cpu()-initial[k]).abs().max()) for k,v in model.state_dict().items() if k.startswith(prefix) and v.is_floating_point()) for prefix in ['rgb_stem','spatial','temporal','head']}
    assert all(v>0 for v in updates.values())
    assert frozen_hash=={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in codec.state_dict().items()}
    report=dict(complete=True,steps=a.steps,arm=a.arm,train_windows=len(data.train),dev_select_windows=len(data.select),audit_windows_unopened=len(data.audit),
        gradient_max=dict(max_gradient),weight_update_max=updates,FK_buffers_unchanged=True,batch_plan_sha=plan_sha,optimizer_and_RNG_saved=True,
        supervision='fitted MANO pseudo-parameters plus loss-only natural GT20 FK/true-time motion; correct-point protection loss only in protected arm',
        visual_backbone='full frozen WiLoR features; trainable spatial adapter/attention; no claim of backbone fine-tuning',seconds=time.time()-start,safe_default_admitted=False)
    (output/'done.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
