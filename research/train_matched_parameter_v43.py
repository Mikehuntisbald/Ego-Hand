"""Matched parameter regression/DiT: identical initialization and batch plan.

The model, observation-only fitted seeds, RGB features, targets, optimizer,
budget, and checkpoint selection are held fixed. Kind-specific objectives
and diffusion sampling are the experimental difference.
"""
import argparse,hashlib,json,time
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from parameter_codec_v31 import OUT,observation_batch
from semantic_parameter_model_v36 import SemanticParameterHand


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['regression','dit'],required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--recipe',choices=['rollout','denoise'],default='rollout')
    a=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(202610131)
    assert json.loads((OUT/'refined_target_representability.json').read_text())['passed']
    root=V7.parent;run=root/'matched_parameter_v43'/(a.kind if a.recipe=='rollout' else a.kind+'_denoise');run.mkdir(parents=True,exist_ok=True)
    if (run/'best.pt').exists():raise RuntimeError('Do not overwrite an existing matched training run')
    data=torch.load(root/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    data={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in data.items()}
    target={k:v.to(a.device) for k,v in torch.load(OUT/'refined_targets.pt',weights_only=False,mmap=True).items()}
    coarse=torch.load(root/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(a.device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(a.device)
    pr=torch.load(root/'side_data_v16/consensus/risk_dense/risk_probabilities.pt',weights_only=False)
    prob,devprob=pr['train_oof'].to(a.device),pr['joint'].to(a.device)
    bank=torch.load(root/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True).to(a.device)
    train=torch.tensor([i for i,r in enumerate(data['roles']) if r=='train'],device=a.device)
    dev=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'],device=a.device)
    plan=torch.randint(len(train),(a.steps,a.batch),generator=torch.Generator().manual_seed(202610151))
    torch.save(plan,run/'train_batch_plan.pt');plan_sha=hashlib.sha256(plan.numpy().tobytes()).hexdigest()
    if a.recipe=='denoise':
        assert a.kind=='dit'
        from denoise_parameter_model_v43 import DenoiseParameterHand
        cls=DenoiseParameterHand
    else:cls=SemanticParameterHand
    model=cls(a.kind,a.device).to(a.device)
    initial=torch.load(root/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=a.device)
    transferred=model.load_visual_initial(initial['model'])
    config=dict(kind=a.kind,recipe=a.recipe,steps=a.steps,batch=a.batch,seed=202610131,batch_plan_sha256=plan_sha,
        train_windows=len(train),dev_select_windows=len(dev),query_semantics='34 scalar parameter queries associated with actual anatomical observations',
        coarse_seed='GT-free fitted WiLoR observationIK v38, preserving fitted root and shape',
        visual='Frozen pretrained32layer backbone, native192x1280 spatial features; same spatial localization/projection and temporal4layer192/6heads',
        initial_v16_step=initial['step'],transferred_tensors=transferred,optimizer='AdamW same .0002 schedule and .01 decay',
        targets='Same refined train-IK hand parameters and GTXYZ; no GT or participant calibration in inference inputs',
        selection='Same original v39 raw-generator dev_select score every250steps; eventual v42 output comparison is separate',
        evaluation_scope='Previously reused dev_select. Preexisting data/target files include calibration-role rows; no calibration-role training or metrics in this run',
        kind_specific='Regression supervised parameter residual; DiT additionally diffusion velocity target, DDIM10steps/4draws',
        default_changed=False,natural_rgb_only=True)
    save(run/'config.json',config)
    def get(ids,p):
        b=observation_batch(data,ids,p,coarse,right,preserve_fitted=True);b['rgb_native']=bank[data['feature_ids'][ids]];return b
    @torch.inference_mode()
    def evaluate():
        model.eval();pred=[];states=[];sides=[]
        for begin in range(0,len(dev),8):
            ix=dev[begin:begin+8];b=get(ix,devprob)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_parameters(b,seed=202610131+begin)
            pred.append(p['xyz_camera_m']);states.append(p['state']);sides.append(p['right'])
        xyz=torch.cat(pred);m=metrics(xyz,data['original_base_for_evaluation'][dev],data['gt'][dev],data['valid'][dev]);rank,_=score(m)
        return rank,m,dict(prediction=xyz.cpu(),state=torch.cat(states).cpu(),right=torch.cat(sides).cpu(),indices=dev.cpu())
    start=time.time();optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
    ix=train[:a.batch];b=get(ix,prob);tt={k:v[ix] for k,v in target.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(b,tt,202610131)
    loss.backward();grad={k:float(p.grad.norm()) for k,p in model.named_parameters() if k in ['semantic_head.weight','rgb_project.weight','side_head.1.weight'] and p.grad is not None}
    assert all(np.isfinite(v) for v in grad.values()) and grad['semantic_head.weight']>0
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        clean=model.predict_parameters(b);poison=dict(b,gt=torch.full_like(tt['gt'],float('nan')),gt_shape=torch.zeros(len(ix),10,device=a.device),gt_right=1-tt['right'])
        dirty=model.predict_parameters(poison)
        assert torch.equal(clean['state'],dirty['state']) and torch.equal(clean['xyz_camera_m'],dirty['xyz_camera_m'])
    save(run/'preflight.json',dict(passed=True,gt_poison_exact=True,output_shape=list(clean['xyz_camera_m'].shape),gradient_norms=grad,loss=float(loss.detach()),seconds=time.time()-start))
    print(json.dumps(dict(kind=a.kind,stage='preflight',seconds=time.time()-start,train=len(train),select=len(dev))),flush=True)
    best=None;history=[]
    for step in range(1,a.steps+1):
        model.train();ix=train[plan[step-1].to(a.device)];b=get(ix,prob);tt={k:v[ix] for k,v in target.items()}
        lr=.0002*min(step/100,1.)*(.2+.8*.5*(1+np.cos(np.pi*step/a.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(b,tt,202610131+step)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==0:print(json.dumps(dict(kind=a.kind,step=step,loss=float(loss.detach()),parts=parts,seconds=time.time()-start)),flush=True)
        if step%250==0:
            rank,m,p=evaluate();entry=dict(step=step,loss=float(loss.detach()),metrics=m,rank=list(rank),seconds=time.time()-start)
            history.append(entry);save(run/'history.json',history)
            if best is None or rank<best:
                best=rank;torch.save(dict(model=model.state_dict(),step=step,kind=a.kind,config=config),run/'best.pt');torch.save(p,run/'development_predictions.pt')
            if step%1000==0:torch.save(dict(model=model.state_dict(),step=step,kind=a.kind,config=config),run/f'step{step}.pt')
            print(json.dumps(dict(kind=a.kind,stage='evaluation',**entry)),flush=True)
    ck=torch.load(run/'best.pt',weights_only=False,map_location='cpu')
    save(run/'done.json',dict(complete=True,steps=a.steps,selected_step=ck['step'],seconds=time.time()-start,batch_plan_sha256=plan_sha,default_changed=False))
    print((run/'done.json').read_text(),flush=True)


if __name__=='__main__':main()
