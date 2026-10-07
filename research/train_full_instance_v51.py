"""Resumable full 3D core pilot, with real human instance-mask supervision."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,metrics,score,save
from parameter_codec_v31 import OUT,observation_batch
from online_parameter_model_v47 import OnlineVisual,RUN as SOURCE
from semantic_parameter_model_v36 import SemanticParameterHand
from instance_parameter_model_v51 import InstanceParameterHand
from cache_instance_conditions_v51 import RUN

SEED=2026100751
INITIAL=V7.parent/'online_rgb_iterative_v48/protected/best.pt'

def atomic(value,path):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=40);ap.add_argument('--device',default='cuda:0');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    folder=RUN/'core/dit_joint';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():print((folder/'done.json').read_text());return
    data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True);data={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in data.items()}
    target={k:v.to(a.device) for k,v in torch.load(SOURCE/'targets.pt',weights_only=False,mmap=True).items()}
    target['teacher_xyz']=torch.load(V7.parent/'online_rgb_iterative_v48/teacher_centers.pt',weights_only=False).to(a.device)
    masks=torch.load(RUN/'mask_conditions.pt',weights_only=False);indices=masks['indices']
    masks={k:v.to(a.device) for k,v in masks.items() if torch.is_tensor(v)}
    domain=torch.load(RUN/'domain_masks.pt',weights_only=False);dpixels=np.load(RUN/'domain_pixels.npy',mmap_mode='r')
    dtrain=[i for i,r in enumerate(domain['metadata']) if r['split']=='train'];ddev=[i for i,r in enumerate(domain['metadata']) if r['split']=='dev']
    assert dtrain and ddev
    coarse=torch.load(V7.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(a.device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(a.device)
    rp=torch.load(SOURCE/'risk.pt',weights_only=False);trainrisk,devrisk=rp['train_oof'].to(a.device),rp['joint'].to(a.device)
    pixels=np.load(SOURCE/'pixels.npy',mmap_mode='r');ck=torch.load(INITIAL,weights_only=False,map_location='cpu')
    model=InstanceParameterHand('dit',a.device).to(a.device);missing,unexpected=model.load_state_dict(ck['model'],strict=False)
    assert not unexpected and all(k.startswith(('instance_','ownership_head.')) for k in missing),(missing,unexpected)
    visual=OnlineVisual(a.device,True);visual.load_tail(ck['visual_tail'])
    original_buffers={k:v.detach().cpu().clone() for k,v in model.codec.state_dict().items()}
    plan=np.random.default_rng(SEED).choice(indices['train'],(a.steps,2),replace=True)
    domain_plan=np.random.default_rng(SEED+1).choice(dtrain,(a.steps,2),replace=True)
    config=dict(version=51,full_model=True,steps=a.steps,initial=str(INITIAL),initial_sha256=hashlib.file_digest(INITIAL.open('rb'),'sha256').hexdigest(),initial_development_admitted=True,
        train_windows=len(indices['train']),development_windows=len(indices['dev_select']),mask_train_instances=len(dtrain),mask_dev_instances=len(ddev),mask_labels='Human EgoHands polygons, loss only',
        RGB='Natural images; unchanged 192x1280 spatial tokens',context_s=1.6,decoder='bounded UmeTrack FK, 34 parameters, shape shared',
        trained_visual_blocks=[28,29,30,31],speed_unchanged=True,soft_hard_acceleration_x2=True,detector_tracking_solver_outside_backprop=True,
        synthetic_occlusion=False,GT_never_inference_input=True,default_changed=False,batch_plan_sha256=hashlib.sha256(plan.tobytes()+domain_plan.tobytes()).hexdigest(),
        diagnostic_pilot=True,manicure_coverage_verified=False,external_3D_GT=False)
    save(folder/'config.json',config)
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(('instance_','ownership_head.'))],lr=5e-6),
            dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('instance_','ownership_head.'))],lr=2e-4),
            dict(params=[p for p in visual.parameters() if p.requires_grad],lr=2e-7)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    def batch(ix,risk):
        ix=torch.as_tensor(ix,device=a.device);b=observation_batch(data,ix,risk,coarse,right,preserve_fitted=True);f=data['feature_ids'][ix]
        b=visual.replace(b,f,pixels);b.update(instance_own=masks['own'][f],instance_other=masks['other'][f],instance_quality=masks['quality'][f]);return b
    def dbatch(ids):
        ids=list(map(int,ids));native=visual.encode_pixels([dpixels[i] for i in ids],a.device)
        return dict(rgb_native=native[:,None],instance_own=domain['conditions'][ids,0].to(a.device)[:,None],instance_other=domain['conditions'][ids,1].to(a.device)[:,None],instance_quality=domain['quality'][ids].to(a.device)[:,None])
    @torch.no_grad()
    def predict_dev():
        model.eval();visual.configure();pieces=[]
        for start in range(0,len(indices['dev_select']),2):
            ix=indices['dev_select'][start:start+2]
            with torch.autocast('cuda',dtype=torch.bfloat16):pieces.append(model.predict_parameters(batch(ix,devrisk),SEED+start)['xyz_camera_m'])
        return torch.cat(pieces)
    @torch.no_grad()
    def mask_dev():
        intersection=union=0.;losses=[]
        for start in range(0,len(ddev),4):
            ids=ddev[start:start+4];bb=dbatch(ids);gt=domain['target'][ids].to(a.device)[:,None]
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model.ownership(bb)
            p=out['probability']>=.5;g=gt>=.5;intersection+=float((p&g).sum());union+=float((p|g).sum())
            losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],gt)))
        return dict(cell_IoU=intersection/max(union,1),BCE=float(np.mean(losses)),instances=len(ddev))
    devix=torch.tensor(indices['dev_select'],device=a.device);gt=target['gt'][devix,8];valid=target['valid'][devix,8]
    original=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)['original_base_for_evaluation'][devix.cpu()].to(a.device)
    start=time.time();history=[];first=0
    def checkpoint(step):return dict(model=model.state_dict(),visual_tail=visual.tail_state(),kind='dit',step=step,config=config)
    def save_resume(step):
        atomic(dict(checkpoint=checkpoint(step),optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),step=step,history=history,reference=reference.cpu(),baseline_mask=baseline_mask,elapsed=time.time()-start),folder/'resume.pt')
    if a.resume and (folder/'resume.pt').exists():
        rr=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);model.load_state_dict(rr['checkpoint']['model']);visual.load_tail(rr['checkpoint']['visual_tail']);optimizer.load_state_dict(rr['optimizer']);torch.set_rng_state(rr['cpu_rng']);torch.cuda.set_rng_state(rr['cuda_rng']);np.random.set_state(rr['numpy_rng']);first=rr['step'];history=rr['history'];reference=rr['reference'].to(a.device);baseline_mask=rr['baseline_mask'];start=time.time()-rr['elapsed']
        assert rr['checkpoint']['config']['batch_plan_sha256']==config['batch_plan_sha256']
    else:
        assert not (folder/'resume.pt').exists(),'Existing pilot needs --resume'
        reference=predict_dev();baseline_mask=mask_dev();atomic(checkpoint(0),folder/'initial.pt');atomic(checkpoint(0),folder/'best.pt')
        model.eval();visual.configure();ix=indices['train'][:2];bb=batch(ix,trainrisk)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            clean=model.predict_parameters(bb,SEED);dirty=model.predict_parameters(dict(bb,gt=torch.full((2,17,20,3),float('nan'),device=a.device),teacher_xyz=torch.full((2,20,3),float('nan'),device=a.device),instance_gt=torch.full((2,17,192),float('nan'),device=a.device)),SEED)
        assert torch.equal(clean['state'],dirty['state']) and torch.equal(clean['xyz_camera_m'],dirty['xyz_camera_m'])
        # Initial zero residual branches must retain the admitted predecessor.
        plain=SemanticParameterHand('dit',a.device).to(a.device).eval();plain.load_state_dict(ck['model'])
        with torch.autocast('cuda',dtype=torch.bfloat16):baseline=plain.predict_parameters(bb,SEED)
        delta=float((clean['xyz_camera_m']-baseline['xyz_camera_m']).abs().max());assert delta<1e-5,delta
        del plain,baseline,clean,dirty;torch.cuda.empty_cache()
        save(folder/'preflight.json',dict(passed=True,GT_poison_exact=True,zero_branch_max_abs_m=delta,full_visual_forward=True))
        history.append(dict(step=0,metrics=metrics(reference,original,gt,valid),mask=baseline_mask,admitted=True,reference_only=True));save(folder/'history.json',history);save_resume(0)
    reference_metrics=metrics(reference,original,gt,valid)
    for step in range(first+1,a.steps+1):
        model.train();visual.configure();optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(SEED+step);ix=plan[step-1].tolist();bb=batch(ix,trainrisk);tt={k:v[ix] for k,v in target.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(bb,tt,SEED+step)
        assert torch.isfinite(loss);loss.backward();del bb
        ids=domain_plan[step-1].tolist();bb=dbatch(ids);gg=domain['target'][ids].to(a.device)[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):mask_loss=model.ownership_loss(bb,gg,torch.ones_like(gg))
        assert torch.isfinite(mask_loss);(.2*mask_loss).backward();del bb
        if step==1:
            norms={str(i):sum(float(p.grad.float().square().sum()) for p in visual.backbone.blocks[i].parameters() if p.grad is not None)**.5 for i in range(28,32)}
            branch=float(model.instance_joint[-1].weight.grad.norm());own=float(model.ownership_head[-1].weight.grad.norm())
            assert all(np.isfinite(x) and x>0 for x in norms.values()) and branch>0 and own>0
            before=visual.backbone.blocks[31].attn.qkv.weight.detach().clone();save(folder/'gradients.json',dict(visual_blocks=norms,instance_branch=branch,ownership_branch=own,passed=True))
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],5);optimizer.step()
        if step==1:
            update=float((visual.backbone.blocks[31].attn.qkv.weight-before).abs().max());assert update>0
            save(folder/'first_update.json',dict(passed=True,visual_weight_update_max=update));del before
        if step%5==0:print(json.dumps(dict(stage='train_full_model',step=step,loss=float(loss.detach()),mask_loss=float(mask_loss.detach()),parts=parts,seconds=time.time()-start)),flush=True)
        if step%20==0 or step==a.steps:
            xyz=predict_dev();m=metrics(xyz,original,gt,valid);guard=metrics(xyz,reference,gt,valid);mk=mask_dev();_,oldfeasible=score(m)
            admitted=bool(oldfeasible and guard['camera_good_harm_rate']<=.01 and guard['relative_good_harm_rate']<=.01 and m['camera_mm']<=reference_metrics['camera_mm']+.1 and m['relative_mm']<=reference_metrics['relative_mm']+.1 and mk['cell_IoU']>=baseline_mask['cell_IoU'])
            entry=dict(step=step,metrics=m,correct_point_protection=guard,mask=mk,admitted=admitted,seconds=time.time()-start);history.append(entry);save(folder/'history.json',history);print(json.dumps(entry),flush=True)
            atomic(checkpoint(step),folder/'last.pt')
            if admitted:atomic(checkpoint(step),folder/'best.pt');atomic(dict(prediction=xyz.cpu(),indices=devix.cpu()),folder/'admitted_development_predictions.pt')
            save_resume(step)
    assert all(torch.equal(v.detach().cpu(),original_buffers[k]) for k,v in model.codec.state_dict().items())
    saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
    for key,value in model.state_dict().items():assert torch.equal(value.cpu(),saved['checkpoint']['model'][key])
    model.eval();visual.configure();bb=batch(indices['dev_select'][:2],devrisk)
    with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_parameters(bb,SEED)
    model.load_state_dict(saved['checkpoint']['model']);visual.load_tail(saved['checkpoint']['visual_tail']);optimizer.load_state_dict(saved['optimizer'])
    with torch.autocast('cuda',dtype=torch.bfloat16):q=model.predict_parameters(bb,SEED)
    assert torch.equal(p['state'],q['state'])
    assert torch.equal(torch.get_rng_state(),saved['cpu_rng']) and torch.equal(torch.cuda.get_rng_state(),saved['cuda_rng'])
    optstate=optimizer.state_dict()
    for key,value in optstate['state'].items():
        for name,tensor in value.items():
            if torch.is_tensor(tensor):assert torch.equal(tensor.cpu(),saved['optimizer']['state'][key][name])
    best=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
    save(folder/'verification.json',dict(passed=True,FK_buffers_unchanged=True,saved_state_exact=True,inference_reload_exact=True,optimizer_moments_exact=True,RNG_exact=True,GT_poison_exact=True))
    save(folder/'done.json',dict(complete=True,steps=a.steps,selected_step=best['step'],trained_development_admitted=best['step']>0,external_full_pipeline_validation_pending=True,seconds=time.time()-start,default_changed=False))

if __name__=='__main__':main()
