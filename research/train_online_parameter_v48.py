"""Resumable matched continuation; frozen teacher guards already recovered points."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from parameter_codec_v31 import OUT,observation_batch
from online_parameter_model_v47 import RUN as SOURCE,OnlineVisual,load_head
from protected_parameter_model_v48 import ProtectedParameterHand

RUN=V7.parent/'online_rgb_iterative_v48'
SEED=2026100517

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_torch(value,path):
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state())
def restore_rng(value):
    torch.set_rng_state(value['cpu']);torch.cuda.set_rng_state_all(value['cuda']);np.random.set_state(value['numpy'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['frozen','joint','protected'],required=True)
    ap.add_argument('--device',default='cuda:3');ap.add_argument('--steps',type=int,default=1800)
    ap.add_argument('--batch',type=int,default=2);ap.add_argument('--accumulation',type=int,default=2)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    assert json.loads((RUN/'teacher_ready.json').read_text())['complete']
    folder=RUN/args.arm;folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():print((folder/'done.json').read_text(),flush=True);return
    source_mode='frozen' if args.arm=='frozen' else 'joint';joint=args.arm!='frozen'
    path=SOURCE/f'dit_{source_mode}/last.pt';initial=torch.load(path,weights_only=False,map_location='cpu')
    assert initial['step']==600
    teacher_path=SOURCE/'dit_joint/best.pt';admitted=torch.load(teacher_path,weights_only=False,map_location='cpu')
    data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True)
    data={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in data.items()}
    target={k:v.to(args.device) for k,v in torch.load(SOURCE/'targets.pt',weights_only=False,mmap=True).items()}
    target['teacher_xyz']=torch.load(RUN/'teacher_centers.pt',weights_only=False).to(args.device)
    coarse=torch.load(V7.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(args.device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(args.device)
    pr=torch.load(SOURCE/'risk.pt',weights_only=False);prob,ep=pr['train_oof'].to(args.device),pr['joint'].to(args.device)
    pixels=np.load(SOURCE/'pixels.npy',mmap_mode='r')
    train=torch.tensor([i for i,r in enumerate(data['roles']) if r=='train'],device=args.device)
    dev=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'],device=args.device)
    model,_,_=load_head('dit',args.device)
    if args.arm=='protected':
        replacement=ProtectedParameterHand('dit',args.device,teacher_weight=4.).to(args.device)
        replacement.load_state_dict(model.state_dict());model=replacement
    visual=OnlineVisual(args.device,joint)
    model.load_state_dict(initial['model']);visual.load_tail(initial['visual_tail'])
    plan=torch.randint(len(train),(args.steps,args.accumulation,args.batch),generator=torch.Generator().manual_seed(SEED+4801))
    atomic_torch(plan,folder/'batch_plan.pt')
    config=dict(**vars(args),seed=SEED,cumulative_target_steps=600+args.steps,source_step=600,
        initial_checkpoint=str(path),initial_checkpoint_sha256=digest(path),
        teacher_checkpoint=str(teacher_path),teacher_checkpoint_sha256=digest(teacher_path),
        teacher_predictions_sha256=digest(RUN/'teacher_centers.pt'),batch_plan_sha256=hashlib.sha256(plan.numpy().tobytes()).hexdigest(),
        lr_head=5e-6,lr_backbone=2e-7 if joint else 0.,teacher_guard_weight=4. if args.arm=='protected' else 0.,
        optimizer_restart_from_v47=True,optimizer_state_saved_in_v48=True,
        online_RGB=True,spatial='192x1280 native tokens; full32 blocks; last4 and norm trainable in joint arms',
        context_s=1.6,trained_visual_blocks=[28,29,30,31] if joint else [],
        same_online_crop_and_BF16_padded16=True,train_windows=len(train),dev_select_windows=len(dev),natural_only=True,
        GT_only_supervised_loss_and_locked_dev_selection=True,teacher_not_an_inference_input=True,
        selection='Original admission plus teacher camera/relative good-point harm <=1 percent and teacher means noninferior0.1mm',
        best_pt='Admitted reference remains unless a trained model passes all gates; best_trained is diagnostic',
        speed_limits_unchanged=True,acceleration_soft_and_hard_x2=True,
        detector_tracking_wilor_200step_ik_fixed=True,decoder_geometry_fixed=True,stable_solver_outside_graph=True,
        default_changed=False)
    if (folder/'config.json').exists():
        old=json.loads((folder/'config.json').read_text())
        for key in ['steps','batch','accumulation','initial_checkpoint_sha256','teacher_predictions_sha256','batch_plan_sha256','teacher_guard_weight','lr_head','lr_backbone']:
            assert old[key]==config[key],(key,old[key],config[key])
        config=old
    else:save(folder/'config.json',config)
    original=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    original_base=original['original_base_for_evaluation'][dev.cpu()].to(args.device);del original
    def get(ids,risk):
        return visual.replace(observation_batch(data,ids,risk,coarse,right,preserve_fitted=True),data['feature_ids'][ids],pixels)
    def evaluate_predictions():
        model.eval();visual.configure();pieces=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for begin in range(0,len(dev),args.batch):
                ix=dev[begin:begin+args.batch];pieces.append(model.predict_parameters(get(ix,ep),seed=SEED+begin)['xyz_camera_m'])
        return torch.cat(pieces)
    # The admitted reference is evaluated with exactly the student batching/seed.
    teacher_dev_path=RUN/f'teacher_dev_batch{args.batch}.pt'
    if not teacher_dev_path.exists():
        model.load_state_dict(admitted['model']);visual.load_tail(admitted['visual_tail'])
        teacher_dev=evaluate_predictions();atomic_torch(dict(prediction=teacher_dev.cpu(),indices=dev.cpu()),teacher_dev_path)
        model.load_state_dict(initial['model']);visual.load_tail(initial['visual_tail'])
    else:teacher_dev=torch.load(teacher_dev_path,weights_only=False)['prediction'].to(args.device)
    assert len(teacher_dev)==len(dev)
    teacher_metrics=metrics(teacher_dev,original_base,target['gt'][dev,8],target['valid'][dev,8])
    def admission(xyz):
        m=metrics(xyz,original_base,target['gt'][dev,8],target['valid'][dev,8]);rank,old_feasible=score(m)
        protection=metrics(xyz,teacher_dev,target['gt'][dev,8],target['valid'][dev,8])
        feasible=bool(old_feasible and protection['camera_good_harm_rate']<=.01 and protection['relative_good_harm_rate']<=.01
            and m['camera_mm']<=teacher_metrics['camera_mm']+.1 and m['relative_mm']<=teacher_metrics['relative_mm']+.1)
        penalty=max(0,protection['camera_good_harm_rate']-.01)*200+max(0,protection['relative_good_harm_rate']-.01)*200
        penalty+=max(0,m['camera_mm']-teacher_metrics['camera_mm']-.1)*2+max(0,m['relative_mm']-teacher_metrics['relative_mm']-.1)*2
        return m,protection,(0 if feasible else 1,float(rank[1])+penalty),feasible
    def ckvalue(step):
        return dict(model=model.state_dict(),visual_tail=visual.tail_state() if joint else None,step=step,
            source_step=600,cumulative_step=600+step,kind='dit',config=config)
    groups=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=5e-6,initial_lr=5e-6)]
    if joint:groups.append(dict(params=[p for p in visual.parameters() if p.requires_grad],lr=2e-7,initial_lr=2e-7))
    opt=torch.optim.AdamW(groups,weight_decay=.01);parameters=[p for group in groups for p in group['params']]
    start=time.time();history=[];first_step=0;best_trained=None;best_admitted=None
    def validate(step):
        xyz=evaluate_predictions();m,p,rank,feasible=admission(xyz)
        entry=dict(step=step,cumulative_step=600+step,metrics=m,teacher_protection=p,rank=list(rank),feasible=feasible,seconds=time.time()-start)
        history.append(entry);save(folder/'history.json',history)
        print(json.dumps(dict(stage='validation',arm=args.arm,**entry)),flush=True)
        return rank,feasible,xyz
    def save_resume(step):
        atomic_torch(dict(checkpoint=ckvalue(step),optimizer=opt.state_dict(),rng=rng_state(),step=step,history=history,
            best_trained=best_trained,best_admitted=best_admitted,elapsed=time.time()-start),folder/'resume.pt')
        save(folder/'progress.json',dict(stage='training',step=step,cumulative_step=600+step,total_steps=args.steps,
            timestamp=time.time(),pid=os.getpid(),last_validation=history[-1] if history else None,default_changed=False))
    def verify_resume(step):
        keep_rng=rng_state();model.eval();visual.configure()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            bb=get(dev[:args.batch],ep);before=model.predict_parameters(bb,SEED+48991)
        disk=torch.load(folder/'resume.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(disk['checkpoint']['model']);visual.load_tail(disk['checkpoint']['visual_tail']);opt.load_state_dict(disk['optimizer'])
        for state_id,value in opt.state_dict()['state'].items():
            for key,tensor in value.items():
                if torch.is_tensor(tensor):assert torch.equal(tensor.cpu(),disk['optimizer']['state'][state_id][key])
        restore_rng(disk['rng'])
        assert torch.equal(torch.get_rng_state(),disk['rng']['cpu'])
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),disk['rng']['cuda']))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=model.predict_parameters(bb,SEED+48991)
        assert torch.equal(before['state'],after['state']) and torch.equal(before['xyz_camera_m'],after['xyz_camera_m'])
        restore_rng(keep_rng)
        save(folder/'resume_verification.json',dict(passed=True,step=step,model_prediction_exact=True,
            optimizer_moments_exact=True,cpu_cuda_RNG_exact=True,checkpoint_sha256=digest(folder/'resume.pt')))
        del bb,before,after,disk;torch.cuda.empty_cache()
    if args.resume and (folder/'resume.pt').exists():
        resumed=torch.load(folder/'resume.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(resumed['checkpoint']['model']);visual.load_tail(resumed['checkpoint']['visual_tail'])
        opt.load_state_dict(resumed['optimizer']);restore_rng(resumed['rng']);first_step=resumed['step']
        history=resumed['history'];best_trained=resumed['best_trained'];best_admitted=resumed['best_admitted'];start=time.time()-resumed['elapsed']
        print(json.dumps(dict(stage='resumed',arm=args.arm,step=first_step)),flush=True)
    else:
        assert not (folder/'resume.pt').exists(),'Use --resume for existing training'
        # Keep an actual admitted starting reference, never mislabel warm600 as safe.
        reference=dict(admitted);reference['config']=dict(config,reference_only=True,admitted_source_step=admitted['step'])
        reference['source_checkpoint']=str(teacher_path);reference['cumulative_step']=0
        atomic_torch(reference,folder/'best.pt')
        _,_,reference_rank,reference_feasible=admission(teacher_dev);assert reference_feasible
        best_admitted=reference_rank
        model.eval();visual.configure()
        tg=target['gt'][:,8];tv=target['valid'][:,8].clone();tv[:,5]=False;te=target['teacher_xyz']
        rel=lambda x:x-x[:,5:6]
        teacher_good=tv&(((te-tg).norm(dim=-1)<=.01)|((rel(te)-rel(tg)).norm(dim=-1)<=.01))
        ix=train[teacher_good[train].sum(-1).argsort(descending=True)[:args.batch]];b=get(ix,prob)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            clean=model.predict_parameters(b,seed=SEED+4800)
            dirty=model.predict_parameters(dict(b,gt=torch.full_like(target['gt'][ix],float('nan')),teacher_xyz=torch.full((len(ix),20,3),float('nan'),device=args.device),gt_right=1-target['right'][ix]),seed=SEED+4800)
            assert torch.equal(clean['state'],dirty['state']) and torch.equal(clean['xyz_camera_m'],dirty['xyz_camera_m'])
        # Verify same-state inference and differentiable teacher protection.
        plain,_,_=load_head('dit',args.device);plain.load_state_dict(model.state_dict());plain.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):plain_pred=plain.predict_parameters(b,seed=SEED+4800)
        assert torch.equal(clean['state'],plain_pred['state']);del plain
        tt={k:v[ix] for k,v in target.items()}
        model.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,parts=model.objective(b,tt,SEED+4800)
            if args.arm=='protected':
                generated=model.generate(b,SEED+4800)
                guard,_=model.teacher_penalty(generated['xyz'][:,8],tt)
        assert torch.isfinite(loss)
        if args.arm=='protected':
            gg=torch.autograd.grad(guard,model.semantic_head.weight,retain_graph=True)[0];guard_norm=float(gg.norm())
            assert np.isfinite(guard_norm)
            # An inactive hinge legitimately has zero gradient. Probe only the
            # FK output in this check; never perturb training images/targets.
            offset=torch.tensor([.03,-.02,.02],device=args.device)
            probe,_=model.teacher_penalty(generated['xyz'][:,8]+offset,tt)
            pg=torch.autograd.grad(probe,model.semantic_head.weight,retain_graph=True)[0];probe_norm=float(pg.norm())
            assert np.isfinite(probe_norm) and probe_norm>0;del gg,pg,probe,generated,guard
        else:guard_norm=None;probe_norm=None
        if joint:
            generated=model.generate(b,SEED+4800);vm=tt['valid'].float()
            fk=((generated['xyz']-tt['gt']).norm(dim=-1)*vm).sum()/vm.sum().clamp_min(1)
            gg=torch.autograd.grad(fk,visual.backbone.blocks[31].attn.qkv.weight,retain_graph=True)[0]
            fk_norm=float(gg.norm());assert np.isfinite(fk_norm) and fk_norm>0;del gg,generated,fk
        else:fk_norm=None
        loss.backward();norms={str(i):sum(float(p.grad.float().square().sum()) for p in visual.backbone.blocks[i].parameters() if p.grad is not None)**.5 for i in range(28,32)}
        assert all(np.isfinite(x) and x>0 for x in norms.values()) if joint else all(x==0 for x in norms.values())
        opt.zero_grad(set_to_none=True)
        save(folder/'preflight.json',dict(passed=True,GT_poison_exact=True,inference_architecture_state_parity_exact=True,
            teacher_guard_head_gradient=guard_norm,activated_teacher_guard_probe_gradient=probe_norm,
            guard_probe_only_FK_output_offset_m=[.03,-.02,.02] if args.arm=='protected' else None,
            pure_fk_visual_gradient=fk_norm,visual_block_gradient_norms=norms,
            teacher_not_input=True,teacher_guard_loss_only=True,loss=float(loss.detach()),parts=parts))
        del b,loss,clean,dirty,plain_pred;torch.cuda.empty_cache()
        rank,feasible,xyz=validate(0);best_trained=rank
        atomic_torch(ckvalue(0),folder/'best_trained.pt');atomic_torch(dict(indices=dev.cpu(),prediction=xyz.cpu()),folder/'development_predictions.pt')
        if feasible and rank<best_admitted:
            best_admitted=rank;atomic_torch(ckvalue(0),folder/'best.pt')
        save_resume(0)
    if args.preflight_only:return
    for step in range(first_step+1,args.steps+1):
        for group in opt.param_groups:group['lr']=group['initial_lr']*min(1.,step/20)*(.2+.8*.5*(1+np.cos(np.pi*step/args.steps)))
        model.train();visual.configure();opt.zero_grad(set_to_none=True);total=0.
        before=visual.backbone.blocks[31].attn.qkv.weight.detach().clone() if step==1 else None
        for micro in range(args.accumulation):
            torch.manual_seed(SEED+48000+step*args.accumulation+micro)
            ix=train[plan[step-1,micro].to(args.device)];b=get(ix,prob);tt={k:v[ix] for k,v in target.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(b,tt,SEED+48000+step*args.accumulation+micro);loss=loss/args.accumulation
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del b,loss
        torch.nn.utils.clip_grad_norm_(parameters,5.);opt.step()
        if step==1:
            change=float((visual.backbone.blocks[31].attn.qkv.weight-before).abs().max())
            # Record first update; low-LR cosine warmup may be below FP32 ULP.
            save(folder/'first_update.json',dict(last_block_max_weight_change=change,frozen_exact=(change==0) if not joint else None))
            if not joint:assert change==0
            del before
        if step%20==0:print(json.dumps(dict(stage='train',arm=args.arm,step=step,cumulative_step=600+step,loss=total,parts=parts,seconds=time.time()-start)),flush=True)
        if step%100==0 or step==args.steps:
            rank,feasible,xyz=validate(step)
            if rank<best_trained:
                best_trained=rank;atomic_torch(ckvalue(step),folder/'best_trained.pt')
                atomic_torch(dict(indices=dev.cpu(),prediction=xyz.cpu()),folder/'development_predictions.pt')
            if feasible and rank<best_admitted:
                best_admitted=rank;atomic_torch(ckvalue(step),folder/'best.pt')
                atomic_torch(ckvalue(step),folder/'best_feasible_trained.pt')
        if step%100==0 or step==args.steps:
            save_resume(step)
            if not (folder/'resume_verification.json').exists():verify_resume(step)
    atomic_torch(ckvalue(args.steps),folder/'last.pt')
    ck=torch.load(folder/'best.pt',weights_only=False,map_location='cpu');bc=torch.load(folder/'best_trained.pt',weights_only=False,map_location='cpu')
    save(folder/'done.json',dict(complete=True,steps=args.steps,cumulative_steps=600+args.steps,
        selected_step=ck['step'],selected_reference_only=ck['config'].get('reference_only',False),
        best_diagnostic_step=bc['step'],trained_checkpoint_admitted=(folder/'best_feasible_trained.pt').exists(),seconds=time.time()-start,default_changed=False))
    print((folder/'done.json').read_text(),flush=True)

if __name__=='__main__':main()
