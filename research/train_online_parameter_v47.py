"""Matched live-RGB frozen/joint continuation with FK and motion supervision."""
import argparse,hashlib,json,time
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from parameter_codec_v31 import OUT,observation_batch
from online_parameter_model_v47 import RUN,OnlineVisual,load_head

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['regression','dit'],default='dit')
    ap.add_argument('--mode',choices=['frozen','joint'],required=True);ap.add_argument('--device',default='cuda:3')
    ap.add_argument('--steps',type=int,default=600);ap.add_argument('--batch',type=int,default=2);ap.add_argument('--accumulation',type=int,default=2)
    args=ap.parse_args();torch.set_num_threads(4);seed=2026100517;torch.manual_seed(seed);np.random.seed(seed)
    folder=RUN/f'{args.kind}_{args.mode}';folder.mkdir(exist_ok=True)
    assert not (folder/'best.pt').exists(),'Existing run must remain immutable'
    assert json.loads((RUN/'ready.json').read_text())['complete']
    data=torch.load(RUN/'inputs.pt',weights_only=False,mmap=True);data={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in data.items()}
    target={k:v.to(args.device) for k,v in torch.load(RUN/'targets.pt',weights_only=False,mmap=True).items()}
    coarse=torch.load(V7.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(args.device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(args.device)
    pr=torch.load(RUN/'risk.pt',weights_only=False);prob,ep=pr['train_oof'].to(args.device),pr['joint'].to(args.device)
    pixels=np.load(RUN/'pixels.npy',mmap_mode='r');native=torch.load(V7.parent/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True)
    train=torch.tensor([i for i,r in enumerate(data['roles']) if r=='train'],device=args.device);dev=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'],device=args.device)
    model,path,initial=load_head(args.kind,args.device);visual=OnlineVisual(args.device,args.mode=='joint')
    plan=torch.randint(len(train),(args.steps,args.accumulation,args.batch),generator=torch.Generator().manual_seed(seed+1));torch.save(plan,folder/'batch_plan.pt')
    config=dict(**vars(args),seed=seed,batch_plan_sha256=hashlib.sha256(plan.numpy().tobytes()).hexdigest(),initial_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        initial_step=initial['step'],online_RGB=True,spatial='192x1280 native features; no global pooling',context_s=1.6,
        trained_visual_blocks=[28,29,30,31] if args.mode=='joint' else [],prefix_frozen=list(range(28)),last_norm_trained=args.mode=='joint',
        lr_head=2e-5,lr_backbone=1e-6,loss='Original 3D FK fit, diffusion/parameter, accurate-point protection, GT velocity, shape consistency, side',
        selection='Original feasible-first dev_select score; step0 included; no fresh/calibration metrics',
        same_online_crop_and_BF16_padded16=True,detector_and_tracks_fixed=True,wilor_xyz_and_200step_ik_fixed=True,
        decoder='Existing differentiable bounded UmeTrack/FK; geometry buffers fixed; parameter generator trained',
        postprocessor='Independent v46 speed-fixed/acceleration-x2 stability solver; outside backprop graph',
        default_changed=False,train_windows=len(train),dev_select_windows=len(dev),natural_only=True)
    save(folder/'config.json',config)
    def get(ids,risk):
        b=observation_batch(data,ids,risk,coarse,right,preserve_fitted=True)
        return visual.replace(b,data['feature_ids'][ids],pixels)
    def checkpoint(step,name='best.pt'):
        torch.save(dict(model=model.state_dict(),visual_tail=visual.tail_state() if args.mode=='joint' else None,step=step,kind=args.kind,config=config),folder/name)
    model.eval();visual.configure();ix=dev[:args.batch]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        cached=observation_batch(data,ix,ep,coarse,right,preserve_fitted=True);cached['rgb_native']=native[data['feature_ids'][ix].cpu()].to(args.device)
        live=get(ix,ep);p0=model.predict_parameters(cached,seed=seed);p1=model.predict_parameters(live,seed=seed)
        parity=dict(feature_max_abs=float((cached['rgb_native'].float()-live['rgb_native'].float()).abs().max()),feature_mean_abs=float((cached['rgb_native'].float()-live['rgb_native'].float()).abs().mean()),
            prediction_max_mm=float((p0['xyz_camera_m']-p1['xyz_camera_m']).norm(dim=-1).max()*1000),prediction_mean_mm=float((p0['xyz_camera_m']-p1['xyz_camera_m']).norm(dim=-1).mean()*1000))
        assert parity['prediction_max_mm']<.5,parity
        poison=dict(live,gt=torch.full_like(target['gt'][ix],float('nan')),gt_right=1-target['right'][ix],gt_shape=torch.ones(len(ix),10,device=args.device)*999)
        dirty=model.predict_parameters(poison,seed=seed)
        assert torch.equal(p1['state'],dirty['state']) and torch.equal(p1['xyz_camera_m'],dirty['xyz_camera_m'])
    save(folder/'initial_parity.json',dict(passed=True,gt_poison_exact=True,**parity))
    # Prove that pure FK XYZ loss alone reaches the visual backbone.
    if args.mode=='joint':
        model.eval();visual.configure();ix=train[:args.batch];b=get(ix,prob);tt={k:v[ix] for k,v in target.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            p=model.generate(b,seed);vm=tt['valid'].float();fk=((p['xyz']-tt['gt']).norm(dim=-1)*vm).sum()/vm.sum().clamp_min(1)
        tracked=visual.backbone.blocks[31].attn.qkv.weight
        g=torch.autograd.grad(fk,tracked)[0];norm=float(g.norm());assert np.isfinite(norm) and norm>0
        save(folder/'fk_gradient_path.json',dict(passed=True,pure_fk_3d_loss_to_visual_block31_qkv=norm,gt_xyz_loss_not_parameter_auxiliary=True))
        del b,p,g,fk
    groups=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=2e-5,initial_lr=2e-5)]
    if args.mode=='joint':groups.append(dict(params=[p for p in visual.parameters() if p.requires_grad],lr=1e-6,initial_lr=1e-6))
    opt=torch.optim.AdamW(groups,weight_decay=.01);parameters=[p for group in groups for p in group['params']]
    history=[];start=time.time();original=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    @torch.no_grad()
    def evaluate(step):
        model.eval();visual.configure();pred=[]
        for begin in range(0,len(dev),args.batch):
            ix=dev[begin:begin+args.batch]
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_parameters(get(ix,ep),seed=seed+begin)
            pred.append(p['xyz_camera_m'])
        xyz=torch.cat(pred);m=metrics(xyz,original['original_base_for_evaluation'][dev.cpu()].to(args.device),target['gt'][dev,8],target['valid'][dev,8]);rank,feasible=score(m)
        history.append(dict(step=step,metrics=m,rank=list(rank),feasible=feasible,seconds=time.time()-start));save(folder/'history.json',history)
        print(json.dumps(dict(stage='validation',mode=args.mode,**history[-1])),flush=True)
        return rank,xyz
    best,xyz=evaluate(0);checkpoint(0);torch.save(dict(indices=dev.cpu(),prediction=xyz.cpu()),folder/'development_predictions.pt')
    for step in range(1,args.steps+1):
        for group in opt.param_groups:group['lr']=group['initial_lr']*min(1.,step/30)*(.2+.8*.5*(1+np.cos(np.pi*step/args.steps)))
        model.train();visual.configure();opt.zero_grad(set_to_none=True);total=0.
        for micro in range(args.accumulation):
            torch.manual_seed(seed+step*args.accumulation+micro);ix=train[plan[step-1,micro].to(args.device)];b=get(ix,prob);tt={k:v[ix] for k,v in target.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(b,tt,seed+step*args.accumulation+micro);loss=loss/args.accumulation
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del b,loss
        if step==1:
            grads={str(i):sum(float(p.grad.float().square().sum()) for p in visual.backbone.blocks[i].parameters() if p.grad is not None)**.5 for i in range(28,32)}
            if args.mode=='joint':assert all(np.isfinite(x) and x>0 for x in grads.values()),grads
            else:assert all(x==0 for x in grads.values())
            before=visual.backbone.blocks[31].attn.qkv.weight.detach().clone();save(folder/'gradients.json',dict(passed=True,block_gradient_norms=grads))
        torch.nn.utils.clip_grad_norm_(parameters,5.);opt.step()
        if step==1:
            change=float((visual.backbone.blocks[31].attn.qkv.weight-before).abs().max())
            assert (change>0) if args.mode=='joint' else change==0
            save(folder/'weight_update.json',dict(passed=True,last_block_max_weight_change=change));del before
        if step%20==0:print(json.dumps(dict(stage='train',mode=args.mode,step=step,loss=total,parts=parts,seconds=time.time()-start)),flush=True)
        if step%100==0 or step==args.steps:
            rank,xyz=evaluate(step)
            if rank<best:
                best=rank;checkpoint(step);torch.save(dict(indices=dev.cpu(),prediction=xyz.cpu()),folder/'development_predictions.pt')
    checkpoint(args.steps,'last.pt');ck=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
    save(folder/'done.json',dict(complete=True,steps=args.steps,selected_step=ck['step'],seconds=time.time()-start,default_changed=False))
    print((folder/'done.json').read_text(),flush=True)
if __name__=='__main__':main()
