"""Train on disjoint R clips; selection only P0003; held-out tests untouched."""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np,torch
from residual_models import ResidualModel
from temporal_residual import TemporalResidualModel,attach_context
from train_coarse_pose import RUN
from metrics_3d import compare

def evaluate(model,data,indices,device,gated=False):
    coarse=[];gt=[];refined=[];model.eval();gen=torch.Generator(device=device).manual_seed(81003)
    with torch.inference_mode():
        for start in range(0,len(indices),256):
            ids=indices[start:start+256];c=data['coarse'][ids].to(device);f=data['confidence'][ids].to(device);rgb=data['rgb'][ids].to(device).float()
            attach_context(model,data,ids,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                proposal=model.propose(c,f,rgb,steps=10,samples=4 if model.kind=='dit' else 1,generator=gen)
                gates=model.gates(proposal,c,f,rgb)[1] if gated else torch.ones_like(proposal[...,0])
                pred=model.unpack(model.pack(c)+gates[...,None]*proposal)
            coarse.append(c.cpu().numpy());gt.append(data['gt'][ids].numpy());refined.append(pred.float().cpu().numpy())
    result=compare(np.concatenate(coarse),np.concatenate(refined),np.concatenate(gt))
    for key in ['coarse','refined']:result[key].pop('sample_mpjpe19_mm')
    return result

def train(kind,steps,gate_steps,device,temporal=False):
    torch.set_num_threads(4);torch.manual_seed(20261003);np.random.seed(20261003);random.seed(20261003)
    data=torch.load(RUN/'coarse_cache.pt',map_location='cpu',weights_only=False)
    if temporal:data['temporal']=torch.load(RUN/'temporal_context.pt',map_location='cpu',weights_only=False)
    train_ids=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='residual'])
    tune_ids=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='tune'])
    assert len(train_ids)>500 and len(tune_ids)>100
    assert not set(train_ids.tolist())&set(tune_ids.tolist())
    # Test records are cached for later evaluation; they never enter gradients or
    # checkpoint selection here.
    model=(TemporalResidualModel(kind=kind) if temporal else ResidualModel(kind=kind)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.01)
    folder=RUN/(f'temporal_residual_{kind}' if temporal else f'residual_{kind}');folder.mkdir(exist_ok=True);history=[];best=float('inf');started=time.time()
    for step in range(1,steps+1):
        model.train();ids=train_ids[torch.randint(len(train_ids),(256,))]
        c=data['coarse'][ids].to(device);f=data['confidence'][ids].to(device);rgb=data['rgb'][ids].to(device).float();gt=data['gt'][ids].to(device)
        # Small perturbation to predicted poses only, shared by both methods.
        if not temporal:c=c+torch.randn_like(c)*.001
        attach_context(model,data,ids,device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.prediction_loss(gt,c,f,rgb)
        assert torch.isfinite(loss),parts;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:print(json.dumps(dict(stage='denoiser',step=step,loss=float(loss.detach()),parts=parts,seconds=time.time()-started)),flush=True)
        if step%500==0 or step==steps:
            result=evaluate(model,data,tune_ids,device,gated=False);score=result['refined']['mpjpe19_mm']+result['refined']['wrist_relative_mpjpe19_mm']
            row=dict(stage='denoiser',step=step,tune=result,seconds=time.time()-started);history.append(row)
            ckpt=dict(model=model.state_dict(),kind=kind,stage='denoiser',step=step,history=history)
            torch.save(ckpt,folder/'denoiser_last.pt')
            if score<best:best=score;torch.save(ckpt,folder/'denoiser_best.pt')
            (folder/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    model.load_state_dict(torch.load(folder/'denoiser_best.pt',map_location=device,weights_only=False)['model']);model.eval()
    if gate_steps==0:
        (folder/'done.json').write_text(json.dumps(dict(completed=True,checkpoint=str(folder/'denoiser_best.pt'),stage='denoiser_only')))
        return
    # Freeze the proposal model. Gate supervision comes from real DDIM proposals
    # (different seeds), generated without GT residual injection.
    for p in model.parameters():p.requires_grad_(False)
    proposals=[]
    for seed in [901,902]:
        pieces=[];gen=torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            for start in range(0,len(train_ids),256):
                ids=train_ids[start:start+256];c=data['coarse'][ids].to(device);f=data['confidence'][ids].to(device);rgb=data['rgb'][ids].to(device).float()
                attach_context(model,data,ids,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model.propose(c,f,rgb,steps=10,samples=4 if kind=='dit' else 1,generator=gen)
                pieces.append(p.float().cpu())
        proposals.append(torch.cat(pieces))
    for p in model.gate.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(model.gate.parameters(),lr=.0003,weight_decay=.001);best=float('inf')
    for step in range(1,gate_steps+1):
        positions=torch.randint(len(train_ids),(256,));ids=train_ids[positions]
        proposal=proposals[step%len(proposals)][positions].to(device)
        c=data['coarse'][ids].to(device);f=data['confidence'][ids].to(device);rgb=data['rgb'][ids].to(device).float();gt=data['gt'][ids].to(device)
        attach_context(model,data,ids,device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.gate_loss(proposal,gt,c,f,rgb)
        assert torch.isfinite(loss),parts;loss.backward();torch.nn.utils.clip_grad_norm_(model.gate.parameters(),1.);opt.step()
        if step%100==0:print(json.dumps(dict(stage='gate',step=step,loss=float(loss.detach()),parts=parts,seconds=time.time()-started)),flush=True)
        if step%250==0 or step==gate_steps:
            result=evaluate(model,data,tune_ids,device,gated=True)
            score=result['refined']['mpjpe19_mm']+result['refined']['wrist_relative_mpjpe19_mm']+100*max(0,result['correct_joints_harmed_fraction']-.05)
            row=dict(stage='gate',step=step,tune=result,seconds=time.time()-started);history.append(row)
            ckpt=dict(model=model.state_dict(),kind=kind,stage='gate',step=step,history=history)
            torch.save(ckpt,folder/'last.pt')
            if score<best:best=score;torch.save(ckpt,folder/'best.pt')
            (folder/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    (folder/'done.json').write_text(json.dumps(dict(completed=True,checkpoint=str(folder/'best.pt'))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['dit','regression'],default='dit');ap.add_argument('--steps',type=int,default=5000)
    ap.add_argument('--gate-steps',type=int,default=1500);ap.add_argument('--device',default='cuda:2');ap.add_argument('--temporal',action='store_true');a=ap.parse_args()
    train(a.kind,a.steps,a.gate_steps,a.device,getattr(a,'temporal',False))

if __name__=='__main__':main()
